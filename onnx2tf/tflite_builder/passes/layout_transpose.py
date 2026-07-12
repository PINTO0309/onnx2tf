from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
def _is_identity_perm(perm: List[int]) -> bool:
    return perm == [int(i) for i in range(len(perm))]

def _is_inverse_perm(perm_a: List[int], perm_b: List[int]) -> bool:
    if len(perm_a) != len(perm_b):
        return False
    rank = len(perm_a)
    if sorted(perm_a) != [int(i) for i in range(rank)]:
        return False
    if sorted(perm_b) != [int(i) for i in range(rank)]:
        return False
    for idx, value in enumerate(perm_a):
        if perm_b[value] != idx:
            return False
    return True

def _optimize_layout_transpose_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate redundant TRANSPOSE chains introduced by channel-first/channel-last bridging.

    This pass removes:
    - identity transpose: X --Transpose(identity)--> Y
    - inverse transpose pairs with single-edge bridge:
        A --Transpose(P)--> B --Transpose(inv(P))--> C
      by directly reconnecting C consumers to A.

    The optimization is output-safe: graph outputs are never renamed.
    """
    removed_identity = 0
    removed_inverse_pairs = 0
    removed_inverse_fanout_branches = 0
    composed_consecutive_pairs = 0
    iterations = 0
    preserve_layout_boundary_marker = "__preserve_layout_boundary__"
    rank4_perm_nhwc_to_nchw = [0, 3, 1, 2]
    rank4_perm_nchw_to_nwhc = [0, 3, 2, 1]

    def _compose_permutations(perm_pre: List[int], perm_post: List[int]) -> Optional[List[int]]:
        if len(perm_pre) != len(perm_post) or len(perm_pre) == 0:
            return None
        rank = len(perm_pre)
        if sorted(int(v) for v in perm_pre) != list(range(rank)):
            return None
        if sorted(int(v) for v in perm_post) != list(range(rank)):
            return None
        try:
            return [int(perm_pre[int(v)]) for v in perm_post]
        except Exception:
            return None

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        iterations += 1
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        # 1) Remove identity transpose.
        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue
            perm = _read_transpose_perm(model_ir, op)
            if perm is None or not _is_identity_perm(perm):
                continue
            transposed_output = op.outputs[0]
            transposed_input = op.inputs[0]
            if transposed_output in model_ir.outputs:
                continue
            _replace_tensor_inputs(model_ir, transposed_output, transposed_input)
            del model_ir.operators[op_idx]
            removed_identity += 1
            changed = True
            break
        if changed:
            continue

        # 2) Remove inverse transpose pair.
        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue
            bridge_tensor = op.outputs[0]
            bridge_users = consumers.get(bridge_tensor, [])
            if len(bridge_users) != 1:
                continue
            next_op_idx = int(bridge_users[0])
            if next_op_idx == op_idx:
                continue
            next_op = model_ir.operators[next_op_idx]
            if str(next_op.op_type) != "TRANSPOSE":
                continue
            if len(next_op.inputs) < 2 or len(next_op.outputs) != 1:
                continue
            if next_op.inputs[0] != bridge_tensor:
                continue

            perm_1 = _read_transpose_perm(model_ir, op)
            perm_2 = _read_transpose_perm(model_ir, next_op)
            if perm_1 is None or perm_2 is None:
                continue
            if not _is_inverse_perm(perm_1, perm_2):
                continue

            transpose_1_input = op.inputs[0]
            transpose_1_output = op.outputs[0]
            transpose_2_output = next_op.outputs[0]

            # Keep user-visible output names stable.
            if transpose_1_output in model_ir.outputs or transpose_2_output in model_ir.outputs:
                continue

            _replace_tensor_inputs(model_ir, transpose_2_output, transpose_1_input)
            for remove_idx in sorted([op_idx, next_op_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_inverse_pairs += 1
            changed = True
            break

        if not changed:
            # 2b) Remove inverse transpose branches even when the first transpose
            # fans out to multiple consumers.
            for op_idx, op in enumerate(model_ir.operators):
                if str(op.op_type) != "TRANSPOSE":
                    continue
                if len(op.inputs) < 2 or len(op.outputs) != 1:
                    continue

                bridge_tensor = str(op.outputs[0])
                bridge_users = [int(v) for v in consumers.get(bridge_tensor, [])]
                if len(bridge_users) <= 1:
                    continue

                perm_pre = _read_transpose_perm(model_ir, op)
                if perm_pre is None:
                    continue

                for post_idx in bridge_users:
                    if int(post_idx) == int(op_idx):
                        continue
                    post_op = model_ir.operators[int(post_idx)]
                    if str(post_op.op_type) != "TRANSPOSE":
                        continue
                    if len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                        continue
                    if str(post_op.inputs[0]) != bridge_tensor:
                        continue

                    post_opts = dict(post_op.options) if isinstance(post_op.options, dict) else {}
                    if bool(post_opts.get(preserve_layout_boundary_marker, False)):
                        continue

                    perm_post = _read_transpose_perm(model_ir, post_op)
                    if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                        continue

                    post_output = str(post_op.outputs[0])
                    if bridge_tensor in model_ir.outputs or post_output in model_ir.outputs:
                        continue

                    _replace_tensor_inputs(model_ir, post_output, str(op.inputs[0]))
                    del model_ir.operators[int(post_idx)]
                    removed_inverse_fanout_branches += 1
                    changed = True
                    break
                if changed:
                    break

        if not changed:
            # 3) Compose consecutive transpose pairs into one transpose.
            for op_idx, op in enumerate(model_ir.operators):
                if str(op.op_type) != "TRANSPOSE":
                    continue
                if len(op.inputs) < 2 or len(op.outputs) != 1:
                    continue
                op_opts = dict(op.options) if isinstance(op.options, dict) else {}
                if bool(op_opts.get(preserve_layout_boundary_marker, False)):
                    continue

                bridge_tensor = str(op.outputs[0])
                bridge_users = [int(v) for v in consumers.get(bridge_tensor, [])]
                if len(bridge_users) != 1:
                    continue
                # The first transpose may be observable independently from the
                # downstream chain (for example ONNX NonZero exposes [rank, N]
                # while a following Transpose consumes it as [N, rank]).  It is
                # valid to compose the pair for the consumer only when deleting
                # the first operator cannot orphan a public output tensor.
                if bridge_tensor in set(str(v) for v in model_ir.outputs):
                    continue

                next_op_idx = int(bridge_users[0])
                if next_op_idx == int(op_idx):
                    continue
                next_op = model_ir.operators[int(next_op_idx)]
                if str(next_op.op_type) != "TRANSPOSE":
                    continue
                if len(next_op.inputs) < 2 or len(next_op.outputs) != 1:
                    continue
                if str(next_op.inputs[0]) != bridge_tensor:
                    continue
                next_opts = dict(next_op.options) if isinstance(next_op.options, dict) else {}
                if bool(next_opts.get(preserve_layout_boundary_marker, False)):
                    continue

                perm_1 = _read_transpose_perm(model_ir, op)
                perm_2 = _read_transpose_perm(model_ir, next_op)
                if perm_1 is None or perm_2 is None:
                    continue
                # Keep this pattern intact for `_canonicalize_softmax_transpose_chains`.
                if (
                    [int(v) for v in perm_1] == rank4_perm_nhwc_to_nchw
                    and [int(v) for v in perm_2] == rank4_perm_nchw_to_nwhc
                ):
                    next_out_name = str(next_op.outputs[0])
                    next_out_users = [int(v) for v in consumers.get(next_out_name, [])]
                    if len(next_out_users) == 1:
                        softmax_idx = int(next_out_users[0])
                        if softmax_idx != int(next_op_idx):
                            softmax_op = model_ir.operators[softmax_idx]
                            if (
                                str(softmax_op.op_type) == "SOFTMAX"
                                and len(softmax_op.inputs) == 1
                                and len(softmax_op.outputs) == 1
                                and str(softmax_op.inputs[0]) == next_out_name
                            ):
                                softmax_out_name = str(softmax_op.outputs[0])
                                softmax_out_users = [int(v) for v in consumers.get(softmax_out_name, [])]
                                if len(softmax_out_users) == 1:
                                    post_idx = int(softmax_out_users[0])
                                    if post_idx != int(softmax_idx):
                                        post_op = model_ir.operators[post_idx]
                                        if (
                                            str(post_op.op_type) == "TRANSPOSE"
                                            and len(post_op.inputs) >= 2
                                            and len(post_op.outputs) == 1
                                            and str(post_op.inputs[0]) == softmax_out_name
                                            and _read_transpose_perm(model_ir, post_op) == rank4_perm_nchw_to_nwhc
                                        ):
                                            continue
                composed_perm = _compose_permutations(
                    [int(v) for v in perm_1],
                    [int(v) for v in perm_2],
                )
                if composed_perm is None:
                    continue

                next_perm_name = str(next_op.inputs[1])
                next_perm_users = [int(v) for v in consumers.get(next_perm_name, [])]
                if len(next_perm_users) == 1 and int(next_perm_users[0]) == int(next_op_idx):
                    perm_tensor = model_ir.tensors.get(next_perm_name, None)
                    if perm_tensor is None:
                        continue
                    perm_arr = np.asarray(composed_perm, dtype=np.int32)
                    perm_tensor.data = perm_arr
                    perm_tensor.dtype = "INT32"
                    perm_tensor.shape = [int(perm_arr.size)]
                    perm_tensor.shape_signature = [int(perm_arr.size)]
                    new_perm_name = next_perm_name
                else:
                    new_perm_name = _unique_tensor_name(f"{next_perm_name}_composed")
                    perm_arr = np.asarray(composed_perm, dtype=np.int32)
                    model_ir.tensors[new_perm_name] = TensorIR(
                        name=new_perm_name,
                        dtype="INT32",
                        shape=[int(perm_arr.size)],
                        shape_signature=[int(perm_arr.size)],
                        data=perm_arr,
                        is_variable=False,
                        quantization=None,
                    )

                _set_operator_inputs(
                    model_ir=model_ir,
                    op=next_op,
                    new_inputs=[str(op.inputs[0]), str(new_perm_name)],
                )
                del model_ir.operators[int(op_idx)]
                composed_consecutive_pairs += 1
                changed = True
                break

            if not changed:
                break

    _prune_unused_tensors(model_ir)
    return {
        "iterations": int(iterations),
        "removed_identity_transpose": int(removed_identity),
        "removed_inverse_transpose_pairs": int(removed_inverse_pairs),
        "removed_inverse_transpose_fanout_branches": int(removed_inverse_fanout_branches),
        "composed_consecutive_transpose_pairs": int(composed_consecutive_pairs),
    }

