from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _broadcast_shape_signatures,
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _invert_perm,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _shapes_match_if_known,
)
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_TRANSPOSE_UNARY_PASSTHROUGH_OPS = frozenset(
    {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "GELU",
        "ABS",
        "NEG",
        "SQRT",
        "EXP",
        "FLOOR",
        "CEIL",
    }
)
_TRANSPOSE_UNARY_FANOUT_OPS = frozenset(
    {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "GELU",
    }
)


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

def _optimize_layout_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
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
        consumers = graph_index.consumers

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
            _replace_tensor_inputs(
                model_ir,
                transposed_output,
                transposed_input,
                graph_index=graph_index,
            )
            graph_index.remove_operator(op_idx)
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

            _replace_tensor_inputs(
                model_ir,
                transpose_2_output,
                transpose_1_input,
                graph_index=graph_index,
            )
            for remove_idx in sorted([op_idx, next_op_idx], reverse=True):
                graph_index.remove_operator(remove_idx)
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

                    _replace_tensor_inputs(
                        model_ir,
                        post_output,
                        str(op.inputs[0]),
                        graph_index=graph_index,
                    )
                    graph_index.remove_operator(int(post_idx))
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
                    graph_index=graph_index,
                )
                graph_index.remove_operator(int(op_idx))
                composed_consecutive_pairs += 1
                changed = True
                break

            if not changed:
                break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "iterations": int(iterations),
        "removed_identity_transpose": int(removed_identity),
        "removed_inverse_transpose_pairs": int(removed_inverse_pairs),
        "removed_inverse_transpose_fanout_branches": int(removed_inverse_fanout_branches),
        "composed_consecutive_transpose_pairs": int(composed_consecutive_pairs),
    }


def run_layout_transpose_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run identity, inverse, fan-out, and composed Transpose cleanup."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        for visited, operator in enumerate(candidate_model.operators, start=1):
            if str(operator.op_type) == "TRANSPOSE":
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        for transpose_op in candidate_model.operators:
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.inputs) < 2
                or len(transpose_op.outputs) != 1
            ):
                continue
            perm_pre = _read_transpose_perm(candidate_model, transpose_op)
            if perm_pre is None:
                continue
            if _is_identity_perm(perm_pre):
                return True
            bridge_name = str(transpose_op.outputs[0])
            for user_index in pass_state.graph_index.consumer_indices(bridge_name):
                post_op = candidate_model.operators[int(user_index)]
                if (
                    str(post_op.op_type) != "TRANSPOSE"
                    or len(post_op.inputs) < 2
                    or len(post_op.outputs) != 1
                    or str(post_op.inputs[0]) != bridge_name
                ):
                    continue
                perm_post = _read_transpose_perm(candidate_model, post_op)
                if perm_post is not None and len(perm_pre) == len(perm_post):
                    return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_layout_transpose_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        changed_count = sum(
            int(stats.get(key, 0))
            for key in (
                "removed_identity_transpose",
                "removed_inverse_transpose_pairs",
                "removed_inverse_transpose_fanout_branches",
                "composed_consecutive_transpose_pairs",
            )
        )
        return {**stats, "changed": bool(changed_count)}

    default_details = {
        "iterations": 0,
        "removed_identity_transpose": 0,
        "removed_inverse_transpose_pairs": 0,
        "removed_inverse_transpose_fanout_branches": 0,
        "composed_consecutive_transpose_pairs": 0,
    }
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.transpose_chain_cleanup",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_transpose_unary_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Fold transpose wrappers around strict unary passthrough chains.

    Target:
      X --TRANSPOSE(P)--> A --(UNARY)*--> B --TRANSPOSE(inv(P))--> Y

    Rewrite:
      X --(UNARY)*--> Y

    Safety:
    - Unary chain must be strictly linear (single consumer per main-path output).
    - Only layout-agnostic unary ops are allowed.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    while True:
        changed = False
        consumers = graph_index.consumers

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE":
                continue
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_ir.outputs:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            # Only fold the canonical NHWC->NCHW->NHWC bridge direction.
            # Reverse-direction folds can corrupt downstream layout contracts.
            if [int(v) for v in list(perm_pre)] != perm_nhwc_to_nchw:
                continue
            perm_post_expected = _invert_perm(perm_pre)
            if perm_post_expected is None:
                continue

            chain_indices: List[int] = []
            chain_ops: List[OperatorIR] = []
            chain_output_names: List[str] = []
            chain_main_input_index: Dict[int, int] = {}
            current_tensor = pre_output_name

            while True:
                current_users = [int(v) for v in consumers.get(current_tensor, [])]
                if len(current_users) != 1:
                    break
                op_idx = int(current_users[0])
                op = model_ir.operators[op_idx]
                op_type = str(op.op_type)
                if op_type not in _TRANSPOSE_UNARY_PASSTHROUGH_OPS:
                    break
                if len(op.inputs) != 1 or len(op.outputs) != 1:
                    break
                if str(op.inputs[0]) != current_tensor:
                    break
                out_name = str(op.outputs[0])
                if out_name in model_ir.outputs:
                    break
                chain_indices.append(int(op_idx))
                chain_ops.append(op)
                chain_output_names.append(out_name)
                current_tensor = out_name

            if len(chain_ops) == 0:
                continue

            tail_users = [int(v) for v in consumers.get(current_tensor, [])]
            if len(tail_users) != 1:
                continue
            post_idx = int(tail_users[0])
            if post_idx in set(chain_indices):
                continue
            post_op = model_ir.operators[int(post_idx)]
            if str(post_op.op_type) != "TRANSPOSE":
                continue
            if len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != current_tensor:
                continue
            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or perm_post != perm_post_expected:
                continue
            post_output_name = str(post_op.outputs[0])

            # Rewire chain head to the pre-transpose source.
            _set_operator_inputs(
                model_ir=model_ir,
                op=chain_ops[0],
                new_inputs=[pre_input_name],
                graph_index=graph_index,
            )
            # Preserve output name expected after post-transpose.
            _set_operator_outputs(
                model_ir=model_ir,
                op=chain_ops[-1],
                new_outputs=[post_output_name],
                graph_index=graph_index,
            )

            # Update intermediate metadata to post-transpose layout.
            for out_name in chain_output_names[:-1]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(out_name, None),
                    perm_post_expected,
                )

            old_last_name = str(chain_output_names[-1])
            old_last_tensor = model_ir.tensors.get(old_last_name, None)
            post_output_tensor = model_ir.tensors.get(post_output_name, None)
            if old_last_tensor is not None and post_output_tensor is not None:
                post_output_tensor.dtype = str(old_last_tensor.dtype)
                post_output_tensor.quantization = _clone_quantization(old_last_tensor.quantization)
                post_output_tensor.shape = [int(v) for v in list(old_last_tensor.shape)]
                post_output_tensor.shape_signature = (
                    [int(v) for v in list(old_last_tensor.shape_signature)]
                    if old_last_tensor.shape_signature is not None
                    else [int(v) for v in list(old_last_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_output_tensor,
                    perm_post_expected,
                )

            remove_indices = sorted(list({int(pre_idx), int(post_idx)}), reverse=True)
            for remove_idx in remove_indices:
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_transpose_unary_passthrough_chains": int(rewritten)}


def run_transpose_unary_passthrough_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run strict NHWC/NCHW unary Transpose passthrough cleanup."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        for visited, operator in enumerate(candidate_model.operators, start=1):
            if str(operator.op_type) == "TRANSPOSE":
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        graph_outputs = set(str(name) for name in candidate_model.outputs)
        for pre_op in candidate_model.operators:
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or _read_transpose_perm(candidate_model, pre_op) != [0, 3, 1, 2]
            ):
                continue
            current_tensor = str(pre_op.outputs[0])
            if current_tensor in graph_outputs:
                continue
            chain_length = 0
            while True:
                users = pass_state.graph_index.consumer_indices(current_tensor)
                if len(users) != 1:
                    break
                operator = candidate_model.operators[int(users[0])]
                if (
                    str(operator.op_type) not in _TRANSPOSE_UNARY_PASSTHROUGH_OPS
                    or len(operator.inputs) != 1
                    or len(operator.outputs) != 1
                    or str(operator.inputs[0]) != current_tensor
                ):
                    break
                current_tensor = str(operator.outputs[0])
                if current_tensor in graph_outputs:
                    break
                chain_length += 1
            if chain_length == 0 or current_tensor in graph_outputs:
                continue
            users = pass_state.graph_index.consumer_indices(current_tensor)
            if len(users) != 1:
                continue
            post_op = candidate_model.operators[int(users[0])]
            if (
                str(post_op.op_type) == "TRANSPOSE"
                and len(post_op.inputs) >= 2
                and len(post_op.outputs) == 1
                and str(post_op.inputs[0]) == current_tensor
                and _read_transpose_perm(candidate_model, post_op) == [0, 2, 3, 1]
            ):
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_unary_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("rewritten_transpose_unary_passthrough_chains", 0)
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.transpose_unary_passthrough",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"rewritten_transpose_unary_passthrough_chains": 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_transpose_unary_fanout_inverse_post_bridges(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate transpose wrappers around unary fanout branches with inverse post-transpose consumers.

    Target:
      X --TRANSPOSE(P)--> A --(RELU|RELU6|LOGISTIC|TANH)--> B
      B --TRANSPOSE(inv(P))--> Y0
      B --TRANSPOSE(inv(P))--> Y1
      ...
      (optionally: B also has non-transpose legacy consumers)

    Rewrite:
      X --(UNARY)--> Y0
      (all uses of Y1... are rewired to Y0)
      If legacy consumers exist:
        keep one adapter TRANSPOSE(P): Y0 -> B for legacy consumers.

    Safety:
    - Pre-transpose output is consumed only by the unary op.
    - Inverse-post transpose consumers must use inv(P).
    - Fanout post-transpose outputs are not graph outputs.
    - If non-transpose legacy consumers exist, preserve B via one adapter.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue
            perm_post_expected = _invert_perm(perm_pre)
            if perm_post_expected is None:
                continue

            unary_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(unary_users) != 1:
                continue
            unary_idx = int(unary_users[0])
            unary_op = model_ir.operators[unary_idx]
            if str(unary_op.op_type) not in _TRANSPOSE_UNARY_FANOUT_OPS:
                continue
            if len(unary_op.inputs) != 1 or len(unary_op.outputs) != 1:
                continue
            if str(unary_op.inputs[0]) != pre_output_name:
                continue

            unary_output_name = str(unary_op.outputs[0])
            if unary_output_name in model_outputs:
                continue
            output_users = [int(v) for v in consumers.get(unary_output_name, [])]
            if len(output_users) == 0:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_users: List[int] = []
            valid_users = True
            for user_idx in output_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == unary_output_name
                ):
                    perm_post = _read_transpose_perm(model_ir, user_op)
                    if perm_post is None or perm_post != perm_post_expected:
                        valid_users = False
                        break
                    post_output_name = str(user_op.outputs[0])
                    if post_output_name in model_outputs:
                        valid_users = False
                        break
                    post_indices.append(int(user_idx))
                    post_output_names.append(post_output_name)
                else:
                    legacy_users.append(int(user_idx))
            if not valid_users or len(post_indices) == 0:
                continue

            representative_output_name = str(post_output_names[0])
            _set_operator_inputs(
                model_ir=model_ir,
                op=unary_op,
                new_inputs=[pre_input_name],
                graph_index=graph_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=unary_op,
                new_outputs=[representative_output_name],
                graph_index=graph_index,
            )

            for post_output_name in post_output_names[1:]:
                _replace_tensor_inputs(
                    model_ir,
                    post_output_name,
                    representative_output_name,
                    graph_index=graph_index,
                )

            old_unary_tensor = model_ir.tensors.get(unary_output_name, None)
            representative_tensor = model_ir.tensors.get(representative_output_name, None)
            if old_unary_tensor is not None and representative_tensor is not None:
                representative_tensor.dtype = str(old_unary_tensor.dtype)
                representative_tensor.quantization = _clone_quantization(old_unary_tensor.quantization)

            if len(legacy_users) > 0:
                keep_post_idx = int(post_indices[0])
                keep_post_op = model_ir.operators[keep_post_idx]
                keep_perm_name = str(keep_post_op.inputs[1])
                keep_perm_tensor = model_ir.tensors.get(keep_perm_name, None)
                if keep_perm_tensor is not None:
                    keep_perm_tensor.data = np.asarray(perm_pre, dtype=np.int32)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_inputs=[representative_output_name, keep_perm_name],
                    graph_index=graph_index,
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_outputs=[unary_output_name],
                    graph_index=graph_index,
                )
                remove_post_indices = [int(v) for v in post_indices[1:]]
            else:
                remove_post_indices = [int(v) for v in post_indices]

            remove_indices = sorted(list({int(pre_idx), *remove_post_indices}), reverse=True)
            for remove_idx in remove_indices:
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_transpose_unary_fanout_inverse_post_bridges": int(rewritten)}


def run_transpose_unary_fanout_bridge_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run inverse-post Transpose cleanup around unary fan-out branches."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        found_transpose = False
        found_unary = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            found_transpose = found_transpose or operator_type == "TRANSPOSE"
            found_unary = (
                found_unary or operator_type in _TRANSPOSE_UNARY_FANOUT_OPS
            )
            if found_transpose and found_unary:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        model_outputs = set(str(name) for name in candidate_model.outputs)
        for pre_op in candidate_model.operators:
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
            ):
                continue
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue
            perm_pre = _read_transpose_perm(candidate_model, pre_op)
            if perm_pre is None:
                continue
            expected_post_perm = _invert_perm(perm_pre)
            if expected_post_perm is None:
                continue
            unary_users = pass_state.graph_index.consumer_indices(pre_output_name)
            if len(unary_users) != 1:
                continue
            unary_op = candidate_model.operators[int(unary_users[0])]
            if (
                str(unary_op.op_type) not in _TRANSPOSE_UNARY_FANOUT_OPS
                or len(unary_op.inputs) != 1
                or len(unary_op.outputs) != 1
                or str(unary_op.inputs[0]) != pre_output_name
            ):
                continue
            unary_output_name = str(unary_op.outputs[0])
            if unary_output_name in model_outputs:
                continue
            output_users = pass_state.graph_index.consumer_indices(unary_output_name)
            if len(output_users) == 0:
                continue
            post_count = 0
            valid_users = True
            for user_index in output_users:
                user_op = candidate_model.operators[int(user_index)]
                if (
                    str(user_op.op_type) != "TRANSPOSE"
                    or len(user_op.inputs) < 2
                    or len(user_op.outputs) != 1
                    or str(user_op.inputs[0]) != unary_output_name
                ):
                    continue
                if (
                    _read_transpose_perm(candidate_model, user_op)
                    != expected_post_perm
                    or str(user_op.outputs[0]) in model_outputs
                ):
                    valid_users = False
                    break
                post_count += 1
            if valid_users and post_count > 0:
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_unary_fanout_inverse_post_bridges(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "rewritten_transpose_unary_fanout_inverse_post_bridges",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.transpose_unary_fanout_bridge",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "rewritten_transpose_unary_fanout_inverse_post_bridges": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_transpose_unary_binary_full_post_fanout_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose wrappers around unary->binary chains with inverse post-transpose fanout.

    Target:
      X --T(P)--> X_t --(RELU|RELU6|LOGISTIC|TANH)--> U_t
      Y --T(P)--> Y_t
      U_t,Y_t --(ADD|SUB|MUL|DIV)--> Z_t
      Z_t --T(inv(P))--> Z0
      Z_t --T(inv(P))--> Z1
      ...

    Rewrite:
      X --(UNARY)--> U
      U,Y --(BINARY)--> Z0
      (all uses of Z1... are rewired to Z0)

    Safety:
    - The unary-side transpose output is consumed only by the unary op.
    - The unary output and the other transpose output are consumed only by the binary op.
    - All binary consumers are inverse post-transpose ops.
    - Intermediate tensors and post outputs are not graph outputs.
    """
    rewritten = 0
    unary_ops = {"RELU", "RELU6", "RELU_0_TO_1", "HARD_SWISH", "LOGISTIC", "TANH", "GELU"}
    binary_ops = {"ADD", "SUB", "MUL", "DIV"}

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in binary_ops:
                continue
            if len(mid_op.inputs) != 2 or len(mid_op.outputs) != 1:
                continue

            in0_name = str(mid_op.inputs[0])
            in1_name = str(mid_op.inputs[1])
            out_name = str(mid_op.outputs[0])

            if in0_name in model_outputs or in1_name in model_outputs or out_name in model_outputs:
                continue

            # One binary input must be produced by unary-from-transpose,
            # and the other by a matching transpose.
            candidate = None
            for unary_on_lhs in [True, False]:
                unary_input_name = in0_name if unary_on_lhs else in1_name
                other_input_name = in1_name if unary_on_lhs else in0_name

                unary_idx = producers.get(unary_input_name, None)
                if unary_idx is None:
                    continue
                unary_op = model_ir.operators[int(unary_idx)]
                if (
                    str(unary_op.op_type) not in unary_ops
                    or len(unary_op.inputs) != 1
                    or len(unary_op.outputs) != 1
                    or str(unary_op.outputs[0]) != unary_input_name
                ):
                    continue

                pre_unary_out_name = str(unary_op.inputs[0])
                pre_unary_idx = producers.get(pre_unary_out_name, None)
                if pre_unary_idx is None:
                    continue
                pre_unary_op = model_ir.operators[int(pre_unary_idx)]
                if (
                    str(pre_unary_op.op_type) != "TRANSPOSE"
                    or len(pre_unary_op.inputs) < 2
                    or len(pre_unary_op.outputs) != 1
                    or str(pre_unary_op.outputs[0]) != pre_unary_out_name
                ):
                    continue

                pre_other_idx = producers.get(other_input_name, None)
                if pre_other_idx is None:
                    continue
                pre_other_op = model_ir.operators[int(pre_other_idx)]
                if (
                    str(pre_other_op.op_type) != "TRANSPOSE"
                    or len(pre_other_op.inputs) < 2
                    or len(pre_other_op.outputs) != 1
                    or str(pre_other_op.outputs[0]) != other_input_name
                ):
                    continue

                if set(consumers.get(pre_unary_out_name, [])) != {int(unary_idx)}:
                    continue
                if set(consumers.get(unary_input_name, [])) != {int(mid_idx)}:
                    continue
                if set(consumers.get(other_input_name, [])) != {int(mid_idx)}:
                    continue

                perm_pre_unary = _read_transpose_perm(model_ir, pre_unary_op)
                perm_pre_other = _read_transpose_perm(model_ir, pre_other_op)
                if perm_pre_unary is None or perm_pre_other is None:
                    continue
                if perm_pre_unary != perm_pre_other:
                    continue

                candidate = {
                    "unary_on_lhs": bool(unary_on_lhs),
                    "unary_idx": int(unary_idx),
                    "unary_op": unary_op,
                    "unary_input_name": unary_input_name,
                    "pre_unary_idx": int(pre_unary_idx),
                    "pre_unary_op": pre_unary_op,
                    "pre_unary_out_name": pre_unary_out_name,
                    "other_input_name": other_input_name,
                    "pre_other_idx": int(pre_other_idx),
                    "pre_other_op": pre_other_op,
                    "perm_pre": perm_pre_unary,
                }
                break

            if candidate is None:
                continue

            unary_idx = int(candidate["unary_idx"])
            unary_op = candidate["unary_op"]
            unary_input_name = str(candidate["unary_input_name"])
            pre_unary_idx = int(candidate["pre_unary_idx"])
            pre_unary_op = candidate["pre_unary_op"]
            pre_unary_out_name = str(candidate["pre_unary_out_name"])
            other_input_name = str(candidate["other_input_name"])
            pre_other_idx = int(candidate["pre_other_idx"])
            pre_other_op = candidate["pre_other_op"]
            perm_pre = candidate["perm_pre"]
            unary_on_lhs = bool(candidate["unary_on_lhs"])

            out_users = [int(v) for v in consumers.get(out_name, []) if int(v) != int(mid_idx)]
            if len(out_users) == 0:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_users: List[int] = []
            valid_users = True
            for user_idx in out_users:
                post_op = model_ir.operators[int(user_idx)]
                if (
                    str(post_op.op_type) == "TRANSPOSE"
                    and len(post_op.inputs) >= 2
                    and len(post_op.outputs) == 1
                    and str(post_op.inputs[0]) == out_name
                ):
                    perm_post = _read_transpose_perm(model_ir, post_op)
                    if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                        valid_users = False
                        break
                    post_output_name = str(post_op.outputs[0])
                    if post_output_name in model_outputs:
                        valid_users = False
                        break
                    post_indices.append(int(user_idx))
                    post_output_names.append(post_output_name)
                else:
                    legacy_users.append(int(user_idx))

            if not valid_users or len(post_indices) == 0:
                continue

            if pre_unary_out_name in model_outputs:
                continue

            raw_unary_input_name = str(pre_unary_op.inputs[0])
            raw_other_input_name = str(pre_other_op.inputs[0])
            raw_unary_tensor = model_ir.tensors.get(raw_unary_input_name, None)
            raw_other_tensor = model_ir.tensors.get(raw_other_input_name, None)
            unary_input_tensor = model_ir.tensors.get(unary_input_name, None)
            other_input_tensor = model_ir.tensors.get(other_input_name, None)
            out_tensor = model_ir.tensors.get(out_name, None)
            if not _all_per_tensor_quantized(
                [raw_unary_tensor, raw_other_tensor, unary_input_tensor, other_input_tensor, out_tensor]
            ):
                continue

            raw_shape_unary = list(raw_unary_tensor.shape) if raw_unary_tensor is not None else None
            raw_shape_other = list(raw_other_tensor.shape) if raw_other_tensor is not None else None
            raw_broadcast_shape = _broadcast_static_shapes(raw_shape_unary, raw_shape_other)
            if raw_broadcast_shape is not None:
                mid_expected_shape = _permute_shape(list(raw_broadcast_shape), perm_pre)
                if mid_expected_shape is None:
                    continue

                in0_tensor = model_ir.tensors.get(in0_name, None)
                in1_tensor = model_ir.tensors.get(in1_name, None)
                in_shape0 = list(in0_tensor.shape) if in0_tensor is not None else None
                in_shape1 = list(in1_tensor.shape) if in1_tensor is not None else None
                mid_broadcast_shape = _broadcast_static_shapes(in_shape0, in_shape1)
                if mid_broadcast_shape is not None and not _shapes_match_if_known(mid_broadcast_shape, mid_expected_shape):
                    continue

                out_shape = list(out_tensor.shape) if out_tensor is not None else None
                if not _shapes_match_if_known(out_shape, mid_expected_shape):
                    continue

                valid_post_shapes = True
                for post_output_name in post_output_names:
                    post_tensor = model_ir.tensors.get(post_output_name, None)
                    post_shape = list(post_tensor.shape) if post_tensor is not None else None
                    if not _shapes_match_if_known(post_shape, raw_broadcast_shape):
                        valid_post_shapes = False
                        break
                if not valid_post_shapes:
                    continue

            canonical_post_output = str(post_output_names[0])
            canonical_tensor = model_ir.tensors.get(canonical_post_output, None)
            keep_post_idx = int(post_indices[0])
            keep_post_op = model_ir.operators[keep_post_idx]
            keep_post_perm_name = str(keep_post_op.inputs[1])

            _set_operator_inputs(
                model_ir=model_ir,
                op=unary_op,
                new_inputs=[raw_unary_input_name],
            )

            if unary_on_lhs:
                new_binary_inputs = [unary_input_name, raw_other_input_name]
            else:
                new_binary_inputs = [raw_other_input_name, unary_input_name]
            _set_operator_inputs(
                model_ir=model_ir,
                op=mid_op,
                new_inputs=new_binary_inputs,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=mid_op,
                new_outputs=[canonical_post_output],
            )

            unary_out_tensor = model_ir.tensors.get(unary_input_name, None)
            if unary_out_tensor is not None and raw_unary_tensor is not None:
                unary_out_tensor.shape = [int(v) for v in list(raw_unary_tensor.shape)]
                unary_out_tensor.shape_signature = (
                    [int(v) for v in list(raw_unary_tensor.shape_signature)]
                    if raw_unary_tensor.shape_signature is not None
                    else [int(v) for v in list(raw_unary_tensor.shape)]
                )

            if canonical_tensor is not None and raw_broadcast_shape is not None:
                canonical_tensor.shape = [int(v) for v in list(raw_broadcast_shape)]
                raw_sig_unary = (
                    list(raw_unary_tensor.shape_signature)
                    if raw_unary_tensor is not None and raw_unary_tensor.shape_signature is not None
                    else list(raw_shape_unary) if raw_shape_unary is not None else None
                )
                raw_sig_other = (
                    list(raw_other_tensor.shape_signature)
                    if raw_other_tensor is not None and raw_other_tensor.shape_signature is not None
                    else list(raw_shape_other) if raw_shape_other is not None else None
                )
                raw_broadcast_signature = _broadcast_shape_signatures(raw_sig_unary, raw_sig_other)
                if raw_broadcast_signature is None:
                    raw_broadcast_signature = [int(v) for v in list(raw_broadcast_shape)]
                canonical_tensor.shape_signature = [int(v) for v in list(raw_broadcast_signature)]
                if out_tensor is not None:
                    canonical_tensor.dtype = str(out_tensor.dtype)
                    canonical_tensor.quantization = _clone_quantization(out_tensor.quantization)

            for post_output_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, post_output_name, canonical_post_output)

            if len(legacy_users) > 0:
                # Keep one adapter transpose for legacy NCHW consumers.
                keep_perm_tensor = model_ir.tensors.get(keep_post_perm_name, None)
                if keep_perm_tensor is not None:
                    keep_perm_tensor.data = np.asarray(perm_pre, dtype=np.int32)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_inputs=[canonical_post_output, keep_post_perm_name],
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_outputs=[out_name],
                )
                post_remove_indices = [int(v) for v in post_indices[1:]]
            else:
                post_remove_indices = [int(v) for v in post_indices]

            remove_indices = sorted(
                list(set([int(pre_unary_idx), int(pre_other_idx)] + post_remove_indices)),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_transpose_unary_binary_full_post_fanout_bridges": int(rewritten)}



def _optimize_transpose_gather_transpose_axis_remap_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Remove NHWC<->NCHW transpose wrappers around GATHER by remapping axis.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --GATHER(axis=a_nchw, batchDims=0)--> g_nchw
      g_nchw --TRANSPOSE(0,2,3,1)--> y_nhwc

    Rewrite:
      x_nhwc --GATHER(axis=a_nhwc, batchDims=0)--> y_nhwc

    where a_nhwc = perm_nhwc_to_nchw[a_nchw].
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for gather_idx, gather_op in enumerate(model_ir.operators):
            if str(gather_op.op_type) != "GATHER" or len(gather_op.inputs) < 2 or len(gather_op.outputs) != 1:
                continue
            gather_in_name = str(gather_op.inputs[0])
            gather_out_name = str(gather_op.outputs[0])
            if gather_out_name in model_outputs:
                continue

            gather_options = dict(gather_op.options) if isinstance(gather_op.options, dict) else {}
            batch_dims = int(gather_options.get("batchDims", 0))
            if int(batch_dims) != 0:
                continue

            pre_idx = producers.get(gather_in_name, None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != gather_in_name
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            ):
                continue
            pre_out_name = str(pre_op.outputs[0])

            gather_users = [int(v) for v in consumers.get(gather_out_name, [])]
            if len(gather_users) != 1:
                continue
            post_idx = int(gather_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != gather_out_name
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue

            gather_in_tensor = model_ir.tensors.get(gather_in_name, None)
            if gather_in_tensor is None or len(list(gather_in_tensor.shape)) != 4:
                continue

            axis = int(gather_options.get("axis", 0))
            if axis < 0:
                axis += 4
            if int(axis) < 0 or int(axis) >= 4:
                continue
            remapped_axis = int(perm_nhwc_to_nchw[int(axis)])

            pre_input_name = str(pre_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if post_output_name not in model_ir.tensors:
                gather_out_tensor = model_ir.tensors.get(gather_out_name, None)
                if gather_out_tensor is not None:
                    model_ir.tensors[post_output_name] = TensorIR(
                        name=post_output_name,
                        dtype=str(gather_out_tensor.dtype),
                        shape=[int(v) for v in list(gather_out_tensor.shape)],
                        shape_signature=(
                            [int(v) for v in list(gather_out_tensor.shape_signature)]
                            if gather_out_tensor.shape_signature is not None
                            else [int(v) for v in list(gather_out_tensor.shape)]
                        ),
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(gather_out_tensor.quantization),
                    )

            gather_inputs = [str(v) for v in list(gather_op.inputs)]
            gather_inputs[0] = pre_input_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=gather_op,
                new_inputs=gather_inputs,
                graph_index=graph_index,
            )
            gather_options["axis"] = int(remapped_axis)
            gather_options["batchDims"] = 0
            gather_op.options = gather_options
            _set_operator_outputs(
                model_ir=model_ir,
                op=gather_op,
                new_outputs=[str(post_output_name)],
                graph_index=graph_index,
            )

            remove_indices: List[int] = [int(post_idx)]
            remaining_pre_users = [
                int(v)
                for v in consumers.get(pre_out_name, [])
                if int(v) != int(gather_idx)
            ]
            if len(remaining_pre_users) == 0 and pre_out_name not in model_outputs:
                remove_indices.append(int(pre_idx))
            for remove_idx in sorted(remove_indices, reverse=True):
                graph_index.remove_operator(int(remove_idx))

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_gather_transpose_axis_remap_nhwc_chains": int(optimized)}


def run_transpose_gather_axis_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run strict NHWC Transpose/Gather axis remapping."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "GATHER"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if len(required) == 0:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        for gather_op in candidate_model.operators:
            if (
                str(gather_op.op_type) != "GATHER"
                or len(gather_op.inputs) < 2
                or len(gather_op.outputs) != 1
                or int(gather_op.options.get("batchDims", 0)) != 0
            ):
                continue
            pre_op = pass_state.graph_index.producer(str(gather_op.inputs[0]))
            if (
                pre_op is None
                or str(pre_op.op_type) != "TRANSPOSE"
                or _read_transpose_perm(candidate_model, pre_op) != [0, 3, 1, 2]
            ):
                continue
            users = pass_state.graph_index.consumer_indices(str(gather_op.outputs[0]))
            if len(users) != 1:
                continue
            post_op = candidate_model.operators[int(users[0])]
            if (
                str(post_op.op_type) == "TRANSPOSE"
                and _read_transpose_perm(candidate_model, post_op) == [0, 2, 3, 1]
            ):
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_gather_transpose_axis_remap_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_gather_transpose_axis_remap_nhwc_chains",
                    0,
                )
            ),
        }

    default_details = {
        "optimized_transpose_gather_transpose_axis_remap_nhwc_chains": 0,
    }
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.transpose_gather_axis_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
