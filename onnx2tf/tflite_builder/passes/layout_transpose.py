from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
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
