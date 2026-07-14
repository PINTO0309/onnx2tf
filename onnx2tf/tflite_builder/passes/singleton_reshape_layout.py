from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_fully_known_positive_shape,
    _invert_perm,
    _permute_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _rename_tensor_globally,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_singleton_layout_reshape_unary_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Remove singleton-layout RESHAPE wrappers around layout-agnostic unary ops.

    Target:
      x --RESHAPE(layout wrapper)--> a --UNARY--> b --RESHAPE(inverse wrapper)--> y

    Rewrite:
      x --UNARY--> y

    Safety:
    - Both RESHAPEs must be rank-4 NHWC<->NCHW permutations.
    - Permutations must be inverse of each other.
    - The permutation must be memory-order equivalent due singleton dimensions.
    - Chain must be strict linear: pre-reshape output and unary output have one consumer.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    unary_passthrough_ops = {
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
        "ROUND",
    }

    def _is_static_rank4_shape(tensor: Optional[TensorIR]) -> bool:
        if tensor is None:
            return False
        if tensor.shape is None or len(list(tensor.shape)) != 4:
            return False
        if any(int(v) <= 0 for v in list(tensor.shape)):
            return False
        signature = (
            list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else list(tensor.shape)
        )
        if len(signature) != 4:
            return False
        if any(int(v) < 0 for v in signature):
            return False
        return True

    def _detect_layout_perm(src_shape: List[int], dst_shape: List[int]) -> Optional[List[int]]:
        for perm in [perm_nhwc_to_nchw, perm_nchw_to_nhwc]:
            expected = _permute_shape(src_shape, perm)
            if expected is not None and [int(v) for v in list(expected)] == [int(v) for v in list(dst_shape)]:
                return [int(v) for v in list(perm)]
        return None

    def _is_singleton_layout_permute_safe(src_shape: List[int], perm: List[int]) -> bool:
        if len(src_shape) != 4:
            return False
        if [int(v) for v in list(perm)] == perm_nhwc_to_nchw:
            return int(src_shape[3]) == 1 or (int(src_shape[1]) == 1 and int(src_shape[2]) == 1)
        if [int(v) for v in list(perm)] == perm_nchw_to_nhwc:
            return int(src_shape[1]) == 1 or (int(src_shape[2]) == 1 and int(src_shape[3]) == 1)
        return False

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "RESHAPE" or len(pre_op.inputs) < 1 or len(pre_op.outputs) != 1:
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) != 1:
                continue

            unary_idx = int(pre_users[0])
            unary_op = model_ir.operators[int(unary_idx)]
            if str(unary_op.op_type) not in unary_passthrough_ops:
                continue
            if len(unary_op.inputs) != 1 or len(unary_op.outputs) != 1:
                continue
            if str(unary_op.inputs[0]) != pre_output_name:
                continue

            unary_output_name = str(unary_op.outputs[0])
            if unary_output_name in model_outputs:
                continue
            unary_users = [int(v) for v in consumers.get(unary_output_name, [])]
            if len(unary_users) != 1:
                continue

            post_idx = int(unary_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if str(post_op.op_type) != "RESHAPE" or len(post_op.inputs) < 1 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != unary_output_name:
                continue

            post_output_name = str(post_op.outputs[0])

            pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
            pre_output_tensor = model_ir.tensors.get(pre_output_name, None)
            unary_output_tensor = model_ir.tensors.get(unary_output_name, None)
            post_output_tensor = model_ir.tensors.get(post_output_name, None)
            if (
                not _is_static_rank4_shape(pre_input_tensor)
                or not _is_static_rank4_shape(pre_output_tensor)
                or not _is_static_rank4_shape(unary_output_tensor)
                or not _is_static_rank4_shape(post_output_tensor)
            ):
                continue

            pre_input_shape = [int(v) for v in list(pre_input_tensor.shape)]
            pre_output_shape = [int(v) for v in list(pre_output_tensor.shape)]
            unary_output_shape = [int(v) for v in list(unary_output_tensor.shape)]
            post_output_shape = [int(v) for v in list(post_output_tensor.shape)]
            if pre_output_shape != unary_output_shape:
                continue

            perm_pre = _detect_layout_perm(pre_input_shape, pre_output_shape)
            if perm_pre is None:
                continue
            if not _is_singleton_layout_permute_safe(pre_input_shape, perm_pre):
                continue
            perm_post = _detect_layout_perm(pre_output_shape, post_output_shape)
            if perm_post is None:
                continue
            if perm_post != _invert_perm(perm_pre):
                continue
            if not _is_singleton_layout_permute_safe(pre_output_shape, perm_post):
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=unary_op,
                new_inputs=[pre_input_name],
                graph_index=graph_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=unary_op,
                new_outputs=[post_output_name],
                graph_index=graph_index,
            )

            if unary_output_tensor is not None and post_output_tensor is not None:
                post_output_tensor.dtype = str(unary_output_tensor.dtype)
                post_output_tensor.quantization = _clone_quantization(unary_output_tensor.quantization)

            remove_indices: List[int] = []
            pre_remove_idx = next((idx for idx, op in enumerate(model_ir.operators) if op is pre_op), None)
            if pre_remove_idx is not None:
                remove_indices.append(int(pre_remove_idx))
            post_remove_idx = next((idx for idx, op in enumerate(model_ir.operators) if op is post_op), None)
            if post_remove_idx is not None:
                remove_indices.append(int(post_remove_idx))
            for remove_idx in sorted(set(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_singleton_layout_reshape_unary_passthrough_chains": int(rewritten)}

def _optimize_consecutive_inverse_singleton_layout_reshapes(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Remove consecutive inverse NHWC<->NCHW layout RESHAPE pairs.

    Target:
      x --RESHAPE(NHWC->NCHW)--> a --RESHAPE(NCHW->NHWC)--> y
      (or reverse direction)

    Rewrite:
      x --> y

    Safety:
    - Both RESHAPEs must be rank-4 and form inverse canonical layout permutations.
    - Memory-order equivalence must hold due singleton channel or singleton spatial dims.
    - Intermediate tensor must be single-consumer.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _is_static_rank4_shape(tensor: Optional[TensorIR]) -> bool:
        if tensor is None:
            return False
        if tensor.shape is None or len(list(tensor.shape)) != 4:
            return False
        if any(int(v) <= 0 for v in list(tensor.shape)):
            return False
        signature = (
            list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else list(tensor.shape)
        )
        if len(signature) != 4:
            return False
        if any(int(v) < 0 for v in signature):
            return False
        return True

    def _detect_layout_perm(src_shape: List[int], dst_shape: List[int]) -> Optional[List[int]]:
        for perm in [perm_nhwc_to_nchw, perm_nchw_to_nhwc]:
            expected = _permute_shape(src_shape, perm)
            if expected is not None and [int(v) for v in list(expected)] == [int(v) for v in list(dst_shape)]:
                return [int(v) for v in list(perm)]
        return None

    def _is_singleton_layout_permute_safe(src_shape: List[int], perm: List[int]) -> bool:
        if len(src_shape) != 4:
            return False
        if [int(v) for v in list(perm)] == perm_nhwc_to_nchw:
            return int(src_shape[3]) == 1 or (int(src_shape[1]) == 1 and int(src_shape[2]) == 1)
        if [int(v) for v in list(perm)] == perm_nchw_to_nhwc:
            return int(src_shape[1]) == 1 or (int(src_shape[2]) == 1 and int(src_shape[3]) == 1)
        return False

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for first_idx, first_op in enumerate(model_ir.operators):
            if str(first_op.op_type) != "RESHAPE" or len(first_op.inputs) < 1 or len(first_op.outputs) != 1:
                continue

            src_name = str(first_op.inputs[0])
            mid_name = str(first_op.outputs[0])
            mid_users = [int(v) for v in consumers.get(mid_name, [])]
            if len(mid_users) != 1:
                continue

            second_idx = int(mid_users[0])
            second_op = model_ir.operators[int(second_idx)]
            if str(second_op.op_type) != "RESHAPE" or len(second_op.inputs) < 1 or len(second_op.outputs) != 1:
                continue
            if str(second_op.inputs[0]) != mid_name:
                continue

            dst_name = str(second_op.outputs[0])

            src_tensor = model_ir.tensors.get(src_name, None)
            mid_tensor = model_ir.tensors.get(mid_name, None)
            dst_tensor = model_ir.tensors.get(dst_name, None)
            if (
                not _is_static_rank4_shape(src_tensor)
                or not _is_static_rank4_shape(mid_tensor)
                or not _is_static_rank4_shape(dst_tensor)
            ):
                continue

            src_shape = [int(v) for v in list(src_tensor.shape)]
            mid_shape = [int(v) for v in list(mid_tensor.shape)]
            dst_shape = [int(v) for v in list(dst_tensor.shape)]

            perm_1 = _detect_layout_perm(src_shape, mid_shape)
            perm_2 = _detect_layout_perm(mid_shape, dst_shape)
            if perm_1 is None or perm_2 is None:
                continue
            if perm_2 != _invert_perm(perm_1):
                continue
            if not _is_singleton_layout_permute_safe(src_shape, perm_1):
                continue
            if not _is_singleton_layout_permute_safe(mid_shape, perm_2):
                continue

            # Keep graph-output tensor names stable when possible.
            if dst_name in model_outputs:
                src_users = [int(v) for v in consumers.get(src_name, [])]
                if (
                    src_name in model_ir.inputs
                    or src_name in model_outputs
                    or set(src_users) != {int(first_idx)}
                ):
                    continue
                _rename_tensor_globally(
                    model_ir=model_ir,
                    old_name=str(src_name),
                    new_name=str(dst_name),
                    layout_state=layout_state,
                    graph_index=graph_index,
                )
            else:
                _replace_tensor_inputs(
                    model_ir=model_ir,
                    src_name=str(dst_name),
                    dst_name=str(src_name),
                    graph_index=graph_index,
                )

            for remove_idx in sorted([int(first_idx), int(second_idx)], reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_consecutive_inverse_singleton_layout_reshapes": int(rewritten)}


def run_singleton_reshape_layout_cleanup(
    model_ir: ModelIR,
    *,
    include_unary_passthrough: bool = True,
    include_inverse_pair: bool = True,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Run adjacent singleton-Reshape cleanups in legacy order."""

    unary_passthrough_ops = {
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
        "ROUND",
    }

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        reshape_count = 0
        for visited, operator in enumerate(candidate_model.operators, start=1):
            if str(operator.op_type) == "RESHAPE":
                reshape_count += 1
                if reshape_count >= 2:
                    return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _single_user(
        pass_state: ModelIRPassState,
        tensor_name: str,
    ) -> tuple[int, OperatorIR] | None:
        users = pass_state.graph_index.consumer_indices(str(tensor_name))
        if len(users) != 1:
            return None
        index = int(users[0])
        return index, pass_state.model_ir.operators[index]

    def _has_unary_candidate(pass_state: ModelIRPassState) -> bool:
        for pre_op in pass_state.model_ir.operators:
            if str(pre_op.op_type) != "RESHAPE" or len(pre_op.outputs) != 1:
                continue
            unary = _single_user(pass_state, str(pre_op.outputs[0]))
            if (
                unary is None
                or str(unary[1].op_type) not in unary_passthrough_ops
                or len(unary[1].outputs) != 1
            ):
                continue
            post = _single_user(pass_state, str(unary[1].outputs[0]))
            if post is not None and str(post[1].op_type) == "RESHAPE":
                return True
        return False

    def _has_inverse_pair_candidate(pass_state: ModelIRPassState) -> bool:
        for first_op in pass_state.model_ir.operators:
            if str(first_op.op_type) != "RESHAPE" or len(first_op.outputs) != 1:
                continue
            second = _single_user(pass_state, str(first_op.outputs[0]))
            if second is not None and str(second[1].op_type) == "RESHAPE":
                return True
        return False

    def _run_unary(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_singleton_layout_reshape_unary_passthrough_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "rewritten_singleton_layout_reshape_unary_passthrough_chains",
                    0,
                )
            ),
        }

    def _run_inverse_pair(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_consecutive_inverse_singleton_layout_reshapes(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("rewritten_consecutive_inverse_singleton_layout_reshapes", 0)
            ),
        }

    specs: List[PassSpec[ModelIRPassState]] = []
    if include_unary_passthrough:
        specs.append(
            PassSpec(
                pass_id="layout.singleton_reshape_unary_passthrough",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_unary,
                precondition=_has_unary_candidate,
                priority=10,
                transactional=True,
            )
        )
    if include_inverse_pair:
        specs.append(
            PassSpec(
                pass_id="layout.consecutive_inverse_singleton_reshapes",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_inverse_pair,
                precondition=_has_inverse_pair_candidate,
                priority=20,
                transactional=True,
            )
        )
    default_details = {
        "rewritten_singleton_layout_reshape_unary_passthrough_chains": 0,
        "rewritten_consecutive_inverse_singleton_layout_reshapes": 0,
    }
    if len(specs) == 0:
        return default_details
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_singleton_channel_layout_transpose_to_reshape(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Replace TRANSPOSE with RESHAPE when permutation is memory-order
    equivalent due to singleton dimensions.

    Safe cases:
    - NHWC -> NCHW with input shape [N,H,W,1]
    - NCHW -> NHWC with input shape [N,1,H,W]
    - NHWC -> NCHW with input shape [N,1,1,C]
    - NCHW -> NHWC with input shape [N,C,1,1]
    In these rank-4 layout cases, moved axes are singleton and the transpose
    is an exact reshape.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    consumers = graph_index.consumers
    producers = graph_index.producers

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _is_memory_order_equivalent_by_singletons(
        *,
        input_signature: List[int],
        perm: List[int],
    ) -> bool:
        rank = len(input_signature)
        if rank < 2 or len(perm) != rank:
            return False
        if any(int(v) <= 0 for v in input_signature):
            return False
        non_singleton_input_axes = [
            int(axis_idx)
            for axis_idx, dim in enumerate(input_signature)
            if int(dim) != 1
        ]
        non_singleton_permuted_axes = [
            int(axis_idx)
            for axis_idx in list(perm)
            if int(input_signature[int(axis_idx)]) != 1
        ]
        return non_singleton_input_axes == non_singleton_permuted_axes

    def _feeds_quantized_logistic_concat_axis1_chain(transpose_out_name: str) -> bool:
        """
        Keep transpose form for detection-head style paths:
          TRANSPOSE -> DQ -> LOGISTIC -> Q -> DQ -> CONCAT(axis=1)
        where concat has any non-singleton channel sibling input.
        """
        lv1_users = [int(v) for v in consumers.get(str(transpose_out_name), [])]
        if len(lv1_users) != 1:
            return False
        dq1_op = model_ir.operators[int(lv1_users[0])]
        if (
            str(dq1_op.op_type) != "DEQUANTIZE"
            or len(dq1_op.inputs) != 1
            or len(dq1_op.outputs) != 1
            or str(dq1_op.inputs[0]) != str(transpose_out_name)
        ):
            return False
        dq1_out = str(dq1_op.outputs[0])

        lv2_users = [int(v) for v in consumers.get(dq1_out, [])]
        if len(lv2_users) != 1:
            return False
        logistic_op = model_ir.operators[int(lv2_users[0])]
        if (
            str(logistic_op.op_type) != "LOGISTIC"
            or len(logistic_op.inputs) != 1
            or len(logistic_op.outputs) != 1
            or str(logistic_op.inputs[0]) != dq1_out
        ):
            return False
        logistic_out = str(logistic_op.outputs[0])

        lv3_users = [int(v) for v in consumers.get(logistic_out, [])]
        if len(lv3_users) != 1:
            return False
        q_op = model_ir.operators[int(lv3_users[0])]
        if (
            str(q_op.op_type) != "QUANTIZE"
            or len(q_op.inputs) != 1
            or len(q_op.outputs) != 1
            or str(q_op.inputs[0]) != logistic_out
        ):
            return False
        q_out = str(q_op.outputs[0])

        lv4_users = [int(v) for v in consumers.get(q_out, [])]
        if len(lv4_users) != 1:
            return False
        dq2_op = model_ir.operators[int(lv4_users[0])]
        if (
            str(dq2_op.op_type) != "DEQUANTIZE"
            or len(dq2_op.inputs) != 1
            or len(dq2_op.outputs) != 1
            or str(dq2_op.inputs[0]) != q_out
        ):
            return False
        dq2_out = str(dq2_op.outputs[0])

        lv5_users = [int(v) for v in consumers.get(dq2_out, [])]
        if len(lv5_users) != 1:
            return False
        concat_op = model_ir.operators[int(lv5_users[0])]
        if (
            str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.inputs) < 2
            or str(dq2_out) not in set(str(v) for v in list(concat_op.inputs))
        ):
            return False
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 4
        if concat_axis != 1:
            return False

        # If siblings include non-singleton channel inputs, this branch is part
        # of an NCHW concat contract and downstream shape passes can optimize it.
        for input_name in [str(v) for v in list(concat_op.inputs)]:
            if str(input_name) == str(dq2_out):
                continue
            input_tensor = model_ir.tensors.get(str(input_name), None)
            if input_tensor is None or len(list(input_tensor.shape)) != 4:
                continue
            if int(input_tensor.shape[1]) > 1:
                return True
        return False

    for op in model_ir.operators:
        if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 2 or len(op.outputs) != 1:
            continue

        perm = _read_transpose_perm(model_ir, op)
        if perm is None:
            continue

        input_name = str(op.inputs[0])
        output_name = str(op.outputs[0])
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        if (
            input_tensor is None
            or output_tensor is None
            or not _is_fully_known_positive_shape(input_tensor.shape)
            or not _is_fully_known_positive_shape(output_tensor.shape)
        ):
            continue

        input_shape = [int(v) for v in list(input_tensor.shape)]
        output_shape = [int(v) for v in list(output_tensor.shape)]
        rank = len(input_shape)
        # This pass is intended only for NHWC<->NCHW style rank-4 layout bridges.
        # Rank-3 permutations (e.g. [0,2,1]) can be axis-semantic transforms and
        # rewriting them to RESHAPE may break downstream ops (e.g. LogSoftmax/Add).
        if rank != 4:
            continue
        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else [int(v) for v in list(input_shape)]
        )
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_shape)]
        )
        if len(output_shape) != rank or len(list(perm)) != rank:
            continue
        input_producer_idx = producers.get(input_name, None)
        input_producer = (
            model_ir.operators[int(input_producer_idx)]
            if input_producer_idx is not None
            and 0 <= int(input_producer_idx) < len(model_ir.operators)
            else None
        )
        obsolete_post_mean_layout_adapter = bool(
            perm == perm_nchw_to_nhwc
            and input_producer is not None
            and str(input_producer.op_type) == "MEAN"
            and isinstance(input_producer.options, dict)
            and bool(
                input_producer.options.get(
                    "__convpool_output_nhwc_axes_remapped__",
                    False,
                )
            )
        )
        if obsolete_post_mean_layout_adapter:
            output_shape = [int(v) for v in input_shape]
            output_signature = [int(v) for v in input_signature]
            output_tensor.shape = list(output_shape)
            output_tensor.shape_signature = list(output_signature)
        # Dynamic dimensions can be represented as placeholder static 1s in `shape`.
        # Replacing transpose with reshape is only safe when the moved layout axes
        # are still provably singleton in signatures. Rank-4 layout bridges keep
        # batch at axis 0, so a dynamic batch dimension there remains safe.
        if (
            len(input_signature) != rank
            or len(output_signature) != rank
            or any(int(v) < 0 for idx, v in enumerate(input_signature) if int(idx) != 0)
            or any(int(v) < 0 for idx, v in enumerate(output_signature) if int(idx) != 0)
        ):
            continue

        canonical_singleton_safe = False
        if perm == perm_nhwc_to_nchw:
            channel_singleton = int(input_shape[3]) == 1 and int(output_shape[1]) == 1
            spatial_singleton = (
                int(input_shape[1]) == 1
                and int(input_shape[2]) == 1
                and int(output_shape[2]) == 1
                and int(output_shape[3]) == 1
            )
            canonical_singleton_safe = bool(channel_singleton or spatial_singleton)
        elif perm == perm_nchw_to_nhwc:
            channel_singleton = int(input_shape[1]) == 1 and int(output_shape[3]) == 1
            spatial_singleton = (
                int(input_shape[2]) == 1
                and int(input_shape[3]) == 1
                and int(output_shape[1]) == 1
                and int(output_shape[2]) == 1
            )
            canonical_singleton_safe = bool(channel_singleton or spatial_singleton)

        expected_output_shape = (
            list(input_shape)
            if obsolete_post_mean_layout_adapter
            else _permute_shape(input_shape, perm)
        )
        if expected_output_shape is None or [int(v) for v in list(expected_output_shape)] != output_shape:
            continue
        expected_output_signature = (
            list(input_signature)
            if obsolete_post_mean_layout_adapter
            else _permute_shape(input_signature, perm)
        )
        if (
            expected_output_signature is None
            or [int(v) for v in list(expected_output_signature)] != output_signature
        ):
            continue

        if not canonical_singleton_safe and not _is_memory_order_equivalent_by_singletons(
            input_signature=[int(v) for v in list(input_signature)],
            perm=[int(v) for v in list(perm)],
        ):
            continue

        # Preserve transpose (do not downcast to reshape) for quantized
        # detection-head singleton gates that feed NCHW concat chains.
        if (
            rank == 4
            and [int(v) for v in list(perm)] == perm_nhwc_to_nchw
            and int(input_shape[3]) == 1
            and _feeds_quantized_logistic_concat_axis1_chain(str(output_name))
        ):
            continue

        reshape_target = [int(v) for v in output_shape]
        if int(output_signature[0]) < 0:
            reshape_target[0] = -1
        reshape_shape_name = _unique_tensor_name(f"{output_name}_reshape_shape")
        model_ir.tensors[reshape_shape_name] = TensorIR(
            name=reshape_shape_name,
            dtype="INT32",
            shape=[len(reshape_target)],
            shape_signature=[len(reshape_target)],
            data=np.asarray(reshape_target, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )

        op.op_type = "RESHAPE"
        _set_operator_inputs(
            model_ir=model_ir,
            op=op,
            new_inputs=[input_name, reshape_shape_name],
            graph_index=graph_index,
        )
        op.options = {
            "newShape": [int(v) for v in reshape_target],
            "layoutTransposeAsReshape": True,
            "layoutTransposePerm": [int(v) for v in perm],
        }
        rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_singleton_channel_layout_transpose_to_reshape": int(rewritten)}


def _optimize_singleton_spatial_nhwc_transpose_reshape_flatten(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Remove NHWC->NCHW transpose before 2D flatten reshape when spatial dims are 1x1.

    Pattern:
      x_nhwc [N,1,1,C] -> TRANSPOSE(0,3,1,2) -> x_nchw [N,C,1,1]
      x_nchw -> RESHAPE -> y [N,C]

    Extended pattern:
      x_nhwc [N,1,1,C] -> TRANSPOSE(0,3,1,2) -> x_nchw [N,C,1,1]
      x_nchw -> RESHAPE(identity) -> x_keep [N,C,1,1]
      x_keep -> RESHAPE -> y [N,C]

    For singleton spatial dims, flatten result is identical without transpose:
      RESHAPE(x_nhwc) -> y
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for transpose_idx, transpose_op in enumerate(model_ir.operators):
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.inputs) < 2
                or len(transpose_op.outputs) != 1
                or _read_transpose_perm(model_ir, transpose_op) != perm_nhwc_to_nchw
            ):
                continue

            transpose_input_name = str(transpose_op.inputs[0])
            transpose_output_name = str(transpose_op.outputs[0])
            users = [int(v) for v in consumers.get(transpose_output_name, [])]
            if len(users) != 1:
                continue

            first_reshape_idx = int(users[0])
            first_reshape_op = model_ir.operators[int(first_reshape_idx)]
            if (
                str(first_reshape_op.op_type) != "RESHAPE"
                or len(first_reshape_op.inputs) < 1
                or str(first_reshape_op.inputs[0]) != transpose_output_name
                or len(first_reshape_op.outputs) != 1
            ):
                continue

            target_reshape_idx = int(first_reshape_idx)
            target_reshape_op = first_reshape_op
            target_reshape_output_name = str(target_reshape_op.outputs[0])
            intermediate_reshape_idx: Optional[int] = None
            intermediate_reshape_output_name = ""

            first_reshape_output_name = str(first_reshape_op.outputs[0])
            first_reshape_output_tensor = model_ir.tensors.get(first_reshape_output_name, None)
            if (
                first_reshape_output_tensor is not None
                and len(list(first_reshape_output_tensor.shape)) == 4
                and first_reshape_output_name not in model_outputs
            ):
                second_users = [int(v) for v in consumers.get(first_reshape_output_name, [])]
                if len(second_users) == 1:
                    second_reshape_idx = int(second_users[0])
                    second_reshape_op = model_ir.operators[int(second_reshape_idx)]
                    if (
                        str(second_reshape_op.op_type) == "RESHAPE"
                        and len(second_reshape_op.inputs) >= 1
                        and str(second_reshape_op.inputs[0]) == first_reshape_output_name
                        and len(second_reshape_op.outputs) == 1
                    ):
                        second_reshape_output_name = str(second_reshape_op.outputs[0])
                        second_reshape_output_tensor = model_ir.tensors.get(second_reshape_output_name, None)
                        if (
                            second_reshape_output_tensor is not None
                            and len(list(second_reshape_output_tensor.shape)) == 2
                        ):
                            target_reshape_idx = int(second_reshape_idx)
                            target_reshape_op = second_reshape_op
                            target_reshape_output_name = second_reshape_output_name
                            intermediate_reshape_idx = int(first_reshape_idx)
                            intermediate_reshape_output_name = first_reshape_output_name

            input_tensor = model_ir.tensors.get(transpose_input_name, None)
            output_tensor = model_ir.tensors.get(transpose_output_name, None)
            reshape_output_tensor = model_ir.tensors.get(target_reshape_output_name, None)
            if (
                input_tensor is None
                or output_tensor is None
                or reshape_output_tensor is None
                or not _is_fully_known_positive_shape(input_tensor.shape)
                or not _is_fully_known_positive_shape(output_tensor.shape)
                or not _is_fully_known_positive_shape(reshape_output_tensor.shape)
            ):
                continue

            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            reshape_output_shape = [int(v) for v in list(reshape_output_tensor.shape)]
            if len(input_shape) != 4 or len(output_shape) != 4 or len(reshape_output_shape) != 2:
                continue
            if int(input_shape[1]) != 1 or int(input_shape[2]) != 1:
                continue
            expected_output_shape = _permute_shape(input_shape, perm_nhwc_to_nchw)
            if expected_output_shape is None or [int(v) for v in list(expected_output_shape)] != output_shape:
                continue
            if intermediate_reshape_idx is not None:
                intermediate_tensor = model_ir.tensors.get(intermediate_reshape_output_name, None)
                if intermediate_tensor is None:
                    continue
                intermediate_shape = [int(v) for v in list(intermediate_tensor.shape)]
                if intermediate_shape != output_shape:
                    continue
            if int(reshape_output_shape[1]) != int(input_shape[3]):
                continue

            remove_intermediate = bool(
                intermediate_reshape_idx is not None
                and intermediate_reshape_output_name not in model_outputs
                and set(
                    int(v)
                    for v in consumers.get(intermediate_reshape_output_name, [])
                )
                == {int(target_reshape_idx)}
            )
            reshape_inputs = [str(v) for v in list(target_reshape_op.inputs)]
            reshape_inputs[0] = str(transpose_input_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=target_reshape_op,
                new_inputs=reshape_inputs,
                graph_index=graph_index,
            )
            remove_indices = [int(transpose_idx)]
            if remove_intermediate and intermediate_reshape_idx is not None:
                remove_indices.append(int(intermediate_reshape_idx))
            for remove_idx in sorted(set(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_singleton_spatial_nhwc_transpose_reshape_flatten": int(rewritten)}


def _optimize_singleton_reshape_concat_post_transpose_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Rewrite singleton-channel NCHW concat blocks back to NHWC and remove trailing transpose.

    Target:
      x_i_nhwc --RESHAPE([N,1,H,W])--> x_i_nchw (subset of inputs; some may be reused elsewhere)
      c_nchw --CONCAT(axis=1)--> y_nchw --TRANSPOSE(0,2,3,1)--> y_nhwc

    Rewrite:
      - CONCAT inputs switch to NHWC aliases (reshape sources when available)
      - direct singleton NCHW inputs receive local NCHW->NHWC RESHAPE adapters
      - CONCAT axis becomes 3
      - trailing TRANSPOSE is removed and CONCAT directly produces y_nhwc
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_nhwc_to_nchw = [0, 3, 1, 2]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx, post_op in enumerate(model_ir.operators):
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue

            concat_output_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if concat_output_name in model_outputs:
                continue

            concat_idx = producers.get(concat_output_name, None)
            if concat_idx is None:
                continue
            concat_op = model_ir.operators[int(concat_idx)]
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue
            if str(concat_op.outputs[0]) != concat_output_name:
                continue
            concat_axis = int(concat_op.options.get("axis", 1))
            if concat_axis < 0:
                concat_axis += 4
            if concat_axis != 1:
                continue
            if set(int(v) for v in consumers.get(concat_output_name, [])) != {int(post_idx)}:
                continue

            concat_inputs = [str(v) for v in list(concat_op.inputs)]
            if len(concat_inputs) < 2:
                continue

            mapped_concat_inputs: List[str] = []
            reshape_ops_to_remove: List[OperatorIR] = []
            adapters_to_insert: List[Tuple[str, str, str, List[int]]] = []
            adapter_by_source: Dict[str, str] = {}
            valid = True

            for input_name in concat_inputs:
                input_tensor = model_ir.tensors.get(str(input_name), None)
                if (
                    input_tensor is None
                    or not _is_fully_known_positive_shape(input_tensor.shape)
                    or len(list(input_tensor.shape)) != 4
                ):
                    valid = False
                    break
                input_shape = [int(v) for v in list(input_tensor.shape)]

                input_prod_idx = producers.get(str(input_name), None)
                if input_prod_idx is not None:
                    input_prod_op = model_ir.operators[int(input_prod_idx)]
                    if (
                        str(input_prod_op.op_type) == "RESHAPE"
                        and len(input_prod_op.inputs) >= 1
                        and len(input_prod_op.outputs) == 1
                        and str(input_prod_op.outputs[0]) == str(input_name)
                    ):
                        source_name = str(input_prod_op.inputs[0])
                        source_tensor = model_ir.tensors.get(source_name, None)
                        if (
                            source_tensor is not None
                            and _is_fully_known_positive_shape(source_tensor.shape)
                            and len(list(source_tensor.shape)) == 4
                        ):
                            source_shape = [int(v) for v in list(source_tensor.shape)]
                            expected_shape = _permute_shape(source_shape, perm_nhwc_to_nchw)
                            if (
                                expected_shape is not None
                                and [int(v) for v in list(expected_shape)] == input_shape
                                and int(source_shape[3]) == 1
                                and int(input_shape[1]) == 1
                            ):
                                mapped_concat_inputs.append(str(source_name))
                                input_users = [int(v) for v in consumers.get(str(input_name), [])]
                                if set(input_users) == {int(concat_idx)}:
                                    reshape_ops_to_remove.append(input_prod_op)
                                continue

                # Direct NCHW singleton input: create local NCHW->NHWC reshape adapter.
                if int(input_shape[1]) != 1:
                    valid = False
                    break
                nhwc_shape = _permute_shape(input_shape, perm_nchw_to_nhwc)
                if nhwc_shape is None or len(nhwc_shape) != 4 or int(nhwc_shape[3]) != 1:
                    valid = False
                    break

                existing_adapter_name = adapter_by_source.get(str(input_name), None)
                if existing_adapter_name is not None:
                    mapped_concat_inputs.append(str(existing_adapter_name))
                    continue

                adapter_name = _unique_tensor_name(f"{input_name}_nhwc_adapter")
                shape_name = _unique_tensor_name(f"{adapter_name}_reshape_shape")
                adapter_by_source[str(input_name)] = str(adapter_name)
                adapters_to_insert.append(
                    (
                        str(input_name),
                        str(adapter_name),
                        str(shape_name),
                        [int(v) for v in list(nhwc_shape)],
                    )
                )
                mapped_concat_inputs.append(str(adapter_name))

            if not valid:
                continue

            # Insert local adapters right before concat.
            if len(adapters_to_insert) > 0:
                concat_pos = int(model_ir.operators.index(concat_op))
                insert_pos = int(concat_pos)
                for source_name, adapter_name, shape_name, nhwc_shape in adapters_to_insert:
                    source_tensor = model_ir.tensors.get(source_name, None)
                    source_dtype = str(source_tensor.dtype) if source_tensor is not None else "FLOAT32"
                    source_quant = (
                        _clone_quantization(source_tensor.quantization)
                        if source_tensor is not None
                        else None
                    )
                    model_ir.tensors[shape_name] = TensorIR(
                        name=str(shape_name),
                        dtype="INT32",
                        shape=[len(nhwc_shape)],
                        shape_signature=[len(nhwc_shape)],
                        data=np.asarray(nhwc_shape, dtype=np.int32),
                        is_variable=False,
                        quantization=None,
                    )
                    model_ir.tensors[adapter_name] = TensorIR(
                        name=str(adapter_name),
                        dtype=str(source_dtype),
                        shape=[int(v) for v in list(nhwc_shape)],
                        shape_signature=[int(v) for v in list(nhwc_shape)],
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(source_quant),
                    )
                    graph_index.insert_operator(
                        int(insert_pos),
                        OperatorIR(
                            op_type="RESHAPE",
                            inputs=[str(source_name), str(shape_name)],
                            outputs=[str(adapter_name)],
                            options={"newShape": [int(v) for v in list(nhwc_shape)]},
                        ),
                    )
                    insert_pos += 1

            # Re-find operators after insertion.
            concat_op = next((op for op in model_ir.operators if op is concat_op), None)
            post_op = next((op for op in model_ir.operators if op is post_op), None)
            if concat_op is None or post_op is None:
                continue

            concat_old_output_name = str(concat_op.outputs[0])
            concat_op.options = {
                **(dict(concat_op.options) if isinstance(concat_op.options, dict) else {}),
                "axis": 3,
            }
            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=[str(v) for v in mapped_concat_inputs],
                graph_index=graph_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=concat_op,
                new_outputs=[str(post_output_name)],
                graph_index=graph_index,
            )
            _replace_tensor_inputs(
                model_ir,
                concat_old_output_name,
                post_output_name,
                graph_index=graph_index,
            )

            post_out_tensor = model_ir.tensors.get(str(post_output_name), None)
            if post_out_tensor is not None and post_out_tensor.shape is not None and len(list(post_out_tensor.shape)) == 4:
                # Keep transpose output metadata authoritative.
                pass
            else:
                concat_old_tensor = model_ir.tensors.get(str(concat_old_output_name), None)
                if concat_old_tensor is not None and concat_old_tensor.shape is not None:
                    mapped_shape = _permute_shape(
                        [int(v) for v in list(concat_old_tensor.shape)],
                        perm_nchw_to_nhwc,
                    )
                    if mapped_shape is not None:
                        model_ir.tensors[str(post_output_name)] = TensorIR(
                            name=str(post_output_name),
                            dtype=str(concat_old_tensor.dtype),
                            shape=[int(v) for v in list(mapped_shape)],
                            shape_signature=[int(v) for v in list(mapped_shape)],
                            data=None,
                            is_variable=False,
                            quantization=_clone_quantization(concat_old_tensor.quantization),
                        )

            remove_indices: List[int] = []
            post_index = next((idx for idx, op in enumerate(model_ir.operators) if op is post_op), None)
            if post_index is not None:
                remove_indices.append(int(post_index))
            for reshape_op in reshape_ops_to_remove:
                reshape_index = next((idx for idx, op in enumerate(model_ir.operators) if op is reshape_op), None)
                if reshape_index is not None:
                    remove_indices.append(int(reshape_index))
            for remove_idx in sorted(set(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_singleton_reshape_concat_post_transpose_nhwc_chains": int(rewritten)}


def _optimize_flatten_concat_expanddims_to_nhwc_concat(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Rewrite 4D->2D flatten concat and 2D->4D reshape back to direct NHWC concat.

    Target:
      a4d[N,1,1,Ca] --RESHAPE--> a2d[N,Ca]
      b2d[N,Cb]
      CONCAT(axis=1, [a2d,b2d]) -> c2d[N,Ca+Cb]
      RESHAPE -> c4d[N,1,1,Ca+Cb]

    Rewrite:
      b2d --RESHAPE--> b4d[N,1,1,Cb]
      CONCAT(axis=3, [a4d,b4d]) -> c4d
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "RESHAPE" or len(pre_op.inputs) < 1 or len(pre_op.outputs) != 1:
                continue

            a4d_name = str(pre_op.inputs[0])
            a2d_name = str(pre_op.outputs[0])
            if a2d_name in model_outputs:
                continue

            concat_users = [int(v) for v in consumers.get(a2d_name, [])]
            if len(concat_users) != 1:
                continue
            concat_idx = int(concat_users[0])
            concat_op = model_ir.operators[int(concat_idx)]
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue
            concat_axis = int(concat_op.options.get("axis", 1))
            if concat_axis < 0:
                concat_axis += 2
            if concat_axis != 1:
                continue
            concat_inputs = [str(v) for v in list(concat_op.inputs)]
            if len(concat_inputs) != 2 or a2d_name not in concat_inputs:
                continue

            c2d_name = str(concat_op.outputs[0])
            if c2d_name in model_outputs:
                continue
            c2d_users = [int(v) for v in consumers.get(c2d_name, [])]
            if len(c2d_users) != 1:
                continue
            post_idx = int(c2d_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if str(post_op.op_type) != "RESHAPE" or len(post_op.inputs) < 1 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != c2d_name:
                continue
            c4d_name = str(post_op.outputs[0])

            a4d_tensor = model_ir.tensors.get(a4d_name, None)
            a2d_tensor = model_ir.tensors.get(a2d_name, None)
            c2d_tensor = model_ir.tensors.get(c2d_name, None)
            c4d_tensor = model_ir.tensors.get(c4d_name, None)
            if (
                a4d_tensor is None
                or a2d_tensor is None
                or c2d_tensor is None
                or c4d_tensor is None
                or not _is_fully_known_positive_shape(a4d_tensor.shape)
                or not _is_fully_known_positive_shape(a2d_tensor.shape)
                or not _is_fully_known_positive_shape(c2d_tensor.shape)
                or not _is_fully_known_positive_shape(c4d_tensor.shape)
            ):
                continue

            a4d_shape = [int(v) for v in list(a4d_tensor.shape)]
            a2d_shape = [int(v) for v in list(a2d_tensor.shape)]
            c2d_shape = [int(v) for v in list(c2d_tensor.shape)]
            c4d_shape = [int(v) for v in list(c4d_tensor.shape)]
            if len(a4d_shape) != 4 or len(a2d_shape) != 2 or len(c2d_shape) != 2 or len(c4d_shape) != 4:
                continue
            if int(a4d_shape[1]) != 1 or int(a4d_shape[2]) != 1:
                continue
            if int(c4d_shape[1]) != 1 or int(c4d_shape[2]) != 1:
                continue
            if int(a2d_shape[0]) != int(a4d_shape[0]) or int(a2d_shape[1]) != int(a4d_shape[3]):
                continue
            if int(c4d_shape[0]) != int(c2d_shape[0]) or int(c4d_shape[3]) != int(c2d_shape[1]):
                continue

            b2d_name = str(concat_inputs[0]) if str(concat_inputs[1]) == a2d_name else str(concat_inputs[1])
            b2d_tensor = model_ir.tensors.get(b2d_name, None)
            if (
                b2d_tensor is None
                or not _is_fully_known_positive_shape(b2d_tensor.shape)
                or len(list(b2d_tensor.shape)) != 2
            ):
                continue
            b2d_shape = [int(v) for v in list(b2d_tensor.shape)]
            if int(b2d_shape[0]) != int(a4d_shape[0]):
                continue

            expected_c = int(a4d_shape[3]) + int(b2d_shape[1])
            if int(c4d_shape[3]) != expected_c:
                continue

            b4d_shape = [int(b2d_shape[0]), 1, 1, int(b2d_shape[1])]
            b4d_name = _unique_tensor_name(f"{b2d_name}_nhwc")
            b4d_shape_name = _unique_tensor_name(f"{b4d_name}_reshape_shape")
            model_ir.tensors[b4d_shape_name] = TensorIR(
                name=str(b4d_shape_name),
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray(b4d_shape, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
            model_ir.tensors[b4d_name] = TensorIR(
                name=str(b4d_name),
                dtype=str(b2d_tensor.dtype),
                shape=[int(v) for v in list(b4d_shape)],
                shape_signature=[int(v) for v in list(b4d_shape)],
                data=None,
                is_variable=False,
                quantization=_clone_quantization(b2d_tensor.quantization),
            )
            concat_pos = int(model_ir.operators.index(concat_op))
            graph_index.insert_operator(
                int(concat_pos),
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[str(b2d_name), str(b4d_shape_name)],
                    outputs=[str(b4d_name)],
                    options={"newShape": [int(v) for v in list(b4d_shape)]},
                ),
            )

            # Re-find after insertion.
            concat_op = next((op for op in model_ir.operators if op is concat_op), None)
            post_op = next((op for op in model_ir.operators if op is post_op), None)
            if concat_op is None or post_op is None:
                continue

            concat_in_4d = [str(a4d_name), str(b4d_name)]
            if str(concat_inputs[0]) != str(a2d_name):
                concat_in_4d = [str(b4d_name), str(a4d_name)]
            concat_op.options = {
                **(dict(concat_op.options) if isinstance(concat_op.options, dict) else {}),
                "axis": 3,
            }
            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=concat_in_4d,
                graph_index=graph_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=concat_op,
                new_outputs=[str(c4d_name)],
                graph_index=graph_index,
            )
            _replace_tensor_inputs(
                model_ir,
                c2d_name,
                c4d_name,
                graph_index=graph_index,
            )

            remove_indices: List[int] = []
            pre_remove_idx = next((idx for idx, op in enumerate(model_ir.operators) if op is pre_op), None)
            if pre_remove_idx is not None:
                remove_indices.append(int(pre_remove_idx))
            post_remove_idx = next((idx for idx, op in enumerate(model_ir.operators) if op is post_op), None)
            if post_remove_idx is not None:
                remove_indices.append(int(post_remove_idx))
            for remove_idx in sorted(set(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"rewritten_flatten_concat_expanddims_to_nhwc_concat": int(rewritten)}


def run_flatten_concat_reshape_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Run the fully static 2D-to-NHWC Concat rewrite."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"RESHAPE", "CONCATENATION"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if len(required) == 0:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _single_user(
        pass_state: ModelIRPassState,
        tensor_name: str,
    ) -> OperatorIR | None:
        users = pass_state.graph_index.consumer_indices(str(tensor_name))
        if len(users) != 1:
            return None
        return pass_state.model_ir.operators[int(users[0])]

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        for pre_op in pass_state.model_ir.operators:
            if str(pre_op.op_type) != "RESHAPE" or len(pre_op.outputs) != 1:
                continue
            concat_op = _single_user(pass_state, str(pre_op.outputs[0]))
            if (
                concat_op is None
                or str(concat_op.op_type) != "CONCATENATION"
                or len(concat_op.outputs) != 1
            ):
                continue
            post_op = _single_user(pass_state, str(concat_op.outputs[0]))
            if post_op is not None and str(post_op.op_type) == "RESHAPE":
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_flatten_concat_expanddims_to_nhwc_concat(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("rewritten_flatten_concat_expanddims_to_nhwc_concat", 0)
            ),
        }

    default_details = {
        "rewritten_flatten_concat_expanddims_to_nhwc_concat": 0,
    }
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.flatten_concat_expanddims_nhwc",
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
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def run_singleton_spatial_reshape_cleanup(
    model_ir: ModelIR,
    *,
    include_spatial_flatten: bool = True,
    include_concat_post_transpose: bool = True,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Run singleton-spatial Reshape rewrites in legacy order."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        for visited, operator in enumerate(candidate_model.operators, start=1):
            if str(operator.op_type) == "TRANSPOSE":
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _single_user(
        pass_state: ModelIRPassState,
        tensor_name: str,
    ) -> OperatorIR | None:
        users = pass_state.graph_index.consumer_indices(str(tensor_name))
        if len(users) != 1:
            return None
        return pass_state.model_ir.operators[int(users[0])]

    def _has_spatial_flatten_candidate(pass_state: ModelIRPassState) -> bool:
        for transpose_op in pass_state.model_ir.operators:
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.outputs) != 1
                or _read_transpose_perm(pass_state.model_ir, transpose_op)
                != [0, 3, 1, 2]
            ):
                continue
            first = _single_user(pass_state, str(transpose_op.outputs[0]))
            if first is not None and str(first.op_type) == "RESHAPE":
                return True
        return False

    def _has_concat_candidate(pass_state: ModelIRPassState) -> bool:
        for post_op in pass_state.model_ir.operators:
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 1
                or _read_transpose_perm(pass_state.model_ir, post_op)
                != [0, 2, 3, 1]
            ):
                continue
            concat_op = pass_state.graph_index.producer(str(post_op.inputs[0]))
            if concat_op is not None and str(concat_op.op_type) == "CONCATENATION":
                return True
        return False

    def _run_spatial_flatten(
        pass_state: ModelIRPassState,
    ) -> Dict[str, int | bool]:
        stats = _optimize_singleton_spatial_nhwc_transpose_reshape_flatten(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("optimized_singleton_spatial_nhwc_transpose_reshape_flatten", 0)
            ),
        }

    def _run_concat(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_singleton_reshape_concat_post_transpose_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "rewritten_singleton_reshape_concat_post_transpose_nhwc_chains",
                    0,
                )
            ),
        }

    specs: List[PassSpec[ModelIRPassState]] = []
    if include_spatial_flatten:
        specs.append(
            PassSpec(
                pass_id="layout.singleton_spatial_flatten",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_spatial_flatten,
                precondition=_has_spatial_flatten_candidate,
                priority=10,
                transactional=True,
            )
        )
    if include_concat_post_transpose:
        specs.append(
            PassSpec(
                pass_id="layout.singleton_reshape_concat_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_concat,
                precondition=_has_concat_candidate,
                priority=20,
                transactional=True,
            )
        )
    default_details = {
        "optimized_singleton_spatial_nhwc_transpose_reshape_flatten": 0,
        "rewritten_singleton_reshape_concat_post_transpose_nhwc_chains": 0,
    }
    if len(specs) == 0:
        return default_details
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def run_singleton_channel_transpose_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Canonicalize singleton-safe rank-4 layout Transposes as Reshapes."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        for visited, operator in enumerate(candidate_model.operators, start=1):
            if str(operator.op_type) == "TRANSPOSE":
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        for transpose_op in pass_state.model_ir.operators:
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.inputs) < 2
                or len(transpose_op.outputs) != 1
            ):
                continue
            perm = _read_transpose_perm(pass_state.model_ir, transpose_op)
            if perm is None or len(perm) != 4:
                continue
            input_tensor = pass_state.model_ir.tensors.get(str(transpose_op.inputs[0]))
            output_tensor = pass_state.model_ir.tensors.get(str(transpose_op.outputs[0]))
            if (
                input_tensor is None
                or output_tensor is None
                or len(list(input_tensor.shape)) != 4
                or len(list(output_tensor.shape)) != 4
            ):
                continue
            input_shape = [int(value) for value in input_tensor.shape]
            if perm == [0, 3, 1, 2] and (
                int(input_shape[3]) == 1
                or (int(input_shape[1]) == 1 and int(input_shape[2]) == 1)
            ):
                return True
            if perm == [0, 2, 3, 1] and (
                int(input_shape[1]) == 1
                or (int(input_shape[2]) == 1 and int(input_shape[3]) == 1)
            ):
                return True
            signature = list(input_tensor.shape_signature or input_tensor.shape)
            non_singleton_input = [
                index for index, value in enumerate(signature) if int(value) != 1
            ]
            non_singleton_permuted = [
                index for index in perm if int(signature[int(index)]) != 1
            ]
            if non_singleton_input == non_singleton_permuted:
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_singleton_channel_layout_transpose_to_reshape(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("rewritten_singleton_channel_layout_transpose_to_reshape", 0)
            ),
        }

    default_details = {
        "rewritten_singleton_channel_layout_transpose_to_reshape": 0,
    }
    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.singleton_channel_transpose_as_reshape",
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
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
