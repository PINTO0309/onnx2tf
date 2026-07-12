from __future__ import annotations

from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _invert_perm,
    _permute_shape,
    _prune_unused_tensors,
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
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
