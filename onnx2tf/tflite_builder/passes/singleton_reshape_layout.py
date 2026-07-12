from __future__ import annotations

from typing import Dict, List, Optional

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _clone_quantization,
    _invert_perm,
    _permute_shape,
    _prune_unused_tensors,
    _rename_tensor_globally,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
def _optimize_singleton_layout_reshape_unary_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
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
        consumers = _build_tensor_consumer_map(model_ir)
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
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=unary_op,
                new_outputs=[post_output_name],
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
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"rewritten_singleton_layout_reshape_unary_passthrough_chains": int(rewritten)}

def _optimize_consecutive_inverse_singleton_layout_reshapes(model_ir: ModelIR) -> Dict[str, int]:
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
        consumers = _build_tensor_consumer_map(model_ir)
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
                )
            else:
                _replace_tensor_inputs(
                    model_ir=model_ir,
                    src_name=str(dst_name),
                    dst_name=str(src_name),
                )

            for remove_idx in sorted([int(first_idx), int(second_idx)], reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"rewritten_consecutive_inverse_singleton_layout_reshapes": int(rewritten)}

