from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _clone_quantization,
    _invert_perm,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _is_inverse_perm(first: list[int], second: list[int]) -> bool:
    inverse = _invert_perm(first)
    return inverse is not None and [int(value) for value in second] == inverse


def optimize_transpose_dequant_relu_quantize_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Remove inverse layout Transposes around a linear DQ-ReLU-Q chain."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        if not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators):
            return {"removed_transpose_dequant_relu_quantize_bridges": 0}
        active_index = ModelIRGraphIndex(model_ir)

    removed_bridges = 0
    while True:
        changed = False
        for pre_index in active_index.operator_indices("TRANSPOSE"):
            pre_op = model_ir.operators[int(pre_index)]
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            pre_perm = _read_transpose_perm(model_ir, pre_op)
            if pre_perm is None:
                continue

            quantized_input_bridge = str(pre_op.outputs[0])
            dequantize_users = active_index.consumer_indices(
                quantized_input_bridge
            )
            if len(dequantize_users) != 1:
                continue
            dequantize_index = int(dequantize_users[0])
            dequantize_op = model_ir.operators[dequantize_index]
            if (
                str(dequantize_op.op_type) != "DEQUANTIZE"
                or len(dequantize_op.inputs) != 1
                or len(dequantize_op.outputs) != 1
                or str(dequantize_op.inputs[0]) != quantized_input_bridge
            ):
                continue

            float_input_bridge = str(dequantize_op.outputs[0])
            activation_users = active_index.consumer_indices(float_input_bridge)
            if len(activation_users) != 1:
                continue
            activation_index = int(activation_users[0])
            activation_op = model_ir.operators[activation_index]
            if (
                str(activation_op.op_type) not in {"RELU", "RELU6"}
                or len(activation_op.inputs) != 1
                or len(activation_op.outputs) != 1
                or str(activation_op.inputs[0]) != float_input_bridge
            ):
                continue

            float_output_bridge = str(activation_op.outputs[0])
            quantize_users = active_index.consumer_indices(float_output_bridge)
            if len(quantize_users) != 1:
                continue
            quantize_index = int(quantize_users[0])
            quantize_op = model_ir.operators[quantize_index]
            if (
                str(quantize_op.op_type) != "QUANTIZE"
                or len(quantize_op.inputs) != 1
                or len(quantize_op.outputs) != 1
                or str(quantize_op.inputs[0]) != float_output_bridge
            ):
                continue

            quantized_output_bridge = str(quantize_op.outputs[0])
            post_users = active_index.consumer_indices(quantized_output_bridge)
            if len(post_users) != 1:
                continue
            post_index = int(post_users[0])
            post_op = model_ir.operators[post_index]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != quantized_output_bridge
            ):
                continue
            post_perm = _read_transpose_perm(model_ir, post_op)
            if post_perm is None or not _is_inverse_perm(pre_perm, post_perm):
                continue

            if any(
                name in model_ir.outputs
                for name in {
                    quantized_input_bridge,
                    float_input_bridge,
                    float_output_bridge,
                    quantized_output_bridge,
                }
            ):
                continue

            source_name = str(pre_op.inputs[0])
            destination_name = str(post_op.outputs[0])
            if source_name in model_ir.outputs:
                continue
            source_tensor = model_ir.tensors.get(source_name)
            quantized_input_tensor = model_ir.tensors.get(
                quantized_input_bridge
            )
            quantized_output_tensor = model_ir.tensors.get(
                quantized_output_bridge
            )
            destination_tensor = model_ir.tensors.get(destination_name)
            if not _all_per_tensor_quantized(
                [
                    source_tensor,
                    quantized_input_tensor,
                    quantized_output_tensor,
                    destination_tensor,
                ]
            ):
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=dequantize_op,
                new_inputs=[source_name],
                graph_index=active_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=quantize_op,
                new_outputs=[destination_name],
                graph_index=active_index,
            )

            source_shape = (
                list(source_tensor.shape) if source_tensor is not None else None
            )
            source_signature = (
                list(source_tensor.shape_signature)
                if source_tensor is not None
                and source_tensor.shape_signature is not None
                else source_shape
            )
            if source_shape is not None:
                for bridge_name in [float_input_bridge, float_output_bridge]:
                    bridge_tensor = model_ir.tensors.get(bridge_name)
                    if bridge_tensor is None:
                        continue
                    bridge_tensor.shape = [int(value) for value in source_shape]
                    bridge_tensor.shape_signature = (
                        [int(value) for value in source_signature]
                        if source_signature is not None
                        else [int(value) for value in source_shape]
                    )
            if destination_tensor is not None and quantized_output_tensor is not None:
                destination_tensor.dtype = str(quantized_output_tensor.dtype)
                destination_tensor.quantization = _clone_quantization(
                    quantized_output_tensor.quantization
                )

            active_index.remove_operators([int(pre_index), int(post_index)])
            removed_bridges += 1
            changed = True
            break
        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_dequant_relu_quantize_bridges": int(
            removed_bridges
        ),
    }

def optimize_transpose_dequant_hardsigmoid_quantize_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Remove inverse Transposes around an expanded HardSigmoid QDQ chain."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        if not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators):
            return {
                "removed_transpose_dequant_hardsigmoid_quantize_bridges": 0,
            }
        active_index = ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    removed_bridges = 0
    while True:
        changed = False
        for pre_index in active_index.operator_indices("TRANSPOSE"):
            pre_op = model_ir.operators[int(pre_index)]
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            pre_perm = _read_transpose_perm(model_ir, pre_op)
            if pre_perm is None:
                continue

            quantized_input_bridge = str(pre_op.outputs[0])
            dequantize_users = active_index.consumer_indices(quantized_input_bridge)
            if len(dequantize_users) != 1:
                continue
            dequantize_index = int(dequantize_users[0])
            dequantize_op = model_ir.operators[dequantize_index]
            if (
                str(dequantize_op.op_type) != "DEQUANTIZE"
                or len(dequantize_op.inputs) != 1
                or len(dequantize_op.outputs) != 1
                or str(dequantize_op.inputs[0]) != quantized_input_bridge
            ):
                continue

            float_input_bridge = str(dequantize_op.outputs[0])
            multiply_users = active_index.consumer_indices(float_input_bridge)
            if len(multiply_users) != 1:
                continue
            multiply_index = int(multiply_users[0])
            multiply_op = model_ir.operators[multiply_index]
            if (
                str(multiply_op.op_type) != "MUL"
                or len(multiply_op.inputs) != 2
                or len(multiply_op.outputs) != 1
            ):
                continue
            if str(multiply_op.inputs[0]) == float_input_bridge:
                multiply_constant_index = 1
            elif str(multiply_op.inputs[1]) == float_input_bridge:
                multiply_constant_index = 0
            else:
                continue

            multiply_output_bridge = str(multiply_op.outputs[0])
            add_users = active_index.consumer_indices(multiply_output_bridge)
            if len(add_users) != 1:
                continue
            add_index = int(add_users[0])
            add_op = model_ir.operators[add_index]
            if (
                str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
            ):
                continue
            if str(add_op.inputs[0]) == multiply_output_bridge:
                add_constant_index = 1
            elif str(add_op.inputs[1]) == multiply_output_bridge:
                add_constant_index = 0
            else:
                continue

            add_output_bridge = str(add_op.outputs[0])
            activation_users = active_index.consumer_indices(add_output_bridge)
            if len(activation_users) != 1:
                continue

            constant_inputs: List[Tuple[int, OperatorIR, int]] = [
                (multiply_index, multiply_op, multiply_constant_index),
                (add_index, add_op, add_constant_index),
            ]
            activation_intermediates = [
                multiply_output_bridge,
                add_output_bridge,
            ]
            activation_index = int(activation_users[0])
            activation_op = model_ir.operators[activation_index]
            if (
                str(activation_op.op_type) == "RELU_0_TO_1"
                and len(activation_op.inputs) == 1
                and len(activation_op.outputs) == 1
                and str(activation_op.inputs[0]) == add_output_bridge
            ):
                activation_output_bridge = str(activation_op.outputs[0])
            elif (
                str(activation_op.op_type) == "MAXIMUM"
                and len(activation_op.inputs) == 2
                and len(activation_op.outputs) == 1
            ):
                if str(activation_op.inputs[0]) == add_output_bridge:
                    maximum_constant_index = 1
                elif str(activation_op.inputs[1]) == add_output_bridge:
                    maximum_constant_index = 0
                else:
                    continue
                maximum_output = str(activation_op.outputs[0])
                minimum_users = active_index.consumer_indices(maximum_output)
                if len(minimum_users) != 1:
                    continue
                minimum_index = int(minimum_users[0])
                minimum_op = model_ir.operators[minimum_index]
                if (
                    str(minimum_op.op_type) != "MINIMUM"
                    or len(minimum_op.inputs) != 2
                    or len(minimum_op.outputs) != 1
                ):
                    continue
                if str(minimum_op.inputs[0]) == maximum_output:
                    minimum_constant_index = 1
                elif str(minimum_op.inputs[1]) == maximum_output:
                    minimum_constant_index = 0
                else:
                    continue
                activation_output_bridge = str(minimum_op.outputs[0])
                activation_intermediates.append(maximum_output)
                constant_inputs.extend(
                    [
                        (
                            activation_index,
                            activation_op,
                            maximum_constant_index,
                        ),
                        (minimum_index, minimum_op, minimum_constant_index),
                    ]
                )
            else:
                continue

            quantize_users = active_index.consumer_indices(activation_output_bridge)
            if len(quantize_users) != 1:
                continue
            quantize_index = int(quantize_users[0])
            quantize_op = model_ir.operators[quantize_index]
            if (
                str(quantize_op.op_type) != "QUANTIZE"
                or len(quantize_op.inputs) != 1
                or len(quantize_op.outputs) != 1
                or str(quantize_op.inputs[0]) != activation_output_bridge
            ):
                continue

            quantized_output_bridge = str(quantize_op.outputs[0])
            post_users = active_index.consumer_indices(quantized_output_bridge)
            if len(post_users) != 1:
                continue
            post_index = int(post_users[0])
            post_op = model_ir.operators[post_index]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != quantized_output_bridge
            ):
                continue
            post_perm = _read_transpose_perm(model_ir, post_op)
            if post_perm is None or not _is_inverse_perm(pre_perm, post_perm):
                continue

            observable_intermediates = {
                quantized_input_bridge,
                float_input_bridge,
                activation_output_bridge,
                quantized_output_bridge,
                *activation_intermediates,
            }
            if any(name in model_ir.outputs for name in observable_intermediates):
                continue

            source_name = str(pre_op.inputs[0])
            destination_name = str(post_op.outputs[0])
            if source_name in model_ir.outputs:
                continue
            source_tensor = model_ir.tensors.get(source_name)
            quantized_input_tensor = model_ir.tensors.get(quantized_input_bridge)
            quantized_output_tensor = model_ir.tensors.get(quantized_output_bridge)
            destination_tensor = model_ir.tensors.get(destination_name)
            if not _all_per_tensor_quantized(
                [
                    source_tensor,
                    quantized_input_tensor,
                    quantized_output_tensor,
                    destination_tensor,
                ]
            ):
                continue

            # Validate every side input before changing any tensor or edge.
            # This makes a rejected match a true no-op instead of leaving an
            # earlier rank-matched constant partially transposed.
            constant_remaps: List[
                Tuple[int, OperatorIR, int, TensorIR, np.ndarray, bool]
            ] = []
            constants_valid = True
            for op_index, op, input_index in constant_inputs:
                constant_name = str(op.inputs[input_index])
                constant_tensor = model_ir.tensors.get(constant_name)
                if constant_tensor is None or constant_tensor.data is None:
                    constants_valid = False
                    break
                constant_data = np.asarray(constant_tensor.data)
                if constant_data.ndim != len(post_perm):
                    continue
                remapped_data = np.transpose(constant_data, axes=post_perm)
                constant_users = active_index.consumer_indices(constant_name)
                can_update_in_place = len(constant_users) == 1 and int(
                    constant_users[0]
                ) == int(op_index)
                constant_remaps.append(
                    (
                        op_index,
                        op,
                        input_index,
                        constant_tensor,
                        np.asarray(remapped_data),
                        can_update_in_place,
                    )
                )
            if not constants_valid:
                continue

            for (
                _op_index,
                op,
                input_index,
                constant_tensor,
                remapped_data,
                can_update_in_place,
            ) in constant_remaps:
                if can_update_in_place:
                    constant_tensor.data = np.asarray(remapped_data)
                    constant_tensor.shape = [
                        int(value) for value in remapped_data.shape
                    ]
                    constant_tensor.shape_signature = [
                        int(value) for value in remapped_data.shape
                    ]
                    continue
                constant_name = str(op.inputs[input_index])
                new_name = _unique_tensor_name(f"{constant_name}_nhwc")
                model_ir.tensors[new_name] = TensorIR(
                    name=new_name,
                    dtype=str(constant_tensor.dtype),
                    shape=[int(value) for value in remapped_data.shape],
                    shape_signature=[int(value) for value in remapped_data.shape],
                    data=np.asarray(remapped_data),
                    is_variable=False,
                    quantization=_clone_quantization(constant_tensor.quantization),
                )
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=op,
                    input_index=int(input_index),
                    new_input_name=new_name,
                    graph_index=active_index,
                )

            _set_operator_inputs(
                model_ir=model_ir,
                op=dequantize_op,
                new_inputs=[source_name],
                graph_index=active_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=quantize_op,
                new_outputs=[destination_name],
                graph_index=active_index,
            )

            source_shape = (
                list(source_tensor.shape) if source_tensor is not None else None
            )
            source_signature = (
                list(source_tensor.shape_signature)
                if source_tensor is not None
                and source_tensor.shape_signature is not None
                else source_shape
            )
            if source_shape is not None:
                for bridge_name in [
                    float_input_bridge,
                    *activation_intermediates,
                    activation_output_bridge,
                ]:
                    bridge_tensor = model_ir.tensors.get(bridge_name)
                    if bridge_tensor is None:
                        continue
                    bridge_tensor.shape = [int(value) for value in source_shape]
                    bridge_tensor.shape_signature = (
                        [int(value) for value in source_signature]
                        if source_signature is not None
                        else [int(value) for value in source_shape]
                    )
            if destination_tensor is not None and quantized_output_tensor is not None:
                destination_tensor.dtype = str(quantized_output_tensor.dtype)
                destination_tensor.quantization = _clone_quantization(
                    quantized_output_tensor.quantization
                )

            active_index.remove_operators([int(pre_index), int(post_index)])
            removed_bridges += 1
            changed = True
            break
        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_dequant_hardsigmoid_quantize_bridges": int(removed_bridges),
    }
