from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _clone_quantization,
    _invert_perm,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


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
