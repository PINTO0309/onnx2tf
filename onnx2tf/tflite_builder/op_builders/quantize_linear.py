from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.quantized_common import (
    _add_scalar_onnx_requantization,
    _normalize_axis,
    _promote_internal_uint8_tensor_to_int8,
    _propagate_shape,
    _require_const,
    _set_tensor_dtype_from_array,
    _set_tensor_quantization,
)


def build_quantize_linear_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    y_scale_name = node.inputs[1].name
    y_zero_point_name = node.inputs[2].name if len(node.inputs) >= 3 else ""
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    y_scale = _require_const(ctx, y_scale_name, "QuantizeLinear scale")
    if y_zero_point_name != "":
        y_zero_point = _require_const(ctx, y_zero_point_name, "QuantizeLinear zero_point")
    else:
        y_zero_point = np.zeros_like(y_scale, dtype=np.int32)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero_point)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    output_rank = len(ctx.get_tensor_shape(output_name))
    axis = int(node.attrs.get("axis", 1))
    qdim = _normalize_axis(axis, output_rank)
    if np.asarray(y_scale).size <= 1:
        qdim = 0
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero_point,
        quantized_dimension=qdim,
    )

    use_onnx_requantization = (
        np.asarray(y_zero_point).dtype == np.dtype(np.uint8)
        and str(ctx.get_tensor_dtype(output_name)).upper() == "INT8"
    )
    if not use_onnx_requantization or not _add_scalar_onnx_requantization(
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
    ):
        ctx.add_operator(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[input_name],
                outputs=[output_name],
            )
        )

def build_dequantize_linear_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_point_name = node.inputs[2].name if len(node.inputs) >= 3 else ""
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    x_scale = _require_const(ctx, x_scale_name, "DequantizeLinear scale")
    if x_zero_point_name != "":
        x_zero_point = _require_const(ctx, x_zero_point_name, "DequantizeLinear zero_point")
    else:
        x_zero_point = np.zeros_like(x_scale, dtype=np.int32)
    _set_tensor_dtype_from_array(ctx, input_name, x_zero_point)
    ctx.model_ir.tensors[output_name].dtype = "FLOAT32"

    input_rank = len(ctx.get_tensor_shape(input_name))
    axis = int(node.attrs.get("axis", 1))
    qdim = _normalize_axis(axis, input_rank)
    if np.asarray(x_scale).size <= 1:
        qdim = 0
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=input_name,
        scale=x_scale,
        zero_point=x_zero_point,
        quantized_dimension=qdim,
    )

    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[input_name],
            outputs=[output_name],
        )
    )
