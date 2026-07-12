from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.quantized import (
    _add_scalar_onnx_requantization,
    _normalize_axis,
    _promote_internal_uint8_tensor_to_int8,
    _propagate_shape,
    _require_const,
    _set_tensor_dtype_from_array,
    _set_tensor_quantization,
)


def _build_qlinear_binary_op(node: Any, ctx: Any, op_type: str) -> None:
    a_name = node.inputs[0].name
    a_scale_name = node.inputs[1].name
    a_zero_name = node.inputs[2].name
    b_name = node.inputs[3].name
    b_scale_name = node.inputs[4].name
    b_zero_name = node.inputs[5].name
    c_scale_name = node.inputs[6].name
    c_zero_name = node.inputs[7].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(a_name)
    ctx.ensure_tensor(b_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, a_name, output_name)

    a_scale = _require_const(ctx, a_scale_name, f"{node.op} input-a scale")
    a_zero = _require_const(ctx, a_zero_name, f"{node.op} input-a zero_point")
    b_scale = _require_const(ctx, b_scale_name, f"{node.op} input-b scale")
    b_zero = _require_const(ctx, b_zero_name, f"{node.op} input-b zero_point")
    c_scale = _require_const(ctx, c_scale_name, f"{node.op} output scale")
    c_zero = _require_const(ctx, c_zero_name, f"{node.op} output zero_point")
    _set_tensor_dtype_from_array(ctx, a_name, a_zero)
    _set_tensor_dtype_from_array(ctx, b_name, b_zero)
    _set_tensor_dtype_from_array(ctx, output_name, c_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, a_name)
    _promote_internal_uint8_tensor_to_int8(ctx, b_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    a_rank = len(ctx.get_tensor_shape(a_name))
    b_rank = len(ctx.get_tensor_shape(b_name))
    out_rank = len(ctx.get_tensor_shape(output_name))

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=a_name,
        scale=a_scale,
        zero_point=a_zero,
        quantized_dimension=0 if np.asarray(a_scale).size <= 1 else _normalize_axis(1, a_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=b_name,
        scale=b_scale,
        zero_point=b_zero,
        quantized_dimension=0 if np.asarray(b_scale).size <= 1 else _normalize_axis(1, b_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=c_scale,
        zero_point=c_zero,
        quantized_dimension=0 if np.asarray(c_scale).size <= 1 else _normalize_axis(1, out_rank),
    )

    op_type_upper = str(op_type).upper()
    use_float_binary_path = (
        op_type_upper == "MUL"
        or (a_name not in ctx.constants and b_name not in ctx.constants)
    )
    if not use_float_binary_path:
        ctx.add_operator(
            OperatorIR(
                op_type=op_type,
                inputs=[a_name, b_name],
                outputs=[output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return

    # TFLite's quantized binary kernels use fixed-point requantization whose
    # tie handling can differ by one quantum from ONNX Runtime.  Preserve the
    # ONNX dequantize -> arithmetic -> round -> saturate contract explicitly;
    # an early one-quantum residual error can otherwise be amplified by later
    # quantized convolutions.
    a_f_name = ctx.add_intermediate_tensor(
        f"{output_name}_{str(op_type).lower()}_a_f32",
        dtype="FLOAT32",
        shape=list(ctx.get_tensor_shape(a_name)),
    )
    b_f_name = ctx.add_intermediate_tensor(
        f"{output_name}_{str(op_type).lower()}_b_f32",
        dtype="FLOAT32",
        shape=list(ctx.get_tensor_shape(b_name)),
    )
    out_f_name = ctx.add_intermediate_tensor(
        f"{output_name}_{str(op_type).lower()}_f32",
        dtype="FLOAT32",
        shape=list(ctx.get_tensor_shape(output_name)),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[a_name],
            outputs=[a_f_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[b_name],
            outputs=[b_f_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[a_f_name, b_f_name],
            outputs=[out_f_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    use_explicit_onnx_requantization = (
        op_type_upper == "ADD"
        and _add_scalar_onnx_requantization(
            ctx=ctx,
            input_name=out_f_name,
            output_name=output_name,
        )
    )
    if not use_explicit_onnx_requantization:
        ctx.add_operator(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[out_f_name],
                outputs=[output_name],
            )
        )


def build_qlinear_add_op(node: Any, ctx: Any) -> None:
    _build_qlinear_binary_op(node, ctx, "ADD")


def build_qlinear_mul_op(node: Any, ctx: Any) -> None:
    _build_qlinear_binary_op(node, ctx, "MUL")
