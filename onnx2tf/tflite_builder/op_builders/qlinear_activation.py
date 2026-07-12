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


def build_qlinear_sigmoid_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, x_name, output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearSigmoid input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearSigmoid input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearSigmoid output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearSigmoid output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_rank = len(ctx.get_tensor_shape(x_name))
    output_rank = len(ctx.get_tensor_shape(output_name))
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(1, input_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, output_rank),
    )

    dq_out = ctx.add_intermediate_tensor(
        f"{node.name}_dq_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[x_name],
            outputs=[dq_out],
        )
    )

    sig_out = ctx.add_intermediate_tensor(
        f"{node.name}_sigmoid_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOGISTIC",
            inputs=[dq_out],
            outputs=[sig_out],
        )
    )

    ctx.add_operator(
        OperatorIR(
            op_type="QUANTIZE",
            inputs=[sig_out],
            outputs=[output_name],
        )
    )


def build_qlinear_leaky_relu_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, x_name, output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearLeakyRelu input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearLeakyRelu input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearLeakyRelu output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearLeakyRelu output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_rank = len(ctx.get_tensor_shape(x_name))
    output_rank = len(ctx.get_tensor_shape(output_name))
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(1, input_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, output_rank),
    )

    dq_out = ctx.add_intermediate_tensor(
        f"{node.name}_dq_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[x_name],
            outputs=[dq_out],
        )
    )

    alpha = float(node.attrs.get("alpha", 0.01))
    alpha_name = ctx.add_const_tensor(
        f"{node.name}_alpha",
        np.asarray([alpha], dtype=np.float32),
    )

    prelu_out = ctx.add_intermediate_tensor(
        f"{node.name}_prelu_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="PRELU",
            inputs=[dq_out, alpha_name],
            outputs=[prelu_out],
        )
    )

    ctx.add_operator(
        OperatorIR(
            op_type="QUANTIZE",
            inputs=[prelu_out],
            outputs=[output_name],
        )
    )


def build_qlinear_softmax_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, x_name, output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearSoftmax input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearSoftmax input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearSoftmax output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearSoftmax output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_rank = len(ctx.get_tensor_shape(x_name))
    output_rank = len(ctx.get_tensor_shape(output_name))
    axis = int(node.attrs.get("axis", 1))
    axis = _normalize_axis(axis, input_rank)
    if axis != input_rank - 1:
        raise NotImplementedError(
            "QLinearSoftmax supports axis=last only in flatbuffer_direct. "
            f"op={node.name} axis={axis} input_rank={input_rank}"
        )

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(1, input_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, output_rank),
    )

    dq_out = ctx.add_intermediate_tensor(
        f"{node.name}_dq_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[x_name],
            outputs=[dq_out],
        )
    )

    softmax_out = ctx.add_intermediate_tensor(
        f"{node.name}_softmax_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SOFTMAX",
            inputs=[dq_out],
            outputs=[softmax_out],
            options={"beta": float(node.attrs.get("beta", 1.0))},
        )
    )

    if not _add_scalar_onnx_requantization(
        ctx=ctx,
        input_name=softmax_out,
        output_name=output_name,
        wrap_on_overflow=True,
    ):
        ctx.add_operator(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[softmax_out],
                outputs=[output_name],
            )
        )
