from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.quantized_common import (
    _add_scalar_onnx_requantization,
    _normalize_quant_params,
    _promote_internal_uint8_tensor_to_int8,
    _require_const,
    _set_tensor_dtype_from_array,
    _set_tensor_quantization,
)


def _build_qlinear_fc_like_op(
    node: Any,
    ctx: Any,
    *,
    has_bias_input: bool,
) -> None:
    a_name = node.inputs[0].name
    a_scale_name = node.inputs[1].name
    a_zero_name = node.inputs[2].name
    b_name = node.inputs[3].name
    b_scale_name = node.inputs[4].name
    b_zero_name = node.inputs[5].name
    bias_name = node.inputs[6].name if has_bias_input else ""
    y_scale_name = node.inputs[7].name if has_bias_input else node.inputs[6].name
    y_zero_name = node.inputs[8].name if has_bias_input else node.inputs[7].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(a_name)
    ctx.ensure_tensor(b_name)
    ctx.ensure_tensor(output_name)

    a_scale = _require_const(ctx, a_scale_name, "QLinearMatMul input-a scale")
    a_zero = _require_const(ctx, a_zero_name, "QLinearMatMul input-a zero_point")
    b_scale = _require_const(ctx, b_scale_name, "QLinearMatMul input-b scale")
    b_zero = _require_const(ctx, b_zero_name, "QLinearMatMul input-b zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearMatMul output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearMatMul output zero_point")
    _set_tensor_dtype_from_array(ctx, a_name, a_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, a_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if trans_a != 0:
        raise NotImplementedError(
            f"{node.op} transA=1 is not supported in flatbuffer_direct. op={node.name}"
        )
    if trans_b not in [0, 1]:
        raise NotImplementedError(
            f"{node.op} transB must be 0 or 1 in flatbuffer_direct. op={node.name} transB={trans_b}"
        )

    weights = _require_const(ctx, b_name, "QLinearMatMul weights")
    if weights.ndim != 2:
        raise NotImplementedError(
            f"QLinearMatMul weight rank must be 2. op={node.name} weight_shape={list(weights.shape)}"
        )
    if trans_b == 0:
        # ONNX B is [K, N] when transB=0. TFLite FC expects [N, K].
        fc_weights = np.asarray(weights.T, dtype=weights.dtype)
    else:
        # ONNX B is [N, K] when transB=1. It already matches TFLite FC layout.
        fc_weights = np.asarray(weights, dtype=weights.dtype)
    out_features = int(fc_weights.shape[0])

    input_shape = ctx.get_tensor_shape(a_name)
    if len(input_shape) != 2:
        output_shape = ctx.get_tensor_shape(output_name)
        if len(output_shape) == 2:
            inferred_shape = [int(output_shape[0]), int(fc_weights.shape[1])]
        else:
            inferred_shape = [1, int(fc_weights.shape[1])]
        ctx.model_ir.tensors[a_name].shape = inferred_shape
        ctx.model_ir.tensors[a_name].shape_signature = list(inferred_shape)
        input_shape = inferred_shape

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=a_name,
        scale=a_scale,
        zero_point=a_zero,
        quantized_dimension=0 if np.asarray(a_scale).size <= 1 else 1,
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else 1,
    )
    if ctx.model_ir.tensors[output_name].shape == [1]:
        ctx.model_ir.tensors[output_name].shape = [
            int(input_shape[0]),
            int(out_features),
        ]
        ctx.model_ir.tensors[output_name].shape_signature = [
            int(input_shape[0]),
            int(out_features),
        ]
    a_fc_name = a_name
    w_q_name = ctx.add_const_tensor(
        f"{node.name}_fc_weights_q",
        fc_weights,
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=w_q_name,
        scale=b_scale,
        zero_point=b_zero,
        quantized_dimension=0 if np.asarray(b_scale).size <= 1 else 0,
    )
    if has_bias_input:
        bias_values = _require_const(ctx, bias_name, f"{node.op} bias")
        bias_values = np.asarray(bias_values, dtype=np.int32).reshape(-1)
        if bias_values.size == 1 and out_features > 1:
            bias_values = np.repeat(bias_values, repeats=out_features, axis=0)
        if int(bias_values.size) != int(out_features):
            raise NotImplementedError(
                f"{node.op} bias size must match output features. "
                f"op={node.name} bias_size={int(bias_values.size)} out_features={out_features}"
            )
    else:
        bias_values = np.zeros((out_features,), dtype=np.int32)

    a_scales, _ = _normalize_quant_params(scale=a_scale, zero_point=a_zero)
    b_scales, _ = _normalize_quant_params(scale=b_scale, zero_point=b_zero)
    if len(b_scales) == 1:
        bias_scales = [float(a_scales[0] * b_scales[0])]
    else:
        bias_scales = [float(a_scales[0] * bs) for bs in b_scales]
    bias_q_name = ctx.add_const_tensor(
        f"{node.name}_fc_bias_q",
        bias_values,
    )
    ctx.model_ir.tensors[bias_q_name].quantization = QuantParamIR(
        scale=bias_scales,
        zero_point=[0 for _ in range(len(bias_scales))],
        quantized_dimension=0,
    )
    use_float_requantization_compatibility = (
        np.issubdtype(np.asarray(a_zero).dtype, np.integer)
        and np.issubdtype(np.asarray(y_zero).dtype, np.integer)
        and np.issubdtype(np.asarray(b_zero).dtype, np.integer)
    )
    w_name_for_fc = w_q_name
    bias_name_for_fc = bias_q_name
    y_fc_name = output_name
    if use_float_requantization_compatibility:
        a_fc_name = ctx.add_intermediate_tensor(
            f"{node.name}_fc_input_f32",
            dtype="FLOAT32",
            shape=list(ctx.get_tensor_shape(a_name)),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[a_name],
                outputs=[a_fc_name],
            )
        )
        w_name_for_fc = ctx.add_intermediate_tensor(
            f"{node.name}_fc_weights_f32",
            dtype="FLOAT32",
            shape=list(ctx.get_tensor_shape(w_q_name)),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[w_q_name],
                outputs=[w_name_for_fc],
            )
        )
        bias_scales_array = np.asarray(bias_scales, dtype=np.float32).reshape(-1)
        if bias_scales_array.size == 1 and bias_values.size > 1:
            bias_scales_array = np.repeat(
                bias_scales_array,
                repeats=int(bias_values.size),
            )
        bias_name_for_fc = ctx.add_const_tensor(
            f"{node.name}_fc_bias_f32",
            np.asarray(bias_values, dtype=np.float32) * bias_scales_array,
        )
        y_fc_name = ctx.add_intermediate_tensor(
            f"{node.name}_fc_output_f32",
            dtype="FLOAT32",
            shape=list(ctx.get_tensor_shape(output_name)),
        )
    ctx.add_operator(
        OperatorIR(
            op_type="FULLY_CONNECTED",
            inputs=[a_fc_name, w_name_for_fc, bias_name_for_fc],
            outputs=[y_fc_name],
            options={
                "fusedActivationFunction": "NONE",
                "weightsFormat": "DEFAULT",
                "keepNumDims": False,
                "asymmetricQuantizeInputs": False,
            },
            version=4,
        )
    )
    if use_float_requantization_compatibility:
        if not _add_scalar_onnx_requantization(
            ctx=ctx,
            input_name=y_fc_name,
            output_name=output_name,
        ):
            ctx.add_operator(
                OperatorIR(
                    op_type="QUANTIZE",
                    inputs=[y_fc_name],
                    outputs=[output_name],
                )
            )


def build_qlinear_matmul_op(node: Any, ctx: Any) -> None:
    _build_qlinear_fc_like_op(
        node=node,
        ctx=ctx,
        has_bias_input=False,
    )


def build_qgemm_op(node: Any, ctx: Any) -> None:
    _build_qlinear_fc_like_op(
        node=node,
        ctx=ctx,
        has_bias_input=True,
    )
