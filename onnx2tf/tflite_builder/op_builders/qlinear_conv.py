from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.conv import (
    _infer_conv2d_output_shape_nchw,
)
from onnx2tf.tflite_builder.op_builders.quantized_common import (
    _add_scalar_onnx_requantization,
    _feeds_max_pool_through_quantized_passthrough,
    _infer_rank4_conv_output_signature,
    _normalize_axis,
    _normalize_quant_params,
    _promote_internal_uint8_tensor_to_int8,
    _require_const,
    _resolve_qlinear_conv_padding_and_explicit_pads,
    _set_tensor_dtype_from_array,
    _set_tensor_quantization,
    _shape_from_rank4_signature,
)
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def build_qlinear_conv_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    w_name = node.inputs[3].name
    w_scale_name = node.inputs[4].name
    w_zero_name = node.inputs[5].name
    y_scale_name = node.inputs[6].name
    y_zero_name = node.inputs[7].name
    bias_name = node.inputs[8].name if len(node.inputs) >= 9 else ""
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(w_name)
    ctx.ensure_tensor(output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearConv input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearConv input zero_point")
    w_scale = _require_const(ctx, w_scale_name, "QLinearConv weight scale")
    w_zero = _require_const(ctx, w_zero_name, "QLinearConv weight zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearConv output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearConv output zero_point")
    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(1, len(ctx.get_tensor_shape(x_name))),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, len(ctx.get_tensor_shape(output_name))),
    )

    weights = _require_const(ctx, w_name, "QLinearConv weights")
    if weights.ndim != 4:
        raise NotImplementedError(
            f"QLinearConv weight rank must be 4. op={node.name} weight_shape={list(weights.shape)}"
        )
    input_shape = [int(v) for v in list(ctx.get_tensor_shape(x_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    input_tensor = ctx.model_ir.tensors[x_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_shape)
    )
    existing_output_signature = (
        list(output_signature)
        if len(list(output_signature)) == 4
        else None
    )
    rank4_input_from_signature = _shape_from_rank4_signature(input_signature)
    if len(input_shape) != 4 and rank4_input_from_signature is not None:
        input_shape = [int(v) for v in list(rank4_input_from_signature)]
        input_tensor.shape = [int(v) for v in list(input_shape)]
    rank4_output_from_signature = _shape_from_rank4_signature(output_signature)
    if len(output_shape) != 4 and rank4_output_from_signature is not None:
        output_shape = [int(v) for v in list(rank4_output_from_signature)]
        output_tensor.shape = [int(v) for v in list(output_shape)]

    group = int(node.attrs.get("group", 1))
    inferred_input_channels = int(weights.shape[1]) * int(group if group > 0 else 1)
    inferred_output_channels = int(weights.shape[0])
    if (
        len(input_shape) == 4
        and len(input_signature) == 4
        and int(input_signature[1]) < 0
        and int(input_shape[1]) <= 1
    ):
        input_shape[1] = int(inferred_input_channels)
        input_tensor.shape = [int(v) for v in list(input_shape)]
    if (
        len(output_shape) == 4
        and len(output_signature) == 4
        and int(output_signature[1]) < 0
        and int(output_shape[1]) <= 1
    ):
        output_shape[1] = int(inferred_output_channels)
        output_tensor.shape = [int(v) for v in list(output_shape)]

    inferred_output_shape = (
        _infer_conv2d_output_shape_nchw(
            node=node,
            input_shape_nchw=input_shape,
            weights=weights,
        )
        if len(input_shape) == 4
        and np.asarray(weights).ndim == 4
        and all(int(v) > 0 for v in input_shape)
        and _feeds_max_pool_through_quantized_passthrough(ctx, output_name)
        else None
    )
    output_shape_disagrees = bool(
        inferred_output_shape is not None
        and [int(v) for v in output_shape]
        != [int(v) for v in inferred_output_shape]
    )
    if inferred_output_shape is not None and (
        len(output_shape) != 4 or output_shape_disagrees
    ):
        output_shape = [int(v) for v in inferred_output_shape]
        output_tensor.shape = [int(v) for v in output_shape]
        # Static convolution geometry is authoritative over stale or
        # placeholder value-info. Dynamic axes are reintroduced from the
        # input signature below.
        existing_output_signature = None
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NotImplementedError(
            "QLinearConv supports only rank-4 tensors in flatbuffer_direct. "
            f"input_shape={input_shape} output_shape={output_shape} op={node.name}"
        )
    inferred_output_signature = _infer_rank4_conv_output_signature(
        input_signature_nchw=input_signature,
        output_shape_nchw=output_shape,
        existing_output_signature_nchw=existing_output_signature,
    )
    output_tensor.shape_signature = [int(v) for v in inferred_output_signature]

    nchw_input = input_shape
    nchw_output = output_shape
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    padding, explicit_pads = _resolve_qlinear_conv_padding_and_explicit_pads(
        node=node,
        input_shape_nchw=nchw_input,
        output_shape_nchw=nchw_output,
    )

    nhwc_input_shape = [nchw_input[0], nchw_input[2], nchw_input[3], nchw_input[1]]
    nhwc_output_shape = [nchw_output[0], nchw_output[2], nchw_output[3], nchw_output[1]]
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(nchw_output)
    )
    nhwc_output_signature = list(nhwc_output_shape)
    if len(output_signature) == 4:
        nhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[1]),
        ]

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(x_name),
        shape=nhwc_input_shape,
    )
    ctx.model_ir.tensors[x_nhwc].quantization = ctx.model_ir.tensors[x_name].quantization
    x_nhwc = make_transpose(
        ctx,
        x_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )
    x_nhwc_conv = x_nhwc
    if explicit_pads is not None:
        pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in explicit_pads]
        if any(int(v) != 0 for v in [pad_top, pad_left, pad_bottom, pad_right]):
            x_tensor = ctx.model_ir.tensors[x_nhwc_conv]
            padded_shape = list(x_tensor.shape)
            padded_shape[1] = int(padded_shape[1]) + int(pad_top) + int(pad_bottom)
            padded_shape[2] = int(padded_shape[2]) + int(pad_left) + int(pad_right)
            x_nhwc_padded = ctx.add_intermediate_tensor(
                f"{node.name}_input_nhwc_padded",
                dtype=ctx.get_tensor_dtype(x_nhwc_conv),
                shape=padded_shape,
            )
            x_nhwc_padded_tensor = ctx.model_ir.tensors[x_nhwc_padded]
            x_nhwc_padded_tensor.quantization = x_tensor.quantization

            x_sig = (
                list(x_tensor.shape_signature)
                if x_tensor.shape_signature is not None
                else list(x_tensor.shape)
            )
            if len(x_sig) == 4:
                padded_sig = list(x_sig)
                if int(padded_sig[1]) >= 0:
                    padded_sig[1] = int(padded_sig[1]) + int(pad_top) + int(pad_bottom)
                if int(padded_sig[2]) >= 0:
                    padded_sig[2] = int(padded_sig[2]) + int(pad_left) + int(pad_right)
                x_nhwc_padded_tensor.shape_signature = [int(v) for v in padded_sig]

            pads_name = ctx.add_const_tensor(
                f"{node.name}_pads_nhwc",
                np.asarray(
                    [
                        [0, 0],
                        [pad_top, pad_bottom],
                        [pad_left, pad_right],
                        [0, 0],
                    ],
                    dtype=np.int32,
                ),
            )
            # NOTE:
            # For quantized activation tensors, TFLite PAD uses tensor quantization
            # parameters and pads with the quantized representation of real-value 0.
            # QLinearConv explicit padding requires padding by input zero_point,
            # so PAD is the semantically-correct choice here.
            ctx.add_operator(
                OperatorIR(
                    op_type="PAD",
                    inputs=[x_nhwc_conv, pads_name],
                    outputs=[x_nhwc_padded],
                )
            )
            x_nhwc_conv = x_nhwc_padded

    out_channels = int(weights.shape[0])
    weight_in_channels_per_group = int(weights.shape[1])
    # Keep depthwise detection aligned with op_registry validator:
    # rely on group/weight shape rather than potentially stale input metadata.
    is_depthwise = (
        group > 1
        and weight_in_channels_per_group == 1
        and (out_channels % group) == 0
    )
    depth_multiplier = 1

    if is_depthwise:
        # For ONNX depthwise-style QLinearConv, logical input channels are `group`.
        depth_multiplier = out_channels // group
        w_dw = weights.reshape(out_channels, int(weights.shape[2]), int(weights.shape[3]))
        w_dw = np.transpose(w_dw, (1, 2, 0))
        w_dw = np.expand_dims(w_dw, axis=0)
        w_q_name = ctx.add_const_tensor(
            f"{node.name}_depthwise_filter_q",
            np.asarray(w_dw, dtype=weights.dtype),
        )
        _set_tensor_quantization(
            ctx=ctx,
            tensor_name=w_q_name,
            scale=w_scale,
            zero_point=w_zero,
            # DEPTHWISE_CONV_2D weights are [1, H, W, I*depth_multiplier].
            # Per-channel quantization axis is channel-last (axis=3).
            quantized_dimension=(len(ctx.model_ir.tensors[w_q_name].shape) - 1) if np.asarray(w_scale).size > 1 else 0,
        )
    else:
        if group != 1:
            allow_custom = bool(getattr(ctx, "allow_custom_ops", False))
            if allow_custom:
                allowlist_raw = getattr(ctx, "custom_op_allowlist", None)
                allow_qlinearconv_custom = True
                if allowlist_raw is not None:
                    if isinstance(allowlist_raw, (str, bytes)):
                        allow_items = [str(allowlist_raw)]
                    else:
                        try:
                            allow_items = [str(v) for v in list(allowlist_raw)]
                        except Exception:
                            allow_items = [str(allowlist_raw)]
                    allow_set = {
                        str(v).strip().upper() for v in allow_items if str(v).strip() != ""
                    }
                    if len(allow_set) > 0 and "QLINEARCONV" not in allow_set:
                        allow_qlinearconv_custom = False
                if allow_qlinearconv_custom:
                    # Keep TF/common path untouched; escape grouped QLinearConv to
                    # flatbuffer_direct CUSTOM flow when requested.
                    from onnx2tf.tflite_builder.op_builders.custom import build_custom_passthrough_op
                    build_custom_passthrough_op(node, ctx)
                    return
            raise NotImplementedError(
                "QLinearConv grouped convolution is supported only for depthwise. "
                f"op={node.name} group={group}"
            )
        # ONNX QLinearConv weights are OIHW; TFLite CONV_2D expects OHWI.
        w_conv = np.transpose(weights, (0, 2, 3, 1))
        w_q_name = ctx.add_const_tensor(
            f"{node.name}_conv_filter_q",
            np.asarray(w_conv, dtype=weights.dtype),
        )
        _set_tensor_quantization(
            ctx=ctx,
            tensor_name=w_q_name,
            scale=w_scale,
            zero_point=w_zero,
            # CONV_2D weights are OHWI. Per-channel quantization axis is O (axis=0).
            quantized_dimension=0 if np.asarray(w_scale).size > 1 else 0,
        )

    if bias_name != "":
        bias_values = _require_const(ctx, bias_name, "QLinearConv bias")
        bias_values = np.asarray(bias_values, dtype=np.int32).reshape(-1)
    else:
        bias_values = np.zeros((out_channels,), dtype=np.int32)
    x_scales, _ = _normalize_quant_params(scale=x_scale, zero_point=x_zero)
    w_scales, _ = _normalize_quant_params(scale=w_scale, zero_point=w_zero)
    if len(w_scales) == 1:
        bias_scales = [float(x_scales[0] * w_scales[0])]
    else:
        bias_scales = [float(x_scales[0] * ws) for ws in w_scales]
    # TFLite's quantized convolution kernels use a fixed-point requantizer whose
    # tie handling can differ by one output quantum from ONNX Runtime's
    # QLinearConv implementation. A single early difference can be amplified by
    # later quantized layers. Keep quantized graph boundaries, but evaluate the
    # exact integer accumulator through float builtins before applying explicit
    # ONNX round-then-saturate requantization.
    use_float_requantization_compatibility = (
        np.issubdtype(np.asarray(x_zero).dtype, np.integer)
        and np.issubdtype(np.asarray(y_zero).dtype, np.integer)
        and np.issubdtype(np.asarray(w_zero).dtype, np.integer)
    )

    if use_float_requantization_compatibility:
        bias_scales_array = np.asarray(bias_scales, dtype=np.float32).reshape(-1)
        if bias_scales_array.size == 1 and bias_values.size > 1:
            bias_scales_array = np.repeat(
                bias_scales_array,
                repeats=int(bias_values.size),
            )
        bias_name_for_conv = ctx.add_const_tensor(
            f"{node.name}_conv_bias_f32",
            np.asarray(bias_values, dtype=np.float32),
        )
    else:
        bias_name_for_conv = ctx.add_const_tensor(
            f"{node.name}_conv_bias_q",
            bias_values,
        )
        ctx.model_ir.tensors[bias_name_for_conv].quantization = QuantParamIR(
            scale=bias_scales,
            zero_point=[0 for _ in range(len(bias_scales))],
            quantized_dimension=0,
        )

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    ctx.model_ir.tensors[y_nhwc].quantization = ctx.model_ir.tensors[output_name].quantization
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in nhwc_output_signature]
    y_nhwc_conv = y_nhwc
    x_name_for_conv = x_nhwc_conv
    w_name_for_conv = w_q_name
    y_nhwc_accumulator = y_nhwc
    if use_float_requantization_compatibility:
        x_cast_name = ctx.add_intermediate_tensor(
            f"{node.name}_input_nhwc_q_f32",
            dtype="FLOAT32",
            shape=list(ctx.model_ir.tensors[x_nhwc_conv].shape),
        )
        ctx.model_ir.tensors[x_cast_name].shape_signature = (
            list(ctx.model_ir.tensors[x_nhwc_conv].shape_signature)
            if ctx.model_ir.tensors[x_nhwc_conv].shape_signature is not None
            else list(ctx.model_ir.tensors[x_nhwc_conv].shape)
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[x_nhwc_conv],
                outputs=[x_cast_name],
                options={
                    "inDataType": str(ctx.get_tensor_dtype(x_nhwc_conv)).upper(),
                    "outDataType": "FLOAT32",
                },
            )
        )
        x_quantization = ctx.model_ir.tensors[x_nhwc_conv].quantization
        x_zero_points = (
            list(x_quantization.zero_point)
            if isinstance(x_quantization, QuantParamIR)
            else [0]
        )
        if len(x_zero_points) != 1:
            raise NotImplementedError(
                "QLinearConv mixed UINT8/INT8 compatibility requires a scalar "
                f"input zero point. op={node.name} zero_points={x_zero_points}"
            )
        x_name_for_conv = x_cast_name
        if int(x_zero_points[0]) != 0:
            x_zero_name = ctx.add_const_tensor(
                f"{node.name}_input_zero_f32",
                np.asarray(float(x_zero_points[0]), dtype=np.float32),
            )
            x_name_for_conv = ctx.add_intermediate_tensor(
                f"{node.name}_input_centered_f32",
                dtype="FLOAT32",
                shape=list(ctx.model_ir.tensors[x_nhwc_conv].shape),
            )
            ctx.model_ir.tensors[x_name_for_conv].shape_signature = list(
                ctx.model_ir.tensors[x_cast_name].shape_signature
                or ctx.model_ir.tensors[x_cast_name].shape
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SUB",
                    inputs=[x_cast_name, x_zero_name],
                    outputs=[x_name_for_conv],
                    options={"fusedActivationFunction": "NONE"},
                )
            )

        weight_tensor = ctx.model_ir.tensors[w_q_name]
        weight_quantization = weight_tensor.quantization
        weight_zero_points = (
            list(weight_quantization.zero_point)
            if isinstance(weight_quantization, QuantParamIR)
            else [0]
        )
        weight_shape = list(weight_tensor.shape)
        weight_cast_name = ctx.add_intermediate_tensor(
            f"{node.name}_filter_q_f32",
            dtype="FLOAT32",
            shape=weight_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[w_q_name],
                outputs=[weight_cast_name],
                options={
                    "inDataType": str(weight_tensor.dtype).upper(),
                    "outDataType": "FLOAT32",
                    "preserveRuntimeCastForQuantizedAccumulator": True,
                },
            )
        )
        w_name_for_conv = weight_cast_name
        if any(int(v) != 0 for v in weight_zero_points):
            if len(weight_zero_points) == 1:
                weight_zero_values = np.asarray(
                    float(weight_zero_points[0]),
                    dtype=np.float32,
                )
            else:
                quantized_dimension = int(weight_quantization.quantized_dimension)
                zero_shape = [1 for _ in range(len(weight_shape))]
                zero_shape[quantized_dimension] = len(weight_zero_points)
                weight_zero_values = np.asarray(
                    weight_zero_points,
                    dtype=np.float32,
                ).reshape(zero_shape)
            weight_zero_name = ctx.add_const_tensor(
                f"{node.name}_filter_zero_f32",
                weight_zero_values,
            )
            w_name_for_conv = ctx.add_intermediate_tensor(
                f"{node.name}_filter_centered_f32",
                dtype="FLOAT32",
                shape=weight_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SUB",
                    inputs=[weight_cast_name, weight_zero_name],
                    outputs=[w_name_for_conv],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
        y_nhwc_accumulator = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc_accumulator_f32",
            dtype="FLOAT32",
            shape=list(nhwc_output_shape),
        )
        ctx.model_ir.tensors[y_nhwc_accumulator].shape_signature = [
            int(v) for v in nhwc_output_signature
        ]
        y_nhwc_conv = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc_f32",
            dtype="FLOAT32",
            shape=list(nhwc_output_shape),
        )
        ctx.model_ir.tensors[y_nhwc_conv].shape_signature = [
            int(v) for v in nhwc_output_signature
        ]

    if is_depthwise:
        ctx.add_operator(
            OperatorIR(
                op_type="DEPTHWISE_CONV_2D",
                inputs=[x_name_for_conv, w_name_for_conv, bias_name_for_conv],
                outputs=[y_nhwc_accumulator],
                options={
                    "padding": padding,
                    "strideH": int(strides[0]),
                    "strideW": int(strides[1]),
                    "dilationHFactor": int(dilations[0]),
                    "dilationWFactor": int(dilations[1]),
                    "depthMultiplier": int(depth_multiplier),
                    "fusedActivationFunction": "NONE",
                },
                version=3,
            )
        )
    else:
        ctx.add_operator(
            OperatorIR(
                op_type="CONV_2D",
                inputs=[x_name_for_conv, w_name_for_conv, bias_name_for_conv],
                outputs=[y_nhwc_accumulator],
                options={
                    "padding": padding,
                    "strideH": int(strides[0]),
                    "strideW": int(strides[1]),
                    "dilationHFactor": int(dilations[0]),
                    "dilationWFactor": int(dilations[1]),
                    "fusedActivationFunction": "NONE",
                },
                version=3,
            )
        )
    if use_float_requantization_compatibility:
        output_quantization = ctx.model_ir.tensors[y_nhwc].quantization
        output_scale = float(output_quantization.scale[0])
        requant_multiplier_name = ctx.add_const_tensor(
            f"{node.name}_accumulator_requant_multiplier",
            np.asarray(
                bias_scales_array / np.float32(output_scale),
                dtype=np.float32,
            ).reshape(
                () if int(bias_scales_array.size) == 1 else (-1,)
            ),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[y_nhwc_accumulator, requant_multiplier_name],
                outputs=[y_nhwc_conv],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        if not _add_scalar_onnx_requantization(
            ctx=ctx,
            input_name=y_nhwc_conv,
            output_name=y_nhwc,
            input_is_scaled_quant_units=True,
        ):
            ctx.add_operator(
                OperatorIR(
                    op_type="QUANTIZE",
                    inputs=[y_nhwc_conv],
                    outputs=[y_nhwc],
                )
            )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )
