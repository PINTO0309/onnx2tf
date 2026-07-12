from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.quantized_common import (
    _infer_rank4_conv_output_signature,
    _require_const,
    _resolve_qlinear_conv_padding_and_explicit_pads,
    _shape_from_rank4_signature,
)
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def build_conv_integer_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    w_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name if len(node.inputs) >= 3 else ""
    w_zero_name = node.inputs[3].name if len(node.inputs) >= 4 else ""
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(w_name)
    ctx.ensure_tensor(output_name)

    weights = _require_const(ctx, w_name, "ConvInteger weights")
    if weights.ndim != 4:
        raise NotImplementedError(
            f"ConvInteger weight rank must be 4. op={node.name} weight_shape={list(weights.shape)}"
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

    if len(output_shape) != 4 and len(input_shape) == 4:
        inferred_output_shape = [
            int(input_shape[0]),
            int(weights.shape[0]),
            int(input_shape[2]),
            int(input_shape[3]),
        ]
        output_tensor.shape = [int(v) for v in list(inferred_output_shape)]
        output_shape = [int(v) for v in list(inferred_output_shape)]
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NotImplementedError(
            "ConvInteger supports only rank-4 tensors in flatbuffer_direct. "
            f"input_shape={input_shape} output_shape={output_shape} op={node.name}"
        )

    inferred_output_signature = _infer_rank4_conv_output_signature(
        input_signature_nchw=input_signature,
        output_shape_nchw=output_shape,
        existing_output_signature_nchw=existing_output_signature,
    )
    output_tensor.shape_signature = [int(v) for v in list(inferred_output_signature)]

    nchw_input = [int(v) for v in list(input_shape)]
    nchw_output = [int(v) for v in list(output_shape)]
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    padding, explicit_pads = _resolve_qlinear_conv_padding_and_explicit_pads(
        node=node,
        input_shape_nchw=nchw_input,
        output_shape_nchw=nchw_output,
    )

    nhwc_input_shape = [int(nchw_input[0]), int(nchw_input[2]), int(nchw_input[3]), int(nchw_input[1])]
    nhwc_output_shape = [int(nchw_output[0]), int(nchw_output[2]), int(nchw_output[3]), int(nchw_output[1])]
    nhwc_output_signature = [int(v) for v in list(nhwc_output_shape)]
    if len(output_signature) == 4:
        nhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[1]),
        ]

    x_f32_nchw = x_name
    x_dtype = str(ctx.get_tensor_dtype(x_name)).upper()
    if x_dtype != "FLOAT32":
        x_f32_nchw = ctx.add_intermediate_tensor(
            f"{node.name}_input_f32_nchw",
            dtype="FLOAT32",
            shape=list(nchw_input),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[x_name],
                outputs=[x_f32_nchw],
                options={"inDataType": x_dtype, "outDataType": "FLOAT32"},
            )
        )

    if x_zero_name != "":
        ctx.ensure_tensor(x_zero_name)
        x_zero_shape = [int(v) for v in list(ctx.get_tensor_shape(x_zero_name))]
        x_zero_f32 = x_zero_name
        x_zero_dtype = str(ctx.get_tensor_dtype(x_zero_name)).upper()
        if x_zero_dtype != "FLOAT32":
            x_zero_f32 = ctx.add_intermediate_tensor(
                f"{node.name}_x_zero_point_f32",
                dtype="FLOAT32",
                shape=list(x_zero_shape) if len(x_zero_shape) > 0 else [],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[x_zero_name],
                    outputs=[x_zero_f32],
                    options={"inDataType": x_zero_dtype, "outDataType": "FLOAT32"},
                )
            )
        x_zero_for_sub = x_zero_f32
        if len(x_zero_shape) == 1 and int(x_zero_shape[0]) > 1:
            x_zero_reshape_shape = [1, int(x_zero_shape[0]), 1, 1]
            x_zero_reshape_shape_name = ctx.add_const_tensor(
                f"{node.name}_x_zero_point_reshape_shape",
                np.asarray(x_zero_reshape_shape, dtype=np.int32),
            )
            x_zero_reshaped = ctx.add_intermediate_tensor(
                f"{node.name}_x_zero_point_reshaped",
                dtype="FLOAT32",
                shape=list(x_zero_reshape_shape),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[x_zero_f32, x_zero_reshape_shape_name],
                    outputs=[x_zero_reshaped],
                    options={"newShape": [int(v) for v in list(x_zero_reshape_shape)]},
                )
            )
            x_zero_for_sub = x_zero_reshaped
        x_centered_nchw = ctx.add_intermediate_tensor(
            f"{node.name}_input_centered_nchw",
            dtype="FLOAT32",
            shape=list(nchw_input),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[x_f32_nchw, x_zero_for_sub],
                outputs=[x_centered_nchw],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        x_f32_nchw = x_centered_nchw

    w_centered = np.asarray(weights, dtype=np.float32)
    if w_zero_name != "":
        w_zero = _require_const(ctx, w_zero_name, "ConvInteger weight zero_point")
        w_zero_arr = np.asarray(w_zero)
        if w_zero_arr.ndim == 0:
            w_centered = w_centered - float(w_zero_arr)
        elif w_zero_arr.ndim == 1:
            if int(w_zero_arr.size) == 1:
                w_centered = w_centered - float(w_zero_arr.reshape(-1)[0])
            elif int(w_zero_arr.size) == int(w_centered.shape[0]):
                w_centered = w_centered - w_zero_arr.astype(np.float32).reshape(-1, 1, 1, 1)
            else:
                raise NotImplementedError(
                    "ConvInteger per-output-channel weight zero_point length mismatch. "
                    f"op={node.name} w_zero_shape={list(w_zero_arr.shape)} weight_shape={list(weights.shape)}"
                )
        else:
            raise NotImplementedError(
                "ConvInteger weight zero_point must be scalar or 1D tensor in flatbuffer_direct. "
                f"op={node.name} w_zero_shape={list(w_zero_arr.shape)}"
            )

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype="FLOAT32",
        shape=list(nhwc_input_shape),
    )
    x_nhwc = make_transpose(
        ctx,
        x_f32_nchw,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )
    x_nhwc_conv = x_nhwc
    if explicit_pads is not None:
        pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in list(explicit_pads)]
        if any(int(v) != 0 for v in [pad_top, pad_left, pad_bottom, pad_right]):
            x_tensor = ctx.model_ir.tensors[x_nhwc_conv]
            padded_shape = [int(v) for v in list(x_tensor.shape)]
            padded_shape[1] = int(padded_shape[1]) + int(pad_top) + int(pad_bottom)
            padded_shape[2] = int(padded_shape[2]) + int(pad_left) + int(pad_right)
            x_nhwc_padded = ctx.add_intermediate_tensor(
                f"{node.name}_input_nhwc_padded",
                dtype="FLOAT32",
                shape=list(padded_shape),
            )
            x_sig = (
                list(x_tensor.shape_signature)
                if x_tensor.shape_signature is not None
                else list(x_tensor.shape)
            )
            if len(x_sig) == 4:
                padded_sig = [int(v) for v in list(x_sig)]
                if int(padded_sig[1]) >= 0:
                    padded_sig[1] = int(padded_sig[1]) + int(pad_top) + int(pad_bottom)
                if int(padded_sig[2]) >= 0:
                    padded_sig[2] = int(padded_sig[2]) + int(pad_left) + int(pad_right)
                ctx.model_ir.tensors[x_nhwc_padded].shape_signature = [int(v) for v in list(padded_sig)]
            pads_name = ctx.add_const_tensor(
                f"{node.name}_pads_nhwc",
                np.asarray(
                    [
                        [0, 0],
                        [int(pad_top), int(pad_bottom)],
                        [int(pad_left), int(pad_right)],
                        [0, 0],
                    ],
                    dtype=np.int32,
                ),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="PAD",
                    inputs=[x_nhwc_conv, pads_name],
                    outputs=[x_nhwc_padded],
                )
            )
            x_nhwc_conv = x_nhwc_padded

    in_channels = int(nchw_input[1])
    out_channels = int(weights.shape[0])
    weight_in_channels_per_group = int(weights.shape[1])
    is_depthwise = (
        group > 1
        and weight_in_channels_per_group == 1
        and (out_channels % group) == 0
    )
    depth_multiplier = 1

    if is_depthwise:
        depth_multiplier = out_channels // group
        w_dw = w_centered.reshape(out_channels, int(weights.shape[2]), int(weights.shape[3]))
        w_dw = np.transpose(w_dw, (1, 2, 0))
        w_dw = np.expand_dims(w_dw, axis=0)
        w_f_name = ctx.add_const_tensor(
            f"{node.name}_depthwise_filter_f32",
            np.asarray(w_dw, dtype=np.float32),
        )
    else:
        if group != 1:
            raise NotImplementedError(
                "ConvInteger grouped convolution is supported only for depthwise in flatbuffer_direct. "
                f"op={node.name} group={group}"
            )
        if int(weights.shape[1]) != int(in_channels):
            raise NotImplementedError(
                "ConvInteger weight input channels do not match input tensor channels. "
                f"op={node.name} in_channels={in_channels} weight_shape={list(weights.shape)}"
            )
        w_conv = np.transpose(w_centered, (0, 2, 3, 1))
        w_f_name = ctx.add_const_tensor(
            f"{node.name}_conv_filter_f32",
            np.asarray(w_conv, dtype=np.float32),
        )

    bias_name = ctx.add_const_tensor(
        f"{node.name}_conv_bias_f32",
        np.zeros((out_channels,), dtype=np.float32),
    )
    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype="FLOAT32",
        shape=list(nhwc_output_shape),
    )
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in list(nhwc_output_signature)]

    def _add_conv2d_op(
        *,
        input_name: str,
        filter_name: str,
        bias_name_local: str,
        output_name_local: str,
    ) -> None:
        ctx.add_operator(
            OperatorIR(
                op_type="CONV_2D",
                inputs=[input_name, filter_name, bias_name_local],
                outputs=[output_name_local],
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

    if is_depthwise:
        ctx.add_operator(
            OperatorIR(
                op_type="DEPTHWISE_CONV_2D",
                inputs=[x_nhwc_conv, w_f_name, bias_name],
                outputs=[y_nhwc],
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
        _add_conv2d_op(
            input_name=x_nhwc_conv,
            filter_name=w_f_name,
            bias_name_local=bias_name,
            output_name_local=y_nhwc,
        )

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    y_nchw_f32 = output_name if output_dtype == "FLOAT32" else ctx.add_intermediate_tensor(
        f"{node.name}_output_nchw_f32",
        dtype="FLOAT32",
        shape=list(nchw_output),
    )
    make_transpose(
        ctx,
        y_nhwc,
        y_nchw_f32,
        [0, 3, 1, 2],
    )

    if output_dtype != "FLOAT32":
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[y_nchw_f32],
                outputs=[output_name],
                options={"inDataType": "FLOAT32", "outDataType": output_dtype},
            )
        )
