from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose, resolve_padding


def build_conv2d_or_depthwise_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(weight_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NotImplementedError(f"Only 2D Conv (rank=4) is supported. op={node.name}")

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(
            f"Conv weights must be constant for flatbuffer_direct. op={node.name}"
        )
    weights = np.asarray(weights)
    if weights.ndim != 4:
        raise NotImplementedError(
            f"Conv weight rank must be 4. op={node.name} shape={weights.shape}"
        )

    strides = list(node.attrs.get("strides", [1, 1]))
    dilations = list(node.attrs.get("dilations", [1, 1]))
    group = int(node.attrs.get("group", 1))
    padding = resolve_padding(node)

    nchw_input = input_shape
    nchw_output = output_shape
    nhwc_input_shape = [nchw_input[0], nchw_input[2], nchw_input[3], nchw_input[1]]
    nhwc_output_shape = [nchw_output[0], nchw_output[2], nchw_output[3], nchw_output[1]]
    output_tensor = ctx.model_ir.tensors[output_name]
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
        dtype=ctx.get_tensor_dtype(input_name),
        shape=nhwc_input_shape,
    )
    x_nhwc = make_transpose(
        ctx,
        input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )

    in_channels = int(nchw_input[1])
    out_channels = int(weights.shape[0])
    is_depthwise = group == in_channels and weights.shape[1] == 1 and group > 1

    if is_depthwise:
        depth_multiplier = out_channels // in_channels
        w_dw = weights.reshape(out_channels, weights.shape[2], weights.shape[3])
        w_dw = np.transpose(w_dw, (1, 2, 0))
        w_dw = np.expand_dims(w_dw, axis=0)
        w_name = ctx.add_const_tensor(
            f"{node.name}_depthwise_filter",
            w_dw.astype(np.float32),
        )

        bias_values = None
        if len(node.inputs) >= 3:
            bias_values = ctx.get_constant_array(node.inputs[2].name)
        if bias_values is None:
            bias_values = np.zeros((out_channels,), dtype=np.float32)
        b_name = ctx.add_const_tensor(
            f"{node.name}_depthwise_bias",
            np.asarray(bias_values, dtype=np.float32).reshape(-1),
        )

        y_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc",
            dtype=ctx.get_tensor_dtype(output_name),
            shape=nhwc_output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="DEPTHWISE_CONV_2D",
                inputs=[x_nhwc, w_name, b_name],
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
            )
        )
    else:
        if group != 1:
            raise NotImplementedError(
                "Grouped Conv is not supported except depthwise. "
                f"op={node.name} group={group}"
            )
        # ONNX Conv weights are OIHW; TFLite CONV_2D expects OHWI.
        w_conv = np.transpose(weights, (0, 2, 3, 1))
        w_name = ctx.add_const_tensor(
            f"{node.name}_conv_filter",
            w_conv.astype(np.float32),
        )

        bias_values = None
        if len(node.inputs) >= 3:
            bias_values = ctx.get_constant_array(node.inputs[2].name)
        if bias_values is None:
            bias_values = np.zeros((out_channels,), dtype=np.float32)
        b_name = ctx.add_const_tensor(
            f"{node.name}_conv_bias",
            np.asarray(bias_values, dtype=np.float32).reshape(-1),
        )

        y_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc",
            dtype=ctx.get_tensor_dtype(output_name),
            shape=nhwc_output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONV_2D",
                inputs=[x_nhwc, w_name, b_name],
                outputs=[y_nhwc],
                options={
                    "padding": padding,
                    "strideH": int(strides[0]),
                    "strideW": int(strides[1]),
                    "dilationHFactor": int(dilations[0]),
                    "dilationWFactor": int(dilations[1]),
                    "fusedActivationFunction": "NONE",
                },
            )
        )
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in nhwc_output_signature]

    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )
