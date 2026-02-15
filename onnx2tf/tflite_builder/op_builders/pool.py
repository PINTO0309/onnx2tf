from __future__ import annotations

import math
from typing import Any

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose, resolve_padding


def _infer_pool_output_hw(
    *,
    node: Any,
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
) -> tuple[int, int]:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    if auto_pad in ["SAME", "SAME_UPPER", "SAME_LOWER"]:
        out_h = int(math.ceil(float(input_h) / float(stride_h)))
        out_w = int(math.ceil(float(input_w) / float(stride_w)))
        return max(out_h, 1), max(out_w, 1)
    if auto_pad == "VALID":
        out_h = int(math.floor((float(input_h) - float(kernel_h)) / float(stride_h) + 1.0))
        out_w = int(math.floor((float(input_w) - float(kernel_w)) / float(stride_w) + 1.0))
        return max(out_h, 1), max(out_w, 1)

    pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(pads) < 4:
        pads = [0, 0, 0, 0]
    pad_top, pad_left, pad_bottom, pad_right = pads[0], pads[1], pads[2], pads[3]
    out_h = int(
        math.floor(
            (float(input_h + pad_top + pad_bottom - kernel_h) / float(stride_h)) + 1.0
        )
    )
    out_w = int(
        math.floor(
            (float(input_w + pad_left + pad_right - kernel_w) / float(stride_w)) + 1.0
        )
    )
    return max(out_h, 1), max(out_w, 1)


def build_pool2d_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    if len(input_shape) != 4:
        raise NotImplementedError(f"Only 2D pooling (rank=4) is supported. op={node.name}")

    kernel = list(node.attrs.get("kernel_shape", [1, 1]))
    strides = list(node.attrs.get("strides", [1, 1]))
    padding = resolve_padding(node)
    if int(node.attrs.get("ceil_mode", 0)) != 0:
        raise NotImplementedError(
            f"ceil_mode is not supported in flatbuffer_direct. op={node.name}"
        )
    if len(output_shape) != 4:
        out_h, out_w = _infer_pool_output_hw(
            node=node,
            input_h=int(input_shape[2]),
            input_w=int(input_shape[3]),
            kernel_h=int(kernel[0]),
            kernel_w=int(kernel[1]),
            stride_h=int(strides[0]),
            stride_w=int(strides[1]),
        )
        output_shape = [int(input_shape[0]), int(input_shape[1]), int(out_h), int(out_w)]
        output_tensor = ctx.model_ir.tensors[output_name]
        output_tensor.shape = list(output_shape)
        input_signature = (
            list(ctx.model_ir.tensors[input_name].shape_signature)
            if ctx.model_ir.tensors[input_name].shape_signature is not None
            else list(input_shape)
        )
        output_signature = list(output_shape)
        if len(input_signature) == 4:
            output_signature[0] = int(input_signature[0])
            output_signature[1] = int(input_signature[1])
        output_tensor.shape_signature = list(output_signature)

    nhwc_input_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
    nhwc_output_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
    output_tensor = ctx.model_ir.tensors[output_name]
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_shape)
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

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in nhwc_output_signature]
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[x_nhwc],
            outputs=[y_nhwc],
            options={
                "padding": padding,
                "strideH": int(strides[0]),
                "strideW": int(strides[1]),
                "filterHeight": int(kernel[0]),
                "filterWidth": int(kernel[1]),
                "fusedActivationFunction": "NONE",
            },
        )
    )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )
