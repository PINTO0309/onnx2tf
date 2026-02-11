from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose, resolve_padding


def build_pool2d_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NotImplementedError(f"Only 2D pooling (rank=4) is supported. op={node.name}")

    kernel = list(node.attrs.get("kernel_shape", [1, 1]))
    strides = list(node.attrs.get("strides", [1, 1]))
    padding = resolve_padding(node)
    if int(node.attrs.get("ceil_mode", 0)) != 0:
        raise NotImplementedError(
            f"ceil_mode is not supported in flatbuffer_direct. op={node.name}"
        )

    nhwc_input_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
    nhwc_output_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=nhwc_input_shape,
    )
    make_transpose(ctx, input_name, x_nhwc, [0, 2, 3, 1])

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
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
    make_transpose(ctx, y_nhwc, output_name, [0, 3, 1, 2])
