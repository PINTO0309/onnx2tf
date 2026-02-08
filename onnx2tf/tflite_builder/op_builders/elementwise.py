from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.ir import OperatorIR


def build_binary_op(node: Any, ctx: Any, op_type: str) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)

    options = {"fusedActivationFunction": "NONE"}
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=input_names,
            outputs=[output_name],
            options=options,
        )
    )


def build_logistic_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    ctx.add_operator(
        OperatorIR(
            op_type="LOGISTIC",
            inputs=[input_name],
            outputs=[output_name],
        )
    )


def build_softmax_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    axis = int(node.attrs.get("axis", 1))
    if axis < 0:
        axis += len(input_shape)
    if axis != len(input_shape) - 1:
        raise NotImplementedError(
            f"Softmax axis != last dim is not supported in flatbuffer_direct. "
            f"op={node.name} axis={axis} shape={input_shape}"
        )

    ctx.add_operator(
        OperatorIR(
            op_type="SOFTMAX",
            inputs=[input_name],
            outputs=[output_name],
            options={"beta": float(node.attrs.get("beta", 1.0))},
        )
    )
