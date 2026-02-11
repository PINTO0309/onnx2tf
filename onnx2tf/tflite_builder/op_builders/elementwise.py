from __future__ import annotations

from typing import Any
import math

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


def build_unary_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[input_name],
            outputs=[output_name],
        )
    )


def _get_clip_bound_value(value: Any, default_value: float) -> float:
    if value is None:
        return float(default_value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        import numpy as np
        arr = np.asarray(value)
        if arr.size == 0:
            return float(default_value)
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default_value)


def build_clip_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    clip_min = _get_clip_bound_value(node.attrs.get("min", None), float("-inf"))
    clip_max = _get_clip_bound_value(node.attrs.get("max", None), float("inf"))
    if len(node.inputs) >= 2:
        min_const = ctx.get_constant_array(node.inputs[1].name)
        clip_min = _get_clip_bound_value(min_const, clip_min)
    if len(node.inputs) >= 3:
        max_const = ctx.get_constant_array(node.inputs[2].name)
        clip_max = _get_clip_bound_value(max_const, clip_max)

    if abs(clip_min - 0.0) <= 1e-6 and abs(clip_max - 6.0) <= 1e-6:
        op_type = "RELU6"
    elif abs(clip_min - 0.0) <= 1e-6 and math.isinf(clip_max) and clip_max > 0.0:
        op_type = "RELU"
    else:
        raise NotImplementedError(
            "Clip is supported only for relu-style ranges: "
            f"min=0,max=6 or min=0,max=+inf. op={node.name} min={clip_min} max={clip_max}"
        )

    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
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
