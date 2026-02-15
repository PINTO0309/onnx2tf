from __future__ import annotations

from typing import Any
import math
import copy

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR


def _propagate_shape(ctx: Any, src_tensor_name: str, dst_tensor_name: str) -> None:
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src = ctx.model_ir.tensors[src_tensor_name]
    dst = ctx.model_ir.tensors[dst_tensor_name]
    if dst.shape == [1] and src.shape != [1]:
        dst.shape = list(src.shape)
        dst.shape_signature = (
            list(src.shape_signature)
            if src.shape_signature is not None
            else list(src.shape)
        )


def build_binary_op(node: Any, ctx: Any, op_type: str) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)
    if len(input_names) > 0:
        _propagate_shape(ctx, input_names[0], output_name)

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
    _propagate_shape(ctx, input_name, output_name)
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
    _propagate_shape(ctx, input_name, output_name)
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
    _propagate_shape(ctx, input_name, output_name)

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


def _clone_quantization(quantization: Any) -> Any:
    if quantization is None:
        return None
    if isinstance(quantization, QuantParamIR):
        return QuantParamIR(
            scale=list(quantization.scale),
            zero_point=list(quantization.zero_point),
            quantized_dimension=int(quantization.quantized_dimension),
            min=list(quantization.min) if quantization.min is not None else None,
            max=list(quantization.max) if quantization.max is not None else None,
        )
    return copy.deepcopy(quantization)


def _reshape_prelu_slope_for_input(
    slope: np.ndarray,
    input_shape: list[int],
) -> np.ndarray:
    if slope.ndim == 0:
        return slope.reshape([1])
    if len(input_shape) == 4 and len(input_shape) >= 2:
        channels = int(input_shape[1])
        if slope.ndim == 1 and slope.size == channels:
            return slope.reshape([1, channels, 1, 1])
        if slope.ndim == 3 and slope.shape[0] == channels and slope.shape[1] == 1 and slope.shape[2] == 1:
            return slope.reshape([1, channels, 1, 1])
    if len(input_shape) == 2 and len(input_shape) >= 2:
        channels = int(input_shape[1])
        if slope.ndim == 1 and slope.size == channels:
            return slope.reshape([1, channels])
    return slope


def _quantize_prelu_slope(
    slope: np.ndarray,
    target_dtype: str,
) -> tuple[np.ndarray, QuantParamIR]:
    if target_dtype == "INT8":
        max_abs = float(np.max(np.abs(slope))) if slope.size > 0 else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        q = np.clip(np.round(slope / scale), -128, 127).astype(np.int8)
        return q, QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            quantized_dimension=0,
        )
    if target_dtype == "UINT8":
        mn = float(np.min(slope)) if slope.size > 0 else 0.0
        mx = float(np.max(slope)) if slope.size > 0 else 0.0
        scale = max((mx - mn) / 255.0, 1e-8)
        zp = int(np.round(-mn / scale))
        zp = int(np.clip(zp, 0, 255))
        q = np.clip(np.round(slope / scale) + zp, 0, 255).astype(np.uint8)
        return q, QuantParamIR(
            scale=[float(scale)],
            zero_point=[int(zp)],
            quantized_dimension=0,
        )
    raise NotImplementedError(
        f"PRelu quantized slope requires INT8/UINT8 input. got={target_dtype}"
    )


def build_prelu_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    slope_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    slope = ctx.get_constant_array(slope_name)
    if slope is None:
        raise NotImplementedError(
            "PRelu slope must be constant for flatbuffer_direct. "
            f"op={node.name} slope_tensor={slope_name}"
        )
    slope_f = _reshape_prelu_slope_for_input(
        np.asarray(slope, dtype=np.float32),
        ctx.get_tensor_shape(input_name),
    )

    input_dtype = str(ctx.get_tensor_dtype(input_name))
    slope_tensor_name = ""
    if input_dtype in {"INT8", "UINT8"}:
        slope_q, slope_qparams = _quantize_prelu_slope(slope_f, input_dtype)
        slope_tensor_name = ctx.add_const_tensor(
            f"{node.name}_prelu_alpha_q",
            slope_q,
        )
        ctx.model_ir.tensors[slope_tensor_name].quantization = slope_qparams
        ctx.model_ir.tensors[output_name].dtype = input_dtype
        in_quant = ctx.model_ir.tensors[input_name].quantization
        if in_quant is not None and ctx.model_ir.tensors[output_name].quantization is None:
            ctx.model_ir.tensors[output_name].quantization = _clone_quantization(in_quant)
    else:
        slope_tensor_name = ctx.add_const_tensor(
            f"{node.name}_prelu_alpha",
            np.asarray(slope_f, dtype=np.float32),
        )

    ctx.add_operator(
        OperatorIR(
            op_type="PRELU",
            inputs=[input_name, slope_tensor_name],
            outputs=[output_name],
        )
    )
