from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.core.op_contracts import (
    NodeValidationError,
    normalize_axis_for_rank as _normalize_axis_for_rank,
    require_const_input as _require_const_input,
)


def _validate_quantize_dequantize_linear(node: Any, ctx: Any) -> None:
    scale = _require_const_input(node, ctx, 1, f"{node.op} scale")
    if len(node.inputs) >= 3:
        _require_const_input(node, ctx, 2, f"{node.op} zero_point")
    if int(np.asarray(scale).size) <= 1:
        return
    axis = int(node.attrs.get("axis", 1))
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank <= 1:
        return
    _ = _normalize_axis_for_rank(
        axis=axis,
        rank=input_rank,
        node=node,
    )


def _validate_dynamic_quantize_linear(node: Any, ctx: Any) -> None:
    x_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if x_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "DynamicQuantizeLinear input dtype must be FLOAT16 or FLOAT32 for builtin lowering. "
                f"input_dtype={x_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    y_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    scale_dtype = str(ctx.get_tensor_dtype(node.outputs[1].name)).upper()
    zero_dtype = str(ctx.get_tensor_dtype(node.outputs[2].name)).upper()
    if y_dtype != "UINT8":
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "DynamicQuantizeLinear output[0] dtype must be UINT8 for builtin lowering. "
                f"output_dtype={y_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if scale_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "DynamicQuantizeLinear output[1] dtype must be FLOAT16 or FLOAT32 for builtin lowering. "
                f"output_dtype={scale_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if zero_dtype != "UINT8":
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "DynamicQuantizeLinear output[2] dtype must be UINT8 for builtin lowering. "
                f"output_dtype={zero_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    for output_index in [1, 2]:
        output_name = node.outputs[output_index].name
        shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
        # Accept scalar or scalar-like placeholder [1] only.
        if len(shape) == 0:
            continue
        if len(shape) == 1 and int(shape[0]) == 1:
            continue
        raise NodeValidationError(
            reason_code="unsupported_output_shape",
            message=(
                "DynamicQuantizeLinear scale/zero_point outputs must be scalar for builtin lowering. "
                f"output_index={output_index} shape={shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_qlinear_binary(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "a_scale"),
        (2, "a_zero_point"),
        (4, "b_scale"),
        (5, "b_zero_point"),
        (6, "c_scale"),
        (7, "c_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"{node.op} {label}")


def _validate_qlinear_concat(node: Any, ctx: Any) -> None:
    if len(node.inputs) < 5 or (len(node.inputs) - 2) % 3 != 0:
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message=(
                "QLinearConcat expects [y_scale, y_zero_point, (x, x_scale, x_zero_point)+]. "
                f"input_count={len(node.inputs)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    _require_const_input(node, ctx, 0, "QLinearConcat y_scale")
    _require_const_input(node, ctx, 1, "QLinearConcat y_zero_point")

    first_input_shape = ctx.get_tensor_shape(node.inputs[2].name)
    rank = len(first_input_shape)
    axis = int(node.attrs.get("axis", 1))
    _ = _normalize_axis_for_rank(axis=axis, rank=rank, node=node)

    for group_idx in range((len(node.inputs) - 2) // 3):
        base = 2 + group_idx * 3
        x_name = node.inputs[base].name
        x_scale_name = node.inputs[base + 1].name
        x_zero_name = node.inputs[base + 2].name
        _require_const_input(node, ctx, base + 1, f"QLinearConcat input[{group_idx}] scale")
        _require_const_input(node, ctx, base + 2, f"QLinearConcat input[{group_idx}] zero_point")
        shape_i = ctx.get_tensor_shape(x_name)
        if len(shape_i) != rank:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    f"QLinearConcat input ranks must match. "
                    f"input={x_name} shape={shape_i} expected_rank={rank}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_qlinear_conv(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    if len(input_shape) not in [1, 4] or len(output_shape) not in [1, 4]:
        raise NodeValidationError(
            reason_code="unsupported_tensor_rank",
            message=(
                "QLinearConv supports only rank-4 tensors. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    weights = _require_const_input(node, ctx, 3, "QLinearConv weights")
    if weights.ndim != 4:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"QLinearConv weight rank must be 4. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (4, "w_scale"),
        (5, "w_zero_point"),
        (6, "y_scale"),
        (7, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearConv {label}")
    group = int(node.attrs.get("group", 1))
    if len(input_shape) == 4:
        in_channels = int(input_shape[1])
        weight_in_channels_per_group = int(weights.shape[1])
        weight_out_channels = int(weights.shape[0])
        # Prefer weight/group-based depthwise detection because some quantized
        # models carry incomplete shape metadata during direct lowering.
        is_depthwise = (
            group > 1
            and weight_in_channels_per_group == 1
            and (weight_out_channels % group) == 0
        )
        if group != 1 and not is_depthwise:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "QLinearConv supports only regular or depthwise group conv. "
                    f"group={group} in_channels={in_channels} "
                    f"weight_shape={list(weights.shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    if len(node.inputs) >= 9:
        _require_const_input(node, ctx, 8, "QLinearConv bias")


def _validate_conv_integer(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    if len(input_shape) not in [1, 4] or len(output_shape) not in [1, 4]:
        raise NodeValidationError(
            reason_code="unsupported_tensor_rank",
            message=(
                "ConvInteger supports only rank-4 tensors. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    supported_input_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32"}
    x_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if x_dtype not in supported_input_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ConvInteger input dtype must be an integer tensor type for builtin lowering. "
                f"input_dtype={x_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"INT32", "INT64"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"ConvInteger output dtype must be INT32 or INT64. got={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )

    weights = _require_const_input(node, ctx, 1, "ConvInteger weights")
    if weights.ndim != 4:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"ConvInteger weight rank must be 4. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )

    group = int(node.attrs.get("group", 1))
    if len(input_shape) == 4:
        in_channels = int(input_shape[1])
        weight_in_channels_per_group = int(weights.shape[1])
        weight_out_channels = int(weights.shape[0])
        is_depthwise = (
            group > 1
            and weight_in_channels_per_group == 1
            and (weight_out_channels % group) == 0
        )
        if group != 1 and not is_depthwise:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "ConvInteger supports only regular or depthwise group conv. "
                    f"group={group} in_channels={in_channels} weight_shape={list(weights.shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 3:
        x_zero_shape = ctx.get_tensor_shape(node.inputs[2].name)
        if len(x_zero_shape) > 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=f"ConvInteger x_zero_point must be scalar or rank-1. shape={x_zero_shape}",
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 4:
        w_zero = _require_const_input(node, ctx, 3, "ConvInteger w_zero_point")
        w_zero_shape = list(np.asarray(w_zero).shape)
        if len(w_zero_shape) > 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=f"ConvInteger w_zero_point must be scalar or rank-1. shape={w_zero_shape}",
                node_name=node.name,
                node_op=node.op,
            )
        if len(w_zero_shape) == 1 and int(w_zero_shape[0]) > 1 and int(w_zero_shape[0]) != int(weights.shape[0]):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "ConvInteger w_zero_point length mismatch. "
                    f"shape={w_zero_shape} expected={int(weights.shape[0])}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_qlinear_matmul(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank not in [1, 2]:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"QLinearMatMul input rank must be 2. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    weights = _require_const_input(node, ctx, 3, "QLinearMatMul weights")
    if weights.ndim != 2:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"QLinearMatMul weight rank must be 2. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    for idx, label in [
        (1, "a_scale"),
        (2, "a_zero_point"),
        (4, "b_scale"),
        (5, "b_zero_point"),
        (6, "y_scale"),
        (7, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearMatMul {label}")


def _validate_qgemm(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank not in [1, 2]:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"QGemm input rank must be 1 or 2. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    weights = _require_const_input(node, ctx, 3, "QGemm weights")
    if weights.ndim != 2:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"QGemm weight rank must be 2. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    for idx, label in [
        (1, "a_scale"),
        (2, "a_zero_point"),
        (4, "b_scale"),
        (5, "b_zero_point"),
        (6, "bias"),
        (7, "y_scale"),
        (8, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QGemm {label}")
    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if trans_a != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QGemm transA must be 0. got={trans_a}",
            node_name=node.name,
            node_op=node.op,
        )
    if trans_b not in [0, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QGemm transB must be 0 or 1. got={trans_b}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_qlinear_sigmoid(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearSigmoid {label}")


def _validate_qlinear_leaky_relu(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearLeakyRelu {label}")


def _validate_qlinear_softmax(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearSoftmax {label}")
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    axis = int(node.attrs.get("axis", 1))
    if axis < 0:
        axis += len(input_shape)
    if axis != len(input_shape) - 1:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearSoftmax axis must be last dimension. axis={axis} shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_qlinear_global_average_pool(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearGlobalAveragePool {label}")

    channels_last = int(node.attrs.get("channels_last", 0))
    if channels_last not in [0, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearGlobalAveragePool channels_last must be 0 or 1. got={channels_last}",
            node_name=node.name,
            node_op=node.op,
        )

    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if input_shape != [1] and len(input_shape) < 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"QLinearGlobalAveragePool input rank must be >=3. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_qlinear_average_pool(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearAveragePool {label}")

    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if input_shape != [1] and len(input_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"QLinearAveragePool supports rank-4 input. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )

    kernel = [int(v) for v in list(node.attrs.get("kernel_shape", []))]
    if len(kernel) != 2:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearAveragePool kernel_shape must be 2D. kernel_shape={kernel}",
            node_name=node.name,
            node_op=node.op,
        )
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    if len(strides) != 2:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearAveragePool strides must be 2D. strides={strides}",
            node_name=node.name,
            node_op=node.op,
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if dilations != [1, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearAveragePool dilations must be [1,1]. dilations={dilations}",
            node_name=node.name,
            node_op=node.op,
        )
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode not in [0, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearAveragePool ceil_mode must be 0 or 1. got={ceil_mode}",
            node_name=node.name,
            node_op=node.op,
        )
    if ceil_mode == 1:
        auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
        pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
        if len(pads) < 4:
            pads = [0, 0, 0, 0]
        if auto_pad not in ["NOTSET", "SAME", "SAME_UPPER", "SAME_LOWER"]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "QLinearAveragePool ceil_mode=1 supports auto_pad "
                    "NOTSET/SAME/SAME_UPPER/SAME_LOWER only."
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if auto_pad == "NOTSET" and any(int(v) != 0 for v in pads):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message="QLinearAveragePool ceil_mode=1 with auto_pad=NOTSET requires pads=[0,0,0,0].",
                node_name=node.name,
                node_op=node.op,
            )
    if int(node.attrs.get("count_include_pad", 0)) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message="QLinearAveragePool count_include_pad must be 0.",
            node_name=node.name,
            node_op=node.op,
        )
