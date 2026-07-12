from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.conv import (
    _infer_conv2d_output_shape_nchw,
)
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def _as_flat_float_list(value: Any) -> List[float]:
    return [float(v) for v in np.asarray(value).reshape(-1).tolist()]


def _as_flat_int_list(value: Any) -> List[int]:
    return [int(v) for v in np.asarray(value).reshape(-1).tolist()]


def _normalize_axis(axis: int, rank: int) -> int:
    if rank <= 0:
        return 0
    a = int(axis)
    if a < 0:
        a += int(rank)
    if a < 0 or a >= int(rank):
        return 0
    return a


def _normalize_quant_params(
    *,
    scale: Any,
    zero_point: Any,
) -> Tuple[List[float], List[int]]:
    scales = _as_flat_float_list(scale)
    if len(scales) == 0:
        scales = [1.0]
    zps = _as_flat_int_list(zero_point)
    if len(zps) == 0:
        zps = [0]
    if len(zps) == 1 and len(scales) > 1:
        zps = [int(zps[0]) for _ in range(len(scales))]
    if len(scales) == 1 and len(zps) > 1:
        scales = [float(scales[0]) for _ in range(len(zps))]
    if len(scales) != len(zps):
        raise NotImplementedError(
            "scale and zero_point sizes must match or be broadcastable. "
            f"scale_len={len(scales)} zero_point_len={len(zps)}"
        )
    return scales, zps


def _tflite_dtype_from_numpy_dtype(np_dtype: np.dtype) -> str:
    dt = np.dtype(np_dtype)
    if dt == np.dtype(np.int8):
        return "INT8"
    if dt == np.dtype(np.uint8):
        return "UINT8"
    if dt == np.dtype(np.int16):
        return "INT16"
    if dt == np.dtype(np.uint16):
        return "UINT16"
    if dt == np.dtype(np.int32):
        return "INT32"
    if dt == np.dtype(np.uint32):
        return "UINT32"
    if dt == np.dtype(np.float16):
        return "FLOAT16"
    if dt == np.dtype(np.float32):
        return "FLOAT32"
    raise NotImplementedError(f"Unsupported dtype for flatbuffer_direct quantized builder: {dt}")


def _numpy_dtype_from_tflite_dtype(tflite_dtype: str) -> np.dtype:
    dt = str(tflite_dtype).upper()
    if dt == "INT8":
        return np.dtype(np.int8)
    if dt == "UINT8":
        return np.dtype(np.uint8)
    if dt == "INT16":
        return np.dtype(np.int16)
    if dt == "UINT16":
        return np.dtype(np.uint16)
    if dt == "INT32":
        return np.dtype(np.int32)
    if dt == "UINT32":
        return np.dtype(np.uint32)
    if dt == "FLOAT16":
        return np.dtype(np.float16)
    if dt == "FLOAT32":
        return np.dtype(np.float32)
    raise NotImplementedError(f"Unsupported TFLite dtype in quantized builder: {tflite_dtype}")


def _set_tensor_dtype_from_array(ctx: Any, tensor_name: str, array: np.ndarray) -> None:
    ctx.ensure_tensor(tensor_name)
    tensor = ctx.model_ir.tensors[tensor_name]
    new_dtype = _tflite_dtype_from_numpy_dtype(np.asarray(array).dtype)
    if tensor.dtype == "FLOAT32" and new_dtype != "FLOAT32":
        tensor.dtype = new_dtype


def _set_tensor_quantization(
    *,
    ctx: Any,
    tensor_name: str,
    scale: Any,
    zero_point: Any,
    quantized_dimension: int,
) -> None:
    ctx.ensure_tensor(tensor_name)
    scales, zps = _normalize_quant_params(scale=scale, zero_point=zero_point)
    tensor_dtype = str(ctx.get_tensor_dtype(tensor_name))
    zero_point_dtype = np.asarray(zero_point).dtype
    if tensor_dtype == "INT8" and zero_point_dtype == np.dtype(np.uint8):
        zps = [int(v) - 128 for v in zps]
    elif tensor_dtype == "UINT8" and zero_point_dtype == np.dtype(np.int8):
        zps = [int(v) + 128 for v in zps]
    ctx.model_ir.tensors[tensor_name].quantization = QuantParamIR(
        scale=[float(v) for v in scales],
        zero_point=[int(v) for v in zps],
        quantized_dimension=int(quantized_dimension),
    )


def _add_scalar_onnx_requantization(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    input_is_scaled_quant_units: bool = False,
    wrap_on_overflow: bool = False,
) -> bool:
    """Requantize float data with ONNX round-then-saturate semantics.

    TFLite's quantized kernels use fixed-point requantizers whose rounding can
    differ by one quantum from ONNX Runtime.  A one-quantum error in an early
    QLinearConv can be amplified by the following quantized layers.  For the
    mixed UINT8-activation/INT8-weight compatibility path, keep the declared
    quantized boundary but make the scalar requantization arithmetic explicit.
    """

    output_tensor = ctx.model_ir.tensors.get(output_name)
    quantization = (
        output_tensor.quantization
        if output_tensor is not None
        and isinstance(output_tensor.quantization, QuantParamIR)
        else None
    )
    if (
        output_tensor is None
        or quantization is None
        or len(quantization.scale) != 1
        or len(quantization.zero_point) != 1
    ):
        return False

    output_dtype = str(output_tensor.dtype).upper()
    dtype_limits = {
        "INT8": (-128.0, 127.0),
        "UINT8": (0.0, 255.0),
        "INT16": (-32768.0, 32767.0),
        "UINT16": (0.0, 65535.0),
    }
    if output_dtype not in dtype_limits:
        return False

    output_shape = list(output_tensor.shape)
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_shape)
    )

    def intermediate(suffix: str) -> str:
        name = ctx.add_intermediate_tensor(
            f"{output_name}_onnx_requant_{suffix}",
            dtype="FLOAT32",
            shape=output_shape,
        )
        ctx.model_ir.tensors[name].shape_signature = list(output_signature)
        return name

    zero_name = ctx.add_const_tensor(
        f"{output_name}_onnx_requant_zero",
        np.asarray(float(quantization.zero_point[0]), dtype=np.float32),
    )
    qmin, qmax = dtype_limits[output_dtype]
    scaled_name = input_name
    rounded_name = intermediate("rounded")
    shifted_name = intermediate("shifted")
    if not bool(input_is_scaled_quant_units):
        scale_name = ctx.add_const_tensor(
            f"{output_name}_onnx_requant_scale",
            np.asarray(float(quantization.scale[0]), dtype=np.float32),
        )
        scaled_name = intermediate("scaled")
        ctx.add_operator(
            OperatorIR(
                op_type="DIV",
                inputs=[input_name, scale_name],
                outputs=[scaled_name],
                options={
                    "fusedActivationFunction": "NONE",
                    "preserveDivisionForOnnxRequantization": True,
                },
            )
        )
    ctx.add_operator(
        OperatorIR(
            op_type="ROUND",
            inputs=[scaled_name],
            outputs=[rounded_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[rounded_name, zero_name],
            outputs=[shifted_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    cast_input_name = shifted_name
    if bool(wrap_on_overflow):
        qmin_name = ctx.add_const_tensor(
            f"{output_name}_onnx_requant_min",
            np.asarray(qmin, dtype=np.float32),
        )
        modulus_name = ctx.add_const_tensor(
            f"{output_name}_onnx_requant_modulus",
            np.asarray(qmax - qmin + 1.0, dtype=np.float32),
        )
        offset_name = intermediate("wrap_offset")
        quotient_name = intermediate("wrap_quotient")
        cycles_name = intermediate("wrap_cycles")
        cycle_span_name = intermediate("wrap_cycle_span")
        wrapped_offset_name = intermediate("wrapped_offset")
        wrapped_name = intermediate("wrapped")
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[shifted_name, qmin_name],
                outputs=[offset_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="DIV",
                inputs=[offset_name, modulus_name],
                outputs=[quotient_name],
                options={
                    "fusedActivationFunction": "NONE",
                    "preserveDivisionForOnnxRequantization": True,
                },
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="FLOOR",
                inputs=[quotient_name],
                outputs=[cycles_name],
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[cycles_name, modulus_name],
                outputs=[cycle_span_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[offset_name, cycle_span_name],
                outputs=[wrapped_offset_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[wrapped_offset_name, qmin_name],
                outputs=[wrapped_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        cast_input_name = wrapped_name
    else:
        qmin_name = ctx.add_const_tensor(
            f"{output_name}_onnx_requant_min",
            np.asarray(qmin, dtype=np.float32),
        )
        qmax_name = ctx.add_const_tensor(
            f"{output_name}_onnx_requant_max",
            np.asarray(qmax, dtype=np.float32),
        )
        lower_clamped_name = intermediate("lower_clamped")
        clamped_name = intermediate("clamped")
        ctx.add_operator(
            OperatorIR(
                op_type="MAXIMUM",
                inputs=[shifted_name, qmin_name],
                outputs=[lower_clamped_name],
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MINIMUM",
                inputs=[lower_clamped_name, qmax_name],
                outputs=[clamped_name],
            )
        )
        cast_input_name = clamped_name
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[cast_input_name],
            outputs=[output_name],
            options={"inDataType": "FLOAT32", "outDataType": output_dtype},
        )
    )
    return True


def _promote_internal_uint8_tensor_to_int8(ctx: Any, tensor_name: str) -> None:
    if str(tensor_name) in set(ctx.graph_output_names):
        return
    ctx.ensure_tensor(tensor_name)
    tensor = ctx.model_ir.tensors.get(tensor_name, None)
    if tensor is None or str(tensor.dtype) != "UINT8":
        return
    tensor.dtype = "INT8"
    if isinstance(tensor.data, np.ndarray):
        shifted = np.asarray(tensor.data, dtype=np.int16) - 128
        tensor.data = np.clip(shifted, -128, 127).astype(np.int8)
        if tensor_name in ctx.constants:
            ctx.constants[tensor_name] = tensor.data
    if isinstance(tensor.quantization, QuantParamIR):
        tensor.quantization.zero_point = [
            int(v) - 128 for v in list(tensor.quantization.zero_point)
        ]


def _propagate_shape(ctx: Any, src_tensor_name: str, dst_tensor_name: str) -> None:
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src = ctx.model_ir.tensors[src_tensor_name]
    dst = ctx.model_ir.tensors[dst_tensor_name]
    src_signature = (
        list(src.shape_signature)
        if src.shape_signature is not None
        else list(src.shape)
    )
    same_rank_all_ones_placeholder = bool(
        len(list(dst.shape)) == len(list(src.shape))
        and list(dst.shape) != list(src.shape)
        and len(list(dst.shape)) > 0
        and all(int(v) == 1 for v in list(dst.shape))
        and _feeds_max_pool_through_quantized_passthrough(ctx, dst_tensor_name)
    )
    if (dst.shape == [1] and src.shape != [1]) or same_rank_all_ones_placeholder:
        dst.shape = list(src.shape)
        dst.shape_signature = list(src_signature)
    elif len(list(dst.shape)) == len(list(src.shape)) and list(dst.shape) == list(src.shape):
        dst.shape_signature = list(src_signature)


def _feeds_max_pool_through_quantized_passthrough(
    ctx: Any,
    tensor_name: str,
) -> bool:
    passthrough_ops = {
        "DequantizeLinear",
        "Identity",
        "QLinearLeakyRelu",
    }
    frontier = [str(tensor_name)]
    visited: set[str] = set()
    for _ in range(4):
        next_frontier: List[str] = []
        for current_name in frontier:
            if current_name in visited:
                continue
            visited.add(current_name)
            for consumer in ctx.onnx_tensor_consumers.get(current_name, []):
                op_type = str(getattr(consumer, "op_type", ""))
                if op_type == "MaxPool":
                    strides = [
                        int(value)
                        for attr in consumer.attribute
                        if str(attr.name) == "strides"
                        for value in attr.ints
                    ]
                    if len(strides) == 0:
                        strides = [1, 1]
                    if any(int(value) > 1 for value in strides):
                        return True
                    continue
                if op_type not in passthrough_ops:
                    continue
                next_frontier.extend(
                    str(output_name)
                    for output_name in consumer.output
                    if str(output_name) != ""
                )
        frontier = next_frontier
        if len(frontier) == 0:
            break
    return False


def _infer_rank4_conv_output_signature(
    *,
    input_signature_nchw: List[int],
    output_shape_nchw: List[int],
    existing_output_signature_nchw: Optional[List[int]] = None,
) -> List[int]:
    signature = [int(v) for v in list(output_shape_nchw)]
    if len(signature) != 4:
        return signature
    if existing_output_signature_nchw is not None and len(existing_output_signature_nchw) == 4:
        for axis in range(4):
            if int(existing_output_signature_nchw[axis]) < 0:
                signature[axis] = -1
    if len(input_signature_nchw) == 4:
        if int(input_signature_nchw[0]) < 0:
            signature[0] = -1
        if int(input_signature_nchw[2]) < 0:
            signature[2] = -1
        if int(input_signature_nchw[3]) < 0:
            signature[3] = -1
    return [int(v) for v in signature]


def _shape_from_rank4_signature(signature: List[int]) -> Optional[List[int]]:
    if len(signature) != 4:
        return None
    return [int(v) if int(v) > 0 else 1 for v in list(signature)]


def _require_const(ctx: Any, tensor_name: str, label: str) -> np.ndarray:
    value = ctx.get_constant_array(tensor_name)
    if value is None:
        raise NotImplementedError(
            f"{label} must be constant for flatbuffer_direct. tensor={tensor_name}"
        )
    return np.asarray(value)


def _infer_pool_output_hw_for_qlinear(
    *,
    node: Any,
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    ceil_mode: int = 0,
) -> tuple[int, int]:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    if auto_pad in ["SAME", "SAME_UPPER", "SAME_LOWER"]:
        out_h = int(math.ceil(float(input_h) / float(stride_h)))
        out_w = int(math.ceil(float(input_w) / float(stride_w)))
        return max(out_h, 1), max(out_w, 1)
    if auto_pad == "VALID":
        if int(ceil_mode) == 1:
            out_h = int(math.ceil((float(input_h) - float(kernel_h)) / float(stride_h) + 1.0))
            out_w = int(math.ceil((float(input_w) - float(kernel_w)) / float(stride_w) + 1.0))
        else:
            out_h = int(math.floor((float(input_h) - float(kernel_h)) / float(stride_h) + 1.0))
            out_w = int(math.floor((float(input_w) - float(kernel_w)) / float(stride_w) + 1.0))
        return max(out_h, 1), max(out_w, 1)

    pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(pads) < 4:
        pads = [0, 0, 0, 0]
    pad_top, pad_left, pad_bottom, pad_right = pads[0], pads[1], pads[2], pads[3]
    if int(ceil_mode) == 1:
        out_h = int(
            math.ceil(
                (float(input_h + pad_top + pad_bottom - kernel_h) / float(stride_h)) + 1.0
            )
        )
        out_w = int(
            math.ceil(
                (float(input_w + pad_left + pad_right - kernel_w) / float(stride_w)) + 1.0
            )
        )
    else:
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


def _resolve_qlinear_conv_padding_and_explicit_pads(
    *,
    node: Any,
    input_shape_nchw: List[int],
    output_shape_nchw: List[int],
) -> Tuple[str, List[int] | None]:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    raw_pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(raw_pads) < 4:
        raw_pads = [0, 0, 0, 0]
    pads = [int(raw_pads[0]), int(raw_pads[1]), int(raw_pads[2]), int(raw_pads[3])]
    pad_top, pad_left, pad_bottom, pad_right = pads
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    kernel = [int(v) for v in list(node.attrs.get("kernel_shape", [1, 1]))]
    explicit_pads_match_same_upper = False
    if strides == [1, 1] and len(kernel) == 2 and len(dilations) == 2:
        effective_h = (int(kernel[0]) - 1) * int(dilations[0]) + 1
        effective_w = (int(kernel[1]) - 1) * int(dilations[1]) + 1
        same_top = (effective_h - 1) // 2
        same_bottom = (effective_h - 1) - same_top
        same_left = (effective_w - 1) // 2
        same_right = (effective_w - 1) - same_left
        explicit_pads_match_same_upper = pads == [
            same_top,
            same_left,
            same_bottom,
            same_right,
        ]

    if auto_pad == "NOTSET":
        if (
            explicit_pads_match_same_upper
            and len(input_shape_nchw) == 4
            and len(output_shape_nchw) == 4
            and list(input_shape_nchw[2:]) == list(output_shape_nchw[2:])
        ):
            return "SAME", None
        if any(int(v) != 0 for v in pads):
            return "VALID", pads
        return "VALID", None

    if auto_pad == "SAME_UPPER":
        return "SAME", None
    if auto_pad == "VALID":
        if any(int(v) != 0 for v in pads):
            return "VALID", pads
        return "VALID", None
    if auto_pad == "SAME_LOWER":
        raise NotImplementedError(
            f"QLinearConv auto_pad=SAME_LOWER is not supported in flatbuffer_direct. op={node.name}"
        )
    raise NotImplementedError(
        f"QLinearConv auto_pad attribute is invalid for flatbuffer_direct. op={node.name} auto_pad={auto_pad}"
    )


def build_quantize_linear_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    y_scale_name = node.inputs[1].name
    y_zero_point_name = node.inputs[2].name if len(node.inputs) >= 3 else ""
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    y_scale = _require_const(ctx, y_scale_name, "QuantizeLinear scale")
    if y_zero_point_name != "":
        y_zero_point = _require_const(ctx, y_zero_point_name, "QuantizeLinear zero_point")
    else:
        y_zero_point = np.zeros_like(y_scale, dtype=np.int32)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero_point)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    output_rank = len(ctx.get_tensor_shape(output_name))
    axis = int(node.attrs.get("axis", 1))
    qdim = _normalize_axis(axis, output_rank)
    if np.asarray(y_scale).size <= 1:
        qdim = 0
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero_point,
        quantized_dimension=qdim,
    )

    use_onnx_requantization = (
        np.asarray(y_zero_point).dtype == np.dtype(np.uint8)
        and str(ctx.get_tensor_dtype(output_name)).upper() == "INT8"
    )
    if not use_onnx_requantization or not _add_scalar_onnx_requantization(
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
    ):
        ctx.add_operator(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[input_name],
                outputs=[output_name],
            )
        )

def build_dequantize_linear_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_point_name = node.inputs[2].name if len(node.inputs) >= 3 else ""
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    x_scale = _require_const(ctx, x_scale_name, "DequantizeLinear scale")
    if x_zero_point_name != "":
        x_zero_point = _require_const(ctx, x_zero_point_name, "DequantizeLinear zero_point")
    else:
        x_zero_point = np.zeros_like(x_scale, dtype=np.int32)
    _set_tensor_dtype_from_array(ctx, input_name, x_zero_point)
    ctx.model_ir.tensors[output_name].dtype = "FLOAT32"

    input_rank = len(ctx.get_tensor_shape(input_name))
    axis = int(node.attrs.get("axis", 1))
    qdim = _normalize_axis(axis, input_rank)
    if np.asarray(x_scale).size <= 1:
        qdim = 0
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=input_name,
        scale=x_scale,
        zero_point=x_zero_point,
        quantized_dimension=qdim,
    )

    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[input_name],
            outputs=[output_name],
        )
    )


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


def build_qlinear_concat_op(node: Any, ctx: Any) -> None:
    y_scale_name = node.inputs[0].name
    y_zero_name = node.inputs[1].name
    output_name = node.outputs[0].name

    if (len(node.inputs) - 2) % 3 != 0 or len(node.inputs) < 5:
        raise NotImplementedError(
            f"QLinearConcat inputs must be [y_scale, y_zero_point, (x, x_scale, x_zero_point)+]. "
            f"op={node.name} input_count={len(node.inputs)}"
        )

    input_groups = (len(node.inputs) - 2) // 3
    input_names: list[str] = []
    input_scale_names: list[str] = []
    input_zero_names: list[str] = []
    for i in range(input_groups):
        base = 2 + i * 3
        input_names.append(node.inputs[base].name)
        input_scale_names.append(node.inputs[base + 1].name)
        input_zero_names.append(node.inputs[base + 2].name)

    for input_name in input_names:
        ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    y_scale = _require_const(ctx, y_scale_name, "QLinearConcat output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearConcat output zero_point")
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    first_shape = [int(v) for v in ctx.get_tensor_shape(input_names[0])]
    rank = len(first_shape)
    axis = int(node.attrs.get("axis", 1))
    axis = _normalize_axis(axis, rank)

    input_signatures: list[list[int]] = []
    for idx, input_name in enumerate(input_names):
        input_scale = _require_const(ctx, input_scale_names[idx], f"QLinearConcat input[{idx}] scale")
        input_zero = _require_const(ctx, input_zero_names[idx], f"QLinearConcat input[{idx}] zero_point")
        _set_tensor_dtype_from_array(ctx, input_name, input_zero)
        _promote_internal_uint8_tensor_to_int8(ctx, input_name)
        _set_tensor_quantization(
            ctx=ctx,
            tensor_name=input_name,
            scale=input_scale,
            zero_point=input_zero,
            quantized_dimension=0 if np.asarray(input_scale).size <= 1 else _normalize_axis(1, rank),
        )
        input_tensor = ctx.model_ir.tensors[input_name]
        input_signature = (
            list(input_tensor.shape_signature)
            if input_tensor.shape_signature is not None
            else list(input_tensor.shape)
        )
        input_signatures.append(input_signature)

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, rank),
    )

    output_shape = [int(v) for v in first_shape]
    output_signature = list(input_signatures[0]) if len(input_signatures) > 0 else list(output_shape)
    concat_dim = 0
    concat_sig_dim = 0
    for idx, input_name in enumerate(input_names):
        shape_i = [int(v) for v in ctx.get_tensor_shape(input_name)]
        sig_i = input_signatures[idx]
        if len(shape_i) != rank:
            raise NotImplementedError(
                f"QLinearConcat input ranks must match. op={node.name} input={input_name} shape={shape_i}"
            )
        concat_dim += int(shape_i[axis])
        concat_sig_dim += int(sig_i[axis]) if int(sig_i[axis]) >= 0 else 0
    output_shape[axis] = int(concat_dim)
    output_signature[axis] = int(concat_sig_dim) if concat_sig_dim > 0 else -1
    ctx.model_ir.tensors[output_name].shape = list(output_shape)
    ctx.model_ir.tensors[output_name].shape_signature = list(output_signature)

    dq_inputs: list[str] = []
    for input_name in input_names:
        dq_name = ctx.add_intermediate_tensor(
            f"{node.name}_{input_name}_dq",
            dtype="FLOAT32",
            shape=ctx.get_tensor_shape(input_name),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[input_name],
                outputs=[dq_name],
            )
        )
        dq_inputs.append(dq_name)

    concat_out = ctx.add_intermediate_tensor(
        f"{node.name}_concat_out",
        dtype="FLOAT32",
        shape=list(output_shape),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=dq_inputs,
            outputs=[concat_out],
            options={
                "axis": int(axis),
                "fusedActivationFunction": "NONE",
            },
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="QUANTIZE",
            inputs=[concat_out],
            outputs=[output_name],
        )
    )
