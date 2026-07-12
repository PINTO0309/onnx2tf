from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR


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
