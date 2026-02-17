from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose, resolve_padding


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
    if dst.shape == [1] and src.shape != [1]:
        dst.shape = list(src.shape)
        dst.shape_signature = list(src_signature)
    elif len(list(dst.shape)) == len(list(src.shape)) and list(dst.shape) == list(src.shape):
        dst.shape_signature = list(src_signature)


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
    pads_axes_opposite_same = bool((pad_top == pad_bottom) and (pad_left == pad_right))

    if auto_pad == "NOTSET":
        if (
            pads_axes_opposite_same
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


def _build_qlinear_binary_op(node: Any, ctx: Any, op_type: str) -> None:
    a_name = node.inputs[0].name
    a_scale_name = node.inputs[1].name
    a_zero_name = node.inputs[2].name
    b_name = node.inputs[3].name
    b_scale_name = node.inputs[4].name
    b_zero_name = node.inputs[5].name
    c_scale_name = node.inputs[6].name
    c_zero_name = node.inputs[7].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(a_name)
    ctx.ensure_tensor(b_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, a_name, output_name)

    a_scale = _require_const(ctx, a_scale_name, f"{node.op} input-a scale")
    a_zero = _require_const(ctx, a_zero_name, f"{node.op} input-a zero_point")
    b_scale = _require_const(ctx, b_scale_name, f"{node.op} input-b scale")
    b_zero = _require_const(ctx, b_zero_name, f"{node.op} input-b zero_point")
    c_scale = _require_const(ctx, c_scale_name, f"{node.op} output scale")
    c_zero = _require_const(ctx, c_zero_name, f"{node.op} output zero_point")
    _set_tensor_dtype_from_array(ctx, a_name, a_zero)
    _set_tensor_dtype_from_array(ctx, b_name, b_zero)
    _set_tensor_dtype_from_array(ctx, output_name, c_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, a_name)
    _promote_internal_uint8_tensor_to_int8(ctx, b_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    a_rank = len(ctx.get_tensor_shape(a_name))
    b_rank = len(ctx.get_tensor_shape(b_name))
    out_rank = len(ctx.get_tensor_shape(output_name))

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=a_name,
        scale=a_scale,
        zero_point=a_zero,
        quantized_dimension=0 if np.asarray(a_scale).size <= 1 else _normalize_axis(1, a_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=b_name,
        scale=b_scale,
        zero_point=b_zero,
        quantized_dimension=0 if np.asarray(b_scale).size <= 1 else _normalize_axis(1, b_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=c_scale,
        zero_point=c_zero,
        quantized_dimension=0 if np.asarray(c_scale).size <= 1 else _normalize_axis(1, out_rank),
    )

    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[a_name, b_name],
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )


def build_qlinear_add_op(node: Any, ctx: Any) -> None:
    _build_qlinear_binary_op(node, ctx, "ADD")


def build_qlinear_mul_op(node: Any, ctx: Any) -> None:
    _build_qlinear_binary_op(node, ctx, "MUL")


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
    input_shape = ctx.get_tensor_shape(x_name)
    output_shape = ctx.get_tensor_shape(output_name)
    input_tensor = ctx.model_ir.tensors[x_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )
    existing_output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None and len(list(output_tensor.shape_signature)) == 4
        else None
    )
    if len(output_shape) != 4 and len(input_shape) == 4:
        inferred_output_shape = [
            int(input_shape[0]),
            int(weights.shape[0]),
            int(input_shape[2]),
            int(input_shape[3]),
        ]
        ctx.model_ir.tensors[output_name].shape = inferred_output_shape
        output_shape = inferred_output_shape
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
    group = int(node.attrs.get("group", 1))
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

    in_channels = int(nchw_input[1])
    out_channels = int(weights.shape[0])
    weight_in_channels_per_group = int(weights.shape[1])
    # Keep depthwise detection aligned with op_registry validator:
    # rely on group/weight shape rather than potentially stale input metadata.
    is_depthwise = (
        group > 1
        and weight_in_channels_per_group == 1
        and (out_channels % group) == 0
    )

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
    bias_q_name = ctx.add_const_tensor(
        f"{node.name}_conv_bias_q",
        bias_values,
    )
    x_scales, _ = _normalize_quant_params(scale=x_scale, zero_point=x_zero)
    w_scales, _ = _normalize_quant_params(scale=w_scale, zero_point=w_zero)
    if len(w_scales) == 1:
        bias_scales = [float(x_scales[0] * w_scales[0])]
    else:
        bias_scales = [float(x_scales[0] * ws) for ws in w_scales]
    ctx.model_ir.tensors[bias_q_name].quantization = QuantParamIR(
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

    if is_depthwise:
        ctx.add_operator(
            OperatorIR(
                op_type="DEPTHWISE_CONV_2D",
                inputs=[x_nhwc_conv, w_q_name, bias_q_name],
                outputs=[y_nhwc_conv],
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
                inputs=[x_nhwc_conv, w_q_name, bias_q_name],
                outputs=[y_nhwc_conv],
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
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )


def _build_qlinear_fc_like_op(
    node: Any,
    ctx: Any,
    *,
    has_bias_input: bool,
) -> None:
    a_name = node.inputs[0].name
    a_scale_name = node.inputs[1].name
    a_zero_name = node.inputs[2].name
    b_name = node.inputs[3].name
    b_scale_name = node.inputs[4].name
    b_zero_name = node.inputs[5].name
    bias_name = node.inputs[6].name if has_bias_input else ""
    y_scale_name = node.inputs[7].name if has_bias_input else node.inputs[6].name
    y_zero_name = node.inputs[8].name if has_bias_input else node.inputs[7].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(a_name)
    ctx.ensure_tensor(b_name)
    ctx.ensure_tensor(output_name)

    a_scale = _require_const(ctx, a_scale_name, "QLinearMatMul input-a scale")
    a_zero = _require_const(ctx, a_zero_name, "QLinearMatMul input-a zero_point")
    b_scale = _require_const(ctx, b_scale_name, "QLinearMatMul input-b scale")
    b_zero = _require_const(ctx, b_zero_name, "QLinearMatMul input-b zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearMatMul output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearMatMul output zero_point")
    _set_tensor_dtype_from_array(ctx, a_name, a_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, a_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if trans_a != 0:
        raise NotImplementedError(
            f"{node.op} transA=1 is not supported in flatbuffer_direct. op={node.name}"
        )
    if trans_b not in [0, 1]:
        raise NotImplementedError(
            f"{node.op} transB must be 0 or 1 in flatbuffer_direct. op={node.name} transB={trans_b}"
        )

    weights = _require_const(ctx, b_name, "QLinearMatMul weights")
    if weights.ndim != 2:
        raise NotImplementedError(
            f"QLinearMatMul weight rank must be 2. op={node.name} weight_shape={list(weights.shape)}"
        )
    if trans_b == 0:
        # ONNX B is [K, N] when transB=0. TFLite FC expects [N, K].
        fc_weights = np.asarray(weights.T, dtype=weights.dtype)
    else:
        # ONNX B is [N, K] when transB=1. It already matches TFLite FC layout.
        fc_weights = np.asarray(weights, dtype=weights.dtype)
    out_features = int(fc_weights.shape[0])

    input_shape = ctx.get_tensor_shape(a_name)
    if len(input_shape) != 2:
        output_shape = ctx.get_tensor_shape(output_name)
        if len(output_shape) == 2:
            inferred_shape = [int(output_shape[0]), int(fc_weights.shape[1])]
        else:
            inferred_shape = [1, int(fc_weights.shape[1])]
        ctx.model_ir.tensors[a_name].shape = inferred_shape
        ctx.model_ir.tensors[a_name].shape_signature = list(inferred_shape)
        input_shape = inferred_shape

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=a_name,
        scale=a_scale,
        zero_point=a_zero,
        quantized_dimension=0 if np.asarray(a_scale).size <= 1 else 1,
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else 1,
    )
    if ctx.model_ir.tensors[output_name].shape == [1]:
        ctx.model_ir.tensors[output_name].shape = [int(input_shape[0]), int(out_features)]
        ctx.model_ir.tensors[output_name].shape_signature = [int(input_shape[0]), int(out_features)]
    a_fc_name = a_name
    w_q_name = ctx.add_const_tensor(
        f"{node.name}_fc_weights_q",
        fc_weights,
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=w_q_name,
        scale=b_scale,
        zero_point=b_zero,
        quantized_dimension=0 if np.asarray(b_scale).size <= 1 else 0,
    )
    if has_bias_input:
        bias_values = _require_const(ctx, bias_name, f"{node.op} bias")
        bias_values = np.asarray(bias_values, dtype=np.int32).reshape(-1)
        if bias_values.size == 1 and out_features > 1:
            bias_values = np.repeat(bias_values, repeats=out_features, axis=0)
        if int(bias_values.size) != int(out_features):
            raise NotImplementedError(
                f"{node.op} bias size must match output features. "
                f"op={node.name} bias_size={int(bias_values.size)} out_features={out_features}"
            )
    else:
        bias_values = np.zeros((out_features,), dtype=np.int32)

    a_scales, _ = _normalize_quant_params(scale=a_scale, zero_point=a_zero)
    b_scales, _ = _normalize_quant_params(scale=b_scale, zero_point=b_zero)
    if len(b_scales) == 1:
        bias_scales = [float(a_scales[0] * b_scales[0])]
    else:
        bias_scales = [float(a_scales[0] * bs) for bs in b_scales]
    bias_q_name = ctx.add_const_tensor(
        f"{node.name}_fc_bias_q",
        bias_values,
    )
    ctx.model_ir.tensors[bias_q_name].quantization = QuantParamIR(
        scale=bias_scales,
        zero_point=[0 for _ in range(len(bias_scales))],
        quantized_dimension=0,
    )
    y_fc_name = output_name
    ctx.add_operator(
        OperatorIR(
            op_type="FULLY_CONNECTED",
            inputs=[a_fc_name, w_q_name, bias_q_name],
            outputs=[y_fc_name],
            options={
                "fusedActivationFunction": "NONE",
                "weightsFormat": "DEFAULT",
                "keepNumDims": False,
                "asymmetricQuantizeInputs": False,
            },
            version=4,
        )
    )

def build_qlinear_matmul_op(node: Any, ctx: Any) -> None:
    _build_qlinear_fc_like_op(
        node=node,
        ctx=ctx,
        has_bias_input=False,
    )


def build_qgemm_op(node: Any, ctx: Any) -> None:
    _build_qlinear_fc_like_op(
        node=node,
        ctx=ctx,
        has_bias_input=True,
    )


def build_qlinear_average_pool_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearAveragePool input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearAveragePool input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearAveragePool output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearAveragePool output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    if len(input_shape) != 4:
        raise NotImplementedError(
            f"QLinearAveragePool supports only rank-4 input for flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )

    kernel = [int(v) for v in list(node.attrs.get("kernel_shape", [1, 1]))]
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    if len(kernel) != 2 or len(strides) != 2:
        raise NotImplementedError(
            f"QLinearAveragePool supports only 2D pooling in flatbuffer_direct. "
            f"op={node.name} kernel_shape={kernel} strides={strides}"
        )
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode not in [0, 1]:
        raise NotImplementedError(
            f"QLinearAveragePool ceil_mode must be 0 or 1 for flatbuffer_direct. op={node.name} ceil_mode={ceil_mode}"
        )
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(pads) < 4:
        pads = [0, 0, 0, 0]
    if ceil_mode == 1:
        if auto_pad not in ["NOTSET", "SAME", "SAME_UPPER", "SAME_LOWER"]:
            raise NotImplementedError(
                f"QLinearAveragePool ceil_mode=1 supports auto_pad NOTSET/SAME only. op={node.name} auto_pad={auto_pad}"
            )
        if auto_pad == "NOTSET" and any(int(v) != 0 for v in pads):
            raise NotImplementedError(
                f"QLinearAveragePool ceil_mode=1 with auto_pad=NOTSET requires pads=[0,0,0,0]. op={node.name} pads={pads}"
            )
    if int(node.attrs.get("count_include_pad", 0)) != 0:
        raise NotImplementedError(
            f"QLinearAveragePool count_include_pad must be 0 for flatbuffer_direct. op={node.name}"
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if dilations != [1, 1]:
        raise NotImplementedError(
            f"QLinearAveragePool dilations must be [1,1] for flatbuffer_direct. "
            f"op={node.name} dilations={dilations}"
        )

    if ceil_mode == 1:
        # TFLite has no explicit ceil_mode flag; use SAME-style output sizing for this supported subset.
        padding = "SAME"
    else:
        padding = resolve_padding(node)

    input_tensor = ctx.model_ir.tensors[x_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )
    if len(output_shape) != 4:
        out_h, out_w = _infer_pool_output_hw_for_qlinear(
            node=node,
            input_h=int(input_shape[2]),
            input_w=int(input_shape[3]),
            kernel_h=int(kernel[0]),
            kernel_w=int(kernel[1]),
            stride_h=int(strides[0]),
            stride_w=int(strides[1]),
            ceil_mode=ceil_mode,
        )
        output_shape = [int(input_shape[0]), int(input_shape[1]), int(out_h), int(out_w)]
        output_tensor = ctx.model_ir.tensors[output_name]
        output_tensor.shape = list(output_shape)
        output_signature = list(output_shape)
        if len(input_signature) == 4:
            output_signature[0] = int(input_signature[0])
            output_signature[1] = int(input_signature[1])
        output_tensor.shape_signature = list(output_signature)

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(1, 4),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, 4),
    )

    nhwc_input_shape = [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
    nhwc_output_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[3]), int(output_shape[1])]
    output_tensor = ctx.model_ir.tensors[output_name]
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_shape)
    )
    nhwc_output_signature = [int(v) for v in nhwc_output_shape]
    if len(output_signature) == 4:
        nhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[1]),
        ]
    nhwc_input_signature = [
        int(input_signature[0]),
        int(input_signature[2]),
        int(input_signature[3]),
        int(input_signature[1]),
    ]

    def _lower_via_dequantize_avgpool_quantize() -> None:
        dq_out = ctx.add_intermediate_tensor(
            f"{node.name}_dq_out",
            dtype="FLOAT32",
            shape=list(input_shape),
        )
        ctx.model_ir.tensors[dq_out].shape_signature = list(input_signature)
        ctx.add_operator(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[x_name],
                outputs=[dq_out],
            )
        )

        x_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_input_nhwc",
            dtype="FLOAT32",
            shape=nhwc_input_shape,
        )
        ctx.model_ir.tensors[x_nhwc].shape_signature = list(nhwc_input_signature)
        x_nhwc = make_transpose(
            ctx,
            dq_out,
            x_nhwc,
            [0, 2, 3, 1],
            allow_elide_inverse_chain=True,
        )

        y_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc",
            dtype="FLOAT32",
            shape=nhwc_output_shape,
        )
        ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in nhwc_output_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="AVERAGE_POOL_2D",
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

        pool_out_nchw = ctx.add_intermediate_tensor(
            f"{node.name}_pool_out_nchw",
            dtype="FLOAT32",
            shape=list(output_shape),
        )
        ctx.model_ir.tensors[pool_out_nchw].shape_signature = list(output_signature)
        make_transpose(
            ctx,
            y_nhwc,
            pool_out_nchw,
            [0, 3, 1, 2],
        )

        ctx.add_operator(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[pool_out_nchw],
                outputs=[output_name],
            )
        )

    # Prefer quantized AVERAGE_POOL_2D and avoid DEQUANTIZE/QUANTIZE bridges.
    # Keep this conservative: only per-tensor activation quantization.
    can_use_quantized_pool = (
        len(input_shape) == 4
        and len(output_shape) == 4
        and np.asarray(x_scale).size <= 1
        and np.asarray(y_scale).size <= 1
    )
    if not can_use_quantized_pool:
        _lower_via_dequantize_avgpool_quantize()
        return

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(x_name),
        shape=nhwc_input_shape,
    )
    ctx.model_ir.tensors[x_nhwc].shape_signature = list(nhwc_input_signature)
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_nhwc,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0,
    )
    x_nhwc = make_transpose(
        ctx,
        x_name,
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
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=y_nhwc,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="AVERAGE_POOL_2D",
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


def build_qlinear_global_average_pool_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearGlobalAveragePool input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearGlobalAveragePool input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearGlobalAveragePool output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearGlobalAveragePool output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    input_tensor = ctx.model_ir.tensors[x_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )
    channels_last = bool(int(node.attrs.get("channels_last", 0)))
    if len(input_shape) < 3 and input_shape != [1]:
        raise NotImplementedError(
            f"QLinearGlobalAveragePool requires rank>=3. op={node.name} input_shape={input_shape}"
        )

    spatial_axes = (
        [int(v) for v in range(1, len(input_shape) - 1)]
        if channels_last
        else [int(v) for v in range(2, len(input_shape))]
    )
    if len(spatial_axes) == 0:
        raise NotImplementedError(
            f"QLinearGlobalAveragePool requires at least one spatial axis. op={node.name} input_shape={input_shape}"
        )

    output_shape = [int(v) for v in input_shape]
    output_signature = list(input_signature)
    for axis in spatial_axes:
        output_shape[axis] = 1
        if axis < len(output_signature):
            output_signature[axis] = 1
    ctx.model_ir.tensors[output_name].shape = list(output_shape)
    ctx.model_ir.tensors[output_name].shape_signature = list(output_signature)

    input_rank = len(input_shape)
    output_rank = len(output_shape)
    input_channel_axis = (input_rank - 1) if channels_last else 1
    output_channel_axis = (output_rank - 1) if channels_last else 1
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(input_channel_axis, input_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(output_channel_axis, output_rank),
    )

    def _lower_via_dequantize_mean_quantize() -> None:
        dq_out = ctx.add_intermediate_tensor(
            f"{node.name}_dq_out",
            dtype="FLOAT32",
            shape=input_shape,
        )
        ctx.model_ir.tensors[dq_out].shape_signature = list(input_signature)
        ctx.add_operator(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[x_name],
                outputs=[dq_out],
            )
        )

        axes_const = ctx.add_const_tensor(
            f"{node.name}_mean_axes",
            np.asarray(spatial_axes, dtype=np.int32),
        )
        mean_out = ctx.add_intermediate_tensor(
            f"{node.name}_mean_out",
            dtype="FLOAT32",
            shape=output_shape,
        )
        ctx.model_ir.tensors[mean_out].shape_signature = list(output_signature)
        ctx.add_operator(
            OperatorIR(
                op_type="MEAN",
                inputs=[dq_out, axes_const],
                outputs=[mean_out],
                options={"keepDims": True},
            )
        )

        ctx.add_operator(
            OperatorIR(
                op_type="QUANTIZE",
                inputs=[mean_out],
                outputs=[output_name],
            )
        )

    # Use MEAN-based lowering unconditionally for stability across delegates/runtimes.
    _lower_via_dequantize_mean_quantize()


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


def build_qlinear_sigmoid_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, x_name, output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearSigmoid input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearSigmoid input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearSigmoid output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearSigmoid output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_rank = len(ctx.get_tensor_shape(x_name))
    output_rank = len(ctx.get_tensor_shape(output_name))
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(1, input_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, output_rank),
    )

    dq_out = ctx.add_intermediate_tensor(
        f"{node.name}_dq_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[x_name],
            outputs=[dq_out],
        )
    )

    sig_out = ctx.add_intermediate_tensor(
        f"{node.name}_sigmoid_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOGISTIC",
            inputs=[dq_out],
            outputs=[sig_out],
        )
    )

    ctx.add_operator(
        OperatorIR(
            op_type="QUANTIZE",
            inputs=[sig_out],
            outputs=[output_name],
        )
    )


def build_qlinear_leaky_relu_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, x_name, output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearLeakyRelu input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearLeakyRelu input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearLeakyRelu output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearLeakyRelu output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_rank = len(ctx.get_tensor_shape(x_name))
    output_rank = len(ctx.get_tensor_shape(output_name))
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(1, input_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, output_rank),
    )

    dq_out = ctx.add_intermediate_tensor(
        f"{node.name}_dq_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[x_name],
            outputs=[dq_out],
        )
    )

    alpha = float(node.attrs.get("alpha", 0.01))
    alpha_name = ctx.add_const_tensor(
        f"{node.name}_alpha",
        np.asarray([alpha], dtype=np.float32),
    )

    prelu_out = ctx.add_intermediate_tensor(
        f"{node.name}_prelu_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="PRELU",
            inputs=[dq_out, alpha_name],
            outputs=[prelu_out],
        )
    )

    ctx.add_operator(
        OperatorIR(
            op_type="QUANTIZE",
            inputs=[prelu_out],
            outputs=[output_name],
        )
    )


def build_qlinear_softmax_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    x_scale_name = node.inputs[1].name
    x_zero_name = node.inputs[2].name
    y_scale_name = node.inputs[3].name
    y_zero_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, x_name, output_name)

    x_scale = _require_const(ctx, x_scale_name, "QLinearSoftmax input scale")
    x_zero = _require_const(ctx, x_zero_name, "QLinearSoftmax input zero_point")
    y_scale = _require_const(ctx, y_scale_name, "QLinearSoftmax output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearSoftmax output zero_point")

    _set_tensor_dtype_from_array(ctx, x_name, x_zero)
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, x_name)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    input_rank = len(ctx.get_tensor_shape(x_name))
    output_rank = len(ctx.get_tensor_shape(output_name))
    axis = int(node.attrs.get("axis", 1))
    axis = _normalize_axis(axis, input_rank)
    if axis != input_rank - 1:
        raise NotImplementedError(
            "QLinearSoftmax supports axis=last only in flatbuffer_direct. "
            f"op={node.name} axis={axis} input_rank={input_rank}"
        )

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=x_name,
        scale=x_scale,
        zero_point=x_zero,
        quantized_dimension=0 if np.asarray(x_scale).size <= 1 else _normalize_axis(1, input_rank),
    )
    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, output_rank),
    )

    dq_out = ctx.add_intermediate_tensor(
        f"{node.name}_dq_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=[x_name],
            outputs=[dq_out],
        )
    )

    softmax_out = ctx.add_intermediate_tensor(
        f"{node.name}_softmax_out",
        dtype="FLOAT32",
        shape=ctx.get_tensor_shape(x_name),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SOFTMAX",
            inputs=[dq_out],
            outputs=[softmax_out],
            options={"beta": float(node.attrs.get("beta", 1.0))},
        )
    )

    ctx.add_operator(
        OperatorIR(
            op_type="QUANTIZE",
            inputs=[softmax_out],
            outputs=[output_name],
        )
    )
