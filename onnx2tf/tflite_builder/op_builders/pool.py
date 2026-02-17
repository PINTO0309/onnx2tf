from __future__ import annotations

import copy
import math
from typing import Any, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
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
    ceil_mode: int = 0,
    effective_pads: Optional[List[int]] = None,
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

    pads = (
        [int(v) for v in list(effective_pads)]
        if effective_pads is not None
        else [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    )
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


def _calc_same_pads_2d(
    *,
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
) -> List[int]:
    out_h = int(math.floor((float(input_h) - 1.0) / float(stride_h)) + 1.0)
    out_w = int(math.floor((float(input_w) - 1.0) / float(stride_w)) + 1.0)
    pad_h = max(int((out_h - 1) * stride_h + kernel_h - input_h), 0)
    pad_w = max(int((out_w - 1) * stride_w + kernel_w - input_w), 0)
    pad_top = int(pad_h // 2)
    pad_bottom = int(pad_h - pad_top)
    pad_left = int(pad_w // 2)
    pad_right = int(pad_w - pad_left)
    return [pad_top, pad_left, pad_bottom, pad_right]


def _calc_extra_padding_with_ceil_2d(
    *,
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    pads: List[int],
    dilations: List[int],
    strides: List[int],
) -> List[int]:
    pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in pads]
    dilation_h, dilation_w = int(dilations[0]), int(dilations[1])
    stride_h, stride_w = int(strides[0]), int(strides[1])
    pads_h = int(pad_top + pad_bottom)
    pads_w = int(pad_left + pad_right)

    out_h = int(
        math.ceil(
            (float(input_h + pads_h - dilation_h * (kernel_h - 1) - 1) / float(stride_h)) + 1.0
        )
    )
    out_w = int(
        math.ceil(
            (float(input_w + pads_w - dilation_w * (kernel_w - 1) - 1) / float(stride_w)) + 1.0
        )
    )
    last_stride_h = int((out_h - 1) * stride_h)
    last_stride_w = int((out_w - 1) * stride_w)

    valid_h = bool(last_stride_h < (int(input_h) + int(pad_top)))
    valid_w = bool(last_stride_w < (int(input_w) + int(pad_left)))
    extra_h = (
        int(last_stride_h + (kernel_h - 1) * dilation_h + 1 - (input_h + pads_h))
        if valid_h
        else 0
    )
    extra_w = (
        int(last_stride_w + (kernel_w - 1) * dilation_w + 1 - (input_w + pads_w))
        if valid_w
        else 0
    )
    return [int(max(extra_h, 0)), int(max(extra_w, 0))]


def _resolve_max_pool_padding_and_explicit_pads(
    *,
    node: Any,
    input_shape_nchw: List[int],
    kernel: List[int],
    strides: List[int],
    dilations: List[int],
    ceil_mode: int,
) -> Tuple[str, Optional[List[int]], List[int]]:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    raw_pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(raw_pads) < 4:
        raw_pads = [0, 0, 0, 0]
    pads = [int(raw_pads[0]), int(raw_pads[1]), int(raw_pads[2]), int(raw_pads[3])]

    input_h = int(input_shape_nchw[2])
    input_w = int(input_shape_nchw[3])
    kernel_h = int(kernel[0])
    kernel_w = int(kernel[1])
    stride_h = int(strides[0])
    stride_w = int(strides[1])

    if auto_pad in ["SAME", "SAME_UPPER"]:
        return "SAME", None, [0, 0, 0, 0]
    if auto_pad == "VALID":
        return "VALID", None, [0, 0, 0, 0]
    if auto_pad == "SAME_LOWER":
        same_upper_pads = _calc_same_pads_2d(
            input_h=input_h,
            input_w=input_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
        )
        top, left, bottom, right = [int(v) for v in same_upper_pads]
        same_lower_pads = [bottom, right, top, left]
        if any(int(v) != 0 for v in same_lower_pads):
            return "VALID", same_lower_pads, list(same_lower_pads)
        return "VALID", None, list(same_lower_pads)
    if auto_pad != "NOTSET":
        raise NotImplementedError(
            f"MaxPool auto_pad attribute is invalid for flatbuffer_direct. op={node.name} auto_pad={auto_pad}"
        )

    same_upper_pads = _calc_same_pads_2d(
        input_h=input_h,
        input_w=input_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
    )
    if int(ceil_mode) == 0 and any(int(v) != 0 for v in pads) and same_upper_pads == pads:
        return "SAME", None, list(pads)

    explicit_pads = list(pads)
    if int(ceil_mode) == 1:
        extra_h, extra_w = _calc_extra_padding_with_ceil_2d(
            input_h=input_h,
            input_w=input_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            pads=explicit_pads,
            dilations=dilations,
            strides=strides,
        )
        explicit_pads[2] = int(explicit_pads[2]) + int(extra_h)
        explicit_pads[3] = int(explicit_pads[3]) + int(extra_w)

    if any(int(v) != 0 for v in explicit_pads):
        return "VALID", explicit_pads, list(explicit_pads)
    return "VALID", None, list(explicit_pads)


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
    raise NotImplementedError(f"Unsupported TFLite dtype in pool builder: {tflite_dtype}")


def _max_pool_pad_value_for_tensor(tensor_dtype: str, tensor_quant: Optional[Any]) -> Any:
    if isinstance(tensor_quant, QuantParamIR) or isinstance(tensor_quant, dict):
        return 0
    np_dtype = _numpy_dtype_from_tflite_dtype(tensor_dtype)
    if np.issubdtype(np_dtype, np.floating):
        return np.finfo(np_dtype).min
    return np.iinfo(np_dtype).min


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


def build_pool2d_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    if len(input_shape) != 4:
        raise NotImplementedError(f"Only 2D pooling (rank=4) is supported. op={node.name}")

    kernel = [int(v) for v in list(node.attrs.get("kernel_shape", [1, 1]))]
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    if len(kernel) != 2 or len(strides) != 2:
        raise NotImplementedError(
            f"Only 2D pooling is supported in flatbuffer_direct. op={node.name} kernel={kernel} strides={strides}"
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if dilations != [1, 1]:
        raise NotImplementedError(
            f"{node.op} dilations must be [1,1] for flatbuffer_direct. op={node.name} dilations={dilations}"
        )
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    explicit_pads: Optional[List[int]] = None
    effective_pads: List[int] = [0, 0, 0, 0]
    if op_type == "MAX_POOL_2D":
        if ceil_mode not in [0, 1]:
            raise NotImplementedError(
                f"MaxPool ceil_mode must be 0 or 1 for flatbuffer_direct. op={node.name} ceil_mode={ceil_mode}"
            )
        padding, explicit_pads, effective_pads = _resolve_max_pool_padding_and_explicit_pads(
            node=node,
            input_shape_nchw=[int(v) for v in input_shape],
            kernel=kernel,
            strides=strides,
            dilations=dilations,
            ceil_mode=ceil_mode,
        )
    else:
        if ceil_mode != 0:
            raise NotImplementedError(
                f"ceil_mode is not supported for {node.op} in flatbuffer_direct. op={node.name}"
            )
        padding = resolve_padding(node)

    if len(output_shape) != 4:
        out_h, out_w = _infer_pool_output_hw(
            node=node,
            input_h=int(input_shape[2]),
            input_w=int(input_shape[3]),
            kernel_h=int(kernel[0]),
            kernel_w=int(kernel[1]),
            stride_h=int(strides[0]),
            stride_w=int(strides[1]),
            ceil_mode=ceil_mode,
            effective_pads=effective_pads,
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

    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_dtype = str(input_tensor.dtype).upper()
    if input_dtype in {"INT8", "UINT8"} and str(output_tensor.dtype).upper() != input_dtype:
        # TFLite quantized pooling requires input/output tensor types to be identical.
        output_tensor.dtype = input_dtype
    if output_tensor.quantization is None and input_tensor.quantization is not None:
        output_tensor.quantization = _clone_quantization(input_tensor.quantization)

    nhwc_input_shape = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
    nhwc_output_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
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
    x_quant = input_tensor.quantization
    if x_quant is not None:
        ctx.model_ir.tensors[x_nhwc].quantization = _clone_quantization(x_quant)
    x_nhwc = make_transpose(
        ctx,
        input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )
    x_nhwc_pool = x_nhwc
    if op_type == "MAX_POOL_2D" and explicit_pads is not None:
        pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in explicit_pads]
        if any(int(v) != 0 for v in [pad_top, pad_left, pad_bottom, pad_right]):
            x_tensor = ctx.model_ir.tensors[x_nhwc_pool]
            padded_shape = list(x_tensor.shape)
            padded_shape[1] = int(padded_shape[1]) + int(pad_top) + int(pad_bottom)
            padded_shape[2] = int(padded_shape[2]) + int(pad_left) + int(pad_right)
            x_nhwc_padded = ctx.add_intermediate_tensor(
                f"{node.name}_input_nhwc_padded",
                dtype=ctx.get_tensor_dtype(x_nhwc_pool),
                shape=padded_shape,
            )
            x_nhwc_padded_tensor = ctx.model_ir.tensors[x_nhwc_padded]
            x_nhwc_padded_tensor.quantization = _clone_quantization(x_tensor.quantization)

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
            # Use PAD for quantized tensors so the runtime pads with the
            # tensor quantization zero-point (real-value 0) consistently.
            if x_tensor.quantization is not None:
                ctx.add_operator(
                    OperatorIR(
                        op_type="PAD",
                        inputs=[x_nhwc_pool, pads_name],
                        outputs=[x_nhwc_padded],
                    )
                )
            else:
                pad_value = _max_pool_pad_value_for_tensor(
                    tensor_dtype=ctx.get_tensor_dtype(x_nhwc_pool),
                    tensor_quant=x_tensor.quantization,
                )
                pad_value_name = ctx.add_const_tensor(
                    f"{node.name}_pad_value",
                    np.asarray(
                        [pad_value],
                        dtype=_numpy_dtype_from_tflite_dtype(ctx.get_tensor_dtype(x_nhwc_pool)),
                    ),
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="PADV2",
                        inputs=[x_nhwc_pool, pads_name, pad_value_name],
                        outputs=[x_nhwc_padded],
                    )
                )
            x_nhwc_pool = x_nhwc_padded

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in nhwc_output_signature]
    y_quant = output_tensor.quantization
    if y_quant is not None:
        ctx.model_ir.tensors[y_nhwc].quantization = _clone_quantization(y_quant)
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[x_nhwc_pool],
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
