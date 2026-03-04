from __future__ import annotations

import copy
import math
from typing import Any, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


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


def _resolve_avg_pool_padding_and_explicit_pads(
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
        effective = _calc_same_pads_2d(
            input_h=input_h,
            input_w=input_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
        )
        return "SAME", None, effective
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
            return "VALID", list(same_lower_pads), list(same_lower_pads)
        return "VALID", None, list(same_lower_pads)
    if auto_pad == "VALID":
        return "VALID", None, [0, 0, 0, 0]
    if auto_pad != "NOTSET":
        raise NotImplementedError(
            f"AveragePool auto_pad attribute is invalid for flatbuffer_direct. op={node.name} auto_pad={auto_pad}"
        )
    if int(ceil_mode) == 1:
        explicit_pads = list(pads)
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
            return "VALID", list(explicit_pads), list(explicit_pads)
        return "VALID", None, list(explicit_pads)

    same_upper_pads = _calc_same_pads_2d(
        input_h=input_h,
        input_w=input_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
    )
    if pads == same_upper_pads:
        return "SAME", None, list(pads)
    if any(int(v) != 0 for v in pads):
        return "VALID", list(pads), list(pads)
    return "VALID", None, list(pads)


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


class _NameRef:
    def __init__(self, name: str):
        self.name = str(name)


class _PoolNodeProxy:
    def __init__(
        self,
        *,
        name: str,
        op: str,
        attrs: dict[str, Any],
        input_names: List[str],
        output_names: List[str],
    ) -> None:
        self.name = str(name)
        self.op = str(op)
        self.attrs = dict(attrs)
        self.inputs = [_NameRef(name=v) for v in list(input_names)]
        self.outputs = [_NameRef(name=v) for v in list(output_names)]


def _normalize_1d_pads(raw_pads: List[int]) -> List[int]:
    pads = [int(v) for v in list(raw_pads)]
    if len(pads) == 0:
        return [0, 0]
    if len(pads) == 1:
        return [int(pads[0]), int(pads[0])]
    if len(pads) == 2:
        return [int(pads[0]), int(pads[1])]
    if len(pads) == 4:
        # Some exporters may accidentally materialize as NCHW-style pairs.
        # Interpret as [top, left, bottom, right] and collapse to width-side.
        return [int(pads[1]), int(pads[3])]
    raise NotImplementedError(f"1D MaxPool pads must be length 0/1/2/4. pads={pads}")


def _build_maxpool1d_indices(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    indices_output_name: str,
    kernel_1d: int,
    stride_1d: int,
    pad_left: int,
    pad_right: int,
) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(indices_output_name)]
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise NotImplementedError(
            "MaxPool1D indices lowering expects rank-3 input/output. "
            f"op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )
    n_dim, c_dim, l_dim = [int(v) for v in list(input_shape)]
    out_n, out_c, out_l = [int(v) for v in list(output_shape)]
    if any(int(v) <= 0 for v in [n_dim, c_dim, l_dim, out_n, out_c, out_l]):
        raise NotImplementedError(
            "MaxPool1D indices lowering requires static positive NCL shapes. "
            f"op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )
    if int(out_n) != int(n_dim) or int(out_c) != int(c_dim):
        raise NotImplementedError(
            "MaxPool1D indices lowering requires output N/C to match input. "
            f"op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )

    padded_l = int(l_dim) + int(pad_left) + int(pad_right)
    if int(kernel_1d) != int(stride_1d):
        raise NotImplementedError(
            "MaxPool1D indices lowering currently supports non-overlapping windows only "
            "(kernel==stride). "
            f"op={node.name} kernel={kernel_1d} stride={stride_1d}"
        )
    target_window_length = int(out_l) * int(kernel_1d)
    if int(padded_l) < int(target_window_length):
        raise NotImplementedError(
            "MaxPool1D indices lowering requires padded input length >= output_length * kernel. "
            f"op={node.name} padded_length={padded_l} output_length={out_l} kernel={kernel_1d}"
        )

    indices_input_name = input_name
    if int(pad_left) != 0 or int(pad_right) != 0:
        input_tensor = ctx.model_ir.tensors[input_name]
        input_dtype = str(input_tensor.dtype).upper()
        if input_tensor.quantization is not None:
            raise NotImplementedError(
                "MaxPool1D indices lowering with padding currently supports non-quantized tensors only. "
                f"op={node.name} dtype={input_dtype}"
            )
        padded_name = ctx.add_intermediate_tensor(
            f"{node.name}_indices_input_padded",
            dtype=input_dtype,
            shape=[int(n_dim), int(c_dim), int(padded_l)],
        )
        pads_name = ctx.add_const_tensor(
            f"{node.name}_indices_pads_ncl",
            np.asarray(
                [
                    [0, 0],
                    [0, 0],
                    [int(pad_left), int(pad_right)],
                ],
                dtype=np.int32,
            ),
        )
        pad_value = _max_pool_pad_value_for_tensor(
            tensor_dtype=input_dtype,
            tensor_quant=input_tensor.quantization,
        )
        pad_value_name = ctx.add_const_tensor(
            f"{node.name}_indices_pad_value",
            np.asarray(
                [pad_value],
                dtype=_numpy_dtype_from_tflite_dtype(input_dtype),
            ),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="PADV2",
                inputs=[input_name, pads_name, pad_value_name],
                outputs=[padded_name],
            )
        )
        indices_input_name = padded_name
    if int(padded_l) > int(target_window_length):
        # ceil_mode=0 with stride=kernel can legally drop the tail region.
        # Trim trailing elements that do not participate in output windows.
        sliced_name = ctx.add_intermediate_tensor(
            f"{node.name}_indices_input_sliced",
            dtype=str(ctx.get_tensor_dtype(indices_input_name)).upper(),
            shape=[int(n_dim), int(c_dim), int(target_window_length)],
        )
        begin_name = ctx.add_const_tensor(
            f"{node.name}_indices_slice_begin",
            np.asarray([0, 0, 0], dtype=np.int32),
        )
        size_name = ctx.add_const_tensor(
            f"{node.name}_indices_slice_size",
            np.asarray([int(n_dim), int(c_dim), int(target_window_length)], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[indices_input_name, begin_name, size_name],
                outputs=[sliced_name],
            )
        )
        indices_input_name = sliced_name
        padded_l = int(target_window_length)

    windows_name = ctx.add_intermediate_tensor(
        f"{node.name}_indices_windows",
        dtype=str(ctx.get_tensor_dtype(indices_input_name)).upper(),
        shape=[int(n_dim), int(c_dim), int(out_l), int(kernel_1d)],
    )
    windows_shape_name = ctx.add_const_tensor(
        f"{node.name}_indices_windows_shape",
        np.asarray([int(n_dim), int(c_dim), int(out_l), int(kernel_1d)], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[indices_input_name, windows_shape_name],
            outputs=[windows_name],
            options={"newShape": [int(n_dim), int(c_dim), int(out_l), int(kernel_1d)]},
        )
    )

    block_argmax_name = ctx.add_intermediate_tensor(
        f"{node.name}_indices_block_argmax",
        dtype="INT32",
        shape=[int(n_dim), int(c_dim), int(out_l)],
    )
    axis_name = ctx.add_const_tensor(
        f"{node.name}_indices_argmax_axis",
        np.asarray([3], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ARG_MAX",
            inputs=[windows_name, axis_name],
            outputs=[block_argmax_name],
            options={"outputType": "INT32"},
        )
    )

    out_grid_name = ctx.add_const_tensor(
        f"{node.name}_indices_out_grid",
        np.arange(int(out_l), dtype=np.int32).reshape(1, 1, int(out_l)),
    )
    stride_name = ctx.add_const_tensor(
        f"{node.name}_indices_stride",
        np.asarray([int(stride_1d)], dtype=np.int32),
    )
    base_name = ctx.add_intermediate_tensor(
        f"{node.name}_indices_base",
        dtype="INT32",
        shape=[1, 1, int(out_l)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[out_grid_name, stride_name],
            outputs=[base_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    base_shifted_name = base_name
    if int(pad_left) != 0:
        minus_pad_name = ctx.add_const_tensor(
            f"{node.name}_indices_minus_pad_left",
            np.asarray([-int(pad_left)], dtype=np.int32),
        )
        shifted_name = ctx.add_intermediate_tensor(
            f"{node.name}_indices_base_shifted",
            dtype="INT32",
            shape=[1, 1, int(out_l)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[base_name, minus_pad_name],
                outputs=[shifted_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        base_shifted_name = shifted_name

    l_index_name = ctx.add_intermediate_tensor(
        f"{node.name}_indices_l_index",
        dtype="INT32",
        shape=[int(n_dim), int(c_dim), int(out_l)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[base_shifted_name, block_argmax_name],
            outputs=[l_index_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    c_grid_name = ctx.add_const_tensor(
        f"{node.name}_indices_c_grid",
        np.arange(int(c_dim), dtype=np.int32).reshape(1, int(c_dim), 1),
    )
    l_dim_name = ctx.add_const_tensor(
        f"{node.name}_indices_l_dim",
        np.asarray([int(l_dim)], dtype=np.int32),
    )
    c_mul_name = ctx.add_intermediate_tensor(
        f"{node.name}_indices_c_mul",
        dtype="INT32",
        shape=[1, int(c_dim), 1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[c_grid_name, l_dim_name],
            outputs=[c_mul_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    linear_i32_name = ctx.add_intermediate_tensor(
        f"{node.name}_indices_linear_i32",
        dtype="INT32",
        shape=[int(n_dim), int(c_dim), int(out_l)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[c_mul_name, l_index_name],
            outputs=[linear_i32_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    target_dtype = str(ctx.get_tensor_dtype(indices_output_name)).upper()
    if target_dtype == "INT32":
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[
                    linear_i32_name,
                    ctx.add_const_tensor(
                        f"{node.name}_indices_out_shape_i32",
                        np.asarray([int(n_dim), int(c_dim), int(out_l)], dtype=np.int32),
                    ),
                ],
                outputs=[indices_output_name],
                options={"newShape": [int(n_dim), int(c_dim), int(out_l)]},
            )
        )
        return

    if target_dtype not in {"INT64"}:
        ctx.model_ir.tensors[indices_output_name].dtype = "INT64"
        target_dtype = "INT64"
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[linear_i32_name],
            outputs=[indices_output_name],
            options={"inDataType": "INT32", "outDataType": target_dtype},
        )
    )


def _build_maxpool1d_op(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
    indices_output_name: Optional[str],
) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    if len(input_shape) != 3:
        raise NotImplementedError(
            f"MaxPool1D lowering requires rank-3 input. op={node.name} input_shape={input_shape}"
        )

    kernel_raw = [int(v) for v in list(node.attrs.get("kernel_shape", []))]
    strides_raw = [int(v) for v in list(node.attrs.get("strides", []))]
    dilations_raw = [int(v) for v in list(node.attrs.get("dilations", []))]
    if len(kernel_raw) != 1:
        raise NotImplementedError(
            f"MaxPool1D lowering requires 1D kernel_shape. op={node.name} kernel_shape={kernel_raw}"
        )
    if len(strides_raw) == 0:
        strides_raw = [1]
    if len(strides_raw) != 1:
        raise NotImplementedError(
            f"MaxPool1D lowering requires 1D strides. op={node.name} strides={strides_raw}"
        )
    if len(dilations_raw) == 0:
        dilations_raw = [1]
    if len(dilations_raw) != 1 or int(dilations_raw[0]) != 1:
        raise NotImplementedError(
            f"MaxPool1D lowering currently supports dilations=[1]. op={node.name} dilations={dilations_raw}"
        )
    kernel_1d = int(kernel_raw[0])
    stride_1d = int(strides_raw[0])
    if int(kernel_1d) not in [1, 2] or int(stride_1d) != int(kernel_1d):
        raise NotImplementedError(
            "MaxPool1D lowering currently supports kernel/stride [1]/[1] or [2]/[2]. "
            f"op={node.name} kernel={kernel_raw} strides={strides_raw}"
        )

    auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
    if auto_pad not in {"NOTSET", "VALID"}:
        raise NotImplementedError(
            f"MaxPool1D lowering supports auto_pad NOTSET/VALID only. op={node.name} auto_pad={auto_pad}"
        )
    raw_pads = [int(v) for v in list(node.attrs.get("pads", []))]
    pad_left, pad_right = _normalize_1d_pads(raw_pads)
    if auto_pad == "VALID":
        pad_left, pad_right = 0, 0
    if int(kernel_1d) == 1 and (int(pad_left) != 0 or int(pad_right) != 0):
        raise NotImplementedError(
            "MaxPool1D kernel=1/stride=1 with indices currently supports zero pads only. "
            f"op={node.name} pads={[pad_left, pad_right]}"
        )
    if int(pad_left) < 0 or int(pad_right) < 0:
        raise NotImplementedError(
            f"MaxPool1D lowering requires non-negative pads. op={node.name} pads={[pad_left, pad_right]}"
        )

    expanded_input_name = ctx.add_intermediate_tensor(
        f"{node.name}_input_ncl_to_nchw",
        dtype=str(ctx.get_tensor_dtype(input_name)).upper(),
        shape=[int(input_shape[0]), int(input_shape[1]), 1, int(input_shape[2])],
    )
    input_tensor = ctx.model_ir.tensors[input_name]
    expanded_tensor = ctx.model_ir.tensors[expanded_input_name]
    expanded_tensor.quantization = _clone_quantization(input_tensor.quantization)
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_tensor.shape)
    )
    if len(input_signature) == 3:
        expanded_tensor.shape_signature = [
            int(input_signature[0]),
            int(input_signature[1]),
            1,
            int(input_signature[2]),
        ]
    expand_axis_name = ctx.add_const_tensor(
        f"{node.name}_input_expand_axis",
        np.asarray([2], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="EXPAND_DIMS",
            inputs=[input_name, expand_axis_name],
            outputs=[expanded_input_name],
        )
    )

    output_shape_4d = (
        [int(output_shape[0]), int(output_shape[1]), 1, int(output_shape[2])]
        if len(output_shape) == 3
        else [int(output_shape[0]), int(output_shape[1]), int(output_shape[2]), int(output_shape[3])]
    )
    pooled_output_name = ctx.add_intermediate_tensor(
        f"{node.name}_output_nchw_1d",
        dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
        shape=output_shape_4d,
    )
    pooled_output_tensor = ctx.model_ir.tensors[pooled_output_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_tensor.shape)
    )
    if len(output_signature) == 3:
        pooled_output_tensor.shape_signature = [
            int(output_signature[0]),
            int(output_signature[1]),
            1,
            int(output_signature[2]),
        ]

    proxy_attrs = dict(node.attrs)
    proxy_attrs["kernel_shape"] = [1, int(kernel_1d)]
    proxy_attrs["strides"] = [1, int(stride_1d)]
    proxy_attrs["dilations"] = [1, 1]
    if auto_pad in {"NOTSET", "VALID"}:
        proxy_attrs["pads"] = [0, int(pad_left), 0, int(pad_right)]

    proxy_node = _PoolNodeProxy(
        name=f"{node.name}_maxpool1d_proxy",
        op=str(node.op),
        attrs=proxy_attrs,
        input_names=[expanded_input_name],
        output_names=[pooled_output_name],
    )
    build_pool2d_op(proxy_node, ctx, "MAX_POOL_2D")

    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[pooled_output_name],
            outputs=[output_name],
            options={"squeezeDims": [2]},
        )
    )

    if indices_output_name is not None:
        _build_maxpool1d_indices(
            node=node,
            ctx=ctx,
            input_name=input_name,
            indices_output_name=indices_output_name,
            kernel_1d=int(kernel_1d),
            stride_1d=int(stride_1d),
            pad_left=int(pad_left),
            pad_right=int(pad_right),
        )


def build_pool2d_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    indices_output_name = node.outputs[1].name if len(node.outputs) >= 2 else None
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    if indices_output_name is not None:
        ctx.ensure_tensor(indices_output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    if len(input_shape) == 3:
        if str(op_type) != "MAX_POOL_2D":
            raise NotImplementedError(
                f"Only MaxPool rank-3 (1D) is currently supported in flatbuffer_direct. op={node.name}"
            )
        _build_maxpool1d_op(
            node=node,
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
            indices_output_name=indices_output_name,
        )
        return
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
    average_count_include_pad = 0
    average_needs_exclude_pad_correction = False
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
        if ceil_mode not in [0, 1]:
            raise NotImplementedError(
                f"AveragePool ceil_mode must be 0 or 1 for flatbuffer_direct. op={node.name} ceil_mode={ceil_mode}"
            )
        average_count_include_pad = int(node.attrs.get("count_include_pad", 0))
        if average_count_include_pad not in [0, 1]:
            raise NotImplementedError(
                f"AveragePool count_include_pad must be 0 or 1 for flatbuffer_direct. "
                f"op={node.name} count_include_pad={average_count_include_pad}"
            )
        padding, explicit_pads, effective_pads = _resolve_avg_pool_padding_and_explicit_pads(
            node=node,
            input_shape_nchw=[int(v) for v in input_shape],
            kernel=kernel,
            strides=strides,
            dilations=dilations,
            ceil_mode=ceil_mode,
        )
        average_needs_exclude_pad_correction = (
            int(average_count_include_pad) == 0
            and any(int(v) != 0 for v in effective_pads)
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
    output_dtype = str(output_tensor.dtype).upper()
    if average_needs_exclude_pad_correction:
        if input_dtype != output_dtype:
            raise NotImplementedError(
                "AveragePool count_include_pad=0 correction requires identical input/output dtype. "
                f"op={node.name} input_dtype={input_dtype} output_dtype={output_dtype}"
            )
        if output_dtype not in {"FLOAT16", "FLOAT32"}:
            raise NotImplementedError(
                "AveragePool count_include_pad=0 correction supports FLOAT16/FLOAT32 only. "
                f"op={node.name} dtype={output_dtype}"
            )

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
    x_nhwc_prepad = x_nhwc_pool
    if explicit_pads is not None:
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
                if op_type == "MAX_POOL_2D":
                    pad_value = _max_pool_pad_value_for_tensor(
                        tensor_dtype=ctx.get_tensor_dtype(x_nhwc_pool),
                        tensor_quant=x_tensor.quantization,
                    )
                else:
                    pad_value = 0.0
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
    pool_output_name = y_nhwc
    if op_type == "AVERAGE_POOL_2D" and average_needs_exclude_pad_correction:
        pool_output_name = ctx.add_intermediate_tensor(
            f"{node.name}_output_nhwc_include_pad",
            dtype=ctx.get_tensor_dtype(output_name),
            shape=nhwc_output_shape,
        )
        pool_output_tensor = ctx.model_ir.tensors[pool_output_name]
        pool_output_tensor.shape_signature = [int(v) for v in nhwc_output_signature]
        if y_quant is not None:
            pool_output_tensor.quantization = _clone_quantization(y_quant)
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[x_nhwc_pool],
            outputs=[pool_output_name],
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
    if op_type == "AVERAGE_POOL_2D" and average_needs_exclude_pad_correction:
        mask_shape = [int(v) for v in list(ctx.get_tensor_shape(x_nhwc_prepad))]
        if len(mask_shape) != 4 or any(int(v) <= 0 for v in mask_shape):
            raise NotImplementedError(
                "AveragePool count_include_pad=0 correction requires fully known static rank-4 input shape. "
                f"op={node.name} input_shape={mask_shape}"
            )
        mask_name = ctx.add_const_tensor(
            f"{node.name}_exclude_pad_mask",
            np.ones(
                mask_shape,
                dtype=np.float16 if output_dtype == "FLOAT16" else np.float32,
            ),
        )
        mask_input_name = mask_name
        if explicit_pads is not None:
            pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in explicit_pads]
            if any(int(v) != 0 for v in [pad_top, pad_left, pad_bottom, pad_right]):
                mask_padded_name = ctx.add_intermediate_tensor(
                    f"{node.name}_exclude_pad_mask_padded",
                    dtype=output_dtype,
                    shape=list(ctx.get_tensor_shape(x_nhwc_pool)),
                )
                pads_name = ctx.add_const_tensor(
                    f"{node.name}_exclude_pad_mask_pads_nhwc",
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
                pad_zero_name = ctx.add_const_tensor(
                    f"{node.name}_exclude_pad_mask_pad_zero",
                    np.asarray(
                        [0.0],
                        dtype=np.float16 if output_dtype == "FLOAT16" else np.float32,
                    ),
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="PADV2",
                        inputs=[mask_name, pads_name, pad_zero_name],
                        outputs=[mask_padded_name],
                    )
                )
                mask_input_name = mask_padded_name

        mask_pool_name = ctx.add_intermediate_tensor(
            f"{node.name}_exclude_pad_mask_pool",
            dtype=output_dtype,
            shape=nhwc_output_shape,
        )
        ctx.model_ir.tensors[mask_pool_name].shape_signature = [int(v) for v in nhwc_output_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="AVERAGE_POOL_2D",
                inputs=[mask_input_name],
                outputs=[mask_pool_name],
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

        ctx.add_operator(
            OperatorIR(
                op_type="DIV",
                inputs=[pool_output_name, mask_pool_name],
                outputs=[y_nhwc],
                options={"fusedActivationFunction": "NONE"},
            )
        )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )

    if op_type == "MAX_POOL_2D" and indices_output_name is not None:
        if (
            list(kernel) != [2, 2]
            or list(strides) != [2, 2]
            or list(dilations) != [1, 1]
            or int(ceil_mode) != 0
            or padding != "VALID"
            or explicit_pads is not None
        ):
            raise NotImplementedError(
                "MaxPool with indices currently supports only "
                "kernel_shape=[2,2], strides=[2,2], dilations=[1,1], "
                "ceil_mode=0, VALID/no-explicit-padding in flatbuffer_direct. "
                f"op={node.name} kernel={kernel} strides={strides} "
                f"dilations={dilations} ceil_mode={ceil_mode} "
                f"padding={padding} explicit_pads={explicit_pads}"
            )

        n_dim, c_dim, h_dim, w_dim = [int(v) for v in list(input_shape)]
        _, _, out_h, out_w = [int(v) for v in list(output_shape)]
        if any(int(v) <= 0 for v in [n_dim, c_dim, h_dim, w_dim, out_h, out_w]):
            raise NotImplementedError(
                "MaxPool with indices requires fully-known static positive "
                f"NCHW shapes in flatbuffer_direct. op={node.name} "
                f"input_shape={input_shape} output_shape={output_shape}"
            )
        if int(h_dim) != int(out_h) * 2 or int(w_dim) != int(out_w) * 2:
            raise NotImplementedError(
                "MaxPool with indices requires output spatial size exactly "
                "half of input for kernel=2/stride=2. "
                f"op={node.name} input_shape={input_shape} output_shape={output_shape}"
            )

        s2d_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_space_to_depth",
            dtype=ctx.get_tensor_dtype(x_nhwc),
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim) * 4],
        )
        x_nhwc_tensor = ctx.model_ir.tensors[x_nhwc]
        s2d_tensor = ctx.model_ir.tensors[s2d_name]
        s2d_tensor.quantization = _clone_quantization(x_nhwc_tensor.quantization)
        x_nhwc_signature = (
            list(x_nhwc_tensor.shape_signature)
            if x_nhwc_tensor.shape_signature is not None
            else list(x_nhwc_tensor.shape)
        )
        if len(x_nhwc_signature) == 4:
            s2d_signature = [
                int(x_nhwc_signature[0]),
                int(x_nhwc_signature[1]) // 2 if int(x_nhwc_signature[1]) > 0 else int(x_nhwc_signature[1]),
                int(x_nhwc_signature[2]) // 2 if int(x_nhwc_signature[2]) > 0 else int(x_nhwc_signature[2]),
                int(x_nhwc_signature[3]) * 4 if int(x_nhwc_signature[3]) > 0 else int(x_nhwc_signature[3]),
            ]
            s2d_tensor.shape_signature = [int(v) for v in s2d_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="SPACE_TO_DEPTH",
                inputs=[x_nhwc],
                outputs=[s2d_name],
                options={"blockSize": 2},
            )
        )

        s2d_reshaped_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_space_to_depth_reshaped",
            dtype=ctx.get_tensor_dtype(x_nhwc),
            shape=[int(n_dim), int(out_h), int(out_w), 4, int(c_dim)],
        )
        s2d_reshaped_tensor = ctx.model_ir.tensors[s2d_reshaped_name]
        s2d_reshaped_tensor.quantization = _clone_quantization(x_nhwc_tensor.quantization)
        reshape_shape_name = ctx.add_const_tensor(
            f"{node.name}_argmax_s2d_reshape_shape",
            np.asarray([int(n_dim), int(out_h), int(out_w), 4, int(c_dim)], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[s2d_name, reshape_shape_name],
                outputs=[s2d_reshaped_name],
                options={"newShape": [int(n_dim), int(out_h), int(out_w), 4, int(c_dim)]},
            )
        )

        block_argmax_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_block_index",
            dtype="INT32",
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
        )
        axis_name = ctx.add_const_tensor(
            f"{node.name}_argmax_block_axis",
            np.asarray([3], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ARG_MAX",
                inputs=[s2d_reshaped_name, axis_name],
                outputs=[block_argmax_name],
                options={"outputType": "INT32"},
            )
        )

        two_name = ctx.add_const_tensor(
            f"{node.name}_argmax_two",
            np.asarray([2], dtype=np.int32),
        )
        block_h_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_block_h",
            dtype="INT32",
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
        )
        block_w_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_block_w",
            dtype="INT32",
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="DIV",
                inputs=[block_argmax_name, two_name],
                outputs=[block_h_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="FLOOR_MOD",
                inputs=[block_argmax_name, two_name],
                outputs=[block_w_name],
            )
        )

        oh_grid_name = ctx.add_const_tensor(
            f"{node.name}_argmax_oh_grid",
            np.arange(int(out_h), dtype=np.int32).reshape(1, int(out_h), 1, 1),
        )
        ow_grid_name = ctx.add_const_tensor(
            f"{node.name}_argmax_ow_grid",
            np.arange(int(out_w), dtype=np.int32).reshape(1, 1, int(out_w), 1),
        )
        c_grid_name = ctx.add_const_tensor(
            f"{node.name}_argmax_c_grid",
            np.arange(int(c_dim), dtype=np.int32).reshape(1, 1, 1, int(c_dim)),
        )

        oh2_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_oh2",
            dtype="INT32",
            shape=[1, int(out_h), 1, 1],
        )
        ow2_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_ow2",
            dtype="INT32",
            shape=[1, 1, int(out_w), 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[oh_grid_name, two_name],
                outputs=[oh2_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[ow_grid_name, two_name],
                outputs=[ow2_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

        h_index_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_h_index",
            dtype="INT32",
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
        )
        w_index_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_w_index",
            dtype="INT32",
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[oh2_name, block_h_name],
                outputs=[h_index_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[ow2_name, block_w_name],
                outputs=[w_index_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

        hw_name = ctx.add_const_tensor(
            f"{node.name}_argmax_hw",
            np.asarray([int(h_dim) * int(w_dim)], dtype=np.int32),
        )
        w_name = ctx.add_const_tensor(
            f"{node.name}_argmax_w",
            np.asarray([int(w_dim)], dtype=np.int32),
        )
        c_hw_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_c_hw",
            dtype="INT32",
            shape=[1, 1, 1, int(c_dim)],
        )
        h_w_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_h_w",
            dtype="INT32",
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
        )
        linear_tmp_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_linear_tmp",
            dtype="INT32",
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
        )
        linear_nhwc_name = ctx.add_intermediate_tensor(
            f"{node.name}_argmax_linear_nhwc",
            dtype="INT32",
            shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[c_grid_name, hw_name],
                outputs=[c_hw_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[h_index_name, w_name],
                outputs=[h_w_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[c_hw_name, h_w_name],
                outputs=[linear_tmp_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[linear_tmp_name, w_index_name],
                outputs=[linear_nhwc_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

        target_indices_dtype = str(ctx.get_tensor_dtype(indices_output_name)).upper()
        linear_output_name = linear_nhwc_name
        if target_indices_dtype == "INT64":
            linear_i64_name = ctx.add_intermediate_tensor(
                f"{node.name}_argmax_linear_nhwc_i64",
                dtype="INT64",
                shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[linear_nhwc_name],
                    outputs=[linear_i64_name],
                    options={"inDataType": "INT32", "outDataType": "INT64"},
                )
            )
            linear_output_name = linear_i64_name
        elif target_indices_dtype != "INT32":
            # Keep ONNX MaxPool indices contract (int64/int32). Unknown dtypes
            # are normalized to int64 to preserve downstream index arithmetic.
            ctx.model_ir.tensors[indices_output_name].dtype = "INT64"
            linear_i64_name = ctx.add_intermediate_tensor(
                f"{node.name}_argmax_linear_nhwc_i64",
                dtype="INT64",
                shape=[int(n_dim), int(out_h), int(out_w), int(c_dim)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[linear_nhwc_name],
                    outputs=[linear_i64_name],
                    options={"inDataType": "INT32", "outDataType": "INT64"},
                )
            )
            linear_output_name = linear_i64_name

        make_transpose(
            ctx,
            linear_output_name,
            indices_output_name,
            [0, 3, 1, 2],
        )
