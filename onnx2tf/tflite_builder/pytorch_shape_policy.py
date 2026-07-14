from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from onnx2tf.tflite_builder.ir import (
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _perm_cl_to_cf,
    _permute_shape,
)


def _conv_output_spatial_shape(
    *,
    input_spatial: Sequence[int],
    kernel_spatial: Sequence[int],
    stride_spatial: Sequence[int],
    dilation_spatial: Sequence[int],
    padding_mode: str,
    spatial_rank: int,
) -> Optional[List[int]]:
    input_items = [int(value) for value in list(input_spatial)]
    kernel_items = [int(value) for value in list(kernel_spatial)]
    stride_items = [max(1, int(value)) for value in list(stride_spatial)]
    dilation_items = [max(1, int(value)) for value in list(dilation_spatial)]
    if any(
        len(items) != int(spatial_rank)
        for items in (input_items, kernel_items, stride_items, dilation_items)
    ):
        return None
    if any(int(value) <= 0 for value in input_items + kernel_items):
        return None
    padding_key = str(padding_mode).upper()
    output_spatial: List[int] = []
    for input_dim, kernel_dim, stride_dim, dilation_dim in zip(
        input_items,
        kernel_items,
        stride_items,
        dilation_items,
    ):
        effective_kernel = (int(kernel_dim) - 1) * int(dilation_dim) + 1
        if padding_key == "SAME":
            output_dim = int(math.ceil(float(input_dim) / float(stride_dim)))
        elif padding_key == "VALID":
            output_dim = int(
                math.floor(
                    (float(input_dim) - float(effective_kernel))
                    / float(stride_dim)
                )
            ) + 1
        else:
            return None
        if int(output_dim) <= 0:
            return None
        output_spatial.append(int(output_dim))
    return output_spatial


def _conv2d_output_spatial_shape_for_codegen(
    *,
    input_hw: Sequence[int],
    kernel_hw: Sequence[int],
    stride_hw: Sequence[int],
    dilation_hw: Sequence[int],
    padding_mode: str,
) -> Optional[List[int]]:
    return _conv_output_spatial_shape(
        input_spatial=input_hw,
        kernel_spatial=kernel_hw,
        stride_spatial=stride_hw,
        dilation_spatial=dilation_hw,
        padding_mode=padding_mode,
        spatial_rank=2,
    )


def _conv3d_output_spatial_shape_for_codegen(
    *,
    input_dhw: Sequence[int],
    kernel_dhw: Sequence[int],
    stride_dhw: Sequence[int],
    dilation_dhw: Sequence[int],
    padding_mode: str,
) -> Optional[List[int]]:
    return _conv_output_spatial_shape(
        input_spatial=input_dhw,
        kernel_spatial=kernel_dhw,
        stride_spatial=stride_dhw,
        dilation_spatial=dilation_dhw,
        padding_mode=padding_mode,
        spatial_rank=3,
    )


def _conv3d_transpose_output_spatial_shape_for_codegen(
    *,
    input_dhw: Sequence[int],
    kernel_dhw: Sequence[int],
    stride_dhw: Sequence[int],
    dilation_dhw: Sequence[int],
    padding_mode: str,
) -> Optional[List[int]]:
    input_items = [int(value) for value in list(input_dhw)]
    kernel_items = [int(value) for value in list(kernel_dhw)]
    stride_items = [max(1, int(value)) for value in list(stride_dhw)]
    dilation_items = [max(1, int(value)) for value in list(dilation_dhw)]
    if any(
        len(items) != 3
        for items in (input_items, kernel_items, stride_items, dilation_items)
    ):
        return None
    if any(int(value) <= 0 for value in input_items + kernel_items):
        return None
    padding_key = str(padding_mode).upper()
    output_dhw: List[int] = []
    for input_dim, kernel_dim, stride_dim, dilation_dim in zip(
        input_items,
        kernel_items,
        stride_items,
        dilation_items,
    ):
        effective_kernel = (int(kernel_dim) - 1) * int(dilation_dim) + 1
        if padding_key == "SAME":
            output_dim = int(input_dim) * int(stride_dim)
        elif padding_key == "VALID":
            output_dim = (
                (int(input_dim) - 1) * int(stride_dim)
                + int(effective_kernel)
            )
        else:
            return None
        if int(output_dim) <= 0:
            return None
        output_dhw.append(int(output_dim))
    return output_dhw


def _conv2d_same_pad_padding_arg_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_pre_permute: Optional[Sequence[int]] = None,
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
) -> Optional[List[int]]:
    if str((options or {}).get("padding", "SAME")).upper() != "SAME":
        return None
    if input_shape is None or output_shape is None or weight_shape is None:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
        return None
    if (
        is_channel_first_logical_layout(normalized_input_layout)
        and is_channel_first_logical_layout(normalized_output_layout)
        and int(kernel_shape[0]) > 0
        and int(out_shape[1]) != int(kernel_shape[0])
        and int(out_shape[3]) == int(kernel_shape[0])
    ):
        perm_to_cf = _perm_cl_to_cf(4)
        if perm_to_cf is not None:
            permuted_input_shape = _permute_shape(in_shape, perm_to_cf)
            permuted_output_shape = _permute_shape(out_shape, perm_to_cf)
            if (
                permuted_input_shape is not None
                and permuted_output_shape is not None
            ):
                in_shape = [int(v) for v in list(permuted_input_shape)]
                out_shape = [int(v) for v in list(permuted_output_shape)]
    if input_pre_permute is not None:
        perm = [int(v) for v in list(input_pre_permute)]
        if len(perm) != 4:
            return None
        if not is_channel_first_logical_layout(normalized_input_layout):
            in_shape = [int(in_shape[idx]) for idx in perm]
        should_permute_output_shape = normalized_output_layout == "NHWC"
        if (
            not should_permute_output_shape
            and int(kernel_shape[0]) > 0
            and int(out_shape[1]) != int(kernel_shape[0])
            and int(out_shape[-1]) == int(kernel_shape[0])
        ):
            should_permute_output_shape = True
        if should_permute_output_shape:
            out_shape = [int(out_shape[idx]) for idx in perm]
    else:
        if is_channel_last_logical_layout(normalized_input_layout):
            perm = _perm_cl_to_cf(4)
            if perm is not None:
                in_shape = [int(in_shape[idx]) for idx in perm]
        if is_channel_last_logical_layout(normalized_output_layout):
            perm = _perm_cl_to_cf(4)
            if perm is not None:
                out_shape = [int(out_shape[idx]) for idx in perm]
    stride_hw = [
        max(1, int((options or {}).get("strideH", 1))),
        max(1, int((options or {}).get("strideW", 1))),
    ]
    dilation_hw = [
        max(1, int((options or {}).get("dilationHFactor", 1))),
        max(1, int((options or {}).get("dilationWFactor", 1))),
    ]
    input_hw = [int(in_shape[2]), int(in_shape[3])]
    output_hw = [int(out_shape[2]), int(out_shape[3])]
    if output_hw[0] <= 0:
        output_hw[0] = max(
            1,
            int(math.ceil(float(input_hw[0]) / float(stride_hw[0]))),
        )
    if output_hw[1] <= 0:
        output_hw[1] = max(
            1,
            int(math.ceil(float(input_hw[1]) / float(stride_hw[1]))),
        )
    for idx in range(2):
        if int(input_hw[idx]) <= 0 and int(output_hw[idx]) > 0:
            input_hw[idx] = max(
                1,
                int(output_hw[idx]) * int(stride_hw[idx]),
            )
    kernel_hw_candidates: List[List[int]] = []
    expected_in_channels = int(in_shape[1]) if len(in_shape) == 4 else -1
    if expected_in_channels > 0:
        if (
            int(kernel_shape[1]) == expected_in_channels
            or (
                int(kernel_shape[1]) == 1
                and int(kernel_shape[0]) == expected_in_channels
            )
        ):
            kernel_hw_candidates.append([2, 3])
        if (
            int(kernel_shape[3]) == expected_in_channels
            or (
                int(kernel_shape[0]) == 1
                and int(kernel_shape[3]) == expected_in_channels
            )
        ):
            kernel_hw_candidates.append([1, 2])
    if len(kernel_hw_candidates) == 0:
        kernel_hw_candidates.append([2, 3])
        if [1, 2] not in kernel_hw_candidates:
            kernel_hw_candidates.append([1, 2])
    best_pad_totals: Optional[Tuple[int, int]] = None
    effective_kernel: Optional[List[int]] = None
    for kernel_hw_indices in kernel_hw_candidates:
        candidate_kernel = [
            (int(kernel_shape[int(kernel_hw_indices[idx])]) - 1)
            * int(dilation_hw[idx])
            + 1
            for idx in range(2)
        ]
        candidate_pad_h_total = max(
            (int(output_hw[0]) - 1) * int(stride_hw[0])
            + int(candidate_kernel[0])
            - int(input_hw[0]),
            0,
        )
        candidate_pad_w_total = max(
            (int(output_hw[1]) - 1) * int(stride_hw[1])
            + int(candidate_kernel[1])
            - int(input_hw[1]),
            0,
        )
        candidate_pad_totals = (
            int(candidate_pad_h_total),
            int(candidate_pad_w_total),
        )
        if best_pad_totals is None or candidate_pad_totals < best_pad_totals:
            best_pad_totals = candidate_pad_totals
            effective_kernel = candidate_kernel
    if effective_kernel is None or best_pad_totals is None:
        return None
    pad_h_total, pad_w_total = best_pad_totals
    if pad_h_total == 0 and pad_w_total == 0:
        return None
    pad_top = int(pad_h_total // 2)
    pad_bottom = int(pad_h_total - pad_top)
    pad_left = int(pad_w_total // 2)
    pad_right = int(pad_w_total - pad_left)
    return [pad_left, pad_right, pad_top, pad_bottom]


def _fast_precanonicalize_rank4_layout_hint(
    shape: Sequence[int],
    preferred_channel_count: int | None = None,
) -> str | None:
    if len(shape) != 4:
        return None
    shape_values = [int(value) for value in list(shape)]
    if preferred_channel_count is not None:
        if shape_values[1] == int(preferred_channel_count) and shape_values[3] != int(
            preferred_channel_count
        ):
            return "cf"
        if shape_values[3] == int(preferred_channel_count) and shape_values[1] != int(
            preferred_channel_count
        ):
            return "nhwc"
    if shape_values[1] == 1 and shape_values[3] != 1:
        return "cf"
    if shape_values[3] == 1 and shape_values[1] != 1:
        return "nhwc"
    return None


def _normalize_cf_rank4_shape(
    shape: Sequence[int],
    preferred_channel_count: int | None = None,
    out_hw: Tuple[int, int] | None = None,
) -> List[int]:
    shape_values = [int(value) for value in list(shape)]
    if len(shape_values) != 4:
        return shape_values
    n = int(shape_values[0])
    dims = [int(value) for value in shape_values[1:]]
    channel_index: int | None = None
    if preferred_channel_count is not None:
        for index, dim_value in enumerate(dims):
            if int(dim_value) == int(preferred_channel_count):
                channel_index = index
                break
    if channel_index is None and out_hw is not None:
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        non_spatial_indexes = [
            index
            for index, dim_value in enumerate(dims)
            if int(dim_value) not in {out_h, out_w}
        ]
        if len(non_spatial_indexes) == 1:
            channel_index = int(non_spatial_indexes[0])
    if channel_index is None:
        channel_index = len(dims) - 1
    channel_value = int(dims[channel_index])
    remaining_dims = [
        int(dim_value) for index, dim_value in enumerate(dims) if index != channel_index
    ]
    if len(remaining_dims) < 2:
        remaining_dims = remaining_dims + [
            remaining_dims[0] if remaining_dims else channel_value
        ]
    if out_hw is not None:
        out_h, out_w = int(out_hw[0]), int(out_hw[1])
        if out_h in remaining_dims:
            remaining_dims.remove(out_h)
            normalized_h = out_h
        else:
            normalized_h = int(remaining_dims.pop(0))
        if out_w in remaining_dims:
            remaining_dims.remove(out_w)
            normalized_w = out_w
        else:
            normalized_w = int(remaining_dims.pop(0)) if remaining_dims else out_w
    else:
        normalized_h = int(remaining_dims[0])
        normalized_w = int(remaining_dims[1])
    return [n, channel_value, normalized_h, normalized_w]


def _normalize_nhwc_rank4_shape(
    shape: Sequence[int],
    preferred_channel_count: int | None = None,
    out_hw: Tuple[int, int] | None = None,
) -> List[int]:
    shape_values = [int(value) for value in list(shape)]
    if len(shape_values) != 4:
        return shape_values
    if preferred_channel_count is None:
        layout_hint = _fast_precanonicalize_rank4_layout_hint(shape_values)
        if layout_hint == "cf":
            return [
                int(shape_values[0]),
                int(shape_values[2]),
                int(shape_values[3]),
                int(shape_values[1]),
            ]
    normalized_cf_shape = _normalize_cf_rank4_shape(
        shape_values,
        preferred_channel_count=preferred_channel_count,
        out_hw=out_hw,
    )
    if len(normalized_cf_shape) != 4:
        return shape_values
    return [
        int(normalized_cf_shape[0]),
        int(normalized_cf_shape[2]),
        int(normalized_cf_shape[3]),
        int(normalized_cf_shape[1]),
    ]


def _reshape_special_layout_plan(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    input_layout: Optional[str],
    output_layout: Optional[str],
) -> Optional[Dict[str, Any]]:
    if input_shape is None or output_shape is None:
        return None
    src = [int(v) for v in list(input_shape)]
    dst = [int(v) for v in list(output_shape)]
    in_layout = str(input_layout or "").upper()
    out_layout = str(output_layout or "").upper()
    if (
        in_layout == "NCHW"
        and len(src) == 4
        and len(dst) == 3
        and int(src[0]) == 1
        and int(src[2]) == int(dst[0])
        and int(src[3]) == int(dst[1])
        and int(src[1]) == int(dst[2])
    ):
        return {
            "pre_perm": [0, 2, 3, 1],
            "reshape_shape": list(dst),
            "post_perm": None,
        }
    if (
        len(src) == 3
        and len(dst) == 4
        and int(dst[0]) == 1
        and int(src[0]) == int(dst[2])
        and int(src[1]) == int(dst[3])
        and int(src[2]) == int(dst[1])
        and out_layout == "NCHW"
    ):
        return {
            "pre_perm": None,
            "reshape_shape": [1, int(src[0]), int(src[1]), int(src[2])],
            "post_perm": [0, 3, 1, 2],
        }
    if (
        len(src) == 4
        and len(dst) == 4
        and int(src[0]) == int(dst[0])
        and any(int(v) == 1 for v in src[1:])
    ):
        if [int(src[0]), int(src[3]), int(src[1]), int(src[2])] == dst:
            return {
                "pre_perm": [0, 3, 1, 2],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
        if [int(src[0]), int(src[2]), int(src[3]), int(src[1])] == dst:
            return {
                "pre_perm": [0, 2, 3, 1],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
    if (
        in_layout == "NCHW"
        and out_layout == "NCDHW"
        and len(src) == 4
        and len(dst) == 5
        and int(src[0]) == int(dst[0])
        and int(src[1]) == 1
        and int(dst[2]) == 1
        and int(dst[3]) == 1
        and int(src[2]) == int(dst[4])
        and int(src[3]) == int(dst[1])
    ):
        return {
            "pre_perm": [0, 3, 1, 2],
            "reshape_shape": list(dst),
            "post_perm": None,
        }
    if (
        in_layout == "NCHW"
        and len(src) == 4
        and len(dst) >= 5
        and int(src[0]) == int(dst[0])
        and int(src[2]) == int(dst[1])
        and int(src[3]) == int(dst[2])
    ):
        trailing_product = 1
        for dim in dst[3:]:
            trailing_product *= int(dim)
        if int(src[1]) == int(trailing_product):
            return {
                "pre_perm": [0, 2, 3, 1],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
    return None
