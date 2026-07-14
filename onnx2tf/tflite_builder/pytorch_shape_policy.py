from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


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
