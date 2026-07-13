from __future__ import annotations

from typing import List, Sequence, Tuple


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
