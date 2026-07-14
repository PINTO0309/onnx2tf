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


def _infer_conv2d_layout_candidate_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    depthwise: bool,
) -> Optional[Tuple[List[int], int, int]]:
    if input_shape is None or output_shape is None or weight_shape is None:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
        return None
    if (
        in_shape[0] > 0
        and out_shape[0] > 0
        and int(in_shape[0]) != int(out_shape[0])
    ):
        return None
    out_channels = int(kernel_shape[0])
    if (
        out_shape[1] > 0
        and out_channels > 0
        and int(out_shape[1]) != int(out_channels)
    ):
        return None
    stride_hw = [
        int((options or {}).get("strideH", 1)),
        int((options or {}).get("strideW", 1)),
    ]
    dilation_hw = [
        int((options or {}).get("dilationHFactor", 1)),
        int((options or {}).get("dilationWFactor", 1)),
    ]
    padding_mode = str((options or {}).get("padding", "SAME"))
    import itertools

    for tail_perm in itertools.permutations((1, 2, 3)):
        perm = [0, *[int(v) for v in tail_perm]]
        permuted_shape = [int(in_shape[idx]) for idx in perm]
        in_channels = int(permuted_shape[1])
        if in_channels <= 0:
            continue
        if depthwise:
            if (
                int(kernel_shape[1]) != 1
                or int(out_channels) % int(in_channels) != 0
            ):
                continue
            groups = int(in_channels)
        else:
            weight_in_channels = int(kernel_shape[1])
            if (
                weight_in_channels <= 0
                or int(in_channels) % int(weight_in_channels) != 0
            ):
                continue
            groups = int(in_channels) // int(weight_in_channels)
            if groups <= 0 or int(out_channels) % int(groups) != 0:
                continue
        expected_output_hw = _conv2d_output_spatial_shape_for_codegen(
            input_hw=[int(permuted_shape[2]), int(permuted_shape[3])],
            kernel_hw=[int(kernel_shape[2]), int(kernel_shape[3])],
            stride_hw=stride_hw,
            dilation_hw=dilation_hw,
            padding_mode=padding_mode,
        )
        if expected_output_hw is None:
            continue
        if (
            out_shape[2] > 0
            and out_shape[3] > 0
            and expected_output_hw != [int(out_shape[2]), int(out_shape[3])]
        ):
            continue
        return (perm, int(in_channels), int(groups))
    return None


def _conv2d_input_pre_permute_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
    depthwise: bool = False,
) -> Optional[List[int]]:
    if input_shape is None or output_shape is None or weight_shape is None:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
        return None
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    if is_channel_last_logical_layout(normalized_input_layout):
        return [0, 3, 1, 2]
    if depthwise:
        depthwise_channels = int(kernel_shape[0])
        if (
            depthwise_channels > 0
            and int(in_shape[1]) == depthwise_channels
            and int(out_shape[1]) == depthwise_channels
        ):
            return None
        if (
            depthwise_channels > 0
            and int(in_shape[3]) == depthwise_channels
            and int(out_shape[3]) == depthwise_channels
        ):
            return [0, 3, 1, 2]
    if (
        depthwise
        and int(kernel_shape[1]) == 1
        and int(in_shape[1]) == int(kernel_shape[0])
        and int(out_shape[1]) == int(kernel_shape[0])
    ):
        return None
    candidate_channels: List[int] = []
    for candidate in (int(in_shape[1]), int(in_shape[3])):
        if candidate > 0 and candidate not in candidate_channels:
            candidate_channels.append(candidate)
    expected_in_channels = None
    best_groups = None
    for candidate in candidate_channels:
        if (
            int(kernel_shape[1]) <= 0
            or int(candidate) % int(kernel_shape[1]) != 0
        ):
            continue
        inferred_groups = int(candidate) // int(kernel_shape[1])
        if (
            inferred_groups <= 0
            or int(kernel_shape[0]) % int(inferred_groups) != 0
        ):
            continue
        if best_groups is None or int(inferred_groups) < int(best_groups):
            expected_in_channels = int(candidate)
            best_groups = int(inferred_groups)
    if expected_in_channels is None:
        expected_in_channels = int(kernel_shape[1])
    if is_channel_last_logical_layout(normalized_output_layout):
        if (
            int(in_shape[1]) == int(expected_in_channels)
            and int(in_shape[3]) != int(expected_in_channels)
        ):
            return None
        if (
            int(in_shape[3]) == int(expected_in_channels)
            and int(in_shape[1]) != int(expected_in_channels)
        ):
            return [0, 3, 1, 2]
    if (
        int(in_shape[1]) != expected_in_channels
        and int(in_shape[3]) == expected_in_channels
    ):
        return [0, 3, 1, 2]
    if (
        kernel_shape[2] == 1
        and kernel_shape[3] > 1
        and in_shape[2] > 1
        and in_shape[3] == 1
        and out_shape[2] == 1
        and out_shape[3] > 1
    ):
        return [0, 1, 3, 2]

    inferred_layout = _infer_conv2d_layout_candidate_for_codegen(
        input_shape=in_shape,
        output_shape=out_shape,
        weight_shape=kernel_shape,
        options=options,
        depthwise=depthwise,
    )
    if inferred_layout is None:
        if is_channel_first_logical_layout(normalized_input_layout):
            return None
        return None
    perm, _, _ = inferred_layout
    if perm == [0, 1, 2, 3]:
        return None
    return perm


def _infer_conv2d_ctor_params_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
    depthwise: bool,
) -> Tuple[int, int]:
    if input_shape is None or weight_shape is None:
        return (1, 1)
    in_shape = [int(v) for v in list(input_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 4 or len(kernel_shape) != 4:
        return (1, 1)
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    preferred_input_channels: Optional[int] = None
    if (
        output_shape is not None
        and is_channel_first_logical_layout(normalized_input_layout)
        and is_channel_first_logical_layout(normalized_output_layout)
    ):
        out_shape_values = [int(v) for v in list(output_shape)]
        if (
            len(out_shape_values) == 4
            and int(kernel_shape[0]) > 0
            and int(out_shape_values[1]) != int(kernel_shape[0])
            and int(out_shape_values[3]) == int(kernel_shape[0])
        ):
            perm_to_cf = _perm_cl_to_cf(4)
            if perm_to_cf is not None:
                permuted_input_shape = _permute_shape(in_shape, perm_to_cf)
                permuted_output_shape = _permute_shape(
                    out_shape_values,
                    perm_to_cf,
                )
                if (
                    permuted_input_shape is not None
                    and permuted_output_shape is not None
                ):
                    in_shape = [int(v) for v in list(permuted_input_shape)]
                    output_shape = [int(v) for v in list(permuted_output_shape)]

    def _choose_unknown_input_channel_candidate() -> Optional[int]:
        candidate_channels: List[int] = []
        for candidate in (int(in_shape[1]), int(in_shape[3])):
            if candidate > 0 and candidate not in candidate_channels:
                candidate_channels.append(candidate)
        if len(candidate_channels) == 0:
            return None
        out_channels = max(1, int(kernel_shape[0]))
        if depthwise:
            valid_candidates = [
                int(candidate)
                for candidate in candidate_channels
                if int(candidate) > 0
                and int(out_channels) % int(candidate) == 0
            ]
            if len(valid_candidates) > 0:
                return int(max(valid_candidates))
            return None
        weight_in_channels = max(1, int(kernel_shape[1]))
        best_choice: Optional[Tuple[int, int]] = None
        for candidate in candidate_channels:
            if int(candidate) % int(weight_in_channels) != 0:
                continue
            groups = int(candidate) // int(weight_in_channels)
            if groups <= 0 or int(out_channels) % int(groups) != 0:
                continue
            choice = (int(candidate), int(groups))
            if best_choice is None or int(choice[1]) < int(best_choice[1]):
                best_choice = choice
        if best_choice is not None:
            return int(best_choice[0])
        return None

    if is_channel_first_logical_layout(normalized_input_layout):
        preferred_input_channels = int(in_shape[1])
    elif is_channel_last_logical_layout(normalized_input_layout):
        preferred_input_channels = int(in_shape[3])
    else:
        preferred_input_channels = _choose_unknown_input_channel_candidate()
        if preferred_input_channels is None and output_shape is not None:
            out_shape = [int(v) for v in list(output_shape)]
            if len(out_shape) == 4:
                if is_channel_first_logical_layout(normalized_output_layout):
                    preferred_input_channels = int(in_shape[1])
                elif is_channel_last_logical_layout(normalized_output_layout):
                    preferred_input_channels = int(in_shape[3])
    if (
        preferred_input_channels is not None
        and int(preferred_input_channels) > 0
    ):
        if depthwise:
            out_channels = max(1, int(kernel_shape[0]))
            if int(out_channels) % int(preferred_input_channels) == 0:
                return (
                    int(preferred_input_channels),
                    int(preferred_input_channels),
                )
        weight_in_channels = max(1, int(kernel_shape[1]))
        if int(preferred_input_channels) % int(weight_in_channels) == 0:
            preferred_groups = int(preferred_input_channels) // int(
                weight_in_channels
            )
            out_channels = max(1, int(kernel_shape[0]))
            if (
                preferred_groups > 0
                and int(out_channels) % int(preferred_groups) == 0
            ):
                return (int(preferred_input_channels), int(preferred_groups))
    inferred_layout = _infer_conv2d_layout_candidate_for_codegen(
        input_shape=in_shape,
        output_shape=output_shape,
        weight_shape=kernel_shape,
        options=options,
        depthwise=depthwise,
    )
    if inferred_layout is not None:
        _, inferred_in_channels, inferred_groups = inferred_layout
        return (
            max(1, int(inferred_in_channels)),
            max(1, int(inferred_groups)),
        )
    candidate_channels: List[int] = []
    for candidate in (int(in_shape[1]), int(in_shape[3])):
        if candidate > 0 and candidate not in candidate_channels:
            candidate_channels.append(candidate)
    out_channels = max(1, int(kernel_shape[0]))
    if depthwise:
        valid_candidates = [
            int(candidate)
            for candidate in candidate_channels
            if int(candidate) > 0 and int(out_channels) % int(candidate) == 0
        ]
        if len(valid_candidates) == 0:
            inferred_in_channels = (
                int(candidate_channels[-1])
                if len(candidate_channels) > 0
                else 1
            )
        else:
            inferred_in_channels = int(max(valid_candidates))
        return (
            max(1, inferred_in_channels),
            max(1, inferred_in_channels),
        )

    weight_in_channels = max(1, int(kernel_shape[1]))
    best_choice: Optional[Tuple[int, int]] = None
    for candidate in candidate_channels:
        if int(candidate) % int(weight_in_channels) != 0:
            continue
        groups = int(candidate) // int(weight_in_channels)
        if groups <= 0 or int(out_channels) % int(groups) != 0:
            continue
        choice = (int(candidate), int(groups))
        if best_choice is None or int(choice[1]) < int(best_choice[1]):
            best_choice = choice
    if best_choice is not None:
        return best_choice
    fallback_in_channels = (
        int(candidate_channels[-1])
        if len(candidate_channels) > 0
        else int(weight_in_channels)
    )
    return (max(1, fallback_in_channels), 1)


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


def _infer_conv3d_ctor_params_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
) -> Tuple[int, int, int, List[int]]:
    if input_shape is None or output_shape is None or weight_shape is None:
        return (1, 1, 1, [1, 1, 1])
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 5 or len(out_shape) != 5 or len(kernel_shape) != 5:
        return (
            1,
            1,
            max(1, int(kernel_shape[0]) if len(kernel_shape) > 0 else 1),
            [1, 1, 1],
        )
    if (
        in_shape[0] > 0
        and out_shape[0] > 0
        and int(in_shape[0]) != int(out_shape[0])
    ):
        return (
            max(1, int(in_shape[1])),
            1,
            max(1, int(out_shape[1])),
            [int(v) for v in list(kernel_shape[2:5])],
        )
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    input_channels = max(
        1,
        int(in_shape[-1])
        if is_channel_last_logical_layout(normalized_input_layout)
        else int(in_shape[1]),
    )
    expected_out_channels = max(
        1,
        int(out_shape[-1])
        if is_channel_last_logical_layout(normalized_output_layout)
        else int(out_shape[1]),
    )
    input_dhw = (
        [int(in_shape[1]), int(in_shape[2]), int(in_shape[3])]
        if is_channel_last_logical_layout(normalized_input_layout)
        else [int(in_shape[2]), int(in_shape[3]), int(in_shape[4])]
    )
    output_dhw = (
        [int(out_shape[1]), int(out_shape[2]), int(out_shape[3])]
        if is_channel_last_logical_layout(normalized_output_layout)
        else [int(out_shape[2]), int(out_shape[3]), int(out_shape[4])]
    )
    stride_dhw = [
        int((options or {}).get("strideD", 1)),
        int((options or {}).get("strideH", 1)),
        int((options or {}).get("strideW", 1)),
    ]
    dilation_dhw = [
        int((options or {}).get("dilationDFactor", 1)),
        int((options or {}).get("dilationHFactor", 1)),
        int((options or {}).get("dilationWFactor", 1)),
    ]
    padding_mode = str((options or {}).get("padding", "SAME"))

    import itertools

    best_choice: Optional[Tuple[int, int, int, List[int]]] = None
    for out_axis in range(5):
        out_channels = int(kernel_shape[out_axis])
        if out_channels <= 0 or out_channels != expected_out_channels:
            continue
        for in_axis in range(5):
            if in_axis == out_axis:
                continue
            weight_in_channels = int(kernel_shape[in_axis])
            if (
                weight_in_channels <= 0
                or int(input_channels) % int(weight_in_channels) != 0
            ):
                continue
            groups = int(input_channels) // int(weight_in_channels)
            if groups <= 0 or int(out_channels) % int(groups) != 0:
                continue
            kernel_axes = [
                idx for idx in range(5) if idx not in {out_axis, in_axis}
            ]
            if len(kernel_axes) != 3:
                continue
            for kernel_order in itertools.permutations(kernel_axes):
                kernel_dhw = [int(kernel_shape[idx]) for idx in kernel_order]
                expected_output_dhw = _conv3d_output_spatial_shape_for_codegen(
                    input_dhw=input_dhw,
                    kernel_dhw=kernel_dhw,
                    stride_dhw=stride_dhw,
                    dilation_dhw=dilation_dhw,
                    padding_mode=padding_mode,
                )
                if expected_output_dhw is None:
                    continue
                if expected_output_dhw != output_dhw:
                    continue
                choice = (
                    int(input_channels),
                    int(groups),
                    int(out_channels),
                    [int(v) for v in kernel_dhw],
                )
                if best_choice is None or int(choice[1]) < int(best_choice[1]):
                    best_choice = choice
    if best_choice is not None:
        return best_choice
    fallback_kernel = [int(v) for v in list(kernel_shape[2:5])]
    return (
        max(1, int(input_channels)),
        1,
        max(1, int(expected_out_channels)),
        fallback_kernel,
    )


def _infer_conv3d_transpose_ctor_params_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    weight_shape: Optional[Sequence[int]],
    options: Optional[Dict[str, Any]],
    input_logical_layout: Optional[str] = None,
    output_logical_layout: Optional[str] = None,
) -> Tuple[int, int, List[int], int]:
    if input_shape is None or output_shape is None or weight_shape is None:
        return (1, 1, [1, 1, 1], 1)
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    kernel_shape = [int(v) for v in list(weight_shape)]
    if len(in_shape) != 5 or len(out_shape) != 5 or len(kernel_shape) != 5:
        return (
            1,
            max(1, int(out_shape[1]) if len(out_shape) > 1 else 1),
            [1, 1, 1],
            1,
        )
    normalized_input_layout = normalize_logical_layout(input_logical_layout)
    normalized_output_layout = normalize_logical_layout(output_logical_layout)
    input_channels = max(
        1,
        int(in_shape[-1])
        if is_channel_last_logical_layout(normalized_input_layout)
        else int(in_shape[1]),
    )
    expected_out_channels = max(
        1,
        int(out_shape[-1])
        if is_channel_last_logical_layout(normalized_output_layout)
        else int(out_shape[1]),
    )
    input_dhw = (
        [int(in_shape[1]), int(in_shape[2]), int(in_shape[3])]
        if is_channel_last_logical_layout(normalized_input_layout)
        else [int(in_shape[2]), int(in_shape[3]), int(in_shape[4])]
    )
    output_dhw = (
        [int(out_shape[1]), int(out_shape[2]), int(out_shape[3])]
        if is_channel_last_logical_layout(normalized_output_layout)
        else [int(out_shape[2]), int(out_shape[3]), int(out_shape[4])]
    )
    stride_dhw = [
        int((options or {}).get("strideD", 1)),
        int((options or {}).get("strideH", 1)),
        int((options or {}).get("strideW", 1)),
    ]
    dilation_dhw = [
        int((options or {}).get("dilationDFactor", 1)),
        int((options or {}).get("dilationHFactor", 1)),
        int((options or {}).get("dilationWFactor", 1)),
    ]
    padding_mode = str((options or {}).get("padding", "SAME"))

    import itertools

    best_choice: Optional[Tuple[int, int, List[int], int]] = None
    for in_axis in range(5):
        weight_in_channels = int(kernel_shape[in_axis])
        if weight_in_channels <= 0 or weight_in_channels != input_channels:
            continue
        for out_axis in range(5):
            if out_axis == in_axis:
                continue
            weight_out_per_group = int(kernel_shape[out_axis])
            if (
                weight_out_per_group <= 0
                or int(expected_out_channels) % int(weight_out_per_group) != 0
            ):
                continue
            groups = int(expected_out_channels) // int(weight_out_per_group)
            if groups <= 0 or int(input_channels) % int(groups) != 0:
                continue
            kernel_axes = [
                idx for idx in range(5) if idx not in {in_axis, out_axis}
            ]
            if len(kernel_axes) != 3:
                continue
            for kernel_order in itertools.permutations(kernel_axes):
                kernel_dhw = [int(kernel_shape[idx]) for idx in kernel_order]
                expected_output_dhw = (
                    _conv3d_transpose_output_spatial_shape_for_codegen(
                        input_dhw=input_dhw,
                        kernel_dhw=kernel_dhw,
                        stride_dhw=stride_dhw,
                        dilation_dhw=dilation_dhw,
                        padding_mode=padding_mode,
                    )
                )
                if expected_output_dhw is None:
                    continue
                if expected_output_dhw != output_dhw:
                    continue
                choice = (
                    int(input_channels),
                    int(expected_out_channels),
                    [int(v) for v in kernel_dhw],
                    int(groups),
                )
                if best_choice is None or int(choice[3]) < int(best_choice[3]):
                    best_choice = choice
    if best_choice is not None:
        return best_choice
    fallback_kernel = [int(v) for v in list(kernel_shape[2:5])]
    return (
        max(1, int(input_channels)),
        max(1, int(expected_out_channels)),
        fallback_kernel,
        1,
    )


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


def _reshape_preserves_channel_last_sequence_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    input_layout: Optional[str],
) -> Optional[List[int]]:
    if input_shape is None or output_shape is None:
        return None
    src = [int(v) for v in list(input_shape)]
    dst = [int(v) for v in list(output_shape)]
    layout = str(input_layout or "").upper()
    if layout == "NCHW" and len(src) == 4 and len(dst) == 3:
        flattened_spatial = int(src[2]) * int(src[3])
        sequence_extent_matches = (
            dst[1] == -1
            or (
                dst[2] > 0
                and flattened_spatial
                * max(1, int(src[1]) // int(dst[2]))
                == dst[1]
            )
            or flattened_spatial == dst[1]
        )
        if (
            src[0] == dst[0]
            and dst[2] > 0
            and int(src[1]) % int(dst[2]) == 0
            and src[2] > 0
            and src[3] > 0
            and sequence_extent_matches
        ):
            return [0, 2, 3, 1]
    if layout == "NCDHW" and len(src) == 5 and len(dst) == 3:
        spatial = src[2] * src[3] * src[4]
        sequence_extent_matches = (
            dst[1] == -1
            or (
                dst[2] > 0
                and spatial * max(1, int(src[1]) // int(dst[2])) == dst[1]
            )
            or spatial == dst[1]
        )
        if (
            src[0] == dst[0]
            and dst[2] > 0
            and int(src[1]) % int(dst[2]) == 0
            and sequence_extent_matches
        ):
            return [0, 2, 3, 4, 1]
    if layout == "NCW" and len(src) == 3 and len(dst) == 3:
        sequence_extent_matches = (
            dst[1] == -1
            or (
                dst[2] > 0
                and int(src[2])
                * max(1, int(src[1]) // int(dst[2]))
                == dst[1]
            )
            or src[2] == dst[1]
        )
        if (
            src[0] == dst[0]
            and dst[2] > 0
            and int(src[1]) % int(dst[2]) == 0
            and sequence_extent_matches
        ):
            return [0, 2, 1]
    return None


def _matmul_broadcast_shape_for_codegen(
    *,
    lhs_batch: Sequence[int],
    rhs_batch: Sequence[int],
) -> Optional[List[int]]:
    lhs_items = [int(v) for v in list(lhs_batch)]
    rhs_items = [int(v) for v in list(rhs_batch)]
    result: List[int] = []
    for lhs_dim, rhs_dim in zip(reversed(lhs_items), reversed(rhs_items)):
        if int(lhs_dim) == int(rhs_dim):
            result.append(int(lhs_dim))
        elif int(lhs_dim) == 1:
            result.append(int(rhs_dim))
        elif int(rhs_dim) == 1:
            result.append(int(lhs_dim))
        else:
            return None
    if len(lhs_items) > len(rhs_items):
        result.extend(reversed(lhs_items[: len(lhs_items) - len(rhs_items)]))
    elif len(rhs_items) > len(lhs_items):
        result.extend(reversed(rhs_items[: len(rhs_items) - len(lhs_items)]))
    return list(reversed(result))


def _infer_batch_matmul_shape_for_codegen(
    *,
    lhs_shape: Optional[Sequence[int]],
    rhs_shape: Optional[Sequence[int]],
    adj_x: bool,
    adj_y: bool,
) -> Optional[List[int]]:
    if lhs_shape is None or rhs_shape is None:
        return None
    lhs_items = [int(v) for v in list(lhs_shape)]
    rhs_items = [int(v) for v in list(rhs_shape)]
    if len(lhs_items) == 0 or len(rhs_items) == 0:
        return None
    if len(lhs_items) == 1:
        lhs_items = [1, int(lhs_items[0])]
    if len(rhs_items) == 1:
        rhs_items = [int(rhs_items[0]), 1]
    if len(lhs_items) < 2 or len(rhs_items) < 2:
        return None
    lhs_m = int(lhs_items[-1 if adj_x else -2])
    lhs_k = int(lhs_items[-2 if adj_x else -1])
    rhs_k = int(rhs_items[-1 if adj_y else -2])
    rhs_n = int(rhs_items[-2 if adj_y else -1])
    if int(lhs_k) != int(rhs_k):
        return None
    batch_shape = _matmul_broadcast_shape_for_codegen(
        lhs_batch=lhs_items[:-2],
        rhs_batch=rhs_items[:-2],
    )
    if batch_shape is None:
        return None
    return list(batch_shape) + [int(lhs_m), int(rhs_n)]


def _infer_reduction_shape_for_codegen(
    *,
    input_shape: Optional[Sequence[int]],
    axes: Optional[Sequence[int]],
    keepdims: bool,
) -> Optional[List[int]]:
    if input_shape is None:
        return None
    dims = [int(v) for v in list(input_shape)]
    if axes is None:
        return [1 for _ in dims] if keepdims else []
    normalized_axes = sorted({int(v) for v in list(axes)})
    if keepdims:
        return [
            1 if idx in normalized_axes else int(dim)
            for idx, dim in enumerate(dims)
        ]
    return [
        int(dim)
        for idx, dim in enumerate(dims)
        if idx not in normalized_axes
    ]


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
