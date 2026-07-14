from __future__ import annotations

from typing import List, Optional, Sequence

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    is_channel_first_logical_layout,
    normalize_logical_layout,
)


def _reshape_is_plain_singleton_axis_drop(
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
) -> bool:
    if input_shape is None or output_shape is None:
        return False
    src = [int(v) for v in list(input_shape)]
    dst = [int(v) for v in list(output_shape)]
    if len(src) != len(dst) + 1:
        return False
    singleton_axes = [axis for axis, dim in enumerate(src) if int(dim) == 1]
    if len(singleton_axes) != 1:
        return False
    axis = int(singleton_axes[0])
    return src[:axis] + src[axis + 1 :] == dst


def _direct_slice_expr(
    *,
    x_expr: str,
    begin_values: Sequence[int],
    size_values: Sequence[int],
    input_rank: int,
    input_shape: Optional[Sequence[int]] = None,
) -> Optional[str]:
    if len(begin_values) != int(input_rank) or len(size_values) != int(input_rank):
        return None
    direct_x_expr = f"{x_expr}.reshape(-1)" if int(input_rank) == 1 else x_expr
    parts: List[str] = []
    for axis, (start, length) in enumerate(zip(begin_values, size_values)):
        dim_size: Optional[int] = None
        if input_shape is not None and axis < len(input_shape):
            try:
                dim_size = int(input_shape[axis])
            except Exception:
                dim_size = None
        resolved_start = int(start)
        if int(length) < 0:
            resolved_stop: Optional[int] = None
        else:
            resolved_stop = resolved_start + int(length)
            if dim_size is not None and int(dim_size) > 0:
                resolved_stop = min(int(resolved_stop), int(dim_size))
        if resolved_start == 0 and resolved_stop is None:
            parts.append(":")
        else:
            start_str = "" if resolved_start == 0 else str(resolved_start)
            stop_str = "" if resolved_stop is None else str(int(resolved_stop))
            parts.append(f"{start_str}:{stop_str}")
    return f"{direct_x_expr}[{', '.join(parts)}]"


def _direct_strided_slice_expr(
    *,
    x_expr: str,
    begin_values: Sequence[int],
    end_values: Sequence[int],
    stride_values: Sequence[int],
    begin_mask: int,
    end_mask: int,
    input_rank: int,
) -> Optional[str]:
    if (
        len(begin_values) != int(input_rank)
        or len(end_values) != int(input_rank)
        or len(stride_values) != int(input_rank)
    ):
        return None
    direct_x_expr = f"{x_expr}.reshape(-1)" if int(input_rank) == 1 else x_expr
    parts: List[str] = []
    for axis, (start, stop, step) in enumerate(
        zip(begin_values, end_values, stride_values)
    ):
        resolved_start = None if ((int(begin_mask) >> axis) & 1) else int(start)
        resolved_stop = None if ((int(end_mask) >> axis) & 1) else int(stop)
        if resolved_stop is not None and int(resolved_stop) >= 2147483647:
            resolved_stop = None
        resolved_step = int(step)
        if resolved_step == 0:
            return None
        if resolved_start is None and resolved_stop is None and resolved_step == 1:
            parts.append(":")
            continue
        start_str = "" if resolved_start is None else str(int(resolved_start))
        stop_str = "" if resolved_stop is None else str(int(resolved_stop))
        if resolved_step == 1:
            parts.append(f"{start_str}:{stop_str}")
        else:
            parts.append(f"{start_str}:{stop_str}:{resolved_step}")
    return f"{direct_x_expr}[{', '.join(parts)}]"


def _direct_symbolic_strided_slice_expr(
    *,
    x_expr: str,
    begin_values: Sequence[int],
    stride_values: Sequence[int],
    begin_mask: int,
    end_mask: int,
    input_rank: int,
    end_list_expr: Optional[str] = None,
    end_scalar_expr: Optional[str] = None,
) -> Optional[str]:
    if len(begin_values) != int(input_rank) or len(stride_values) != int(input_rank):
        return None
    if end_list_expr is None and end_scalar_expr is None:
        return None
    direct_x_expr = f"{x_expr}.reshape(-1)" if int(input_rank) == 1 else x_expr
    parts: List[str] = []
    for axis, (start, step) in enumerate(zip(begin_values, stride_values)):
        resolved_start = None if ((int(begin_mask) >> axis) & 1) else int(start)
        if (int(end_mask) >> axis) & 1:
            resolved_stop_expr = None
        elif int(input_rank) == 1 and end_scalar_expr is not None:
            resolved_stop_expr = str(end_scalar_expr)
        elif end_list_expr is not None:
            resolved_stop_expr = f"({end_list_expr})[{int(axis)}]"
        else:
            return None
        resolved_step = int(step)
        if resolved_step == 0:
            return None
        start_str = (
            ""
            if resolved_start is None or int(resolved_start) == 0
            else str(int(resolved_start))
        )
        stop_str = "" if resolved_stop_expr is None else str(resolved_stop_expr)
        if resolved_step == 1:
            parts.append(f"{start_str}:{stop_str}")
        else:
            parts.append(f"{start_str}:{stop_str}:{resolved_step}")
    return f"{direct_x_expr}[{', '.join(parts)}]"


def _direct_gather_expr(
    *,
    params_expr: str,
    indices_values: Sequence[int],
    indices_shape: Optional[Sequence[int]],
    axis: int,
    batch_dims: int,
    input_rank: int,
) -> Optional[str]:
    if int(batch_dims) != 0:
        return None
    if len(indices_values) == 0:
        return None
    normalized_indices_shape = (
        [int(v) for v in list(indices_shape)]
        if indices_shape is not None and len(list(indices_shape)) > 0
        else [int(len(indices_values))]
    )
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis < 0 or resolved_axis >= int(input_rank):
        return None
    if indices_shape is not None and len(list(indices_shape)) == 0:
        literal = int(indices_values[0])
        return (
            f"torch.index_select({params_expr}, {resolved_axis}, "
            f"torch.as_tensor([{literal}], dtype=torch.int64, device={params_expr}.device))"
            f".squeeze({resolved_axis})"
        )
    if len(normalized_indices_shape) > 1:
        literal = repr([int(v) for v in indices_values])
        return (
            f"_reshape_gather_output("
            f"torch.index_select({params_expr}, {resolved_axis}, "
            f"torch.as_tensor({literal}, dtype=torch.int64, device={params_expr}.device)), "
            f"{params_expr}, {repr(normalized_indices_shape)}, axis={resolved_axis})"
        )
    if int(input_rank) == 1 and int(resolved_axis) == 0:
        return f"{params_expr}.reshape(-1)[{repr([int(v) for v in indices_values])}]"
    parts = [":" for _ in range(int(input_rank))]
    parts[resolved_axis] = repr([int(v) for v in indices_values])
    return f"{params_expr}[{', '.join(parts)}]"


def _direct_dynamic_gather_expr(
    *,
    params_expr: str,
    indices_expr: str,
    axis: int,
    batch_dims: int,
    input_rank: int,
    indices_name: str,
    indices_shape: Optional[Sequence[int]] = None,
    indices_shape_signature: Optional[Sequence[int]] = None,
) -> Optional[str]:
    if int(batch_dims) != 0:
        return None
    if str(indices_name).endswith("_crd_to_dcr_indices"):
        return None
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis < 0 or resolved_axis >= int(input_rank):
        return None
    flat_indices_expr = f"{indices_expr}.to(dtype=torch.int64).reshape(-1)"
    normalized_indices_shape = (
        [int(v) for v in list(indices_shape)] if indices_shape is not None else None
    )
    normalized_indices_shape_signature = (
        [int(v) for v in list(indices_shape_signature)]
        if indices_shape_signature is not None
        else None
    )
    indices_shape_expr = (
        repr(normalized_indices_shape)
        if (
            normalized_indices_shape is not None
            and all(int(v) > 0 for v in normalized_indices_shape)
            and (
                normalized_indices_shape_signature is None
                or all(int(v) > 0 for v in normalized_indices_shape_signature)
            )
        )
        else f"_shape_tensor({indices_expr}, dtype=torch.int64, device={indices_expr}.device)"
    )
    reshaped_expr = (
        f"_reshape_gather_output("
        f"torch.index_select({params_expr}, {resolved_axis}, {flat_indices_expr}), "
        f"{params_expr}, {indices_shape_expr}, axis={resolved_axis})"
    )
    return reshaped_expr


def _is_suffix_flatten_gather_reshape(
    gather_output_shape: Optional[Sequence[int]],
    reshape_output_shape: Optional[Sequence[int]],
) -> bool:
    if gather_output_shape is None or reshape_output_shape is None:
        return False
    gather_shape = [int(v) for v in list(gather_output_shape)]
    reshape_shape = [int(v) for v in list(reshape_output_shape)]
    if (
        len(gather_shape) < 2
        or len(reshape_shape) < 2
        or len(reshape_shape) >= len(gather_shape)
    ):
        return False
    prefix_len = len(reshape_shape) - 1
    if prefix_len <= 0:
        return False
    for gather_dim, reshape_dim in zip(
        gather_shape[:prefix_len], reshape_shape[:prefix_len]
    ):
        if int(gather_dim) == int(reshape_dim):
            continue
        if int(gather_dim) <= 0 or int(reshape_dim) <= 0:
            continue
        return False
    flattened_dims = gather_shape[prefix_len:]
    if len(flattened_dims) < 2 or any(int(dim) <= 0 for dim in flattened_dims):
        return False
    expected_flattened = 1
    for dim in flattened_dims:
        expected_flattened *= int(dim)
    reshape_last_dim = int(reshape_shape[-1])
    if reshape_last_dim <= 0:
        return False
    return int(expected_flattened) == int(reshape_last_dim)


def _direct_gather_reshape_expr(
    *,
    params_expr: str,
    indices_expr: str,
    indices_values: Optional[Sequence[int]],
    indices_shape: Optional[Sequence[int]],
    indices_shape_signature: Optional[Sequence[int]],
    axis: int,
    batch_dims: int,
    input_rank: int,
    indices_name: str,
    final_shape_values: Optional[Sequence[int]],
) -> Optional[str]:
    if int(batch_dims) != 0:
        return None
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis < 0 or resolved_axis >= int(input_rank):
        return None
    flat_indices_expr: Optional[str] = None
    if indices_values is not None:
        flat_indices_expr = (
            f"torch.as_tensor({repr([int(v) for v in list(indices_values)])}, "
            f"dtype=torch.int64, device={params_expr}.device)"
        )
    else:
        if str(indices_name).endswith("_crd_to_dcr_indices"):
            return None
        normalized_indices_shape = (
            [int(v) for v in list(indices_shape)] if indices_shape is not None else None
        )
        normalized_indices_shape_signature = (
            [int(v) for v in list(indices_shape_signature)]
            if indices_shape_signature is not None
            else None
        )
        if (
            normalized_indices_shape is None
            or len(normalized_indices_shape) <= 1
            or not all(int(v) > 0 for v in normalized_indices_shape)
            or (
                normalized_indices_shape_signature is not None
                and not all(int(v) > 0 for v in normalized_indices_shape_signature)
            )
        ):
            return None
        flat_indices_expr = f"{indices_expr}.to(dtype=torch.int64).reshape(-1)"
    if flat_indices_expr is None:
        return None
    if final_shape_values is None:
        return None
    resolved_shape_values = [int(v) for v in list(final_shape_values)]
    if len(resolved_shape_values) == 0:
        return None
    if all(int(v) > 0 for v in resolved_shape_values):
        final_shape_expr = repr(resolved_shape_values)
    elif int(resolved_shape_values[0]) <= 0 and all(
        int(v) > 0 for v in resolved_shape_values[1:]
    ):
        final_shape_expr = (
            f"[int({params_expr}.shape[0]), "
            + ", ".join(str(int(v)) for v in resolved_shape_values[1:])
            + "]"
        )
    else:
        return None
    return (
        f"torch.reshape("
        f"torch.index_select({params_expr}, {resolved_axis}, {flat_indices_expr}), "
        f"{final_shape_expr})"
    )


def _should_elide_crd_to_dcr_gather_for_depth_to_space(
    *,
    model_ir: ModelIR,
    params_name: str,
    indices_name: str,
    output_name: str,
    axis: int,
    batch_dims: int,
) -> bool:
    if int(batch_dims) != 0 or not str(indices_name).endswith("_crd_to_dcr_indices"):
        return False
    input_tensor = model_ir.tensors.get(str(params_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if input_tensor is None or output_tensor is None:
        return False
    input_rank = len(list(input_tensor.shape))
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis != 1:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if not (
        is_channel_first_logical_layout(input_layout)
        or is_channel_first_logical_layout(output_layout)
    ):
        return False
    consumer_op_types = [
        str(consumer.op_type)
        for consumer in model_ir.operators
        if str(output_name) in {str(v) for v in consumer.inputs}
    ]
    return len(consumer_op_types) > 0 and all(
        op_type == "DEPTH_TO_SPACE" for op_type in consumer_op_types
    )
