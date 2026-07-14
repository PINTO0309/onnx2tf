from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import _constant_int_list


def _channel_first_reduction_plan_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    op: OperatorIR,
    input_name: str,
) -> Optional[Tuple[str, List[int]]]:
    op_type = str(op.op_type)
    if op_type not in {
        "SUM",
        "MEAN",
        "REDUCE_MAX",
        "REDUCE_MIN",
        "REDUCE_PROD",
        "REDUCE_ANY",
    }:
        return None
    if len(op.inputs) < 2 or str(op.inputs[0]) != str(input_name):
        return None
    input_tensor = model_ir.tensors.get(str(input_name), None)
    if input_tensor is None:
        return None
    input_rank = len(list(input_tensor.shape))
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    if not is_channel_last_logical_layout(input_layout):
        return None
    if input_rank == 3:
        return None
    if (
        input_rank == 4
        and len(op.inputs) >= 2
        and _constant_int_list(
            model_ir.tensors.get(str(op.inputs[1]), None)
        )
        == [1, 2]
    ):
        input_shape = [int(v) for v in list(input_tensor.shape)]
        non_batch_dims = [dim for dim in input_shape[1:] if dim > 0]
        if len(set(non_batch_dims)) != len(non_batch_dims):
            return None
    input_expr = channel_first_tensor_expr_aliases.get(str(input_name), None)
    if input_expr is None:
        return None
    axis_values = _constant_int_list(
        model_ir.tensors.get(str(op.inputs[1]), None)
    )
    if axis_values is None:
        return None
    cl_spatial_axes = {
        3: [1],
        4: [1, 2],
        5: [1, 2, 3],
    }.get(input_rank, None)
    cf_spatial_axes = {
        3: [2],
        4: [2, 3],
        5: [2, 3, 4],
    }.get(input_rank, None)
    if cl_spatial_axes is None or cf_spatial_axes is None:
        return None
    if [int(v) for v in list(axis_values)] != [
        int(v) for v in list(cl_spatial_axes)
    ]:
        return None
    return str(input_expr), [int(v) for v in list(cf_spatial_axes)]


def _normalized_constant_reduction_axes_for_codegen(
    *,
    axis_values: Optional[Sequence[int]],
    rank: int,
) -> Optional[List[int]]:
    if axis_values is None:
        return None
    normalized_axes: List[int] = []
    for axis in list(axis_values):
        normalized_axis = int(axis)
        if normalized_axis < 0:
            normalized_axis += int(rank)
        if normalized_axis < 0 or normalized_axis >= int(rank):
            return None
        if normalized_axis not in normalized_axes:
            normalized_axes.append(normalized_axis)
    normalized_axes.sort()
    return normalized_axes


def _direct_mean_reduction_expr_for_codegen(
    *,
    normalized_constant_reduction_axes_fn: Callable[
        [Optional[Sequence[int]], int], Optional[List[int]]
    ],
    input_expr: str,
    axes: Optional[Sequence[int]],
    input_rank: int,
    keepdims: bool,
) -> Optional[str]:
    normalized_axes = normalized_constant_reduction_axes_fn(axes, input_rank)
    if normalized_axes is None:
        return None
    if len(normalized_axes) == 0:
        return input_expr
    dim_literal: Any = (
        normalized_axes[0] if len(normalized_axes) == 1 else normalized_axes
    )
    return (
        f"torch.mean({input_expr}, dim={repr(dim_literal)}, keepdim={keepdims})"
    )
