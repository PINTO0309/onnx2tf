from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    channel_first_logical_layout,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import _constant_int_list
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _perm_cf_to_cl,
    _perm_cl_to_cf,
    _permute_shape,
    _tensor_name_suggests_channel_last_layout_for_codegen,
)


def _channel_first_concat_input_expr_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_name: str,
    tensor_expr_fn: Callable[[str], str],
) -> Optional[str]:
    alias_expr = channel_first_tensor_expr_aliases.get(str(tensor_name), None)
    if alias_expr is not None:
        return str(alias_expr)
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    tensor_layout = normalize_logical_layout(tensor.logical_layout)
    tensor_rank = len(list(tensor.shape))
    if is_channel_first_logical_layout(tensor_layout):
        return tensor_expr_fn(str(tensor_name))
    if tensor_rank not in {4, 5}:
        return None
    perm_to_cf = _perm_cl_to_cf(tensor_rank)
    if (
        perm_to_cf is not None
        and (
            is_channel_last_logical_layout(tensor_layout)
            or (
                tensor_layout == LOGICAL_LAYOUT_UNKNOWN
                and _tensor_name_suggests_channel_last_layout_for_codegen(
                    str(tensor_name)
                )
            )
        )
    ):
        base_expr = tensor_expr_fn(str(tensor_name))
        return (
            f"{base_expr}.permute("
            f"{', '.join(str(int(v)) for v in perm_to_cf)}).contiguous()"
        )
    return None


def _can_fold_channel_last_alias_slice_consumer_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    expected_input_name: str,
) -> bool:
    op_type = str(op.op_type)
    if op_type == "SLICE":
        return (
            len(op.inputs) >= 3
            and str(op.inputs[0]) == str(expected_input_name)
            and _constant_int_list(
                model_ir.tensors.get(str(op.inputs[1]), None)
            )
            is not None
            and _constant_int_list(
                model_ir.tensors.get(str(op.inputs[2]), None)
            )
            is not None
        )
    if op_type == "STRIDED_SLICE":
        options = dict(op.options)
        return (
            len(op.inputs) >= 4
            and str(op.inputs[0]) == str(expected_input_name)
            and _constant_int_list(
                model_ir.tensors.get(str(op.inputs[1]), None)
            )
            is not None
            and _constant_int_list(
                model_ir.tensors.get(str(op.inputs[2]), None)
            )
            is not None
            and _constant_int_list(
                model_ir.tensors.get(str(op.inputs[3]), None)
            )
            is not None
            and int(options.get("ellipsisMask", 0)) == 0
            and int(options.get("newAxisMask", 0)) == 0
            and int(options.get("shrinkAxisMask", 0)) == 0
        )
    return False


def _is_valid_concat_axis_for_channel_first_shapes_for_codegen(
    *,
    input_shapes: Sequence[Sequence[int]],
    output_shape: Sequence[int],
    axis: int,
) -> bool:
    output_items = [int(v) for v in list(output_shape)]
    rank = len(output_items)
    if axis < 0 or axis >= rank:
        return False
    expected_axis_extent = 0
    axis_extent_static = int(output_items[axis]) > 0
    for input_shape in input_shapes:
        input_items = [int(v) for v in list(input_shape)]
        if len(input_items) != rank:
            return False
        for dim_index, (input_dim, output_dim) in enumerate(
            zip(input_items, output_items)
        ):
            if dim_index == axis:
                continue
            if (
                int(input_dim) > 0
                and int(output_dim) > 0
                and int(input_dim) != int(output_dim)
            ):
                return False
        if int(input_items[axis]) <= 0:
            axis_extent_static = False
        else:
            expected_axis_extent += int(input_items[axis])
    if axis_extent_static:
        return expected_axis_extent == int(output_items[axis])
    return True


def _resolve_concat_axis_for_channel_first_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    channel_first_shape_for_tensor_fn: Callable[
        [str], Optional[List[int]]
    ],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
) -> Optional[Tuple[int, List[int], List[int]]]:
    if len(op.outputs) != 1:
        return None
    input_shapes_cf: List[List[int]] = []
    for input_name in op.inputs:
        input_shape_cf = channel_first_shape_for_tensor_fn(str(input_name))
        if input_shape_cf is None:
            return None
        input_shapes_cf.append([int(v) for v in list(input_shape_cf)])
    stored_output_shape = tensor_shape_list_fn(str(op.outputs[0]))
    if stored_output_shape is None:
        return None
    rank = len(list(stored_output_shape))
    axis = int(op.options.get("axis", 0))
    if axis < 0:
        axis += int(rank)
    candidate_output_specs: List[Tuple[List[int], List[int]]] = [
        (
            [int(v) for v in list(stored_output_shape)],
            [int(v) for v in list(range(rank))],
        ),
    ]
    perm_to_cf = _perm_cl_to_cf(rank)
    perm_from_cf = _perm_cf_to_cl(rank)
    if perm_to_cf is not None and perm_from_cf is not None:
        permuted_output_shape = _permute_shape(stored_output_shape, perm_to_cf)
        if permuted_output_shape is not None:
            candidate_spec = (
                [int(v) for v in list(permuted_output_shape)],
                [int(v) for v in list(perm_from_cf)],
            )
            if candidate_spec not in candidate_output_specs:
                candidate_output_specs.append(candidate_spec)
    if perm_from_cf is not None and perm_to_cf is not None:
        permuted_output_shape = _permute_shape(
            stored_output_shape,
            perm_from_cf,
        )
        if permuted_output_shape is not None:
            candidate_spec = (
                [int(v) for v in list(permuted_output_shape)],
                [int(v) for v in list(perm_to_cf)],
            )
            if candidate_spec not in candidate_output_specs:
                candidate_output_specs.append(candidate_spec)
    candidate_axes: List[int] = []
    if 0 <= int(axis) < int(rank):
        candidate_axes.append(int(axis))
    if perm_to_cf is not None:
        mapped_axis = next(
            (
                int(index)
                for index, source_axis in enumerate(perm_to_cf)
                if int(source_axis) == int(axis)
            ),
            None,
        )
        if mapped_axis is not None and int(mapped_axis) not in candidate_axes:
            candidate_axes.append(int(mapped_axis))
    for output_shape_cf, perm_from_candidate in candidate_output_specs:
        for candidate_axis in candidate_axes:
            if _is_valid_concat_axis_for_channel_first_shapes_for_codegen(
                input_shapes=input_shapes_cf,
                output_shape=output_shape_cf,
                axis=int(candidate_axis),
            ):
                return (
                    int(candidate_axis),
                    [int(v) for v in list(output_shape_cf)],
                    [int(v) for v in list(perm_from_candidate)],
                )
    return None


def _can_keep_channel_first_slice_output_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    output_name: str,
    resolve_concat_axis_for_channel_first_fn: Callable[
        [OperatorIR], Optional[Tuple[int, List[int], List[int]]]
    ],
) -> bool:
    consumer_indices = consumer_index.get(str(output_name), [])
    if len(consumer_indices) == 0:
        return False
    for consumer_idx in consumer_indices:
        consumer_op = model_ir.operators[int(consumer_idx)]
        if str(consumer_op.op_type) != "CONCATENATION":
            return False
        concat_cf_spec = resolve_concat_axis_for_channel_first_fn(consumer_op)
        if concat_cf_spec is None:
            return False
        consumer_output_name = (
            str(consumer_op.outputs[0]) if len(consumer_op.outputs) == 1 else ""
        )
        if consumer_output_name == "":
            return False
        consumer_output_tensor = model_ir.tensors.get(
            consumer_output_name,
            None,
        )
        if consumer_output_tensor is None:
            return False
        consumer_output_rank = len(list(consumer_output_tensor.shape))
        consumer_output_layout = normalize_logical_layout(
            consumer_output_tensor.logical_layout
        )
        if consumer_output_layout in {
            LOGICAL_LAYOUT_UNKNOWN,
            channel_first_logical_layout(consumer_output_rank),
        }:
            continue
        return False
    return True
