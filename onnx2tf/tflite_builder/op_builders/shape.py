from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import copy

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR, normalize_onnx_shape
from onnx2tf.tflite_builder.op_builders.shared import make_transpose

_BICUBIC_MATRIX_CACHE: Dict[Tuple[int, int, str, float, bool], np.ndarray] = {}


def _is_unresolved_placeholder_shape(shape: List[int], signature: List[int] | None) -> bool:
    if len(shape) == 0 or not all(int(v) == 1 for v in shape):
        return False
    if signature is None:
        return len(shape) == 1
    if len(signature) != len(shape):
        return False
    if any(int(v) < 0 for v in signature):
        return True
    if len(shape) == 1 and int(signature[0]) == 1:
        return True
    return False


def _materialize_tensor_shape_from_signature(tensor: Any, *, signature: List[int]) -> None:
    tensor.shape = [int(v) if int(v) > 0 else 1 for v in list(signature)]
    tensor.shape_signature = [int(v) if int(v) > 0 else -1 for v in list(signature)]


def _numpy_dtype_from_tflite_dtype(tflite_dtype: str) -> np.dtype:
    dt = str(tflite_dtype).upper()
    if dt == "BOOL":
        return np.dtype(np.bool_)
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
    if dt == "INT64":
        return np.dtype(np.int64)
    if dt == "UINT64":
        return np.dtype(np.uint64)
    if dt == "FLOAT16":
        return np.dtype(np.float16)
    if dt == "FLOAT32":
        return np.dtype(np.float32)
    if dt == "FLOAT64":
        return np.dtype(np.float64)
    raise NotImplementedError(
        f"Unsupported TFLite dtype in shape builder: {tflite_dtype}"
    )


def _prefer_int32_index_output_dtype(
    *,
    ctx: Any,
    tensor_name: str,
    requested_dtype: str,
) -> str:
    dtype = str(requested_dtype).upper()
    if dtype == "INT32":
        return "INT32"
    if dtype == "INT64":
        tensor = ctx.model_ir.tensors.get(tensor_name, None)
        if tensor is not None:
            tensor.dtype = "INT32"
        if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
            ctx.dtype_map[str(tensor_name)] = "INT32"
        return "INT32"
    return dtype


def _normalize_axis(axis: int, rank: int, *, op_name: str) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += int(rank)
    if normalized < 0 or normalized >= int(rank):
        raise NotImplementedError(
            f"Slice axis out of range in flatbuffer_direct. op={op_name} axis={axis} rank={rank}"
        )
    return normalized


def _propagate_passthrough_dtype_and_quantization(
    *,
    ctx: Any,
    src_tensor_name: str,
    dst_tensor_name: str,
) -> None:
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src_tensor = ctx.model_ir.tensors[src_tensor_name]
    dst_tensor = ctx.model_ir.tensors[dst_tensor_name]
    dst_tensor.dtype = str(src_tensor.dtype)
    if src_tensor.quantization is not None:
        dst_tensor.quantization = _clone_quantization(src_tensor.quantization)


def _propagate_passthrough_shape_signature(
    *,
    ctx: Any,
    src_tensor_name: str,
    dst_tensor_name: str,
) -> None:
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src_tensor = ctx.model_ir.tensors[src_tensor_name]
    dst_tensor = ctx.model_ir.tensors[dst_tensor_name]
    dst_tensor.shape = [int(v) for v in list(src_tensor.shape)]
    if src_tensor.shape_signature is not None:
        dst_tensor.shape_signature = [int(v) for v in list(src_tensor.shape_signature)]
    else:
        dst_tensor.shape_signature = [int(v) for v in list(src_tensor.shape)]


def _parse_slice_axes_or_steps(
    *,
    node: Any,
    ctx: Any,
    input_index: int,
    attr_name: str,
    default_values: list[int],
    label: str,
) -> list[int]:
    values: list[int] | None = None
    if len(node.inputs) > input_index and str(node.inputs[input_index].name) != "":
        arr = ctx.get_constant_array(node.inputs[input_index].name)
        if arr is None:
            raise NotImplementedError(
                f"Slice {label} must be constant for flatbuffer_direct. op={node.name}"
            )
        values = [int(v) for v in np.asarray(arr).reshape(-1).tolist()]
    elif attr_name in node.attrs:
        attr_val = node.attrs.get(attr_name)
        if isinstance(attr_val, (list, tuple, np.ndarray)):
            values = [int(v) for v in np.asarray(attr_val).reshape(-1).tolist()]
        elif attr_val is None:
            values = []
        else:
            values = [int(attr_val)]
    if values is None:
        values = [int(v) for v in default_values]
    return [int(v) for v in values]


def _parse_slice_indices(
    *,
    node: Any,
    ctx: Any,
    input_index: int,
    attr_name: str,
    label: str,
) -> list[int]:
    values: list[int] | None = None
    if len(node.inputs) > input_index and str(node.inputs[input_index].name) != "":
        arr = ctx.get_constant_array(node.inputs[input_index].name)
        if arr is None:
            raise NotImplementedError(
                f"Slice {label} must be constant for flatbuffer_direct. op={node.name}"
            )
        values = [int(v) for v in np.asarray(arr).reshape(-1).tolist()]
    elif attr_name in node.attrs:
        attr_val = node.attrs.get(attr_name)
        if isinstance(attr_val, (list, tuple, np.ndarray)):
            values = [int(v) for v in np.asarray(attr_val).reshape(-1).tolist()]
        elif attr_val is None:
            values = []
        else:
            values = [int(attr_val)]
    if values is None:
        raise NotImplementedError(
            f"Slice {label} must be provided as constant input[{input_index}] "
            f"or attribute '{attr_name}'. op={node.name}"
        )
    return [int(v) for v in values]


def _get_slice_rank_limit(ctx: Any) -> int:
    return int(
        max(
            1,
            min(
                5,
                int(
                    getattr(
                        ctx,
                        "number_of_dimensions_after_flexstridedslice_compression",
                        5,
                    )
                ),
            ),
        )
    )


def _remap_axis_mask(mask: int, remaining_axes: list[int]) -> int:
    remapped = 0
    mask_i = int(mask)
    for new_axis, old_axis in enumerate(remaining_axes):
        if ((mask_i >> int(old_axis)) & 1) != 0:
            remapped |= (1 << int(new_axis))
    return int(remapped)


def _emit_slice_or_stridedslice(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    use_strided_slice: bool,
    begin: list[int],
    size: list[int],
    end_for_strided: list[int],
    strides_for_strided: list[int],
    strided_slice_options: dict[str, int | bool],
    name_prefix: str,
) -> None:
    if use_strided_slice:
        begin_name = ctx.add_const_tensor(
            f"{name_prefix}_stridedslice_begin",
            np.asarray(begin, dtype=np.int32),
        )
        end_name = ctx.add_const_tensor(
            f"{name_prefix}_stridedslice_end",
            np.asarray(end_for_strided, dtype=np.int32),
        )
        strides_name = ctx.add_const_tensor(
            f"{name_prefix}_stridedslice_strides",
            np.asarray(strides_for_strided, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="STRIDED_SLICE",
                inputs=[input_name, begin_name, end_name, strides_name],
                outputs=[output_name],
                options={
                    "beginMask": int(strided_slice_options.get("beginMask", 0)),
                    "endMask": int(strided_slice_options.get("endMask", 0)),
                    "ellipsisMask": int(strided_slice_options.get("ellipsisMask", 0)),
                    "newAxisMask": int(strided_slice_options.get("newAxisMask", 0)),
                    "shrinkAxisMask": int(strided_slice_options.get("shrinkAxisMask", 0)),
                    "offset": bool(strided_slice_options.get("offset", False)),
                },
            )
        )
        return

    begin_name = ctx.add_const_tensor(
        f"{name_prefix}_slice_begin",
        np.asarray(begin, dtype=np.int32),
    )
    size_name = ctx.add_const_tensor(
        f"{name_prefix}_slice_size",
        np.asarray(size, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[input_name, begin_name, size_name],
            outputs=[output_name],
        )
    )


def _decompose_high_rank_slice_like(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    preferred_slice_axes: list[int],
    use_strided_slice: bool,
    begin: list[int],
    size: list[int],
    end_for_strided: list[int],
    strides_for_strided: list[int],
    strided_slice_options: dict[str, int | bool],
) -> bool:
    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    input_rank = len(input_shape)
    target_rank = _get_slice_rank_limit(ctx)
    if input_rank <= int(target_rank):
        return False
    if any(int(v) <= 0 for v in input_shape):
        return False

    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    output_shape = (
        [int(v) for v in list(output_tensor.shape)]
        if output_tensor is not None
        else []
    )
    if len(output_shape) != input_rank:
        return False
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in list(output_shape)]
    )

    begin_mask = int(strided_slice_options.get("beginMask", 0))
    end_mask = int(strided_slice_options.get("endMask", 0))
    avoid_axes = set(int(v) for v in list(preferred_slice_axes))
    required_compress = int(input_rank - int(target_rank))

    def _is_passthrough_axis(axis: int) -> bool:
        dim = int(input_shape[axis])
        if not use_strided_slice:
            return int(begin[axis]) == 0 and int(size[axis]) == int(dim)
        begin_ok = (((begin_mask >> int(axis)) & 1) != 0) or int(begin[axis]) == 0
        end_ok = (((end_mask >> int(axis)) & 1) != 0) or int(end_for_strided[axis]) == int(dim)
        return begin_ok and end_ok and int(strides_for_strided[axis]) == 1

    candidate_axes = [
        int(axis)
        for axis in range(input_rank)
        if axis not in avoid_axes and _is_passthrough_axis(axis)
    ]
    if len(candidate_axes) < required_compress:
        candidate_axes = [
            int(axis)
            for axis in range(input_rank)
            if _is_passthrough_axis(axis)
        ]
    if len(candidate_axes) < required_compress:
        return False

    candidate_axes = sorted(
        [int(v) for v in list(candidate_axes)],
        key=lambda axis: (
            0 if int(input_shape[axis]) == 1 else 1,
            int(input_shape[axis]),
            int(axis),
        ),
    )
    split_axes = sorted([int(v) for v in candidate_axes[:required_compress]])
    split_dims = [int(input_shape[axis]) for axis in split_axes]
    remaining_axes = [int(axis) for axis in range(input_rank) if axis not in set(split_axes)]
    if len(remaining_axes) != target_rank:
        return False

    reduced_begin = [int(begin[axis]) for axis in remaining_axes]
    reduced_size = [int(size[axis]) for axis in remaining_axes]
    reduced_end = [int(end_for_strided[axis]) for axis in remaining_axes]
    reduced_strides = [int(strides_for_strided[axis]) for axis in remaining_axes]
    reduced_options = dict(strided_slice_options)
    reduced_options["beginMask"] = int(_remap_axis_mask(begin_mask, remaining_axes))
    reduced_options["endMask"] = int(_remap_axis_mask(end_mask, remaining_axes))
    reduced_options["ellipsisMask"] = 0
    reduced_options["newAxisMask"] = 0
    reduced_options["shrinkAxisMask"] = 0
    reduced_options["offset"] = bool(strided_slice_options.get("offset", False))
    reduced_out_shape = [int(output_shape[axis]) for axis in remaining_axes]
    reduced_out_sig = [int(output_signature[axis]) for axis in remaining_axes]
    rank_tag = f"rank{int(target_rank)}"

    split_tensors: list[str] = [str(input_name)]
    work_axes = [int(v) for v in split_axes]
    split_step = 0
    while split_step < len(work_axes):
        axis = int(work_axes[split_step])
        next_split_tensors: list[str] = []
        for tensor_idx, split_tensor_name in enumerate(split_tensors):
            split_tensor_shape = [int(v) for v in list(ctx.get_tensor_shape(split_tensor_name))]
            axis_dim = int(split_tensor_shape[axis])
            if axis_dim <= 0:
                return False
            split_tensor_ir = ctx.model_ir.tensors.get(split_tensor_name, None)
            split_tensor_sig = (
                [int(v) for v in list(split_tensor_ir.shape_signature)]
                if split_tensor_ir is not None and split_tensor_ir.shape_signature is not None
                else [int(v) for v in list(split_tensor_shape)]
            )
            for gather_index in range(axis_dim):
                gather_index_name = ctx.add_const_tensor(
                    f"{output_name}_{rank_tag}_slice_split_axis{axis}_idx{gather_index}",
                    np.asarray(int(gather_index), dtype=np.int32),
                )
                gather_index_ir = ctx.model_ir.tensors.get(gather_index_name, None)
                if gather_index_ir is not None:
                    gather_index_ir.shape = []
                    gather_index_ir.shape_signature = []
                gathered_shape = [
                    int(v) for i, v in enumerate(split_tensor_shape) if int(i) != int(axis)
                ]
                gathered_sig = [
                    int(v) for i, v in enumerate(split_tensor_sig) if int(i) != int(axis)
                ]
                gathered_name = ctx.add_intermediate_tensor(
                    f"{output_name}_{rank_tag}_slice_split_{split_step}_{tensor_idx}_{gather_index}",
                    dtype=ctx.get_tensor_dtype(split_tensor_name),
                    shape=list(gathered_shape),
                )
                gathered_ir = ctx.model_ir.tensors.get(gathered_name, None)
                if gathered_ir is not None:
                    gathered_ir.shape_signature = [int(v) for v in list(gathered_sig)]
                    if split_tensor_ir is not None:
                        gathered_ir.quantization = _clone_quantization(split_tensor_ir.quantization)
                ctx.add_operator(
                    OperatorIR(
                        op_type="GATHER",
                        inputs=[split_tensor_name, gather_index_name],
                        outputs=[gathered_name],
                        options={"axis": int(axis), "batchDims": 0},
                    )
                )
                next_split_tensors.append(str(gathered_name))
        split_tensors = next_split_tensors
        split_step += 1
        if split_step >= len(work_axes):
            break
        current_axis = int(axis)
        work_axes = [
            int(v) if int(v) <= int(current_axis) else int(v) - 1
            for v in work_axes
        ]

    sliced_tensors: list[str] = []
    for idx, split_tensor_name in enumerate(split_tensors):
        sliced_name = ctx.add_intermediate_tensor(
            f"{output_name}_{rank_tag}_slice_core_{idx}",
            dtype=ctx.get_tensor_dtype(split_tensor_name),
            shape=list(reduced_out_shape),
        )
        sliced_ir = ctx.model_ir.tensors.get(sliced_name, None)
        if sliced_ir is not None:
            sliced_ir.shape_signature = [int(v) for v in list(reduced_out_sig)]
            split_ir = ctx.model_ir.tensors.get(split_tensor_name, None)
            if split_ir is not None:
                sliced_ir.quantization = _clone_quantization(split_ir.quantization)
        _emit_slice_or_stridedslice(
            ctx=ctx,
            input_name=split_tensor_name,
            output_name=sliced_name,
            use_strided_slice=bool(use_strided_slice),
            begin=[int(v) for v in list(reduced_begin)],
            size=[int(v) for v in list(reduced_size)],
            end_for_strided=[int(v) for v in list(reduced_end)],
            strides_for_strided=[int(v) for v in list(reduced_strides)],
            strided_slice_options=dict(reduced_options),
            name_prefix=f"{output_name}_{rank_tag}_slice_core_{idx}",
        )
        sliced_tensors.append(str(sliced_name))

    expanded_tensors = list(sliced_tensors)
    for expand_axis in sorted([int(v) for v in list(split_axes)]):
        axis_name = ctx.add_const_tensor(
            f"{output_name}_{rank_tag}_slice_expand_axis_{expand_axis}",
            np.asarray([int(expand_axis)], dtype=np.int32),
        )
        next_expanded: list[str] = []
        for idx, tensor_name in enumerate(expanded_tensors):
            tensor_shape = [int(v) for v in list(ctx.get_tensor_shape(tensor_name))]
            expanded_shape = (
                [int(v) for v in tensor_shape[: int(expand_axis)]]
                + [1]
                + [int(v) for v in tensor_shape[int(expand_axis):]]
            )
            expanded_name = ctx.add_intermediate_tensor(
                f"{output_name}_{rank_tag}_slice_expanded_{expand_axis}_{idx}",
                dtype=ctx.get_tensor_dtype(tensor_name),
                shape=list(expanded_shape),
            )
            tensor_ir = ctx.model_ir.tensors.get(tensor_name, None)
            expanded_ir = ctx.model_ir.tensors.get(expanded_name, None)
            if tensor_ir is not None and expanded_ir is not None:
                tensor_sig = (
                    [int(v) for v in list(tensor_ir.shape_signature)]
                    if tensor_ir.shape_signature is not None
                    else [int(v) for v in list(tensor_shape)]
                )
                expanded_ir.shape_signature = (
                    [int(v) for v in tensor_sig[: int(expand_axis)]]
                    + [1]
                    + [int(v) for v in tensor_sig[int(expand_axis):]]
                )
                expanded_ir.quantization = _clone_quantization(tensor_ir.quantization)
            ctx.add_operator(
                OperatorIR(
                    op_type="EXPAND_DIMS",
                    inputs=[tensor_name, axis_name],
                    outputs=[expanded_name],
                )
            )
            next_expanded.append(str(expanded_name))
        expanded_tensors = next_expanded

    grouped_tensors = list(expanded_tensors)
    concat_axes = list(reversed([int(v) for v in list(split_axes)]))
    grouping_dims = list(reversed([int(v) for v in list(split_dims)]))
    for stage_idx, (concat_axis, target_concat_dim) in enumerate(zip(concat_axes, grouping_dims)):
        if int(target_concat_dim) <= 0:
            return False
        next_grouped: list[str] = []
        for group_idx in range(0, len(grouped_tensors), int(target_concat_dim)):
            chunk = grouped_tensors[group_idx: group_idx + int(target_concat_dim)]
            if len(chunk) == 0:
                continue
            if len(chunk) == 1:
                next_grouped.append(str(chunk[0]))
                continue
            concat_out = (
                output_name
                if stage_idx == int(len(concat_axes) - 1) and len(grouped_tensors) == len(chunk)
                else f"{output_name}_{rank_tag}_slice_concat_{stage_idx}_{group_idx // int(target_concat_dim)}"
            )
            concat_shape = [int(v) for v in list(ctx.get_tensor_shape(chunk[0]))]
            concat_shape[int(concat_axis)] = int(
                sum(int(ctx.get_tensor_shape(name)[int(concat_axis)]) for name in chunk)
            )
            if concat_out != output_name:
                ctx.add_intermediate_tensor(
                    concat_out,
                    dtype=ctx.get_tensor_dtype(chunk[0]),
                    shape=list(concat_shape),
                )
                concat_out_ir = ctx.model_ir.tensors.get(concat_out, None)
                chunk_ir = ctx.model_ir.tensors.get(chunk[0], None)
                if concat_out_ir is not None and chunk_ir is not None:
                    chunk_sig = (
                        [int(v) for v in list(chunk_ir.shape_signature)]
                        if chunk_ir.shape_signature is not None
                        else [int(v) for v in list(ctx.get_tensor_shape(chunk[0]))]
                    )
                    concat_sig = [int(v) for v in list(chunk_sig)]
                    concat_sig[int(concat_axis)] = int(
                        sum(int(ctx.get_tensor_shape(name)[int(concat_axis)]) for name in chunk)
                    )
                    concat_out_ir.shape_signature = [int(v) for v in list(concat_sig)]
                    concat_out_ir.quantization = _clone_quantization(chunk_ir.quantization)
            ctx.add_operator(
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=[str(v) for v in chunk],
                    outputs=[concat_out],
                    options={"axis": int(concat_axis), "fusedActivationFunction": "NONE"},
                )
            )
            next_grouped.append(str(concat_out))
        grouped_tensors = next_grouped

    if len(grouped_tensors) != 1:
        return False
    final_name = str(grouped_tensors[0])
    if final_name != str(output_name):
        final_shape = [int(v) for v in list(ctx.get_tensor_shape(final_name))]
        reshape_shape_name = ctx.add_const_tensor(
            f"{output_name}_{rank_tag}_slice_identity_shape",
            np.asarray(final_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[final_name, reshape_shape_name],
                outputs=[output_name],
                options={"newShape": [int(v) for v in final_shape]},
            )
        )
        final_tensor = ctx.model_ir.tensors.get(final_name, None)
        output_tensor_final = ctx.model_ir.tensors.get(output_name, None)
        if final_tensor is not None and output_tensor_final is not None:
            output_tensor_final.shape = [int(v) for v in list(final_shape)]
            if final_tensor.shape_signature is not None:
                output_tensor_final.shape_signature = [
                    int(v) for v in list(final_tensor.shape_signature)
                ]
            output_tensor_final.quantization = _clone_quantization(final_tensor.quantization)
    return True


def build_slice_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    dynamic_start_input_name = ""
    starts: list[int] = []
    if (
        len(node.inputs) > 1
        and str(node.inputs[1].name) != ""
        and ctx.get_constant_array(node.inputs[1].name) is None
        and "starts" not in node.attrs
    ):
        dynamic_start_input_name = str(node.inputs[1].name)
    else:
        starts = _parse_slice_indices(
            node=node,
            ctx=ctx,
            input_index=1,
            attr_name="starts",
            label="starts",
        )

    dynamic_end_input_name = ""
    ends: list[int] = []
    if (
        len(node.inputs) > 2
        and str(node.inputs[2].name) != ""
        and ctx.get_constant_array(node.inputs[2].name) is None
        and "ends" not in node.attrs
    ):
        dynamic_end_input_name = str(node.inputs[2].name)
    else:
        ends = _parse_slice_indices(
            node=node,
            ctx=ctx,
            input_index=2,
            attr_name="ends",
            label="ends",
        )
    if (
        dynamic_start_input_name == ""
        and dynamic_end_input_name == ""
        and len(starts) != len(ends)
    ):
        raise NotImplementedError(
            f"Slice starts and ends length mismatch. op={node.name} "
            f"starts_len={len(starts)} ends_len={len(ends)}"
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor is not None and input_tensor.shape_signature is not None
        else list(input_shape)
    )
    rank = len(input_shape)
    default_slice_len = int(
        len(starts)
        if len(starts) > 0
        else (len(ends) if len(ends) > 0 else 1)
    )
    axes = _parse_slice_axes_or_steps(
        node=node,
        ctx=ctx,
        input_index=3,
        attr_name="axes",
        default_values=[int(v) for v in range(default_slice_len)],
        label="axes",
    )
    steps = _parse_slice_axes_or_steps(
        node=node,
        ctx=ctx,
        input_index=4,
        attr_name="steps",
        default_values=[1 for _ in range(len(axes))],
        label="steps",
    )
    if len(steps) != len(axes):
        raise NotImplementedError(
            f"Slice starts/axes/steps length mismatch. op={node.name} "
            f"starts_len={len(starts)} axes_len={len(axes)} steps_len={len(steps)}"
        )
    normalized_axes = [
        _normalize_axis(axis_raw, rank, op_name=node.name)
        for axis_raw in axes
    ]

    if dynamic_start_input_name == "" and len(normalized_axes) != len(starts):
        raise NotImplementedError(
            f"Slice starts/axes length mismatch. op={node.name} "
            f"starts_len={len(starts)} axes_len={len(normalized_axes)}"
        )
    if dynamic_end_input_name == "" and len(normalized_axes) != len(ends):
        raise NotImplementedError(
            f"Slice ends/axes length mismatch. op={node.name} "
            f"ends_len={len(ends)} axes_len={len(normalized_axes)}"
        )

    if dynamic_start_input_name != "" or (
        dynamic_end_input_name != "" and len(normalized_axes) == 1
    ):
        if not (
            len(normalized_axes) == 1
            and len(steps) == 1
            and int(steps[0]) > 0
            and (
                dynamic_start_input_name != ""
                or (len(starts) == 1 and int(starts[0]) >= 0)
            )
        ):
            raise NotImplementedError(
                "Slice with dynamic starts/ends currently supports only "
                "single-axis positive-step slicing in flatbuffer_direct. "
                f"op={node.name} starts={starts} ends={ends} axes={axes} steps={steps}"
            )

        axis = int(normalized_axes[0])
        step = int(steps[0])
        int32_max = int(np.iinfo(np.int32).max)

        def _prepare_dynamic_len1_i32(dynamic_name: str, suffix: str) -> str:
            dynamic_shape = [int(v) for v in ctx.get_tensor_shape(dynamic_name)]
            if len(dynamic_shape) != 1 or (int(dynamic_shape[0]) > 0 and int(dynamic_shape[0]) != 1):
                raise NotImplementedError(
                    "Slice dynamic starts/ends must be rank-1 length-1 "
                    f"for builtin lowering. op={node.name} tensor={dynamic_name} shape={dynamic_shape}"
                )
            dynamic_dtype = str(ctx.get_tensor_dtype(dynamic_name)).upper()
            out_name = dynamic_name
            if dynamic_dtype != "INT32":
                out_name = ctx.add_intermediate_tensor(
                    f"{output_name}_stridedslice_{suffix}_i32",
                    dtype="INT32",
                    shape=[1],
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[dynamic_name],
                        outputs=[out_name],
                        options={
                            "inDataType": dynamic_dtype,
                            "outDataType": "INT32",
                        },
                    )
                )
            return out_name

        def _compose_rank_vector_with_dynamic_axis(
            *,
            dynamic_len1_name: str,
            axis_index: int,
            fill_value: int,
            suffix: str,
        ) -> str:
            parts: list[str] = []
            if axis_index > 0:
                prefix_name = ctx.add_const_tensor(
                    f"{output_name}_stridedslice_{suffix}_prefix",
                    np.asarray([int(fill_value) for _ in range(axis_index)], dtype=np.int32),
                )
                parts.append(prefix_name)
            parts.append(dynamic_len1_name)
            if axis_index + 1 < rank:
                suffix_name = ctx.add_const_tensor(
                    f"{output_name}_stridedslice_{suffix}_suffix",
                    np.asarray(
                        [int(fill_value) for _ in range(rank - axis_index - 1)],
                        dtype=np.int32,
                    ),
                )
                parts.append(suffix_name)
            if len(parts) == 1:
                return parts[0]
            vector_name = ctx.add_intermediate_tensor(
                f"{output_name}_stridedslice_{suffix}_full",
                dtype="INT32",
                shape=[int(rank)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=parts,
                    outputs=[vector_name],
                    options={"axis": 0, "fusedActivationFunction": "NONE"},
                )
            )
            return vector_name

        if dynamic_start_input_name != "":
            begin_name = _prepare_dynamic_len1_i32(
                dynamic_name=dynamic_start_input_name,
                suffix="begin",
            )
            begin_name = _compose_rank_vector_with_dynamic_axis(
                dynamic_len1_name=begin_name,
                axis_index=axis,
                fill_value=0,
                suffix="begin",
            )
        else:
            begin_vec = [0 for _ in range(rank)]
            start_const = int(starts[0])
            known_axis_dim = int(input_shape[axis]) if int(axis) < int(len(input_shape)) else -1
            if start_const < 0:
                if known_axis_dim > 0:
                    start_const += int(known_axis_dim)
                else:
                    raise NotImplementedError(
                        "Slice negative constant start with dynamic shape is not supported "
                        f"in this dynamic starts/ends path. op={node.name} start={starts[0]}"
                    )
            begin_vec[axis] = int(start_const)
            begin_name = ctx.add_const_tensor(
                f"{output_name}_stridedslice_begin",
                np.asarray(begin_vec, dtype=np.int32),
            )

        if dynamic_end_input_name != "":
            end_name = _prepare_dynamic_len1_i32(
                dynamic_name=dynamic_end_input_name,
                suffix="end",
            )
            end_name = _compose_rank_vector_with_dynamic_axis(
                dynamic_len1_name=end_name,
                axis_index=axis,
                fill_value=int32_max,
                suffix="end",
            )
        else:
            end_vec = [int32_max for _ in range(rank)]
            end_const = int(ends[0])
            known_axis_dim = int(input_shape[axis]) if int(axis) < int(len(input_shape)) else -1
            if end_const < 0 and known_axis_dim > 0:
                end_const += int(known_axis_dim)
            end_vec[axis] = int(end_const)
            end_name = ctx.add_const_tensor(
                f"{output_name}_stridedslice_end",
                np.asarray(end_vec, dtype=np.int32),
            )

        strides = [1 for _ in range(rank)]
        strides[axis] = int(step)
        strides_name = ctx.add_const_tensor(
            f"{output_name}_stridedslice_strides",
            np.asarray(strides, dtype=np.int32),
        )
        end_mask = 0
        for axis_idx in range(rank):
            if int(axis_idx) != int(axis):
                end_mask |= (1 << int(axis_idx))
        ctx.add_operator(
            OperatorIR(
                op_type="STRIDED_SLICE",
                inputs=[input_name, begin_name, end_name, strides_name],
                outputs=[output_name],
                options={
                    "beginMask": 0,
                    "endMask": int(end_mask),
                    "ellipsisMask": 0,
                    "newAxisMask": 0,
                    "shrinkAxisMask": 0,
                    "offset": False,
                },
            )
        )
        return

    if dynamic_end_input_name != "":
        dynamic_prefix_len = len(starts)
        dynamic_end_shape = [int(v) for v in ctx.get_tensor_shape(dynamic_end_input_name)]
        dynamic_end_len = int(dynamic_end_shape[0]) if len(dynamic_end_shape) == 1 else -1
        dynamic_end_len_ok = (
            len(dynamic_end_shape) == 1
            and (dynamic_end_len <= 0 or dynamic_end_len == dynamic_prefix_len)
        )
        axes_are_prefix = normalized_axes == [int(v) for v in range(len(normalized_axes))]
        starts_non_negative = all(int(v) >= 0 for v in starts)
        steps_positive = all(int(v) > 0 for v in steps)
        if not (
            rank >= 1
            and dynamic_prefix_len >= 1
            and len(normalized_axes) == dynamic_prefix_len
            and len(steps) == dynamic_prefix_len
            and dynamic_prefix_len <= rank
            and dynamic_end_len_ok
            and axes_are_prefix
            and starts_non_negative
            and steps_positive
        ):
            raise NotImplementedError(
                "Slice with dynamic end is supported only for prefix-axis "
                "slicing (axes=[0..k-1], start>=0, step>0) in flatbuffer_direct. "
                f"op={node.name} rank={rank} starts={starts} axes={axes} steps={steps}"
            )
        dynamic_end_name = dynamic_end_input_name
        dynamic_end_dtype = str(ctx.get_tensor_dtype(dynamic_end_input_name)).upper()
        if len(dynamic_end_shape) != 1 or (
            dynamic_end_len > 0 and dynamic_end_len != dynamic_prefix_len
        ):
            reshape_shape_name = ctx.add_const_tensor(
                f"{output_name}_stridedslice_end_shape",
                np.asarray([int(dynamic_prefix_len)], dtype=np.int32),
            )
            reshaped_end_name = ctx.add_intermediate_tensor(
                f"{output_name}_stridedslice_end_flat",
                dtype=dynamic_end_dtype,
                shape=[int(dynamic_prefix_len)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[dynamic_end_input_name, reshape_shape_name],
                    outputs=[reshaped_end_name],
                    options={"newShape": [int(dynamic_prefix_len)]},
                )
            )
            dynamic_end_name = reshaped_end_name
            dynamic_end_shape = [int(dynamic_prefix_len)]
        if dynamic_end_dtype != "INT32":
            dynamic_end_i32 = ctx.add_intermediate_tensor(
                f"{output_name}_stridedslice_end_i32",
                dtype="INT32",
                shape=dynamic_end_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[dynamic_end_name],
                    outputs=[dynamic_end_i32],
                    options={
                        "inDataType": dynamic_end_dtype,
                        "outDataType": "INT32",
                    },
                )
            )
            dynamic_end_name = dynamic_end_i32

        if dynamic_prefix_len < rank:
            tail_dims = [int(np.iinfo(np.int32).max) for _ in range(rank - dynamic_prefix_len)]
            tail_name = ctx.add_const_tensor(
                f"{output_name}_stridedslice_end_tail",
                np.asarray(tail_dims, dtype=np.int32),
            )
            end_full_name = ctx.add_intermediate_tensor(
                f"{output_name}_stridedslice_end_full",
                dtype="INT32",
                shape=[int(rank)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=[dynamic_end_name, tail_name],
                    outputs=[end_full_name],
                    options={"axis": 0, "fusedActivationFunction": "NONE"},
                )
            )
            dynamic_end_name = end_full_name

        begin = [0 for _ in range(rank)]
        strides = [1 for _ in range(rank)]
        for idx, axis in enumerate(normalized_axes):
            begin[axis] = int(starts[idx])
            strides[axis] = int(steps[idx])
        end_mask = 0
        normalized_axis_set = set(int(v) for v in normalized_axes)
        for axis in range(rank):
            if axis not in normalized_axis_set:
                end_mask |= (1 << axis)
        begin_name = ctx.add_const_tensor(
            f"{output_name}_stridedslice_begin",
            np.asarray(begin, dtype=np.int32),
        )
        strides_name = ctx.add_const_tensor(
            f"{output_name}_stridedslice_strides",
            np.asarray(strides, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="STRIDED_SLICE",
                inputs=[input_name, begin_name, dynamic_end_name, strides_name],
                outputs=[output_name],
                options={
                    "beginMask": 0,
                    "endMask": int(end_mask),
                    "ellipsisMask": 0,
                    "newAxisMask": 0,
                    "shrinkAxisMask": 0,
                    "offset": False,
                },
            )
        )
        return

    if any(int(step) < 0 for step in steps):
        is_supported_full_reverse = (
            len(starts) == 1
            and len(ends) == 1
            and len(normalized_axes) == 1
            and len(steps) == 1
            and int(steps[0]) == -1
            and int(starts[0]) == -1
            and int(ends[0]) <= -int(np.iinfo(np.int32).max)
        )
        if not is_supported_full_reverse:
            raise NotImplementedError(
                f"Slice negative step is not supported for flatbuffer_direct. op={node.name} "
                f"starts={starts} ends={ends} axes={axes} steps={steps}"
            )
        axis_name = ctx.add_const_tensor(
            f"{output_name}_reverse_axis",
            np.asarray([int(normalized_axes[0])], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="REVERSE_V2",
                inputs=[input_name, axis_name],
                outputs=[output_name],
            )
        )
        return

    known_dim_flags = [
        (
            axis < len(input_signature)
            and int(input_signature[axis]) >= 0
            and axis < len(input_shape)
            and int(input_shape[axis]) > 0
        )
        for axis in range(rank)
    ]
    begin = [0 for _ in range(rank)]
    end_for_strided = [
        int(input_shape[axis]) if known_dim_flags[axis] else int(np.iinfo(np.int32).max)
        for axis in range(rank)
    ]
    strides_for_strided = [1 for _ in range(rank)]
    size = [int(input_shape[axis]) if known_dim_flags[axis] else -1 for axis in range(rank)]
    large_int = int(np.iinfo(np.int64).max // 2)
    int32_min = int(np.iinfo(np.int32).min)
    int32_max = int(np.iinfo(np.int32).max)
    use_strided_slice = any(not flag for flag in known_dim_flags)

    for idx, axis in enumerate(normalized_axes):
        step = int(steps[idx])
        if step == 0:
            raise NotImplementedError(
                f"Slice step must not be 0 for flatbuffer_direct. op={node.name} step={step}"
            )
        if step != 1:
            use_strided_slice = True
        start = int(starts[idx])
        end = int(ends[idx])
        dim_is_known = (
            axis < len(input_signature)
            and int(input_signature[axis]) >= 0
        )
        dim = int(input_shape[axis]) if dim_is_known and axis < len(input_shape) else -1

        if dim > 0:
            if start < 0:
                start += dim
            if end < 0:
                end += dim
            start = max(0, min(start, dim))
            end = max(0, min(end, dim))
            begin[axis] = int(start)
            end_for_strided[axis] = int(end)
            strides_for_strided[axis] = int(step)
            if step == 1:
                size[axis] = int(max(end - start, 0))
            else:
                size[axis] = -1
        else:
            if start < 0 or end < 0:
                # Keep negative indices and defer resolution to runtime via STRIDED_SLICE.
                use_strided_slice = True
            begin[axis] = int(max(min(start, int32_max), int32_min))
            if end >= large_int:
                end_for_strided[axis] = int32_max
                size[axis] = -1
            elif end <= -large_int:
                end_for_strided[axis] = int32_min
                size[axis] = -1
            else:
                end_for_strided[axis] = int(max(min(end, int32_max), int32_min))
                if step == 1 and start >= 0 and end >= 0:
                    size[axis] = int(max(end - start, 0))
                else:
                    size[axis] = -1
            strides_for_strided[axis] = int(step)

    output_shape_hint = [int(v) for v in ctx.get_tensor_shape(output_name)]
    if (
        not bool(use_strided_slice)
        and len(output_shape_hint) == int(rank)
        and all(int(v) > 0 for v in output_shape_hint)
        and all(int(step) == 1 for step in steps)
    ):
        # Prefer graph output metadata for static step=1 SLICE when available.
        # This avoids propagating stale input-shape metadata into size constants.
        size = [int(v) for v in output_shape_hint]

    strided_slice_options: dict[str, int | bool] = {
        "beginMask": 0,
        "endMask": 0,
        "ellipsisMask": 0,
        "newAxisMask": 0,
        "shrinkAxisMask": 0,
        "offset": False,
    }
    if _decompose_high_rank_slice_like(
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
        preferred_slice_axes=[int(v) for v in list(normalized_axes)],
        use_strided_slice=bool(use_strided_slice),
        begin=[int(v) for v in list(begin)],
        size=[int(v) for v in list(size)],
        end_for_strided=[int(v) for v in list(end_for_strided)],
        strides_for_strided=[int(v) for v in list(strides_for_strided)],
        strided_slice_options=dict(strided_slice_options),
    ):
        return

    _emit_slice_or_stridedslice(
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
        use_strided_slice=bool(use_strided_slice),
        begin=[int(v) for v in list(begin)],
        size=[int(v) for v in list(size)],
        end_for_strided=[int(v) for v in list(end_for_strided)],
        strides_for_strided=[int(v) for v in list(strides_for_strided)],
        strided_slice_options=dict(strided_slice_options),
        name_prefix=output_name,
    )


def _parse_split_sizes(
    *,
    node: Any,
    ctx: Any,
    input_axis_dim: int,
    output_count: int,
) -> list[int]:
    split_sizes: list[int] | None = None
    if len(node.inputs) >= 2:
        split_arr = ctx.get_constant_array(node.inputs[1].name)
        if split_arr is None:
            raise NotImplementedError(
                f"Split split tensor must be constant for flatbuffer_direct. op={node.name}"
            )
        split_sizes = [int(v) for v in np.asarray(split_arr).reshape(-1).tolist()]
    elif "split" in node.attrs:
        split_attr = node.attrs.get("split")
        if isinstance(split_attr, (list, tuple, np.ndarray)):
            split_sizes = [int(v) for v in np.asarray(split_attr).reshape(-1).tolist()]
        elif split_attr is not None:
            split_sizes = [int(split_attr)]

    if split_sizes is None or len(split_sizes) == 0:
        if input_axis_dim <= 0 or input_axis_dim % int(output_count) != 0:
            raise NotImplementedError(
                "Split requires explicit split sizes when axis dimension is unknown "
                "or not divisible by number of outputs in flatbuffer_direct. "
                f"op={node.name} axis_dim={input_axis_dim} outputs={output_count}"
            )
        each = int(input_axis_dim // int(output_count))
        split_sizes = [int(each) for _ in range(int(output_count))]

    if len(split_sizes) != int(output_count):
        raise NotImplementedError(
            f"Split split size count must match outputs. op={node.name} "
            f"split_len={len(split_sizes)} outputs={output_count}"
        )
    # If shape metadata is stale, axis_dim may not match explicit split sizes.
    # In that case, prefer explicit split sizes from ONNX and continue.
    if any(int(v) < 0 for v in split_sizes):
        raise NotImplementedError(
            f"Split split sizes must be non-negative. op={node.name} split={split_sizes}"
        )
    return [int(v) for v in split_sizes]


def _find_producer_op(ctx: Any, tensor_name: str) -> Any:
    for op in reversed(ctx.model_ir.operators):
        if str(tensor_name) in set(str(v) for v in op.outputs):
            return op
    return None


def _find_onnx_producer_node(ctx: Any, tensor_name: str) -> Any:
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None or getattr(onnx_model, "graph", None) is None:
        return None
    for graph_node in onnx_model.graph.node:
        for output_name in graph_node.output:
            if str(output_name) == str(tensor_name):
                return graph_node
    return None


def _is_optional_onnx_tensor(ctx: Any, tensor_name: str) -> bool:
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None or getattr(onnx_model, "graph", None) is None:
        return False
    graph = onnx_model.graph
    value_infos = list(graph.input) + list(graph.value_info) + list(graph.output)
    for value_info in value_infos:
        if str(getattr(value_info, "name", "")) != str(tensor_name):
            continue
        type_proto = getattr(value_info, "type", None)
        if type_proto is not None and hasattr(type_proto, "HasField"):
            return bool(type_proto.HasField("optional_type"))
    return False


def _infer_optional_has_element_result(ctx: Any, input_name: str) -> bool | None:
    producer_node = _find_onnx_producer_node(ctx, input_name)
    if producer_node is None:
        return None
    producer_op = str(getattr(producer_node, "op_type", ""))
    if producer_op == "Optional":
        producer_inputs = [str(v) for v in list(getattr(producer_node, "input", []))]
        return bool(any(v != "" for v in producer_inputs))
    if producer_op == "OptionalGetElement":
        return True
    return None


def _infer_static_template_from_dynamic_reshape_shape_input(
    *,
    ctx: Any,
    shape_input_name: str,
) -> list[int] | None:
    """
    Infer static reshape template from:
      CONCATENATION(axis=0, [const_prefix, SHAPE(x)])
    """
    concat_op = _find_producer_op(ctx, shape_input_name)
    if concat_op is None or str(concat_op.op_type) != "CONCATENATION":
        return None
    if int(concat_op.options.get("axis", -1)) != 0 or len(concat_op.inputs) != 2:
        return None

    prefix_tensor = ctx.model_ir.tensors.get(str(concat_op.inputs[0]), None)
    if prefix_tensor is None or prefix_tensor.data is None:
        return None
    prefix_vals = [int(v) for v in np.asarray(prefix_tensor.data).reshape(-1).tolist()]
    if len(prefix_vals) == 0 or any(int(v) <= 0 for v in prefix_vals):
        return None

    shape_output_name = str(concat_op.inputs[1])
    shape_op = _find_producer_op(ctx, shape_output_name)
    if shape_op is None or str(shape_op.op_type) != "SHAPE" or len(shape_op.inputs) != 1:
        return None

    source_tensor = ctx.model_ir.tensors.get(str(shape_op.inputs[0]), None)
    if source_tensor is None:
        return None
    source_signature = (
        [int(v) for v in list(source_tensor.shape_signature)]
        if source_tensor.shape_signature is not None
        else [int(v) for v in list(source_tensor.shape)]
    )
    if len(source_signature) == 0:
        return None

    template = [int(v) for v in prefix_vals]
    template.extend(int(v) if int(v) >= 0 else -1 for v in source_signature)
    if any(int(v) == 0 for v in template):
        return None
    if sum(1 for v in template if int(v) < 0) > 1:
        return None
    if not any(int(v) < 0 for v in template):
        template[-1] = -1
    return [int(v) for v in template]


def build_split_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_names = [o.name for o in node.outputs]
    if len(output_names) == 0:
        return

    ctx.ensure_tensor(input_name)
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    rank = len(input_shape)
    axis = _normalize_axis(int(node.attrs.get("axis", 0)), rank, op_name=node.name)
    input_axis_dim = int(input_shape[axis]) if axis < len(input_shape) else -1
    split_sizes = _parse_split_sizes(
        node=node,
        ctx=ctx,
        input_axis_dim=input_axis_dim,
        output_count=len(output_names),
    )

    input_tensor = ctx.model_ir.tensors[input_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )

    offset = 0
    for out_idx, output_name in enumerate(output_names):
        ctx.ensure_tensor(output_name)
        output_tensor = ctx.model_ir.tensors[output_name]
        output_tensor.dtype = input_tensor.dtype
        output_tensor.quantization = _clone_quantization(input_tensor.quantization)

        out_shape = list(input_shape)
        out_shape[axis] = int(split_sizes[out_idx])
        output_tensor.shape = [int(v) for v in out_shape]
        output_signature = list(input_signature)
        if axis < len(output_signature):
            output_signature[axis] = int(split_sizes[out_idx])
        output_tensor.shape_signature = [int(v) for v in output_signature]

        begin = [0 for _ in range(rank)]
        begin[axis] = int(offset)
        # Split should preserve all non-split axes even when static shape
        # metadata is stale. Use -1 to consume full extent on those axes.
        size = [-1 for _ in range(rank)]
        size[axis] = int(split_sizes[out_idx])
        offset += int(split_sizes[out_idx])

        begin_name = ctx.add_const_tensor(
            f"{output_name}_split_begin",
            np.asarray(begin, dtype=np.int32),
        )
        size_name = ctx.add_const_tensor(
            f"{output_name}_split_size",
            np.asarray(size, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[input_name, begin_name, size_name],
                outputs=[output_name],
            )
        )


def _resolve_reshape_shape_with_static_dims(
    *,
    new_shape: list[int],
    input_tensor: Any,
    output_tensor: Any,
    allowzero: bool,
) -> list[int]:
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_tensor.shape)
    )
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_tensor.shape)
    )
    if len(output_signature) == len(new_shape) and all(int(dim) > 0 for dim in output_signature):
        if len(input_signature) > 0 and all(int(dim) > 0 for dim in input_signature):
            output_product = int(np.prod(np.asarray(output_signature, dtype=np.int64)))
            input_product = int(np.prod(np.asarray(input_signature, dtype=np.int64)))
            if output_product == input_product:
                return [int(v) for v in output_tensor.shape]
        else:
            return [int(v) for v in output_tensor.shape]

    minus_one_indices = [idx for idx, dim in enumerate(new_shape) if int(dim) == -1]
    resolved_shape = [int(v) for v in new_shape]

    if not allowzero:
        for idx, dim in enumerate(resolved_shape):
            if int(dim) != 0:
                continue
            if idx >= len(input_signature):
                return [int(v) for v in new_shape]
            in_dim = int(input_signature[idx])
            if in_dim <= 0:
                return [int(v) for v in new_shape]
            resolved_shape[idx] = in_dim

    if len(minus_one_indices) != 1:
        return [int(v) for v in resolved_shape]
    if len(input_signature) == 0 or any(int(dim) <= 0 for dim in input_signature):
        return [int(v) for v in resolved_shape]

    known_product = 1
    for idx, raw_dim in enumerate(resolved_shape):
        dim = int(raw_dim)
        if dim == -1:
            continue
        if dim <= 0:
            return [int(v) for v in resolved_shape]
        known_product *= dim
    if known_product <= 0:
        return [int(v) for v in resolved_shape]

    input_product = int(np.prod(np.asarray(input_signature, dtype=np.int64)))
    if input_product <= 0 or input_product % known_product != 0:
        return [int(v) for v in resolved_shape]
    inferred = int(input_product // known_product)
    if inferred <= 0:
        return [int(v) for v in resolved_shape]
    resolved_shape[minus_one_indices[0]] = inferred
    return resolved_shape


def _rewrite_dynamic_reshape_shape_allowzero_copy_dim0(
    *,
    ctx: Any,
    input_name: str,
    shape_input_name: str,
    output_name: str,
) -> str:
    """
    ONNX Reshape with allowzero=0 treats 0 in shape tensor as "copy from input".
    TFLite RESHAPE does not support that semantic, so rewrite leading dim0 at runtime:
      shape[0] = (shape[0] == 0) ? SHAPE(input)[0] : shape[0]
    """
    shape_input_shape = [int(v) for v in ctx.get_tensor_shape(shape_input_name)]
    if len(shape_input_shape) != 1:
        return shape_input_name

    first_begin_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape_dim0_begin",
        np.asarray([0], dtype=np.int32),
    )
    first_size_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape_dim0_size",
        np.asarray([1], dtype=np.int32),
    )
    shape_dim0_name = ctx.add_intermediate_tensor(
        f"{output_name}_reshape_shape_dim0",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[shape_input_name, first_begin_name, first_size_name],
            outputs=[shape_dim0_name],
        )
    )

    input_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_reshape_input_shape",
        dtype="INT32",
        shape=[max(int(len(ctx.get_tensor_shape(input_name))), 1)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[input_shape_name],
            options={"outType": "INT32"},
        )
    )
    input_dim0_name = ctx.add_intermediate_tensor(
        f"{output_name}_reshape_input_dim0",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[input_shape_name, first_begin_name, first_size_name],
            outputs=[input_dim0_name],
        )
    )

    zero_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape_dim0_zero",
        np.asarray([0], dtype=np.int32),
    )
    dim0_is_zero_name = ctx.add_intermediate_tensor(
        f"{output_name}_reshape_shape_dim0_is_zero",
        dtype="BOOL",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="EQUAL",
            inputs=[shape_dim0_name, zero_name],
            outputs=[dim0_is_zero_name],
            options={},
        )
    )

    fixed_dim0_name = ctx.add_intermediate_tensor(
        f"{output_name}_reshape_shape_dim0_fixed",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT",
            inputs=[dim0_is_zero_name, input_dim0_name, shape_dim0_name],
            outputs=[fixed_dim0_name],
            options={},
        )
    )

    tail_begin_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape_tail_begin",
        np.asarray([1], dtype=np.int32),
    )
    tail_name = ctx.add_intermediate_tensor(
        f"{output_name}_reshape_shape_tail",
        dtype="INT32",
        shape=[1],
    )
    # Keep shape tail dynamic: shape[1:] works for both known and unknown length vectors.
    tail_end_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape_tail_end",
        np.asarray([0], dtype=np.int32),
    )
    tail_stride_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape_tail_stride",
        np.asarray([1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="STRIDED_SLICE",
            inputs=[shape_input_name, tail_begin_name, tail_end_name, tail_stride_name],
            outputs=[tail_name],
            options={
                "beginMask": 0,
                "endMask": 1,
                "ellipsisMask": 0,
                "newAxisMask": 0,
                "shrinkAxisMask": 0,
                "offset": False,
            },
        )
    )

    fixed_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_reshape_shape_fixed",
        dtype="INT32",
        shape=[1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[fixed_dim0_name, tail_name],
            outputs=[fixed_shape_name],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )
    return fixed_shape_name


def build_reshape_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    shape_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    allowzero = bool(node.attrs.get("allowzero", 0))
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    shape_values = ctx.get_constant_array(shape_name)
    raw_new_shape: list[int] = []
    new_shape: list[int] = []
    reshape_shape_input_name = shape_name

    if shape_values is not None:
        raw_new_shape = [int(v) for v in np.asarray(shape_values).reshape(-1).tolist()]
        new_shape = _resolve_reshape_shape_with_static_dims(
            new_shape=list(raw_new_shape),
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            allowzero=allowzero,
        )
        if len(new_shape) > 0 and all(int(dim) >= 0 for dim in new_shape):
            output_tensor.shape = [int(dim) for dim in new_shape]
            output_tensor.shape_signature = [int(dim) for dim in new_shape]
        reshape_shape_input_name = ctx.add_const_tensor(
            f"{output_name}_reshape_shape",
            np.asarray(new_shape, dtype=np.int32),
        )
    else:
        inferred_template = _infer_static_template_from_dynamic_reshape_shape_input(
            ctx=ctx,
            shape_input_name=shape_name,
        )
        if inferred_template is not None:
            new_shape = [int(v) for v in inferred_template]
            raw_new_shape = [int(v) for v in inferred_template]
            output_tensor.shape_signature = [int(v) for v in inferred_template]
            output_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in inferred_template]
            reshape_shape_input_name = ctx.add_const_tensor(
                f"{output_name}_reshape_shape",
                np.asarray(inferred_template, dtype=np.int32),
            )
        else:
            shape_dtype = str(ctx.get_tensor_dtype(shape_name)).upper()
            if shape_dtype not in {"INT32", "INT64"}:
                raise NotImplementedError(
                    f"Reshape dynamic shape input dtype must be INT32/INT64. "
                    f"op={node.name} tensor={shape_name} dtype={shape_dtype}"
                )
            if shape_dtype == "INT64":
                reshape_shape_input_name = ctx.add_intermediate_tensor(
                    f"{output_name}_reshape_shape_i32",
                    dtype="INT32",
                    shape=[int(v) for v in ctx.get_tensor_shape(shape_name)],
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[shape_name],
                        outputs=[reshape_shape_input_name],
                        options={
                            "inDataType": "INT64",
                            "outDataType": "INT32",
                        },
                    )
                )
            if not bool(allowzero):
                reshape_shape_input_name = _rewrite_dynamic_reshape_shape_allowzero_copy_dim0(
                    ctx=ctx,
                    input_name=input_name,
                    shape_input_name=reshape_shape_input_name,
                    output_name=output_name,
                )
            shape_vector_shape = [int(v) for v in ctx.get_tensor_shape(reshape_shape_input_name)]
            if len(shape_vector_shape) == 1 and int(shape_vector_shape[0]) > 0:
                output_rank = int(shape_vector_shape[0])
                output_tensor.shape = [1 for _ in range(output_rank)]
                output_tensor.shape_signature = [-1 for _ in range(output_rank)]
            # Dynamic shape input drives runtime reshape dimensions. Keep options empty
            # to avoid static-shape rewrite passes from clobbering this tensor.
            new_shape = []
            raw_new_shape = []

    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, reshape_shape_input_name],
            outputs=[output_name],
            options={
                "newShape": new_shape,
                "onnxRawNewShape": raw_new_shape,
                "allowZero": bool(allowzero),
            },
        )
    )


def build_transpose_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    perm = None
    if len(node.inputs) >= 2:
        perm_name = node.inputs[1].name
        perm = ctx.get_constant_array(perm_name)
    if perm is None and "perm" in node.attrs:
        attr_perm = node.attrs.get("perm")
        if isinstance(attr_perm, (list, tuple)):
            perm = np.asarray([int(v) for v in attr_perm], dtype=np.int32)
        elif attr_perm is not None:
            perm = np.asarray([int(attr_perm)], dtype=np.int32)
    if perm is None:
        input_rank = len(ctx.get_tensor_shape(input_name))
        perm = np.asarray(list(reversed(range(input_rank))), dtype=np.int32)

    if perm is None:
        raise NotImplementedError(
            f"Transpose permutation must be resolvable for flatbuffer_direct. op={node.name}"
        )
    make_transpose(
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
        perm_values=[int(v) for v in np.asarray(perm, dtype=np.int32).reshape(-1).tolist()],
    )


def build_concat_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)

    output_shape = ctx.get_tensor_shape(output_name)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += len(output_shape)

    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=input_names,
            outputs=[output_name],
            options={
                "axis": int(axis),
                "fusedActivationFunction": "NONE",
            },
        )
    )


def _string_normalizer_stopwords(raw_stopwords: Any) -> List[str]:
    if raw_stopwords is None:
        return []
    if isinstance(raw_stopwords, str):
        return [str(raw_stopwords)]
    values: List[str] = []
    for item in list(raw_stopwords):
        values.append(str(item))
    return values


def _string_normalizer_apply_case(tokens: np.ndarray, case_change_action: str) -> np.ndarray:
    action = str(case_change_action).strip().upper()
    if action not in {"LOWER", "UPPER"}:
        return np.asarray(tokens, dtype=object)
    transformed = []
    for item in np.asarray(tokens, dtype=object).reshape(-1).tolist():
        text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        transformed.append(text.lower() if action == "LOWER" else text.upper())
    return np.asarray(transformed, dtype=object)


def _string_normalizer_stopword_mask(
    *,
    tokens: np.ndarray,
    stopwords: List[str],
    is_case_sensitive: bool,
) -> np.ndarray:
    if len(stopwords) == 0:
        return np.ones(np.asarray(tokens).shape, dtype=np.bool_)

    token_texts = []
    for item in np.asarray(tokens, dtype=object).reshape(-1).tolist():
        token_texts.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    stopword_texts = [str(v) for v in stopwords]

    if not bool(is_case_sensitive):
        token_texts = [t.lower() for t in token_texts]
        stopword_texts = [s.lower() for s in stopword_texts]

    stopword_set = set(stopword_texts)
    mask_list = [text not in stopword_set for text in token_texts]
    return np.asarray(mask_list, dtype=np.bool_)


def _evaluate_string_normalizer_constant(
    *,
    input_values: np.ndarray,
    case_change_action: str,
    is_case_sensitive: bool,
    stopwords: List[str],
) -> np.ndarray:
    values = np.asarray(input_values, dtype=object)
    if values.ndim <= 1:
        flat_tokens = values.reshape(-1)
        mask = _string_normalizer_stopword_mask(
            tokens=flat_tokens,
            stopwords=stopwords,
            is_case_sensitive=is_case_sensitive,
        )
        filtered = flat_tokens[mask]
        filtered = _string_normalizer_apply_case(filtered, case_change_action)
        if filtered.size == 0:
            return np.asarray([""], dtype=object)
        return np.asarray(filtered, dtype=object)

    row = np.asarray(values[0], dtype=object).reshape(-1)
    mask = _string_normalizer_stopword_mask(
        tokens=row,
        stopwords=stopwords,
        is_case_sensitive=is_case_sensitive,
    )
    filtered = row[mask]
    filtered = _string_normalizer_apply_case(filtered, case_change_action)
    filtered = np.expand_dims(np.asarray(filtered, dtype=object), axis=0)
    if filtered.size == 0:
        return np.asarray([[""]], dtype=object)
    return np.asarray(filtered, dtype=object)


def _build_string_normalizer_keep_mask(
    *,
    ctx: Any,
    input_name: str,
    stopwords: List[str],
    base_name: str,
) -> str:
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    matched_name: Optional[str] = None
    for idx, stopword in enumerate(stopwords):
        stopword_name = ctx.add_const_tensor(
            f"{base_name}_stopword_{idx}",
            np.asarray(stopword, dtype=object),
        )
        eq_name = ctx.add_intermediate_tensor(
            f"{base_name}_stopword_eq_{idx}",
            dtype="BOOL",
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EQUAL",
                inputs=[input_name, stopword_name],
                outputs=[eq_name],
            )
        )
        if matched_name is None:
            matched_name = eq_name
            continue
        merged_name = ctx.add_intermediate_tensor(
            f"{base_name}_stopword_or_{idx}",
            dtype="BOOL",
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="LOGICAL_OR",
                inputs=[matched_name, eq_name],
                outputs=[merged_name],
            )
        )
        matched_name = merged_name

    if matched_name is None:
        raise NotImplementedError("StringNormalizer stopwords mask generation requires non-empty stopwords.")

    keep_name = ctx.add_intermediate_tensor(
        f"{base_name}_keep_mask",
        dtype="BOOL",
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOGICAL_NOT",
            inputs=[matched_name],
            outputs=[keep_name],
        )
    )
    return keep_name


def _build_string_normalizer_rank1_runtime(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    stopwords: List[str],
    base_name: str,
) -> None:
    keep_name = _build_string_normalizer_keep_mask(
        ctx=ctx,
        input_name=input_name,
        stopwords=stopwords,
        base_name=base_name,
    )
    where_name = ctx.add_intermediate_tensor(
        f"{base_name}_where_indices",
        dtype="INT64",
        shape=[-1, 1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="WHERE",
            inputs=[keep_name],
            outputs=[where_name],
        )
    )
    flat_indices_name = ctx.add_intermediate_tensor(
        f"{base_name}_flat_indices",
        dtype="INT64",
        shape=[-1],
    )
    flat_shape_name = ctx.add_const_tensor(
        f"{base_name}_flat_indices_shape",
        np.asarray([-1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[where_name, flat_shape_name],
            outputs=[flat_indices_name],
            options={"newShape": [-1]},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[input_name, flat_indices_name],
            outputs=[output_name],
            options={
                "axis": 0,
                "batchDims": 0,
            },
        )
    )


def build_string_normalizer_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    case_change_action = str(node.attrs.get("case_change_action", "NONE")).strip().upper()
    is_case_sensitive = bool(node.attrs.get("is_case_sensitive", 1))
    locale = str(node.attrs.get("locale", "en_US")).strip()
    stopwords = _string_normalizer_stopwords(node.attrs.get("stopwords", []))

    constant_input = ctx.get_constant_array(input_name)
    if constant_input is not None:
        normalized = _evaluate_string_normalizer_constant(
            input_values=constant_input,
            case_change_action=case_change_action,
            is_case_sensitive=is_case_sensitive,
            stopwords=stopwords,
        )
        out_tensor = ctx.model_ir.tensors[output_name]
        out_tensor.dtype = "STRING"
        out_tensor.data = np.asarray(normalized, dtype=object)
        out_tensor.shape, out_tensor.shape_signature = normalize_onnx_shape(
            [int(v) for v in list(out_tensor.data.shape)]
        )
        ctx.constants[output_name] = np.asarray(out_tensor.data, dtype=object)
        return

    if locale not in {"", "en_US"}:
        raise NotImplementedError(
            f"StringNormalizer locale is not supported in builtin lowering. op={node.name} locale={locale}"
        )
    if case_change_action not in {"", "NONE"}:
        raise NotImplementedError(
            "StringNormalizer case_change_action LOWER/UPPER requires string transform support "
            f"that is unavailable in flatbuffer_direct builtin lowering. op={node.name}"
        )
    if len(stopwords) == 0:
        build_identity_op(node, ctx)
        return
    if not is_case_sensitive:
        raise NotImplementedError(
            "StringNormalizer case-insensitive stopword matching requires string transform support "
            f"that is unavailable in flatbuffer_direct builtin lowering. op={node.name}"
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_rank = len(input_shape)
    if input_rank == 1:
        _build_string_normalizer_rank1_runtime(
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
            stopwords=stopwords,
            base_name=output_name,
        )
        return

    if input_rank == 2:
        gather_index_name = ctx.add_const_tensor(
            f"{output_name}_row0_index",
            np.asarray([0], dtype=np.int32),
        )
        row2d_name = ctx.add_intermediate_tensor(
            f"{output_name}_row0_2d",
            dtype="STRING",
            shape=[1, int(input_shape[1]) if len(input_shape) >= 2 else 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[input_name, gather_index_name],
                outputs=[row2d_name],
                options={
                    "axis": 0,
                    "batchDims": 0,
                },
            )
        )
        row1d_name = ctx.add_intermediate_tensor(
            f"{output_name}_row0_1d",
            dtype="STRING",
            shape=[int(input_shape[1]) if len(input_shape) >= 2 else 1],
        )
        row_shape_name = ctx.add_const_tensor(
            f"{output_name}_row0_1d_shape",
            np.asarray([-1], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[row2d_name, row_shape_name],
                outputs=[row1d_name],
                options={"newShape": [-1]},
            )
        )
        filtered_row_name = ctx.add_intermediate_tensor(
            f"{output_name}_row0_filtered",
            dtype="STRING",
            shape=[1],
        )
        _build_string_normalizer_rank1_runtime(
            ctx=ctx,
            input_name=row1d_name,
            output_name=filtered_row_name,
            stopwords=stopwords,
            base_name=f"{output_name}_row0",
        )
        axis_name = ctx.add_const_tensor(
            f"{output_name}_expand_axis",
            np.asarray([0], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[filtered_row_name, axis_name],
                outputs=[output_name],
            )
        )
        return

    raise NotImplementedError(
        f"StringNormalizer builtin lowering supports only rank1/rank2 input. op={node.name} rank={input_rank}"
    )


def build_optional_has_element_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    has_element = _infer_optional_has_element_result(ctx, input_name)
    if has_element is None:
        if _is_optional_onnx_tensor(ctx, input_name):
            raise NotImplementedError(
                "OptionalHasElement with runtime-optional input is not supported in "
                f"flatbuffer_direct builtin lowering. op={node.name}"
            )
        has_element = True

    output_tensor = ctx.model_ir.tensors[output_name]
    output_tensor.dtype = "BOOL"
    output_tensor.quantization = None
    output_data = np.asarray(bool(has_element), dtype=np.bool_)
    output_tensor.data = output_data
    shape, signature = normalize_onnx_shape(list(output_data.shape))
    output_tensor.shape = [int(v) for v in shape]
    output_tensor.shape_signature = [int(v) for v in signature]
    ctx.constants[output_name] = output_data


def build_identity_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    output_shape = ctx.get_tensor_shape(output_name)
    shape_const = ctx.add_const_tensor(
        f"{output_name}_identity_shape",
        np.asarray(output_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape]},
        )
    )


def build_dropout_op(node: Any, ctx: Any) -> None:
    # Dropout is treated as inference-time no-op in flatbuffer_direct.
    build_identity_op(node, ctx)

    if len(node.outputs) < 2:
        return

    input_name = node.inputs[0].name
    mask_output_name = node.outputs[1].name
    if str(mask_output_name) == "":
        return

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(mask_output_name)
    _propagate_passthrough_shape_signature(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=mask_output_name,
    )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_rank = len(input_shape)
    shape_name = ctx.add_intermediate_tensor(
        f"{mask_output_name}_dropout_mask_shape",
        dtype="INT32",
        shape=[int(input_rank)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[shape_name],
        )
    )

    mask_dtype = str(ctx.get_tensor_dtype(mask_output_name)).upper()
    mask_np_dtype = _numpy_dtype_from_tflite_dtype(mask_dtype)
    fill_value = True if mask_dtype == "BOOL" else 1
    fill_value_name = ctx.add_const_tensor(
        f"{mask_output_name}_dropout_mask_fill_value",
        np.asarray(fill_value, dtype=mask_np_dtype),
    )
    fill_value_tensor = ctx.model_ir.tensors.get(fill_value_name, None)
    if fill_value_tensor is not None:
        fill_value_tensor.shape = []
        fill_value_tensor.shape_signature = []
    ctx.add_operator(
        OperatorIR(
            op_type="FILL",
            inputs=[shape_name, fill_value_name],
            outputs=[mask_output_name],
        )
    )


def build_eyelike_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    output_tensor = ctx.model_ir.tensors[output_name]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in output_shape]
    )
    if len(output_shape) != 2:
        raise NotImplementedError(
            f"EyeLike requires rank-2 output shape in flatbuffer_direct. op={node.name} output_shape={output_shape}"
        )
    if any(int(v) <= 0 for v in output_shape) or any(int(v) < 0 for v in output_signature):
        raise NotImplementedError(
            "EyeLike requires fully static positive shape in flatbuffer_direct. "
            f"op={node.name} output_shape={output_shape} output_signature={output_signature}"
        )

    rows = int(output_shape[0])
    cols = int(output_shape[1])
    k = int(node.attrs.get("k", 0))
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    output_np_dtype = _numpy_dtype_from_tflite_dtype(output_dtype)
    eye_const_name = ctx.add_const_tensor(
        f"{output_name}_eyelike_const",
        np.eye(rows, cols, k=k, dtype=output_np_dtype),
    )
    shape_const_name = ctx.add_const_tensor(
        f"{output_name}_eyelike_shape",
        np.asarray(output_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[eye_const_name, shape_const_name],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape]},
        )
    )
    output_tensor.shape = [int(v) for v in output_shape]
    output_tensor.shape_signature = [int(v) for v in output_signature]
    output_tensor.quantization = None


def build_trilu_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )
    _propagate_passthrough_shape_signature(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) < 2:
        raise NotImplementedError(
            f"Trilu requires rank >= 2 in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    m = int(input_shape[-2])
    n = int(input_shape[-1])
    if m <= 0 or n <= 0:
        raise NotImplementedError(
            "Trilu requires static positive matrix dimensions in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )
    k = 0
    if len(node.inputs) >= 2 and node.inputs[1].name != "":
        k_arr = ctx.get_constant_array(node.inputs[1].name)
        if k_arr is None:
            raise NotImplementedError(
                f"Trilu k input must be constant for flatbuffer_direct. op={node.name}"
            )
        k_flat = np.asarray(k_arr).reshape(-1)
        if int(k_flat.size) > 0:
            k = int(k_flat[0])

    upper = int(node.attrs.get("upper", 1)) != 0
    if upper:
        mask_2d = np.triu(np.ones((m, n), dtype=np.int32), k=int(k))
    else:
        mask_2d = np.tril(np.ones((m, n), dtype=np.int32), k=int(k))

    broadcast_shape = [1] * (len(input_shape) - 2) + [int(m), int(n)]
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    mask_name = ""
    op_type = "MUL"
    op_options: dict[str, Any] = {"fusedActivationFunction": "NONE"}
    if input_dtype == "BOOL":
        mask_name = ctx.add_const_tensor(
            f"{output_name}_trilu_mask",
            np.reshape(mask_2d.astype(np.bool_), broadcast_shape),
        )
        op_type = "LOGICAL_AND"
        op_options = {}
    else:
        mask_name = ctx.add_const_tensor(
            f"{output_name}_trilu_mask",
            np.reshape(
                mask_2d.astype(_numpy_dtype_from_tflite_dtype(input_dtype)),
                broadcast_shape,
            ),
        )
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[input_name, mask_name],
            outputs=[output_name],
            options=op_options,
        )
    )


def build_range_op(node: Any, ctx: Any) -> None:
    start_name = node.inputs[0].name
    limit_name = node.inputs[1].name
    delta_name = node.inputs[2].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(start_name)
    ctx.ensure_tensor(limit_name)
    ctx.ensure_tensor(delta_name)
    ctx.ensure_tensor(output_name)

    output_dtype = _prefer_int32_index_output_dtype(
        ctx=ctx,
        tensor_name=output_name,
        requested_dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
    )
    if output_dtype not in {
        "INT8", "INT16", "INT32", "INT64",
        "UINT8", "UINT16", "UINT32", "UINT64",
        "FLOAT16", "FLOAT32",
    }:
        raise NotImplementedError(
            f"Range output dtype is not supported in flatbuffer_direct. op={node.name} dtype={output_dtype}"
        )

    converted_inputs: list[str] = []
    for src_name, label in [
        (start_name, "start"),
        (limit_name, "limit"),
        (delta_name, "delta"),
    ]:
        src_dtype = str(ctx.get_tensor_dtype(src_name)).upper()
        if src_dtype == output_dtype:
            converted_inputs.append(src_name)
            continue
        cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_range_{label}_{output_dtype.lower()}",
            dtype=output_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(src_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[src_name],
                outputs=[cast_name],
                options={
                    "inDataType": src_dtype,
                    "outDataType": output_dtype,
                },
            )
        )
        converted_inputs.append(cast_name)

    scalar_inputs: list[str] = []
    for idx, src_name in enumerate(converted_inputs):
        src_shape = [int(v) for v in ctx.get_tensor_shape(src_name)]
        if len(src_shape) != 1 or int(src_shape[0]) != 1:
            raise NotImplementedError(
                "Range expects scalar-like inputs represented as shape [1] in flatbuffer_direct. "
                f"op={node.name} input_index={idx} input_shape={src_shape}"
            )
        scalar_name = ctx.add_intermediate_tensor(
            f"{output_name}_range_scalar_{idx}",
            dtype=output_dtype,
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[src_name],
                outputs=[scalar_name],
                options={"squeezeDims": [0]},
            )
        )
        scalar_inputs.append(scalar_name)

    ctx.add_operator(
        OperatorIR(
            op_type="RANGE",
            inputs=scalar_inputs,
            outputs=[output_name],
        )
    )


def build_random_normal_like_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    input_rank = len(ctx.get_tensor_shape(input_name))
    scale = float(node.attrs.get("scale", 1.0))
    mean = float(node.attrs.get("mean", 0.0))
    has_scale = not np.isclose(scale, 1.0)
    has_mean = not np.isclose(mean, 0.0)

    shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_random_shape",
        dtype="INT32",
        shape=[int(input_rank)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[shape_name],
            options={"outType": "INT32"},
        )
    )

    random_raw_name = output_name
    if has_scale or has_mean or output_dtype != compute_dtype:
        random_raw_name = ctx.add_intermediate_tensor(
            f"{output_name}_random_raw",
            dtype=compute_dtype,
            shape=output_shape,
        )
    seed_attr = node.attrs.get("seed", None)
    random_options: dict[str, Any] = {}
    if seed_attr is not None:
        seed = int(float(seed_attr))
        random_options["seed"] = int(seed)
        random_options["seed2"] = int(seed)
    ctx.add_operator(
        OperatorIR(
            op_type="RANDOM_STANDARD_NORMAL",
            inputs=[shape_name],
            outputs=[random_raw_name],
            options=random_options,
        )
    )

    current_name = random_raw_name
    if has_scale:
        scale_name = ctx.add_const_tensor(
            f"{output_name}_random_scale",
            np.asarray(scale, dtype=np.float16 if compute_dtype == "FLOAT16" else np.float32),
        )
        scaled_name = ctx.add_intermediate_tensor(
            f"{output_name}_random_scaled",
            dtype=compute_dtype,
            shape=output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[current_name, scale_name],
                outputs=[scaled_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        current_name = scaled_name

    if has_mean:
        mean_name = ctx.add_const_tensor(
            f"{output_name}_random_mean",
            np.asarray(mean, dtype=np.float16 if compute_dtype == "FLOAT16" else np.float32),
        )
        shifted_name = ctx.add_intermediate_tensor(
            f"{output_name}_random_shifted",
            dtype=compute_dtype,
            shape=output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[current_name, mean_name],
                outputs=[shifted_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        current_name = shifted_name

    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[current_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )
    elif current_name != output_name:
        output_shape_const = ctx.add_const_tensor(
            f"{output_name}_random_identity_shape",
            np.asarray(output_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[current_name, output_shape_const],
                outputs=[output_name],
                options={"newShape": [int(v) for v in output_shape]},
            )
        )


def _normalize_shape_slice_index(index: int, rank: int) -> int:
    normalized = int(index)
    if normalized < 0:
        normalized += int(rank)
    if normalized < 0:
        normalized = 0
    if normalized > int(rank):
        normalized = int(rank)
    return int(normalized)


def build_shape_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_rank = len(ctx.get_tensor_shape(input_name))
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    shape_output_dtype = _prefer_int32_index_output_dtype(
        ctx=ctx,
        tensor_name=output_name,
        requested_dtype=output_dtype,
    )
    if shape_output_dtype not in {"INT32", "INT64"}:
        shape_output_dtype = "INT32"

    start = int(node.attrs.get("start", 0))
    end = int(node.attrs.get("end", input_rank))
    start = _normalize_shape_slice_index(start, input_rank)
    end = _normalize_shape_slice_index(end, input_rank)
    if end < start:
        end = start

    output_tensor = ctx.model_ir.tensors[output_name]
    static_len = max(int(end - start), 0)
    output_tensor.shape = [int(static_len)]
    output_tensor.shape_signature = [int(static_len)]
    output_tensor.dtype = shape_output_dtype

    if start == 0 and end == int(input_rank):
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[input_name],
                outputs=[output_name],
                options={"outType": shape_output_dtype},
            )
        )
        return

    full_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_shape_full",
        dtype=shape_output_dtype,
        shape=[int(input_rank)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[full_shape_name],
            options={"outType": shape_output_dtype},
        )
    )
    begin_name = ctx.add_const_tensor(
        f"{output_name}_shape_slice_begin",
        np.asarray([int(start)], dtype=np.int32),
    )
    size_name = ctx.add_const_tensor(
        f"{output_name}_shape_slice_size",
        np.asarray([int(static_len)], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[full_shape_name, begin_name, size_name],
            outputs=[output_name],
        )
    )


def build_size_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor is not None and input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )
    input_rank = int(max(len(input_shape), len(input_signature)))

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = _prefer_int32_index_output_dtype(
        ctx=ctx,
        tensor_name=output_name,
        requested_dtype=output_dtype,
    )
    if compute_dtype not in {"INT32", "INT64"}:
        compute_dtype = "INT32"

    output_tensor = ctx.model_ir.tensors[output_name]
    output_tensor.shape = []
    output_tensor.shape_signature = []
    output_tensor.dtype = str(output_dtype)

    shape_out_name = ctx.add_intermediate_tensor(
        f"{output_name}_size_shape",
        dtype=compute_dtype,
        shape=[int(input_rank)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[shape_out_name],
            options={"outType": compute_dtype},
        )
    )

    size_core_name = output_name
    if output_dtype != compute_dtype:
        size_core_name = ctx.add_intermediate_tensor(
            f"{output_name}_size_core",
            dtype=compute_dtype,
            shape=[],
        )
    reduce_axes_name = ctx.add_const_tensor(
        f"{output_name}_size_axes",
        np.asarray([0], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="REDUCE_PROD",
            inputs=[shape_out_name, reduce_axes_name],
            outputs=[size_core_name],
            options={"keepDims": False},
        )
    )

    if size_core_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[size_core_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_constant_of_shape_op(node: Any, ctx: Any) -> None:
    shape_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(shape_name)
    ctx.ensure_tensor(output_name)

    shape_dtype = str(ctx.get_tensor_dtype(shape_name)).upper()
    fill_dims_name = shape_name
    if shape_dtype not in {"INT32", "INT64"}:
        shape_cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_constofshape_dims_i32",
            dtype="INT32",
            shape=[int(v) for v in ctx.get_tensor_shape(shape_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[shape_name],
                outputs=[shape_cast_name],
                options={
                    "inDataType": shape_dtype,
                    "outDataType": "INT32",
                },
            )
        )
        fill_dims_name = shape_cast_name

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    output_np_dtype = _numpy_dtype_from_tflite_dtype(output_dtype)
    value_attr = node.attrs.get("value", None)
    value_array: np.ndarray
    if value_attr is None:
        value_array = np.asarray(0, dtype=output_np_dtype)
    elif hasattr(value_attr, "values"):
        value_array = np.asarray(getattr(value_attr, "values"), dtype=output_np_dtype)
    else:
        value_array = np.asarray(value_attr, dtype=output_np_dtype)
    if int(value_array.size) == 0:
        value_array = np.asarray(0, dtype=output_np_dtype)
    scalar_value = np.asarray(value_array).reshape(-1)[0]

    fill_value_name = ctx.add_const_tensor(
        f"{output_name}_constofshape_value",
        np.asarray(scalar_value, dtype=output_np_dtype),
    )
    fill_value_tensor = ctx.model_ir.tensors.get(fill_value_name, None)
    if fill_value_tensor is not None:
        fill_value_tensor.shape = []
        fill_value_tensor.shape_signature = []
    ctx.add_operator(
        OperatorIR(
            op_type="FILL",
            inputs=[fill_dims_name, fill_value_name],
            outputs=[output_name],
        )
    )


def build_cast_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_shape_signature(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = _prefer_int32_index_output_dtype(
        ctx=ctx,
        tensor_name=output_name,
        requested_dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[input_name],
            outputs=[output_name],
            options={
                "inDataType": input_dtype,
                "outDataType": output_dtype,
            },
        )
    )


def build_expand_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    shape_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(shape_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    shape_shape = [int(v) for v in ctx.get_tensor_shape(shape_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_tensor = ctx.model_ir.tensors.get(output_name)
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in output_shape]
    )
    shape_const = ctx.get_constant_array(shape_name)
    dynamic_expand_shape = bool(shape_const is None or any(int(v) < 0 for v in output_signature))
    dynamic_output_signature = (
        [int(v) for v in output_signature]
        if any(int(v) < 0 for v in output_signature)
        else [-1 for _ in output_shape]
    )
    if (
        _is_unresolved_placeholder_shape(output_shape, output_signature)
        and len(output_shape) < len(input_shape)
    ):
        inferred_rank = int(len(input_shape))
        if shape_const is not None:
            inferred_rank = max(1, int(np.asarray(shape_const).size))
        output_shape = [1 for _ in range(max(1, inferred_rank))]
        if dynamic_expand_shape:
            dynamic_output_signature = [-1 for _ in output_shape]
            output_signature = [int(v) for v in dynamic_output_signature]
        else:
            output_signature = [int(v) for v in output_shape]
            dynamic_output_signature = [int(v) for v in output_signature]
        if output_tensor is not None:
            output_tensor.shape = [int(v) for v in output_shape]
            output_tensor.shape_signature = [int(v) for v in output_signature]
    shape_for_fill = shape_name
    shape_dtype = str(ctx.get_tensor_dtype(shape_name)).upper()
    if dynamic_expand_shape and shape_dtype != "INT32":
        shape_for_fill = ctx.add_intermediate_tensor(
            f"{output_name}_expand_shape_i32",
            dtype="INT32",
            shape=shape_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[shape_name],
                outputs=[shape_for_fill],
                options={
                    "inDataType": shape_dtype,
                    "outDataType": "INT32",
                },
            )
        )
        shape_dtype = "INT32"

    if len(output_shape) < len(input_shape):
        raise NotImplementedError(
            f"Expand output rank must be >= input rank in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )
    if not dynamic_expand_shape and any(int(v) <= 0 for v in output_shape):
        raise NotImplementedError(
            f"Expand requires static positive output shape for MUL-broadcast lowering in flatbuffer_direct. "
            f"op={node.name} output_shape={output_shape}"
        )

    rank_pad = int(len(output_shape) - len(input_shape))
    aligned_input_shape = [1] * rank_pad + [int(v) for v in input_shape]
    if any(int(v) <= 0 for v in aligned_input_shape):
        raise NotImplementedError(
            f"Expand requires static positive input shape for MUL-broadcast lowering in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )

    if not dynamic_expand_shape:
        for in_dim, out_dim in zip(aligned_input_shape, output_shape):
            if int(in_dim) == int(out_dim):
                continue
            if int(in_dim) == 1 and int(out_dim) > 0:
                continue
            raise NotImplementedError(
                f"Expand shape is not broadcast-compatible for MUL-broadcast lowering in flatbuffer_direct. "
                f"op={node.name} input_shape={input_shape} output_shape={output_shape}"
            )

    mul_input_name = input_name
    if aligned_input_shape != input_shape:
        reshaped_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_expand_input_reshape",
            dtype=ctx.get_tensor_dtype(input_name),
            shape=[int(v) for v in aligned_input_shape],
        )
        reshape_shape = ctx.add_const_tensor(
            f"{output_name}_expand_input_shape",
            np.asarray(aligned_input_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, reshape_shape],
                outputs=[reshaped_input_name],
                options={"newShape": [int(v) for v in aligned_input_shape]},
            )
        )
        mul_input_name = reshaped_input_name

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    mul_input_dtype = str(ctx.get_tensor_dtype(mul_input_name)).upper()
    mul_lhs_name = mul_input_name
    mul_lhs_dtype = mul_input_dtype

    # tf backend Expand path also realizes broadcast via multiply-by-ones.
    # For BOOL, emulate tf behavior by int32 multiply then cast back.
    if output_dtype == "BOOL":
        if mul_lhs_dtype == "BOOL":
            mul_lhs_i32_name = ctx.add_intermediate_tensor(
                f"{output_name}_expand_input_i32",
                dtype="INT32",
                shape=[int(v) for v in aligned_input_shape],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[mul_lhs_name],
                    outputs=[mul_lhs_i32_name],
                    options={
                        "inDataType": "BOOL",
                        "outDataType": "INT32",
                    },
                )
            )
            mul_lhs_name = mul_lhs_i32_name
            mul_lhs_dtype = "INT32"
        else:
            raise NotImplementedError(
                f"Expand BOOL output expects BOOL input in flatbuffer_direct. "
                f"op={node.name} input_dtype={mul_lhs_dtype} output_dtype={output_dtype}"
            )

    ones_dtype = _numpy_dtype_from_tflite_dtype(mul_lhs_dtype)
    if dynamic_expand_shape:
        ones_name = ctx.add_intermediate_tensor(
            f"{output_name}_expand_ones",
            dtype=mul_lhs_dtype,
            shape=[int(v) for v in output_shape],
        )
        ones_tensor = ctx.model_ir.tensors.get(ones_name)
        if ones_tensor is not None:
            ones_tensor.shape_signature = [int(v) for v in dynamic_output_signature]
        one_const_name = ctx.add_const_tensor(
            f"{output_name}_expand_one",
            np.asarray(1, dtype=ones_dtype),
        )
        one_const_tensor = ctx.model_ir.tensors.get(one_const_name, None)
        if one_const_tensor is not None:
            one_const_tensor.shape = []
            one_const_tensor.shape_signature = []
        ctx.add_operator(
            OperatorIR(
                op_type="FILL",
                inputs=[shape_for_fill, one_const_name],
                outputs=[ones_name],
            )
        )
        if output_tensor is not None:
            output_tensor.shape_signature = [int(v) for v in dynamic_output_signature]
    else:
        ones_name = ctx.add_const_tensor(
            f"{output_name}_expand_ones",
            np.ones(output_shape, dtype=ones_dtype),
        )

    mul_output_name = output_name
    if output_dtype == "BOOL":
        mul_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_expand_mul_out",
            dtype=mul_lhs_dtype,
            shape=[int(v) for v in output_shape],
        )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[mul_lhs_name, ones_name],
            outputs=[mul_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    if output_dtype == "BOOL":
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[mul_output_name],
                outputs=[output_name],
                options={
                    "inDataType": mul_lhs_dtype,
                    "outDataType": "BOOL",
                },
            )
        )


def build_tile_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    multiples_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(multiples_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    multiples_for_tile = multiples_name
    multiples_dtype = str(ctx.get_tensor_dtype(multiples_name)).upper()
    if multiples_dtype != "INT32":
        multiples_shape = [int(v) for v in ctx.get_tensor_shape(multiples_name)]
        multiples_for_tile = ctx.add_intermediate_tensor(
            f"{output_name}_tile_multiples_i32",
            dtype="INT32",
            shape=multiples_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[multiples_name],
                outputs=[multiples_for_tile],
                options={
                    "inDataType": multiples_dtype,
                    "outDataType": "INT32",
                },
            )
        )

    ctx.add_operator(
        OperatorIR(
            op_type="TILE",
            inputs=[input_name, multiples_for_tile],
            outputs=[output_name],
        )
    )


def build_pad_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    mode_raw = node.attrs.get("mode", "constant")
    if isinstance(mode_raw, (bytes, bytearray)):
        mode = mode_raw.decode("utf-8").lower()
    else:
        mode = str(mode_raw).lower()
    if mode not in ["constant", "reflect"]:
        raise NotImplementedError(
            f"Pad mode is not supported in flatbuffer_direct. op={node.name} mode={mode}"
        )

    input_rank = len(ctx.get_tensor_shape(input_name))
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_for_pad = str(input_name)
    pads_arr = None
    pads_input_name = ""
    if len(node.inputs) >= 2:
        pads_input_name = str(node.inputs[1].name)
        if pads_input_name != "":
            pads_arr = ctx.get_constant_array(pads_input_name)
    if pads_arr is None and "pads" in node.attrs:
        pads_arr = node.attrs.get("pads")

    pads_name = ""
    if pads_arr is not None:
        pads_flat = [int(v) for v in np.asarray(pads_arr).reshape(-1).tolist()]
        if len(pads_flat) != int(input_rank * 2):
            raise NotImplementedError(
                "Pad pads length must be 2 * input_rank for flatbuffer_direct. "
                f"op={node.name} rank={input_rank} pads_len={len(pads_flat)}"
            )
        pads_begin_raw = [int(v) for v in pads_flat[:input_rank]]
        pads_end_raw = [int(v) for v in pads_flat[input_rank:]]

        crop_begin = [max(-int(v), 0) for v in pads_begin_raw]
        crop_end = [max(-int(v), 0) for v in pads_end_raw]
        if any(int(v) > 0 for v in crop_begin + crop_end):
            # TFLite PAD does not accept negative paddings. Emulate ONNX negative
            # pads by pre-cropping input with STRIDED_SLICE.
            begin_name = ctx.add_const_tensor(
                f"{output_name}_pad_crop_begin",
                np.asarray(crop_begin, dtype=np.int32),
            )
            end_values = [
                int(np.iinfo(np.int32).max) if int(crop_end[i]) == 0 else -int(crop_end[i])
                for i in range(input_rank)
            ]
            end_name = ctx.add_const_tensor(
                f"{output_name}_pad_crop_end",
                np.asarray(end_values, dtype=np.int32),
            )
            strides_name = ctx.add_const_tensor(
                f"{output_name}_pad_crop_strides",
                np.asarray([1 for _ in range(input_rank)], dtype=np.int32),
            )
            end_mask = 0
            for axis in range(input_rank):
                if int(crop_end[axis]) == 0:
                    end_mask |= (1 << axis)

            cropped_shape: list[int] = []
            for axis in range(input_rank):
                dim = int(input_shape[axis]) if axis < len(input_shape) else -1
                if dim > 0:
                    cropped_dim = int(dim - int(crop_begin[axis]) - int(crop_end[axis]))
                    cropped_shape.append(int(max(cropped_dim, 0)))
                else:
                    cropped_shape.append(-1)
            cropped_name = ctx.add_intermediate_tensor(
                f"{output_name}_pad_cropped",
                dtype=str(ctx.get_tensor_dtype(input_name)),
                shape=cropped_shape,
            )
            cropped_tensor = ctx.model_ir.tensors[cropped_name]
            input_tensor = ctx.model_ir.tensors[input_name]
            if input_tensor.quantization is not None:
                cropped_tensor.quantization = _clone_quantization(input_tensor.quantization)
            if input_tensor.shape_signature is not None:
                cropped_tensor.shape_signature = [
                    int(v)
                    for v in list(input_tensor.shape_signature)
                ]
                for axis in range(input_rank):
                    sig_dim = int(cropped_tensor.shape_signature[axis])
                    if sig_dim > 0:
                        cropped_tensor.shape_signature[axis] = int(
                            max(sig_dim - int(crop_begin[axis]) - int(crop_end[axis]), 0)
                        )

            ctx.add_operator(
                OperatorIR(
                    op_type="STRIDED_SLICE",
                    inputs=[input_for_pad, begin_name, end_name, strides_name],
                    outputs=[cropped_name],
                    options={
                        "beginMask": 0,
                        "endMask": int(end_mask),
                        "ellipsisMask": 0,
                        "newAxisMask": 0,
                        "shrinkAxisMask": 0,
                        "offset": False,
                    },
                )
            )
            input_for_pad = str(cropped_name)

        pads_begin = [max(int(v), 0) for v in pads_begin_raw]
        pads_end = [max(int(v), 0) for v in pads_end_raw]
        paddings = np.asarray(
            [[int(b), int(e)] for b, e in zip(pads_begin, pads_end)],
            dtype=np.int32,
        )
        pads_name = ctx.add_const_tensor(
            f"{output_name}_pads",
            paddings,
        )
    else:
        if pads_input_name == "":
            raise NotImplementedError(
                f"Pad pads must be constant for flatbuffer_direct. op={node.name}"
            )
        ctx.ensure_tensor(pads_input_name)
        pads_vector_name = pads_input_name
        pads_dtype = str(ctx.get_tensor_dtype(pads_vector_name)).upper()
        if pads_dtype != "INT32":
            pads_vector_name_i32 = ctx.add_intermediate_tensor(
                f"{output_name}_pads_i32",
                dtype="INT32",
                shape=[int(input_rank * 2)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[pads_vector_name],
                    outputs=[pads_vector_name_i32],
                    options={
                        "inDataType": pads_dtype,
                        "outDataType": "INT32",
                    },
                )
            )
            pads_vector_name = pads_vector_name_i32

        pads_2xrank_name = ctx.add_intermediate_tensor(
            f"{output_name}_pads_2xrank",
            dtype="INT32",
            shape=[2, int(input_rank)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[pads_vector_name],
                outputs=[pads_2xrank_name],
                options={"newShape": [2, int(input_rank)]},
            )
        )

        perm_name = ctx.add_const_tensor(
            f"{output_name}_pads_transpose_perm",
            np.asarray([1, 0], dtype=np.int32),
        )
        pads_name = ctx.add_intermediate_tensor(
            f"{output_name}_pads",
            dtype="INT32",
            shape=[int(input_rank), 2],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[pads_2xrank_name, perm_name],
                outputs=[pads_name],
            )
        )

    if mode == "constant":
        pad_constant_tensor_name: str = ""
        use_padv2 = False
        if len(node.inputs) >= 3 and str(node.inputs[2].name) != "":
            constant_value_arr = ctx.get_constant_array(node.inputs[2].name)
            if constant_value_arr is None:
                raise NotImplementedError(
                    f"Pad constant value input must be constant for flatbuffer_direct. op={node.name}"
                )
            constant_value_vec = np.asarray(constant_value_arr).reshape(-1)
            if constant_value_vec.size == 0:
                raise NotImplementedError(
                    f"Pad constant value input must contain at least one element for flatbuffer_direct. op={node.name}"
                )
            constant_value = constant_value_vec[0]
            if np.issubdtype(np.asarray(constant_value).dtype, np.floating):
                is_zero_padding = bool(np.isfinite(constant_value)) and abs(float(constant_value)) <= 1e-12
            else:
                is_zero_padding = bool(constant_value == 0)
            use_padv2 = not is_zero_padding
            if use_padv2:
                input_dtype = str(ctx.get_tensor_dtype(input_for_pad)).upper()
                input_tensor = ctx.model_ir.tensors[input_for_pad]
                if input_tensor.quantization is not None:
                    raise NotImplementedError(
                        "Pad with non-zero constant value is not supported for quantized tensors in "
                        f"flatbuffer_direct. op={node.name}"
                    )
                pad_value = np.asarray(
                    [constant_value],
                    dtype=_numpy_dtype_from_tflite_dtype(input_dtype),
                )
                pad_constant_tensor_name = ctx.add_const_tensor(
                    f"{output_name}_pad_value",
                    pad_value,
                )
        if use_padv2:
            ctx.add_operator(
                OperatorIR(
                    op_type="PADV2",
                    inputs=[input_for_pad, pads_name, pad_constant_tensor_name],
                    outputs=[output_name],
                )
            )
        else:
            ctx.add_operator(
                OperatorIR(
                    op_type="PAD",
                    inputs=[input_for_pad, pads_name],
                    outputs=[output_name],
                )
            )
    else:
        ctx.add_operator(
            OperatorIR(
                op_type="MIRROR_PAD",
                inputs=[input_for_pad, pads_name],
                outputs=[output_name],
                options={"mode": "REFLECT"},
            )
        )


def _resolve_axes_from_attr_or_input(node: Any, ctx: Any) -> list[int]:
    axes = None
    if len(node.inputs) >= 2:
        axes_arr = ctx.get_constant_array(node.inputs[1].name)
        if axes_arr is None:
            raise NotImplementedError(
                f"{node.op} axes must be constant for flatbuffer_direct. op={node.name}"
            )
        axes = [int(v) for v in np.asarray(axes_arr).reshape(-1).tolist()]
    elif "axes" in node.attrs:
        attr_axes = node.attrs["axes"]
        if isinstance(attr_axes, (list, tuple)):
            axes = [int(v) for v in attr_axes]
        else:
            axes = [int(attr_axes)]
    return [] if axes is None else axes


def build_squeeze_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = ctx.get_tensor_shape(input_name)
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )
    rank = len(input_shape)
    axes = _resolve_axes_from_attr_or_input(node, ctx)
    axes_source_provided = (
        (len(node.inputs) >= 2 and str(node.inputs[1].name) != "")
        or ("axes" in node.attrs)
    )
    if len(axes) == 0 and not axes_source_provided:
        # ONNX Squeeze without axes removes all runtime singleton dimensions.
        # Preserve this behavior by passing empty squeezeDims.
        existing_output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_tensor.shape)]
        )
        output_tensor.shape_signature = [int(v) for v in existing_output_signature]
        output_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in existing_output_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[input_name],
                outputs=[output_name],
                options={"squeezeDims": []},
            )
        )
        return

    explicit_axes = axes_source_provided and len(axes) > 0
    if len(axes) == 0:
        axes = []
        for idx in range(rank):
            if idx < len(input_signature):
                if int(input_signature[idx]) == 1:
                    axes.append(int(idx))
            elif int(input_shape[idx]) == 1:
                axes.append(int(idx))

    normalized_axes: list[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += rank
        if a < 0 or a >= rank:
            raise NotImplementedError(
                f"Squeeze axis is out of range. op={node.name} axis={axis} rank={rank}"
            )
        if a not in normalized_axes:
            normalized_axes.append(a)

    axes_to_remove = set(int(v) for v in normalized_axes)
    if not explicit_axes and len(normalized_axes) == 0:
        axes_to_remove = set()
    inferred_output_shape = [
        int(input_shape[idx])
        for idx in range(rank)
        if idx not in axes_to_remove
    ]
    inferred_output_signature = [
        int(input_signature[idx])
        for idx in range(rank)
        if idx not in axes_to_remove
    ]
    output_tensor.shape = [int(v) for v in list(inferred_output_shape)]
    output_tensor.shape_signature = [int(v) for v in list(inferred_output_signature)]

    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[input_name],
            outputs=[output_name],
            options={"squeezeDims": [int(v) for v in normalized_axes]},
        )
    )


def build_unsqueeze_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    axes = _resolve_axes_from_attr_or_input(node, ctx)
    input_raw_shape = ctx.shape_map.get(input_name, None)
    logical_scalar_input = (
        isinstance(input_raw_shape, (list, tuple))
        and len(list(input_raw_shape)) == 0
    )
    input_rank = 0 if logical_scalar_input else len(ctx.get_tensor_shape(input_name))
    output_tensor = ctx.model_ir.tensors[output_name]
    output_shape = ctx.get_tensor_shape(output_name)
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_shape)]
    )
    if input_rank == 0 and len(output_signature) > 0:
        input_rank = int(max(len(output_signature) - len(axes), 0))

    if logical_scalar_input and len(axes) > 0:
        scalar_output_rank = int(len(axes))
        scalar_axes_valid = True
        scalar_seen_axes: set[int] = set()
        for axis in axes:
            scalar_axis = int(axis)
            if scalar_axis < 0:
                scalar_axis += scalar_output_rank
            if (
                scalar_axis < 0
                or scalar_axis >= scalar_output_rank
                or scalar_axis in scalar_seen_axes
            ):
                scalar_axes_valid = False
                break
            scalar_seen_axes.add(int(scalar_axis))
        if not scalar_axes_valid:
            # Some models lose rank info and appear as scalar placeholders ([] -> [1]).
            # If axes are incompatible with true scalar semantics, reinterpret as rank-1.
            logical_scalar_input = False
            input_rank = int(len(ctx.get_tensor_shape(input_name)))

    output_rank = int(input_rank + len(axes))
    normalized_axes: list[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += output_rank
        if a < 0 or a >= output_rank:
            raise NotImplementedError(
                "Unsqueeze axis out of range in flatbuffer_direct. "
                f"op={node.name} axis={axis} input_rank={input_rank} output_rank={output_rank}"
            )
        if a in normalized_axes:
            raise NotImplementedError(
                f"Unsqueeze axes must be unique in flatbuffer_direct. op={node.name} axes={axes}"
            )
        normalized_axes.append(int(a))
    normalized_axes = sorted([int(v) for v in normalized_axes])
    input_tensor = ctx.model_ir.tensors[input_name]
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor.shape_signature is not None
        else [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    )
    if logical_scalar_input:
        input_signature = []
    if input_rank == 0 and len(output_signature) > 0:
        reshape_shape = [int(v) for v in output_signature]
        output_tensor.shape_signature = [int(v) for v in reshape_shape]
        output_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in reshape_shape]
        reshape_options_shape = (
            []
            if any(int(v) < 0 for v in reshape_shape)
            else [int(v) for v in reshape_shape]
        )
        reshape_shape_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_shape",
            np.asarray(reshape_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, reshape_shape_name],
                outputs=[output_name],
                options={"newShape": reshape_options_shape},
            )
        )
        return
    if input_rank == 1 and len(normalized_axes) == 1:
        axis = int(normalized_axes[0])
        if axis not in (0, 1):
            raise NotImplementedError(
                f"Unsqueeze axis out of range for dynamic rank-1 input in flatbuffer_direct. "
                f"op={node.name} axis={axis}"
            )
        reshape_shape = [1, -1] if axis == 0 else [-1, 1]
        inferred_signature = [1, -1] if axis == 0 else [-1, 1]
        output_tensor.shape_signature = [int(v) for v in inferred_signature]
        output_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in inferred_signature]
        reshape_options_shape = (
            []
            if any(int(v) < 0 for v in reshape_shape)
            else [int(v) for v in reshape_shape]
        )
        reshape_shape_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_shape",
            np.asarray(reshape_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, reshape_shape_name],
                outputs=[output_name],
                options={"newShape": reshape_options_shape},
            )
        )
        return

    has_dynamic_dim = any(int(v) < 0 for v in output_signature)

    if len(input_signature) == int(input_rank) and all(int(v) >= 0 for v in input_signature):
        inferred_shape: list[int] = []
        src_dim_idx = 0
        output_rank = int(input_rank + len(normalized_axes))
        axes_set = set(int(v) for v in normalized_axes)
        for out_axis in range(output_rank):
            if out_axis in axes_set:
                inferred_shape.append(1)
            else:
                inferred_shape.append(int(input_signature[src_dim_idx]))
                src_dim_idx += 1
        output_tensor.shape = [int(v) for v in inferred_shape]
        output_tensor.shape_signature = [int(v) for v in inferred_shape]
        shape_const = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_shape",
            np.asarray(inferred_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, shape_const],
                outputs=[output_name],
                options={"newShape": [int(v) for v in inferred_shape]},
            )
        )
        return

    if not has_dynamic_dim:
        shape_const = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_shape",
            np.asarray(output_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, shape_const],
                outputs=[output_name],
                options={"newShape": [int(v) for v in output_shape]},
            )
        )
        return

    input_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_unsqueeze_input_shape",
        dtype="INT32",
        shape=[input_rank],
    )
    # Preserve dynamic dimensions in output metadata. Dynamic RESHAPE must not
    # be pinned to placeholder-1 static dimensions.
    output_rank = int(input_rank + len(normalized_axes))
    axes_set = set(int(v) for v in normalized_axes)
    inferred_signature: list[int] = []
    src_idx = 0
    for out_axis in range(output_rank):
        if out_axis in axes_set:
            inferred_signature.append(1)
        else:
            if src_idx < len(input_signature):
                inferred_signature.append(int(input_signature[src_idx]))
            else:
                inferred_signature.append(-1)
            src_idx += 1
    output_tensor.shape_signature = [int(v) for v in inferred_signature]
    output_tensor.shape = [int(v) if int(v) > 0 else 1 for v in inferred_signature]

    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[input_shape_name],
            options={"outType": "INT32"},
        )
    )

    current_shape_name = input_shape_name
    current_rank = int(input_rank)
    for axis in normalized_axes:
        prefix_name = ctx.add_intermediate_tensor(
            f"{output_name}_unsqueeze_prefix_{axis}",
            dtype="INT32",
            shape=[max(int(axis), 0)],
        )
        prefix_begin_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_prefix_begin_{axis}",
            np.asarray([0], dtype=np.int32),
        )
        prefix_size_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_prefix_size_{axis}",
            np.asarray([int(axis)], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[current_shape_name, prefix_begin_name, prefix_size_name],
                outputs=[prefix_name],
            )
        )

        suffix_len = int(max(current_rank - int(axis), 0))
        suffix_name = ctx.add_intermediate_tensor(
            f"{output_name}_unsqueeze_suffix_{axis}",
            dtype="INT32",
            shape=[suffix_len],
        )
        suffix_begin_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_suffix_begin_{axis}",
            np.asarray([int(axis)], dtype=np.int32),
        )
        suffix_size_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_suffix_size_{axis}",
            np.asarray([suffix_len], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[current_shape_name, suffix_begin_name, suffix_size_name],
                outputs=[suffix_name],
            )
        )

        one_dim_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_one_{axis}",
            np.asarray([1], dtype=np.int32),
        )
        merged_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_unsqueeze_shape_merged_{axis}",
            dtype="INT32",
            shape=[current_rank + 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[prefix_name, one_dim_name, suffix_name],
                outputs=[merged_shape_name],
                options={
                    "axis": 0,
                    "fusedActivationFunction": "NONE",
                },
            )
        )
        current_shape_name = merged_shape_name
        current_rank += 1

    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, current_shape_name],
            outputs=[output_name],
            options={"newShape": []},
        )
    )


def build_space_to_depth_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NotImplementedError(
            f"SpaceToDepth supports rank-4 input/output only. op={node.name} "
            f"input_shape={input_shape} output_shape={output_shape}"
        )

    block_size = int(node.attrs.get("blocksize", 0))
    if block_size <= 1:
        raise NotImplementedError(
            f"SpaceToDepth blocksize must be > 1. op={node.name} blocksize={block_size}"
        )
    mode_raw = node.attrs.get("mode", "DCR")
    if isinstance(mode_raw, (bytes, bytearray)):
        mode = mode_raw.decode("utf-8").upper()
    else:
        mode = str(mode_raw).upper()
    if mode != "DCR":
        raise NotImplementedError(
            f"SpaceToDepth mode must be DCR for flatbuffer_direct builtin lowering. op={node.name} mode={mode}"
        )

    nhwc_input_shape = [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
    nhwc_output_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[3]), int(output_shape[1])]
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    nhwc_input_signature = (
        [int(v) for v in list(nhwc_input_shape)]
        if input_tensor is None or input_tensor.shape_signature is None
        else [
            int(input_tensor.shape_signature[0]),
            int(input_tensor.shape_signature[2]),
            int(input_tensor.shape_signature[3]),
            int(input_tensor.shape_signature[1]),
        ]
    )
    nhwc_output_signature = (
        [int(v) for v in list(nhwc_output_shape)]
        if output_tensor is None or output_tensor.shape_signature is None
        else [
            int(output_tensor.shape_signature[0]),
            int(output_tensor.shape_signature[2]),
            int(output_tensor.shape_signature[3]),
            int(output_tensor.shape_signature[1]),
        ]
    )

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=nhwc_input_shape,
    )
    ctx.model_ir.tensors[x_nhwc].shape_signature = [int(v) for v in list(nhwc_input_signature)]
    x_nhwc = make_transpose(
        ctx,
        input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in list(nhwc_output_signature)]
    if ctx.model_ir.tensors[x_nhwc].quantization is not None:
        ctx.model_ir.tensors[y_nhwc].quantization = _clone_quantization(
            ctx.model_ir.tensors[x_nhwc].quantization
        )

    ctx.add_operator(
        OperatorIR(
            op_type="SPACE_TO_DEPTH",
            inputs=[x_nhwc],
            outputs=[y_nhwc],
            options={"blockSize": int(block_size)},
        )
    )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
        allow_elide_inverse_chain=True,
    )


def build_depth_to_space_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    output_shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NotImplementedError(
            f"DepthToSpace supports rank-4 input/output only. op={node.name} "
            f"input_shape={input_shape} output_shape={output_shape}"
        )

    block_size = int(node.attrs.get("blocksize", 0))
    if block_size <= 1:
        raise NotImplementedError(
            f"DepthToSpace blocksize must be > 1. op={node.name} blocksize={block_size}"
        )
    mode_raw = node.attrs.get("mode", "DCR")
    if isinstance(mode_raw, (bytes, bytearray)):
        mode = mode_raw.decode("utf-8").upper()
    else:
        mode = str(mode_raw).upper()
    if mode not in {"DCR", "CRD"}:
        raise NotImplementedError(
            f"DepthToSpace mode must be DCR or CRD for flatbuffer_direct builtin lowering. op={node.name} mode={mode}"
        )

    nhwc_input_shape = [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
    nhwc_output_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[3]), int(output_shape[1])]
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    nhwc_input_signature = (
        [int(v) for v in list(nhwc_input_shape)]
        if input_tensor is None or input_tensor.shape_signature is None
        else [
            int(input_tensor.shape_signature[0]),
            int(input_tensor.shape_signature[2]),
            int(input_tensor.shape_signature[3]),
            int(input_tensor.shape_signature[1]),
        ]
    )
    nhwc_output_signature = (
        [int(v) for v in list(nhwc_output_shape)]
        if output_tensor is None or output_tensor.shape_signature is None
        else [
            int(output_tensor.shape_signature[0]),
            int(output_tensor.shape_signature[2]),
            int(output_tensor.shape_signature[3]),
            int(output_tensor.shape_signature[1]),
        ]
    )

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=nhwc_input_shape,
    )
    ctx.model_ir.tensors[x_nhwc].shape_signature = [int(v) for v in list(nhwc_input_signature)]
    x_nhwc = make_transpose(
        ctx,
        input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in list(nhwc_output_signature)]
    if ctx.model_ir.tensors[x_nhwc].quantization is not None:
        ctx.model_ir.tensors[y_nhwc].quantization = _clone_quantization(
            ctx.model_ir.tensors[x_nhwc].quantization
        )

    if mode == "DCR":
        ctx.add_operator(
            OperatorIR(
                op_type="DEPTH_TO_SPACE",
                inputs=[x_nhwc],
                outputs=[y_nhwc],
                options={"blockSize": int(block_size)},
            )
        )
    else:
        block_area = int(block_size * block_size)
        channels = int(nhwc_input_shape[3])
        input_channel_signature = int(nhwc_input_signature[3]) if len(nhwc_input_signature) == 4 else int(channels)
        channel_dim_for_crd = int(channels) if int(channels) > 0 else int(input_channel_signature)
        if channel_dim_for_crd <= 0:
            raise NotImplementedError(
                "DepthToSpace CRD requires static channel dimension in flatbuffer_direct builtin lowering. "
                f"op={node.name} channel_dim={channel_dim_for_crd}"
            )
        if channel_dim_for_crd % block_area != 0:
            raise NotImplementedError(
                f"DepthToSpace CRD requires input channels divisible by blocksize^2. "
                f"op={node.name} channels={channel_dim_for_crd} blocksize={block_size}"
            )
        out_channels = int(channel_dim_for_crd // block_area)

        # ONNX CRD layout can be lowered by reordering channels to DCR layout first,
        # then applying TFLite DEPTH_TO_SPACE.
        # CRD channel index: ((c * b) + by) * b + bx
        # DCR channel index: ((by * b) + bx) * c + c_idx
        gather_indices = []
        for by in range(int(block_size)):
            for bx in range(int(block_size)):
                for c_idx in range(int(out_channels)):
                    gather_indices.append(
                        int(((int(c_idx) * int(block_size)) + int(by)) * int(block_size) + int(bx))
                    )
        gather_indices_name = ctx.add_const_tensor(
            f"{node.name}_crd_to_dcr_indices",
            np.asarray(gather_indices, dtype=np.int32),
        )
        x_dcr = ctx.add_intermediate_tensor(
            f"{node.name}_crd_input_reordered",
            dtype=ctx.get_tensor_dtype(output_name),
            shape=[int(v) for v in list(nhwc_input_shape)],
        )
        x_dcr_tensor = ctx.model_ir.tensors.get(x_dcr, None)
        if x_dcr_tensor is not None:
            x_dcr_tensor.shape_signature = [int(v) for v in list(nhwc_input_signature)]
            if ctx.model_ir.tensors[x_nhwc].quantization is not None:
                x_dcr_tensor.quantization = _clone_quantization(
                    ctx.model_ir.tensors[x_nhwc].quantization
                )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[x_nhwc, gather_indices_name],
                outputs=[x_dcr],
                options={
                    "axis": 3,
                    "batchDims": 0,
                },
            )
        )

        if len(nhwc_output_signature) == 4 and int(nhwc_output_signature[3]) < 0:
            nhwc_output_signature[3] = int(out_channels)
            y_nhwc_tensor = ctx.model_ir.tensors.get(y_nhwc, None)
            if y_nhwc_tensor is not None:
                y_nhwc_tensor.shape_signature = [int(v) for v in list(nhwc_output_signature)]
                y_nhwc_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in list(nhwc_output_signature)]

        ctx.add_operator(
            OperatorIR(
                op_type="DEPTH_TO_SPACE",
                inputs=[x_dcr],
                outputs=[y_nhwc],
                options={"blockSize": int(block_size)},
            )
        )
        if ctx.model_ir.tensors[x_nhwc].quantization is not None:
            ctx.model_ir.tensors[y_nhwc].quantization = _clone_quantization(
                ctx.model_ir.tensors[x_nhwc].quantization
            )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
        allow_elide_inverse_chain=True,
    )


def build_flatten_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    output_tensor = ctx.model_ir.tensors[output_name]
    output_shape_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_tensor.shape)]
    )
    output_shape = [int(v) if int(v) >= 0 else 1 for v in output_shape_signature]
    output_tensor.shape_signature = [int(v) for v in output_shape_signature]
    output_tensor.shape = [int(v) for v in output_shape]
    shape_const = ctx.add_const_tensor(
        f"{output_name}_flatten_shape",
        np.asarray(output_shape_signature, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape_signature]},
        )
    )


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


def _get_original_node_input_names(node: Any, ctx: Any) -> list[str]:
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None or getattr(onnx_model, "graph", None) is None:
        return [str(v.name) for v in node.inputs]
    for graph_node in onnx_model.graph.node:
        graph_node_name = str(graph_node.name) if str(graph_node.name) != "" else str(graph_node.op_type)
        if graph_node_name == str(node.name) and str(graph_node.op_type) == str(node.op):
            return [str(v) for v in graph_node.input]
    return [str(v.name) for v in node.inputs]


def _resolve_resize_target_hw(node: Any, ctx: Any, input_shape: list[int]) -> tuple[int, int]:
    def _resolve_from_sizes(arr: np.ndarray) -> tuple[int, int]:
        values = np.asarray(arr).reshape(-1).astype(np.int64)
        if values.size >= 4:
            return int(values[-2]), int(values[-1])
        if values.size == 2:
            return int(values[0]), int(values[1])
        raise NotImplementedError(
            f"Resize sizes must have at least 2 values. op={node.name} sizes_shape={list(values.shape)}"
        )

    def _resolve_from_scales(arr: np.ndarray) -> tuple[int, int]:
        values = np.asarray(arr).reshape(-1).astype(np.float32)
        in_h = int(input_shape[2])
        in_w = int(input_shape[3])
        if values.size >= 4:
            out_h = int(round(float(in_h) * float(values[-2])))
            out_w = int(round(float(in_w) * float(values[-1])))
            return max(out_h, 1), max(out_w, 1)
        if values.size == 2:
            out_h = int(round(float(in_h) * float(values[0])))
            out_w = int(round(float(in_w) * float(values[1])))
            return max(out_h, 1), max(out_w, 1)
        raise NotImplementedError(
            f"Resize scales must have at least 2 values. op={node.name} scales_shape={list(values.shape)}"
        )

    original_inputs = _get_original_node_input_names(node, ctx)

    if len(original_inputs) >= 4:
        sizes_name = str(original_inputs[3])
        if sizes_name != "":
            sizes = ctx.get_constant_array(sizes_name)
            if sizes is not None and int(np.asarray(sizes).size) >= 2:
                return _resolve_from_sizes(np.asarray(sizes))
    if len(original_inputs) >= 3:
        scales_name = str(original_inputs[2])
        if scales_name != "":
            scales = ctx.get_constant_array(scales_name)
            if scales is not None and int(np.asarray(scales).size) >= 2:
                arr = np.asarray(scales)
                if np.issubdtype(arr.dtype, np.integer):
                    return _resolve_from_sizes(arr)
                return _resolve_from_scales(arr)
    if len(original_inputs) == 2:
        param_name = str(original_inputs[1])
        if param_name != "":
            param = ctx.get_constant_array(param_name)
            if param is not None and int(np.asarray(param).size) >= 2:
                arr = np.asarray(param)
                if np.issubdtype(arr.dtype, np.integer):
                    return _resolve_from_sizes(arr)
                return _resolve_from_scales(arr)
    raise NotImplementedError(
        f"Resize target size must be resolvable from constant sizes/scales. op={node.name}"
    )


def _build_resize_dynamic_size_input(node: Any, ctx: Any) -> str | None:
    original_inputs = _get_original_node_input_names(node, ctx)
    sizes_name = ""
    if len(original_inputs) >= 4:
        sizes_name = str(original_inputs[3])
    elif len(original_inputs) == 2:
        # _NodeWrap drops empty optional inputs, so
        # Resize(x, "", "", sizes) may appear as 2-input form.
        sizes_name = str(original_inputs[1])
    if sizes_name == "":
        return None
    sizes_const = ctx.get_constant_array(sizes_name)
    if sizes_const is not None and int(np.asarray(sizes_const).size) >= 2:
        return None

    ctx.ensure_tensor(sizes_name)
    sizes_shape = [int(v) for v in ctx.get_tensor_shape(sizes_name)]
    if sizes_shape != [1] and len(sizes_shape) != 1:
        raise NotImplementedError(
            f"Resize dynamic sizes input must be rank-1. op={node.name} sizes_shape={sizes_shape}"
        )
    sizes_tensor = ctx.model_ir.tensors[sizes_name]
    sizes_signature = (
        [int(v) for v in list(sizes_tensor.shape_signature)]
        if sizes_tensor.shape_signature is not None and len(list(sizes_tensor.shape_signature)) == 1
        else [int(v) for v in list(sizes_shape)]
    )
    sizes_len = int(sizes_shape[0]) if len(sizes_shape) == 1 else 1
    if sizes_len <= 0 and len(sizes_signature) == 1 and int(sizes_signature[0]) > 0:
        sizes_len = int(sizes_signature[0])
    if sizes_len == 1:
        # Placeholder rank-1 length from symbolic shape inference.
        # ONNX Resize `sizes` for rank-4 tensors is typically length-4 (N,C,H,W).
        sizes_len = 4

    sizes_i32 = sizes_name
    sizes_dtype = str(ctx.get_tensor_dtype(sizes_name)).upper()
    if sizes_dtype != "INT32":
        cast_sizes_name = ctx.add_intermediate_tensor(
            f"{node.name}_resize_sizes_i32",
            dtype="INT32",
            shape=[int(v) for v in list(sizes_shape)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[sizes_name],
                outputs=[cast_sizes_name],
                options={
                    "inDataType": sizes_dtype,
                    "outDataType": "INT32",
                },
            )
        )
        sizes_i32 = cast_sizes_name

    min_hw_const = ctx.add_const_tensor(
        f"{node.name}_resize_min_hw",
        np.asarray([1, 1], dtype=np.int32),
    )

    def _clamp_hw_size(size_tensor_name: str, *, suffix: str) -> str:
        clamped_name = ctx.add_intermediate_tensor(
            f"{node.name}_resize_size_hw_clamped_{suffix}",
            dtype="INT32",
            shape=[2],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MAXIMUM",
                inputs=[size_tensor_name, min_hw_const],
                outputs=[clamped_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return clamped_name

    if sizes_len == 2:
        return _clamp_hw_size(sizes_i32, suffix="from_sizes2")
    if sizes_len == 4:
        size_hw = ctx.add_intermediate_tensor(
            f"{node.name}_resize_size_hw",
            dtype="INT32",
            shape=[2],
        )
        begin = ctx.add_const_tensor(
            f"{node.name}_resize_sizes_begin",
            np.asarray([2], dtype=np.int32),
        )
        size = ctx.add_const_tensor(
            f"{node.name}_resize_sizes_size",
            np.asarray([2], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[sizes_i32, begin, size],
                outputs=[size_hw],
            )
        )
        return _clamp_hw_size(size_hw, suffix="from_sizes4")

    raise NotImplementedError(
        f"Resize dynamic sizes input length must be 2 or 4. "
        f"op={node.name} sizes_shape={sizes_shape} sizes_signature={sizes_signature}"
    )


def _infer_resize_output_signature_nchw(
    *,
    input_signature_nchw: list[int],
    output_shape_nchw: list[int],
    onnx_sizes_hw: list[int] | None,
    onnx_scales_hw: list[float] | None,
    existing_output_signature_nchw: list[int] | None,
) -> list[int]:
    signature = [int(v) for v in list(output_shape_nchw)]
    if existing_output_signature_nchw is not None and len(existing_output_signature_nchw) == 4:
        for axis in range(4):
            if int(existing_output_signature_nchw[axis]) < 0:
                signature[axis] = -1
    if len(input_signature_nchw) == 4:
        if int(input_signature_nchw[0]) < 0:
            signature[0] = -1
        if int(input_signature_nchw[1]) < 0:
            signature[1] = -1
        if onnx_sizes_hw is not None and len(onnx_sizes_hw) >= 2:
            signature[2] = int(onnx_sizes_hw[0])
            signature[3] = int(onnx_sizes_hw[1])
        elif onnx_scales_hw is not None and len(onnx_scales_hw) >= 2:
            if int(input_signature_nchw[2]) < 0:
                signature[2] = -1
            else:
                signature[2] = max(int(round(float(input_signature_nchw[2]) * float(onnx_scales_hw[0]))), 1)
            if int(input_signature_nchw[3]) < 0:
                signature[3] = -1
            else:
                signature[3] = max(int(round(float(input_signature_nchw[3]) * float(onnx_scales_hw[1]))), 1)
    return [int(v) for v in signature]


def _resolve_integer_resize_scales_hw(onnx_scales_hw: list[float] | None) -> list[int] | None:
    if onnx_scales_hw is None or len(onnx_scales_hw) < 2:
        return None
    scales_int: list[int] = []
    for raw_scale in onnx_scales_hw[:2]:
        scale = float(raw_scale)
        rounded = int(round(scale))
        if rounded <= 0:
            return None
        if abs(scale - float(rounded)) > 1e-6:
            return None
        scales_int.append(int(rounded))
    return scales_int


def _extract_resize_onnx_hw_hints(node: Any, ctx: Any) -> tuple[list[int] | None, list[float] | None]:
    onnx_sizes_hw: list[int] | None = None
    onnx_scales_hw: list[float] | None = None

    original_inputs = _get_original_node_input_names(node, ctx)

    if len(original_inputs) >= 4:
        sizes_name = str(original_inputs[3])
        if sizes_name != "":
            sizes = ctx.get_constant_array(sizes_name)
            if sizes is not None and int(np.asarray(sizes).size) >= 2:
                values = np.asarray(sizes).reshape(-1).astype(np.int64)
                if values.size >= 4:
                    onnx_sizes_hw = [int(values[-2]), int(values[-1])]
                elif values.size == 2:
                    onnx_sizes_hw = [int(values[0]), int(values[1])]

    if len(original_inputs) >= 3:
        scales_name = str(original_inputs[2])
        if scales_name != "":
            scales = ctx.get_constant_array(scales_name)
            if scales is not None and int(np.asarray(scales).size) >= 2:
                values = np.asarray(scales).reshape(-1).astype(np.float32)
                if values.size >= 4:
                    onnx_scales_hw = [float(values[-2]), float(values[-1])]
                elif values.size == 2:
                    onnx_scales_hw = [float(values[0]), float(values[1])]
    elif len(original_inputs) == 2:
        param_name = str(original_inputs[1])
        if param_name != "":
            param = ctx.get_constant_array(param_name)
            if param is not None and int(np.asarray(param).size) >= 2:
                values = np.asarray(param).reshape(-1)
                if np.issubdtype(values.dtype, np.integer):
                    v = values.astype(np.int64)
                    if v.size >= 4:
                        onnx_sizes_hw = [int(v[-2]), int(v[-1])]
                    elif v.size == 2:
                        onnx_sizes_hw = [int(v[0]), int(v[1])]
                else:
                    v = values.astype(np.float32)
                    if v.size >= 4:
                        onnx_scales_hw = [float(v[-2]), float(v[-1])]
                    elif v.size == 2:
                        onnx_scales_hw = [float(v[0]), float(v[1])]

    return onnx_sizes_hw, onnx_scales_hw


def _resolve_resize_flags(node: Any) -> tuple[str, str, bool, bool]:
    mode = str(node.attrs.get("mode", "nearest")).lower()
    ctm = str(node.attrs.get("coordinate_transformation_mode", "half_pixel")).lower()
    align_corners = bool(ctm == "align_corners")
    half_pixel_centers = bool(ctm in {"half_pixel", "pytorch_half_pixel"})
    if mode == "nearest" and ctm == "asymmetric":
        align_corners = False
        half_pixel_centers = False
    return mode, ctm, align_corners, half_pixel_centers


def _normalize_resize_coordinate_transformation_mode(
    *,
    coordinate_transformation_mode: str,
) -> str | None:
    ctm = str(coordinate_transformation_mode).lower()
    if ctm in {"align_corners", "asymmetric", "half_pixel", "pytorch_half_pixel"}:
        return ctm
    return None


def _compute_resize_source_index(
    *,
    out_index: int,
    input_size: int,
    output_size: int,
    coordinate_transformation_mode: str,
) -> float:
    ctm = str(coordinate_transformation_mode).lower()
    in_size = int(input_size)
    out_size = int(output_size)
    i = float(out_index)
    if ctm == "align_corners":
        if out_size <= 1:
            return 0.0
        return i * float(in_size - 1) / float(out_size - 1)
    if ctm == "asymmetric":
        return i * float(in_size) / float(out_size)
    if ctm == "half_pixel":
        return (i + 0.5) * float(in_size) / float(out_size) - 0.5
    if ctm == "pytorch_half_pixel":
        if out_size > 1:
            return (i + 0.5) * float(in_size) / float(out_size) - 0.5
        return 0.0
    raise NotImplementedError(
        f"Unsupported coordinate_transformation_mode for Resize(cubic): {coordinate_transformation_mode}"
    )


def _cubic_kernel_weight(*, distance: float, cubic_coeff_a: float) -> float:
    t = float(abs(distance))
    a = float(cubic_coeff_a)
    if t <= 1.0:
        return ((a + 2.0) * t * t * t) - ((a + 3.0) * t * t) + 1.0
    if t < 2.0:
        return (a * t * t * t) - (5.0 * a * t * t) + (8.0 * a * t) - (4.0 * a)
    return 0.0


def _build_resize_cubic_matrix_from_onnx(
    *,
    input_size: int,
    output_size: int,
    coordinate_transformation_mode: str,
    cubic_coeff_a: float,
    exclude_outside: bool,
) -> np.ndarray:
    ctm = _normalize_resize_coordinate_transformation_mode(
        coordinate_transformation_mode=coordinate_transformation_mode,
    )
    if ctm is None:
        raise NotImplementedError(
            "Resize(cubic) strict lowering supports coordinate_transformation_mode "
            f"only in [align_corners, asymmetric, half_pixel, pytorch_half_pixel]. "
            f"got={coordinate_transformation_mode}"
        )

    key = (
        int(input_size),
        int(output_size),
        str(ctm),
        float(cubic_coeff_a),
        bool(exclude_outside),
    )
    cached = _BICUBIC_MATRIX_CACHE.get(key, None)
    if cached is not None:
        return np.asarray(cached, dtype=np.float32)
    in_size = int(input_size)
    out_size = int(output_size)
    if in_size <= 0 or out_size <= 0:
        raise NotImplementedError(
            f"Resize(cubic) requires positive input/output size. input_size={in_size} output_size={out_size}"
        )

    matrix = np.zeros((out_size, in_size), dtype=np.float32)
    for out_idx in range(out_size):
        src = _compute_resize_source_index(
            out_index=out_idx,
            input_size=in_size,
            output_size=out_size,
            coordinate_transformation_mode=ctm,
        )
        src_floor = int(np.floor(src))
        row_weight_sum = 0.0
        for offset in [-1, 0, 1, 2]:
            src_idx = int(src_floor + offset)
            weight = _cubic_kernel_weight(
                distance=src - float(src_idx),
                cubic_coeff_a=float(cubic_coeff_a),
            )
            if bool(exclude_outside) and (src_idx < 0 or src_idx >= in_size):
                continue
            if src_idx < 0:
                src_idx = 0
            elif src_idx >= in_size:
                src_idx = in_size - 1
            matrix[out_idx, src_idx] += np.float32(weight)
            row_weight_sum += float(weight)
        if bool(exclude_outside) and abs(row_weight_sum) > 0.0:
            matrix[out_idx, :] = matrix[out_idx, :] / np.float32(row_weight_sum)

    _BICUBIC_MATRIX_CACHE[key] = matrix
    return np.asarray(matrix, dtype=np.float32)


def _add_reshape_operator(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    new_shape: list[int],
    preserve_dynamic_shape: bool = False,
) -> None:
    shape_const = ctx.add_const_tensor(
        f"{output_name}_reshape_shape",
        np.asarray([int(v) for v in list(new_shape)], dtype=np.int32),
    )
    options = {"newShape": [int(v) for v in list(new_shape)]}
    if bool(preserve_dynamic_shape):
        options["preserveDynamicShape"] = True
        output_tensor = ctx.model_ir.tensors.get(output_name, None)
        if output_tensor is not None:
            target_signature = [int(v) for v in list(new_shape)]
            output_tensor.shape_signature = [int(v) for v in target_signature]
            output_tensor.shape = [
                int(v) if int(v) >= 0 else 1 for v in target_signature
            ]
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options=options,
        )
    )


def _build_resize_cubic_strict_op(
    *,
    node: Any,
    ctx: Any,
    x_nhwc: str,
    y_nhwc: str,
    input_signature_nchw: list[int],
    output_signature_nchw: list[int],
    coordinate_transformation_mode: str,
    cubic_coeff_a: float,
    exclude_outside: bool,
) -> None:
    if len(input_signature_nchw) != 4 or len(output_signature_nchw) != 4:
        raise NotImplementedError(
            f"Resize(cubic) strict lowering supports rank-4 signature only. "
            f"op={node.name} input_signature={input_signature_nchw} output_signature={output_signature_nchw}"
        )
    if any(int(input_signature_nchw[idx]) < 0 for idx in [1, 2, 3]):
        raise NotImplementedError(
            "Resize(cubic) strict lowering requires static input C/H/W for flatbuffer_direct. "
            f"op={node.name} input_signature={input_signature_nchw}"
        )
    if any(int(output_signature_nchw[idx]) < 0 for idx in [2, 3]):
        raise NotImplementedError(
            "Resize(cubic) strict lowering requires static output H/W for flatbuffer_direct. "
            f"op={node.name} output_signature={output_signature_nchw}"
        )

    in_c = int(input_signature_nchw[1])
    in_h = int(input_signature_nchw[2])
    in_w = int(input_signature_nchw[3])
    out_c = int(output_signature_nchw[1])
    out_h = int(output_signature_nchw[2])
    out_w = int(output_signature_nchw[3])
    if out_c >= 0 and out_c != in_c:
        raise NotImplementedError(
            "Resize(cubic) strict lowering expects channel-preserving resize. "
            f"op={node.name} input_c={in_c} output_c={out_c}"
        )
    if min(in_c, in_h, in_w, out_h, out_w) <= 0:
        raise NotImplementedError(
            "Resize(cubic) strict lowering requires positive static C/H/W values. "
            f"op={node.name} in=[{in_c},{in_h},{in_w}] out=[{out_h},{out_w}]"
        )

    x_shape_nhwc = [int(v) for v in list(ctx.get_tensor_shape(x_nhwc))]
    input_batch_shape = int(x_shape_nhwc[0]) if len(x_shape_nhwc) == 4 else -1
    input_batch_signature = int(input_signature_nchw[0]) if len(input_signature_nchw) == 4 else -1
    output_batch_signature = int(output_signature_nchw[0]) if len(output_signature_nchw) == 4 else -1
    static_batch = -1
    if input_batch_signature > 0:
        static_batch = int(input_batch_signature)
    elif output_batch_signature > 0:
        static_batch = int(output_batch_signature)
    elif input_batch_shape > 0:
        static_batch = int(input_batch_shape)

    # Keep ONNX cubic semantics by deriving 1D interpolation matrices
    # from ONNX Resize attributes and applying them separably.
    h_matrix = _build_resize_cubic_matrix_from_onnx(
        input_size=in_h,
        output_size=out_h,
        coordinate_transformation_mode=coordinate_transformation_mode,
        cubic_coeff_a=float(cubic_coeff_a),
        exclude_outside=bool(exclude_outside),
    )
    w_matrix = _build_resize_cubic_matrix_from_onnx(
        input_size=in_w,
        output_size=out_w,
        coordinate_transformation_mode=coordinate_transformation_mode,
        cubic_coeff_a=float(cubic_coeff_a),
        exclude_outside=bool(exclude_outside),
    )
    h_matrix_name = ctx.add_const_tensor(
        f"{node.name}_resize_cubic_h_matrix",
        np.asarray(h_matrix, dtype=np.float32).reshape(1, out_h, in_h),
    )
    w_matrix_name = ctx.add_const_tensor(
        f"{node.name}_resize_cubic_w_matrix",
        np.asarray(w_matrix, dtype=np.float32).reshape(1, 1, out_w, in_w),
    )

    x_work = x_nhwc
    x_dtype = str(ctx.get_tensor_dtype(x_nhwc)).upper()
    if x_dtype != "FLOAT32":
        x_cast = ctx.add_intermediate_tensor(
            f"{node.name}_resize_cubic_input_f32",
            dtype="FLOAT32",
            shape=[int(static_batch if static_batch > 0 else -1), in_h, in_w, in_c],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[x_nhwc],
                outputs=[x_cast],
                options={
                    "inDataType": x_dtype,
                    "outDataType": "FLOAT32",
                },
            )
        )
        x_work = x_cast

    # Height interpolation:
    # [N,H,W,C] -> reshape [N,H,W*C] -> BMM([1,out_h,H], ...) -> [N,out_h,W*C] -> reshape [N,out_h,W,C]
    flatten_wc = int(in_w * in_c)
    x_h_3d = ctx.add_intermediate_tensor(
        f"{node.name}_resize_cubic_h_in",
        dtype="FLOAT32",
        shape=[int(static_batch if static_batch > 0 else -1), in_h, flatten_wc],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=x_work,
        output_name=x_h_3d,
        new_shape=[int(static_batch if static_batch > 0 else -1), in_h, flatten_wc],
    )
    y_h_3d = ctx.add_intermediate_tensor(
        f"{node.name}_resize_cubic_h_out",
        dtype="FLOAT32",
        shape=[int(static_batch if static_batch > 0 else -1), out_h, flatten_wc],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[h_matrix_name, x_h_3d],
            outputs=[y_h_3d],
            options={
                "adjX": False,
                "adjY": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )
    y_h_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_resize_cubic_h_nhwc",
        dtype="FLOAT32",
        shape=[int(static_batch if static_batch > 0 else -1), out_h, in_w, in_c],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=y_h_3d,
        output_name=y_h_nhwc,
        new_shape=[int(static_batch if static_batch > 0 else -1), out_h, in_w, in_c],
    )

    y_float_nhwc = y_nhwc
    output_dtype = str(ctx.get_tensor_dtype(y_nhwc)).upper()
    if output_dtype != "FLOAT32":
        y_float_nhwc = ctx.add_intermediate_tensor(
            f"{node.name}_resize_cubic_output_f32",
            dtype="FLOAT32",
            shape=[int(static_batch if static_batch > 0 else -1), out_h, out_w, in_c],
        )
    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[w_matrix_name, y_h_nhwc],
            outputs=[y_float_nhwc],
            options={
                "adjX": False,
                "adjY": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    # Keep batch metadata stable on the NHWC cubic output tensor.
    y_float_tensor = ctx.model_ir.tensors.get(y_float_nhwc, None)
    if y_float_tensor is not None:
        y_float_tensor.shape = [
            int(static_batch if static_batch > 0 else -1),
            int(out_h),
            int(out_w),
            int(in_c),
        ]
        y_float_tensor.shape_signature = [
            int(output_batch_signature if output_batch_signature > 0 else -1),
            int(output_signature_nchw[2]) if int(output_signature_nchw[2]) > 0 else -1,
            int(output_signature_nchw[3]) if int(output_signature_nchw[3]) > 0 else -1,
            int(output_signature_nchw[1]) if int(output_signature_nchw[1]) > 0 else -1,
        ]

    if output_dtype != "FLOAT32":
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[y_float_nhwc],
                outputs=[y_nhwc],
                options={
                    "inDataType": "FLOAT32",
                    "outDataType": output_dtype,
                },
            )
        )


def build_grid_sample_op(node: Any, ctx: Any) -> None:
    image_name = node.inputs[0].name
    grid_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(image_name)
    ctx.ensure_tensor(grid_name)
    ctx.ensure_tensor(output_name)

    image_shape = [int(v) for v in ctx.get_tensor_shape(image_name)]
    grid_shape = [int(v) for v in ctx.get_tensor_shape(grid_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]

    grid_tensor = ctx.model_ir.tensors.get(grid_name, None)
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    grid_signature = (
        [int(v) for v in list(grid_tensor.shape_signature)]
        if grid_tensor is not None and grid_tensor.shape_signature is not None
        else list(grid_shape)
    )
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else list(output_shape)
    )

    def _shape_dim_from_signature(
        signature: List[int],
        index: int,
        fallback: int,
    ) -> int:
        if int(index) < len(signature):
            value = int(signature[int(index)])
            return int(value) if int(value) > 0 else -1
        return int(fallback)

    grid_shape = [
        _shape_dim_from_signature(grid_signature, idx, dim)
        for idx, dim in enumerate(grid_shape)
    ]
    output_shape = [
        _shape_dim_from_signature(output_signature, idx, dim)
        for idx, dim in enumerate(output_shape)
    ]

    image_rank = int(len(image_shape))
    if image_rank == 4:
        n, c, h, w = [int(v) for v in image_shape]
        out_n, out_c, out_h, out_w = [int(v) for v in output_shape]
    elif image_rank == 5:
        n, c, d, h, w = [int(v) for v in image_shape]
        out_n, out_c, out_d, out_h, out_w = [int(v) for v in output_shape]
    else:
        raise NotImplementedError(
            f"GridSample supports rank-4/5 tensors in flatbuffer_direct. op={node.name} image_shape={image_shape}"
        )
    align_corners = bool(int(node.attrs.get("align_corners", 0)))

    image_dtype = str(ctx.get_tensor_dtype(image_name)).upper()
    grid_dtype = str(ctx.get_tensor_dtype(grid_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = (
        "FLOAT32"
        if image_dtype == "FLOAT32" or grid_dtype == "FLOAT32"
        else "FLOAT16"
    )
    compute_np_dtype = np.float16 if compute_dtype == "FLOAT16" else np.float32
    replace_to_pseudo_operators = getattr(ctx, "replace_to_pseudo_operators", set())
    rtpo_gather = "gather" in set(replace_to_pseudo_operators or set())

    def _add_float_const(base_name: str, value: float) -> str:
        return ctx.add_const_tensor(
            base_name,
            np.asarray(value, dtype=compute_np_dtype),
        )

    def _add_binary_op(op_type: str, lhs: str, rhs: str, out: str) -> None:
        options: dict[str, Any] = {}
        if op_type in {"ADD", "SUB", "MUL", "DIV"}:
            options = {"fusedActivationFunction": "NONE"}
        ctx.add_operator(
            OperatorIR(
                op_type=op_type,
                inputs=[lhs, rhs],
                outputs=[out],
                options=options,
            )
        )

    def _squeeze_last_dim(name: str, tag: str, dtype: str) -> str:
        squeezed = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_squeezed",
            dtype=dtype,
            shape=[int(n), int(out_h), int(out_w)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[name],
                outputs=[squeezed],
                options={"squeezeDims": [3]},
            )
        )
        return squeezed

    def _build_linear_index(y_idx: str, x_idx: str, tag: str, width_const: str) -> str:
        mul_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_mul",
            dtype="INT32",
            shape=[int(n), int(out_h), int(out_w)],
        )
        linear_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_linear",
            dtype="INT32",
            shape=[int(n), int(out_h), int(out_w)],
        )
        _add_binary_op("MUL", y_idx, width_const, mul_name)
        _add_binary_op("ADD", mul_name, x_idx, linear_name)
        return linear_name

    def _build_gather_rtpo(
        *,
        linear_idx_name: str,
        params_name: str,
        gathered_name: str,
        tag: str,
        spatial_shape: List[int],
        flattened_axis_size: int,
    ) -> None:
        if not bool(rtpo_gather):
            ctx.add_operator(
                OperatorIR(
                    op_type="GATHER",
                    inputs=[params_name, linear_idx_name],
                    outputs=[gathered_name],
                    options={
                        "axis": 2,
                        "batchDims": 1,
                    },
                )
            )
            return

        batch_size = int(n)
        channels = int(c)
        spatial_dims = [int(v) for v in list(spatial_shape)]
        if (
            int(batch_size) <= 0
            or int(channels) <= 0
            or int(flattened_axis_size) <= 0
            or any(int(v) <= 0 for v in spatial_dims)
        ):
            ctx.add_operator(
                OperatorIR(
                    op_type="GATHER",
                    inputs=[params_name, linear_idx_name],
                    outputs=[gathered_name],
                    options={
                        "axis": 2,
                        "batchDims": 1,
                    },
                )
            )
            return

        indices_shape = [int(batch_size)] + [int(v) for v in spatial_dims]
        offsets_shape = [int(batch_size)] + [1] * len(spatial_dims)
        batch_offsets = (
            np.arange(int(batch_size), dtype=np.int32).reshape(offsets_shape)
            * np.asarray(int(flattened_axis_size), dtype=np.int32)
        )
        offsets_name = ctx.add_const_tensor(
            f"{output_name}_gridsample_{tag}_gather_offsets",
            batch_offsets,
        )
        global_idx_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_gather_global_idx",
            dtype="INT32",
            shape=indices_shape,
        )
        _add_binary_op("ADD", linear_idx_name, offsets_name, global_idx_name)

        flat_index_count = int(np.prod(np.asarray(indices_shape, dtype=np.int64)))
        global_idx_flat_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_gather_global_idx_flat",
            dtype="INT32",
            shape=[int(flat_index_count)],
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=global_idx_name,
            output_name=global_idx_flat_name,
            new_shape=[int(flat_index_count)],
            preserve_dynamic_shape=True,
        )

        params_nlc_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_params_nlc",
            dtype=compute_dtype,
            shape=[int(batch_size), int(flattened_axis_size), int(channels)],
        )
        params_nlc_name = make_transpose(
            ctx=ctx,
            input_name=params_name,
            output_name=params_nlc_name,
            perm_values=[0, 2, 1],
            allow_elide_inverse_chain=True,
        )
        params_linear_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_params_linear",
            dtype=compute_dtype,
            shape=[int(batch_size) * int(flattened_axis_size), int(channels)],
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=params_nlc_name,
            output_name=params_linear_name,
            new_shape=[int(batch_size) * int(flattened_axis_size), int(channels)],
            preserve_dynamic_shape=True,
        )

        gathered_flat_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_gather_flat",
            dtype=compute_dtype,
            shape=[int(flat_index_count), int(channels)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[params_linear_name, global_idx_flat_name],
                outputs=[gathered_flat_name],
                options={
                    "axis": 0,
                    "batchDims": 0,
                },
            )
        )

        gathered_nspatialc_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_gather_nspatialc",
            dtype=compute_dtype,
            shape=[int(batch_size)] + [int(v) for v in spatial_dims] + [int(channels)],
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=gathered_flat_name,
            output_name=gathered_nspatialc_name,
            new_shape=[int(batch_size)] + [int(v) for v in spatial_dims] + [int(channels)],
            preserve_dynamic_shape=True,
        )
        gather_perm = [0, len(spatial_dims) + 1] + [idx + 1 for idx in range(len(spatial_dims))]
        make_transpose(
            ctx=ctx,
            input_name=gathered_nspatialc_name,
            output_name=gathered_name,
            perm_values=[int(v) for v in gather_perm],
            allow_elide_inverse_chain=True,
        )

    def _build_gather(linear_idx_name: str, tag: str, params_name: str) -> str:
        gathered_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_{tag}_gather",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_h), int(out_w)],
        )
        _build_gather_rtpo(
            linear_idx_name=linear_idx_name,
            params_name=params_name,
            gathered_name=gathered_name,
            tag=tag,
            spatial_shape=[int(out_h), int(out_w)],
            flattened_axis_size=int((w + 2) * (h + 2)),
        )
        return gathered_name

    image_compute_name = image_name
    if image_dtype != compute_dtype:
        image_compute_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_image_{compute_dtype.lower()}",
            dtype=compute_dtype,
            shape=image_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[image_name],
                outputs=[image_compute_name],
                options={
                    "inDataType": image_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    grid_compute_name = grid_name
    if grid_dtype != compute_dtype:
        grid_compute_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_grid_{compute_dtype.lower()}",
            dtype=compute_dtype,
            shape=grid_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[grid_name],
                outputs=[grid_compute_name],
                options={
                    "inDataType": grid_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    if image_rank == 5:
        def _squeeze_last_dim_3d(name: str, tag: str, dtype: str) -> str:
            squeezed = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_{tag}_squeezed",
                dtype=dtype,
                shape=[int(n), int(out_d), int(out_h), int(out_w)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SQUEEZE",
                    inputs=[name],
                    outputs=[squeezed],
                    options={"squeezeDims": [4]},
                )
            )
            return squeezed

        def _build_linear_index_3d(z_idx: str, y_idx: str, x_idx: str, tag: str) -> str:
            z_mul_name = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_{tag}_z_mul",
                dtype="INT32",
                shape=[int(n), int(out_d), int(out_h), int(out_w)],
            )
            y_mul_name = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_{tag}_y_mul",
                dtype="INT32",
                shape=[int(n), int(out_d), int(out_h), int(out_w)],
            )
            zy_sum_name = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_{tag}_zy_sum",
                dtype="INT32",
                shape=[int(n), int(out_d), int(out_h), int(out_w)],
            )
            linear_name = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_{tag}_linear",
                dtype="INT32",
                shape=[int(n), int(out_d), int(out_h), int(out_w)],
            )
            _add_binary_op("MUL", z_idx, linear_plane_const, z_mul_name)
            _add_binary_op("MUL", y_idx, linear_width_const, y_mul_name)
            _add_binary_op("ADD", z_mul_name, y_mul_name, zy_sum_name)
            _add_binary_op("ADD", zy_sum_name, x_idx, linear_name)
            return linear_name

        def _build_gather_3d(linear_idx_name: str, tag: str, params_name: str) -> str:
            gathered_name = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_{tag}_gather",
                dtype=compute_dtype,
                shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
            )
            _build_gather_rtpo(
                linear_idx_name=linear_idx_name,
                params_name=params_name,
                gathered_name=gathered_name,
                tag=tag,
                spatial_shape=[int(out_d), int(out_h), int(out_w)],
                flattened_axis_size=int((d + 2) * (h + 2) * (w + 2)),
            )
            return gathered_name

        paddings_name = ctx.add_const_tensor(
            f"{output_name}_gridsample_paddings",
            np.asarray([[0, 0], [0, 0], [1, 1], [1, 1], [1, 1]], dtype=np.int32),
        )
        image_padded_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_image_padded",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(d + 2), int(h + 2), int(w + 2)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="PAD",
                inputs=[image_compute_name, paddings_name],
                outputs=[image_padded_name],
            )
        )
        image_flat_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_image_flat",
            dtype=compute_dtype,
            shape=[int(n), int(c), int((d + 2) * (h + 2) * (w + 2))],
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=image_padded_name,
            output_name=image_flat_name,
            new_shape=[-1, int(c), int((d + 2) * (h + 2) * (w + 2))],
            preserve_dynamic_shape=True,
        )

        grid_x_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_grid_x",
            dtype=compute_dtype,
            shape=[int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]), int(grid_shape[3]), 1],
        )
        grid_y_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_grid_y",
            dtype=compute_dtype,
            shape=[int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]), int(grid_shape[3]), 1],
        )
        grid_z_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_grid_z",
            dtype=compute_dtype,
            shape=[int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]), int(grid_shape[3]), 1],
        )
        grid_x_index = ctx.add_const_tensor(
            f"{output_name}_gridsample_grid_x_index",
            np.asarray([0], dtype=np.int32),
        )
        grid_y_index = ctx.add_const_tensor(
            f"{output_name}_gridsample_grid_y_index",
            np.asarray([1], dtype=np.int32),
        )
        grid_z_index = ctx.add_const_tensor(
            f"{output_name}_gridsample_grid_z_index",
            np.asarray([2], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[grid_compute_name, grid_x_index],
                outputs=[grid_x_name],
                options={"axis": 4, "batchDims": 0},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[grid_compute_name, grid_y_index],
                outputs=[grid_y_name],
                options={"axis": 4, "batchDims": 0},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[grid_compute_name, grid_z_index],
                outputs=[grid_z_name],
                options={"axis": 4, "batchDims": 0},
            )
        )

        one_const = _add_float_const(f"{output_name}_gridsample_one", 1.0)
        zero_const = _add_float_const(f"{output_name}_gridsample_zero", 0.0)
        half_const = _add_float_const(f"{output_name}_gridsample_half", 0.5)
        neg_one_const = _add_float_const(f"{output_name}_gridsample_neg_one", -1.0)
        w_const = _add_float_const(f"{output_name}_gridsample_w", float(w))
        h_const = _add_float_const(f"{output_name}_gridsample_h", float(h))
        d_const = _add_float_const(f"{output_name}_gridsample_d", float(d))
        w_pad_max_const = _add_float_const(f"{output_name}_gridsample_w_pad_max", float(w + 1))
        h_pad_max_const = _add_float_const(f"{output_name}_gridsample_h_pad_max", float(h + 1))
        d_pad_max_const = _add_float_const(f"{output_name}_gridsample_d_pad_max", float(d + 1))
        x_scale_const = _add_float_const(
            f"{output_name}_gridsample_x_scale",
            float((w - 1) * 0.5 if align_corners else w * 0.5),
        )
        y_scale_const = _add_float_const(
            f"{output_name}_gridsample_y_scale",
            float((h - 1) * 0.5 if align_corners else h * 0.5),
        )
        z_scale_const = _add_float_const(
            f"{output_name}_gridsample_z_scale",
            float((d - 1) * 0.5 if align_corners else d * 0.5),
        )

        x_plus_one = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x_plus_one",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y_plus_one = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y_plus_one",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z_plus_one = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z_plus_one",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        x_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        _add_binary_op("ADD", grid_x_name, one_const, x_plus_one)
        _add_binary_op("ADD", grid_y_name, one_const, y_plus_one)
        _add_binary_op("ADD", grid_z_name, one_const, z_plus_one)
        x_name_pre = x_name
        y_name_pre = y_name
        z_name_pre = z_name
        if not align_corners:
            x_name_pre = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_x_pre",
                dtype=compute_dtype,
                shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
            )
            y_name_pre = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_y_pre",
                dtype=compute_dtype,
                shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
            )
            z_name_pre = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_z_pre",
                dtype=compute_dtype,
                shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
            )
        _add_binary_op("MUL", x_plus_one, x_scale_const, x_name_pre)
        _add_binary_op("MUL", y_plus_one, y_scale_const, y_name_pre)
        _add_binary_op("MUL", z_plus_one, z_scale_const, z_name_pre)
        if not align_corners:
            _add_binary_op("SUB", x_name_pre, half_const, x_name)
            _add_binary_op("SUB", y_name_pre, half_const, y_name)
            _add_binary_op("SUB", z_name_pre, half_const, z_name)

        x_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        x_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        x_shift_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x_shift",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y_shift_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y_shift",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z_shift_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z_shift",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        _add_binary_op("MAXIMUM", x_name, neg_one_const, x_clip_low_name)
        _add_binary_op("MINIMUM", x_clip_low_name, w_const, x_clip_name)
        _add_binary_op("MAXIMUM", y_name, neg_one_const, y_clip_low_name)
        _add_binary_op("MINIMUM", y_clip_low_name, h_const, y_clip_name)
        _add_binary_op("MAXIMUM", z_name, neg_one_const, z_clip_low_name)
        _add_binary_op("MINIMUM", z_clip_low_name, d_const, z_clip_name)
        _add_binary_op("ADD", x_clip_name, one_const, x_shift_name)
        _add_binary_op("ADD", y_clip_name, one_const, y_shift_name)
        _add_binary_op("ADD", z_clip_name, one_const, z_shift_name)

        x0_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x0",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y0_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y0",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z0_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z0",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        x1_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x1",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y1_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y1",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z1_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z1",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        ctx.add_operator(OperatorIR(op_type="FLOOR", inputs=[x_shift_name], outputs=[x0_name]))
        ctx.add_operator(OperatorIR(op_type="FLOOR", inputs=[y_shift_name], outputs=[y0_name]))
        ctx.add_operator(OperatorIR(op_type="FLOOR", inputs=[z_shift_name], outputs=[z0_name]))
        _add_binary_op("ADD", x0_name, one_const, x1_name)
        _add_binary_op("ADD", y0_name, one_const, y1_name)
        _add_binary_op("ADD", z0_name, one_const, z1_name)

        x0_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x0_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        x0_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x0_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        x1_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x1_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        x1_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x1_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y0_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y0_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y0_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y0_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y1_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y1_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y1_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y1_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z0_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z0_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z0_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z0_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z1_clip_low_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z1_clip_low",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z1_clip_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z1_clip",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        _add_binary_op("MAXIMUM", x0_name, zero_const, x0_clip_low_name)
        _add_binary_op("MINIMUM", x0_clip_low_name, w_pad_max_const, x0_clip_name)
        _add_binary_op("MAXIMUM", x1_name, zero_const, x1_clip_low_name)
        _add_binary_op("MINIMUM", x1_clip_low_name, w_pad_max_const, x1_clip_name)
        _add_binary_op("MAXIMUM", y0_name, zero_const, y0_clip_low_name)
        _add_binary_op("MINIMUM", y0_clip_low_name, h_pad_max_const, y0_clip_name)
        _add_binary_op("MAXIMUM", y1_name, zero_const, y1_clip_low_name)
        _add_binary_op("MINIMUM", y1_clip_low_name, h_pad_max_const, y1_clip_name)
        _add_binary_op("MAXIMUM", z0_name, zero_const, z0_clip_low_name)
        _add_binary_op("MINIMUM", z0_clip_low_name, d_pad_max_const, z0_clip_name)
        _add_binary_op("MAXIMUM", z1_name, zero_const, z1_clip_low_name)
        _add_binary_op("MINIMUM", z1_clip_low_name, d_pad_max_const, z1_clip_name)

        dx_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_dx",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        dy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_dy",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        dz_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_dz",
            dtype=compute_dtype,
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        _add_binary_op("SUB", x_shift_name, x0_clip_name, dx_name)
        _add_binary_op("SUB", y_shift_name, y0_clip_name, dy_name)
        _add_binary_op("SUB", z_shift_name, z0_clip_name, dz_name)

        x0_i_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x0_i",
            dtype="INT32",
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        x1_i_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x1_i",
            dtype="INT32",
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y0_i_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y0_i",
            dtype="INT32",
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        y1_i_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y1_i",
            dtype="INT32",
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z0_i_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z0_i",
            dtype="INT32",
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        z1_i_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_z1_i",
            dtype="INT32",
            shape=[int(n), int(out_d), int(out_h), int(out_w), 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[x0_clip_name],
                outputs=[x0_i_name],
                options={"inDataType": compute_dtype, "outDataType": "INT32"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[x1_clip_name],
                outputs=[x1_i_name],
                options={"inDataType": compute_dtype, "outDataType": "INT32"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[y0_clip_name],
                outputs=[y0_i_name],
                options={"inDataType": compute_dtype, "outDataType": "INT32"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[y1_clip_name],
                outputs=[y1_i_name],
                options={"inDataType": compute_dtype, "outDataType": "INT32"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[z0_clip_name],
                outputs=[z0_i_name],
                options={"inDataType": compute_dtype, "outDataType": "INT32"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[z1_clip_name],
                outputs=[z1_i_name],
                options={"inDataType": compute_dtype, "outDataType": "INT32"},
            )
        )

        x0_4d = _squeeze_last_dim_3d(x0_i_name, "x0", "INT32")
        x1_4d = _squeeze_last_dim_3d(x1_i_name, "x1", "INT32")
        y0_4d = _squeeze_last_dim_3d(y0_i_name, "y0", "INT32")
        y1_4d = _squeeze_last_dim_3d(y1_i_name, "y1", "INT32")
        z0_4d = _squeeze_last_dim_3d(z0_i_name, "z0", "INT32")
        z1_4d = _squeeze_last_dim_3d(z1_i_name, "z1", "INT32")
        linear_width_const = ctx.add_const_tensor(
            f"{output_name}_gridsample_linear_width",
            np.asarray(int(w + 2), dtype=np.int32),
        )
        linear_plane_const = ctx.add_const_tensor(
            f"{output_name}_gridsample_linear_plane",
            np.asarray(int((h + 2) * (w + 2)), dtype=np.int32),
        )
        idx000_name = _build_linear_index_3d(z0_4d, y0_4d, x0_4d, "idx000")
        idx001_name = _build_linear_index_3d(z0_4d, y0_4d, x1_4d, "idx001")
        idx010_name = _build_linear_index_3d(z0_4d, y1_4d, x0_4d, "idx010")
        idx011_name = _build_linear_index_3d(z0_4d, y1_4d, x1_4d, "idx011")
        idx100_name = _build_linear_index_3d(z1_4d, y0_4d, x0_4d, "idx100")
        idx101_name = _build_linear_index_3d(z1_4d, y0_4d, x1_4d, "idx101")
        idx110_name = _build_linear_index_3d(z1_4d, y1_4d, x0_4d, "idx110")
        idx111_name = _build_linear_index_3d(z1_4d, y1_4d, x1_4d, "idx111")

        val000_name = _build_gather_3d(idx000_name, "val000", image_flat_name)
        val001_name = _build_gather_3d(idx001_name, "val001", image_flat_name)
        val010_name = _build_gather_3d(idx010_name, "val010", image_flat_name)
        val011_name = _build_gather_3d(idx011_name, "val011", image_flat_name)
        val100_name = _build_gather_3d(idx100_name, "val100", image_flat_name)
        val101_name = _build_gather_3d(idx101_name, "val101", image_flat_name)
        val110_name = _build_gather_3d(idx110_name, "val110", image_flat_name)
        val111_name = _build_gather_3d(idx111_name, "val111", image_flat_name)

        dx_n1dhw = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_dx_n1dhw",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        dy_n1dhw = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_dy_n1dhw",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        dz_n1dhw = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_dz_n1dhw",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        dx_n1dhw = make_transpose(
            ctx=ctx,
            input_name=dx_name,
            output_name=dx_n1dhw,
            perm_values=[0, 4, 1, 2, 3],
            allow_elide_inverse_chain=True,
        )
        dy_n1dhw = make_transpose(
            ctx=ctx,
            input_name=dy_name,
            output_name=dy_n1dhw,
            perm_values=[0, 4, 1, 2, 3],
            allow_elide_inverse_chain=True,
        )
        dz_n1dhw = make_transpose(
            ctx=ctx,
            input_name=dz_name,
            output_name=dz_n1dhw,
            perm_values=[0, 4, 1, 2, 3],
            allow_elide_inverse_chain=True,
        )

        one_minus_dx_n1dhw = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_one_minus_dx",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        one_minus_dy_n1dhw = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_one_minus_dy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        one_minus_dz_n1dhw = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_one_minus_dz",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        _add_binary_op("SUB", one_const, dx_n1dhw, one_minus_dx_n1dhw)
        _add_binary_op("SUB", one_const, dy_n1dhw, one_minus_dy_n1dhw)
        _add_binary_op("SUB", one_const, dz_n1dhw, one_minus_dz_n1dhw)

        w000_xy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w000_xy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w001_xy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w001_xy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w010_xy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w010_xy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w011_xy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w011_xy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w100_xy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w100_xy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w101_xy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w101_xy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w110_xy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w110_xy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w111_xy_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w111_xy",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        _add_binary_op("MUL", one_minus_dx_n1dhw, one_minus_dy_n1dhw, w000_xy_name)
        _add_binary_op("MUL", dx_n1dhw, one_minus_dy_n1dhw, w001_xy_name)
        _add_binary_op("MUL", one_minus_dx_n1dhw, dy_n1dhw, w010_xy_name)
        _add_binary_op("MUL", dx_n1dhw, dy_n1dhw, w011_xy_name)
        _add_binary_op("MUL", one_minus_dx_n1dhw, one_minus_dy_n1dhw, w100_xy_name)
        _add_binary_op("MUL", dx_n1dhw, one_minus_dy_n1dhw, w101_xy_name)
        _add_binary_op("MUL", one_minus_dx_n1dhw, dy_n1dhw, w110_xy_name)
        _add_binary_op("MUL", dx_n1dhw, dy_n1dhw, w111_xy_name)

        w000_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w000",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w001_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w001",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w010_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w010",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w011_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w011",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w100_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w100",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w101_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w101",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w110_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w110",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        w111_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_w111",
            dtype=compute_dtype,
            shape=[int(n), 1, int(out_d), int(out_h), int(out_w)],
        )
        _add_binary_op("MUL", w000_xy_name, one_minus_dz_n1dhw, w000_name)
        _add_binary_op("MUL", w001_xy_name, one_minus_dz_n1dhw, w001_name)
        _add_binary_op("MUL", w010_xy_name, one_minus_dz_n1dhw, w010_name)
        _add_binary_op("MUL", w011_xy_name, one_minus_dz_n1dhw, w011_name)
        _add_binary_op("MUL", w100_xy_name, dz_n1dhw, w100_name)
        _add_binary_op("MUL", w101_xy_name, dz_n1dhw, w101_name)
        _add_binary_op("MUL", w110_xy_name, dz_n1dhw, w110_name)
        _add_binary_op("MUL", w111_xy_name, dz_n1dhw, w111_name)

        term000_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_term000",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        term001_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_term001",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        term010_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_term010",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        term011_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_term011",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        term100_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_term100",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        term101_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_term101",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        term110_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_term110",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        term111_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_term111",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        _add_binary_op("MUL", val000_name, w000_name, term000_name)
        _add_binary_op("MUL", val001_name, w001_name, term001_name)
        _add_binary_op("MUL", val010_name, w010_name, term010_name)
        _add_binary_op("MUL", val011_name, w011_name, term011_name)
        _add_binary_op("MUL", val100_name, w100_name, term100_name)
        _add_binary_op("MUL", val101_name, w101_name, term101_name)
        _add_binary_op("MUL", val110_name, w110_name, term110_name)
        _add_binary_op("MUL", val111_name, w111_name, term111_name)

        sum01_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_sum01",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        sum23_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_sum23",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        sum45_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_sum45",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        sum67_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_sum67",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        sum0123_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_sum0123",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        sum4567_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_sum4567",
            dtype=compute_dtype,
            shape=[int(n), int(c), int(out_d), int(out_h), int(out_w)],
        )
        output_compute_name = output_name
        if output_dtype != compute_dtype:
            output_compute_name = ctx.add_intermediate_tensor(
                f"{output_name}_gridsample_output_compute",
                dtype=compute_dtype,
                shape=[int(out_n), int(out_c), int(out_d), int(out_h), int(out_w)],
            )
        _add_binary_op("ADD", term000_name, term001_name, sum01_name)
        _add_binary_op("ADD", term010_name, term011_name, sum23_name)
        _add_binary_op("ADD", term100_name, term101_name, sum45_name)
        _add_binary_op("ADD", term110_name, term111_name, sum67_name)
        _add_binary_op("ADD", sum01_name, sum23_name, sum0123_name)
        _add_binary_op("ADD", sum45_name, sum67_name, sum4567_name)
        _add_binary_op("ADD", sum0123_name, sum4567_name, output_compute_name)

        if output_dtype != compute_dtype:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[output_compute_name],
                    outputs=[output_name],
                    options={
                        "inDataType": compute_dtype,
                        "outDataType": output_dtype,
                    },
                )
            )

        in_quant = ctx.model_ir.tensors[image_name].quantization
        if in_quant is not None and ctx.model_ir.tensors[output_name].quantization is None:
            ctx.model_ir.tensors[output_name].quantization = _clone_quantization(in_quant)
        return

    # zeros-padding fast path (same idea as tf backend): pad by 1 on H/W to
    # eliminate explicit in-bound mask ops and gather zeros from border.
    paddings_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_paddings",
        np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int32),
    )
    image_padded_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_image_padded",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(h + 2), int(w + 2)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="PAD",
            inputs=[image_compute_name, paddings_name],
            outputs=[image_padded_name],
        )
    )
    image_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_image_flat",
        dtype=compute_dtype,
        shape=[int(n), int(c), int((h + 2) * (w + 2))],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=image_padded_name,
        output_name=image_flat_name,
        new_shape=[-1, int(c), int((h + 2) * (w + 2))],
        preserve_dynamic_shape=True,
    )

    grid_x_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_grid_x",
        dtype=compute_dtype,
        shape=[int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]), 1],
    )
    grid_y_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_grid_y",
        dtype=compute_dtype,
        shape=[int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]), 1],
    )
    grid_x_index = ctx.add_const_tensor(
        f"{output_name}_gridsample_grid_x_index",
        np.asarray([0], dtype=np.int32),
    )
    grid_y_index = ctx.add_const_tensor(
        f"{output_name}_gridsample_grid_y_index",
        np.asarray([1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[grid_compute_name, grid_x_index],
            outputs=[grid_x_name],
            options={
                "axis": 3,
                "batchDims": 0,
            },
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[grid_compute_name, grid_y_index],
            outputs=[grid_y_name],
            options={
                "axis": 3,
                "batchDims": 0,
            },
        )
    )

    one_const = _add_float_const(f"{output_name}_gridsample_one", 1.0)
    zero_const = _add_float_const(f"{output_name}_gridsample_zero", 0.0)
    half_const = _add_float_const(f"{output_name}_gridsample_half", 0.5)
    neg_one_const = _add_float_const(f"{output_name}_gridsample_neg_one", -1.0)
    w_const = _add_float_const(f"{output_name}_gridsample_w", float(w))
    h_const = _add_float_const(f"{output_name}_gridsample_h", float(h))
    w_pad_max_const = _add_float_const(f"{output_name}_gridsample_w_pad_max", float(w + 1))
    h_pad_max_const = _add_float_const(f"{output_name}_gridsample_h_pad_max", float(h + 1))
    x_scale_const = _add_float_const(
        f"{output_name}_gridsample_x_scale",
        float((w - 1) * 0.5 if align_corners else w * 0.5),
    )
    y_scale_const = _add_float_const(
        f"{output_name}_gridsample_y_scale",
        float((h - 1) * 0.5 if align_corners else h * 0.5),
    )

    x_plus_one = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x_plus_one",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y_plus_one = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y_plus_one",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    x_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    _add_binary_op("ADD", grid_x_name, one_const, x_plus_one)
    _add_binary_op("ADD", grid_y_name, one_const, y_plus_one)
    x_name_pre = x_name
    y_name_pre = y_name
    if not align_corners:
        x_name_pre = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_x_pre",
            dtype=compute_dtype,
            shape=[int(n), int(out_h), int(out_w), 1],
        )
        y_name_pre = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_y_pre",
            dtype=compute_dtype,
            shape=[int(n), int(out_h), int(out_w), 1],
        )
    _add_binary_op("MUL", x_plus_one, x_scale_const, x_name_pre)
    _add_binary_op("MUL", y_plus_one, y_scale_const, y_name_pre)
    if not align_corners:
        _add_binary_op("SUB", x_name_pre, half_const, x_name)
        _add_binary_op("SUB", y_name_pre, half_const, y_name)

    x_clip_low_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x_clip_low",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    x_clip_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x_clip",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y_clip_low_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y_clip_low",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y_clip_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y_clip",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    x_shift_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x_shift",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y_shift_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y_shift",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    _add_binary_op("MAXIMUM", x_name, neg_one_const, x_clip_low_name)
    _add_binary_op("MINIMUM", x_clip_low_name, w_const, x_clip_name)
    _add_binary_op("MAXIMUM", y_name, neg_one_const, y_clip_low_name)
    _add_binary_op("MINIMUM", y_clip_low_name, h_const, y_clip_name)
    _add_binary_op("ADD", x_clip_name, one_const, x_shift_name)
    _add_binary_op("ADD", y_clip_name, one_const, y_shift_name)

    x0_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x0",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y0_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y0",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    x1_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x1",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y1_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y1",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    ctx.add_operator(OperatorIR(op_type="FLOOR", inputs=[x_shift_name], outputs=[x0_name]))
    ctx.add_operator(OperatorIR(op_type="FLOOR", inputs=[y_shift_name], outputs=[y0_name]))
    _add_binary_op("ADD", x0_name, one_const, x1_name)
    _add_binary_op("ADD", y0_name, one_const, y1_name)

    x0_clip_low_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x0_clip_low",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    x0_clip_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x0_clip",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    x1_clip_low_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x1_clip_low",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    x1_clip_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x1_clip",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y0_clip_low_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y0_clip_low",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y0_clip_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y0_clip",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y1_clip_low_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y1_clip_low",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y1_clip_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y1_clip",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    _add_binary_op("MAXIMUM", x0_name, zero_const, x0_clip_low_name)
    _add_binary_op("MINIMUM", x0_clip_low_name, w_pad_max_const, x0_clip_name)
    _add_binary_op("MAXIMUM", x1_name, zero_const, x1_clip_low_name)
    _add_binary_op("MINIMUM", x1_clip_low_name, w_pad_max_const, x1_clip_name)
    _add_binary_op("MAXIMUM", y0_name, zero_const, y0_clip_low_name)
    _add_binary_op("MINIMUM", y0_clip_low_name, h_pad_max_const, y0_clip_name)
    _add_binary_op("MAXIMUM", y1_name, zero_const, y1_clip_low_name)
    _add_binary_op("MINIMUM", y1_clip_low_name, h_pad_max_const, y1_clip_name)

    dx_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_dx",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    dy_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_dy",
        dtype=compute_dtype,
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    _add_binary_op("SUB", x_shift_name, x0_clip_name, dx_name)
    _add_binary_op("SUB", y_shift_name, y0_clip_name, dy_name)

    x0_i_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x0_i",
        dtype="INT32",
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    x1_i_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_x1_i",
        dtype="INT32",
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y0_i_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y0_i",
        dtype="INT32",
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    y1_i_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_y1_i",
        dtype="INT32",
        shape=[int(n), int(out_h), int(out_w), 1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[x0_clip_name],
            outputs=[x0_i_name],
            options={"inDataType": compute_dtype, "outDataType": "INT32"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[x1_clip_name],
            outputs=[x1_i_name],
            options={"inDataType": compute_dtype, "outDataType": "INT32"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[y0_clip_name],
            outputs=[y0_i_name],
            options={"inDataType": compute_dtype, "outDataType": "INT32"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[y1_clip_name],
            outputs=[y1_i_name],
            options={"inDataType": compute_dtype, "outDataType": "INT32"},
        )
    )

    x0_3d = _squeeze_last_dim(x0_i_name, "x0", "INT32")
    x1_3d = _squeeze_last_dim(x1_i_name, "x1", "INT32")
    y0_3d = _squeeze_last_dim(y0_i_name, "y0", "INT32")
    y1_3d = _squeeze_last_dim(y1_i_name, "y1", "INT32")
    linear_width_const = ctx.add_const_tensor(
        f"{output_name}_gridsample_linear_width",
        np.asarray(int(w + 2), dtype=np.int32),
    )
    idx00_name = _build_linear_index(y0_3d, x0_3d, "idx00", linear_width_const)
    idx01_name = _build_linear_index(y1_3d, x0_3d, "idx01", linear_width_const)
    idx10_name = _build_linear_index(y0_3d, x1_3d, "idx10", linear_width_const)
    idx11_name = _build_linear_index(y1_3d, x1_3d, "idx11", linear_width_const)

    val00_name = _build_gather(idx00_name, "val00", image_flat_name)
    val01_name = _build_gather(idx01_name, "val01", image_flat_name)
    val10_name = _build_gather(idx10_name, "val10", image_flat_name)
    val11_name = _build_gather(idx11_name, "val11", image_flat_name)

    dx_n1hw = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_dx_n1hw",
        dtype=compute_dtype,
        shape=[int(n), 1, int(out_h), int(out_w)],
    )
    dy_n1hw = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_dy_n1hw",
        dtype=compute_dtype,
        shape=[int(n), 1, int(out_h), int(out_w)],
    )
    dx_n1hw = make_transpose(
        ctx=ctx,
        input_name=dx_name,
        output_name=dx_n1hw,
        perm_values=[0, 3, 1, 2],
        allow_elide_inverse_chain=True,
    )
    dy_n1hw = make_transpose(
        ctx=ctx,
        input_name=dy_name,
        output_name=dy_n1hw,
        perm_values=[0, 3, 1, 2],
        allow_elide_inverse_chain=True,
    )

    one_minus_dx_n1hw = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_one_minus_dx",
        dtype=compute_dtype,
        shape=[int(n), 1, int(out_h), int(out_w)],
    )
    one_minus_dy_n1hw = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_one_minus_dy",
        dtype=compute_dtype,
        shape=[int(n), 1, int(out_h), int(out_w)],
    )
    _add_binary_op("SUB", one_const, dx_n1hw, one_minus_dx_n1hw)
    _add_binary_op("SUB", one_const, dy_n1hw, one_minus_dy_n1hw)

    w00_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_w00",
        dtype=compute_dtype,
        shape=[int(n), 1, int(out_h), int(out_w)],
    )
    w01_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_w01",
        dtype=compute_dtype,
        shape=[int(n), 1, int(out_h), int(out_w)],
    )
    w10_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_w10",
        dtype=compute_dtype,
        shape=[int(n), 1, int(out_h), int(out_w)],
    )
    w11_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_w11",
        dtype=compute_dtype,
        shape=[int(n), 1, int(out_h), int(out_w)],
    )
    _add_binary_op("MUL", one_minus_dx_n1hw, one_minus_dy_n1hw, w00_name)
    _add_binary_op("MUL", one_minus_dx_n1hw, dy_n1hw, w01_name)
    _add_binary_op("MUL", dx_n1hw, one_minus_dy_n1hw, w10_name)
    _add_binary_op("MUL", dx_n1hw, dy_n1hw, w11_name)

    term00_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_term00",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(out_h), int(out_w)],
    )
    term01_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_term01",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(out_h), int(out_w)],
    )
    term10_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_term10",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(out_h), int(out_w)],
    )
    term11_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_term11",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(out_h), int(out_w)],
    )
    _add_binary_op("MUL", val00_name, w00_name, term00_name)
    _add_binary_op("MUL", val01_name, w01_name, term01_name)
    _add_binary_op("MUL", val10_name, w10_name, term10_name)
    _add_binary_op("MUL", val11_name, w11_name, term11_name)

    sum0_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_sum0",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(out_h), int(out_w)],
    )
    sum1_name = ctx.add_intermediate_tensor(
        f"{output_name}_gridsample_sum1",
        dtype=compute_dtype,
        shape=[int(n), int(c), int(out_h), int(out_w)],
    )
    output_compute_name = output_name
    if output_dtype != compute_dtype:
        output_compute_name = ctx.add_intermediate_tensor(
            f"{output_name}_gridsample_output_compute",
            dtype=compute_dtype,
            shape=[int(out_n), int(out_c), int(out_h), int(out_w)],
        )
    _add_binary_op("ADD", term00_name, term01_name, sum0_name)
    _add_binary_op("ADD", term10_name, term11_name, sum1_name)
    _add_binary_op("ADD", sum0_name, sum1_name, output_compute_name)

    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[output_compute_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )

    in_quant = ctx.model_ir.tensors[image_name].quantization
    if in_quant is not None and ctx.model_ir.tensors[output_name].quantization is None:
        ctx.model_ir.tensors[output_name].quantization = _clone_quantization(in_quant)


def build_resize_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )
    existing_output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None and len(list(output_tensor.shape_signature)) == 4
        else None
    )
    if len(input_shape) != 4:
        if len(input_signature) == 4:
            _materialize_tensor_shape_from_signature(input_tensor, signature=input_signature)
            input_shape = [int(v) for v in list(input_tensor.shape)]
        elif _is_unresolved_placeholder_shape(input_shape, input_signature):
            _materialize_tensor_shape_from_signature(
                input_tensor,
                signature=[-1, -1, -1, -1],
            )
            input_shape = [int(v) for v in list(input_tensor.shape)]
            input_signature = [int(v) for v in list(input_tensor.shape_signature)]
    if len(input_shape) != 4:
        raise NotImplementedError(
            f"Resize supports only rank-4 tensors in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    has_dynamic_sizes_input = False
    original_inputs = _get_original_node_input_names(node, ctx)
    dynamic_sizes_name = ""
    if len(original_inputs) >= 4 and str(original_inputs[3]) != "":
        dynamic_sizes_name = str(original_inputs[3])
    elif len(original_inputs) == 2 and str(original_inputs[1]) != "":
        # _NodeWrap may compact Resize(x, "", "", sizes) into 2 inputs.
        dynamic_sizes_name = str(original_inputs[1])
    if dynamic_sizes_name != "":
        sizes_const = ctx.get_constant_array(dynamic_sizes_name)
        if sizes_const is None or int(np.asarray(sizes_const).size) == 0:
            sizes_dtype = str(ctx.get_tensor_dtype(dynamic_sizes_name)).upper()
            has_dynamic_sizes_input = sizes_dtype in {"INT32", "INT64"}
    if len(output_shape) != 4 or _is_unresolved_placeholder_shape(output_shape, existing_output_signature) or output_shape == [1]:
        if has_dynamic_sizes_input:
            output_shape = [int(input_shape[0]), int(input_shape[1]), 1, 1]
        else:
            out_h, out_w = _resolve_resize_target_hw(node, ctx, input_shape)
            output_shape = [int(input_shape[0]), int(input_shape[1]), int(out_h), int(out_w)]
        ctx.model_ir.tensors[output_name].shape = list(output_shape)

    input_dtype = str(ctx.get_tensor_dtype(input_name))
    output_dtype = str(ctx.get_tensor_dtype(output_name))
    if output_dtype == "FLOAT32" and input_dtype != "FLOAT32":
        ctx.model_ir.tensors[output_name].dtype = input_dtype
    if ctx.model_ir.tensors[output_name].quantization is None:
        in_quant = ctx.model_ir.tensors[input_name].quantization
        if in_quant is not None:
            ctx.model_ir.tensors[output_name].quantization = _clone_quantization(in_quant)

    mode, coordinate_transformation_mode, align_corners, half_pixel_centers = _resolve_resize_flags(node)
    tflite_op = "RESIZE_NEAREST_NEIGHBOR" if mode == "nearest" else "RESIZE_BILINEAR"
    onnx_sizes_hw, onnx_scales_hw = _extract_resize_onnx_hw_hints(node, ctx)
    output_signature = _infer_resize_output_signature_nchw(
        input_signature_nchw=input_signature,
        output_shape_nchw=output_shape,
        onnx_sizes_hw=onnx_sizes_hw,
        onnx_scales_hw=onnx_scales_hw,
        existing_output_signature_nchw=existing_output_signature,
    )
    output_tensor.shape_signature = [int(v) for v in list(output_signature)]

    nhwc_input_shape = [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
    nhwc_output_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[3]), int(output_shape[1])]
    nhwc_output_signature = [int(v) for v in list(nhwc_output_shape)]
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
    x_quant = ctx.model_ir.tensors[input_name].quantization
    if x_quant is not None:
        ctx.model_ir.tensors[x_nhwc].quantization = _clone_quantization(x_quant)
    x_nhwc = make_transpose(
        ctx,
        input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    y_quant = ctx.model_ir.tensors[output_name].quantization
    if y_quant is not None:
        ctx.model_ir.tensors[y_nhwc].quantization = _clone_quantization(y_quant)
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in list(nhwc_output_signature)]

    if mode == "cubic":
        exclude_outside_attr = node.attrs.get("exclude_outside", 0)
        try:
            exclude_outside = bool(int(exclude_outside_attr))
        except Exception:
            exclude_outside = bool(exclude_outside_attr)
        cubic_coeff_a_attr = node.attrs.get("cubic_coeff_a", -0.75)
        try:
            cubic_coeff_a = float(cubic_coeff_a_attr)
        except Exception:
            cubic_coeff_a = -0.75
        _build_resize_cubic_strict_op(
            node=node,
            ctx=ctx,
            x_nhwc=x_nhwc,
            y_nhwc=y_nhwc,
            input_signature_nchw=input_signature,
            output_signature_nchw=output_signature,
            coordinate_transformation_mode=str(coordinate_transformation_mode),
            cubic_coeff_a=float(cubic_coeff_a),
            exclude_outside=bool(exclude_outside),
        )
    else:
        size_input_name = ""
        dynamic_size_input_name = _build_resize_dynamic_size_input(node, ctx)
        has_dynamic_spatial_input = (
            len(input_signature) == 4 and (int(input_signature[2]) < 0 or int(input_signature[3]) < 0)
        )
        resize_scales_hw_int = _resolve_integer_resize_scales_hw(onnx_scales_hw)
        if dynamic_size_input_name is not None:
            size_input_name = dynamic_size_input_name
        elif has_dynamic_spatial_input and onnx_sizes_hw is None and onnx_scales_hw is not None:
            if resize_scales_hw_int is None:
                raise NotImplementedError(
                    f"Resize with dynamic spatial input supports integer scales only in flatbuffer_direct. "
                    f"op={node.name} scales_hw={onnx_scales_hw}"
                )
            input_hw_shape = ctx.add_intermediate_tensor(
                f"{node.name}_input_hw_shape",
                dtype="INT32",
                shape=[2],
            )
            shape_vec = ctx.add_intermediate_tensor(
                f"{node.name}_input_shape_vec",
                dtype="INT32",
                shape=[4],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SHAPE",
                    inputs=[x_nhwc],
                    outputs=[shape_vec],
                    options={"outType": "INT32"},
                )
            )
            shape_begin = ctx.add_const_tensor(
                f"{node.name}_shape_slice_begin",
                np.asarray([1], dtype=np.int32),
            )
            shape_size = ctx.add_const_tensor(
                f"{node.name}_shape_slice_size",
                np.asarray([2], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[shape_vec, shape_begin, shape_size],
                    outputs=[input_hw_shape],
                )
            )
            scales_const = ctx.add_const_tensor(
                f"{node.name}_resize_scales_hw_int",
                np.asarray([int(resize_scales_hw_int[0]), int(resize_scales_hw_int[1])], dtype=np.int32),
            )
            dynamic_size = ctx.add_intermediate_tensor(
                f"{node.name}_resize_size_dynamic",
                dtype="INT32",
                shape=[2],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MUL",
                    inputs=[input_hw_shape, scales_const],
                    outputs=[dynamic_size],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            size_input_name = dynamic_size
        else:
            size_const = ctx.add_const_tensor(
                f"{node.name}_resize_size",
                np.asarray([int(output_shape[2]), int(output_shape[3])], dtype=np.int32),
            )
            size_input_name = size_const
        ctx.add_operator(
            OperatorIR(
                op_type=tflite_op,
                inputs=[x_nhwc, size_input_name],
                outputs=[y_nhwc],
                options={
                    "alignCorners": bool(align_corners),
                    "halfPixelCenters": bool(half_pixel_centers),
                    "onnxSizesHW": list(onnx_sizes_hw) if onnx_sizes_hw is not None else None,
                    "onnxScalesHW": list(onnx_scales_hw) if onnx_scales_hw is not None else None,
                },
            )
        )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )
