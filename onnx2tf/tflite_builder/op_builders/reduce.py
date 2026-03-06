from __future__ import annotations

from typing import Any, List

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


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


def _normalize_axes(axes: List[int], rank: int, node_name: str) -> List[int]:
    normalized: List[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += rank
        if a < 0 or a >= rank:
            raise NotImplementedError(
                f"Reduce axis is out of range. op={node_name} axis={axis} rank={rank}"
            )
        if a not in normalized:
            normalized.append(a)
    return normalized


def _resolve_reduce_axes(
    node: Any,
    ctx: Any,
    input_rank: int,
    *,
    preserve_raw_axes: bool = False,
) -> List[int]:
    axes: List[int]
    if len(node.inputs) >= 2:
        axes_arr = ctx.get_constant_array(node.inputs[1].name)
        if axes_arr is None:
            raise NotImplementedError(
                f"Reduce axes must be constant for flatbuffer_direct. op={node.name}"
            )
        axes = [int(v) for v in np.asarray(axes_arr).reshape(-1).tolist()]
    else:
        attr_axes = node.attrs.get("axes", None)
        if attr_axes is None:
            axes = [int(v) for v in range(input_rank)]
        elif isinstance(attr_axes, (list, tuple)):
            axes = [int(v) for v in attr_axes]
        else:
            axes = [int(attr_axes)]

    if len(axes) == 0:
        if int(node.attrs.get("noop_with_empty_axes", 0)) == 1:
            return []
        axes = [int(v) for v in range(input_rank)]
    if preserve_raw_axes:
        # Keep raw ONNX axes when rank metadata is unreliable. TFLite accepts
        # negative axes for MEAN/REDUCE ops, and preserving -1 avoids
        # accidental remap to axis=0 from temporary rank-1 placeholders.
        deduped: List[int] = []
        for axis in axes:
            value = int(axis)
            if value not in deduped:
                deduped.append(value)
        return deduped
    return _normalize_axes(axes, input_rank, node.name)


def _resolve_reduce_input_rank(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
    input_shape: List[int],
    output_shape: List[int],
) -> int:
    """Resolve robust input rank for Reduce* ops.

    Some dynamic paths temporarily collapse input tensor metadata to rank-1
    placeholder shape (e.g. [1]) even though shape_signature keeps true rank.
    Using that collapsed rank incorrectly normalizes negative axes.
    """
    base_rank = int(len(input_shape))
    if base_rank > 1:
        return int(base_rank)

    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor is not None and input_tensor.shape_signature is not None
        else None
    )
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else None
    )

    candidate_ranks = [int(base_rank)]
    # Prefer ONNX node rank metadata when available. This is often stable even
    # when intermediate IR tensors are temporarily materialized as rank-1
    # placeholders during lowering.
    try:
        node_input_shape = getattr(node.inputs[0], "shape", None) if len(node.inputs) > 0 else None
        if node_input_shape is not None:
            candidate_ranks.append(int(len(list(node_input_shape))))
    except Exception:
        pass
    try:
        node_output_shape = getattr(node.outputs[0], "shape", None) if len(node.outputs) > 0 else None
        if node_output_shape is not None:
            candidate_ranks.append(int(len(list(node_output_shape))))
    except Exception:
        pass
    if input_signature is not None:
        candidate_ranks.append(int(len(input_signature)))
    if bool(int(node.attrs.get("keepdims", 1))):
        candidate_ranks.append(int(len(output_shape)))
        if output_signature is not None:
            candidate_ranks.append(int(len(output_signature)))

    resolved_rank = int(max(candidate_ranks))
    if resolved_rank <= 0:
        return int(base_rank)

    if _is_unresolved_placeholder_shape(input_shape, input_signature):
        return int(resolved_rank)
    if int(base_rank) == 1 and int(resolved_rank) > 1:
        return int(resolved_rank)
    return int(base_rank)


def _resolve_cumsum_axis(node: Any, ctx: Any, input_rank: int) -> int:
    axis_raw: int
    if len(node.inputs) >= 2 and str(node.inputs[1].name) != "":
        axis_arr = ctx.get_constant_array(node.inputs[1].name)
        if axis_arr is None:
            raise NotImplementedError(
                f"CumSum axis must be constant for flatbuffer_direct. op={node.name}"
            )
        axis_values = np.asarray(axis_arr).reshape(-1)
        if int(axis_values.size) != 1:
            raise NotImplementedError(
                f"CumSum axis must be scalar. op={node.name} axis_shape={list(np.asarray(axis_arr).shape)}"
            )
        axis_raw = int(axis_values[0])
    else:
        axis_raw = int(node.attrs.get("axis", 0))

    axis = int(axis_raw)
    if axis < 0:
        axis += int(input_rank)
    if axis < 0 or axis >= int(input_rank):
        raise NotImplementedError(
            f"CumSum axis is out of range. op={node.name} axis={axis_raw} rank={input_rank}"
        )
    return int(axis)


def _is_integer_dtype(dtype: str) -> bool:
    return str(dtype).upper() in {
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
    }


def _harmonize_reduce_input_output_dtype(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
) -> str:
    """
    Ensure Reduce* runtime dtype consistency.

    Flatbuffer-direct lowering prefers INT32 for index/control tensors. When a
    producer has already normalized INT64 -> INT32 but ONNX metadata keeps the
    downstream Reduce output as INT64, emitting REDUCE_* with mismatched
    input/output dtypes can lead to corrupted values. In integer-only mismatch
    cases, keep runtime dtype aligned to input tensor dtype.
    """
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if input_dtype == output_dtype:
        return str(input_name)

    if _is_integer_dtype(input_dtype) and _is_integer_dtype(output_dtype):
        output_tensor = ctx.model_ir.tensors.get(str(output_name), None)
        if output_tensor is not None:
            output_tensor.dtype = str(input_dtype)
        if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
            ctx.dtype_map[str(output_name)] = str(input_dtype)
        return str(input_name)

    cast_input_name = ctx.add_intermediate_tensor(
        f"{output_name}_{node.op_type.lower()}_input_cast",
        dtype=output_dtype,
        shape=[int(v) for v in ctx.get_tensor_shape(input_name)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[input_name],
            outputs=[cast_input_name],
            options={
                "inDataType": input_dtype,
                "outDataType": output_dtype,
            },
        )
    )
    return str(cast_input_name)


def build_cumsum_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    axis = _resolve_cumsum_axis(node=node, ctx=ctx, input_rank=len(input_shape))
    axis_const = ctx.add_const_tensor(
        f"{output_name}_cumsum_axis",
        np.asarray([axis], dtype=np.int32),
    )
    exclusive = bool(int(node.attrs.get("exclusive", 0)))
    reverse = bool(int(node.attrs.get("reverse", 0)))

    ctx.add_operator(
        OperatorIR(
            op_type="CUMSUM",
            inputs=[input_name, axis_const],
            outputs=[output_name],
            options={
                "exclusive": exclusive,
                "reverse": reverse,
            },
        )
    )


def _set_scalar_tensor_metadata(ctx: Any, tensor_name: str) -> None:
    tensor = ctx.model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return
    tensor.shape = []
    tensor.shape_signature = []


def build_cumprod_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) < 1:
        raise NotImplementedError(
            f"CumProd requires rank>=1 input. op={node.name} input_shape={input_shape}"
        )
    if any(int(v) <= 0 for v in input_shape):
        raise NotImplementedError(
            "CumProd builtin lowering requires static positive input shape in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )

    axis = _resolve_cumsum_axis(node=node, ctx=ctx, input_rank=len(input_shape))
    axis_size = int(input_shape[axis])
    exclusive = bool(int(node.attrs.get("exclusive", 0)))
    reverse = bool(int(node.attrs.get("reverse", 0)))

    working_input_name = str(input_name)
    reverse_axis_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_reverse_axis",
        np.asarray([int(axis)], dtype=np.int32),
    )
    if reverse:
        reverse_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_cumprod_reverse_input",
            dtype=input_dtype,
            shape=[int(v) for v in input_shape],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="REVERSE_V2",
                inputs=[working_input_name, reverse_axis_name],
                outputs=[reverse_input_name],
            )
        )
        working_input_name = str(reverse_input_name)

    zero_scalar_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_range_start",
        np.asarray(0, dtype=np.int32),
    )
    limit_scalar_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_range_limit",
        np.asarray(axis_size, dtype=np.int32),
    )
    one_scalar_i32_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_range_delta",
        np.asarray(1, dtype=np.int32),
    )
    _set_scalar_tensor_metadata(ctx, zero_scalar_name)
    _set_scalar_tensor_metadata(ctx, limit_scalar_name)
    _set_scalar_tensor_metadata(ctx, one_scalar_i32_name)

    range_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_range",
        dtype="INT32",
        shape=[int(axis_size)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RANGE",
            inputs=[zero_scalar_name, limit_scalar_name, one_scalar_i32_name],
            outputs=[range_name],
        )
    )

    row_shape = [int(axis_size), 1]
    col_shape = [1, int(axis_size)]
    row_shape_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_row_shape",
        np.asarray(row_shape, dtype=np.int32),
    )
    col_shape_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_col_shape",
        np.asarray(col_shape, dtype=np.int32),
    )
    row_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_row_indices",
        dtype="INT32",
        shape=[int(v) for v in row_shape],
    )
    col_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_col_indices",
        dtype="INT32",
        shape=[int(v) for v in col_shape],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[range_name, row_shape_name],
            outputs=[row_name],
            options={"newShape": [int(v) for v in row_shape]},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[range_name, col_shape_name],
            outputs=[col_name],
            options={"newShape": [int(v) for v in col_shape]},
        )
    )

    mask_2d_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_mask_2d",
        dtype="BOOL",
        shape=[int(axis_size), int(axis_size)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LESS" if exclusive else "LESS_EQUAL",
            inputs=[row_name, col_name],
            outputs=[mask_2d_name],
        )
    )

    rank = int(len(input_shape))
    expanded_shape = (
        [int(v) for v in input_shape[: axis + 1]]
        + [1]
        + [int(v) for v in input_shape[axis + 1 :]]
    )
    full_shape = (
        [int(v) for v in input_shape[:axis]]
        + [int(axis_size), int(axis_size)]
        + [int(v) for v in input_shape[axis + 1 :]]
    )

    input_expand_shape_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_input_expand_shape",
        np.asarray(expanded_shape, dtype=np.int32),
    )
    tiled_input_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_input_tiled",
        dtype=input_dtype,
        shape=[int(v) for v in full_shape],
    )
    expanded_input_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_input_expanded",
        dtype=input_dtype,
        shape=[int(v) for v in expanded_shape],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[working_input_name, input_expand_shape_name],
            outputs=[expanded_input_name],
            options={"newShape": [int(v) for v in expanded_shape]},
        )
    )
    input_tile_multiples = [1 for _ in range(rank + 1)]
    input_tile_multiples[axis + 1] = int(axis_size)
    input_tile_multiples_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_input_tile_multiples",
        np.asarray(input_tile_multiples, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="TILE",
            inputs=[expanded_input_name, input_tile_multiples_name],
            outputs=[tiled_input_name],
        )
    )

    mask_expand_shape = (
        [1 for _ in range(axis)]
        + [int(axis_size), int(axis_size)]
        + [1 for _ in range(rank - axis - 1)]
    )
    mask_expand_shape_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_mask_expand_shape",
        np.asarray(mask_expand_shape, dtype=np.int32),
    )
    expanded_mask_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_mask_expanded",
        dtype="BOOL",
        shape=[int(v) for v in mask_expand_shape],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[mask_2d_name, mask_expand_shape_name],
            outputs=[expanded_mask_name],
            options={"newShape": [int(v) for v in mask_expand_shape]},
        )
    )
    mask_tile_multiples = (
        [int(v) for v in input_shape[:axis]]
        + [1, 1]
        + [int(v) for v in input_shape[axis + 1 :]]
    )
    mask_tile_multiples_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_mask_tile_multiples",
        np.asarray(mask_tile_multiples, dtype=np.int32),
    )
    tiled_mask_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_mask_tiled",
        dtype="BOOL",
        shape=[int(v) for v in full_shape],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="TILE",
            inputs=[expanded_mask_name, mask_tile_multiples_name],
            outputs=[tiled_mask_name],
        )
    )

    one_scalar_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_one_scalar",
        np.asarray(1.0, dtype=np.float16 if input_dtype == "FLOAT16" else np.float32),
    )
    _set_scalar_tensor_metadata(ctx, one_scalar_name)
    full_shape_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_full_shape",
        np.asarray(full_shape, dtype=np.int32),
    )
    ones_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_ones",
        dtype=input_dtype,
        shape=[int(v) for v in full_shape],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="FILL",
            inputs=[full_shape_name, one_scalar_name],
            outputs=[ones_name],
        )
    )

    masked_values_name = ctx.add_intermediate_tensor(
        f"{output_name}_cumprod_masked_values",
        dtype=input_dtype,
        shape=[int(v) for v in full_shape],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[tiled_mask_name, tiled_input_name, ones_name],
            outputs=[masked_values_name],
        )
    )

    reduce_axes_name = ctx.add_const_tensor(
        f"{output_name}_cumprod_reduce_axes",
        np.asarray([int(axis)], dtype=np.int32),
    )
    cumprod_core_output_name = output_name
    if reverse:
        cumprod_core_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_cumprod_reversed_output",
            dtype=input_dtype,
            shape=[int(v) for v in input_shape],
        )
    ctx.add_operator(
        OperatorIR(
            op_type="REDUCE_PROD",
            inputs=[masked_values_name, reduce_axes_name],
            outputs=[cumprod_core_output_name],
            options={"keepDims": False},
        )
    )

    if reverse:
        ctx.add_operator(
            OperatorIR(
                op_type="REVERSE_V2",
                inputs=[cumprod_core_output_name, reverse_axis_name],
                outputs=[output_name],
            )
        )


def build_reduce_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    runtime_input_name = _harmonize_reduce_input_output_dtype(
        node=node,
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
    )

    input_shape = ctx.get_tensor_shape(runtime_input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    input_tensor = ctx.model_ir.tensors.get(runtime_input_name, None)
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor is not None and input_tensor.shape_signature is not None
        else None
    )
    input_rank = _resolve_reduce_input_rank(
        node=node,
        ctx=ctx,
        input_name=runtime_input_name,
        output_name=output_name,
        input_shape=[int(v) for v in list(input_shape)],
        output_shape=[int(v) for v in list(output_shape)],
    )
    rank_unreliable = bool(
        int(len(input_shape)) <= 1
        and _is_unresolved_placeholder_shape(
            [int(v) for v in list(input_shape)],
            input_signature,
        )
    )
    axes = _resolve_reduce_axes(
        node,
        ctx,
        input_rank,
        preserve_raw_axes=rank_unreliable,
    )
    if len(axes) == 0 and int(node.attrs.get("noop_with_empty_axes", 0)) == 1:
        shape_const = ctx.add_const_tensor(
            f"{output_name}_reduce_noop_shape",
            np.asarray(output_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[runtime_input_name, shape_const],
                outputs=[output_name],
                options={"newShape": [int(v) for v in output_shape]},
            )
        )
        return

    axes_const = ctx.add_const_tensor(
        f"{output_name}_{op_type.lower()}_axes",
        np.asarray(axes, dtype=np.int32),
    )
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[runtime_input_name, axes_const],
            outputs=[output_name],
            options={"keepDims": keepdims},
        )
    )


def build_global_average_pool_op(node: Any, ctx: Any) -> None:
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
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_shape)]
    )
    if len(input_shape) < 3:
        if len(input_signature) >= 3:
            _materialize_tensor_shape_from_signature(input_tensor, signature=input_signature)
            input_shape = [int(v) for v in list(input_tensor.shape)]
        elif _is_unresolved_placeholder_shape(input_shape, input_signature):
            _materialize_tensor_shape_from_signature(
                input_tensor,
                signature=[-1, -1, -1, -1],
            )
            input_shape = [int(v) for v in list(input_tensor.shape)]
    if len(input_shape) < 3:
        raise NotImplementedError(
            f"GlobalAveragePool requires rank>=3. op={node.name} input_shape={input_shape}"
        )
    if len(output_shape) != len(input_shape):
        if len(output_signature) == len(input_shape):
            _materialize_tensor_shape_from_signature(output_tensor, signature=output_signature)
        else:
            inferred_output_signature = [int(v) for v in list(input_signature)]
            for axis in range(2, len(inferred_output_signature)):
                inferred_output_signature[axis] = 1
            _materialize_tensor_shape_from_signature(output_tensor, signature=inferred_output_signature)

    spatial_axes = [int(v) for v in range(2, len(input_shape))]
    axes_const = ctx.add_const_tensor(
        f"{output_name}_global_avg_axes",
        np.asarray(spatial_axes, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MEAN",
            inputs=[input_name, axes_const],
            outputs=[output_name],
            options={"keepDims": True},
        )
    )


def build_global_max_pool_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) < 3:
        raise NotImplementedError(
            f"GlobalMaxPool requires rank>=3. op={node.name} input_shape={input_shape}"
        )

    spatial_axes = [int(v) for v in range(2, len(input_shape))]
    axes_const = ctx.add_const_tensor(
        f"{output_name}_global_max_axes",
        np.asarray(spatial_axes, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="REDUCE_MAX",
            inputs=[input_name, axes_const],
            outputs=[output_name],
            options={"keepDims": True},
        )
    )


def _build_sum_reduce_from_input(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
) -> None:
    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    axes = _resolve_reduce_axes(node, ctx, len(input_shape))
    if len(axes) == 0 and int(node.attrs.get("noop_with_empty_axes", 0)) == 1:
        shape_const = ctx.add_const_tensor(
            f"{output_name}_reduce_noop_shape",
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
    axes_const = ctx.add_const_tensor(
        f"{output_name}_sum_axes",
        np.asarray(axes, dtype=np.int32),
    )
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    ctx.add_operator(
        OperatorIR(
            op_type="SUM",
            inputs=[input_name, axes_const],
            outputs=[output_name],
            options={"keepDims": keepdims},
        )
    )


def build_reduce_l1_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    abs_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_l1_abs",
        dtype=str(ctx.get_tensor_dtype(input_name)).upper(),
        shape=[int(v) for v in ctx.get_tensor_shape(input_name)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ABS",
            inputs=[input_name],
            outputs=[abs_name],
        )
    )
    _build_sum_reduce_from_input(
        node=node,
        ctx=ctx,
        input_name=abs_name,
        output_name=output_name,
    )


def build_reduce_l2_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"

    square_input_name = input_name
    if input_dtype != compute_dtype:
        cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_l2_input_cast",
            dtype=compute_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[cast_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )
        square_input_name = cast_name

    squared_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_l2_squared",
        dtype=compute_dtype,
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[square_input_name, square_input_name],
            outputs=[squared_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_l2_sum",
        dtype=compute_dtype,
        shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
    )
    _build_sum_reduce_from_input(
        node=node,
        ctx=ctx,
        input_name=squared_name,
        output_name=sum_name,
    )

    sqrt_name = output_name
    if output_dtype != compute_dtype:
        sqrt_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_l2_sqrt",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
        )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[sum_name],
            outputs=[sqrt_name],
        )
    )

    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[sqrt_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )
