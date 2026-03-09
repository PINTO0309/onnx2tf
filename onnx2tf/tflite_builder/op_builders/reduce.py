from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, List

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


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


def _clone_quantization(quantization: Any) -> Any:
    if quantization is None:
        return None
    return copy.deepcopy(quantization)


def _tensor_shape_with_signature(ctx: Any, tensor_name: str) -> list[int]:
    shape = [int(v) for v in ctx.get_tensor_shape(tensor_name)]
    tensor = ctx.model_ir.tensors.get(tensor_name, None)
    signature = (
        [int(v) for v in list(tensor.shape_signature)]
        if tensor is not None and tensor.shape_signature is not None
        else [int(v) for v in shape]
    )
    if len(signature) != len(shape):
        return [int(v) for v in shape]
    return [
        int(signature[idx]) if int(signature[idx]) < 0 else int(shape[idx])
        for idx in range(len(shape))
    ]


def _normalize_axis_for_rank(axis: int, rank: int) -> int:
    axis_norm = int(axis)
    if axis_norm < 0:
        axis_norm += int(rank)
    if axis_norm < 0 or axis_norm >= int(rank):
        raise NotImplementedError(f"axis is out of range. axis={axis} rank={rank}")
    return int(axis_norm)


def _propagate_shape(ctx: Any, src_tensor_name: str, dst_tensor_name: str) -> None:
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src = ctx.model_ir.tensors[src_tensor_name]
    dst = ctx.model_ir.tensors[dst_tensor_name]
    src_signature = (
        list(src.shape_signature)
        if src.shape_signature is not None
        else list(src.shape)
    )
    dst.shape = [int(v) for v in list(src.shape)]
    dst.shape_signature = [int(v) for v in list(src_signature)]
    dst.dtype = str(src.dtype)
    dst.quantization = _clone_quantization(src.quantization)


def _add_binary_op(
    *,
    ctx: Any,
    op_type: str,
    lhs_name: str,
    rhs_name: str,
    output_name: str,
) -> None:
    options: dict[str, Any] = {}
    if op_type in {"ADD", "SUB", "MUL", "DIV"}:
        options = {"fusedActivationFunction": "NONE"}
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[lhs_name, rhs_name],
            outputs=[output_name],
            options=options,
        )
    )


def _build_reduce_sum_all_axes(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
) -> None:
    input_shape = _tensor_shape_with_signature(ctx, input_name)
    if len(input_shape) == 0:
        if str(input_name) != str(output_name):
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[input_name, ctx.add_const_tensor(f"{output_name}_identity_shape", np.asarray([], dtype=np.int32))],
                    outputs=[output_name],
                    options={"newShape": []},
                )
            )
        return
    axes_name = ctx.add_const_tensor(
        f"{output_name}_reduce_axes",
        np.asarray(list(range(len(input_shape))), dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUM",
            inputs=[input_name, axes_name],
            outputs=[output_name],
            options={"keepDims": False},
        )
    )


def _build_safe_divide_no_nan(
    *,
    ctx: Any,
    numerator_name: str,
    denominator_name: str,
    output_name: str,
    dtype: str,
) -> None:
    zero_name = ctx.add_const_tensor(
        f"{output_name}_safe_div_zero",
        np.asarray(0.0, dtype=np.float16 if str(dtype).upper() == "FLOAT16" else np.float32),
    )
    one_name = ctx.add_const_tensor(
        f"{output_name}_safe_div_one",
        np.asarray(1.0, dtype=np.float16 if str(dtype).upper() == "FLOAT16" else np.float32),
    )
    positive_name = ctx.add_intermediate_tensor(
        f"{output_name}_safe_div_positive",
        dtype="BOOL",
        shape=[],
    )
    safe_denom_name = ctx.add_intermediate_tensor(
        f"{output_name}_safe_div_denominator",
        dtype=dtype,
        shape=[],
    )
    div_name = ctx.add_intermediate_tensor(
        f"{output_name}_safe_div_value",
        dtype=dtype,
        shape=[],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GREATER",
            inputs=[denominator_name, zero_name],
            outputs=[positive_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[positive_name, denominator_name, one_name],
            outputs=[safe_denom_name],
        )
    )
    _add_binary_op(
        ctx=ctx,
        op_type="DIV",
        lhs_name=numerator_name,
        rhs_name=safe_denom_name,
        output_name=div_name,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[positive_name, div_name, zero_name],
            outputs=[output_name],
        )
    )


def _move_axis_to_last(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    axis: int,
) -> str:
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    rank = int(len(input_shape))
    axis_norm = _normalize_axis_for_rank(axis=axis, rank=rank)
    if int(axis_norm) == int(rank - 1):
        return str(input_name)
    perm_to_last = [int(v) for v in range(rank) if int(v) != int(axis_norm)] + [int(axis_norm)]
    permuted_shape = [int(input_shape[int(v)]) for v in perm_to_last]
    transposed_name = ctx.add_intermediate_tensor(
        output_name,
        dtype=str(ctx.get_tensor_dtype(input_name)).upper(),
        shape=permuted_shape,
    )
    return make_transpose(
        ctx=ctx,
        input_name=input_name,
        output_name=transposed_name,
        perm_values=perm_to_last,
        allow_elide_inverse_chain=False,
    )


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


def build_global_lp_pool_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) < 3:
        raise NotImplementedError(
            f"GlobalLpPool requires rank>=3. op={node.name} input_shape={input_shape}"
        )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    np_dtype = np.float16 if input_dtype == "FLOAT16" else np.float32
    p = float(node.attrs.get("p", 2.0))

    abs_name = ctx.add_intermediate_tensor(
        f"{node.name}_global_lp_abs",
        dtype=input_dtype,
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ABS",
            inputs=[input_name],
            outputs=[abs_name],
        )
    )

    powered_name = abs_name
    if abs(float(p) - 1.0) > 1e-12:
        p_const_name = ctx.add_const_tensor(
            f"{node.name}_global_lp_p",
            np.asarray(float(p), dtype=np_dtype),
        )
        powered_name = ctx.add_intermediate_tensor(
            f"{node.name}_global_lp_powered",
            dtype=input_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="POW",
                inputs=[abs_name, p_const_name],
                outputs=[powered_name],
            )
        )

    reduced_shape = [int(input_shape[0]), int(input_shape[1])] + [1] * int(len(input_shape) - 2)
    spatial_axes = [int(v) for v in range(2, len(input_shape))]
    axes_const_name = ctx.add_const_tensor(
        f"{node.name}_global_lp_axes",
        np.asarray(spatial_axes, dtype=np.int32),
    )
    reduced_name = ctx.add_intermediate_tensor(
        f"{node.name}_global_lp_reduced",
        dtype=input_dtype,
        shape=reduced_shape,
    )
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    reduced_tensor = ctx.model_ir.tensors.get(reduced_name, None)
    if output_tensor is not None and reduced_tensor is not None:
        reduced_tensor.shape_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(reduced_shape)]
        )
    ctx.add_operator(
        OperatorIR(
            op_type="SUM",
            inputs=[powered_name, axes_const_name],
            outputs=[reduced_name],
            options={"keepDims": True},
        )
    )

    root_name = reduced_name
    if abs(float(p) - 1.0) > 1e-12:
        inv_p_const_name = ctx.add_const_tensor(
            f"{node.name}_global_lp_inv_p",
            np.asarray(float(1.0 / p), dtype=np_dtype),
        )
        root_name = ctx.add_intermediate_tensor(
            f"{node.name}_global_lp_root",
            dtype=input_dtype,
            shape=reduced_shape,
        )
        root_tensor = ctx.model_ir.tensors.get(root_name, None)
        if reduced_tensor is not None and root_tensor is not None:
            root_tensor.shape_signature = (
                [int(v) for v in list(reduced_tensor.shape_signature)]
                if reduced_tensor.shape_signature is not None
                else [int(v) for v in list(reduced_shape)]
            )
        ctx.add_operator(
            OperatorIR(
                op_type="POW",
                inputs=[reduced_name, inv_p_const_name],
                outputs=[root_name],
            )
        )

    if output_dtype == input_dtype:
        shape_const_name = ctx.add_const_tensor(
            f"{output_name}_global_lp_shape",
            np.asarray([int(v) for v in ctx.get_tensor_shape(output_name)], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[root_name, shape_const_name],
                outputs=[output_name],
                options={"newShape": [int(v) for v in ctx.get_tensor_shape(output_name)]},
            )
        )
        return

    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[root_name],
            outputs=[output_name],
            options={"inDataType": input_dtype, "outDataType": output_dtype},
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


def build_reduce_sum_square_op(node: Any, ctx: Any) -> None:
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
        square_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_sum_square_input_cast",
            dtype=compute_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[square_input_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    squared_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_sum_square_squared",
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

    sum_name = output_name
    if output_dtype != compute_dtype:
        sum_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_sum_square_sum",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
        )
    _build_sum_reduce_from_input(
        node=node,
        ctx=ctx,
        input_name=squared_name,
        output_name=sum_name,
    )

    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[sum_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_reduce_log_sum_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"

    sum_input_name = input_name
    if input_dtype != compute_dtype:
        sum_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_log_sum_input_cast",
            dtype=compute_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[sum_input_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_log_sum_sum",
        dtype=compute_dtype,
        shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
    )
    log_name = output_name
    if output_dtype != compute_dtype:
        log_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_log_sum_log",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
        )
    _build_sum_reduce_from_input(
        node=node,
        ctx=ctx,
        input_name=sum_input_name,
        output_name=sum_name,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOG",
            inputs=[sum_name],
            outputs=[log_name],
        )
    )
    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[log_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_reduce_log_sum_exp_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"

    exp_input_name = input_name
    if input_dtype != compute_dtype:
        exp_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_log_sum_exp_input_cast",
            dtype=compute_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[exp_input_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    exp_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_log_sum_exp_exp",
        dtype=compute_dtype,
        shape=input_shape,
    )
    sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_log_sum_exp_sum",
        dtype=compute_dtype,
        shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
    )
    log_name = output_name
    if output_dtype != compute_dtype:
        log_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_log_sum_exp_log",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
        )
    ctx.add_operator(
        OperatorIR(
            op_type="EXP",
            inputs=[exp_input_name],
            outputs=[exp_name],
        )
    )
    _build_sum_reduce_from_input(
        node=node,
        ctx=ctx,
        input_name=exp_name,
        output_name=sum_name,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOG",
            inputs=[sum_name],
            outputs=[log_name],
        )
    )
    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[log_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_negative_log_likelihood_loss_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    target_name = node.inputs[1].name
    weight_name = node.inputs[2].name if len(node.inputs) > 2 and str(node.inputs[2].name) != "" else ""
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(target_name)
    if str(weight_name) != "":
        ctx.ensure_tensor(weight_name)
    ctx.ensure_tensor(output_name)

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    input_shape = _tensor_shape_with_signature(ctx, input_name)
    rank = int(len(input_shape))

    working_input_name = _move_axis_to_last(
        ctx=ctx,
        input_name=input_name,
        output_name=f"{output_name}_nll_input_axis_last",
        axis=1,
    )
    working_input_shape = _tensor_shape_with_signature(ctx, working_input_name)
    class_depth = int(working_input_shape[-1])

    target_i32_name = target_name
    target_dtype = str(ctx.get_tensor_dtype(target_name)).upper()
    target_shape = _tensor_shape_with_signature(ctx, target_name)
    if target_dtype != "INT32":
        target_i32_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_target_i32",
            dtype="INT32",
            shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
        )
        target_i32_tensor = ctx.model_ir.tensors.get(target_i32_name, None)
        if target_i32_tensor is not None:
            target_i32_tensor.shape_signature = [int(v) for v in target_shape]
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[target_name],
                outputs=[target_i32_name],
                options={"inDataType": target_dtype, "outDataType": "INT32"},
            )
        )

    ignore_mask_name = ""
    labels_safe_name = target_i32_name
    ignore_index = node.attrs.get("ignore_index", None)
    if ignore_index is not None:
        ignore_const_name = ctx.add_const_tensor(
            f"{output_name}_nll_ignore_index",
            np.asarray(int(ignore_index), dtype=np.int32),
        )
        ignore_mask_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_ignore_mask",
            dtype="BOOL",
            shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
        )
        labels_zero_name = ctx.add_const_tensor(
            f"{output_name}_nll_zero_i32",
            np.asarray(0, dtype=np.int32),
        )
        labels_safe_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_labels_safe",
            dtype="INT32",
            shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
        )
        labels_safe_tensor = ctx.model_ir.tensors.get(labels_safe_name, None)
        if labels_safe_tensor is not None:
            labels_safe_tensor.shape_signature = [int(v) for v in target_shape]
        ctx.add_operator(
            OperatorIR(
                op_type="EQUAL",
                inputs=[target_i32_name, ignore_const_name],
                outputs=[ignore_mask_name],
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SELECT_V2",
                inputs=[ignore_mask_name, labels_zero_name, target_i32_name],
                outputs=[labels_safe_name],
            )
        )

    depth_name = ctx.add_const_tensor(
        f"{output_name}_nll_depth",
        np.asarray(int(class_depth), dtype=np.int32),
    )
    values_name = ctx.add_const_tensor(
        f"{output_name}_nll_one_hot_values",
        np.asarray([0.0, 1.0], dtype=np.float16 if input_dtype == "FLOAT16" else np.float32),
    )
    one_hot_name = ctx.add_intermediate_tensor(
        f"{output_name}_nll_one_hot",
        dtype=input_dtype,
        shape=[int(v) if int(v) > 0 else 1 for v in target_shape] + [int(class_depth)],
    )
    one_hot_tensor = ctx.model_ir.tensors.get(one_hot_name, None)
    if one_hot_tensor is not None:
        one_hot_tensor.shape_signature = [int(v) for v in target_shape] + [int(class_depth)]
    ctx.add_operator(
        OperatorIR(
            op_type="ONE_HOT",
            inputs=[labels_safe_name, depth_name, ctx.add_const_tensor(f"{output_name}_nll_on_value", np.asarray(1.0, dtype=np.float16 if input_dtype == "FLOAT16" else np.float32)), ctx.add_const_tensor(f"{output_name}_nll_off_value", np.asarray(0.0, dtype=np.float16 if input_dtype == "FLOAT16" else np.float32))],
            outputs=[one_hot_name],
            options={"axis": -1},
        )
    )

    selected_mul_name = ctx.add_intermediate_tensor(
        f"{output_name}_nll_selected_mul",
        dtype=input_dtype,
        shape=[int(v) if int(v) > 0 else 1 for v in working_input_shape],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=working_input_name,
        rhs_name=one_hot_name,
        output_name=selected_mul_name,
    )
    selected_name = ctx.add_intermediate_tensor(
        f"{output_name}_nll_selected",
        dtype=input_dtype,
        shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
    )
    selected_tensor = ctx.model_ir.tensors.get(selected_name, None)
    if selected_tensor is not None:
        selected_tensor.shape_signature = [int(v) for v in target_shape]
    selected_axes_name = ctx.add_const_tensor(
        f"{output_name}_nll_selected_axes",
        np.asarray([-1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUM",
            inputs=[selected_mul_name, selected_axes_name],
            outputs=[selected_name],
            options={"keepDims": False},
        )
    )

    zero_float_name = ctx.add_const_tensor(
        f"{output_name}_nll_zero",
        np.asarray(0.0, dtype=np.float16 if input_dtype == "FLOAT16" else np.float32),
    )
    loss_name = ctx.add_intermediate_tensor(
        f"{output_name}_nll_loss",
        dtype=input_dtype,
        shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
    )
    loss_tensor = ctx.model_ir.tensors.get(loss_name, None)
    if loss_tensor is not None:
        loss_tensor.shape_signature = [int(v) for v in target_shape]
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=zero_float_name,
        rhs_name=selected_name,
        output_name=loss_name,
    )

    denominator_name = ""
    if str(weight_name) != "":
        gathered_weight_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_weight_gather",
            dtype=str(ctx.get_tensor_dtype(weight_name)).upper(),
            shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
        )
        gathered_weight_tensor = ctx.model_ir.tensors.get(gathered_weight_name, None)
        if gathered_weight_tensor is not None:
            gathered_weight_tensor.shape_signature = [int(v) for v in target_shape]
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[weight_name, labels_safe_name],
                outputs=[gathered_weight_name],
                options={"axis": 0, "batchDims": 0},
            )
        )
        weight_runtime_name = gathered_weight_name
        gathered_dtype = str(ctx.get_tensor_dtype(weight_runtime_name)).upper()
        if gathered_dtype != input_dtype:
            weight_runtime_name = ctx.add_intermediate_tensor(
                f"{output_name}_nll_weight_cast",
                dtype=input_dtype,
                shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
            )
            weight_runtime_tensor = ctx.model_ir.tensors.get(weight_runtime_name, None)
            if weight_runtime_tensor is not None:
                weight_runtime_tensor.shape_signature = [int(v) for v in target_shape]
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[gathered_weight_name],
                    outputs=[weight_runtime_name],
                    options={"inDataType": gathered_dtype, "outDataType": input_dtype},
                )
            )
        if str(ignore_mask_name) != "":
            weight_masked_name = ctx.add_intermediate_tensor(
                f"{output_name}_nll_weight_masked",
                dtype=input_dtype,
                shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
            )
            weight_masked_tensor = ctx.model_ir.tensors.get(weight_masked_name, None)
            if weight_masked_tensor is not None:
                weight_masked_tensor.shape_signature = [int(v) for v in target_shape]
            ctx.add_operator(
                OperatorIR(
                    op_type="SELECT_V2",
                    inputs=[ignore_mask_name, zero_float_name, weight_runtime_name],
                    outputs=[weight_masked_name],
                )
            )
            weight_runtime_name = weight_masked_name
        weighted_loss_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_weighted_loss",
            dtype=input_dtype,
            shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
        )
        weighted_loss_tensor = ctx.model_ir.tensors.get(weighted_loss_name, None)
        if weighted_loss_tensor is not None:
            weighted_loss_tensor.shape_signature = [int(v) for v in target_shape]
        _add_binary_op(
            ctx=ctx,
            op_type="MUL",
            lhs_name=loss_name,
            rhs_name=weight_runtime_name,
            output_name=weighted_loss_name,
        )
        loss_name = weighted_loss_name
        denominator_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_weight_denominator",
            dtype=input_dtype,
            shape=[],
        )
        _build_reduce_sum_all_axes(
            ctx=ctx,
            input_name=weight_runtime_name,
            output_name=denominator_name,
        )
    elif str(ignore_mask_name) != "":
        valid_mask_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_valid_mask",
            dtype=input_dtype,
            shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
        )
        valid_mask_tensor = ctx.model_ir.tensors.get(valid_mask_name, None)
        if valid_mask_tensor is not None:
            valid_mask_tensor.shape_signature = [int(v) for v in target_shape]
        one_float_name = ctx.add_const_tensor(
            f"{output_name}_nll_one",
            np.asarray(1.0, dtype=np.float16 if input_dtype == "FLOAT16" else np.float32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SELECT_V2",
                inputs=[ignore_mask_name, zero_float_name, one_float_name],
                outputs=[valid_mask_name],
            )
        )
        masked_loss_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_masked_loss",
            dtype=input_dtype,
            shape=[int(v) if int(v) > 0 else 1 for v in target_shape],
        )
        masked_loss_tensor = ctx.model_ir.tensors.get(masked_loss_name, None)
        if masked_loss_tensor is not None:
            masked_loss_tensor.shape_signature = [int(v) for v in target_shape]
        _add_binary_op(
            ctx=ctx,
            op_type="MUL",
            lhs_name=loss_name,
            rhs_name=valid_mask_name,
            output_name=masked_loss_name,
        )
        loss_name = masked_loss_name
        denominator_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_valid_denominator",
            dtype=input_dtype,
            shape=[],
        )
        _build_reduce_sum_all_axes(
            ctx=ctx,
            input_name=valid_mask_name,
            output_name=denominator_name,
        )

    reduction = str(node.attrs.get("reduction", "mean")).lower()
    if reduction == "none":
        output_tensor = ctx.model_ir.tensors[output_name]
        target_tensor = ctx.model_ir.tensors.get(target_name, None)
        output_tensor.dtype = output_dtype
        output_tensor.shape = [int(v) if int(v) > 0 else 1 for v in target_shape]
        output_tensor.shape_signature = [int(v) for v in target_shape]
        if target_tensor is not None:
            output_tensor.quantization = _clone_quantization(target_tensor.quantization)
        if output_dtype == input_dtype:
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[loss_name, ctx.add_const_tensor(f"{output_name}_nll_output_shape", np.asarray([int(v) for v in target_shape], dtype=np.int32))],
                    outputs=[output_name],
                    options={"newShape": [int(v) for v in target_shape], "preserveDynamicShape": True},
                )
            )
        else:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[loss_name],
                    outputs=[output_name],
                    options={"inDataType": input_dtype, "outDataType": output_dtype},
                )
            )
        return

    sum_loss_name = ctx.add_intermediate_tensor(
        f"{output_name}_nll_sum_loss",
        dtype=input_dtype,
        shape=[],
    )
    _build_reduce_sum_all_axes(
        ctx=ctx,
        input_name=loss_name,
        output_name=sum_loss_name,
    )
    output_core_name = output_name
    if output_dtype != input_dtype:
        output_core_name = ctx.add_intermediate_tensor(
            f"{output_name}_nll_output_core",
            dtype=input_dtype,
            shape=[],
        )

    output_tensor = ctx.model_ir.tensors[output_name]
    output_tensor.shape = []
    output_tensor.shape_signature = []
    output_tensor.dtype = output_dtype
    if reduction == "sum":
        if str(output_core_name) != str(output_name):
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[sum_loss_name],
                    outputs=[output_name],
                    options={"inDataType": input_dtype, "outDataType": output_dtype},
                )
            )
        else:
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[sum_loss_name, ctx.add_const_tensor(f"{output_name}_nll_scalar_shape", np.asarray([], dtype=np.int32))],
                    outputs=[output_name],
                    options={"newShape": []},
                )
            )
        return

    if str(denominator_name) == "":
        ctx.add_operator(
            OperatorIR(
                op_type="MEAN",
                inputs=[loss_name, ctx.add_const_tensor(f"{output_name}_nll_mean_axes", np.asarray(list(range(max(int(rank - 1), 1))), dtype=np.int32))],
                outputs=[output_core_name],
                options={"keepDims": False},
            )
        )
    else:
        _build_safe_divide_no_nan(
            ctx=ctx,
            numerator_name=sum_loss_name,
            denominator_name=denominator_name,
            output_name=output_core_name,
            dtype=input_dtype,
        )
    if str(output_core_name) != str(output_name):
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[output_core_name],
                outputs=[output_name],
                options={"inDataType": input_dtype, "outDataType": output_dtype},
            )
        )


def build_softmax_cross_entropy_loss_op(node: Any, ctx: Any) -> None:
    scores_name = node.inputs[0].name
    loss_output_name = node.outputs[0].name
    log_prob_output_name = node.outputs[1].name if len(node.outputs) > 1 and str(node.outputs[1].name) != "" else ""
    ctx.ensure_tensor(scores_name)
    ctx.ensure_tensor(loss_output_name)
    if str(log_prob_output_name) != "":
        ctx.ensure_tensor(log_prob_output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(scores_name)]
    rank = int(len(input_shape))
    class_axis = 1
    axis_norm = _normalize_axis_for_rank(class_axis, rank)
    log_prob_name = log_prob_output_name if str(log_prob_output_name) != "" else ctx.add_intermediate_tensor(
        f"{loss_output_name}_softmaxce_log_prob",
        dtype=str(ctx.get_tensor_dtype(scores_name)).upper(),
        shape=[int(v) for v in input_shape],
    )
    if str(log_prob_output_name) != "":
        _propagate_shape(ctx, scores_name, log_prob_output_name)

    if int(axis_norm) == int(rank - 1):
        softmax_name = ctx.add_intermediate_tensor(
            f"{loss_output_name}_softmaxce_softmax",
            dtype=str(ctx.get_tensor_dtype(scores_name)).upper(),
            shape=[int(v) for v in input_shape],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SOFTMAX",
                inputs=[scores_name],
                outputs=[softmax_name],
                options={"beta": 1.0},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="LOG",
                inputs=[softmax_name],
                outputs=[log_prob_name],
            )
        )
    else:
        perm_to_last = [int(v) for v in range(rank) if int(v) != int(axis_norm)] + [int(axis_norm)]
        perm_from_last = [0] * int(rank)
        for out_axis, in_axis in enumerate(perm_to_last):
            perm_from_last[int(in_axis)] = int(out_axis)
        axis_last_shape = [int(input_shape[int(v)]) for v in perm_to_last]
        input_axis_last_name = ctx.add_intermediate_tensor(
            f"{loss_output_name}_softmaxce_input_axis_last",
            dtype=str(ctx.get_tensor_dtype(scores_name)).upper(),
            shape=axis_last_shape,
        )
        input_axis_last_name = make_transpose(
            ctx=ctx,
            input_name=scores_name,
            output_name=input_axis_last_name,
            perm_values=perm_to_last,
            allow_elide_inverse_chain=False,
        )
        softmax_axis_last_name = ctx.add_intermediate_tensor(
            f"{loss_output_name}_softmaxce_softmax_axis_last",
            dtype=str(ctx.get_tensor_dtype(scores_name)).upper(),
            shape=axis_last_shape,
        )
        log_axis_last_name = ctx.add_intermediate_tensor(
            f"{loss_output_name}_softmaxce_log_axis_last",
            dtype=str(ctx.get_tensor_dtype(scores_name)).upper(),
            shape=axis_last_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SOFTMAX",
                inputs=[input_axis_last_name],
                outputs=[softmax_axis_last_name],
                options={"beta": 1.0},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="LOG",
                inputs=[softmax_axis_last_name],
                outputs=[log_axis_last_name],
            )
        )
        transposed_log_prob_name = make_transpose(
            ctx=ctx,
            input_name=log_axis_last_name,
            output_name=log_prob_name,
            perm_values=perm_from_last,
            allow_elide_inverse_chain=False,
        )
        if str(transposed_log_prob_name) != str(log_prob_name):
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[
                        transposed_log_prob_name,
                        ctx.add_const_tensor(
                            f"{loss_output_name}_softmaxce_log_prob_shape",
                            np.asarray([int(v) for v in input_shape], dtype=np.int32),
                        ),
                    ],
                    outputs=[log_prob_name],
                    options={"newShape": [int(v) for v in input_shape]},
                )
            )

    proxy_node = SimpleNamespace(
        name=f"{node.name}_softmaxce_nll_proxy",
        op="NegativeLogLikelihoodLoss",
        attrs={
            "reduction": node.attrs.get("reduction", "mean"),
            "ignore_index": node.attrs.get("ignore_index", None),
        },
        inputs=[
            SimpleNamespace(name=log_prob_name),
            SimpleNamespace(name=node.inputs[1].name),
        ] + ([SimpleNamespace(name=node.inputs[2].name)] if len(node.inputs) > 2 and str(node.inputs[2].name) != "" else []),
        outputs=[SimpleNamespace(name=loss_output_name)],
    )
    build_negative_log_likelihood_loss_op(proxy_node, ctx)
