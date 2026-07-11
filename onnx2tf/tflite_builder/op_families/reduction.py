from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.op_contracts import (
    NodeValidationError,
    is_unknown_rank_placeholder_tensor as _is_unknown_rank_placeholder_tensor,
    require_const_input as _require_const_input,
    tensor_shape_with_signature as _tensor_shape_with_signature,
)


def _extract_axes(
    *,
    node: Any,
    ctx: Any,
    input_index: int = 1,
    attr_name: str = "axes",
    default_if_missing: Optional[List[int]] = None,
) -> List[int]:
    axes: Optional[List[int]] = None
    if len(node.inputs) > input_index and str(node.inputs[input_index].name) != "":
        axes_arr = _require_const_input(node, ctx, input_index, f"{node.op} axes")
        axes = [int(v) for v in np.asarray(axes_arr).reshape(-1).tolist()]
    elif attr_name in node.attrs:
        attr_axes = node.attrs.get(attr_name)
        if isinstance(attr_axes, (list, tuple)):
            axes = [int(v) for v in attr_axes]
        elif attr_axes is None:
            axes = []
        else:
            axes = [int(attr_axes)]
    if axes is None:
        axes = [] if default_if_missing is None else [int(v) for v in default_if_missing]
    return [int(v) for v in axes]


def _normalize_axes_for_rank(
    *,
    axes: List[int],
    rank: int,
    node: Any,
) -> List[int]:
    normalized: List[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += rank
        if a < 0 or a >= rank:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"axis out of range. axis={axis} normalized={a} rank={rank}",
                node_name=node.name,
                node_op=node.op,
            )
        if a not in normalized:
            normalized.append(a)
    return normalized


def _validate_reduce(node: Any, ctx: Any) -> None:
    # Reduce axes must be compile-time constant in flatbuffer_direct.
    # Keep rank/axis normalization in builder-side logic, but fail early in
    # coverage/dispatch checks when non-constant axes input is provided.
    if len(node.inputs) >= 2 and str(node.inputs[1].name) != "":
        axes_arr = ctx.get_constant_array(node.inputs[1].name)
        if axes_arr is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message=(
                    "Reduce axes input must be constant in flatbuffer_direct. "
                    f"op={node.name}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_cumsum(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"CumSum input rank must be >= 1. input_rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.inputs) >= 2 and str(node.inputs[1].name) != "":
        axis_arr = _require_const_input(node, ctx, 1, "CumSum axis")
        axis_values = np.asarray(axis_arr).reshape(-1)
        if int(axis_values.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "CumSum axis must be scalar or single-element tensor. "
                    f"axis_shape={list(np.asarray(axis_arr).shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        axis_raw = int(axis_values[0])
    else:
        axis_raw = int(node.attrs.get("axis", 0))

    axis = int(axis_raw)
    if axis < 0:
        axis += int(input_rank)
    if axis < 0 or axis >= int(input_rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"CumSum axis out of range. axis={axis_raw} normalized={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    for attr_name in ["exclusive", "reverse"]:
        attr_value = int(node.attrs.get(attr_name, 0))
        if attr_value not in [0, 1]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"CumSum {attr_name} must be 0 or 1. got={attr_value}",
                node_name=node.name,
                node_op=node.op,
            )


def _validate_cumprod(node: Any, ctx: Any) -> None:
    _validate_cumsum(node, ctx)

    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    input_shape = _tensor_shape_with_signature(ctx, input_name)
    if any(int(dim) <= 0 for dim in input_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "CumProd builtin lowering requires static positive input shape in flatbuffer_direct. "
                f"input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "CumProd builtin lowering currently supports FLOAT16/FLOAT32 input only. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype != input_dtype:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "CumProd output dtype must match input dtype for builtin lowering. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_squeeze(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axes = _extract_axes(
        node=node,
        ctx=ctx,
        input_index=1,
        attr_name="axes",
    )
    if len(axes) == 0:
        return
    _normalize_axes_for_rank(axes=axes, rank=input_rank, node=node)


def _validate_unsqueeze(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    input_shape = ctx.get_tensor_shape(input_name)
    input_rank = len(input_shape)
    raw_shape = None
    if hasattr(ctx, "shape_map"):
        raw_shape = ctx.shape_map.get(str(input_name), None)
    input_rank_unknown = bool(
        len(input_shape) == 1
        and _is_unknown_rank_placeholder_tensor(ctx, input_name)
        and not (isinstance(raw_shape, (list, tuple)) and len(list(raw_shape)) > 0)
    )
    axes = _extract_axes(
        node=node,
        ctx=ctx,
        input_index=1,
        attr_name="axes",
    )
    if len(axes) == 0:
        raise NodeValidationError(
            reason_code="missing_required_attribute",
            message="Unsqueeze requires axes via input tensor or attribute.",
            node_name=node.name,
            node_op=node.op,
        )
    if input_rank == 0:
        output_rank = len(ctx.get_tensor_shape(node.outputs[0].name))
        if output_rank > 0:
            input_rank = int(max(output_rank - len(axes), 0))
    if input_rank_unknown and len(axes) > 0:
        output_name = node.outputs[0].name
        output_shape = ctx.get_tensor_shape(output_name)
        output_raw_shape = None
        if hasattr(ctx, "shape_map"):
            output_raw_shape = ctx.shape_map.get(str(output_name), None)
        output_rank_unknown = bool(
            len(output_shape) == 1
            and _is_unknown_rank_placeholder_tensor(ctx, output_name)
            and not (
                isinstance(output_raw_shape, (list, tuple))
                and len(list(output_raw_shape)) > 0
            )
        )
        if not output_rank_unknown:
            input_rank = int(max(input_rank, len(output_shape) - len(axes)))
        positive_axes = [int(v) for v in axes if int(v) >= 0]
        if len(positive_axes) > 0:
            min_output_rank = int(max(max(positive_axes) + 1, len(axes)))
            input_rank = int(max(input_rank, min_output_rank - len(axes)))
    output_rank = int(input_rank + len(axes))
    normalized_axes: List[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += output_rank
        if a < 0 or a >= output_rank:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    f"unsqueeze axis out of range. axis={axis} normalized={a} "
                    f"input_rank={input_rank} output_rank={output_rank}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if a in normalized_axes:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"unsqueeze axes must be unique. axes={axes}",
                node_name=node.name,
                node_op=node.op,
            )
        normalized_axes.append(int(a))
