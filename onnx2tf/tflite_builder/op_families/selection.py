from __future__ import annotations

from typing import Any, List

import numpy as np

from onnx2tf.tflite_builder.core.op_contracts import (
    NodeValidationError,
    is_integer_dtype as _is_integer_dtype,
    is_unknown_rank_placeholder_tensor as _is_unknown_rank_placeholder_tensor,
    require_const_input as _require_const_input,
)


def _validate_gather(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Gather axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(node.attrs.get("batch_dims", 0)) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Gather batch_dims must be 0. batch_dims={int(node.attrs.get('batch_dims', 0))}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_gather_nd(node: Any, ctx: Any) -> None:
    params_shape = ctx.get_tensor_shape(node.inputs[0].name)
    indices_shape = ctx.get_tensor_shape(node.inputs[1].name)
    if len(params_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GatherND params rank must be >= 1. params_shape={params_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(indices_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GatherND indices rank must be >= 1. indices_shape={indices_shape}",
            node_name=node.name,
            node_op=node.op,
        )

    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims < 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GatherND batch_dims must be >= 0. batch_dims={batch_dims}",
            node_name=node.name,
            node_op=node.op,
        )
    if batch_dims >= len(indices_shape):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GatherND batch_dims must be < indices rank. "
                f"batch_dims={batch_dims} indices_rank={len(indices_shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if batch_dims > len(params_shape):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GatherND batch_dims must be <= params rank. "
                f"batch_dims={batch_dims} params_rank={len(params_shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if batch_dims > 0:
        params_batch_shape = [int(v) for v in params_shape[:batch_dims]]
        indices_batch_shape = [int(v) for v in indices_shape[:batch_dims]]
        if params_batch_shape != indices_batch_shape:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GatherND batch_dims requires params/indices batch prefix match. "
                    f"params_batch_shape={params_batch_shape} "
                    f"indices_batch_shape={indices_batch_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if any(int(v) <= 0 for v in params_batch_shape):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GatherND batch_dims>0 requires static positive batch prefix dimensions "
                    "in flatbuffer_direct. "
                    f"params_batch_shape={params_batch_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        non_batch_indices_shape = [int(v) for v in indices_shape[batch_dims:-1]]
        if any(int(v) < 0 for v in non_batch_indices_shape):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GatherND batch_dims>0 requires static non-negative non-batch index dimensions "
                    "in flatbuffer_direct. "
                    f"indices_shape={indices_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    indices_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    supported_indices_dtypes = {
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
    }
    if indices_dtype not in supported_indices_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "GatherND indices dtype must be integer. "
                f"indices_dtype={indices_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    k_dim = int(indices_shape[-1]) if len(indices_shape) > 0 else -1
    if k_dim <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GatherND requires static positive indices last dimension in flatbuffer_direct. "
                f"indices_shape={indices_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if k_dim > int(len(params_shape) - int(batch_dims)):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GatherND indices last dimension must be <= params rank after batch_dims. "
                f"indices_last_dim={k_dim} params_rank={len(params_shape)} batch_dims={batch_dims}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_argmax(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    input_rank = len(input_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ArgMax axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    select_last_index = int(node.attrs.get("select_last_index", 0))
    if select_last_index != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ArgMax select_last_index must be 0. got={select_last_index}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_argmin(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    input_rank = len(input_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ArgMin axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    select_last_index = int(node.attrs.get("select_last_index", 0))
    if select_last_index != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ArgMin select_last_index must be 0. got={select_last_index}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_topk(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    input_rank = len(input_shape)
    if input_rank <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"TopK input rank must be >= 1. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )

    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32", "INT32", "INT64"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "TopK supports FLOAT16/FLOAT32/INT32/INT64 input in flatbuffer_direct. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    axis = int(node.attrs.get("axis", -1))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"TopK axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    largest = int(node.attrs.get("largest", 1))
    if largest not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"TopK largest must be 0 or 1. largest={largest}",
            node_name=node.name,
            node_op=node.op,
        )
    sorted_attr = int(node.attrs.get("sorted", 1))
    if sorted_attr not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"TopK sorted must be 0 or 1 in flatbuffer_direct builtin lowering. sorted={sorted_attr}",
            node_name=node.name,
            node_op=node.op,
        )

    k_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
    if len(k_shape) == 0:
        pass
    elif len(k_shape) == 1 and int(k_shape[0]) <= 1:
        pass
    else:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "TopK k input must be scalar-like (shape [] or [1]) in flatbuffer_direct. "
                f"k_shape={k_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    k_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if not _is_integer_dtype(k_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"TopK k input must be integer dtype. k_dtype={k_dtype}",
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.outputs) >= 2:
        indices_dtype = str(ctx.get_tensor_dtype(node.outputs[1].name)).upper()
        if indices_dtype not in {"INT32", "INT64"}:
            raise NodeValidationError(
                reason_code="unsupported_output_dtype",
                message=(
                    "TopK indices output dtype must be INT32 or INT64 in flatbuffer_direct. "
                    f"indices_dtype={indices_dtype}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_hardmax(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    input_rank = len(input_shape)
    if input_rank <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Hardmax input rank must be >= 1. shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    axis = int(node.attrs.get("axis", 1))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Hardmax axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(input_shape[axis]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Hardmax requires static positive dimension on target axis in flatbuffer_direct. "
                f"axis={axis} input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_nonzero(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    if len(input_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"NonZero input rank must be >= 1. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    output_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
    if len(output_shape) != 2:
        raise NodeValidationError(
            reason_code="unsupported_output_rank",
            message=f"NonZero output rank must be 2. output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_non_max_suppression(node: Any, ctx: Any) -> None:
    boxes_shape = ctx.get_tensor_shape(node.inputs[0].name)
    scores_shape = ctx.get_tensor_shape(node.inputs[1].name)
    output_nms_with_argmax = bool(getattr(ctx, "output_nms_with_argmax", False))
    if len(boxes_shape) != 3 or len(scores_shape) != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "NonMaxSuppression builtin lowering currently supports rank-3 boxes and scores only. "
                f"boxes_shape={boxes_shape} scores_shape={scores_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(node.attrs.get("center_point_box", 0)) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "NonMaxSuppression center_point_box=1 is not supported in flatbuffer_direct builtin lowering."
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(boxes_shape[0]) != 1 or int(scores_shape[0]) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "NonMaxSuppression builtin lowering currently supports only batch=1. "
                f"boxes_shape={boxes_shape} scores_shape={scores_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not output_nms_with_argmax and int(scores_shape[1]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "NonMaxSuppression requires static positive class dimension when "
                "--output_nms_with_argmax is disabled for flatbuffer_direct builtin lowering. "
                f"scores_shape={scores_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(boxes_shape[2]) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"NonMaxSuppression boxes last dimension must be 4. boxes_shape={boxes_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(boxes_shape[1]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "NonMaxSuppression requires static positive num_boxes in flatbuffer_direct builtin lowering. "
                f"boxes_shape={boxes_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(scores_shape[2]) != int(boxes_shape[1]):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "NonMaxSuppression requires scores_shape[2] == boxes_shape[1] in builtin lowering. "
                f"boxes_shape={boxes_shape} scores_shape={scores_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.inputs) >= 3:
        max_output_arr = _require_const_input(
            node,
            ctx,
            2,
            "NonMaxSuppression max_output_boxes_per_class",
        )
        max_output_flat = np.asarray(max_output_arr).reshape(-1)
        if int(max_output_flat.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "NonMaxSuppression max_output_boxes_per_class must be scalar or single-element tensor. "
                    f"shape={list(np.asarray(max_output_arr).shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 4:
        iou_threshold_arr = _require_const_input(
            node,
            ctx,
            3,
            "NonMaxSuppression iou_threshold",
        )
        iou_threshold_flat = np.asarray(iou_threshold_arr).reshape(-1)
        if int(iou_threshold_flat.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "NonMaxSuppression iou_threshold must be scalar or single-element tensor. "
                    f"shape={list(np.asarray(iou_threshold_arr).shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 5:
        score_threshold_arr = _require_const_input(
            node,
            ctx,
            4,
            "NonMaxSuppression score_threshold",
        )
        score_threshold_flat = np.asarray(score_threshold_arr).reshape(-1)
        if int(score_threshold_flat.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "NonMaxSuppression score_threshold must be scalar or single-element tensor. "
                    f"shape={list(np.asarray(score_threshold_arr).shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"INT32", "INT64"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "NonMaxSuppression output dtype must be INT32 or INT64 in flatbuffer_direct builtin lowering. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_gather_elements(node: Any, ctx: Any) -> None:
    def _rank_is_unknown_placeholder(tensor_name: str, shape: List[int]) -> bool:
        raw_shape = None
        if hasattr(ctx, "shape_map"):
            raw_shape = ctx.shape_map.get(str(tensor_name), None)
        if isinstance(raw_shape, (list, tuple)) and len(list(raw_shape)) > 0:
            # Rank is known even when dimensions are symbolic/unknown.
            return False
        return bool(
            len(shape) == 1
            and _is_unknown_rank_placeholder_tensor(ctx, tensor_name)
        )

    data_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    output_name = node.outputs[0].name
    data_shape = ctx.get_tensor_shape(node.inputs[0].name)
    indices_shape = ctx.get_tensor_shape(node.inputs[1].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    data_rank_unknown = _rank_is_unknown_placeholder(data_name, data_shape)
    indices_rank_unknown = _rank_is_unknown_placeholder(indices_name, indices_shape)
    output_rank_unknown = _rank_is_unknown_placeholder(output_name, output_shape)
    if output_shape == [1] and len(indices_shape) > 1:
        # GatherElements output rank is defined by indices. Some exporters
        # retain a rank-one placeholder even after the indices rank is known.
        output_rank_unknown = True

    if (
        len(data_shape) != len(indices_shape)
        and not data_rank_unknown
        and not indices_rank_unknown
    ):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                "GatherElements requires data and indices with same rank. "
                f"data_shape={data_shape} indices_shape={indices_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if (
        len(indices_shape) != len(output_shape)
        and not indices_rank_unknown
        and not output_rank_unknown
    ):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "GatherElements requires output rank equal to indices rank. "
                f"indices_shape={indices_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    rank_candidates: List[int] = []
    if not data_rank_unknown:
        rank_candidates.append(len(data_shape))
    if not indices_rank_unknown:
        rank_candidates.append(len(indices_shape))
    if not output_rank_unknown:
        rank_candidates.append(len(output_shape))
    if len(rank_candidates) == 0:
        return
    rank = int(rank_candidates[0])
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GatherElements axis out of range. axis={axis} rank={rank}",
            node_name=node.name,
            node_op=node.op,
        )
