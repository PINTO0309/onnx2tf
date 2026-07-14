from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, cast

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    channel_first_logical_layout,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    logical_layout_permutation,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _broadcast_shapes_relaxed,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _perm_cf_to_cl,
    _perm_cl_to_cf,
    _permute_shape,
    _tensor_name_suggests_channel_last_layout_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_shape_policy import (
    _infer_conv2d_ctor_params_for_codegen,
)


def _native_codegen_cache_bucket_for_model_ir(
    *,
    model_ir: ModelIR,
) -> Dict[str, Any]:
    metadata = model_ir.metadata
    bucket = metadata.get("_native_codegen_cache", None)
    if not isinstance(bucket, dict):
        bucket = {}
        metadata["_native_codegen_cache"] = bucket
    return cast(Dict[str, Any], bucket)


def _native_codegen_graph_index_for_model_ir(
    *,
    model_ir: ModelIR,
) -> ModelIRGraphIndex:
    bucket = _native_codegen_cache_bucket_for_model_ir(model_ir=model_ir)
    cached_index = bucket.get("graph_index", None)
    if (
        isinstance(cached_index, ModelIRGraphIndex)
        and cached_index.model_ir is model_ir
    ):
        return cached_index
    graph_index = ModelIRGraphIndex(model_ir)
    bucket["graph_index"] = graph_index
    return graph_index


def _native_codegen_expected_channel_dim_cache_for_model_ir(
    *,
    model_ir: ModelIR,
) -> Dict[str, Optional[int]]:
    bucket = _native_codegen_cache_bucket_for_model_ir(model_ir=model_ir)
    cached_dims = bucket.get("expected_channel_dims", None)
    if isinstance(cached_dims, dict):
        return cast(Dict[str, Optional[int]], cached_dims)
    graph_index = _native_codegen_graph_index_for_model_ir(model_ir=model_ir)
    candidates_by_tensor_name: Dict[str, Set[int]] = {}
    for op_index in graph_index.operator_indices("CONV_2D"):
        op = model_ir.operators[int(op_index)]
        if len(op.inputs) < 2:
            continue
        weight_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        if weight_tensor is None or len(list(weight_tensor.shape)) != 4:
            continue
        weight_shape = [int(v) for v in list(weight_tensor.shape)]
        input_name = str(op.inputs[0]) if len(op.inputs) >= 1 else ""
        output_name = str(op.outputs[0]) if len(op.outputs) >= 1 else ""
        if input_name != "":
            input_tensor = model_ir.tensors.get(input_name, None)
            output_tensor = (
                model_ir.tensors.get(output_name, None)
                if output_name != ""
                else None
            )
            inferred_input_channels, _ = _infer_conv2d_ctor_params_for_codegen(
                input_shape=(
                    None if input_tensor is None else list(input_tensor.shape)
                ),
                output_shape=(
                    None if output_tensor is None else list(output_tensor.shape)
                ),
                weight_shape=weight_shape,
                options=op.options,
                input_logical_layout=(
                    None
                    if input_tensor is None
                    else str(input_tensor.logical_layout)
                ),
                output_logical_layout=(
                    None
                    if output_tensor is None
                    else str(output_tensor.logical_layout)
                ),
                depthwise=False,
            )
            if int(inferred_input_channels) > 0:
                candidates_by_tensor_name.setdefault(input_name, set()).add(
                    int(inferred_input_channels)
                )
        if output_name != "" and int(weight_shape[0]) > 0:
            candidates_by_tensor_name.setdefault(output_name, set()).add(
                int(weight_shape[0])
            )
    expected_channel_dims = {
        str(tensor_name): (
            next(iter(candidates)) if len(candidates) == 1 else None
        )
        for tensor_name, candidates in candidates_by_tensor_name.items()
    }
    bucket["expected_channel_dims"] = expected_channel_dims
    return expected_channel_dims


def _producer_op_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[OperatorIR]:
    return _native_codegen_graph_index_for_model_ir(
        model_ir=model_ir,
    ).producer(str(tensor_name))


def _expected_channel_dim_for_tensor_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[int]:
    expected_channel_dims = (
        _native_codegen_expected_channel_dim_cache_for_model_ir(
            model_ir=model_ir,
        )
    )
    return expected_channel_dims.get(str(tensor_name), None)


def _expected_channel_dim_for_channel_last_named_tensor_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[int]:
    return _expected_channel_dim_for_tensor_for_codegen(
        model_ir=model_ir,
        tensor_name=tensor_name,
    )


def _target_shape_values_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    public_tensor_names = {
        str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)
    }
    if str(tensor_name) not in public_tensor_names:
        return [int(v) for v in list(tensor.shape)]
    resolved = _base_target_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
    )
    inferred_channel_first = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
    )
    if inferred_channel_first is None:
        return resolved
    layout = normalize_logical_layout(tensor.logical_layout)
    rank = len(inferred_channel_first)
    if is_channel_last_logical_layout(layout):
        perm_to_layout = logical_layout_permutation(
            source_layout=channel_first_logical_layout(rank),
            target_layout=layout,
        )
        permuted = _permute_shape(inferred_channel_first, perm_to_layout or [])
        if permuted is not None:
            return [int(v) for v in list(permuted)]
    if is_channel_first_logical_layout(layout):
        return [int(v) for v in list(inferred_channel_first)]
    return resolved


def _base_target_shape_values_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    if tensor.shape_signature is not None:
        signature = [int(v) for v in list(tensor.shape_signature)]
        if len(signature) == len(list(tensor.shape)):
            preferred = signature
        else:
            preferred = [int(v) for v in list(tensor.shape)]
    else:
        preferred = [int(v) for v in list(tensor.shape)]
    return _resolve_channel_first_named_tensor_shape_for_codegen(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
        preferred=preferred,
        logical_layout=str(tensor.logical_layout),
    )


def _channel_first_shape_values_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
    _seen: Optional[Set[str]] = None,
) -> Optional[List[int]]:
    current_name = str(tensor_name)
    if _seen is None:
        _seen = set()
    bucket = _native_codegen_cache_bucket_for_model_ir(model_ir=model_ir)
    cached_shapes = bucket.setdefault("channel_first_shapes", {})
    in_progress = bucket.setdefault("channel_first_shapes_in_progress", set())
    if not isinstance(cached_shapes, dict):
        cached_shapes = {}
        bucket["channel_first_shapes"] = cached_shapes
    if not isinstance(in_progress, set):
        in_progress = set()
        bucket["channel_first_shapes_in_progress"] = in_progress
    if current_name in cached_shapes:
        cached_value = cached_shapes.get(current_name, None)
        return (
            None
            if cached_value is None
            else [int(v) for v in list(cached_value)]
        )
    if current_name in _seen:
        base_shape = _base_target_shape_values_for_model_ir(
            model_ir=model_ir,
            tensor_name=current_name,
        )
        return _to_channel_first_shape_for_model_ir(
            model_ir=model_ir,
            tensor_name=current_name,
            shape_values=base_shape,
        )
    if current_name in in_progress:
        base_shape = _base_target_shape_values_for_model_ir(
            model_ir=model_ir,
            tensor_name=current_name,
        )
        return _to_channel_first_shape_for_model_ir(
            model_ir=model_ir,
            tensor_name=current_name,
            shape_values=base_shape,
        )
    in_progress.add(current_name)
    base_shape = _base_target_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=current_name,
    )
    channel_first_shape = _to_channel_first_shape_for_model_ir(
        model_ir=model_ir,
        tensor_name=current_name,
        shape_values=base_shape,
    )
    if channel_first_shape is None:
        cached_shapes[current_name] = None
        in_progress.discard(current_name)
        return None
    tensor = model_ir.tensors.get(current_name, None)
    if tensor is None:
        cached_shapes[current_name] = tuple(
            int(v) for v in list(channel_first_shape)
        )
        in_progress.discard(current_name)
        return channel_first_shape
    rank = len(channel_first_shape)
    if rank not in {3, 4, 5}:
        cached_shapes[current_name] = tuple(
            int(v) for v in list(channel_first_shape)
        )
        in_progress.discard(current_name)
        return channel_first_shape
    producer_op = _producer_op_for_model_ir(
        model_ir=model_ir,
        tensor_name=current_name,
    )
    if producer_op is None or str(producer_op.op_type) not in {
        "ADD",
        "DIV",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "SUB",
    }:
        cached_shapes[current_name] = tuple(
            int(v) for v in list(channel_first_shape)
        )
        in_progress.discard(current_name)
        return channel_first_shape
    next_seen = set(_seen)
    next_seen.add(current_name)
    broadcast_shape: Optional[List[int]] = None
    for input_name in list(producer_op.inputs)[:2]:
        input_tensor = model_ir.tensors.get(str(input_name), None)
        if input_tensor is None:
            cached_shapes[current_name] = tuple(
                int(v) for v in list(channel_first_shape)
            )
            in_progress.discard(current_name)
            return channel_first_shape
        if (
            isinstance(input_tensor.data, np.ndarray)
            and int(np.asarray(input_tensor.data).size) == 1
        ):
            continue
        input_shape = _channel_first_shape_values_for_model_ir(
            model_ir=model_ir,
            tensor_name=str(input_name),
            _seen=next_seen,
        )
        if input_shape is None or len(input_shape) != rank:
            cached_shapes[current_name] = tuple(
                int(v) for v in list(channel_first_shape)
            )
            in_progress.discard(current_name)
            return channel_first_shape
        broadcast_shape = (
            [int(v) for v in list(input_shape)]
            if broadcast_shape is None
            else _broadcast_shapes_relaxed(broadcast_shape, input_shape)
        )
        if broadcast_shape is None:
            cached_shapes[current_name] = tuple(
                int(v) for v in list(channel_first_shape)
            )
            in_progress.discard(current_name)
            return channel_first_shape
    if broadcast_shape is not None:
        cached_shapes[current_name] = tuple(int(v) for v in broadcast_shape)
        in_progress.discard(current_name)
        return [int(v) for v in broadcast_shape]
    cached_shapes[current_name] = tuple(int(v) for v in channel_first_shape)
    in_progress.discard(current_name)
    return channel_first_shape


def _to_channel_first_shape_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
    shape_values: Optional[Sequence[int]],
) -> Optional[List[int]]:
    if shape_values is None:
        return None
    values = [int(v) for v in list(shape_values)]
    rank = len(values)
    if rank not in {3, 4, 5}:
        return values
    tensor = model_ir.tensors.get(str(tensor_name), None)
    layout = (
        normalize_logical_layout(tensor.logical_layout)
        if tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    perm_to_cf = _perm_cl_to_cf(rank)
    if is_channel_last_logical_layout(layout) and perm_to_cf is not None:
        expected_channels = _expected_channel_dim_for_tensor_for_codegen(
            model_ir=model_ir,
            tensor_name=str(tensor_name),
        )
        if expected_channels is not None and len(values) >= 3:
            second_axis_matches = int(values[1]) == int(expected_channels)
            last_axis_matches = int(values[-1]) == int(expected_channels)
            if second_axis_matches and not last_axis_matches:
                return values
            if last_axis_matches and not second_axis_matches:
                permuted = _permute_shape(values, perm_to_cf)
                if permuted is not None:
                    return [int(v) for v in list(permuted)]
                return values
        if (
            rank in {4, 5}
            and int(values[1]) > 0
            and int(values[1]) <= 4
            and int(values[-1]) > 4
        ):
            return values
        permuted = _permute_shape(values, perm_to_cf)
        if permuted is not None:
            return [int(v) for v in list(permuted)]
    return values


def _resolve_channel_first_named_tensor_shape_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
    preferred: Sequence[int],
    logical_layout: str,
) -> List[int]:
    resolved = [int(v) for v in list(preferred)]
    rank = len(list(resolved))
    perm_to_cf = _perm_cl_to_cf(rank)
    if (
        perm_to_cf is None
        or not is_channel_first_logical_layout(
            normalize_logical_layout(logical_layout)
        )
        or not _tensor_name_suggests_channel_last_layout_for_codegen(
            str(tensor_name)
        )
    ):
        return resolved
    expected_channels = (
        _expected_channel_dim_for_channel_last_named_tensor_for_codegen(
            model_ir=model_ir,
            tensor_name=str(tensor_name),
        )
    )
    if expected_channels is None and rank == 4:
        return resolved
    if expected_channels is not None and len(resolved) >= 3:
        second_axis_matches = int(resolved[1]) == int(expected_channels)
        last_axis_matches = int(resolved[-1]) == int(expected_channels)
        if second_axis_matches and not last_axis_matches:
            return resolved
        if last_axis_matches and not second_axis_matches:
            permuted = _permute_shape(resolved, perm_to_cf)
            if permuted is not None:
                return [int(v) for v in list(permuted)]
            return resolved
    permuted = _permute_shape(resolved, perm_to_cf)
    if permuted is not None:
        return [int(v) for v in list(permuted)]
    return resolved


def _target_shape_literal_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> str:
    target_shape = _target_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
    )
    if target_shape is None:
        return "None"
    return repr([int(v) for v in list(target_shape)])


def _resize_target_shape_literal_for_model_ir(
    *,
    model_ir: ModelIR,
    output_name: str,
    input_name: str,
) -> str:
    output_tensor = model_ir.tensors.get(str(output_name), None)
    input_tensor = model_ir.tensors.get(str(input_name), None)
    if output_tensor is None:
        return "None"
    target_shape = [int(v) for v in list(output_tensor.shape)]
    if input_tensor is None:
        return repr(target_shape)
    input_shape = [int(v) for v in list(input_tensor.shape)]
    if target_shape == input_shape:
        return repr(target_shape)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    if (
        len(input_shape) == 4
        and len(target_shape) == 4
        and is_channel_first_logical_layout(input_layout)
        and is_channel_first_logical_layout(output_layout)
    ):
        if (
            int(target_shape[1]) != int(input_shape[1])
            and int(target_shape[-1]) == int(input_shape[1])
        ):
            return repr(
                [
                    int(input_shape[0]),
                    int(input_shape[1]),
                    int(target_shape[1]),
                    int(target_shape[2]),
                ]
            )
        if (
            int(target_shape[1]) != int(input_shape[-1])
            and int(target_shape[-1]) == int(input_shape[-1])
        ):
            return repr(
                [
                    int(input_shape[0]),
                    int(input_shape[-1]),
                    int(target_shape[1]),
                    int(target_shape[2]),
                ]
            )
    return repr(target_shape)


def _tensor_shape_list_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    return [int(v) for v in list(tensor.shape)]


def _rank4_channel_first_shape_for_tensor_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_name: str,
) -> Optional[List[int]]:
    inferred_shape = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
    )
    if inferred_shape is None or len(inferred_shape) != 4:
        return None
    tensor_shape = list(inferred_shape)
    tensor = model_ir.tensors.get(str(tensor_name), None)
    tensor_layout = (
        normalize_logical_layout(tensor.logical_layout)
        if tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    if (
        str(tensor_name) in channel_first_tensor_expr_aliases
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
        return [
            int(tensor_shape[0]),
            int(tensor_shape[3]),
            int(tensor_shape[1]),
            int(tensor_shape[2]),
        ]
    if is_channel_last_logical_layout(tensor_layout):
        return [
            int(tensor_shape[0]),
            int(tensor_shape[3]),
            int(tensor_shape[1]),
            int(tensor_shape[2]),
        ]
    return [int(v) for v in list(tensor_shape)]


def _channel_first_shape_for_tensor_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_name: str,
) -> Optional[List[int]]:
    tensor_shape = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(tensor_name),
    )
    if tensor_shape is None:
        return None
    rank = len(list(tensor_shape))
    if rank not in {3, 4, 5}:
        return [int(v) for v in list(tensor_shape)]
    tensor = model_ir.tensors.get(str(tensor_name), None)
    perm_to_cf = _perm_cl_to_cf(rank)
    tensor_layout = (
        normalize_logical_layout(tensor.logical_layout)
        if tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    if (
        str(tensor_name) in channel_first_tensor_expr_aliases
        and (
            is_channel_last_logical_layout(tensor_layout)
            or (
                tensor_layout == LOGICAL_LAYOUT_UNKNOWN
                and _tensor_name_suggests_channel_last_layout_for_codegen(
                    str(tensor_name)
                )
            )
        )
        and perm_to_cf is not None
    ):
        permuted_shape = _permute_shape(tensor_shape, perm_to_cf)
        if permuted_shape is not None:
            return [int(v) for v in list(permuted_shape)]
    if is_channel_last_logical_layout(tensor_layout) and perm_to_cf is not None:
        permuted_shape = _permute_shape(tensor_shape, perm_to_cf)
        if permuted_shape is not None:
            return [int(v) for v in list(permuted_shape)]
    return [int(v) for v in list(tensor_shape)]


def _gather_input_pre_permute_for_codegen(
    *,
    model_ir: ModelIR,
    params_name: str,
    output_name: str,
    axis: int,
    batch_dims: int,
) -> Optional[List[int]]:
    if int(batch_dims) != 0:
        return None
    params_tensor = model_ir.tensors.get(str(params_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if params_tensor is None or output_tensor is None:
        return None
    params_shape = [int(v) for v in list(params_tensor.shape)]
    rank = len(params_shape)
    if rank not in {3, 4, 5}:
        return None
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        and len(list(output_tensor.shape_signature)) == rank
        else [int(v) for v in list(output_tensor.shape)]
    )
    if len(output_signature) != rank:
        return None
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += rank
    if resolved_axis < 0 or resolved_axis >= rank:
        return None

    def _matches_signature(shape: Sequence[int]) -> bool:
        for dim_idx, expected_dim in enumerate(output_signature):
            if dim_idx == resolved_axis or int(expected_dim) <= 0:
                continue
            if int(shape[dim_idx]) != int(expected_dim):
                return False
        return True

    if _matches_signature(params_shape):
        return None
    for perm in (_perm_cf_to_cl(rank), _perm_cl_to_cf(rank)):
        if perm is None:
            continue
        perm_values = [int(v) for v in list(perm)]
        permuted_shape = [int(params_shape[int(idx)]) for idx in perm_values]
        if _matches_signature(permuted_shape):
            return perm_values
    return None


def _infer_effective_rank4_runtime_layout_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    consumer_index: Dict[str, List[int]],
    tensor_name: str,
) -> Optional[str]:
    current_name = str(tensor_name)
    visited: Set[str] = set()
    passthrough_ops = {
        "ADD",
        "AVERAGE_POOL_2D",
        "CAST",
        "IDENTITY",
        "LEAKY_RELU",
        "LOGISTIC",
        "MAX_POOL_2D",
        "MUL",
        "PAD",
        "PADV2",
        "RELU",
        "RELU6",
        "RELU_N1_TO_1",
        "RELU_0_TO_1",
        "TANH",
    }
    while current_name not in visited:
        visited.add(current_name)
        current_tensor = model_ir.tensors.get(current_name, None)
        if current_tensor is None or len(list(current_tensor.shape)) != 4:
            return None
        current_shape = [int(v) for v in list(current_tensor.shape)]
        current_layout = normalize_logical_layout(current_tensor.logical_layout)
        if is_channel_last_logical_layout(current_layout):
            return "NHWC"
        if is_channel_first_logical_layout(current_layout):
            return "NCHW"

        producer_idx = producer_index.get(current_name, None)
        if producer_idx is None:
            consumer_indices = consumer_index.get(current_name, [])
            if len(consumer_indices) != 1:
                return None
            consumer_op = model_ir.operators[int(consumer_indices[0])]
            if (
                str(consumer_op.op_type) != "CONV_2D"
                or len(consumer_op.inputs) < 2
            ):
                return None
            filter_tensor = model_ir.tensors.get(
                str(consumer_op.inputs[1]),
                None,
            )
            if filter_tensor is None or len(list(filter_tensor.shape)) != 4:
                return None
            filter_shape = [int(v) for v in list(filter_tensor.shape)]
            input_channels = int(filter_shape[3])
            if (
                current_shape[3] == input_channels
                and current_shape[1] != input_channels
            ):
                return "NHWC"
            if (
                current_shape[1] == input_channels
                and current_shape[3] != input_channels
            ):
                return "NCHW"
            return None

        producer_op = model_ir.operators[int(producer_idx)]
        producer_type = str(producer_op.op_type)
        if producer_type == "CONV_2D" and len(producer_op.inputs) >= 2:
            filter_tensor = model_ir.tensors.get(
                str(producer_op.inputs[1]),
                None,
            )
            if filter_tensor is None or len(list(filter_tensor.shape)) != 4:
                return None
            filter_shape = [int(v) for v in list(filter_tensor.shape)]
            out_channels = int(filter_shape[0])
            if (
                current_shape[3] == out_channels
                and current_shape[1] != out_channels
            ):
                return "NHWC"
            if (
                current_shape[1] == out_channels
                and current_shape[3] != out_channels
            ):
                return "NCHW"
            return None
        if producer_type not in passthrough_ops or len(producer_op.inputs) <= 0:
            return None

        exact_shape_name = None
        permuted_shape_name = None
        fallback_name = None
        perm_cl_to_cf = _perm_cl_to_cf(4)
        perm_cf_to_cl = _perm_cf_to_cl(4)
        for candidate_input in list(producer_op.inputs):
            candidate_name = str(candidate_input)
            candidate_tensor = model_ir.tensors.get(candidate_name, None)
            if candidate_tensor is None or candidate_tensor.data is not None:
                continue
            if len(list(candidate_tensor.shape)) == 4:
                candidate_shape = [int(v) for v in list(candidate_tensor.shape)]
                if candidate_shape == current_shape:
                    exact_shape_name = candidate_name
                    break
                if (
                    perm_cl_to_cf is not None
                    and _permute_shape(candidate_shape, perm_cl_to_cf)
                    == current_shape
                ) or (
                    perm_cf_to_cl is not None
                    and _permute_shape(candidate_shape, perm_cf_to_cl)
                    == current_shape
                ):
                    if permuted_shape_name is None:
                        permuted_shape_name = candidate_name
                    continue
                if fallback_name is None:
                    fallback_name = candidate_name
        next_name = exact_shape_name or permuted_shape_name or fallback_name
        if next_name is None:
            return None
        current_name = next_name
    return None
