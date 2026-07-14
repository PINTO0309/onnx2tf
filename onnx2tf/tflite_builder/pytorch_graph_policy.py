from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, cast

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _perm_cf_to_cl,
    _perm_cl_to_cf,
    _permute_shape,
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
