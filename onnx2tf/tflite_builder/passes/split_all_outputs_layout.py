from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
)


_STATS_KEY = "optimized_transpose_relu_split_all_outputs_to_nhwc_chains"
_CONV_CONCAT_STATS_KEY = (
    "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)


@dataclass(frozen=True)
class _View:
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _InputRewrite:
    operator: OperatorIR
    original_inputs: Tuple[str, ...]
    new_inputs: Tuple[str, ...]


@dataclass(frozen=True)
class _MetadataUpdate:
    name: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _AxisUpdate:
    source_name: str
    target_name: str
    in_place: bool
    dtype: str
    numpy_dtype: str


@dataclass(frozen=True)
class _Plan:
    pre: OperatorIR
    relu: OperatorIR
    split: OperatorIR
    posts: Tuple[OperatorIR, ...]
    input_rewrites: Tuple[_InputRewrite, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    axis_update: _AxisUpdate
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class _ConcatAxisUpdate:
    operator: OperatorIR
    original_axis: int
    new_axis: int


@dataclass(frozen=True)
class _ConvConcatPlan:
    pre: OperatorIR
    pre_relu: OperatorIR
    split: OperatorIR
    pre_branch: OperatorIR
    conv: OperatorIR
    post_branch: OperatorIR
    post_relu: OperatorIR
    concat: OperatorIR
    post_concat: OperatorIR
    input_rewrites: Tuple[_InputRewrite, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    axis_update: _AxisUpdate
    options_update: _ConcatAxisUpdate
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]


def _op_type(operator: OperatorIR) -> str:
    return str(operator.op_type).upper()


def _operator_index(
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
) -> Optional[int]:
    value = graph_index.operator_index(operator)
    return None if value is None else int(value)


def _signature(tensor: TensorIR) -> Tuple[int, ...]:
    values = tensor.shape if tensor.shape_signature is None else tensor.shape_signature
    return tuple(int(value) for value in values)


def _view(tensor: TensorIR) -> _View:
    return _View(
        shape=tuple(int(value) for value in tensor.shape),
        signature=_signature(tensor),
        dtype=str(tensor.dtype),
    )


def _rank4_positive(view: _View) -> bool:
    return bool(
        len(view.shape) == 4
        and len(view.signature) == 4
        and all(int(value) > 0 for value in view.shape)
    )


def _layout_of(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
) -> str:
    if layout_state is not None:
        return str(layout_state.physical_of(str(name))).upper()
    return str(tensor.physical_layout).upper()


def _resolved_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    before_index: int,
) -> bool:
    normalized_name = str(name)
    tensor = model_ir.tensors.get(normalized_name)
    if (
        tensor is None
        or bool(tensor.is_variable)
        or normalized_name in graph_index.duplicate_producers
    ):
        return False
    producer_index = graph_index.producers.get(normalized_name)
    if producer_index is not None:
        return int(producer_index) < int(before_index)
    return bool(
        normalized_name in {str(value) for value in model_ir.inputs}
        or tensor.data is not None
    )


def _per_tensor_quantization(quantization: Any) -> bool:
    if quantization is None:
        return True
    scale = getattr(quantization, "scale", None)
    if scale is None and isinstance(quantization, dict):
        scale = quantization.get("scale")
    if scale is None:
        return True
    try:
        return int(np.asarray(scale).size) <= 1
    except Exception:
        return False


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            str(array.dtype),
            tuple(int(item) for item in array.shape),
            sha256(array.tobytes()).digest(),
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "scale") and hasattr(value, "zero_point"):
        return (
            tuple(float(item) for item in value.scale),
            tuple(int(item) for item in value.zero_point),
            int(value.quantized_dimension),
            _freeze(value.min),
            _freeze(value.max),
        )
    return value


def _tensor_contract(name: str, tensor: TensorIR) -> Tuple[Any, ...]:
    return (
        str(name),
        id(tensor),
        str(tensor.name),
        str(tensor.dtype),
        tuple(int(value) for value in tensor.shape),
        _signature(tensor),
        _freeze(tensor.data),
        bool(tensor.is_variable),
        _freeze(tensor.quantization),
        str(tensor.logical_layout),
        str(tensor.physical_layout),
        tensor.onnx_tensor_name,
    )


def _operator_contract(operator: OperatorIR) -> Tuple[Any, ...]:
    return (
        id(operator),
        str(operator.op_type),
        tuple(str(value) for value in operator.inputs),
        tuple(str(value) for value in operator.outputs),
        _freeze(operator.options),
        _freeze(operator.axis_semantics),
        int(operator.version),
        operator.onnx_node_name,
        operator.onnx_op_type,
    )


def _typed_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    shape: Sequence[int],
) -> Optional[Tuple[TensorIR, np.ndarray]]:
    tensor = model_ir.tensors.get(str(name))
    if (
        tensor is None
        or tensor.data is None
        or str(tensor.dtype) not in {"INT32", "INT64"}
        or bool(tensor.is_variable)
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or str(name) in {str(value) for value in model_ir.inputs}
        or str(name) in {str(value) for value in model_ir.outputs}
        or tuple(int(value) for value in tensor.shape) != tuple(int(value) for value in shape)
        or _signature(tensor) != tuple(int(value) for value in shape)
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return None
    data = np.asarray(tensor.data)
    if (
        data.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}
        or tuple(int(value) for value in data.shape)
        != tuple(int(value) for value in shape)
    ):
        return None
    return tensor, data


def _typed_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    expected: Sequence[int],
) -> bool:
    if (
        _op_type(operator) != "TRANSPOSE"
        or len(operator.inputs) != 2
        or len(operator.outputs) != 1
    ):
        return False
    resolved = _typed_constant(
        model_ir,
        graph_index,
        str(operator.inputs[1]),
        shape=(4,),
    )
    return bool(
        resolved is not None
        and tuple(int(value) for value in resolved[1].reshape(-1))
        == tuple(int(value) for value in expected)
    )


def _unique_name(base: str, occupied: set[str]) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in occupied:
        candidate = f"{base}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _consumer_slots(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Tuple[Tuple[OperatorIR, int], ...]:
    slots = []
    for operator_index in graph_index.consumer_indices(str(name)):
        operator = model_ir.operators[int(operator_index)]
        slots.extend(
            (operator, input_index)
            for input_index, input_name in enumerate(operator.inputs)
            if str(input_name) == str(name)
        )
    return tuple(slots)


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        id(plan.relu),
        id(plan.split),
        tuple(id(operator) for operator in plan.posts),
        tuple(
            (id(rewrite.operator), rewrite.original_inputs, rewrite.new_inputs)
            for rewrite in plan.input_rewrites
        ),
        tuple(
            (update.name, update.shape, update.signature)
            for update in plan.metadata_updates
        ),
        (
            plan.axis_update.source_name,
            plan.axis_update.target_name,
            plan.axis_update.in_place,
            plan.axis_update.dtype,
            plan.axis_update.numpy_dtype,
        ),
        tuple(id(operator) for operator in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
    )


def _permuted_view(
    view: _View,
    permutation: Sequence[int],
) -> Optional[_View]:
    shape = _permute_shape(list(view.shape), list(permutation))
    signature = _permute_shape(list(view.signature), list(permutation))
    if shape is None or signature is None:
        return None
    return _View(
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        dtype=view.dtype,
    )


def _concat_view(
    left: _View,
    right: _View,
    *,
    axis: int,
) -> Optional[_View]:
    if (
        left.dtype != right.dtype
        or len(left.shape) != 4
        or len(right.shape) != 4
        or len(left.signature) != 4
        or len(right.signature) != 4
        or int(axis) not in range(4)
    ):
        return None
    output_shape = []
    output_signature = []
    for index, (left_dim, right_dim) in enumerate(zip(left.shape, right.shape)):
        if index == int(axis):
            output_shape.append(int(left_dim) + int(right_dim))
            left_signature = int(left.signature[index])
            right_signature = int(right.signature[index])
            output_signature.append(
                left_signature + right_signature
                if left_signature >= 0 and right_signature >= 0
                else -1
            )
            continue
        if int(left_dim) != int(right_dim):
            return None
        output_shape.append(int(left_dim))
        left_signature = int(left.signature[index])
        right_signature = int(right.signature[index])
        if (
            left_signature >= 0
            and right_signature >= 0
            and left_signature != right_signature
        ):
            return None
        output_signature.append(
            left_signature if left_signature >= 0 else right_signature
        )
    return _View(
        shape=tuple(output_shape),
        signature=tuple(output_signature),
        dtype=left.dtype,
    )


def _conv_concat_plan_signature(plan: _ConvConcatPlan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        id(plan.pre_relu),
        id(plan.split),
        id(plan.pre_branch),
        id(plan.conv),
        id(plan.post_branch),
        id(plan.post_relu),
        id(plan.concat),
        id(plan.post_concat),
        tuple(
            (id(rewrite.operator), rewrite.original_inputs, rewrite.new_inputs)
            for rewrite in plan.input_rewrites
        ),
        tuple(
            (update.name, update.shape, update.signature)
            for update in plan.metadata_updates
        ),
        (
            plan.axis_update.source_name,
            plan.axis_update.target_name,
            plan.axis_update.in_place,
            plan.axis_update.dtype,
            plan.axis_update.numpy_dtype,
        ),
        (
            id(plan.options_update.operator),
            plan.options_update.original_axis,
            plan.options_update.new_axis,
        ),
        tuple(id(operator) for operator in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    split: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    split_index = _operator_index(graph_index, split)
    if (
        split_index is None
        or _op_type(split) != "SPLIT"
        or len(split.inputs) != 2
        or len(split.outputs) < 2
    ):
        return None
    axis_name = str(split.inputs[0])
    axis_resolved = _typed_constant(
        model_ir,
        graph_index,
        axis_name,
        shape=(1,),
    )
    if axis_resolved is None:
        return None
    axis_value = int(axis_resolved[1].reshape(-1)[0])
    if axis_value < 0:
        axis_value += 4
    if axis_value != 1:
        return None
    if not isinstance(split.options, dict):
        return None
    num_splits = split.options.get("numSplits")
    if num_splits is not None:
        try:
            if int(num_splits) != len(split.outputs):
                return None
        except Exception:
            return None

    graph_inputs = {str(value) for value in model_ir.inputs}
    graph_outputs = {str(value) for value in model_ir.outputs}
    public_names = graph_inputs | graph_outputs
    relu_output_name = str(split.inputs[1])
    if relu_output_name in public_names or relu_output_name in graph_index.duplicate_producers:
        return None
    relu_index = graph_index.producers.get(relu_output_name)
    if relu_index is None or int(relu_index) >= int(split_index):
        return None
    relu = model_ir.operators[int(relu_index)]
    if (
        _op_type(relu) != "RELU"
        or len(relu.inputs) != 1
        or len(relu.outputs) != 1
        or str(relu.outputs[0]) != relu_output_name
    ):
        return None
    pre_output_name = str(relu.inputs[0])
    if pre_output_name in public_names or pre_output_name in graph_index.duplicate_producers:
        return None
    pre_index = graph_index.producers.get(pre_output_name)
    if pre_index is None or int(pre_index) >= int(relu_index):
        return None
    pre = model_ir.operators[int(pre_index)]
    if not _typed_permutation(
        model_ir,
        graph_index,
        pre,
        _PERM_NHWC_TO_NCHW,
    ) or str(pre.outputs[0]) != pre_output_name:
        return None
    source_name = str(pre.inputs[0])
    if (
        source_name in graph_outputs
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(pre_index),
        )
    ):
        return None
    source_tensor = model_ir.tensors.get(source_name)
    pre_output_tensor = model_ir.tensors.get(pre_output_name)
    relu_output_tensor = model_ir.tensors.get(relu_output_name)
    if source_tensor is None or pre_output_tensor is None or relu_output_tensor is None:
        return None
    source_view = _view(source_tensor)
    pre_view = _view(pre_output_tensor)
    relu_view = _view(relu_output_tensor)
    if (
        not _rank4_positive(source_view)
        or not _rank4_positive(pre_view)
        or not _rank4_positive(relu_view)
        or source_view.dtype != pre_view.dtype
        or pre_view != relu_view
        or bool(source_tensor.is_variable)
        or pre_output_tensor.data is not None
        or bool(pre_output_tensor.is_variable)
        or relu_output_tensor.data is not None
        or bool(relu_output_tensor.is_variable)
        or not _per_tensor_quantization(source_tensor.quantization)
        or not _per_tensor_quantization(pre_output_tensor.quantization)
        or not _per_tensor_quantization(relu_output_tensor.quantization)
        or tuple(
            _permute_shape(list(source_view.shape), list(_PERM_NHWC_TO_NCHW))
            or ()
        )
        != pre_view.shape
        or tuple(
            _permute_shape(
                list(source_view.signature),
                list(_PERM_NHWC_TO_NCHW),
            )
            or ()
        )
        != pre_view.signature
        or graph_index.consumer_indices(pre_output_name) != [int(relu_index)]
        or graph_index.consumer_indices(relu_output_name) != [int(split_index)]
        or _layout_of(source_name, source_tensor, layout_state)
        not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}
        or _layout_of(pre_output_name, pre_output_tensor, layout_state)
        not in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN}
        or _layout_of(relu_output_name, relu_output_tensor, layout_state)
        not in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN}
    ):
        return None

    output_count = len(split.outputs)
    channel_count = int(relu_view.shape[1])
    if channel_count % output_count != 0:
        return None
    branch_channels = channel_count // output_count
    if int(relu_view.signature[1]) > 0:
        if int(relu_view.signature[1]) % output_count != 0:
            return None
        branch_channel_signature = int(relu_view.signature[1]) // output_count
    else:
        branch_channel_signature = -1
    expected_old_shape = (
        int(relu_view.shape[0]),
        branch_channels,
        int(relu_view.shape[2]),
        int(relu_view.shape[3]),
    )
    expected_old_signature = (
        int(relu_view.signature[0]),
        branch_channel_signature,
        int(relu_view.signature[2]),
        int(relu_view.signature[3]),
    )
    expected_new_shape = tuple(
        _permute_shape(list(expected_old_shape), list(_PERM_NCHW_TO_NHWC)) or ()
    )
    expected_new_signature = tuple(
        _permute_shape(
            list(expected_old_signature),
            list(_PERM_NCHW_TO_NHWC),
        )
        or ()
    )

    posts = []
    consumer_replacements: Dict[int, list[str]] = {}
    downstream_operators: Dict[int, OperatorIR] = {}
    metadata_updates = [
        _MetadataUpdate(
            name=relu_output_name,
            shape=source_view.shape,
            signature=source_view.signature,
        )
    ]
    for split_output in split.outputs:
        split_output_name = str(split_output)
        if split_output_name in public_names or split_output_name in graph_index.duplicate_producers:
            return None
        split_tensor = model_ir.tensors.get(split_output_name)
        if split_tensor is None:
            return None
        split_view = _view(split_tensor)
        if (
            split_view.shape != expected_old_shape
            or split_view.signature != expected_old_signature
            or split_view.dtype != relu_view.dtype
            or split_tensor.data is not None
            or bool(split_tensor.is_variable)
            or not _per_tensor_quantization(split_tensor.quantization)
            or _layout_of(split_output_name, split_tensor, layout_state)
            not in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN}
        ):
            return None
        post_indices = graph_index.consumer_indices(split_output_name)
        if len(post_indices) != 1:
            return None
        post_index = int(post_indices[0])
        if post_index <= int(split_index):
            return None
        post = model_ir.operators[post_index]
        if not _typed_permutation(
            model_ir,
            graph_index,
            post,
            _PERM_NCHW_TO_NHWC,
        ) or str(post.inputs[0]) != split_output_name:
            return None
        post_output_name = str(post.outputs[0])
        if post_output_name in public_names or post_output_name in graph_index.duplicate_producers:
            return None
        post_output_tensor = model_ir.tensors.get(post_output_name)
        if post_output_tensor is None:
            return None
        post_view = _view(post_output_tensor)
        if (
            post_view.shape != expected_new_shape
            or post_view.signature != expected_new_signature
            or post_view.dtype != split_view.dtype
            or post_output_tensor.data is not None
            or bool(post_output_tensor.is_variable)
            or not _per_tensor_quantization(post_output_tensor.quantization)
            or _layout_of(post_output_name, post_output_tensor, layout_state)
            not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}
        ):
            return None
        for consumer, input_index in _consumer_slots(
            model_ir,
            graph_index,
            post_output_name,
        ):
            consumer_index = _operator_index(graph_index, consumer)
            if consumer_index is None or int(consumer_index) <= post_index:
                return None
            downstream_operators[id(consumer)] = consumer
            new_inputs = consumer_replacements.setdefault(
                id(consumer),
                [str(value) for value in consumer.inputs],
            )
            new_inputs[int(input_index)] = split_output_name
        posts.append(post)
        metadata_updates.append(
            _MetadataUpdate(
                name=split_output_name,
                shape=expected_new_shape,
                signature=expected_new_signature,
            )
        )

    axis_slots = {
        (id(operator), int(input_index))
        for operator, input_index in _consumer_slots(
            model_ir,
            graph_index,
            axis_name,
        )
    }
    split_axis_slot = {(id(split), 0)}
    if not split_axis_slot <= axis_slots:
        return None
    axis_in_place = axis_slots == split_axis_slot
    occupied = {str(name) for name in model_ir.tensors}
    target_axis_name = axis_name
    if not axis_in_place:
        target_axis_name = _unique_name(f"{axis_name}_nhwc", occupied)
    axis_tensor, axis_data = axis_resolved
    axis_update = _AxisUpdate(
        source_name=axis_name,
        target_name=target_axis_name,
        in_place=axis_in_place,
        dtype=str(axis_tensor.dtype),
        numpy_dtype=str(axis_data.dtype),
    )

    planned_inputs: Dict[int, list[str]] = {
        id(relu): [source_name],
        id(split): [target_axis_name, relu_output_name],
    }
    planned_inputs.update(consumer_replacements)
    rewritten_operators = [relu, split, *downstream_operators.values()]
    input_rewrites = tuple(
        _InputRewrite(
            operator=operator,
            original_inputs=tuple(str(value) for value in operator.inputs),
            new_inputs=tuple(planned_inputs[id(operator)]),
        )
        for operator in rewritten_operators
        if tuple(str(value) for value in operator.inputs)
        != tuple(planned_inputs[id(operator)])
    )

    contract_names = {
        source_name,
        pre_output_name,
        relu_output_name,
        axis_name,
        str(pre.inputs[1]),
    }
    contract_names.update(str(value) for value in split.outputs)
    for post in posts:
        contract_names.update(str(value) for value in (*post.inputs, *post.outputs))
    tensor_contracts = []
    for name in sorted(contract_names):
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))
    operator_contracts = tuple(
        _operator_contract(operator)
        for operator in (pre, relu, split, *posts, *downstream_operators.values())
    )
    return _Plan(
        pre=pre,
        relu=relu,
        split=split,
        posts=tuple(posts),
        input_rewrites=input_rewrites,
        metadata_updates=tuple(metadata_updates),
        axis_update=axis_update,
        removals=(pre, *posts),
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=operator_contracts,
    )


def _resolve_conv_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    concat: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_ConvConcatPlan]:
    concat_index = _operator_index(graph_index, concat)
    if (
        concat_index is None
        or _op_type(concat) != "CONCATENATION"
        or len(concat.inputs) != 2
        or len(concat.outputs) != 1
        or not isinstance(concat.options, dict)
    ):
        return None
    try:
        original_concat_axis = int(concat.options.get("axis", 0))
    except Exception:
        return None
    normalized_concat_axis = int(original_concat_axis)
    if normalized_concat_axis < 0:
        normalized_concat_axis += 4
    if normalized_concat_axis != 1:
        return None

    graph_inputs = {str(value) for value in model_ir.inputs}
    graph_outputs = {str(value) for value in model_ir.outputs}
    public_names = graph_inputs | graph_outputs
    concat_output_name = str(concat.outputs[0])
    if (
        concat_output_name in public_names
        or concat_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(concat_output_name) != int(concat_index)
    ):
        return None
    post_concat_indices = graph_index.consumer_indices(concat_output_name)
    if len(post_concat_indices) != 1:
        return None
    post_concat_index = int(post_concat_indices[0])
    if post_concat_index <= int(concat_index):
        return None
    post_concat = model_ir.operators[post_concat_index]
    if (
        not _typed_permutation(
            model_ir,
            graph_index,
            post_concat,
            _PERM_NCHW_TO_NHWC,
        )
        or str(post_concat.inputs[0]) != concat_output_name
    ):
        return None
    post_concat_output_name = str(post_concat.outputs[0])
    if (
        post_concat_output_name in public_names
        or post_concat_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(post_concat_output_name)
        != int(post_concat_index)
    ):
        return None

    downstream_replacements: Dict[int, list[str]] = {}
    downstream_operators: Dict[int, OperatorIR] = {}
    for consumer, input_slot in _consumer_slots(
        model_ir,
        graph_index,
        post_concat_output_name,
    ):
        consumer_index = _operator_index(graph_index, consumer)
        if consumer_index is None or int(consumer_index) <= int(post_concat_index):
            return None
        downstream_operators[id(consumer)] = consumer
        new_inputs = downstream_replacements.setdefault(
            id(consumer),
            [str(value) for value in consumer.inputs],
        )
        new_inputs[int(input_slot)] = concat_output_name

    resolved_branch: Optional[Tuple[Any, ...]] = None
    concat_inputs = tuple(str(value) for value in concat.inputs)
    if concat_inputs[0] == concat_inputs[1]:
        return None
    for branch_slot in range(2):
        branch_input_name = concat_inputs[branch_slot]
        keep_input_name = concat_inputs[1 - branch_slot]
        if (
            branch_input_name in public_names
            or branch_input_name in graph_index.duplicate_producers
        ):
            continue
        post_relu_index = graph_index.producers.get(branch_input_name)
        if post_relu_index is None or int(post_relu_index) >= int(concat_index):
            continue
        post_relu = model_ir.operators[int(post_relu_index)]
        if (
            _op_type(post_relu) != "RELU"
            or len(post_relu.inputs) != 1
            or len(post_relu.outputs) != 1
            or str(post_relu.outputs[0]) != branch_input_name
            or graph_index.consumer_indices(branch_input_name)
            != [int(concat_index)]
        ):
            continue

        post_branch_output_name = str(post_relu.inputs[0])
        if (
            post_branch_output_name in public_names
            or post_branch_output_name in graph_index.duplicate_producers
        ):
            continue
        post_branch_index = graph_index.producers.get(post_branch_output_name)
        if (
            post_branch_index is None
            or int(post_branch_index) >= int(post_relu_index)
        ):
            continue
        post_branch = model_ir.operators[int(post_branch_index)]
        if (
            not _typed_permutation(
                model_ir,
                graph_index,
                post_branch,
                _PERM_NHWC_TO_NCHW,
            )
            or str(post_branch.outputs[0]) != post_branch_output_name
            or graph_index.consumer_indices(post_branch_output_name)
            != [int(post_relu_index)]
        ):
            continue

        conv_output_name = str(post_branch.inputs[0])
        if (
            conv_output_name in public_names
            or conv_output_name in graph_index.duplicate_producers
        ):
            continue
        conv_index = graph_index.producers.get(conv_output_name)
        if conv_index is None or int(conv_index) >= int(post_branch_index):
            continue
        conv = model_ir.operators[int(conv_index)]
        if (
            _op_type(conv) != "CONV_2D"
            or len(conv.inputs) not in {2, 3}
            or len(conv.outputs) != 1
            or str(conv.outputs[0]) != conv_output_name
            or not isinstance(conv.options, dict)
            or graph_index.consumer_indices(conv_output_name)
            != [int(post_branch_index)]
            or not all(
                not str(name).strip()
                or _resolved_source(
                    model_ir,
                    graph_index,
                    name=str(name),
                    before_index=int(conv_index),
                )
                for name in conv.inputs
            )
        ):
            continue

        conv_input_name = str(conv.inputs[0])
        if (
            conv_input_name in public_names
            or conv_input_name in graph_index.duplicate_producers
        ):
            continue
        pre_branch_index = graph_index.producers.get(conv_input_name)
        if pre_branch_index is None or int(pre_branch_index) >= int(conv_index):
            continue
        pre_branch = model_ir.operators[int(pre_branch_index)]
        if (
            not _typed_permutation(
                model_ir,
                graph_index,
                pre_branch,
                _PERM_NCHW_TO_NHWC,
            )
            or str(pre_branch.outputs[0]) != conv_input_name
            or graph_index.consumer_indices(conv_input_name) != [int(conv_index)]
        ):
            continue

        branch_split_output_name = str(pre_branch.inputs[0])
        if (
            branch_split_output_name in public_names
            or branch_split_output_name in graph_index.duplicate_producers
            or keep_input_name in public_names
            or keep_input_name in graph_index.duplicate_producers
        ):
            continue
        split_index = graph_index.producers.get(branch_split_output_name)
        if split_index is None or int(split_index) >= int(pre_branch_index):
            continue
        split = model_ir.operators[int(split_index)]
        split_outputs = tuple(str(value) for value in split.outputs)
        if (
            _op_type(split) != "SPLIT"
            or len(split.inputs) != 2
            or len(split_outputs) != 2
            or len(set(split_outputs)) != 2
            or set(split_outputs)
            != {branch_split_output_name, keep_input_name}
            or graph_index.consumer_indices(branch_split_output_name)
            != [int(pre_branch_index)]
            or graph_index.consumer_indices(keep_input_name)
            != [int(concat_index)]
            or not isinstance(split.options, dict)
        ):
            continue
        num_splits = split.options.get("numSplits")
        if num_splits is not None:
            try:
                if int(num_splits) != 2:
                    continue
            except Exception:
                continue
        axis_name = str(split.inputs[0])
        axis_resolved = _typed_constant(
            model_ir,
            graph_index,
            axis_name,
            shape=(1,),
        )
        if axis_resolved is None:
            continue
        split_axis = int(axis_resolved[1].reshape(-1)[0])
        if split_axis < 0:
            split_axis += 4
        if split_axis != 1:
            continue

        pre_relu_output_name = str(split.inputs[1])
        if (
            pre_relu_output_name in public_names
            or pre_relu_output_name in graph_index.duplicate_producers
        ):
            continue
        pre_relu_index = graph_index.producers.get(pre_relu_output_name)
        if pre_relu_index is None or int(pre_relu_index) >= int(split_index):
            continue
        pre_relu = model_ir.operators[int(pre_relu_index)]
        if (
            _op_type(pre_relu) != "RELU"
            or len(pre_relu.inputs) != 1
            or len(pre_relu.outputs) != 1
            or str(pre_relu.outputs[0]) != pre_relu_output_name
            or graph_index.consumer_indices(pre_relu_output_name)
            != [int(split_index)]
        ):
            continue

        pre_output_name = str(pre_relu.inputs[0])
        if (
            pre_output_name in public_names
            or pre_output_name in graph_index.duplicate_producers
        ):
            continue
        pre_index = graph_index.producers.get(pre_output_name)
        if pre_index is None or int(pre_index) >= int(pre_relu_index):
            continue
        pre = model_ir.operators[int(pre_index)]
        if (
            not _typed_permutation(
                model_ir,
                graph_index,
                pre,
                _PERM_NHWC_TO_NCHW,
            )
            or str(pre.outputs[0]) != pre_output_name
            or graph_index.consumer_indices(pre_output_name)
            != [int(pre_relu_index)]
        ):
            continue
        source_name = str(pre.inputs[0])
        if (
            source_name in graph_outputs
            or not _resolved_source(
                model_ir,
                graph_index,
                name=source_name,
                before_index=int(pre_index),
            )
        ):
            continue
        if not (
            int(pre_index)
            < int(pre_relu_index)
            < int(split_index)
            < int(pre_branch_index)
            < int(conv_index)
            < int(post_branch_index)
            < int(post_relu_index)
            < int(concat_index)
            < int(post_concat_index)
        ):
            continue
        resolved_branch = (
            branch_input_name,
            keep_input_name,
            branch_split_output_name,
            source_name,
            pre_output_name,
            pre_relu_output_name,
            conv_input_name,
            conv_output_name,
            post_branch_output_name,
            axis_name,
            axis_resolved,
            pre,
            pre_relu,
            split,
            pre_branch,
            conv,
            post_branch,
            post_relu,
        )
        break

    if resolved_branch is None:
        return None
    (
        branch_input_name,
        keep_input_name,
        branch_split_output_name,
        source_name,
        pre_output_name,
        pre_relu_output_name,
        conv_input_name,
        conv_output_name,
        post_branch_output_name,
        axis_name,
        axis_resolved,
        pre,
        pre_relu,
        split,
        pre_branch,
        conv,
        post_branch,
        post_relu,
    ) = resolved_branch

    tensor_names = (
        source_name,
        pre_output_name,
        pre_relu_output_name,
        branch_split_output_name,
        keep_input_name,
        conv_input_name,
        conv_output_name,
        post_branch_output_name,
        branch_input_name,
        concat_output_name,
        post_concat_output_name,
    )
    tensors = {name: model_ir.tensors.get(name) for name in tensor_names}
    if any(tensor is None for tensor in tensors.values()):
        return None
    source_tensor = tensors[source_name]
    assert source_tensor is not None
    runtime_names = tensor_names[1:]
    if (
        bool(source_tensor.is_variable)
        or any(
            tensors[name] is None
            or tensors[name].data is not None
            or bool(tensors[name].is_variable)
            or not _per_tensor_quantization(tensors[name].quantization)
            for name in runtime_names
        )
        or not _per_tensor_quantization(source_tensor.quantization)
    ):
        return None
    views = {name: _view(tensor) for name, tensor in tensors.items() if tensor is not None}
    if not all(_rank4_positive(view) for view in views.values()):
        return None

    source_view = views[source_name]
    pre_view = views[pre_output_name]
    pre_relu_view = views[pre_relu_output_name]
    branch_old_view = views[branch_split_output_name]
    keep_old_view = views[keep_input_name]
    conv_input_view = views[conv_input_name]
    conv_output_view = views[conv_output_name]
    post_branch_view = views[post_branch_output_name]
    post_relu_view = views[branch_input_name]
    concat_old_view = views[concat_output_name]
    post_concat_view = views[post_concat_output_name]
    expected_pre = _permuted_view(source_view, _PERM_NHWC_TO_NCHW)
    if expected_pre is None or pre_view != expected_pre or pre_relu_view != pre_view:
        return None
    input_channels = int(pre_relu_view.shape[1])
    signature_channels = int(pre_relu_view.signature[1])
    if input_channels % 2 != 0 or (
        signature_channels >= 0 and signature_channels % 2 != 0
    ):
        return None
    expected_split_view = _View(
        shape=(
            int(pre_relu_view.shape[0]),
            input_channels // 2,
            int(pre_relu_view.shape[2]),
            int(pre_relu_view.shape[3]),
        ),
        signature=(
            int(pre_relu_view.signature[0]),
            signature_channels // 2 if signature_channels >= 0 else -1,
            int(pre_relu_view.signature[2]),
            int(pre_relu_view.signature[3]),
        ),
        dtype=pre_relu_view.dtype,
    )
    branch_new_view = _permuted_view(
        expected_split_view,
        _PERM_NCHW_TO_NHWC,
    )
    keep_new_view = _permuted_view(
        expected_split_view,
        _PERM_NCHW_TO_NHWC,
    )
    if (
        branch_old_view != expected_split_view
        or keep_old_view != expected_split_view
        or branch_new_view is None
        or keep_new_view is None
        or conv_input_view != branch_new_view
        or _permuted_view(conv_output_view, _PERM_NHWC_TO_NCHW)
        != post_branch_view
        or post_relu_view != post_branch_view
    ):
        return None
    old_concat_inputs = (
        post_relu_view if concat_inputs[0] == branch_input_name else keep_old_view,
        post_relu_view if concat_inputs[1] == branch_input_name else keep_old_view,
    )
    new_concat_inputs = (
        conv_output_view if concat_inputs[0] == branch_input_name else keep_new_view,
        conv_output_view if concat_inputs[1] == branch_input_name else keep_new_view,
    )
    expected_old_concat = _concat_view(*old_concat_inputs, axis=1)
    expected_new_concat = _concat_view(*new_concat_inputs, axis=3)
    if (
        expected_old_concat is None
        or expected_new_concat is None
        or concat_old_view != expected_old_concat
        or _permuted_view(concat_old_view, _PERM_NCHW_TO_NHWC)
        != post_concat_view
        or expected_new_concat != post_concat_view
    ):
        return None

    adapter_pairs = (
        (source_name, pre_output_name),
        (branch_split_output_name, conv_input_name),
        (conv_output_name, post_branch_output_name),
        (concat_output_name, post_concat_output_name),
    )
    if any(
        views[left].dtype != views[right].dtype
        or _freeze(tensors[left].quantization)
        != _freeze(tensors[right].quantization)
        for left, right in adapter_pairs
    ):
        return None
    if (
        _layout_of(source_name, source_tensor, layout_state)
        not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}
        or any(
            _layout_of(name, tensors[name], layout_state)
            not in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN}
            for name in (
                pre_output_name,
                pre_relu_output_name,
                branch_split_output_name,
                keep_input_name,
                post_branch_output_name,
                branch_input_name,
                concat_output_name,
            )
        )
        or any(
            _layout_of(name, tensors[name], layout_state)
            not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}
            for name in (
                conv_input_name,
                conv_output_name,
                post_concat_output_name,
            )
        )
    ):
        return None

    axis_slots = {
        (id(operator), int(input_slot))
        for operator, input_slot in _consumer_slots(
            model_ir,
            graph_index,
            axis_name,
        )
    }
    split_axis_slot = {(id(split), 0)}
    if not split_axis_slot <= axis_slots:
        return None
    axis_in_place = axis_slots == split_axis_slot
    target_axis_name = axis_name
    if not axis_in_place:
        target_axis_name = _unique_name(
            f"{axis_name}_nhwc",
            {str(name) for name in model_ir.tensors},
        )
    axis_tensor, axis_data = axis_resolved
    axis_update = _AxisUpdate(
        source_name=axis_name,
        target_name=target_axis_name,
        in_place=axis_in_place,
        dtype=str(axis_tensor.dtype),
        numpy_dtype=str(axis_data.dtype),
    )

    planned_inputs: Dict[int, list[str]] = {
        id(pre_relu): [source_name],
        id(split): [target_axis_name, pre_relu_output_name],
        id(conv): [branch_split_output_name, *[str(value) for value in conv.inputs[1:]]],
        id(post_relu): [conv_output_name],
    }
    planned_inputs.update(downstream_replacements)
    rewritten_operators = [
        pre_relu,
        split,
        conv,
        post_relu,
        *downstream_operators.values(),
    ]
    input_rewrites = tuple(
        _InputRewrite(
            operator=operator,
            original_inputs=tuple(str(value) for value in operator.inputs),
            new_inputs=tuple(planned_inputs[id(operator)]),
        )
        for operator in rewritten_operators
        if tuple(str(value) for value in operator.inputs)
        != tuple(planned_inputs[id(operator)])
    )
    metadata_updates = (
        _MetadataUpdate(
            name=pre_relu_output_name,
            shape=source_view.shape,
            signature=source_view.signature,
        ),
        _MetadataUpdate(
            name=branch_split_output_name,
            shape=branch_new_view.shape,
            signature=branch_new_view.signature,
        ),
        _MetadataUpdate(
            name=keep_input_name,
            shape=keep_new_view.shape,
            signature=keep_new_view.signature,
        ),
        _MetadataUpdate(
            name=branch_input_name,
            shape=conv_output_view.shape,
            signature=conv_output_view.signature,
        ),
        _MetadataUpdate(
            name=concat_output_name,
            shape=post_concat_view.shape,
            signature=post_concat_view.signature,
        ),
    )
    relevant_operators = [
        pre,
        pre_relu,
        split,
        pre_branch,
        conv,
        post_branch,
        post_relu,
        concat,
        post_concat,
        *downstream_operators.values(),
    ]
    relevant_operators = sorted(
        relevant_operators,
        key=lambda operator: int(_operator_index(graph_index, operator) or 0),
    )
    contract_names = set()
    for operator in relevant_operators:
        contract_names.update(
            str(value) for value in (*operator.inputs, *operator.outputs) if str(value)
        )
    tensor_contracts = []
    for name in sorted(contract_names):
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))
    return _ConvConcatPlan(
        pre=pre,
        pre_relu=pre_relu,
        split=split,
        pre_branch=pre_branch,
        conv=conv,
        post_branch=post_branch,
        post_relu=post_relu,
        concat=concat,
        post_concat=post_concat,
        input_rewrites=input_rewrites,
        metadata_updates=metadata_updates,
        axis_update=axis_update,
        options_update=_ConcatAxisUpdate(
            operator=concat,
            original_axis=original_concat_axis,
            new_axis=3,
        ),
        removals=(pre, pre_branch, post_branch, post_concat),
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in relevant_operators
        ),
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _Plan,
    *,
    layout_state: Optional[LayoutState],
) -> bool:
    current = _resolve_candidate(
        model_ir,
        graph_index,
        plan.split,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if any(
        _operator_index(graph_index, operator) is None for operator in plan.removals
    ):
        return False
    for rewrite in plan.input_rewrites:
        if tuple(str(value) for value in rewrite.operator.inputs) != rewrite.original_inputs:
            return False
    update = plan.axis_update
    source_axis = model_ir.tensors.get(update.source_name)
    if source_axis is None:
        return False
    if not update.in_place and update.target_name in model_ir.tensors:
        return False
    for metadata in plan.metadata_updates:
        if metadata.name not in model_ir.tensors:
            return False

    numpy_dtype = np.dtype(update.numpy_dtype)
    if update.in_place:
        target_axis = source_axis
    else:
        target_axis = TensorIR(
            name=update.target_name,
            dtype=update.dtype,
            shape=[1],
            shape_signature=[1],
            data=np.asarray([3], dtype=numpy_dtype),
            is_variable=False,
            quantization=_clone_quantization(source_axis.quantization),
            logical_layout=str(source_axis.logical_layout),
            physical_layout=str(source_axis.physical_layout),
            onnx_tensor_name=source_axis.onnx_tensor_name,
        )
        model_ir.tensors[update.target_name] = target_axis
    target_axis.data = np.asarray([3], dtype=numpy_dtype)
    target_axis.shape = [1]
    target_axis.shape_signature = [1]
    if layout_state is not None and not update.in_place:
        layout_state.set(
            update.target_name,
            logical=str(source_axis.logical_layout),
            physical=str(source_axis.physical_layout),
        )

    for rewrite in plan.input_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    for metadata in plan.metadata_updates:
        tensor = model_ir.tensors[metadata.name]
        tensor.shape = [int(value) for value in metadata.shape]
        tensor.shape_signature = [int(value) for value in metadata.signature]
        tensor.logical_layout = LOGICAL_LAYOUT_NHWC
        tensor.physical_layout = LOGICAL_LAYOUT_NHWC
        if layout_state is not None:
            layout_state.set(
                metadata.name,
                logical=LOGICAL_LAYOUT_NHWC,
                physical=LOGICAL_LAYOUT_NHWC,
            )
    removal_indices = [
        int(_operator_index(graph_index, operator)) for operator in plan.removals
    ]
    graph_index.remove_operators(removal_indices)
    return True


def _apply_conv_concat_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _ConvConcatPlan,
    *,
    layout_state: Optional[LayoutState],
) -> bool:
    current = _resolve_conv_concat_candidate(
        model_ir,
        graph_index,
        plan.concat,
        layout_state=layout_state,
    )
    if (
        current is None
        or _conv_concat_plan_signature(current)
        != _conv_concat_plan_signature(plan)
    ):
        return False
    if any(
        _operator_index(graph_index, operator) is None for operator in plan.removals
    ):
        return False
    for rewrite in plan.input_rewrites:
        if tuple(str(value) for value in rewrite.operator.inputs) != rewrite.original_inputs:
            return False
    update = plan.axis_update
    source_axis = model_ir.tensors.get(update.source_name)
    if source_axis is None:
        return False
    if not update.in_place and update.target_name in model_ir.tensors:
        return False
    for metadata in plan.metadata_updates:
        if metadata.name not in model_ir.tensors:
            return False
    options_update = plan.options_update
    if (
        not isinstance(options_update.operator.options, dict)
        or options_update.operator is not plan.concat
    ):
        return False
    try:
        if (
            int(options_update.operator.options.get("axis", 0))
            != int(options_update.original_axis)
        ):
            return False
    except Exception:
        return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))

    numpy_dtype = np.dtype(update.numpy_dtype)
    if update.in_place:
        target_axis = source_axis
    else:
        target_axis = TensorIR(
            name=update.target_name,
            dtype=update.dtype,
            shape=[1],
            shape_signature=[1],
            data=np.asarray([3], dtype=numpy_dtype),
            is_variable=False,
            quantization=_clone_quantization(source_axis.quantization),
            logical_layout=str(source_axis.logical_layout),
            physical_layout=str(source_axis.physical_layout),
            onnx_tensor_name=source_axis.onnx_tensor_name,
        )
        model_ir.tensors[update.target_name] = target_axis
    target_axis.data = np.asarray([3], dtype=numpy_dtype)
    target_axis.shape = [1]
    target_axis.shape_signature = [1]
    if layout_state is not None and not update.in_place:
        layout_state.set(
            update.target_name,
            logical=str(source_axis.logical_layout),
            physical=str(source_axis.physical_layout),
        )

    for rewrite in plan.input_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    concat_options = dict(options_update.operator.options)
    concat_options["axis"] = int(options_update.new_axis)
    options_update.operator.options = concat_options
    for metadata in plan.metadata_updates:
        tensor = model_ir.tensors[metadata.name]
        tensor.shape = [int(value) for value in metadata.shape]
        tensor.shape_signature = [int(value) for value in metadata.signature]
        tensor.logical_layout = LOGICAL_LAYOUT_NHWC
        tensor.physical_layout = LOGICAL_LAYOUT_NHWC
        if layout_state is not None:
            layout_state.set(
                metadata.name,
                logical=LOGICAL_LAYOUT_NHWC,
                physical=LOGICAL_LAYOUT_NHWC,
            )
    graph_index.remove_operators(removal_indices)
    return True


def optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Move a closed RELU/Split/all-inverse-adapter island to NHWC."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    candidates = (
        [candidate]
        if candidate is not None
        else [
            model_ir.operators[index]
            for index in active_index.operator_indices_for_normalized_types({"SPLIT"})
        ]
    )
    rewrite_limit = (
        len(candidates) if max_rewrites is None else max(0, int(max_rewrites))
    )
    if rewrite_limit == 0:
        return {_STATS_KEY: 0}
    rewritten = 0
    for split in candidates:
        if rewritten >= rewrite_limit:
            break
        if split is None or _operator_index(active_index, split) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            split,
            layout_state=layout_state,
        )
        if plan is None:
            continue
        if _apply_plan(
            model_ir,
            active_index,
            plan,
            layout_state=layout_state,
        ):
            rewritten += 1
    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {_STATS_KEY: int(rewritten)}


def optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Move the exact two-branch RELU/Split/Conv/Concat island to NHWC."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    candidates = (
        [candidate]
        if candidate is not None
        else [
            model_ir.operators[index]
            for index in active_index.operator_indices_for_normalized_types(
                {"CONCATENATION"}
            )
        ]
    )
    rewrite_limit = (
        len(candidates) if max_rewrites is None else max(0, int(max_rewrites))
    )
    if rewrite_limit == 0:
        return {_CONV_CONCAT_STATS_KEY: 0}
    rewritten = 0
    for concat in candidates:
        if rewritten >= rewrite_limit:
            break
        if concat is None or _operator_index(active_index, concat) is None:
            continue
        plan = _resolve_conv_concat_candidate(
            model_ir,
            active_index,
            concat,
            layout_state=layout_state,
        )
        if plan is None:
            continue
        if _apply_conv_concat_plan(
            model_ir,
            active_index,
            plan,
            layout_state=layout_state,
        ):
            rewritten += 1
    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {_CONV_CONCAT_STATS_KEY: int(rewritten)}
