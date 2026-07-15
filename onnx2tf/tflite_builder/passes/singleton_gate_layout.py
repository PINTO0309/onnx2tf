from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _broadcast_shape_signatures,
    _broadcast_static_shapes,
    _permute_shape,
    _prune_unused_tensors,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    ModelIR,
    OperatorIR,
    TensorIR,
)


_STATS_KEY = "optimized_singleton_gate_conv_concat_nhwc_bridge_blocks"
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_UNARY_OPS = frozenset(
    {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "LOGISTIC",
        "HARD_SWISH",
        "LEAKY_RELU",
        "TANH",
    }
)


@dataclass(frozen=True)
class _View:
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _MetadataUpdate:
    name: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    logical_layout: str = LOGICAL_LAYOUT_NHWC
    physical_layout: str = LOGICAL_LAYOUT_NHWC


@dataclass(frozen=True)
class _SingletonAdapter:
    operator: OperatorIR
    input_name: str
    output_name: str
    input_view: _View
    output_view: _View


@dataclass(frozen=True)
class _InputRewrite:
    operator: OperatorIR
    original_inputs: Tuple[str, ...]
    new_inputs: Tuple[str, ...]


@dataclass(frozen=True)
class _Plan:
    concat: OperatorIR
    clip3_adapter: _SingletonAdapter
    clip3_unary: OperatorIR
    add: OperatorIR
    gate_mul: OperatorIR
    signal_mul: OperatorIR
    sub: OperatorIR
    clip_adapter: _SingletonAdapter
    aux_adapter: _SingletonAdapter
    signal: Optional[OperatorIR]
    split_adapter: Optional[_SingletonAdapter]
    rgb_mul: Optional[OperatorIR]
    input_transpose: Optional[OperatorIR]
    rewrites: Tuple[_InputRewrite, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
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


def _compatible(expected: Sequence[int], actual: Sequence[int]) -> bool:
    return bool(
        len(expected) == len(actual)
        and all(
            int(left) == int(right) or int(left) < 0 or int(right) < 0
            for left, right in zip(expected, actual)
        )
    )


def _layout_of(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
) -> str:
    if layout_state is not None:
        return str(layout_state.physical_of(str(name))).upper()
    return str(tensor.physical_layout).upper()


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (
            str(value.dtype),
            tuple(int(item) for item in value.shape),
            tuple(value.reshape(-1).tolist()),
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


def _resolved_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    before_index: int,
) -> bool:
    normalized = str(name)
    if normalized in graph_index.duplicate_producers:
        return False
    tensor = model_ir.tensors.get(normalized)
    if tensor is None:
        return False
    producer_index = graph_index.producers.get(normalized)
    return bool(
        (producer_index is not None and int(producer_index) < int(before_index))
        or normalized in {str(value) for value in model_ir.inputs}
        or tensor.data is not None
    )


def _inputs_resolved_before(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    operator_index: int,
) -> bool:
    return all(
        _resolved_source(
            model_ir,
            graph_index,
            name=str(name),
            before_index=operator_index,
        )
        for name in operator.inputs
    )


def _permuted_view(view: _View, permutation: Sequence[int]) -> Optional[_View]:
    shape = _permute_shape(list(view.shape), list(permutation))
    signature = _permute_shape(list(view.signature), list(permutation))
    if shape is None or signature is None:
        return None
    return _View(
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        dtype=view.dtype,
    )


def _broadcast(left: _View, right: _View) -> Optional[_View]:
    if left.dtype != right.dtype:
        return None
    shape = _broadcast_static_shapes(list(left.shape), list(right.shape))
    signature = _broadcast_shape_signatures(
        list(left.signature),
        list(right.signature),
    )
    if shape is None or signature is None:
        return None
    return _View(
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        dtype=left.dtype,
    )


def _metadata_update_for_nchw_tensor(
    model_ir: ModelIR,
    name: str,
    *,
    expected: Optional[_View] = None,
) -> Optional[_MetadataUpdate]:
    tensor = model_ir.tensors.get(str(name))
    if (
        tensor is None
        or tensor.data is not None
        or bool(tensor.is_variable)
        or len(tensor.shape) != 4
        or not _all_per_tensor_quantized([tensor])
    ):
        return None
    converted = _permuted_view(_view(tensor), _PERM_NCHW_TO_NHWC)
    if (
        converted is None
        or (expected is not None and converted.shape != expected.shape)
        or (
            expected is not None
            and not _compatible(converted.signature, expected.signature)
        )
        or (expected is not None and converted.dtype != expected.dtype)
    ):
        return None
    return _MetadataUpdate(
        name=str(name),
        shape=converted.shape,
        signature=converted.signature,
    )


def _typed_shape_input_matches(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    output_shape: Sequence[int],
) -> bool:
    if len(operator.inputs) == 1:
        return True
    if len(operator.inputs) != 2:
        return False
    name = str(operator.inputs[1])
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or bool(tensor.is_variable)
        or tensor.quantization is not None
        or str(tensor.dtype).upper() not in {"INT32", "INT64"}
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
    ):
        return False
    try:
        array = np.asarray(tensor.data)
    except Exception:
        return False
    return bool(
        array.dtype in {np.dtype(np.int32), np.dtype(np.int64)}
        and tuple(int(value) for value in array.reshape(-1).tolist())
        == tuple(int(value) for value in output_shape)
    )


def _resolve_singleton_adapter(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    direction: str,
) -> Optional[_SingletonAdapter]:
    operator_index = _operator_index(graph_index, operator)
    if (
        operator_index is None
        or _op_type(operator) != "RESHAPE"
        or len(operator.inputs) not in {1, 2}
        or len(operator.outputs) != 1
        or not _inputs_resolved_before(
            model_ir,
            graph_index,
            operator,
            operator_index,
        )
    ):
        return None
    input_name = str(operator.inputs[0])
    output_name = str(operator.outputs[0])
    if (
        output_name in graph_index.duplicate_producers
        or graph_index.producers.get(output_name) != operator_index
        or output_name in {str(value) for value in model_ir.inputs + model_ir.outputs}
    ):
        return None
    input_tensor = model_ir.tensors.get(input_name)
    output_tensor = model_ir.tensors.get(output_name)
    if (
        input_tensor is None
        or output_tensor is None
        or bool(input_tensor.is_variable)
        or bool(output_tensor.is_variable)
        or output_tensor.data is not None
        or len(input_tensor.shape) != 4
        or len(output_tensor.shape) != 4
        or str(input_tensor.dtype) != str(output_tensor.dtype)
        or not _all_per_tensor_quantized([input_tensor, output_tensor])
        or not _typed_shape_input_matches(
            model_ir,
            graph_index,
            operator,
            output_tensor.shape,
        )
    ):
        return None
    input_view = _view(input_tensor)
    output_view = _view(output_tensor)
    permutation = (
        _PERM_NHWC_TO_NCHW if direction == "nhwc_to_nchw" else _PERM_NCHW_TO_NHWC
    )
    expected = _permuted_view(input_view, permutation)
    if (
        expected is None
        or expected.shape != output_view.shape
        or not _compatible(expected.signature, output_view.signature)
        or int(input_view.shape[3 if direction == "nhwc_to_nchw" else 1]) != 1
        or int(output_view.shape[1 if direction == "nhwc_to_nchw" else 3]) != 1
    ):
        return None
    option_shape = operator.options.get("newShape")
    if option_shape is not None:
        try:
            if tuple(int(value) for value in option_shape) != output_view.shape:
                return None
        except Exception:
            return None
    return _SingletonAdapter(
        operator=operator,
        input_name=input_name,
        output_name=output_name,
        input_view=input_view,
        output_view=output_view,
    )


def _producer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Optional[OperatorIR]:
    normalized = str(name)
    index = graph_index.producers.get(normalized)
    if index is None or normalized in graph_index.duplicate_producers:
        return None
    return model_ir.operators[int(index)]


def _plain_binary(operator: OperatorIR, op_type: str) -> bool:
    return bool(
        _op_type(operator) == str(op_type).upper()
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
        and str(operator.options.get("fusedActivationFunction", "NONE")).upper()
        in {"", "NONE"}
    )


def _resolve_external_view(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    before_index: int,
    dtype: str,
    rank_limit: int = 4,
) -> Optional[_View]:
    tensor = model_ir.tensors.get(str(name))
    if (
        tensor is None
        or bool(tensor.is_variable)
        or str(tensor.dtype) != dtype
        or len(tensor.shape) > rank_limit
        or not _all_per_tensor_quantized([tensor])
        or not _resolved_source(
            model_ir,
            graph_index,
            name=str(name),
            before_index=before_index,
        )
    ):
        return None
    return _view(tensor)


def _resolve_stale_nchw_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    before_index: int,
    expected: _View,
    layout_state: Optional[LayoutState],
) -> Optional[_MetadataUpdate]:
    normalized = str(name)
    tensor = model_ir.tensors.get(normalized)
    if (
        tensor is None
        or normalized in {str(value) for value in model_ir.inputs + model_ir.outputs}
        or not _resolved_source(
            model_ir,
            graph_index,
            name=normalized,
            before_index=before_index,
        )
        or (
            tensor.data is None
            and _layout_of(normalized, tensor, layout_state) != LOGICAL_LAYOUT_NHWC
        )
    ):
        return None
    if tensor.data is not None:
        try:
            array = np.asarray(tensor.data)
        except Exception:
            return None
        if int(array.size) != int(np.prod(tensor.shape, dtype=np.int64)):
            return None
        converted = _permuted_view(_view(tensor), _PERM_NCHW_TO_NHWC)
        if (
            converted is None
            or converted.shape != expected.shape
            or not _compatible(converted.signature, expected.signature)
            or converted.dtype != expected.dtype
            or not _all_per_tensor_quantized([tensor])
        ):
            return None
        return _MetadataUpdate(
            name=normalized,
            shape=converted.shape,
            signature=converted.signature,
        )
    return _metadata_update_for_nchw_tensor(
        model_ir,
        normalized,
        expected=expected,
    )


def _concat_view(views: Sequence[_View]) -> Optional[_View]:
    if len(views) == 0 or any(len(view.shape) != 4 for view in views):
        return None
    if len({view.dtype for view in views}) != 1:
        return None
    shape = list(views[0].shape)
    signature = list(views[0].signature)
    for view in views[1:]:
        for axis in range(3):
            if int(shape[axis]) != int(view.shape[axis]):
                return None
            if not _compatible([signature[axis]], [view.signature[axis]]):
                return None
            if int(signature[axis]) < 0 or int(view.signature[axis]) < 0:
                signature[axis] = -1
        shape[3] += int(view.shape[3])
        if int(signature[3]) < 0 or int(view.signature[3]) < 0:
            signature[3] = -1
        else:
            signature[3] += int(view.signature[3])
    return _View(tuple(shape), tuple(signature), views[0].dtype)


def _typed_input_transpose(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    consumer: OperatorIR,
) -> Optional[Tuple[str, _View]]:
    operator_index = _operator_index(graph_index, operator)
    consumer_index = _operator_index(graph_index, consumer)
    if (
        operator_index is None
        or consumer_index is None
        or operator_index >= consumer_index
        or _op_type(operator) != "TRANSPOSE"
        or len(operator.inputs) != 2
        or len(operator.outputs) != 1
        or graph_index.consumer_indices(str(operator.outputs[0])) != [consumer_index]
    ):
        return None
    input_name = str(operator.inputs[0])
    output_name = str(operator.outputs[0])
    permutation_name = str(operator.inputs[1])
    permutation = model_ir.tensors.get(permutation_name)
    input_tensor = model_ir.tensors.get(input_name)
    output_tensor = model_ir.tensors.get(output_name)
    if (
        permutation is None
        or permutation.data is None
        or bool(permutation.is_variable)
        or permutation.quantization is not None
        or str(permutation.dtype).upper() not in {"INT32", "INT64"}
        or permutation_name in graph_index.producers
        or permutation_name in graph_index.duplicate_producers
        or input_tensor is None
        or output_tensor is None
        or input_name in {str(value) for value in model_ir.outputs}
        or output_name in {str(value) for value in model_ir.inputs + model_ir.outputs}
        or not _resolved_source(
            model_ir,
            graph_index,
            name=input_name,
            before_index=operator_index,
        )
        or str(input_tensor.dtype) != str(output_tensor.dtype)
        or not _all_per_tensor_quantized([input_tensor, output_tensor])
    ):
        return None
    try:
        array = np.asarray(permutation.data)
    except Exception:
        return None
    if (
        array.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}
        or tuple(int(value) for value in array.reshape(-1).tolist())
        != _PERM_NHWC_TO_NCHW
    ):
        return None
    input_view = _view(input_tensor)
    output_view = _view(output_tensor)
    expected = _permuted_view(input_view, _PERM_NHWC_TO_NCHW)
    if (
        expected is None
        or expected.shape != output_view.shape
        or not _compatible(expected.signature, output_view.signature)
    ):
        return None
    return input_name, input_view


def _metadata_for_binary(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    op_type: str,
    views: Dict[str, _View],
) -> Optional[Tuple[_MetadataUpdate, _View]]:
    operator_index = _operator_index(graph_index, operator)
    if (
        operator_index is None
        or not _plain_binary(operator, op_type)
        or not _inputs_resolved_before(
            model_ir,
            graph_index,
            operator,
            operator_index,
        )
    ):
        return None
    input_names = tuple(str(value) for value in operator.inputs)
    if any(name not in views for name in input_names):
        return None
    expected = _broadcast(views[input_names[0]], views[input_names[1]])
    update = _metadata_update_for_nchw_tensor(
        model_ir,
        str(operator.outputs[0]),
        expected=expected,
    )
    if expected is None or update is None:
        return None
    return update, expected


def _metadata_for_unary(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    input_view: _View,
) -> Optional[Tuple[_MetadataUpdate, _View]]:
    operator_index = _operator_index(graph_index, operator)
    if (
        operator_index is None
        or _op_type(operator) not in _UNARY_OPS
        or len(operator.inputs) != 1
        or len(operator.outputs) != 1
        or not _inputs_resolved_before(
            model_ir,
            graph_index,
            operator,
            operator_index,
        )
    ):
        return None
    update = _metadata_update_for_nchw_tensor(
        model_ir,
        str(operator.outputs[0]),
        expected=input_view,
    )
    if update is None:
        return None
    return update, input_view


def _consumer_set(
    graph_index: ModelIRGraphIndex,
    name: str,
) -> set[int]:
    return {int(value) for value in graph_index.consumer_indices(str(name))}


def _make_rewrites(
    graph_index: ModelIRGraphIndex,
    replacements: Sequence[Tuple[OperatorIR, str, str]],
) -> Optional[Tuple[_InputRewrite, ...]]:
    grouped: Dict[int, Tuple[OperatorIR, Dict[str, str]]] = {}
    for operator, old_name, new_name in replacements:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None or old_name == new_name:
            return None
        if operator_index not in grouped:
            grouped[operator_index] = (operator, {})
        mapping = grouped[operator_index][1]
        if old_name in mapping and mapping[old_name] != new_name:
            return None
        mapping[old_name] = new_name
    rewrites = []
    for operator_index in sorted(grouped):
        operator, mapping = grouped[operator_index]
        original = tuple(str(value) for value in operator.inputs)
        if any(name not in original for name in mapping):
            return None
        new_inputs = tuple(mapping.get(name, name) for name in original)
        if new_inputs == original:
            return None
        rewrites.append(_InputRewrite(operator, original, new_inputs))
    return tuple(rewrites)


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.concat),
        id(plan.clip3_adapter.operator),
        id(plan.clip3_unary),
        id(plan.add),
        id(plan.gate_mul),
        id(plan.signal_mul),
        id(plan.sub),
        id(plan.clip_adapter.operator),
        id(plan.aux_adapter.operator),
        None if plan.signal is None else id(plan.signal),
        None if plan.split_adapter is None else id(plan.split_adapter.operator),
        None if plan.rgb_mul is None else id(plan.rgb_mul),
        None if plan.input_transpose is None else id(plan.input_transpose),
        tuple(
            (id(value.operator), value.original_inputs, value.new_inputs)
            for value in plan.rewrites
        ),
        plan.metadata_updates,
        tuple(id(value) for value in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    concat: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    concat_index = _operator_index(graph_index, concat)
    if (
        concat_index is None
        or _op_type(concat) != "CONCATENATION"
        or len(concat.inputs) < 2
        or len(concat.outputs) != 1
        or not _inputs_resolved_before(
            model_ir,
            graph_index,
            concat,
            concat_index,
        )
    ):
        return None
    try:
        if int(concat.options.get("axis", -1)) != 3:
            return None
    except Exception:
        return None

    clip3_candidates = []
    for name in concat.inputs:
        adapter_operator = _producer(model_ir, graph_index, str(name))
        if adapter_operator is None:
            continue
        adapter = _resolve_singleton_adapter(
            model_ir,
            graph_index,
            adapter_operator,
            direction="nchw_to_nhwc",
        )
        if adapter is None:
            continue
        unary = _producer(model_ir, graph_index, adapter.input_name)
        if (
            unary is None
            or _op_type(unary) not in _UNARY_OPS
            or len(unary.inputs) != 1
            or len(unary.outputs) != 1
        ):
            continue
        add = _producer(model_ir, graph_index, str(unary.inputs[0]))
        if add is None or not _plain_binary(add, "ADD"):
            continue
        clip3_candidates.append((adapter, unary, add))
    if len(clip3_candidates) != 1:
        return None
    clip3_adapter, clip3_unary, add = clip3_candidates[0]

    muls = [_producer(model_ir, graph_index, str(name)) for name in add.inputs]
    if (
        any(value is None or not _plain_binary(value, "MUL") for value in muls)
        or muls[0] is muls[1]
    ):
        return None
    gated_candidates = []
    for signal_mul in (value for value in muls if value is not None):
        for signal_name in tuple(str(value) for value in signal_mul.inputs):
            signal_or_adapter = _producer(model_ir, graph_index, signal_name)
            if signal_or_adapter is None:
                continue
            signal = None
            aux_adapter_operator = signal_or_adapter
            if _op_type(signal_or_adapter) == "LOGISTIC":
                if (
                    len(signal_or_adapter.inputs) != 1
                    or len(signal_or_adapter.outputs) != 1
                ):
                    continue
                signal = signal_or_adapter
                aux_adapter_operator = _producer(
                    model_ir,
                    graph_index,
                    str(signal.inputs[0]),
                )
                if aux_adapter_operator is None:
                    continue
            aux_adapter = _resolve_singleton_adapter(
                model_ir,
                graph_index,
                aux_adapter_operator,
                direction="nhwc_to_nchw",
            )
            if aux_adapter is None:
                continue
            expected_signal_name = (
                aux_adapter.output_name if signal is None else str(signal.outputs[0])
            )
            if signal_name != expected_signal_name:
                continue
            other_names = [
                str(value) for value in signal_mul.inputs if str(value) != signal_name
            ]
            if len(other_names) != 1:
                continue
            sub = _producer(model_ir, graph_index, other_names[0])
            if sub is None or not _plain_binary(sub, "SUB"):
                continue
            clip_adapter_operator = _producer(
                model_ir,
                graph_index,
                str(sub.inputs[1]),
            )
            if clip_adapter_operator is None:
                continue
            clip_adapter = _resolve_singleton_adapter(
                model_ir,
                graph_index,
                clip_adapter_operator,
                direction="nhwc_to_nchw",
            )
            if clip_adapter is None or str(sub.inputs[1]) != clip_adapter.output_name:
                continue
            gated_candidates.append(
                (signal_mul, sub, clip_adapter, aux_adapter, signal)
            )
    if len(gated_candidates) != 1:
        return None
    signal_mul, sub, clip_adapter, aux_adapter, signal = gated_candidates[0]
    gate_mul = next(value for value in muls if value is not signal_mul)
    if gate_mul is None or list(gate_mul.inputs).count(clip_adapter.output_name) != 1:
        return None
    split_names = [
        str(value)
        for value in gate_mul.inputs
        if str(value) != clip_adapter.output_name
    ]
    if len(split_names) != 1:
        return None
    split_name = split_names[0]

    split_adapter_candidates = []
    for concat_input in concat.inputs:
        candidate = _producer(model_ir, graph_index, str(concat_input))
        if candidate is None:
            continue
        adapter = _resolve_singleton_adapter(
            model_ir,
            graph_index,
            candidate,
            direction="nchw_to_nhwc",
        )
        if adapter is not None and adapter.input_name == split_name:
            split_adapter_candidates.append(adapter)
    if len(split_adapter_candidates) > 1:
        return None
    split_adapter = (
        split_adapter_candidates[0] if len(split_adapter_candidates) == 1 else None
    )
    split_output_name = (
        split_name if split_adapter is None else split_adapter.output_name
    )
    if list(concat.inputs).count(split_output_name) != 1:
        return None

    operator_indices = {
        id(operator): _operator_index(graph_index, operator)
        for operator in (
            concat,
            clip3_adapter.operator,
            clip3_unary,
            add,
            gate_mul,
            signal_mul,
            sub,
            clip_adapter.operator,
            aux_adapter.operator,
            *(() if signal is None else (signal,)),
            *(() if split_adapter is None else (split_adapter.operator,)),
        )
    }
    if any(value is None for value in operator_indices.values()):
        return None
    clip3_adapter_index = int(operator_indices[id(clip3_adapter.operator)])
    clip3_unary_index = int(operator_indices[id(clip3_unary)])
    add_index = int(operator_indices[id(add)])
    gate_index = int(operator_indices[id(gate_mul)])
    signal_mul_index = int(operator_indices[id(signal_mul)])
    sub_index = int(operator_indices[id(sub)])
    clip_adapter_index = int(operator_indices[id(clip_adapter.operator)])
    aux_adapter_index = int(operator_indices[id(aux_adapter.operator)])
    if not (
        clip_adapter_index < min(gate_index, sub_index)
        and sub_index < signal_mul_index < add_index < clip3_unary_index
        and gate_index < add_index
        and clip3_unary_index < clip3_adapter_index < concat_index
        and aux_adapter_index < signal_mul_index
    ):
        return None
    if signal is not None:
        signal_index = int(operator_indices[id(signal)])
        if not aux_adapter_index < signal_index < signal_mul_index:
            return None

    clip_consumers = {gate_index, sub_index}
    remaining_clip_consumers = (
        _consumer_set(
            graph_index,
            clip_adapter.output_name,
        )
        - clip_consumers
    )
    rgb_mul = None
    input_transpose = None
    rgb_other_name = None
    rgb_other_view = None
    rgb_other_update = None
    if len(remaining_clip_consumers) > 1:
        return None
    if len(remaining_clip_consumers) == 1:
        rgb_index = next(iter(remaining_clip_consumers))
        rgb_mul = model_ir.operators[rgb_index]
        if (
            not _plain_binary(rgb_mul, "MUL")
            or list(rgb_mul.inputs).count(clip_adapter.output_name) != 1
        ):
            return None
        rgb_other_name = next(
            str(value)
            for value in rgb_mul.inputs
            if str(value) != clip_adapter.output_name
        )
        input_transpose = _producer(model_ir, graph_index, rgb_other_name)
        transpose_result = None
        if input_transpose is not None:
            transpose_result = _typed_input_transpose(
                model_ir,
                graph_index,
                input_transpose,
                consumer=rgb_mul,
            )
        if transpose_result is not None:
            rgb_other_name, rgb_other_view = transpose_result
        else:
            input_transpose = None
            tensor = model_ir.tensors.get(rgb_other_name)
            rgb_index_value = _operator_index(graph_index, rgb_mul)
            if tensor is None or rgb_index_value is None:
                return None
            current_view = _resolve_external_view(
                model_ir,
                graph_index,
                name=rgb_other_name,
                before_index=rgb_index_value,
                dtype=clip_adapter.input_view.dtype,
            )
            if (
                current_view is not None
                and len(current_view.shape) == 4
                and _layout_of(rgb_other_name, tensor, layout_state)
                != LOGICAL_LAYOUT_NCHW
                and _broadcast(clip_adapter.input_view, current_view) is not None
            ):
                rgb_other_view = current_view
            else:
                expected_view = _permuted_view(_view(tensor), _PERM_NCHW_TO_NHWC)
                if expected_view is None:
                    return None
                rgb_other_update = _resolve_stale_nchw_source(
                    model_ir,
                    graph_index,
                    name=rgb_other_name,
                    before_index=rgb_index_value,
                    expected=expected_view,
                    layout_state=layout_state,
                )
                if rgb_other_update is None:
                    return None
                rgb_other_view = expected_view

    exact_consumers = [
        (str(gate_mul.outputs[0]), {add_index}),
        (str(sub.outputs[0]), {signal_mul_index}),
        (str(signal_mul.outputs[0]), {add_index}),
        (str(add.outputs[0]), {clip3_unary_index}),
        (str(clip3_unary.outputs[0]), {clip3_adapter_index}),
        (
            aux_adapter.output_name,
            {signal_mul_index if signal is None else int(operator_indices[id(signal)])},
        ),
    ]
    if signal is not None:
        exact_consumers.append((str(signal.outputs[0]), {signal_mul_index}))
    if any(
        _consumer_set(graph_index, name) != expected
        for name, expected in exact_consumers
    ):
        return None
    split_source_expected_consumers = {gate_index}
    if split_adapter is None:
        split_source_expected_consumers.add(concat_index)
    else:
        split_adapter_index = int(operator_indices[id(split_adapter.operator)])
        split_source_expected_consumers.add(split_adapter_index)
    if _consumer_set(graph_index, split_name) != split_source_expected_consumers:
        return None

    protected = {str(value) for value in model_ir.inputs + model_ir.outputs}
    changed_names = {
        split_name,
        str(gate_mul.outputs[0]),
        str(sub.outputs[0]),
        str(signal_mul.outputs[0]),
        str(add.outputs[0]),
        str(clip3_unary.outputs[0]),
    }
    if signal is not None:
        changed_names.add(str(signal.outputs[0]))
    if rgb_mul is not None:
        changed_names.add(str(rgb_mul.outputs[0]))
    if changed_names & protected:
        return None

    split_target_view = (
        split_adapter.output_view
        if split_adapter is not None
        else _permuted_view(_view(model_ir.tensors[split_name]), _PERM_NCHW_TO_NHWC)
    )
    if split_target_view is None:
        return None
    split_update = _resolve_stale_nchw_source(
        model_ir,
        graph_index,
        name=split_name,
        before_index=gate_index,
        expected=split_target_view,
        layout_state=layout_state,
    )
    if split_update is None:
        return None

    views: Dict[str, _View] = {
        clip_adapter.output_name: clip_adapter.input_view,
        split_name: split_target_view,
    }
    scalar_name = str(sub.inputs[0])
    scalar_view = _resolve_external_view(
        model_ir,
        graph_index,
        name=scalar_name,
        before_index=sub_index,
        dtype=clip_adapter.input_view.dtype,
    )
    if scalar_view is None:
        return None
    views[scalar_name] = scalar_view
    gate_result = _metadata_for_binary(
        model_ir,
        graph_index,
        gate_mul,
        op_type="MUL",
        views=views,
    )
    sub_result = _metadata_for_binary(
        model_ir,
        graph_index,
        sub,
        op_type="SUB",
        views=views,
    )
    if gate_result is None or sub_result is None:
        return None
    gate_update, gate_view = gate_result
    sub_update, sub_view = sub_result
    views[str(gate_mul.outputs[0])] = gate_view
    views[str(sub.outputs[0])] = sub_view

    metadata_updates = [split_update, gate_update, sub_update]
    signal_view = aux_adapter.input_view
    if signal is not None:
        signal_result = _metadata_for_unary(
            model_ir,
            graph_index,
            signal,
            input_view=aux_adapter.input_view,
        )
        if signal_result is None:
            return None
        signal_update, signal_view = signal_result
        metadata_updates.append(signal_update)
    signal_name = aux_adapter.output_name if signal is None else str(signal.outputs[0])
    views[signal_name] = signal_view
    signal_mul_result = _metadata_for_binary(
        model_ir,
        graph_index,
        signal_mul,
        op_type="MUL",
        views=views,
    )
    if signal_mul_result is None:
        return None
    signal_mul_update, signal_mul_view = signal_mul_result
    metadata_updates.append(signal_mul_update)
    views[str(signal_mul.outputs[0])] = signal_mul_view
    add_result = _metadata_for_binary(
        model_ir,
        graph_index,
        add,
        op_type="ADD",
        views=views,
    )
    if add_result is None:
        return None
    add_update, add_view = add_result
    metadata_updates.append(add_update)
    unary_result = _metadata_for_unary(
        model_ir,
        graph_index,
        clip3_unary,
        input_view=add_view,
    )
    if unary_result is None:
        return None
    unary_update, unary_view = unary_result
    if (
        unary_view.shape != clip3_adapter.output_view.shape
        or not _compatible(unary_view.signature, clip3_adapter.output_view.signature)
        or unary_view.dtype != clip3_adapter.output_view.dtype
    ):
        return None
    metadata_updates.extend([unary_update])

    if rgb_mul is not None:
        if rgb_other_name is None or rgb_other_view is None:
            return None
        views[clip_adapter.output_name] = clip_adapter.input_view
        original_rgb_other = next(
            str(value)
            for value in rgb_mul.inputs
            if str(value) != clip_adapter.output_name
        )
        views[original_rgb_other] = rgb_other_view
        rgb_result = _metadata_for_binary(
            model_ir,
            graph_index,
            rgb_mul,
            op_type="MUL",
            views=views,
        )
        if rgb_result is None:
            return None
        rgb_update, _ = rgb_result
        metadata_updates.append(rgb_update)
        if rgb_other_update is not None:
            metadata_updates.append(rgb_other_update)

    replacement_pairs = [
        (gate_mul, clip_adapter.output_name, clip_adapter.input_name),
        (sub, clip_adapter.output_name, clip_adapter.input_name),
    ]
    if signal is None:
        replacement_pairs.append(
            (signal_mul, aux_adapter.output_name, aux_adapter.input_name)
        )
    else:
        replacement_pairs.append(
            (signal, aux_adapter.output_name, aux_adapter.input_name)
        )
    if rgb_mul is not None:
        replacement_pairs.append(
            (rgb_mul, clip_adapter.output_name, clip_adapter.input_name)
        )
        if input_transpose is not None and rgb_other_name is not None:
            replacement_pairs.append(
                (rgb_mul, str(input_transpose.outputs[0]), rgb_other_name)
            )
    if split_adapter is not None:
        for operator in graph_index.consumers_of(split_adapter.output_name):
            replacement_pairs.append(
                (operator, split_adapter.output_name, split_adapter.input_name)
            )
    for operator in graph_index.consumers_of(clip3_adapter.output_name):
        replacement_pairs.append(
            (operator, clip3_adapter.output_name, clip3_adapter.input_name)
        )
    rewrites = _make_rewrites(graph_index, replacement_pairs)
    if rewrites is None:
        return None

    planned_concat_inputs = next(
        rewrite.new_inputs for rewrite in rewrites if rewrite.operator is concat
    )
    concat_views = []
    for name in planned_concat_inputs:
        if name == split_name:
            concat_views.append(split_target_view)
        elif name == clip3_adapter.input_name:
            concat_views.append(unary_view)
        else:
            tensor = model_ir.tensors.get(name)
            view = _resolve_external_view(
                model_ir,
                graph_index,
                name=name,
                before_index=concat_index,
                dtype=unary_view.dtype,
            )
            if (
                tensor is None
                or view is None
                or len(view.shape) != 4
                or _layout_of(name, tensor, layout_state) == LOGICAL_LAYOUT_NCHW
            ):
                return None
            concat_views.append(view)
    expected_concat = _concat_view(concat_views)
    concat_output_name = str(concat.outputs[0])
    concat_tensor = model_ir.tensors.get(concat_output_name)
    if (
        expected_concat is None
        or concat_tensor is None
        or concat_tensor.data is not None
        or bool(concat_tensor.is_variable)
        or concat_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(concat_output_name) != concat_index
        or _view(concat_tensor).shape != expected_concat.shape
        or not _compatible(_view(concat_tensor).signature, expected_concat.signature)
        or str(concat_tensor.dtype) != expected_concat.dtype
        or not _all_per_tensor_quantized([concat_tensor])
    ):
        return None

    removals = [clip_adapter.operator, aux_adapter.operator, clip3_adapter.operator]
    if split_adapter is not None:
        removals.append(split_adapter.operator)
    if input_transpose is not None:
        removals.append(input_transpose)
    if len({id(value) for value in removals}) != len(removals):
        return None
    removals.sort(key=lambda value: int(_operator_index(graph_index, value) or 0))

    relevant_operators = {
        id(operator): operator
        for operator in (
            concat,
            clip3_adapter.operator,
            clip3_unary,
            add,
            gate_mul,
            signal_mul,
            sub,
            clip_adapter.operator,
            aux_adapter.operator,
            *(() if signal is None else (signal,)),
            *(() if split_adapter is None else (split_adapter.operator,)),
            *(() if rgb_mul is None else (rgb_mul,)),
            *(() if input_transpose is None else (input_transpose,)),
            *(rewrite.operator for rewrite in rewrites),
        )
    }
    ordered_operators = sorted(
        relevant_operators.values(),
        key=lambda value: int(_operator_index(graph_index, value) or 0),
    )
    relevant_tensor_names = set()
    for operator in ordered_operators:
        relevant_tensor_names.update(str(value) for value in operator.inputs)
        relevant_tensor_names.update(str(value) for value in operator.outputs)
    tensor_contracts = []
    for name in sorted(relevant_tensor_names):
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))
    return _Plan(
        concat=concat,
        clip3_adapter=clip3_adapter,
        clip3_unary=clip3_unary,
        add=add,
        gate_mul=gate_mul,
        signal_mul=signal_mul,
        sub=sub,
        clip_adapter=clip_adapter,
        aux_adapter=aux_adapter,
        signal=signal,
        split_adapter=split_adapter,
        rgb_mul=rgb_mul,
        input_transpose=input_transpose,
        rewrites=rewrites,
        metadata_updates=tuple(metadata_updates),
        removals=tuple(removals),
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in ordered_operators
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
        plan.concat,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if any(
        _operator_index(graph_index, operator) is None for operator in plan.removals
    ):
        return False
    for rewrite in plan.rewrites:
        if (
            tuple(str(value) for value in rewrite.operator.inputs)
            != rewrite.original_inputs
        ):
            return False
    for update in plan.metadata_updates:
        if update.name not in model_ir.tensors:
            return False

    for rewrite in plan.rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    for update in plan.metadata_updates:
        tensor = model_ir.tensors[update.name]
        if tensor.data is not None:
            array = np.asarray(tensor.data)
            if int(array.size) != int(np.prod(update.shape, dtype=np.int64)):
                raise RuntimeError("validated singleton-gate constant size changed")
            tensor.data = array.reshape(update.shape)
        tensor.shape = [int(value) for value in update.shape]
        tensor.shape_signature = [int(value) for value in update.signature]
        tensor.logical_layout = str(update.logical_layout)
        tensor.physical_layout = str(update.physical_layout)
        if layout_state is not None:
            layout_state.set(
                update.name,
                logical=update.logical_layout,
                physical=update.physical_layout,
            )
    removal_indices = [
        int(_operator_index(graph_index, operator)) for operator in plan.removals
    ]
    for operator_index in sorted(removal_indices, reverse=True):
        graph_index.remove_operator(operator_index)
    return True


def optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Remove one fully owned singleton gate island transactionally."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    rewrite_limit = max(0, int(max_rewrites))
    if rewrite_limit == 0:
        return {_STATS_KEY: 0}
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
    rewritten = 0
    for concat in candidates:
        if rewritten >= rewrite_limit:
            break
        if concat is None or _operator_index(active_index, concat) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            concat,
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
