from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _broadcast_shape_signatures,
    _broadcast_static_shapes,
    _clone_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
)


_BINARY_OPS = frozenset({"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM"})
_UNARY_OPS = frozenset(
    {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "LOGISTIC",
        "HARD_SWISH",
        "LEAKY_RELU",
        "TANH",
        "NEG",
        "EXP",
    }
)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_STATS_KEY = "optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw"
_DIRECT_STATS_KEY = "optimized_transpose_split_channelwise_tail_to_single_post_nchw"


@dataclass(frozen=True)
class _TensorView:
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _MetadataUpdate:
    name: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    logical_layout: str
    physical_layout: str


@dataclass(frozen=True)
class _AxisUpdate:
    operator: OperatorIR
    axis_name: str
    clone_name: Optional[str]
    dtype: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    numpy_dtype: str
    data_shape: Tuple[int, ...]


@dataclass(frozen=True)
class _ConcatUpdate:
    operator: OperatorIR
    original_axis: int
    new_axis: int


@dataclass(frozen=True)
class _InputUse:
    operator: OperatorIR
    input_slot: int


@dataclass(frozen=True)
class _SliceIntent:
    operator: OperatorIR
    update: _MetadataUpdate
    begin_name: str
    begin_values: Tuple[int, ...]
    size_name: str
    size_values: Tuple[int, ...]


@dataclass(frozen=True)
class _ConstantUpdate:
    tensor_name: str
    clone_name: Optional[str]
    uses: Tuple[_InputUse, ...]
    values: Tuple[int, ...]
    dtype: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    numpy_dtype: str
    data_shape: Tuple[int, ...]


@dataclass(frozen=True)
class _OutputAdapter:
    producer: OperatorIR
    output_slot: int
    public_name: str
    private_name: str
    private_shape: Tuple[int, ...]
    private_signature: Tuple[int, ...]
    permutation_name: str


@dataclass(frozen=True)
class _BinarySplitTailPlan:
    pre: OperatorIR
    first: OperatorIR
    second: OperatorIR
    root_split: OperatorIR
    pre_input_slot: int
    pre_input_name: str
    pre_output_name: str
    metadata_updates: Tuple[_MetadataUpdate, ...]
    axis_updates: Tuple[_AxisUpdate, ...]
    concat_updates: Tuple[_ConcatUpdate, ...]
    constant_updates: Tuple[_ConstantUpdate, ...]
    closure_operators: Tuple[OperatorIR, ...]
    adapter: _OutputAdapter
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class _DirectSplitTailPlan:
    pre: OperatorIR
    root_split: OperatorIR
    pre_input_name: str
    pre_output_name: str
    tail: _ClosedTailPlan
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class _PreTransposeRoot:
    operator_index: int
    permutation_name: str
    input_name: str
    output_name: str
    source_view: _TensorView


def _normalized_op_type(operator: OperatorIR) -> str:
    return str(operator.op_type).upper()


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (
            str(value.dtype),
            tuple(int(item) for item in value.shape),
            tuple(value.reshape(-1).tolist()),
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _freeze_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if hasattr(value, "scale") and hasattr(value, "zero_point"):
        return (
            tuple(float(item) for item in value.scale),
            tuple(int(item) for item in value.zero_point),
            int(value.quantized_dimension),
            _freeze_value(value.min),
            _freeze_value(value.max),
        )
    return value


def _tensor_contract(name: str, tensor: TensorIR) -> Tuple[Any, ...]:
    return (
        str(name),
        id(tensor),
        str(tensor.name),
        str(tensor.dtype),
        tuple(int(value) for value in tensor.shape),
        tuple(
            int(value)
            for value in (
                tensor.shape
                if tensor.shape_signature is None
                else tensor.shape_signature
            )
        ),
        _freeze_value(tensor.data),
        bool(tensor.is_variable),
        _freeze_value(tensor.quantization),
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
        _freeze_value(operator.options),
        _freeze_value(operator.axis_semantics),
        int(operator.version),
        operator.onnx_node_name,
        operator.onnx_op_type,
    )


def _unique_name(
    model_ir: ModelIR,
    reserved_names: set[str],
    base: str,
) -> str:
    candidate = str(base)
    serial = 0
    while candidate in model_ir.tensors or candidate in reserved_names:
        serial += 1
        candidate = f"{base}_{serial}"
    reserved_names.add(candidate)
    return candidate


def _operator_index(
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
) -> Optional[int]:
    return graph_index.operator_index(operator)


def _resolved_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    before_index: int,
) -> bool:
    normalized_name = str(name)
    tensor = model_ir.tensors.get(normalized_name)
    if tensor is None or normalized_name in graph_index.duplicate_producers:
        return False
    producer_index = graph_index.producers.get(normalized_name)
    if producer_index is not None:
        return int(producer_index) < int(before_index)
    return bool(
        normalized_name in {str(value) for value in model_ir.inputs}
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


def _typed_nhwc_to_nchw_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    public_names: set[str],
) -> Optional[str]:
    if (
        _normalized_op_type(operator) != "TRANSPOSE"
        or len(operator.inputs) != 2
        or len(operator.outputs) != 1
    ):
        return None
    permutation_name = str(operator.inputs[1])
    tensor = model_ir.tensors.get(permutation_name)
    if (
        tensor is None
        or tensor.data is None
        or permutation_name in public_names
        or permutation_name in graph_index.producers
        or permutation_name in graph_index.duplicate_producers
        or bool(tensor.is_variable)
        or tensor.quantization is not None
        or str(tensor.dtype).upper() not in {"INT32", "INT64"}
    ):
        return None
    try:
        array = np.asarray(tensor.data)
    except Exception:
        return None
    if array.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        return None
    values = tuple(int(value) for value in array.reshape(-1).tolist())
    if values != _PERM_NHWC_TO_NCHW:
        return None
    return permutation_name


def _resolve_pre_transpose_root(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_PreTransposeRoot]:
    pre_index = _operator_index(graph_index, pre)
    public_names = {
        str(value) for value in tuple(model_ir.inputs) + tuple(model_ir.outputs)
    }
    permutation_name = _typed_nhwc_to_nchw_permutation(
        model_ir,
        graph_index,
        pre,
        public_names=public_names,
    )
    if pre_index is None or permutation_name is None:
        return None
    input_name = str(pre.inputs[0])
    output_name = str(pre.outputs[0])
    if (
        input_name in {str(value) for value in model_ir.outputs}
        or output_name in public_names
        or output_name in graph_index.duplicate_producers
        or graph_index.producers.get(output_name) != pre_index
        or not _resolved_source(
            model_ir,
            graph_index,
            name=input_name,
            before_index=pre_index,
        )
    ):
        return None
    source_tensor = model_ir.tensors.get(input_name)
    transposed_tensor = model_ir.tensors.get(output_name)
    if (
        source_tensor is None
        or transposed_tensor is None
        or source_tensor.data is not None
        or transposed_tensor.data is not None
        or bool(source_tensor.is_variable)
        or bool(transposed_tensor.is_variable)
        or len(source_tensor.shape) != 4
        or len(transposed_tensor.shape) != 4
        or str(source_tensor.dtype) != str(transposed_tensor.dtype)
        or not _all_per_tensor_quantized([source_tensor, transposed_tensor])
    ):
        return None
    source_signature = (
        list(source_tensor.shape)
        if source_tensor.shape_signature is None
        else list(source_tensor.shape_signature)
    )
    transposed_signature = (
        list(transposed_tensor.shape)
        if transposed_tensor.shape_signature is None
        else list(transposed_tensor.shape_signature)
    )
    expected_shape = _permute_shape(
        list(source_tensor.shape),
        list(_PERM_NHWC_TO_NCHW),
    )
    expected_signature = _permute_shape(
        source_signature,
        list(_PERM_NHWC_TO_NCHW),
    )
    if (
        expected_shape is None
        or expected_signature is None
        or tuple(expected_shape) != tuple(transposed_tensor.shape)
        or not _signature_compatible(expected_signature, transposed_signature)
        or _layout_of(input_name, source_tensor, layout_state) == LOGICAL_LAYOUT_NCHW
    ):
        return None
    return _PreTransposeRoot(
        operator_index=int(pre_index),
        permutation_name=permutation_name,
        input_name=input_name,
        output_name=output_name,
        source_view=_tensor_view(source_tensor),
    )


def _plain_binary(operator: OperatorIR) -> bool:
    return bool(
        _normalized_op_type(operator) in _BINARY_OPS
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
        and str(operator.options.get("fusedActivationFunction", "NONE")).upper()
        in {"", "NONE"}
    )


def _tensor_view(tensor: TensorIR) -> _TensorView:
    return _TensorView(
        shape=tuple(int(value) for value in tensor.shape),
        signature=tuple(
            int(value)
            for value in (
                tensor.shape
                if tensor.shape_signature is None
                else tensor.shape_signature
            )
        ),
        dtype=str(tensor.dtype),
    )


def _signature_compatible(
    expected: Sequence[int],
    actual: Sequence[int],
) -> bool:
    if len(expected) != len(actual):
        return False
    return all(
        int(left) == int(right) or int(left) < 0 or int(right) < 0
        for left, right in zip(expected, actual)
    )


def _permuted_update(
    model_ir: ModelIR,
    name: str,
) -> Optional[_MetadataUpdate]:
    tensor = model_ir.tensors.get(str(name))
    if tensor is None or tensor.data is not None or len(tensor.shape) != 4:
        return None
    signature = (
        list(tensor.shape)
        if tensor.shape_signature is None
        else list(tensor.shape_signature)
    )
    if len(signature) != 4:
        return None
    shape = _permute_shape(list(tensor.shape), list(_PERM_NCHW_TO_NHWC))
    permuted_signature = _permute_shape(
        signature,
        list(_PERM_NCHW_TO_NHWC),
    )
    if shape is None or permuted_signature is None:
        return None
    return _MetadataUpdate(
        name=str(name),
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in permuted_signature),
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
    )


def _view_for_update(
    model_ir: ModelIR,
    update: _MetadataUpdate,
) -> _TensorView:
    return _TensorView(
        shape=update.shape,
        signature=update.signature,
        dtype=str(model_ir.tensors[update.name].dtype),
    )


def _layout_of(
    tensor_name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
) -> str:
    if layout_state is not None:
        return str(layout_state.physical_of(str(tensor_name))).upper()
    return str(tensor.physical_layout).upper()


def _shape_is_nhwc_relative(
    view: _TensorView,
    anchor: _TensorView,
) -> bool:
    if len(view.shape) > 4 or len(anchor.shape) != 4:
        return False
    padded_shape = (1,) * (4 - len(view.shape)) + tuple(view.shape)
    padded_signature = (1,) * (4 - len(view.signature)) + tuple(view.signature)
    for candidate, reference in zip(padded_shape, anchor.shape):
        if int(candidate) not in {1, int(reference)}:
            return False
    for candidate, reference in zip(padded_signature, anchor.signature):
        if (
            int(candidate) >= 0
            and int(reference) >= 0
            and int(candidate)
            not in {
                1,
                int(reference),
            }
        ):
            return False
    return True


def _external_view(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    before_index: int,
    expected_dtype: str,
    anchor: _TensorView,
    layout_state: Optional[LayoutState],
) -> Optional[_TensorView]:
    tensor = model_ir.tensors.get(str(name))
    if (
        tensor is None
        or bool(tensor.is_variable)
        or not _resolved_source(
            model_ir,
            graph_index,
            name=str(name),
            before_index=before_index,
        )
        or str(tensor.dtype) != str(expected_dtype)
        or not _all_per_tensor_quantized([tensor])
    ):
        return None
    view = _tensor_view(tensor)
    if len(view.shape) == 0 or len(view.shape) > 4:
        return None
    physical_layout = _layout_of(str(name), tensor, layout_state)
    if physical_layout == LOGICAL_LAYOUT_NCHW:
        return None
    if physical_layout not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}:
        return None
    if physical_layout == LOGICAL_LAYOUT_UNKNOWN and not _shape_is_nhwc_relative(
        view,
        anchor,
    ):
        return None
    return view


def _broadcast_views(
    left: _TensorView,
    right: _TensorView,
) -> Optional[_TensorView]:
    if left.dtype != right.dtype:
        return None
    shape = _broadcast_static_shapes(list(left.shape), list(right.shape))
    signature = _broadcast_shape_signatures(
        list(left.signature),
        list(right.signature),
    )
    if shape is None or signature is None:
        return None
    return _TensorView(
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        dtype=left.dtype,
    )


def _resolve_binary_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    nhwc_views: Dict[str, _TensorView],
    converted_names: set[str],
    anchor: _TensorView,
    layout_state: Optional[LayoutState],
) -> Optional[_MetadataUpdate]:
    operator_index = _operator_index(graph_index, operator)
    if (
        operator_index is None
        or not _plain_binary(operator)
        or not _inputs_resolved_before(
            model_ir,
            graph_index,
            operator,
            operator_index,
        )
    ):
        return None
    input_names = tuple(str(value) for value in operator.inputs)
    if not any(name in converted_names for name in input_names):
        return None
    output_name = str(operator.outputs[0])
    if (
        output_name in nhwc_views
        or output_name in graph_index.duplicate_producers
        or graph_index.producers.get(output_name) != operator_index
    ):
        return None
    output_tensor = model_ir.tensors.get(output_name)
    if output_tensor is None or not _all_per_tensor_quantized([output_tensor]):
        return None
    views = []
    for name in input_names:
        view = nhwc_views.get(name)
        if view is None:
            view = _external_view(
                model_ir,
                graph_index,
                name=name,
                before_index=operator_index,
                expected_dtype=str(output_tensor.dtype),
                anchor=anchor,
                layout_state=layout_state,
            )
        if view is None:
            return None
        views.append(view)
    broadcast = _broadcast_views(views[0], views[1])
    update = _permuted_update(model_ir, output_name)
    if (
        broadcast is None
        or update is None
        or str(output_tensor.dtype) != broadcast.dtype
        or update.shape != broadcast.shape
        or not _signature_compatible(update.signature, broadcast.signature)
    ):
        return None
    return update


def _resolve_unary_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    nhwc_views: Dict[str, _TensorView],
    converted_names: set[str],
) -> Optional[_MetadataUpdate]:
    operator_index = _operator_index(graph_index, operator)
    if (
        operator_index is None
        or _normalized_op_type(operator) not in _UNARY_OPS
        or len(operator.inputs) != 1
        or len(operator.outputs) != 1
    ):
        return None
    input_name = str(operator.inputs[0])
    output_name = str(operator.outputs[0])
    if input_name not in converted_names or output_name in nhwc_views:
        return None
    if (
        graph_index.producers.get(output_name) != operator_index
        or output_name in graph_index.duplicate_producers
        or not _resolved_source(
            model_ir,
            graph_index,
            name=input_name,
            before_index=operator_index,
        )
    ):
        return None
    input_tensor = model_ir.tensors.get(input_name)
    output_tensor = model_ir.tensors.get(output_name)
    update = _permuted_update(model_ir, output_name)
    if (
        input_tensor is None
        or output_tensor is None
        or update is None
        or str(input_tensor.dtype) != str(output_tensor.dtype)
        or not _all_per_tensor_quantized([input_tensor, output_tensor])
    ):
        return None
    input_view = nhwc_views[input_name]
    if update.shape != input_view.shape or not _signature_compatible(
        update.signature, input_view.signature
    ):
        return None
    return update


def _concat_view(views: Sequence[_TensorView]) -> Optional[_TensorView]:
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
            if not _signature_compatible(
                [signature[axis]],
                [view.signature[axis]],
            ):
                return None
            if int(signature[axis]) < 0 or int(view.signature[axis]) < 0:
                signature[axis] = -1
        shape[3] += int(view.shape[3])
        if int(signature[3]) < 0 or int(view.signature[3]) < 0:
            signature[3] = -1
        else:
            signature[3] += int(view.signature[3])
    return _TensorView(
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        dtype=views[0].dtype,
    )


def _resolve_concat_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    nhwc_views: Dict[str, _TensorView],
    converted_names: set[str],
) -> Optional[Tuple[_MetadataUpdate, _ConcatUpdate]]:
    operator_index = _operator_index(graph_index, operator)
    if (
        operator_index is None
        or _normalized_op_type(operator) != "CONCATENATION"
        or len(operator.inputs) == 0
        or len(operator.outputs) != 1
        or not all(str(name) in converted_names for name in operator.inputs)
        or not _inputs_resolved_before(
            model_ir,
            graph_index,
            operator,
            operator_index,
        )
    ):
        return None
    try:
        axis = int(operator.options.get("axis", -1))
    except Exception:
        return None
    if axis not in {1, -3}:
        return None
    output_name = str(operator.outputs[0])
    if (
        output_name in nhwc_views
        or output_name in graph_index.duplicate_producers
        or graph_index.producers.get(output_name) != operator_index
    ):
        return None
    tensors = [model_ir.tensors.get(str(name)) for name in operator.inputs]
    output_tensor = model_ir.tensors.get(output_name)
    if (
        output_tensor is None
        or any(tensor is None for tensor in tensors)
        or not _all_per_tensor_quantized([*tensors, output_tensor])
    ):
        return None
    view = _concat_view([nhwc_views[str(name)] for name in operator.inputs])
    update = _permuted_update(model_ir, output_name)
    if (
        view is None
        or update is None
        or str(output_tensor.dtype) != view.dtype
        or update.shape != view.shape
        or not _signature_compatible(update.signature, view.signature)
    ):
        return None
    return update, _ConcatUpdate(
        operator=operator,
        original_axis=axis,
        new_axis=3,
    )


def _resolve_axis_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    reserved_names: set[str],
) -> Optional[_AxisUpdate]:
    operator_index = _operator_index(graph_index, operator)
    if operator_index is None or len(operator.inputs) != 2:
        return None
    axis_name = str(operator.inputs[0])
    tensor = model_ir.tensors.get(axis_name)
    if (
        tensor is None
        or tensor.data is None
        or bool(tensor.is_variable)
        or tensor.quantization is not None
        or str(tensor.dtype).upper() not in {"INT32", "INT64"}
        or axis_name in graph_index.producers
        or axis_name in graph_index.duplicate_producers
        or axis_name in {str(value) for value in model_ir.inputs + model_ir.outputs}
    ):
        return None
    try:
        array = np.asarray(tensor.data)
    except Exception:
        return None
    if (
        array.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}
        or array.size != 1
        or int(array.reshape(-1)[0]) != 1
    ):
        return None
    shape = tuple(int(value) for value in tensor.shape)
    signature = tuple(
        int(value)
        for value in (
            tensor.shape if tensor.shape_signature is None else tensor.shape_signature
        )
    )
    if len(shape) != len(signature) or int(np.prod(shape, dtype=np.int64)) != 1:
        return None
    clone_name = None
    if graph_index.consumer_indices(axis_name) != [operator_index]:
        clone_name = _unique_name(
            model_ir,
            reserved_names,
            f"{axis_name}_nhwc",
        )
    return _AxisUpdate(
        operator=operator,
        axis_name=axis_name,
        clone_name=clone_name,
        dtype=str(tensor.dtype),
        shape=shape,
        signature=signature,
        numpy_dtype=str(array.dtype),
        data_shape=tuple(int(value) for value in array.shape),
    )


def _resolve_split_updates(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    nhwc_views: Dict[str, _TensorView],
    converted_names: set[str],
    reserved_names: set[str],
) -> Optional[Tuple[_AxisUpdate, Tuple[_MetadataUpdate, ...]]]:
    operator_index = _operator_index(graph_index, operator)
    if (
        operator_index is None
        or _normalized_op_type(operator) != "SPLIT"
        or len(operator.inputs) != 2
        or len(operator.outputs) == 0
        or not _inputs_resolved_before(
            model_ir,
            graph_index,
            operator,
            operator_index,
        )
    ):
        return None
    input_name = str(operator.inputs[1])
    if input_name not in converted_names:
        return None
    axis_update = _resolve_axis_update(
        model_ir,
        graph_index,
        operator,
        reserved_names=reserved_names,
    )
    if axis_update is None:
        return None
    num_splits = operator.options.get("numSplits")
    if num_splits is not None:
        try:
            if int(num_splits) != len(operator.outputs):
                return None
        except Exception:
            return None
    input_view = nhwc_views[input_name]
    input_tensor = model_ir.tensors.get(input_name)
    updates = []
    for output_name_value in operator.outputs:
        output_name = str(output_name_value)
        if (
            output_name in nhwc_views
            or output_name in graph_index.duplicate_producers
            or graph_index.producers.get(output_name) != operator_index
        ):
            return None
        output_tensor = model_ir.tensors.get(output_name)
        update = _permuted_update(model_ir, output_name)
        if (
            input_tensor is None
            or output_tensor is None
            or update is None
            or str(output_tensor.dtype) != input_view.dtype
            or not _all_per_tensor_quantized([input_tensor, output_tensor])
            or len(update.shape) != 4
        ):
            return None
        for axis in range(3):
            if int(update.shape[axis]) != int(input_view.shape[axis]):
                return None
            if not _signature_compatible(
                [update.signature[axis]],
                [input_view.signature[axis]],
            ):
                return None
        updates.append(update)
    output_channels = [int(update.shape[3]) for update in updates]
    if len(set(output_channels)) != 1 or sum(output_channels) != int(
        input_view.shape[3]
    ):
        return None
    known_signature_channels = [int(update.signature[3]) for update in updates]
    if int(input_view.signature[3]) >= 0 and all(
        value >= 0 for value in known_signature_channels
    ):
        if sum(known_signature_channels) != int(input_view.signature[3]):
            return None
    return axis_update, tuple(updates)


def _integer_vector_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    length: int,
) -> Optional[Tuple[TensorIR, np.ndarray, Tuple[int, ...]]]:
    normalized_name = str(name)
    tensor = model_ir.tensors.get(normalized_name)
    if (
        tensor is None
        or tensor.data is None
        or bool(tensor.is_variable)
        or tensor.quantization is not None
        or str(tensor.dtype).upper() not in {"INT32", "INT64"}
        or normalized_name in graph_index.producers
        or normalized_name in graph_index.duplicate_producers
        or normalized_name
        in {str(value) for value in model_ir.inputs + model_ir.outputs}
    ):
        return None
    try:
        array = np.asarray(tensor.data)
    except Exception:
        return None
    if array.dtype not in {np.dtype(np.int32), np.dtype(np.int64)} or array.size != int(
        length
    ):
        return None
    shape = tuple(int(value) for value in tensor.shape)
    signature = tuple(
        int(value)
        for value in (
            tensor.shape if tensor.shape_signature is None else tensor.shape_signature
        )
    )
    if len(shape) != len(signature) or int(np.prod(shape, dtype=np.int64)) != int(
        length
    ):
        return None
    values = tuple(int(value) for value in array.reshape(-1).tolist())
    return tensor, array, values


def _resolved_slice_view(
    source: _TensorView,
    begin: Sequence[int],
    size: Sequence[int],
) -> Optional[_TensorView]:
    if (
        len(source.shape) != 4
        or len(source.signature) != 4
        or len(begin) != 4
        or len(size) != 4
    ):
        return None
    output_shape = []
    output_signature = []
    for dimension, signature, offset, extent in zip(
        source.shape,
        source.signature,
        begin,
        size,
    ):
        resolved_dimension = int(dimension)
        resolved_signature = int(signature)
        resolved_offset = int(offset)
        resolved_extent = int(extent)
        if (
            resolved_dimension <= 0
            or resolved_offset < 0
            or resolved_offset > resolved_dimension
            or resolved_extent == 0
            or resolved_extent < -1
        ):
            return None
        if resolved_extent == -1:
            output_dimension = resolved_dimension - resolved_offset
            output_dynamic_dimension = (
                -1 if resolved_signature < 0 else resolved_signature - resolved_offset
            )
        else:
            if resolved_offset + resolved_extent > resolved_dimension:
                return None
            output_dimension = resolved_extent
            output_dynamic_dimension = resolved_extent
        if output_dimension <= 0 or output_dynamic_dimension == 0:
            return None
        output_shape.append(int(output_dimension))
        output_signature.append(int(output_dynamic_dimension))
    return _TensorView(
        shape=tuple(output_shape),
        signature=tuple(output_signature),
        dtype=source.dtype,
    )


def _resolve_slice_intent(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    nhwc_views: Dict[str, _TensorView],
    converted_names: set[str],
) -> Optional[_SliceIntent]:
    operator_index = _operator_index(graph_index, operator)
    if (
        operator_index is None
        or _normalized_op_type(operator) != "SLICE"
        or len(operator.inputs) != 3
        or len(operator.outputs) != 1
        or str(operator.inputs[0]) not in converted_names
        or not _inputs_resolved_before(
            model_ir,
            graph_index,
            operator,
            operator_index,
        )
    ):
        return None
    source_name = str(operator.inputs[0])
    begin_name = str(operator.inputs[1])
    size_name = str(operator.inputs[2])
    output_name = str(operator.outputs[0])
    if (
        output_name in nhwc_views
        or output_name in graph_index.duplicate_producers
        or graph_index.producers.get(output_name) != operator_index
    ):
        return None
    begin_result = _integer_vector_constant(
        model_ir,
        graph_index,
        begin_name,
        length=4,
    )
    size_result = _integer_vector_constant(
        model_ir,
        graph_index,
        size_name,
        length=4,
    )
    if begin_result is None or size_result is None:
        return None
    begin_tensor, begin_array, begin_values = begin_result
    size_tensor, size_array, size_values = size_result
    if (
        str(begin_tensor.dtype) != str(size_tensor.dtype)
        or begin_array.dtype != size_array.dtype
    ):
        return None
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(output_name)
    update = _permuted_update(model_ir, output_name)
    if (
        source_tensor is None
        or output_tensor is None
        or update is None
        or str(source_tensor.dtype) != str(output_tensor.dtype)
        or not _all_per_tensor_quantized([source_tensor, output_tensor])
    ):
        return None
    original_view = _tensor_view(source_tensor)
    original_result = _resolved_slice_view(
        original_view,
        begin_values,
        size_values,
    )
    if (
        original_result is None
        or original_result.shape != tuple(int(value) for value in output_tensor.shape)
        or not _signature_compatible(
            original_result.signature,
            output_tensor.shape
            if output_tensor.shape_signature is None
            else output_tensor.shape_signature,
        )
    ):
        return None
    new_begin = tuple(int(begin_values[index]) for index in _PERM_NCHW_TO_NHWC)
    new_size = tuple(int(size_values[index]) for index in _PERM_NCHW_TO_NHWC)
    converted_result = _resolved_slice_view(
        nhwc_views[source_name],
        new_begin,
        new_size,
    )
    if (
        converted_result is None
        or update.shape != converted_result.shape
        or not _signature_compatible(update.signature, converted_result.signature)
    ):
        return None
    return _SliceIntent(
        operator=operator,
        update=update,
        begin_name=begin_name,
        begin_values=new_begin,
        size_name=size_name,
        size_values=new_size,
    )


def _finalize_slice_constant_updates(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    intents: Sequence[_SliceIntent],
    *,
    reserved_names: set[str],
) -> Optional[Tuple[_ConstantUpdate, ...]]:
    grouped: Dict[str, list[Tuple[_InputUse, Tuple[int, ...]]]] = {}
    for intent in intents:
        grouped.setdefault(intent.begin_name, []).append(
            (_InputUse(intent.operator, 1), intent.begin_values)
        )
        grouped.setdefault(intent.size_name, []).append(
            (_InputUse(intent.operator, 2), intent.size_values)
        )
    updates = []
    for tensor_name in sorted(grouped):
        grouped_uses = grouped[tensor_name]
        values = {item[1] for item in grouped_uses}
        if len(values) != 1:
            return None
        tensor = model_ir.tensors.get(tensor_name)
        if tensor is None or tensor.data is None:
            return None
        array = np.asarray(tensor.data)
        planned_uses = tuple(
            sorted(
                (item[0] for item in grouped_uses),
                key=lambda use: (
                    int(_operator_index(graph_index, use.operator) or -1),
                    int(use.input_slot),
                ),
            )
        )
        actual_uses = []
        for operator_index in sorted(set(graph_index.consumer_indices(tensor_name))):
            operator = model_ir.operators[int(operator_index)]
            actual_uses.extend(
                _InputUse(operator, input_slot)
                for input_slot, name in enumerate(operator.inputs)
                if str(name) == tensor_name
            )
        planned_signature = tuple(
            (id(use.operator), int(use.input_slot)) for use in planned_uses
        )
        actual_signature = tuple(
            (id(use.operator), int(use.input_slot)) for use in actual_uses
        )
        clone_name = None
        if actual_signature != planned_signature:
            clone_name = _unique_name(
                model_ir,
                reserved_names,
                f"{tensor_name}_nhwc",
            )
        updates.append(
            _ConstantUpdate(
                tensor_name=tensor_name,
                clone_name=clone_name,
                uses=planned_uses,
                values=next(iter(values)),
                dtype=str(tensor.dtype),
                shape=tuple(int(value) for value in tensor.shape),
                signature=tuple(
                    int(value)
                    for value in (
                        tensor.shape
                        if tensor.shape_signature is None
                        else tensor.shape_signature
                    )
                ),
                numpy_dtype=str(array.dtype),
                data_shape=tuple(int(value) for value in array.shape),
            )
        )
    return tuple(updates)


def _enqueue_consumers(
    graph_index: ModelIRGraphIndex,
    names: Sequence[str],
    queue: deque[OperatorIR],
) -> None:
    for name in names:
        for operator in graph_index.consumers_of(str(name)):
            queue.append(operator)


@dataclass(frozen=True)
class _ClosedTailPlan:
    metadata_updates: Tuple[_MetadataUpdate, ...]
    axis_updates: Tuple[_AxisUpdate, ...]
    concat_updates: Tuple[_ConcatUpdate, ...]
    constant_updates: Tuple[_ConstantUpdate, ...]
    closure_operators: Tuple[OperatorIR, ...]
    adapter: _OutputAdapter


def _resolve_closed_tail(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    source_view: _TensorView,
    permutation_name: str,
    nhwc_views: Dict[str, _TensorView],
    converted_names: set[str],
    initial_updates: Sequence[_MetadataUpdate],
    initial_axis_updates: Sequence[_AxisUpdate],
    accepted_operators: Sequence[OperatorIR],
    seed_names: Sequence[str],
    reserved_names: set[str],
    layout_state: Optional[LayoutState],
    allow_slice: bool = False,
) -> Optional[_ClosedTailPlan]:
    active_views = dict(nhwc_views)
    active_names = set(converted_names)
    metadata_updates = list(initial_updates)
    axis_updates = list(initial_axis_updates)
    concat_updates: list[_ConcatUpdate] = []
    slice_intents: list[_SliceIntent] = []
    closure_operators: list[OperatorIR] = []
    accepted_ids = {id(operator) for operator in accepted_operators}

    queue: deque[OperatorIR] = deque()
    _enqueue_consumers(graph_index, seed_names, queue)
    work_limit = len(model_ir.operators) + sum(
        max(1, len(operator.inputs)) for operator in model_ir.operators
    )
    work_count = 0
    while queue:
        work_count += 1
        if work_count > work_limit:
            return None
        operator = queue.popleft()
        if id(operator) in accepted_ids:
            continue
        if _operator_index(graph_index, operator) is None:
            return None
        result_updates: Tuple[_MetadataUpdate, ...] = ()
        normalized_type = _normalized_op_type(operator)
        if normalized_type in _UNARY_OPS:
            update = _resolve_unary_update(
                model_ir,
                graph_index,
                operator,
                nhwc_views=active_views,
                converted_names=active_names,
            )
            if update is not None:
                result_updates = (update,)
        elif normalized_type in _BINARY_OPS:
            update = _resolve_binary_update(
                model_ir,
                graph_index,
                operator,
                nhwc_views=active_views,
                converted_names=active_names,
                anchor=source_view,
                layout_state=layout_state,
            )
            if update is not None:
                result_updates = (update,)
        elif normalized_type == "CONCATENATION":
            concat_result = _resolve_concat_update(
                model_ir,
                graph_index,
                operator,
                nhwc_views=active_views,
                converted_names=active_names,
            )
            if concat_result is not None:
                result_updates = (concat_result[0],)
                concat_updates.append(concat_result[1])
        elif normalized_type == "SPLIT":
            split_result = _resolve_split_updates(
                model_ir,
                graph_index,
                operator,
                nhwc_views=active_views,
                converted_names=active_names,
                reserved_names=reserved_names,
            )
            if split_result is not None:
                axis_updates.append(split_result[0])
                result_updates = split_result[1]
        elif normalized_type == "SLICE" and allow_slice:
            slice_intent = _resolve_slice_intent(
                model_ir,
                graph_index,
                operator,
                nhwc_views=active_views,
                converted_names=active_names,
            )
            if slice_intent is not None:
                slice_intents.append(slice_intent)
                result_updates = (slice_intent.update,)
        if len(result_updates) == 0:
            continue
        accepted_ids.add(id(operator))
        closure_operators.append(operator)
        metadata_updates.extend(result_updates)
        new_names = []
        for update in result_updates:
            active_views[update.name] = _view_for_update(model_ir, update)
            active_names.add(update.name)
            new_names.append(update.name)
        _enqueue_consumers(graph_index, new_names, queue)

    model_outputs = tuple(str(value) for value in model_ir.outputs)
    if len(model_outputs) != 1 or model_outputs[0] not in active_names:
        return None
    public_output_name = model_outputs[0]
    if public_output_name in {str(value) for value in model_ir.inputs}:
        return None
    for name in active_names:
        users = graph_index.consumers_of(name)
        if name == public_output_name:
            if len(users) != 0:
                return None
        elif len(users) == 0 or any(
            id(operator) not in accepted_ids for operator in users
        ):
            return None

    output_producer_index = graph_index.producers.get(public_output_name)
    if (
        output_producer_index is None
        or public_output_name in graph_index.duplicate_producers
    ):
        return None
    output_producer = model_ir.operators[int(output_producer_index)]
    output_slots = [
        index
        for index, name in enumerate(output_producer.outputs)
        if str(name) == public_output_name
    ]
    if len(output_slots) != 1:
        return None
    public_updates = [
        update for update in metadata_updates if update.name == public_output_name
    ]
    if len(public_updates) != 1:
        return None
    public_update = public_updates[0]
    metadata_updates = [
        update for update in metadata_updates if update.name != public_output_name
    ]
    private_name = _unique_name(
        model_ir,
        reserved_names,
        f"{public_output_name}_nhwc",
    )
    constant_updates = _finalize_slice_constant_updates(
        model_ir,
        graph_index,
        slice_intents,
        reserved_names=reserved_names,
    )
    if constant_updates is None:
        return None
    return _ClosedTailPlan(
        metadata_updates=tuple(metadata_updates),
        axis_updates=tuple(axis_updates),
        concat_updates=tuple(concat_updates),
        constant_updates=constant_updates,
        closure_operators=tuple(closure_operators),
        adapter=_OutputAdapter(
            producer=output_producer,
            output_slot=int(output_slots[0]),
            public_name=public_output_name,
            private_name=private_name,
            private_shape=public_update.shape,
            private_signature=public_update.signature,
            permutation_name=permutation_name,
        ),
    )


def _closed_tail_signature(tail: _ClosedTailPlan) -> Tuple[Any, ...]:
    def axis_signature(update: _AxisUpdate) -> Tuple[Any, ...]:
        return (
            id(update.operator),
            update.axis_name,
            update.clone_name,
            update.dtype,
            update.shape,
            update.signature,
            update.numpy_dtype,
            update.data_shape,
        )

    def concat_signature(update: _ConcatUpdate) -> Tuple[Any, ...]:
        return (
            id(update.operator),
            update.original_axis,
            update.new_axis,
        )

    def constant_signature(update: _ConstantUpdate) -> Tuple[Any, ...]:
        return (
            update.tensor_name,
            update.clone_name,
            tuple((id(use.operator), int(use.input_slot)) for use in update.uses),
            update.values,
            update.dtype,
            update.shape,
            update.signature,
            update.numpy_dtype,
            update.data_shape,
        )

    def adapter_signature(adapter: _OutputAdapter) -> Tuple[Any, ...]:
        return (
            id(adapter.producer),
            adapter.output_slot,
            adapter.public_name,
            adapter.private_name,
            adapter.private_shape,
            adapter.private_signature,
            adapter.permutation_name,
        )

    return (
        tail.metadata_updates,
        tuple(axis_signature(value) for value in tail.axis_updates),
        tuple(concat_signature(value) for value in tail.concat_updates),
        tuple(constant_signature(value) for value in tail.constant_updates),
        tuple(id(value) for value in tail.closure_operators),
        adapter_signature(tail.adapter),
    )


def _binary_tail(plan: _BinarySplitTailPlan) -> _ClosedTailPlan:
    return _ClosedTailPlan(
        metadata_updates=plan.metadata_updates,
        axis_updates=plan.axis_updates,
        concat_updates=plan.concat_updates,
        constant_updates=plan.constant_updates,
        closure_operators=plan.closure_operators,
        adapter=plan.adapter,
    )


def _plans_equal(
    expected: _BinarySplitTailPlan,
    actual: _BinarySplitTailPlan,
) -> bool:
    return bool(
        expected.pre is actual.pre
        and expected.first is actual.first
        and expected.second is actual.second
        and expected.root_split is actual.root_split
        and expected.pre_input_slot == actual.pre_input_slot
        and expected.pre_input_name == actual.pre_input_name
        and expected.pre_output_name == actual.pre_output_name
        and _closed_tail_signature(_binary_tail(expected))
        == _closed_tail_signature(_binary_tail(actual))
        and expected.tensor_contracts == actual.tensor_contracts
        and expected.operator_contracts == actual.operator_contracts
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_BinarySplitTailPlan]:
    pre_root = _resolve_pre_transpose_root(
        model_ir,
        graph_index,
        pre,
        layout_state=layout_state,
    )
    if pre_root is None:
        return None
    pre_index = pre_root.operator_index
    permutation_name = pre_root.permutation_name
    pre_input_name = pre_root.input_name
    pre_output_name = pre_root.output_name
    source_view = pre_root.source_view

    pre_users = graph_index.consumer_indices(pre_output_name)
    if len(pre_users) != 1 or int(pre_users[0]) <= pre_index:
        return None
    first_index = int(pre_users[0])
    first = model_ir.operators[first_index]
    if not _plain_binary(first) or list(first.inputs).count(pre_output_name) != 1:
        return None
    first_output_name = str(first.outputs[0])
    first_users = graph_index.consumer_indices(first_output_name)
    if len(first_users) != 1 or int(first_users[0]) <= first_index:
        return None
    second_index = int(first_users[0])
    second = model_ir.operators[second_index]
    if not _plain_binary(second) or list(second.inputs).count(first_output_name) != 1:
        return None
    second_output_name = str(second.outputs[0])
    second_users = graph_index.consumer_indices(second_output_name)
    if len(second_users) != 1 or int(second_users[0]) <= second_index:
        return None
    split_index = int(second_users[0])
    root_split = model_ir.operators[split_index]
    if (
        _normalized_op_type(root_split) != "SPLIT"
        or len(root_split.inputs) != 2
        or str(root_split.inputs[1]) != second_output_name
        or split_index <= second_index
    ):
        return None

    reserved_names = set(str(name) for name in model_ir.tensors)
    nhwc_views: Dict[str, _TensorView] = {
        pre_input_name: source_view,
        pre_output_name: source_view,
    }
    converted_names = {pre_output_name}
    first_update = _resolve_binary_update(
        model_ir,
        graph_index,
        first,
        nhwc_views=nhwc_views,
        converted_names=converted_names,
        anchor=source_view,
        layout_state=layout_state,
    )
    if first_update is None:
        return None
    first_view = _view_for_update(model_ir, first_update)
    nhwc_views[first_output_name] = first_view
    converted_names.add(first_output_name)
    nhwc_views.pop(pre_output_name, None)
    converted_names.remove(pre_output_name)
    second_update = _resolve_binary_update(
        model_ir,
        graph_index,
        second,
        nhwc_views=nhwc_views,
        converted_names=converted_names,
        anchor=source_view,
        layout_state=layout_state,
    )
    if second_update is None:
        return None
    second_view = _view_for_update(model_ir, second_update)
    nhwc_views[second_output_name] = second_view
    converted_names.add(second_output_name)
    root_split_result = _resolve_split_updates(
        model_ir,
        graph_index,
        root_split,
        nhwc_views=nhwc_views,
        converted_names=converted_names,
        reserved_names=reserved_names,
    )
    if root_split_result is None:
        return None
    root_axis_update, root_output_updates = root_split_result

    for update in root_output_updates:
        nhwc_views[update.name] = _view_for_update(model_ir, update)
        converted_names.add(update.name)
    tail_plan = _resolve_closed_tail(
        model_ir,
        graph_index,
        source_view=source_view,
        permutation_name=permutation_name,
        nhwc_views=nhwc_views,
        converted_names=converted_names,
        initial_updates=(first_update, second_update, *root_output_updates),
        initial_axis_updates=(root_axis_update,),
        accepted_operators=(first, second, root_split),
        seed_names=tuple(update.name for update in root_output_updates),
        reserved_names=reserved_names,
        layout_state=layout_state,
    )
    if tail_plan is None:
        return None
    metadata_updates = list(tail_plan.metadata_updates)
    axis_updates = list(tail_plan.axis_updates)
    concat_updates = list(tail_plan.concat_updates)
    constant_updates = list(tail_plan.constant_updates)
    closure_operators = list(tail_plan.closure_operators)
    adapter = tail_plan.adapter

    relevant_operators = [pre, first, second, root_split, *closure_operators]
    relevant_operators = sorted(
        relevant_operators,
        key=lambda operator: int(_operator_index(graph_index, operator) or 0),
    )
    relevant_tensor_names = set()
    for operator in relevant_operators:
        relevant_tensor_names.update(str(value) for value in operator.inputs)
        relevant_tensor_names.update(str(value) for value in operator.outputs)
    relevant_tensor_names.add(adapter.public_name)
    tensor_contracts = []
    for name in sorted(relevant_tensor_names):
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))

    return _BinarySplitTailPlan(
        pre=pre,
        first=first,
        second=second,
        root_split=root_split,
        pre_input_slot=int(list(first.inputs).index(pre_output_name)),
        pre_input_name=pre_input_name,
        pre_output_name=pre_output_name,
        metadata_updates=tuple(metadata_updates),
        axis_updates=tuple(axis_updates),
        concat_updates=tuple(concat_updates),
        constant_updates=tuple(constant_updates),
        closure_operators=tuple(closure_operators),
        adapter=adapter,
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in relevant_operators
        ),
    )


def _direct_plans_equal(
    expected: _DirectSplitTailPlan,
    actual: _DirectSplitTailPlan,
) -> bool:
    return bool(
        expected.pre is actual.pre
        and expected.root_split is actual.root_split
        and expected.pre_input_name == actual.pre_input_name
        and expected.pre_output_name == actual.pre_output_name
        and _closed_tail_signature(expected.tail) == _closed_tail_signature(actual.tail)
        and expected.tensor_contracts == actual.tensor_contracts
        and expected.operator_contracts == actual.operator_contracts
    )


def _resolve_direct_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_DirectSplitTailPlan]:
    pre_root = _resolve_pre_transpose_root(
        model_ir,
        graph_index,
        pre,
        layout_state=layout_state,
    )
    if pre_root is None:
        return None
    pre_users = graph_index.consumer_indices(pre_root.output_name)
    if len(pre_users) != 1 or int(pre_users[0]) <= pre_root.operator_index:
        return None
    split_index = int(pre_users[0])
    root_split = model_ir.operators[split_index]
    if (
        _normalized_op_type(root_split) != "SPLIT"
        or len(root_split.inputs) != 2
        or str(root_split.inputs[1]) != pre_root.output_name
    ):
        return None

    reserved_names = set(str(name) for name in model_ir.tensors)
    nhwc_views = {pre_root.output_name: pre_root.source_view}
    converted_names = {pre_root.output_name}
    root_split_result = _resolve_split_updates(
        model_ir,
        graph_index,
        root_split,
        nhwc_views=nhwc_views,
        converted_names=converted_names,
        reserved_names=reserved_names,
    )
    if root_split_result is None:
        return None
    root_axis_update, root_output_updates = root_split_result
    nhwc_views.pop(pre_root.output_name, None)
    converted_names.remove(pre_root.output_name)
    for update in root_output_updates:
        nhwc_views[update.name] = _view_for_update(model_ir, update)
        converted_names.add(update.name)
    tail = _resolve_closed_tail(
        model_ir,
        graph_index,
        source_view=pre_root.source_view,
        permutation_name=pre_root.permutation_name,
        nhwc_views=nhwc_views,
        converted_names=converted_names,
        initial_updates=root_output_updates,
        initial_axis_updates=(root_axis_update,),
        accepted_operators=(root_split,),
        seed_names=tuple(update.name for update in root_output_updates),
        reserved_names=reserved_names,
        layout_state=layout_state,
        allow_slice=True,
    )
    if tail is None:
        return None

    relevant_operators = sorted(
        [pre, root_split, *tail.closure_operators],
        key=lambda operator: int(_operator_index(graph_index, operator) or 0),
    )
    relevant_tensor_names = {tail.adapter.public_name}
    for operator in relevant_operators:
        relevant_tensor_names.update(str(value) for value in operator.inputs)
        relevant_tensor_names.update(str(value) for value in operator.outputs)
    tensor_contracts = []
    for name in sorted(relevant_tensor_names):
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))
    return _DirectSplitTailPlan(
        pre=pre,
        root_split=root_split,
        pre_input_name=pre_root.input_name,
        pre_output_name=pre_root.output_name,
        tail=tail,
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in relevant_operators
        ),
    )


def _preflight_closed_tail(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tail: _ClosedTailPlan,
    *,
    required_operators: Sequence[OperatorIR],
) -> bool:
    created_names = (
        [
            update.clone_name
            for update in tail.axis_updates
            if update.clone_name is not None
        ]
        + [
            update.clone_name
            for update in tail.constant_updates
            if update.clone_name is not None
        ]
        + [tail.adapter.private_name]
    )
    if any(name in model_ir.tensors for name in created_names):
        return False
    if any(
        _operator_index(graph_index, operator) is None
        for operator in required_operators
    ):
        return False
    for update in tail.axis_updates:
        if (
            len(update.operator.inputs) != 2
            or str(update.operator.inputs[0]) != update.axis_name
        ):
            return False
    for update in tail.concat_updates:
        try:
            if int(update.operator.options.get("axis", -1)) != update.original_axis:
                return False
        except Exception:
            return False
    for update in tail.constant_updates:
        tensor = model_ir.tensors.get(update.tensor_name)
        if tensor is None or tensor.data is None:
            return False
        for use in update.uses:
            if (
                use.input_slot < 0
                or use.input_slot >= len(use.operator.inputs)
                or str(use.operator.inputs[use.input_slot]) != update.tensor_name
            ):
                return False
    return bool(
        0 <= tail.adapter.output_slot < len(tail.adapter.producer.outputs)
        and str(tail.adapter.producer.outputs[tail.adapter.output_slot])
        == tail.adapter.public_name
    )


def _apply_closed_tail(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tail: _ClosedTailPlan,
    *,
    layout_state: Optional[LayoutState],
) -> None:
    for update in tail.axis_updates:
        tensor = model_ir.tensors[update.axis_name]
        array = np.asarray([3], dtype=np.dtype(update.numpy_dtype)).reshape(
            update.data_shape
        )
        if update.clone_name is None:
            tensor.data = array
        else:
            model_ir.tensors[update.clone_name] = TensorIR(
                name=update.clone_name,
                dtype=update.dtype,
                shape=[int(value) for value in update.shape],
                shape_signature=[int(value) for value in update.signature],
                data=array,
                is_variable=False,
                quantization=None,
                logical_layout=str(tensor.logical_layout),
                physical_layout=str(tensor.physical_layout),
                onnx_tensor_name=tensor.onnx_tensor_name,
            )
            split_inputs = [str(value) for value in update.operator.inputs]
            split_inputs[0] = update.clone_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=update.operator,
                new_inputs=split_inputs,
                graph_index=graph_index,
            )
            if layout_state is not None:
                layout_state.set(
                    update.clone_name,
                    logical=str(tensor.logical_layout),
                    physical=str(tensor.physical_layout),
                )

    for update in tail.constant_updates:
        tensor = model_ir.tensors[update.tensor_name]
        array = np.asarray(
            update.values,
            dtype=np.dtype(update.numpy_dtype),
        ).reshape(update.data_shape)
        if update.clone_name is None:
            tensor.data = array
            continue
        model_ir.tensors[update.clone_name] = TensorIR(
            name=update.clone_name,
            dtype=update.dtype,
            shape=[int(value) for value in update.shape],
            shape_signature=[int(value) for value in update.signature],
            data=array,
            is_variable=False,
            quantization=None,
            logical_layout=str(tensor.logical_layout),
            physical_layout=str(tensor.physical_layout),
            onnx_tensor_name=tensor.onnx_tensor_name,
        )
        for use in update.uses:
            inputs = [str(value) for value in use.operator.inputs]
            inputs[use.input_slot] = update.clone_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=use.operator,
                new_inputs=inputs,
                graph_index=graph_index,
            )
        if layout_state is not None:
            layout_state.set(
                update.clone_name,
                logical=str(tensor.logical_layout),
                physical=str(tensor.physical_layout),
            )

    for update in tail.concat_updates:
        options = dict(update.operator.options)
        options["axis"] = int(update.new_axis)
        update.operator.options = options

    for update in tail.metadata_updates:
        tensor = model_ir.tensors[update.name]
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

    public_tensor = model_ir.tensors[tail.adapter.public_name]
    model_ir.tensors[tail.adapter.private_name] = TensorIR(
        name=tail.adapter.private_name,
        dtype=str(public_tensor.dtype),
        shape=[int(value) for value in tail.adapter.private_shape],
        shape_signature=[int(value) for value in tail.adapter.private_signature],
        data=None,
        is_variable=False,
        quantization=_clone_quantization(public_tensor.quantization),
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
        onnx_tensor_name=public_tensor.onnx_tensor_name,
    )
    public_tensor.logical_layout = LOGICAL_LAYOUT_NCHW
    public_tensor.physical_layout = LOGICAL_LAYOUT_NCHW
    producer_outputs = [str(value) for value in tail.adapter.producer.outputs]
    producer_outputs[tail.adapter.output_slot] = tail.adapter.private_name
    _set_operator_outputs(
        model_ir=model_ir,
        op=tail.adapter.producer,
        new_outputs=producer_outputs,
        graph_index=graph_index,
    )
    producer_index = _operator_index(graph_index, tail.adapter.producer)
    if producer_index is None:
        raise RuntimeError("validated Split-tail output producer disappeared")
    graph_index.insert_operator(
        producer_index + 1,
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[tail.adapter.private_name, tail.adapter.permutation_name],
            outputs=[tail.adapter.public_name],
        ),
    )
    if layout_state is not None:
        layout_state.set(
            tail.adapter.private_name,
            logical=LOGICAL_LAYOUT_NHWC,
            physical=LOGICAL_LAYOUT_NHWC,
        )
        layout_state.set(
            tail.adapter.public_name,
            logical=LOGICAL_LAYOUT_NCHW,
            physical=LOGICAL_LAYOUT_NCHW,
        )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _BinarySplitTailPlan,
    *,
    layout_state: Optional[LayoutState],
) -> bool:
    current = _resolve_candidate(
        model_ir,
        graph_index,
        plan.pre,
        layout_state=layout_state,
    )
    if current is None or not _plans_equal(plan, current):
        return False
    tail = _binary_tail(plan)
    required_operators = (
        plan.pre,
        plan.first,
        plan.second,
        plan.root_split,
        *plan.closure_operators,
    )
    if (
        plan.pre_input_slot < 0
        or plan.pre_input_slot >= len(plan.first.inputs)
        or str(plan.first.inputs[plan.pre_input_slot]) != plan.pre_output_name
        or not _preflight_closed_tail(
            model_ir,
            graph_index,
            tail,
            required_operators=required_operators,
        )
    ):
        return False

    first_inputs = [str(value) for value in plan.first.inputs]
    first_inputs[plan.pre_input_slot] = plan.pre_input_name
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.first,
        new_inputs=first_inputs,
        graph_index=graph_index,
    )
    _apply_closed_tail(
        model_ir,
        graph_index,
        tail,
        layout_state=layout_state,
    )

    pre_index = _operator_index(graph_index, plan.pre)
    if pre_index is None:
        raise RuntimeError("validated Split-tail input Transpose disappeared")
    graph_index.remove_operator(pre_index)
    return True


def _apply_direct_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _DirectSplitTailPlan,
    *,
    layout_state: Optional[LayoutState],
) -> bool:
    current = _resolve_direct_candidate(
        model_ir,
        graph_index,
        plan.pre,
        layout_state=layout_state,
    )
    if current is None or not _direct_plans_equal(plan, current):
        return False
    if (
        len(plan.root_split.inputs) != 2
        or str(plan.root_split.inputs[1]) != plan.pre_output_name
        or not _preflight_closed_tail(
            model_ir,
            graph_index,
            plan.tail,
            required_operators=(
                plan.pre,
                plan.root_split,
                *plan.tail.closure_operators,
            ),
        )
    ):
        return False
    split_inputs = [str(value) for value in plan.root_split.inputs]
    split_inputs[1] = plan.pre_input_name
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.root_split,
        new_inputs=split_inputs,
        graph_index=graph_index,
    )
    _apply_closed_tail(
        model_ir,
        graph_index,
        plan.tail,
        layout_state=layout_state,
    )
    pre_index = _operator_index(graph_index, plan.pre)
    if pre_index is None:
        raise RuntimeError("validated direct Split-tail input Transpose disappeared")
    graph_index.remove_operator(pre_index)
    return True


def optimize_transpose_split_channelwise_tail_to_single_post_nchw(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Move one closed direct Split tail to NHWC with one public adapter."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    rewrite_limit = max(0, int(max_rewrites))
    if rewrite_limit == 0:
        return {_DIRECT_STATS_KEY: 0}
    candidates = (
        [candidate]
        if candidate is not None
        else [
            model_ir.operators[index]
            for index in active_index.operator_indices_for_normalized_types(
                {"TRANSPOSE"}
            )
        ]
    )
    optimized = 0
    for pre in candidates:
        if optimized >= rewrite_limit:
            break
        if pre is None or _operator_index(active_index, pre) is None:
            continue
        plan = _resolve_direct_candidate(
            model_ir,
            active_index,
            pre,
            layout_state=layout_state,
        )
        if plan is None:
            continue
        if _apply_direct_plan(
            model_ir,
            active_index,
            plan,
            layout_state=layout_state,
        ):
            optimized += 1
    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {_DIRECT_STATS_KEY: int(optimized)}


def optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Move one closed binary/Split tail to NHWC with one public adapter."""

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
                {"TRANSPOSE"}
            )
        ]
    )
    optimized = 0
    for pre in candidates:
        if optimized >= rewrite_limit:
            break
        if pre is None or _operator_index(active_index, pre) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            pre,
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
            optimized += 1
    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {_STATS_KEY: int(optimized)}
