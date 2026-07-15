from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_NWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    _consumer_slots,
    _freeze,
    _op_type,
    _operator_contract,
    _operator_index,
    _per_tensor_quantization,
    _permuted_view,
    _resolved_source,
    _tensor_contract,
    _typed_constant,
    _view,
)


_STATS_KEY = "optimized_center_size_offset_terminal_transpose_chains"
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)


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
    logical_layout: str
    physical_layout: str


@dataclass(frozen=True)
class _OptionsUpdate:
    operator: OperatorIR
    original_options: Any
    new_options: Dict[str, Any]


@dataclass(frozen=True)
class _ConstantUse:
    operator: OperatorIR
    input_slot: int


@dataclass(frozen=True)
class _ConstantUpdate:
    source_name: str
    target_name: str
    uses: Tuple[_ConstantUse, ...]
    original_values: Tuple[int, ...]
    new_values: Tuple[int, ...]
    dtype: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    numpy_dtype: str
    data_shape: Tuple[int, ...]


@dataclass(frozen=True)
class _TransposeRoot:
    pre: OperatorIR
    operator_index: int
    source_name: str
    output_name: str
    source_view: Any
    output_view: Any
    source_logical_layout: str
    source_physical_layout: str


@dataclass(frozen=True)
class _CoordsPlan:
    concat: OperatorIR
    gather: OperatorIR
    axis_reshape: OperatorIR
    batch_name: str
    axis_name: str
    channel_name: str
    original_inputs: Tuple[str, ...]
    new_inputs: Tuple[str, ...]


@dataclass(frozen=True)
class _CenterBranch:
    root: _TransposeRoot
    logistic: OperatorIR
    maximum: OperatorIR
    minimum: OperatorIR
    reshape: OperatorIR
    output_names: Tuple[str, str, str, str]


@dataclass(frozen=True)
class _SizeBranch:
    root: _TransposeRoot
    logistic: OperatorIR
    maximum: OperatorIR
    minimum: OperatorIR
    reshape: OperatorIR
    gather: OperatorIR
    coords: _CoordsPlan
    output_names: Tuple[str, str, str, str]
    shape_name: str
    shape_values: Tuple[int, ...]
    converted_shape_values: Tuple[int, ...]
    converted_shape: Tuple[int, ...]
    converted_signature: Tuple[int, ...]


@dataclass(frozen=True)
class _OffsetBranch:
    root: _TransposeRoot
    reshape: OperatorIR
    gather: OperatorIR
    coords: _CoordsPlan
    output_name: str
    shape_name: str
    shape_values: Tuple[int, ...]
    converted_shape_values: Tuple[int, ...]
    converted_shape: Tuple[int, ...]
    converted_signature: Tuple[int, ...]


@dataclass(frozen=True)
class _Plan:
    center: _CenterBranch
    size: _SizeBranch
    offset: _OffsetBranch
    input_rewrites: Tuple[_InputRewrite, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    options_updates: Tuple[_OptionsUpdate, ...]
    constant_updates: Tuple[_ConstantUpdate, ...]
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _layout_pair(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
) -> Tuple[str, str]:
    if layout_state is None:
        return str(tensor.logical_layout), str(tensor.physical_layout)
    return (
        str(layout_state.logical_of(str(name))),
        str(layout_state.physical_of(str(name))),
    )


def _layout_is(value: str, expected: str) -> bool:
    normalized = str(value).upper()
    return normalized in {LOGICAL_LAYOUT_UNKNOWN, str(expected).upper()}


def _runtime_tensor(
    tensor: Optional[TensorIR],
    *,
    rank: int,
    allow_constant: bool = False,
) -> bool:
    if tensor is None:
        return False
    view = _view(tensor)
    if (
        bool(tensor.is_variable)
        or (tensor.data is not None and not allow_constant)
        or len(view.shape) != int(rank)
        or len(view.signature) != int(rank)
        or any(int(value) <= 0 for value in view.shape)
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return False
    if tensor.data is None:
        return True
    try:
        array = np.asarray(tensor.data)
        return bool(
            tuple(int(value) for value in array.shape) == view.shape
            and str(array.dtype).upper() == str(tensor.dtype).upper()
        )
    except Exception:
        return False


def _same_quantization(left: TensorIR, right: TensorIR) -> bool:
    return _freeze(left.quantization) == _freeze(right.quantization)


def _resolved_transpose_root(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_TransposeRoot]:
    pre_index = _operator_index(graph_index, pre)
    if (
        pre_index is None
        or _op_type(pre) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
    ):
        return None
    permutation = _typed_constant(
        model_ir,
        graph_index,
        str(pre.inputs[1]),
        shape=(4,),
    )
    if (
        permutation is None
        or tuple(int(value) for value in permutation[1].reshape(-1))
        != _PERM_NHWC_TO_NCHW
    ):
        return None
    graph_outputs = {str(value) for value in model_ir.outputs}
    source_name = str(pre.inputs[0])
    output_name = str(pre.outputs[0])
    if (
        source_name in graph_outputs
        or output_name in graph_outputs
        or output_name in graph_index.duplicate_producers
        or graph_index.producers.get(output_name) != int(pre_index)
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(pre_index),
        )
    ):
        return None
    source = model_ir.tensors.get(source_name)
    output = model_ir.tensors.get(output_name)
    if (
        not _runtime_tensor(source, rank=4, allow_constant=True)
        or not _runtime_tensor(output, rank=4)
    ):
        return None
    assert source is not None and output is not None
    source_view = _view(source)
    output_view = _view(output)
    if (
        _permuted_view(source_view, _PERM_NHWC_TO_NCHW) != output_view
        or not _same_quantization(source, output)
    ):
        return None
    source_logical, source_physical = _layout_pair(
        source_name,
        source,
        layout_state,
    )
    output_logical, output_physical = _layout_pair(
        output_name,
        output,
        layout_state,
    )
    if not (
        _layout_is(source_logical, LOGICAL_LAYOUT_NHWC)
        and _layout_is(source_physical, LOGICAL_LAYOUT_NHWC)
        and _layout_is(output_logical, LOGICAL_LAYOUT_NCHW)
        and _layout_is(output_physical, LOGICAL_LAYOUT_NCHW)
    ):
        return None
    return _TransposeRoot(
        pre=pre,
        operator_index=int(pre_index),
        source_name=source_name,
        output_name=output_name,
        source_view=source_view,
        output_view=output_view,
        source_logical_layout=source_logical,
        source_physical_layout=source_physical,
    )


def _sole_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Optional[OperatorIR]:
    slots = _consumer_slots(model_ir, graph_index, str(name))
    return slots[0][0] if len(slots) == 1 else None


def _singleton_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    expected_dtype: str,
) -> Optional[TensorIR]:
    tensor = model_ir.tensors.get(str(name))
    public = {str(value) for value in (*model_ir.inputs, *model_ir.outputs)}
    if (
        tensor is None
        or tensor.data is None
        or bool(tensor.is_variable)
        or str(name) in public
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or str(tensor.dtype) != str(expected_dtype)
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return None
    try:
        array = np.asarray(tensor.data)
        signature = (
            tensor.shape
            if tensor.shape_signature is None
            else tensor.shape_signature
        )
        if (
            int(array.size) != 1
            or str(array.dtype).upper() != str(tensor.dtype).upper()
            or len(tensor.shape) != len(signature)
            or any(int(value) <= 0 for value in tensor.shape)
            or int(np.prod(tensor.shape, dtype=np.int64)) != 1
        ):
            return None
    except Exception:
        return None
    return tensor


def _private_output_matches(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    name: str,
    *,
    expected_view: Any,
    expected_quantization: Any,
    expected_layout: str,
    layout_state: Optional[LayoutState],
) -> bool:
    operator_index = _operator_index(graph_index, operator)
    tensor = model_ir.tensors.get(str(name))
    return bool(
        operator_index is not None
        and str(name) not in {str(value) for value in (*model_ir.inputs, *model_ir.outputs)}
        and str(name) not in graph_index.duplicate_producers
        and graph_index.producers.get(str(name)) == int(operator_index)
        and _runtime_tensor(tensor, rank=len(expected_view.shape))
        and _view(tensor) == expected_view  # type: ignore[arg-type]
        and _freeze(tensor.quantization) == expected_quantization  # type: ignore[union-attr]
        and _layout_is(
            _layout_pair(str(name), tensor, layout_state)[1],  # type: ignore[arg-type]
            expected_layout,
        )
    )


def _next_unary(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    op_type: str,
    *,
    after_index: int,
) -> Optional[Tuple[OperatorIR, str, int]]:
    operator = _sole_consumer(model_ir, graph_index, name)
    operator_index = None if operator is None else _operator_index(graph_index, operator)
    if (
        operator is None
        or operator_index is None
        or int(operator_index) <= int(after_index)
        or _op_type(operator) != str(op_type)
        or len(operator.inputs) != 1
        or len(operator.outputs) != 1
        or str(operator.inputs[0]) != str(name)
    ):
        return None
    return operator, str(operator.outputs[0]), int(operator_index)


def _next_binary_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    op_type: str,
    *,
    after_index: int,
    expected_dtype: str,
) -> Optional[Tuple[OperatorIR, str, str, int]]:
    operator = _sole_consumer(model_ir, graph_index, name)
    operator_index = None if operator is None else _operator_index(graph_index, operator)
    if (
        operator is None
        or operator_index is None
        or int(operator_index) <= int(after_index)
        or _op_type(operator) != str(op_type)
        or len(operator.inputs) != 2
        or len(operator.outputs) != 1
        or list(str(value) for value in operator.inputs).count(str(name)) != 1
    ):
        return None
    side_name = next(
        str(value) for value in operator.inputs if str(value) != str(name)
    )
    if _singleton_constant(
        model_ir,
        graph_index,
        side_name,
        expected_dtype=expected_dtype,
    ) is None:
        return None
    return operator, str(operator.outputs[0]), side_name, int(operator_index)


def _resolved_reshape_shape(
    input_shape: Sequence[int],
    values: Sequence[int],
) -> Optional[Tuple[int, ...]]:
    resolved = []
    unknown_index: Optional[int] = None
    known_product = 1
    for index, raw_value in enumerate(values):
        value = int(raw_value)
        if value == 0:
            if index >= len(input_shape):
                return None
            value = int(input_shape[index])
        elif value == -1:
            if unknown_index is not None:
                return None
            unknown_index = int(index)
            resolved.append(-1)
            continue
        elif value < -1:
            return None
        if value <= 0:
            return None
        known_product *= int(value)
        resolved.append(int(value))
    input_product = int(np.prod(tuple(int(value) for value in input_shape), dtype=np.int64))
    if unknown_index is not None:
        if known_product <= 0 or input_product % known_product != 0:
            return None
        resolved[int(unknown_index)] = int(input_product // known_product)
    elif int(known_product) != int(input_product):
        return None
    return tuple(int(value) for value in resolved)


def _converted_signature(source_view: Any) -> Tuple[int, int, int]:
    batch = int(source_view.signature[0])
    height = int(source_view.signature[1])
    width = int(source_view.signature[2])
    channel = int(source_view.signature[3])
    spatial = int(height * width) if height >= 0 and width >= 0 else -1
    return batch, spatial, channel


def _nchw_flat_signature(source_view: Any) -> Tuple[int, int, int]:
    batch = int(source_view.signature[0])
    height = int(source_view.signature[1])
    width = int(source_view.signature[2])
    channel = int(source_view.signature[3])
    spatial = int(height * width) if height >= 0 and width >= 0 else -1
    return batch, channel, spatial


def _reshape_options_update(
    operator: OperatorIR,
    new_values: Sequence[int],
) -> Optional[_OptionsUpdate]:
    if not isinstance(operator.options, dict):
        return None
    new_options = copy.deepcopy(operator.options)
    if "newShape" in new_options:
        value = new_options["newShape"]
        if not isinstance(value, list):
            return None
        new_options["newShape"] = [int(item) for item in new_values]
    if "onnxRawNewShape" in new_options:
        raw = new_options["onnxRawNewShape"]
        if not isinstance(raw, list) or len(raw) != 3:
            return None
        try:
            new_options["onnxRawNewShape"] = [
                int(raw[0]),
                int(raw[2]),
                int(raw[1]),
            ]
        except Exception:
            return None
    return _OptionsUpdate(
        operator=operator,
        original_options=copy.deepcopy(operator.options),
        new_options=new_options,
    )


def _resolve_coords_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    gather: OperatorIR,
    *,
    data_name: str,
    batch_size: int,
    channel_size: int,
) -> Optional[_CoordsPlan]:
    gather_index = _operator_index(graph_index, gather)
    if (
        gather_index is None
        or _op_type(gather) != "GATHER_ND"
        or len(gather.inputs) != 2
        or len(gather.outputs) != 1
        or str(gather.inputs[0]) != str(data_name)
    ):
        return None
    coords_name = str(gather.inputs[1])
    public_names = {str(value) for value in (*model_ir.inputs, *model_ir.outputs)}
    if coords_name in public_names or coords_name in graph_index.duplicate_producers:
        return None
    concat_index = graph_index.producers.get(coords_name)
    if concat_index is None or int(concat_index) >= int(gather_index):
        return None
    concat = model_ir.operators[int(concat_index)]
    if (
        _op_type(concat) != "CONCATENATION"
        or len(concat.inputs) != 3
        or len(concat.outputs) != 1
        or str(concat.outputs[0]) != coords_name
        or not isinstance(concat.options, dict)
    ):
        return None
    try:
        if int(concat.options.get("axis", -1)) != 3:
            return None
    except Exception:
        return None
    coords_tensor = model_ir.tensors.get(coords_name)
    if not _runtime_tensor(coords_tensor, rank=4):
        return None
    assert coords_tensor is not None
    coords_view = _view(coords_tensor)
    if int(coords_view.shape[-1]) != 3 or int(coords_view.signature[-1]) not in {-1, 3}:
        return None

    input_names = tuple(str(value) for value in concat.inputs)
    input_tensors = [model_ir.tensors.get(name) for name in input_names]
    if any(tensor is None for tensor in input_tensors):
        return None
    coord_shape = tuple(int(value) for value in coords_view.shape[:-1]) + (1,)
    coord_signature = tuple(int(value) for value in coords_view.signature[:-1]) + (1,)
    for tensor in input_tensors:
        assert tensor is not None
        view = _view(tensor)
        if (
            not _runtime_tensor(tensor, rank=4, allow_constant=True)
            or view.shape != coord_shape
            or view.signature != coord_signature
            or view.dtype != coords_view.dtype
            or _freeze(tensor.quantization) != _freeze(coords_tensor.quantization)
        ):
            return None

    axis_candidates = []
    for position, name in enumerate(input_names):
        if name in graph_index.duplicate_producers:
            continue
        producer_index = graph_index.producers.get(name)
        if producer_index is None or int(producer_index) >= int(concat_index):
            continue
        producer = model_ir.operators[int(producer_index)]
        if (
            _op_type(producer) == "RESHAPE"
            and len(producer.inputs) == 2
            and len(producer.outputs) == 1
            and str(producer.outputs[0]) == name
            and _typed_constant(
                model_ir,
                graph_index,
                str(producer.inputs[1]),
                shape=(4,),
            )
            is not None
        ):
            axis_candidates.append((int(position), producer))
    if len(axis_candidates) != 1:
        return None
    axis_position, axis_reshape = axis_candidates[0]
    axis_name = input_names[int(axis_position)]
    axis_reshape_index = _operator_index(graph_index, axis_reshape)
    axis_source_name = str(axis_reshape.inputs[0])
    axis_source = model_ir.tensors.get(axis_source_name)
    axis_shape = _typed_constant(
        model_ir,
        graph_index,
        str(axis_reshape.inputs[1]),
        shape=(4,),
    )
    if (
        axis_name in public_names
        or axis_reshape_index is None
        or not isinstance(axis_reshape.options, dict)
        or not _resolved_source(
            model_ir,
            graph_index,
            name=axis_source_name,
            before_index=int(axis_reshape_index),
        )
        or not _runtime_tensor(axis_source, rank=3, allow_constant=True)
        or _view(axis_source).shape != coord_shape[:-1]  # type: ignore[union-attr]
        or _view(axis_source).signature != coord_signature[:-1]  # type: ignore[union-attr]
        or str(axis_source.dtype) != str(coords_tensor.dtype)  # type: ignore[union-attr]
        or axis_shape is None
        or _resolved_reshape_shape(
            _view(axis_source).shape,  # type: ignore[arg-type]
            tuple(int(value) for value in axis_shape[1].reshape(-1)),
        )
        != coord_shape
    ):
        return None

    remaining_positions = [
        position for position in range(3) if position != int(axis_position)
    ]
    constant_arrays: Dict[int, np.ndarray] = {}
    for position in remaining_positions:
        name = input_names[int(position)]
        tensor = model_ir.tensors[name]
        if (
            tensor.data is None
            or bool(tensor.is_variable)
            or name in graph_index.producers
            or name in graph_index.duplicate_producers
            or name in public_names
            or str(tensor.dtype) not in {"INT32", "INT64"}
        ):
            return None
        array = np.asarray(tensor.data)
        if (
            tuple(int(value) for value in array.shape) != coord_shape
            or str(array.dtype).upper() != str(tensor.dtype).upper()
        ):
            return None
        constant_arrays[int(position)] = array

    assignments = []
    for batch_position, channel_position in (
        (remaining_positions[0], remaining_positions[1]),
        (remaining_positions[1], remaining_positions[0]),
    ):
        batch_values = constant_arrays[int(batch_position)].reshape(-1)
        channel_values = constant_arrays[int(channel_position)].reshape(-1)
        if (
            np.all(batch_values >= 0)
            and np.all(batch_values < int(batch_size))
            and np.all(channel_values >= 0)
            and np.all(channel_values < int(channel_size))
        ):
            assignments.append((int(batch_position), int(channel_position)))
    if len(assignments) == 0:
        return None
    if len(assignments) > 1:
        left = constant_arrays[remaining_positions[0]]
        right = constant_arrays[remaining_positions[1]]
        if not np.array_equal(left, right):
            return None
        assignments = [(remaining_positions[0], remaining_positions[1])]
    batch_position, channel_position = assignments[0]
    batch_name = input_names[int(batch_position)]
    channel_name = input_names[int(channel_position)]

    gather_output = model_ir.tensors.get(str(gather.outputs[0]))
    data_tensor = model_ir.tensors.get(str(data_name))
    if (
        data_tensor is None
        or str(gather.outputs[0]) in {str(value) for value in model_ir.inputs}
        or str(gather.outputs[0]) in graph_index.duplicate_producers
        or graph_index.producers.get(str(gather.outputs[0])) != int(gather_index)
        or not _runtime_tensor(gather_output, rank=3)
        or _view(gather_output).shape != tuple(int(value) for value in coords_view.shape[:-1])  # type: ignore[union-attr]
        or _view(gather_output).signature != tuple(int(value) for value in coords_view.signature[:-1])  # type: ignore[union-attr]
        or str(gather_output.dtype) != str(data_tensor.dtype)  # type: ignore[union-attr]
        or not _same_quantization(gather_output, data_tensor)  # type: ignore[arg-type]
    ):
        return None
    return _CoordsPlan(
        concat=concat,
        gather=gather,
        axis_reshape=axis_reshape,
        batch_name=batch_name,
        axis_name=axis_name,
        channel_name=channel_name,
        original_inputs=input_names,
        new_inputs=(batch_name, axis_name, channel_name),
    )


def _resolve_center_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_CenterBranch]:
    root = _resolved_transpose_root(
        model_ir,
        graph_index,
        pre,
        layout_state=layout_state,
    )
    if root is None or int(root.source_view.shape[3]) != 1:
        return None
    quantization = _freeze(model_ir.tensors[root.output_name].quantization)
    logistic_info = _next_unary(
        model_ir,
        graph_index,
        root.output_name,
        "LOGISTIC",
        after_index=root.operator_index,
    )
    if logistic_info is None:
        return None
    logistic, logistic_output, logistic_index = logistic_info
    maximum_info = _next_binary_constant(
        model_ir,
        graph_index,
        logistic_output,
        "MAXIMUM",
        after_index=logistic_index,
        expected_dtype=root.output_view.dtype,
    )
    if maximum_info is None:
        return None
    maximum, maximum_output, _, maximum_index = maximum_info
    minimum_info = _next_binary_constant(
        model_ir,
        graph_index,
        maximum_output,
        "MINIMUM",
        after_index=maximum_index,
        expected_dtype=root.output_view.dtype,
    )
    if minimum_info is None:
        return None
    minimum, minimum_output, _, minimum_index = minimum_info
    for operator, name in (
        (logistic, logistic_output),
        (maximum, maximum_output),
        (minimum, minimum_output),
    ):
        if not _private_output_matches(
            model_ir,
            graph_index,
            operator,
            name,
            expected_view=root.output_view,
            expected_quantization=quantization,
            expected_layout=LOGICAL_LAYOUT_NCHW,
            layout_state=layout_state,
        ):
            return None
    reshape = _sole_consumer(model_ir, graph_index, minimum_output)
    reshape_index = None if reshape is None else _operator_index(graph_index, reshape)
    if (
        reshape is None
        or reshape_index is None
        or int(reshape_index) <= int(minimum_index)
        or _op_type(reshape) != "RESHAPE"
        or len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != minimum_output
        or not isinstance(reshape.options, dict)
    ):
        return None
    shape_resolved = _typed_constant(
        model_ir,
        graph_index,
        str(reshape.inputs[1]),
        shape=(2,),
    )
    if shape_resolved is None:
        return None
    values = tuple(int(value) for value in shape_resolved[1].reshape(-1))
    target = _resolved_reshape_shape(root.output_view.shape, values)
    expected = (
        int(root.source_view.shape[0]),
        int(root.source_view.shape[1]) * int(root.source_view.shape[2]),
    )
    expected_signature = (
        int(root.source_view.signature[0]),
        (
            int(root.source_view.signature[1])
            * int(root.source_view.signature[2])
            if int(root.source_view.signature[1]) >= 0
            and int(root.source_view.signature[2]) >= 0
            else -1
        ),
    )
    output_name = str(reshape.outputs[0])
    output_tensor = model_ir.tensors.get(output_name)
    if (
        target != expected
        or output_name in {str(value) for value in model_ir.inputs}
        or output_name in graph_index.duplicate_producers
        or graph_index.producers.get(output_name) != int(reshape_index)
        or not _runtime_tensor(output_tensor, rank=2)
        or _view(output_tensor).shape != expected  # type: ignore[union-attr]
        or _view(output_tensor).signature != expected_signature  # type: ignore[union-attr]
        or str(output_tensor.dtype) != root.output_view.dtype  # type: ignore[union-attr]
        or _freeze(output_tensor.quantization) != quantization  # type: ignore[union-attr]
    ):
        return None
    return _CenterBranch(
        root=root,
        logistic=logistic,
        maximum=maximum,
        minimum=minimum,
        reshape=reshape,
        output_names=(
            logistic_output,
            maximum_output,
            minimum_output,
            output_name,
        ),
    )


def _resolve_size_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_SizeBranch]:
    root = _resolved_transpose_root(
        model_ir,
        graph_index,
        pre,
        layout_state=layout_state,
    )
    if root is None:
        return None
    quantization = _freeze(model_ir.tensors[root.output_name].quantization)
    logistic_info = _next_unary(
        model_ir,
        graph_index,
        root.output_name,
        "LOGISTIC",
        after_index=root.operator_index,
    )
    if logistic_info is None:
        return None
    logistic, logistic_output, logistic_index = logistic_info
    maximum_info = _next_binary_constant(
        model_ir,
        graph_index,
        logistic_output,
        "MAXIMUM",
        after_index=logistic_index,
        expected_dtype=root.output_view.dtype,
    )
    if maximum_info is None:
        return None
    maximum, maximum_output, _, maximum_index = maximum_info
    minimum_info = _next_binary_constant(
        model_ir,
        graph_index,
        maximum_output,
        "MINIMUM",
        after_index=maximum_index,
        expected_dtype=root.output_view.dtype,
    )
    if minimum_info is None:
        return None
    minimum, minimum_output, _, minimum_index = minimum_info
    for operator, name in (
        (logistic, logistic_output),
        (maximum, maximum_output),
        (minimum, minimum_output),
    ):
        if not _private_output_matches(
            model_ir,
            graph_index,
            operator,
            name,
            expected_view=root.output_view,
            expected_quantization=quantization,
            expected_layout=LOGICAL_LAYOUT_NCHW,
            layout_state=layout_state,
        ):
            return None
    reshape = _sole_consumer(model_ir, graph_index, minimum_output)
    reshape_index = None if reshape is None else _operator_index(graph_index, reshape)
    if (
        reshape is None
        or reshape_index is None
        or int(reshape_index) <= int(minimum_index)
        or _op_type(reshape) != "RESHAPE"
        or len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != minimum_output
    ):
        return None
    shape_name = str(reshape.inputs[1])
    shape_resolved = _typed_constant(
        model_ir,
        graph_index,
        shape_name,
        shape=(3,),
    )
    if shape_resolved is None:
        return None
    shape_values = tuple(int(value) for value in shape_resolved[1].reshape(-1))
    expected_old = (
        int(root.source_view.shape[0]),
        int(root.source_view.shape[3]),
        int(root.source_view.shape[1]) * int(root.source_view.shape[2]),
    )
    if _resolved_reshape_shape(root.output_view.shape, shape_values) != expected_old:
        return None
    reshape_output = str(reshape.outputs[0])
    reshape_tensor = model_ir.tensors.get(reshape_output)
    if (
        reshape_output in {str(value) for value in (*model_ir.inputs, *model_ir.outputs)}
        or reshape_output in graph_index.duplicate_producers
        or graph_index.producers.get(reshape_output) != int(reshape_index)
        or not _runtime_tensor(reshape_tensor, rank=3)
        or _view(reshape_tensor).shape != expected_old  # type: ignore[union-attr]
        or _view(reshape_tensor).signature != _nchw_flat_signature(root.source_view)  # type: ignore[union-attr]
        or str(reshape_tensor.dtype) != root.output_view.dtype  # type: ignore[union-attr]
        or _freeze(reshape_tensor.quantization) != quantization  # type: ignore[union-attr]
    ):
        return None
    gather = _sole_consumer(model_ir, graph_index, reshape_output)
    gather_index = None if gather is None else _operator_index(graph_index, gather)
    if gather is None or gather_index is None or int(gather_index) <= int(reshape_index):
        return None
    coords = _resolve_coords_plan(
        model_ir,
        graph_index,
        gather,
        data_name=reshape_output,
        batch_size=expected_old[0],
        channel_size=expected_old[1],
    )
    if coords is None:
        return None
    converted_values = (
        int(shape_values[0]),
        int(shape_values[2]),
        int(shape_values[1]),
    )
    converted_shape = (
        int(root.source_view.shape[0]),
        int(root.source_view.shape[1]) * int(root.source_view.shape[2]),
        int(root.source_view.shape[3]),
    )
    if _resolved_reshape_shape(root.source_view.shape, converted_values) != converted_shape:
        return None
    if _reshape_options_update(reshape, converted_values) is None:
        return None
    return _SizeBranch(
        root=root,
        logistic=logistic,
        maximum=maximum,
        minimum=minimum,
        reshape=reshape,
        gather=gather,
        coords=coords,
        output_names=(
            logistic_output,
            maximum_output,
            minimum_output,
            reshape_output,
        ),
        shape_name=shape_name,
        shape_values=shape_values,
        converted_shape_values=converted_values,
        converted_shape=converted_shape,
        converted_signature=_converted_signature(root.source_view),
    )


def _resolve_offset_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_OffsetBranch]:
    root = _resolved_transpose_root(
        model_ir,
        graph_index,
        pre,
        layout_state=layout_state,
    )
    if root is None:
        return None
    reshape = _sole_consumer(model_ir, graph_index, root.output_name)
    reshape_index = None if reshape is None else _operator_index(graph_index, reshape)
    if (
        reshape is None
        or reshape_index is None
        or int(reshape_index) <= int(root.operator_index)
        or _op_type(reshape) != "RESHAPE"
        or len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != root.output_name
    ):
        return None
    shape_name = str(reshape.inputs[1])
    shape_resolved = _typed_constant(
        model_ir,
        graph_index,
        shape_name,
        shape=(3,),
    )
    if shape_resolved is None:
        return None
    shape_values = tuple(int(value) for value in shape_resolved[1].reshape(-1))
    expected_old = (
        int(root.source_view.shape[0]),
        int(root.source_view.shape[3]),
        int(root.source_view.shape[1]) * int(root.source_view.shape[2]),
    )
    if _resolved_reshape_shape(root.output_view.shape, shape_values) != expected_old:
        return None
    output_name = str(reshape.outputs[0])
    output_tensor = model_ir.tensors.get(output_name)
    expected_quantization = _freeze(model_ir.tensors[root.output_name].quantization)
    if (
        output_name in {str(value) for value in (*model_ir.inputs, *model_ir.outputs)}
        or output_name in graph_index.duplicate_producers
        or graph_index.producers.get(output_name) != int(reshape_index)
        or not _runtime_tensor(output_tensor, rank=3)
        or _view(output_tensor).shape != expected_old  # type: ignore[union-attr]
        or _view(output_tensor).signature != _nchw_flat_signature(root.source_view)  # type: ignore[union-attr]
        or str(output_tensor.dtype) != root.output_view.dtype  # type: ignore[union-attr]
        or _freeze(output_tensor.quantization) != expected_quantization  # type: ignore[union-attr]
    ):
        return None
    gather = _sole_consumer(model_ir, graph_index, output_name)
    gather_index = None if gather is None else _operator_index(graph_index, gather)
    if gather is None or gather_index is None or int(gather_index) <= int(reshape_index):
        return None
    coords = _resolve_coords_plan(
        model_ir,
        graph_index,
        gather,
        data_name=output_name,
        batch_size=expected_old[0],
        channel_size=expected_old[1],
    )
    if coords is None:
        return None
    converted_values = (
        int(shape_values[0]),
        int(shape_values[2]),
        int(shape_values[1]),
    )
    converted_shape = (
        int(root.source_view.shape[0]),
        int(root.source_view.shape[1]) * int(root.source_view.shape[2]),
        int(root.source_view.shape[3]),
    )
    if _resolved_reshape_shape(root.source_view.shape, converted_values) != converted_shape:
        return None
    if _reshape_options_update(reshape, converted_values) is None:
        return None
    return _OffsetBranch(
        root=root,
        reshape=reshape,
        gather=gather,
        coords=coords,
        output_name=output_name,
        shape_name=shape_name,
        shape_values=shape_values,
        converted_shape_values=converted_values,
        converted_shape=converted_shape,
        converted_signature=_converted_signature(root.source_view),
    )


def _same_source_geometry(center: _CenterBranch, size: _SizeBranch) -> bool:
    return bool(
        tuple(int(value) for value in center.root.source_view.shape[:3])
        == tuple(int(value) for value in size.root.source_view.shape[:3])
        and tuple(int(value) for value in center.root.source_view.signature[:3])
        == tuple(int(value) for value in size.root.source_view.signature[:3])
    )


def _size_offset_compatible(size: _SizeBranch, offset: _OffsetBranch) -> bool:
    return bool(
        size.root.source_view == offset.root.source_view
        and _freeze(
            size.root.source_view.dtype
        ) == _freeze(offset.root.source_view.dtype)
        and size.shape_values == offset.shape_values
        and (
            size.coords.batch_name,
            size.coords.axis_name,
            size.coords.channel_name,
        )
        == (
            offset.coords.batch_name,
            offset.coords.axis_name,
            offset.coords.channel_name,
        )
    )


def _unique_name(base: str, occupied: set[str]) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in occupied:
        candidate = f"{base}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _constant_updates(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    size: _SizeBranch,
    offset: _OffsetBranch,
) -> Optional[Tuple[_ConstantUpdate, ...]]:
    grouped: Dict[str, list[Tuple[_ConstantUse, Tuple[int, ...], Tuple[int, ...]]]] = {}
    grouped.setdefault(size.shape_name, []).append(
        (
            _ConstantUse(size.reshape, 1),
            size.shape_values,
            size.converted_shape_values,
        )
    )
    grouped.setdefault(offset.shape_name, []).append(
        (
            _ConstantUse(offset.reshape, 1),
            offset.shape_values,
            offset.converted_shape_values,
        )
    )
    occupied = set(str(name) for name in model_ir.tensors)
    updates = []
    for source_name in sorted(grouped):
        intents = grouped[source_name]
        originals = {intent[1] for intent in intents}
        replacements = {intent[2] for intent in intents}
        if len(originals) != 1 or len(replacements) != 1:
            return None
        tensor = model_ir.tensors.get(source_name)
        if tensor is None or tensor.data is None:
            return None
        array = np.asarray(tensor.data)
        original_values = next(iter(originals))
        if tuple(int(value) for value in array.reshape(-1)) != original_values:
            return None
        planned_uses = tuple(
            sorted(
                (intent[0] for intent in intents),
                key=lambda use: (
                    int(_operator_index(graph_index, use.operator) or -1),
                    int(use.input_slot),
                ),
            )
        )
        actual_uses = _consumer_slots(model_ir, graph_index, source_name)
        planned_signature = tuple(
            (id(use.operator), int(use.input_slot)) for use in planned_uses
        )
        actual_signature = tuple(
            (id(operator), int(input_slot))
            for operator, input_slot in actual_uses
        )
        target_name = source_name
        if actual_signature != planned_signature:
            target_name = _unique_name(f"{source_name}_nhwc", occupied)
        signature = (
            tensor.shape
            if tensor.shape_signature is None
            else tensor.shape_signature
        )
        updates.append(
            _ConstantUpdate(
                source_name=source_name,
                target_name=target_name,
                uses=planned_uses,
                original_values=original_values,
                new_values=next(iter(replacements)),
                dtype=str(tensor.dtype),
                shape=tuple(int(value) for value in tensor.shape),
                signature=tuple(int(value) for value in signature),
                numpy_dtype=str(array.dtype),
                data_shape=tuple(int(value) for value in array.shape),
            )
        )
    return tuple(updates)


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    center_pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    transpose_operators = [
        model_ir.operators[index]
        for index in graph_index.operator_indices_for_normalized_types({"TRANSPOSE"})
    ]
    center = _resolve_center_branch(
        model_ir,
        graph_index,
        center_pre,
        layout_state=layout_state,
    )
    if center is None:
        return None
    centers = [
        branch
        for operator in transpose_operators
        if (
            branch := _resolve_center_branch(
                model_ir,
                graph_index,
                operator,
                layout_state=layout_state,
            )
        )
        is not None
        and tuple(int(value) for value in branch.root.source_view.shape[:3])
        == tuple(int(value) for value in center.root.source_view.shape[:3])
        and tuple(int(value) for value in branch.root.source_view.signature[:3])
        == tuple(int(value) for value in center.root.source_view.signature[:3])
    ]
    if len(centers) != 1 or centers[0].root.pre is not center_pre:
        return None
    sizes = [
        branch
        for operator in transpose_operators
        if operator is not center_pre
        and (
            branch := _resolve_size_branch(
                model_ir,
                graph_index,
                operator,
                layout_state=layout_state,
            )
        )
        is not None
        and _same_source_geometry(center, branch)
    ]
    offsets = [
        branch
        for operator in transpose_operators
        if operator is not center_pre
        and (
            branch := _resolve_offset_branch(
                model_ir,
                graph_index,
                operator,
                layout_state=layout_state,
            )
        )
        is not None
    ]
    pairs = [
        (size, offset)
        for size in sizes
        for offset in offsets
        if size.root.pre is not offset.root.pre
        and _size_offset_compatible(size, offset)
    ]
    if len(pairs) != 1:
        return None
    size, offset = pairs[0]

    concat_uses: Dict[int, Tuple[OperatorIR, set[Tuple[int, int]]]] = {}
    for coords in (size.coords, offset.coords):
        expected = concat_uses.setdefault(
            id(coords.concat),
            (coords.concat, set()),
        )[1]
        expected.add((id(coords.gather), 1))
    for concat, expected in concat_uses.values():
        coords_name = str(concat.outputs[0])
        actual = {
            (id(operator), int(input_slot))
            for operator, input_slot in _consumer_slots(
                model_ir,
                graph_index,
                coords_name,
            )
        }
        if actual != expected:
            return None

    branch_operators = [
        center.root.pre,
        center.logistic,
        center.maximum,
        center.minimum,
        center.reshape,
        size.root.pre,
        size.logistic,
        size.maximum,
        size.minimum,
        size.reshape,
        size.gather,
        offset.root.pre,
        offset.reshape,
        offset.gather,
    ]
    if len({id(operator) for operator in branch_operators}) != len(branch_operators):
        return None

    rewrites_by_operator: Dict[int, Tuple[OperatorIR, list[str]]] = {}

    def plan_input(operator: OperatorIR, slot: int, name: str) -> bool:
        current = rewrites_by_operator.setdefault(
            id(operator),
            (operator, [str(value) for value in operator.inputs]),
        )[1]
        if int(slot) < 0 or int(slot) >= len(current):
            return False
        current[int(slot)] = str(name)
        return True

    if not (
        plan_input(center.logistic, 0, center.root.source_name)
        and plan_input(size.logistic, 0, size.root.source_name)
        and plan_input(offset.reshape, 0, offset.root.source_name)
    ):
        return None
    for coords in (size.coords, offset.coords):
        existing = rewrites_by_operator.get(id(coords.concat))
        if existing is not None and tuple(existing[1]) != coords.new_inputs:
            return None
        rewrites_by_operator[id(coords.concat)] = (
            coords.concat,
            list(coords.new_inputs),
        )
    input_rewrites = tuple(
        _InputRewrite(
            operator=operator,
            original_inputs=tuple(str(value) for value in operator.inputs),
            new_inputs=tuple(new_inputs),
        )
        for operator, new_inputs in sorted(
            rewrites_by_operator.values(),
            key=lambda item: int(_operator_index(graph_index, item[0]) or 0),
        )
        if tuple(str(value) for value in operator.inputs) != tuple(new_inputs)
    )

    constant_updates = _constant_updates(
        model_ir,
        graph_index,
        size,
        offset,
    )
    if constant_updates is None:
        return None
    size_options = _reshape_options_update(
        size.reshape,
        size.converted_shape_values,
    )
    offset_options = _reshape_options_update(
        offset.reshape,
        offset.converted_shape_values,
    )
    if size_options is None or offset_options is None:
        return None

    metadata_updates = []
    for branch, names in (
        (center, center.output_names[:3]),
        (size, size.output_names[:3]),
    ):
        metadata_updates.extend(
            _MetadataUpdate(
                name=name,
                shape=tuple(int(value) for value in branch.root.source_view.shape),
                signature=tuple(
                    int(value) for value in branch.root.source_view.signature
                ),
                logical_layout=branch.root.source_logical_layout,
                physical_layout=branch.root.source_physical_layout,
            )
            for name in names
        )
    metadata_updates.extend(
        (
            _MetadataUpdate(
                name=size.output_names[3],
                shape=size.converted_shape,
                signature=size.converted_signature,
                logical_layout=LOGICAL_LAYOUT_NWC,
                physical_layout=LOGICAL_LAYOUT_NWC,
            ),
            _MetadataUpdate(
                name=offset.output_name,
                shape=offset.converted_shape,
                signature=offset.converted_signature,
                logical_layout=LOGICAL_LAYOUT_NWC,
                physical_layout=LOGICAL_LAYOUT_NWC,
            ),
        )
    )

    relevant_operators = [
        *branch_operators,
        size.coords.concat,
        size.coords.axis_reshape,
        offset.coords.concat,
        offset.coords.axis_reshape,
    ]
    relevant_operators = sorted(
        {id(operator): operator for operator in relevant_operators}.values(),
        key=lambda operator: int(_operator_index(graph_index, operator) or 0),
    )
    contract_names = set()
    for operator in relevant_operators:
        contract_names.update(str(value) for value in operator.inputs)
        contract_names.update(str(value) for value in operator.outputs)
    tensor_contracts = []
    for name in sorted(contract_names):
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))
    return _Plan(
        center=center,
        size=size,
        offset=offset,
        input_rewrites=input_rewrites,
        metadata_updates=tuple(metadata_updates),
        options_updates=(size_options, offset_options),
        constant_updates=constant_updates,
        removals=(center.root.pre, size.root.pre, offset.root.pre),
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in relevant_operators
        ),
        graph_inputs=tuple(str(value) for value in model_ir.inputs),
        graph_outputs=tuple(str(value) for value in model_ir.outputs),
    )


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.center.root.pre),
        id(plan.size.root.pre),
        id(plan.offset.root.pre),
        tuple(
            (id(rewrite.operator), rewrite.original_inputs, rewrite.new_inputs)
            for rewrite in plan.input_rewrites
        ),
        tuple(
            (
                update.name,
                update.shape,
                update.signature,
                update.logical_layout,
                update.physical_layout,
            )
            for update in plan.metadata_updates
        ),
        tuple(
            (
                id(update.operator),
                _freeze(update.original_options),
                _freeze(update.new_options),
            )
            for update in plan.options_updates
        ),
        tuple(
            (
                update.source_name,
                update.target_name,
                tuple(
                    (id(use.operator), int(use.input_slot))
                    for use in update.uses
                ),
                update.original_values,
                update.new_values,
                update.dtype,
                update.shape,
                update.signature,
                update.numpy_dtype,
                update.data_shape,
            )
            for update in plan.constant_updates
        ),
        tuple(id(operator) for operator in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
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
        plan.center.root.pre,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if (
        tuple(str(value) for value in model_ir.inputs) != plan.graph_inputs
        or tuple(str(value) for value in model_ir.outputs) != plan.graph_outputs
        or any(
            tuple(str(value) for value in rewrite.operator.inputs)
            != rewrite.original_inputs
            for rewrite in plan.input_rewrites
        )
        or any(
            _freeze(update.operator.options)
            != _freeze(update.original_options)
            for update in plan.options_updates
        )
        or any(
            update.target_name != update.source_name
            and update.target_name in model_ir.tensors
            for update in plan.constant_updates
        )
    ):
        return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))
    for update in plan.constant_updates:
        tensor = model_ir.tensors.get(update.source_name)
        if tensor is None or tensor.data is None:
            return False
        for use in update.uses:
            if (
                int(use.input_slot) < 0
                or int(use.input_slot) >= len(use.operator.inputs)
                or str(use.operator.inputs[int(use.input_slot)])
                != update.source_name
            ):
                return False

    for rewrite in plan.input_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    for update in plan.constant_updates:
        tensor = model_ir.tensors[update.source_name]
        array = np.asarray(
            update.new_values,
            dtype=np.dtype(update.numpy_dtype),
        ).reshape(update.data_shape)
        if update.target_name == update.source_name:
            tensor.data = array
            continue
        model_ir.tensors[update.target_name] = TensorIR(
            name=update.target_name,
            dtype=update.dtype,
            shape=[int(value) for value in update.shape],
            shape_signature=[int(value) for value in update.signature],
            data=array,
            is_variable=False,
            quantization=_clone_quantization(tensor.quantization),
            logical_layout=str(tensor.logical_layout),
            physical_layout=str(tensor.physical_layout),
            onnx_tensor_name=tensor.onnx_tensor_name,
        )
        for use in update.uses:
            inputs = [str(value) for value in use.operator.inputs]
            inputs[int(use.input_slot)] = update.target_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=use.operator,
                new_inputs=inputs,
                graph_index=graph_index,
            )
        if layout_state is not None:
            layout_state.set(
                update.target_name,
                logical=str(tensor.logical_layout),
                physical=str(tensor.physical_layout),
            )
    for update in plan.options_updates:
        update.operator.options = copy.deepcopy(update.new_options)
    graph_index.remove_operators(removal_indices)
    for update in plan.metadata_updates:
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
    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return True


def optimize_center_size_offset_terminal_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Fold one fully classified center/size/offset terminal head at a time."""

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
                {"TRANSPOSE"}
            )
        ]
    )
    rewrite_limit = (
        len(candidates) if max_rewrites is None else max(0, int(max_rewrites))
    )
    if rewrite_limit == 0:
        return {_STATS_KEY: 0}
    rewritten = 0
    for center_pre in candidates:
        if rewritten >= rewrite_limit:
            break
        if center_pre is None or _operator_index(active_index, center_pre) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            center_pre,
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
    return {_STATS_KEY: int(rewritten)}
