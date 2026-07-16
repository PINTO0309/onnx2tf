from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _replace_operator_input_at,
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
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    _PERM_NCHW_TO_NHWC,
    _PERM_NHWC_TO_NCHW,
    _View,
    _consumer_slots,
    _freeze,
    _layout_of,
    _op_type,
    _operator_contract,
    _operator_index,
    _per_tensor_quantization,
    _permuted_view,
    _resolved_source,
    _tensor_contract,
    _typed_constant,
    _typed_permutation,
    _view,
)


_STATS_KEY = "optimized_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"
_PERM_3D_SWAP = (0, 2, 1)


@dataclass(frozen=True)
class _MetadataUpdate:
    name: str
    old_view: _View
    new_view: _View


@dataclass(frozen=True)
class _ConstantUpdate:
    name: str
    original_values: Tuple[int, ...]
    target_values: Tuple[int, ...]
    dtype: str
    numpy_dtype: str
    data_contract: Any


@dataclass(frozen=True)
class _ShapeUpdate:
    source_name: str
    target_name: str
    in_place: bool
    original_values: Tuple[int, ...]
    target_values: Tuple[int, ...]
    dtype: str
    numpy_dtype: str
    data_contract: Any
    consumer_slots: Tuple[Tuple[OperatorIR, int], ...]


@dataclass(frozen=True)
class _SlicePlan:
    operator: OperatorIR
    original_inputs: Tuple[str, ...]
    source_name: str
    output: _MetadataUpdate
    begin: _ConstantUpdate
    size: _ConstantUpdate
    unary: Optional[OperatorIR]
    unary_output: Optional[_MetadataUpdate]


@dataclass(frozen=True)
class _BranchPlan:
    pre: OperatorIR
    source_name: str
    pre_output_name: str
    permutation_name: str
    slices: Tuple[_SlicePlan, ...]
    concat: OperatorIR
    concat_output: _MetadataUpdate
    reshape: OperatorIR
    reshape_original_inputs: Tuple[str, ...]
    reshape_output: _MetadataUpdate
    reshape_shape: _ShapeUpdate
    reshape_original_options: Any
    reshape_target_options: Any


@dataclass(frozen=True)
class _Plan:
    tail: OperatorIR
    original_tail_inputs: Tuple[str, ...]
    original_output_name: str
    canonical_output_name: str
    output_old_view: _View
    output_new_view: _View
    branches: Tuple[_BranchPlan, ...]
    post_permutation_name: str
    create_post_permutation: bool
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.tail),
        plan.original_tail_inputs,
        plan.original_output_name,
        plan.canonical_output_name,
        plan.output_old_view,
        plan.output_new_view,
        tuple(
            (
                id(branch.pre),
                branch.source_name,
                branch.pre_output_name,
                branch.permutation_name,
                tuple(
                    (
                        id(slice_plan.operator),
                        slice_plan.original_inputs,
                        slice_plan.source_name,
                        slice_plan.output,
                        slice_plan.begin,
                        slice_plan.size,
                        None if slice_plan.unary is None else id(slice_plan.unary),
                        slice_plan.unary_output,
                    )
                    for slice_plan in branch.slices
                ),
                id(branch.concat),
                branch.concat_output,
                id(branch.reshape),
                branch.reshape_original_inputs,
                branch.reshape_output,
                (
                    branch.reshape_shape.source_name,
                    branch.reshape_shape.target_name,
                    branch.reshape_shape.in_place,
                    branch.reshape_shape.original_values,
                    branch.reshape_shape.target_values,
                    branch.reshape_shape.dtype,
                    branch.reshape_shape.numpy_dtype,
                    branch.reshape_shape.data_contract,
                    tuple(
                        (id(operator), int(slot))
                        for operator, slot in branch.reshape_shape.consumer_slots
                    ),
                ),
                _freeze(branch.reshape_original_options),
                _freeze(branch.reshape_target_options),
            )
            for branch in plan.branches
        ),
        plan.post_permutation_name,
        plan.create_post_permutation,
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _rank(view: _View, expected: int) -> bool:
    return bool(
        len(view.shape) == int(expected)
        and len(view.signature) == int(expected)
        and all(int(value) > 0 for value in view.shape)
        and all(int(value) == -1 or int(value) > 0 for value in view.signature)
    )


def _layout_in(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
    allowed: set[str],
) -> bool:
    return str(_layout_of(str(name), tensor, layout_state)).upper() in allowed


def _same_quantization(left: TensorIR, right: TensorIR) -> bool:
    return bool(
        _per_tensor_quantization(left.quantization)
        and _per_tensor_quantization(right.quantization)
        and _freeze(left.quantization) == _freeze(right.quantization)
    )


def _normalized_axis(
    operator: OperatorIR,
    *,
    rank: int,
    default: int,
) -> Optional[int]:
    if not isinstance(operator.options, dict):
        return None
    try:
        axis = int(operator.options.get("axis", int(default)))
    except Exception:
        return None
    if axis < 0:
        axis += int(rank)
    return axis if axis in range(int(rank)) else None


def _unique_name(base: str, occupied: set[str]) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in occupied:
        candidate = f"{base}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _concat_view(
    views: Sequence[_View],
    *,
    axis: int,
) -> Optional[_View]:
    if len(views) < 2 or int(axis) < 0:
        return None
    rank = len(views[0].shape) if views else 0
    if int(axis) >= int(rank) or any(not _rank(view, rank) for view in views):
        return None
    dtype = views[0].dtype
    if any(view.dtype != dtype for view in views[1:]):
        return None

    shape = []
    signature = []
    for dimension in range(rank):
        static_values = [int(view.shape[dimension]) for view in views]
        dynamic_values = [int(view.signature[dimension]) for view in views]
        if dimension == int(axis):
            shape.append(sum(static_values))
            signature.append(
                sum(dynamic_values)
                if all(value >= 0 for value in dynamic_values)
                else -1
            )
            continue
        if len(set(static_values)) != 1:
            return None
        known_dynamic = {value for value in dynamic_values if value >= 0}
        if len(known_dynamic) > 1:
            return None
        shape.append(static_values[0])
        signature.append(
            next(iter(known_dynamic))
            if len(known_dynamic) == 1 and all(value >= 0 for value in dynamic_values)
            else -1
        )
    return _View(
        shape=tuple(shape),
        signature=tuple(signature),
        dtype=dtype,
    )


def _slice_view(
    source: _View,
    begin: Sequence[int],
    size: Sequence[int],
) -> Optional[_View]:
    if (
        not _rank(source, 4)
        or len(begin) != 4
        or len(size) != 4
        or any(int(value) < 0 for value in begin)
        or any(int(value) == 0 or int(value) < -1 for value in size)
    ):
        return None
    shape = []
    signature = []
    for dimension, (raw_begin, raw_size) in enumerate(zip(begin, size)):
        start = int(raw_begin)
        extent = int(raw_size)
        static_source = int(source.shape[dimension])
        dynamic_source = int(source.signature[dimension])
        if start >= static_source:
            return None
        static_extent = static_source - start if extent == -1 else extent
        if static_extent <= 0 or start + static_extent > static_source:
            return None
        shape.append(static_extent)
        if extent >= 0:
            signature.append(extent)
        elif dynamic_source >= 0:
            if start >= dynamic_source:
                return None
            signature.append(dynamic_source - start)
        else:
            signature.append(-1)
    return _View(
        shape=tuple(shape),
        signature=tuple(signature),
        dtype=source.dtype,
    )


def _reshape_static_shape(
    input_shape: Sequence[int],
    values: Sequence[int],
) -> Optional[Tuple[int, ...]]:
    normalized = tuple(int(value) for value in values)
    if (
        len(normalized) != 3
        or normalized.count(-1) > 1
        or any(value == 0 or value < -1 for value in normalized)
        or any(int(value) <= 0 for value in input_shape)
    ):
        return None
    input_elements = int(np.prod(input_shape, dtype=np.int64))
    known_elements = int(
        np.prod(
            [value for value in normalized if value > 0],
            dtype=np.int64,
        )
    )
    if known_elements <= 0:
        return None
    if -1 in normalized:
        if input_elements % known_elements != 0:
            return None
        inferred = input_elements // known_elements
        if inferred <= 0:
            return None
        return tuple(inferred if value == -1 else value for value in normalized)
    if known_elements != input_elements:
        return None
    return normalized


def _reshape_values_match(
    input_view: _View,
    output_view: _View,
    values: Sequence[int],
) -> bool:
    return bool(
        _rank(input_view, 4)
        and _rank(output_view, 3)
        and input_view.dtype == output_view.dtype
        and _reshape_static_shape(input_view.shape, values) == output_view.shape
    )


def _exclusive_slot(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    operator: OperatorIR,
    slot: int,
) -> bool:
    return _consumer_slots(model_ir, graph_index, str(name)) == ((operator, int(slot)),)


def _constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    operator: OperatorIR,
    slot: int,
) -> Optional[_ConstantUpdate]:
    resolved = _typed_constant(
        model_ir,
        graph_index,
        str(name),
        shape=(4,),
    )
    if resolved is None or not _exclusive_slot(
        model_ir,
        graph_index,
        str(name),
        operator,
        int(slot),
    ):
        return None
    tensor, data = resolved
    values = tuple(int(value) for value in data.reshape(-1))
    target = tuple(int(values[index]) for index in _PERM_NCHW_TO_NHWC)
    return _ConstantUpdate(
        name=str(name),
        original_values=values,
        target_values=target,
        dtype=str(tensor.dtype),
        numpy_dtype=str(data.dtype),
        data_contract=_freeze(data),
    )


def _shape_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    reshape: OperatorIR,
    input_old_view: _View,
    output_old_view: _View,
    input_new_view: _View,
    output_new_view: _View,
    occupied: set[str],
) -> Optional[_ShapeUpdate]:
    if len(reshape.inputs) != 2:
        return None
    source_name = str(reshape.inputs[1])
    resolved = _typed_constant(
        model_ir,
        graph_index,
        source_name,
        shape=(3,),
    )
    if resolved is None:
        return None
    tensor, data = resolved
    original_values = tuple(int(value) for value in data.reshape(-1))
    target_values = tuple(int(original_values[index]) for index in _PERM_3D_SWAP)
    if not _reshape_values_match(
        input_old_view,
        output_old_view,
        original_values,
    ) or not _reshape_values_match(
        input_new_view,
        output_new_view,
        target_values,
    ):
        return None
    slots = _consumer_slots(model_ir, graph_index, source_name)
    if len(slots) == 0:
        return None
    in_place = slots == ((reshape, 1),)
    target_name = (
        source_name if in_place else _unique_name(f"{source_name}_nhwc", occupied)
    )
    return _ShapeUpdate(
        source_name=source_name,
        target_name=target_name,
        in_place=in_place,
        original_values=original_values,
        target_values=target_values,
        dtype=str(tensor.dtype),
        numpy_dtype=str(data.dtype),
        data_contract=_freeze(data),
        consumer_slots=slots,
    )


def _target_reshape_options(
    reshape: OperatorIR,
    *,
    input_old_view: _View,
    output_old_view: _View,
    input_new_view: _View,
    output_new_view: _View,
) -> Optional[Dict[str, Any]]:
    if not isinstance(reshape.options, dict):
        return None
    options = dict(reshape.options)
    try:
        allow_zero = bool(options.get("allowZero", False))
    except Exception:
        return None
    changed = False
    for key in ("newShape", "onnxRawNewShape"):
        if key not in options:
            continue
        value = options[key]
        if not isinstance(value, list) or len(value) != 3:
            return None
        try:
            original = tuple(int(item) for item in value)
        except Exception:
            return None
        if allow_zero and 0 in original:
            return None
        target = tuple(int(original[index]) for index in _PERM_3D_SWAP)
        if not _reshape_values_match(
            input_old_view,
            output_old_view,
            original,
        ) or not _reshape_values_match(
            input_new_view,
            output_new_view,
            target,
        ):
            return None
        options[key] = list(target)
        changed = changed or target != original
    return options if changed or options == reshape.options else dict(options)


def _resolve_slice_path(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    branch_concat: OperatorIR,
    branch_concat_index: int,
    branch_input_name: str,
    branch_input_slot: int,
    layout_state: Optional[LayoutState],
) -> Optional[Tuple[_SlicePlan, OperatorIR, str, str, str]]:
    public_names = {
        *(str(value) for value in model_ir.inputs),
        *(str(value) for value in model_ir.outputs),
    }
    input_name = str(branch_input_name)
    if input_name in graph_index.duplicate_producers:
        return None
    producer_index = graph_index.producers.get(input_name)
    if producer_index is None or int(producer_index) >= int(branch_concat_index):
        return None
    producer = model_ir.operators[int(producer_index)]
    unary: Optional[OperatorIR] = None
    unary_output: Optional[_MetadataUpdate] = None
    slice_output_name = input_name
    if _op_type(producer) == "LOGISTIC":
        if (
            len(producer.inputs) != 1
            or len(producer.outputs) != 1
            or input_name in public_names
            or graph_index.producer(input_name) is not producer
            or not _exclusive_slot(
                model_ir,
                graph_index,
                input_name,
                branch_concat,
                int(branch_input_slot),
            )
        ):
            return None
        unary = producer
        slice_output_name = str(producer.inputs[0])
        if slice_output_name in graph_index.duplicate_producers:
            return None
        producer_index = graph_index.producers.get(slice_output_name)
        if producer_index is None or int(producer_index) >= int(
            _operator_index(graph_index, unary) or -1
        ):
            return None
        producer = model_ir.operators[int(producer_index)]

    if (
        _op_type(producer) != "SLICE"
        or len(producer.inputs) != 3
        or len(producer.outputs) != 1
        or str(producer.outputs[0]) != slice_output_name
        or slice_output_name in public_names
        or graph_index.producer(slice_output_name) is not producer
    ):
        return None
    expected_consumer = unary if unary is not None else branch_concat
    expected_slot = 0 if unary is not None else int(branch_input_slot)
    if not _exclusive_slot(
        model_ir,
        graph_index,
        slice_output_name,
        expected_consumer,
        expected_slot,
    ):
        return None

    slice_index = _operator_index(graph_index, producer)
    if slice_index is None:
        return None
    pre_output_name = str(producer.inputs[0])
    if pre_output_name in graph_index.duplicate_producers:
        return None
    pre_index = graph_index.producers.get(pre_output_name)
    if pre_index is None or int(pre_index) >= int(slice_index):
        return None
    pre = model_ir.operators[int(pre_index)]
    if (
        not _typed_permutation(
            model_ir,
            graph_index,
            pre,
            _PERM_NHWC_TO_NCHW,
        )
        or str(pre.outputs[0]) != pre_output_name
    ):
        return None
    source_name = str(pre.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    pre_output_tensor = model_ir.tensors.get(pre_output_name)
    slice_output_tensor = model_ir.tensors.get(slice_output_name)
    if (
        source_tensor is None
        or pre_output_tensor is None
        or slice_output_tensor is None
        or pre_output_name in public_names
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(pre_index),
        )
        or not _rank(_view(source_tensor), 4)
        or not _rank(_view(pre_output_tensor), 4)
        or not _rank(_view(slice_output_tensor), 4)
        or _permuted_view(_view(source_tensor), _PERM_NHWC_TO_NCHW)
        != _view(pre_output_tensor)
        or not _same_quantization(source_tensor, pre_output_tensor)
        or not _same_quantization(pre_output_tensor, slice_output_tensor)
        or not _layout_in(
            source_name,
            source_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not _layout_in(
            pre_output_name,
            pre_output_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not _layout_in(
            slice_output_name,
            slice_output_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    begin = _constant_update(
        model_ir,
        graph_index,
        name=str(producer.inputs[1]),
        operator=producer,
        slot=1,
    )
    size = _constant_update(
        model_ir,
        graph_index,
        name=str(producer.inputs[2]),
        operator=producer,
        slot=2,
    )
    if (
        begin is None
        or size is None
        or begin.name == size.name
        or begin.original_values[0] != 0
        or begin.original_values[2] != 0
        or begin.original_values[3] != 0
        or begin.original_values[1] < 0
        or size.original_values[1] <= 0
        or _slice_view(
            _view(pre_output_tensor),
            begin.original_values,
            size.original_values,
        )
        != _view(slice_output_tensor)
    ):
        return None
    new_slice_view = _permuted_view(
        _view(slice_output_tensor),
        _PERM_NCHW_TO_NHWC,
    )
    if (
        new_slice_view is None
        or _slice_view(
            _view(source_tensor),
            begin.target_values,
            size.target_values,
        )
        != new_slice_view
    ):
        return None

    if unary is not None:
        unary_tensor = model_ir.tensors.get(input_name)
        if (
            unary_tensor is None
            or not _rank(_view(unary_tensor), 4)
            or _view(unary_tensor).shape != _view(slice_output_tensor).shape
            or _view(unary_tensor).signature != _view(slice_output_tensor).signature
            or _view(unary_tensor).dtype != _view(slice_output_tensor).dtype
            or not _per_tensor_quantization(unary_tensor.quantization)
            or not _layout_in(
                input_name,
                unary_tensor,
                layout_state,
                {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            return None
        new_unary_view = _permuted_view(
            _view(unary_tensor),
            _PERM_NCHW_TO_NHWC,
        )
        if new_unary_view is None:
            return None
        unary_output = _MetadataUpdate(
            name=input_name,
            old_view=_view(unary_tensor),
            new_view=new_unary_view,
        )

    return (
        _SlicePlan(
            operator=producer,
            original_inputs=tuple(str(value) for value in producer.inputs),
            source_name=source_name,
            output=_MetadataUpdate(
                name=slice_output_name,
                old_view=_view(slice_output_tensor),
                new_view=new_slice_view,
            ),
            begin=begin,
            size=size,
            unary=unary,
            unary_output=unary_output,
        ),
        pre,
        source_name,
        pre_output_name,
        str(pre.inputs[1]),
    )


def _resolve_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    tail: OperatorIR,
    tail_index: int,
    reshape_output_name: str,
    tail_input_slot: int,
    layout_state: Optional[LayoutState],
    occupied: set[str],
) -> Optional[_BranchPlan]:
    public_names = {
        *(str(value) for value in model_ir.inputs),
        *(str(value) for value in model_ir.outputs),
    }
    if reshape_output_name in graph_index.duplicate_producers:
        return None
    reshape_index = graph_index.producers.get(str(reshape_output_name))
    if reshape_index is None or int(reshape_index) >= int(tail_index):
        return None
    reshape = model_ir.operators[int(reshape_index)]
    reshape_output_tensor = model_ir.tensors.get(str(reshape_output_name))
    if (
        _op_type(reshape) != "RESHAPE"
        or len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.outputs[0]) != str(reshape_output_name)
        or reshape_output_name in public_names
        or graph_index.producer(str(reshape_output_name)) is not reshape
        or not _exclusive_slot(
            model_ir,
            graph_index,
            str(reshape_output_name),
            tail,
            int(tail_input_slot),
        )
        or reshape_output_tensor is None
        or not _rank(_view(reshape_output_tensor), 3)
        or not _per_tensor_quantization(reshape_output_tensor.quantization)
        or not _layout_in(
            str(reshape_output_name),
            reshape_output_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    branch_concat_output_name = str(reshape.inputs[0])
    if branch_concat_output_name in graph_index.duplicate_producers:
        return None
    branch_concat_index = graph_index.producers.get(branch_concat_output_name)
    if branch_concat_index is None or int(branch_concat_index) >= int(reshape_index):
        return None
    branch_concat = model_ir.operators[int(branch_concat_index)]
    branch_concat_output_tensor = model_ir.tensors.get(branch_concat_output_name)
    branch_inputs = tuple(str(value) for value in branch_concat.inputs)
    if (
        _op_type(branch_concat) != "CONCATENATION"
        or len(branch_inputs) != 2
        or len(set(branch_inputs)) != 2
        or len(branch_concat.outputs) != 1
        or str(branch_concat.outputs[0]) != branch_concat_output_name
        or _normalized_axis(branch_concat, rank=4, default=1) != 1
        or branch_concat_output_name in public_names
        or graph_index.producer(branch_concat_output_name) is not branch_concat
        or not _exclusive_slot(
            model_ir,
            graph_index,
            branch_concat_output_name,
            reshape,
            0,
        )
        or branch_concat_output_tensor is None
        or not _rank(_view(branch_concat_output_tensor), 4)
        or not _per_tensor_quantization(branch_concat_output_tensor.quantization)
        or not _layout_in(
            branch_concat_output_name,
            branch_concat_output_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    resolved_paths = []
    for input_slot, input_name in enumerate(branch_inputs):
        resolved = _resolve_slice_path(
            model_ir,
            graph_index,
            branch_concat=branch_concat,
            branch_concat_index=int(branch_concat_index),
            branch_input_name=input_name,
            branch_input_slot=int(input_slot),
            layout_state=layout_state,
        )
        if resolved is None:
            return None
        resolved_paths.append(resolved)
    slices = tuple(value[0] for value in resolved_paths)
    pre = resolved_paths[0][1]
    source_name = resolved_paths[0][2]
    pre_output_name = resolved_paths[0][3]
    permutation_name = resolved_paths[0][4]
    if (
        any(value[1] is not pre for value in resolved_paths[1:])
        or any(value[2] != source_name for value in resolved_paths[1:])
        or any(value[3] != pre_output_name for value in resolved_paths[1:])
        or any(value[4] != permutation_name for value in resolved_paths[1:])
        or len({id(value.operator) for value in slices}) != len(slices)
        or len(
            {update.name for value in slices for update in (value.begin, value.size)}
        )
        != 2 * len(slices)
    ):
        return None
    pre_slots = _consumer_slots(model_ir, graph_index, pre_output_name)
    expected_pre_slots = tuple((slice_plan.operator, 0) for slice_plan in slices)
    if sorted((id(operator), slot) for operator, slot in pre_slots) != sorted(
        (id(operator), slot) for operator, slot in expected_pre_slots
    ):
        return None

    concat_input_tensors = [model_ir.tensors[name] for name in branch_inputs]
    expected_concat = _concat_view(
        [_view(tensor) for tensor in concat_input_tensors],
        axis=1,
    )
    if (
        expected_concat is None
        or expected_concat != _view(branch_concat_output_tensor)
        or any(
            not _same_quantization(tensor, branch_concat_output_tensor)
            for tensor in concat_input_tensors
        )
        or not _same_quantization(
            branch_concat_output_tensor,
            reshape_output_tensor,
        )
    ):
        return None
    concat_new_view = _permuted_view(
        _view(branch_concat_output_tensor),
        _PERM_NCHW_TO_NHWC,
    )
    reshape_new_view = _permuted_view(
        _view(reshape_output_tensor),
        _PERM_3D_SWAP,
    )
    if concat_new_view is None or reshape_new_view is None:
        return None
    expected_new_concat = _concat_view(
        [
            (
                slice_plan.unary_output.new_view
                if slice_plan.unary_output is not None
                else slice_plan.output.new_view
            )
            for slice_plan in slices
        ],
        axis=3,
    )
    if expected_new_concat != concat_new_view:
        return None

    shape_update = _shape_update(
        model_ir,
        graph_index,
        reshape=reshape,
        input_old_view=_view(branch_concat_output_tensor),
        output_old_view=_view(reshape_output_tensor),
        input_new_view=concat_new_view,
        output_new_view=reshape_new_view,
        occupied=occupied,
    )
    target_options = _target_reshape_options(
        reshape,
        input_old_view=_view(branch_concat_output_tensor),
        output_old_view=_view(reshape_output_tensor),
        input_new_view=concat_new_view,
        output_new_view=reshape_new_view,
    )
    if shape_update is None or target_options is None:
        return None
    return _BranchPlan(
        pre=pre,
        source_name=source_name,
        pre_output_name=pre_output_name,
        permutation_name=permutation_name,
        slices=slices,
        concat=branch_concat,
        concat_output=_MetadataUpdate(
            name=branch_concat_output_name,
            old_view=_view(branch_concat_output_tensor),
            new_view=concat_new_view,
        ),
        reshape=reshape,
        reshape_original_inputs=tuple(str(value) for value in reshape.inputs),
        reshape_output=_MetadataUpdate(
            name=str(reshape_output_name),
            old_view=_view(reshape_output_tensor),
            new_view=reshape_new_view,
        ),
        reshape_shape=shape_update,
        reshape_original_options=_freeze(reshape.options),
        reshape_target_options=target_options,
    )


def _find_post_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[str]:
    for name in model_ir.tensors:
        resolved = _typed_constant(
            model_ir,
            graph_index,
            str(name),
            shape=(3,),
        )
        if resolved is None:
            continue
        tensor, data = resolved
        if (
            str(tensor.dtype) == "INT32"
            and data.dtype == np.dtype(np.int32)
            and tuple(int(value) for value in data.reshape(-1)) == _PERM_3D_SWAP
        ):
            return str(name)
    return None


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tail: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    tail_index = _operator_index(graph_index, tail)
    tail_inputs = tuple(str(value) for value in tail.inputs)
    if (
        tail_index is None
        or _op_type(tail) != "CONCATENATION"
        or len(tail_inputs) < 2
        or len(set(tail_inputs)) != len(tail_inputs)
        or len(tail.outputs) != 1
        or _normalized_axis(tail, rank=3, default=2) != 2
    ):
        return None
    output_name = str(tail.outputs[0])
    output_tensor = model_ir.tensors.get(output_name)
    if (
        output_tensor is None
        or output_name in {str(value) for value in model_ir.inputs}
        or output_name in graph_index.duplicate_producers
        or graph_index.producer(output_name) is not tail
        or not _rank(_view(output_tensor), 3)
        or not _per_tensor_quantization(output_tensor.quantization)
        or not _layout_in(
            output_name,
            output_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
        or any(
            int(consumer_index) <= int(tail_index)
            for consumer_index in graph_index.consumer_indices(output_name)
        )
    ):
        return None

    occupied = {str(name) for name in model_ir.tensors}
    branches = []
    for input_slot, input_name in enumerate(tail_inputs):
        branch = _resolve_branch(
            model_ir,
            graph_index,
            tail=tail,
            tail_index=int(tail_index),
            reshape_output_name=input_name,
            tail_input_slot=int(input_slot),
            layout_state=layout_state,
            occupied=occupied,
        )
        if branch is None:
            return None
        branches.append(branch)
    if (
        len({id(branch.pre) for branch in branches}) != len(branches)
        or len({branch.pre_output_name for branch in branches}) != len(branches)
        or len({id(branch.concat) for branch in branches}) != len(branches)
        or len({id(branch.reshape) for branch in branches}) != len(branches)
    ):
        return None

    old_views = [branch.reshape_output.old_view for branch in branches]
    new_views = [branch.reshape_output.new_view for branch in branches]
    expected_old = _concat_view(old_views, axis=2)
    output_new_view = _permuted_view(_view(output_tensor), _PERM_3D_SWAP)
    expected_new = _concat_view(new_views, axis=1)
    if (
        expected_old is None
        or expected_old != _view(output_tensor)
        or output_new_view is None
        or expected_new != output_new_view
        or any(
            not _same_quantization(
                model_ir.tensors[branch.reshape_output.name],
                output_tensor,
            )
            for branch in branches
        )
    ):
        return None

    post_permutation_name = _find_post_permutation(model_ir, graph_index)
    create_post_permutation = post_permutation_name is None
    if post_permutation_name is None:
        post_permutation_name = _unique_name(
            "transpose_tail_3d_nhwc_to_nchw_perm",
            occupied,
        )
    canonical_output_name = _unique_name(f"{output_name}_nhwc", occupied)

    operators: Dict[int, OperatorIR] = {id(tail): tail}
    names = {output_name}
    for branch in branches:
        operators[id(branch.pre)] = branch.pre
        operators[id(branch.concat)] = branch.concat
        operators[id(branch.reshape)] = branch.reshape
        names.update(
            {
                branch.source_name,
                branch.pre_output_name,
                branch.permutation_name,
                branch.concat_output.name,
                branch.reshape_output.name,
                branch.reshape_shape.source_name,
            }
        )
        for slice_plan in branch.slices:
            operators[id(slice_plan.operator)] = slice_plan.operator
            if slice_plan.unary is not None:
                operators[id(slice_plan.unary)] = slice_plan.unary
            names.update(
                {
                    slice_plan.output.name,
                    slice_plan.begin.name,
                    slice_plan.size.name,
                }
            )
            if slice_plan.unary_output is not None:
                names.add(slice_plan.unary_output.name)
        source_producer = graph_index.producer(branch.source_name)
        if source_producer is not None:
            operators[id(source_producer)] = source_producer
        for operator, _ in branch.reshape_shape.consumer_slots:
            operators[id(operator)] = operator
    for operator, _ in _consumer_slots(model_ir, graph_index, output_name):
        operators[id(operator)] = operator
    if not create_post_permutation:
        names.add(str(post_permutation_name))
    if any(name not in model_ir.tensors for name in names):
        return None
    ordered_operators = sorted(
        operators.values(),
        key=lambda operator: int(_operator_index(graph_index, operator) or 0),
    )
    return _Plan(
        tail=tail,
        original_tail_inputs=tail_inputs,
        original_output_name=output_name,
        canonical_output_name=canonical_output_name,
        output_old_view=_view(output_tensor),
        output_new_view=output_new_view,
        branches=tuple(branches),
        post_permutation_name=str(post_permutation_name),
        create_post_permutation=create_post_permutation,
        tensor_contracts=tuple(
            _tensor_contract(name, model_ir.tensors[name])
            for name in sorted(names)
            if name in model_ir.tensors
        ),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in ordered_operators
        ),
        graph_inputs=tuple(str(value) for value in model_ir.inputs),
        graph_outputs=tuple(str(value) for value in model_ir.outputs),
    )


def _set_metadata(
    model_ir: ModelIR,
    update: _MetadataUpdate,
    *,
    layout_state: Optional[LayoutState],
    layout: str,
) -> None:
    tensor = model_ir.tensors[update.name]
    tensor.shape = [int(value) for value in update.new_view.shape]
    tensor.shape_signature = [int(value) for value in update.new_view.signature]
    tensor.logical_layout = str(layout)
    tensor.physical_layout = str(layout)
    if layout_state is not None:
        layout_state.set(update.name, logical=layout, physical=layout)


def _write_constant(
    model_ir: ModelIR,
    update: _ConstantUpdate,
) -> None:
    tensor = model_ir.tensors[update.name]
    data = np.asarray(tensor.data)
    tensor.data = np.asarray(update.target_values, dtype=data.dtype).reshape(data.shape)


def _apply_shape_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    branch: _BranchPlan,
    *,
    layout_state: Optional[LayoutState],
) -> None:
    update = branch.reshape_shape
    source = model_ir.tensors[update.source_name]
    data = np.asarray(source.data)
    target_data = np.asarray(update.target_values, dtype=data.dtype).reshape(data.shape)
    if update.in_place:
        source.data = target_data
        return
    model_ir.tensors[update.target_name] = TensorIR(
        name=update.target_name,
        dtype=str(source.dtype),
        shape=[int(value) for value in source.shape],
        shape_signature=(
            None
            if source.shape_signature is None
            else [int(value) for value in source.shape_signature]
        ),
        data=target_data,
        is_variable=False,
        quantization=_clone_quantization(source.quantization),
    )
    if layout_state is not None:
        layout_state.set(
            update.target_name,
            logical=LOGICAL_LAYOUT_UNKNOWN,
            physical=LOGICAL_LAYOUT_UNKNOWN,
        )
    _set_operator_inputs(
        model_ir=model_ir,
        op=branch.reshape,
        new_inputs=[branch.reshape_original_inputs[0], update.target_name],
        graph_index=graph_index,
    )


def _create_post_permutation(
    model_ir: ModelIR,
    name: str,
    layout_state: Optional[LayoutState],
) -> None:
    model_ir.tensors[str(name)] = TensorIR(
        name=str(name),
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray(_PERM_3D_SWAP, dtype=np.int32),
        is_variable=False,
    )
    if layout_state is not None:
        layout_state.set(
            str(name),
            logical=LOGICAL_LAYOUT_UNKNOWN,
            physical=LOGICAL_LAYOUT_UNKNOWN,
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
        plan.tail,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if (
        tuple(str(value) for value in model_ir.inputs) != plan.graph_inputs
        or tuple(str(value) for value in model_ir.outputs) != plan.graph_outputs
    ):
        return False

    for branch in plan.branches:
        index = _operator_index(graph_index, branch.pre)
        if index is None:
            return False
        for update in (
            *(value.begin for value in branch.slices),
            *(value.size for value in branch.slices),
        ):
            tensor = model_ir.tensors[update.name]
            data = np.asarray(tensor.data)
            if (
                str(tensor.dtype) != update.dtype
                or str(data.dtype) != update.numpy_dtype
                or _freeze(data) != update.data_contract
                or tuple(int(value) for value in data.reshape(-1))
                != update.original_values
            ):
                return False
        shape_update = branch.reshape_shape
        shape_tensor = model_ir.tensors[shape_update.source_name]
        shape_data = np.asarray(shape_tensor.data)
        if (
            str(shape_tensor.dtype) != shape_update.dtype
            or str(shape_data.dtype) != shape_update.numpy_dtype
            or _freeze(shape_data) != shape_update.data_contract
            or tuple(int(value) for value in shape_data.reshape(-1))
            != shape_update.original_values
        ):
            return False

    for branch in plan.branches:
        for slice_plan in branch.slices:
            _write_constant(model_ir, slice_plan.begin)
            _write_constant(model_ir, slice_plan.size)
            _replace_operator_input_at(
                model_ir=model_ir,
                op=slice_plan.operator,
                input_index=0,
                new_input_name=slice_plan.source_name,
                graph_index=graph_index,
            )
            _set_metadata(
                model_ir,
                slice_plan.output,
                layout_state=layout_state,
                layout=LOGICAL_LAYOUT_NHWC,
            )
            if slice_plan.unary_output is not None:
                _set_metadata(
                    model_ir,
                    slice_plan.unary_output,
                    layout_state=layout_state,
                    layout=LOGICAL_LAYOUT_NHWC,
                )

        concat_options = dict(branch.concat.options)
        concat_options["axis"] = 3
        branch.concat.options = concat_options
        _set_metadata(
            model_ir,
            branch.concat_output,
            layout_state=layout_state,
            layout=LOGICAL_LAYOUT_NHWC,
        )
        _apply_shape_update(
            model_ir,
            graph_index,
            branch,
            layout_state=layout_state,
        )
        branch.reshape.options = dict(branch.reshape_target_options)
        _set_metadata(
            model_ir,
            branch.reshape_output,
            layout_state=layout_state,
            layout=LOGICAL_LAYOUT_NHWC,
        )
        source_tensor = model_ir.tensors[branch.source_name]
        source_tensor.logical_layout = LOGICAL_LAYOUT_NHWC
        source_tensor.physical_layout = LOGICAL_LAYOUT_NHWC
        if layout_state is not None:
            layout_state.set(
                branch.source_name,
                logical=LOGICAL_LAYOUT_NHWC,
                physical=LOGICAL_LAYOUT_NHWC,
            )

    output_tensor = model_ir.tensors[plan.original_output_name]
    model_ir.tensors[plan.canonical_output_name] = TensorIR(
        name=plan.canonical_output_name,
        dtype=str(output_tensor.dtype),
        shape=[int(value) for value in plan.output_new_view.shape],
        shape_signature=[int(value) for value in plan.output_new_view.signature],
        data=None,
        is_variable=False,
        quantization=_clone_quantization(output_tensor.quantization),
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
    )
    if layout_state is not None:
        layout_state.set(
            plan.canonical_output_name,
            logical=LOGICAL_LAYOUT_NHWC,
            physical=LOGICAL_LAYOUT_NHWC,
        )
    tail_options = dict(plan.tail.options)
    tail_options["axis"] = 1
    plan.tail.options = tail_options
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.tail,
        new_outputs=[plan.canonical_output_name],
        graph_index=graph_index,
    )
    _set_metadata(
        model_ir,
        _MetadataUpdate(
            name=plan.canonical_output_name,
            old_view=plan.output_new_view,
            new_view=plan.output_new_view,
        ),
        layout_state=layout_state,
        layout=LOGICAL_LAYOUT_NHWC,
    )
    output_tensor.logical_layout = LOGICAL_LAYOUT_NCHW
    output_tensor.physical_layout = LOGICAL_LAYOUT_NCHW
    if layout_state is not None:
        layout_state.set(
            plan.original_output_name,
            logical=LOGICAL_LAYOUT_NCHW,
            physical=LOGICAL_LAYOUT_NCHW,
        )

    if plan.create_post_permutation:
        _create_post_permutation(
            model_ir,
            plan.post_permutation_name,
            layout_state,
        )
    tail_index = _operator_index(graph_index, plan.tail)
    if tail_index is None:
        raise RuntimeError("validated tail Concat disappeared during indexed apply")
    graph_index.insert_operator(
        int(tail_index) + 1,
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[plan.canonical_output_name, plan.post_permutation_name],
            outputs=[plan.original_output_name],
            options={},
        ),
    )
    removal_indices = []
    for branch in plan.branches:
        index = _operator_index(graph_index, branch.pre)
        if index is None:
            raise RuntimeError(
                "validated pre-Transpose disappeared during indexed apply"
            )
        removal_indices.append(int(index))
    graph_index.remove_operators(removal_indices)
    return True


def optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Lift a closed Slice/Logistic/Concat/Reshape detection tail to NHWC."""

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
    rewritten = 0
    for tail in candidates:
        if rewritten >= rewrite_limit:
            break
        if tail is None or _operator_index(active_index, tail) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            tail,
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

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {_STATS_KEY: int(rewritten)}
