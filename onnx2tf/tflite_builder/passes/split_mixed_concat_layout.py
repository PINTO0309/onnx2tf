from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _replace_tensor_inputs,
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


_STATS_KEY = (
    "optimized_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains"
)


@dataclass(frozen=True)
class _AxisUpdate:
    source_name: str
    target_name: str
    in_place: bool
    dtype: str
    numpy_dtype: str
    data_contract: Any


@dataclass(frozen=True)
class _MetadataUpdate:
    name: str
    old_view: _View
    new_view: _View


@dataclass(frozen=True)
class _AliasRewrite:
    source_name: str
    target_name: str
    adapter: OperatorIR


@dataclass(frozen=True)
class _SplitPlan:
    operator: OperatorIR
    axis: _AxisUpdate
    input_name: str
    canonical_input_name: str
    input_old_view: _View
    input_new_view: _View
    outputs: Tuple[_MetadataUpdate, ...]
    aliases: Tuple[_AliasRewrite, ...]
    removals: Tuple[OperatorIR, ...]


@dataclass(frozen=True)
class _Plan:
    concat: OperatorIR
    original_concat_inputs: Tuple[str, ...]
    canonical_concat_inputs: Tuple[str, ...]
    original_concat_output: str
    canonical_concat_output: str
    concat_old_view: _View
    concat_new_view: _View
    direct_sources: Tuple[str, ...]
    direct_removals: Tuple[OperatorIR, ...]
    splits: Tuple[_SplitPlan, ...]
    post_permutation_name: str
    create_post_permutation: bool
    pre_permutation_name: str
    create_pre_permutation: bool
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.concat),
        plan.original_concat_inputs,
        plan.canonical_concat_inputs,
        plan.original_concat_output,
        plan.canonical_concat_output,
        plan.concat_old_view,
        plan.concat_new_view,
        plan.direct_sources,
        tuple(id(operator) for operator in plan.direct_removals),
        tuple(
            (
                id(split.operator),
                (
                    split.axis.source_name,
                    split.axis.target_name,
                    split.axis.in_place,
                    split.axis.dtype,
                    split.axis.numpy_dtype,
                    split.axis.data_contract,
                ),
                split.input_name,
                split.canonical_input_name,
                split.input_old_view,
                split.input_new_view,
                tuple(
                    (update.name, update.old_view, update.new_view)
                    for update in split.outputs
                ),
                tuple(
                    (alias.source_name, alias.target_name, id(alias.adapter))
                    for alias in split.aliases
                ),
                tuple(id(operator) for operator in split.removals),
            )
            for split in plan.splits
        ),
        plan.post_permutation_name,
        plan.create_post_permutation,
        plan.pre_permutation_name,
        plan.create_pre_permutation,
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _rank4(view: _View) -> bool:
    return bool(len(view.shape) == 4 and len(view.signature) == 4)


def _layout_in(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
    allowed: set[str],
) -> bool:
    return str(_layout_of(str(name), tensor, layout_state)).upper() in allowed


def _compatible_nonchannel(left: _View, right: _View) -> bool:
    if left.dtype != right.dtype:
        return False
    for dimension in (0, 1, 2):
        if int(left.shape[dimension]) != int(right.shape[dimension]):
            return False
        left_signature = int(left.signature[dimension])
        right_signature = int(right.signature[dimension])
        if (
            left_signature >= 0
            and right_signature >= 0
            and left_signature != right_signature
        ):
            return False
    return True


def _concat_view(views: Sequence[_View]) -> Optional[_View]:
    if len(views) < 2 or any(not _rank4(view) for view in views):
        return None
    dtype = views[0].dtype
    if any(view.dtype != dtype for view in views[1:]):
        return None
    shape = []
    signature = []
    for dimension in range(4):
        static_values = [int(view.shape[dimension]) for view in views]
        dynamic_values = [int(view.signature[dimension]) for view in views]
        if dimension == 3:
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


def _normalized_axis(operator: OperatorIR) -> Optional[int]:
    if not isinstance(operator.options, dict):
        return None
    try:
        axis = int(operator.options.get("axis", 1))
    except Exception:
        return None
    if axis < 0:
        axis += 4
    return axis if axis in range(4) else None


def _unique_name(base: str, occupied: set[str]) -> str:
    for suffix in range(len(occupied) + 2):
        candidate = str(base) if suffix == 0 else f"{base}_{suffix}"
        if candidate not in occupied:
            occupied.add(candidate)
            return candidate
    raise RuntimeError(f"Unable to allocate ModelIR tensor name: {base}")


def _deduplicate_operators(
    operators: Sequence[OperatorIR],
) -> Tuple[OperatorIR, ...]:
    seen: set[int] = set()
    result = []
    for operator in operators:
        if id(operator) in seen:
            continue
        seen.add(id(operator))
        result.append(operator)
    return tuple(result)


def _find_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    expected: Sequence[int],
) -> Optional[str]:
    target = tuple(int(value) for value in expected)
    for name in model_ir.tensors:
        resolved = _typed_constant(
            model_ir,
            graph_index,
            str(name),
            shape=(4,),
        )
        if resolved is None:
            continue
        if tuple(int(value) for value in resolved[1].reshape(-1)) == target:
            return str(name)
    return None


def _axis_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    split: OperatorIR,
    *,
    occupied: set[str],
) -> Optional[_AxisUpdate]:
    axis_name = str(split.inputs[0])
    resolved = _typed_constant(
        model_ir,
        graph_index,
        axis_name,
        shape=(1,),
    )
    if resolved is None:
        return None
    tensor, data = resolved
    values = tuple(int(value) for value in data.reshape(-1))
    if len(values) != 1:
        return None
    axis = int(values[0])
    if axis < 0:
        axis += 4
    if axis != 1:
        return None
    slots = _consumer_slots(model_ir, graph_index, axis_name)
    in_place = bool(len(slots) == 1 and slots[0][0] is split and int(slots[0][1]) == 0)
    return _AxisUpdate(
        source_name=axis_name,
        target_name=(
            axis_name if in_place else _unique_name(f"{axis_name}_nhwc", occupied)
        ),
        in_place=in_place,
        dtype=str(tensor.dtype),
        numpy_dtype=str(data.dtype),
        data_contract=_freeze(data),
    )


def _resolve_split(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    split: OperatorIR,
    concat: OperatorIR,
    *,
    occupied: set[str],
    layout_state: Optional[LayoutState],
) -> Optional[_SplitPlan]:
    split_index = _operator_index(graph_index, split)
    concat_index = _operator_index(graph_index, concat)
    if (
        split_index is None
        or concat_index is None
        or int(split_index) >= int(concat_index)
        or _op_type(split) != "SPLIT"
        or len(split.inputs) < 2
        or len(split.outputs) == 0
        or len(set(str(value) for value in split.outputs)) != len(split.outputs)
    ):
        return None

    axis = _axis_update(
        model_ir,
        graph_index,
        split,
        occupied=occupied,
    )
    input_name = str(split.inputs[1])
    input_tensor = model_ir.tensors.get(input_name)
    if (
        axis is None
        or input_tensor is None
        or not _resolved_source(
            model_ir,
            graph_index,
            name=input_name,
            before_index=int(split_index),
        )
        or not _rank4(_view(input_tensor))
        or not _per_tensor_quantization(input_tensor.quantization)
        or not _layout_in(
            input_name,
            input_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    input_old_view = _view(input_tensor)
    input_new_view = _permuted_view(input_old_view, _PERM_NCHW_TO_NHWC)
    if input_new_view is None:
        return None

    output_updates = []
    aliases = []
    removals = []
    graph_outputs = {str(value) for value in model_ir.outputs}
    for output_name_value in split.outputs:
        output_name = str(output_name_value)
        output_tensor = model_ir.tensors.get(output_name)
        if (
            output_tensor is None
            or output_name in graph_index.duplicate_producers
            or graph_index.producer(output_name) is not split
            or output_name in graph_outputs
            or not _rank4(_view(output_tensor))
            or _view(output_tensor).dtype != input_old_view.dtype
            or not _per_tensor_quantization(output_tensor.quantization)
            or not _layout_in(
                output_name,
                output_tensor,
                layout_state,
                {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            return None
        output_old_view = _view(output_tensor)
        output_new_view = _permuted_view(
            output_old_view,
            _PERM_NCHW_TO_NHWC,
        )
        if output_new_view is None:
            return None
        output_updates.append(
            _MetadataUpdate(
                name=output_name,
                old_view=output_old_view,
                new_view=output_new_view,
            )
        )

        seen_consumers: set[int] = set()
        for user_index in graph_index.consumer_indices(output_name):
            if int(user_index) in seen_consumers:
                continue
            seen_consumers.add(int(user_index))
            user = model_ir.operators[int(user_index)]
            if user is concat:
                continue
            if (
                not _typed_permutation(
                    model_ir,
                    graph_index,
                    user,
                    _PERM_NCHW_TO_NHWC,
                )
                or str(user.inputs[0]) != output_name
            ):
                return None
            alias_name = str(user.outputs[0])
            alias_tensor = model_ir.tensors.get(alias_name)
            if (
                alias_name in graph_outputs
                or alias_tensor is None
                or alias_name in graph_index.duplicate_producers
                or graph_index.producer(alias_name) is not user
                or _view(alias_tensor) != output_new_view
                or not _per_tensor_quantization(alias_tensor.quantization)
                or not _layout_in(
                    alias_name,
                    alias_tensor,
                    layout_state,
                    {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
                )
            ):
                return None
            aliases.append(
                _AliasRewrite(
                    source_name=alias_name,
                    target_name=output_name,
                    adapter=user,
                )
            )
            removals.append(user)

    return _SplitPlan(
        operator=split,
        axis=axis,
        input_name=input_name,
        canonical_input_name=_unique_name(f"{input_name}_nhwc", occupied),
        input_old_view=input_old_view,
        input_new_view=input_new_view,
        outputs=tuple(output_updates),
        aliases=tuple(aliases),
        removals=_deduplicate_operators(removals),
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
        or _normalized_axis(concat) != 1
    ):
        return None
    concat_output = str(concat.outputs[0])
    concat_tensor = model_ir.tensors.get(concat_output)
    if (
        concat_tensor is None
        or concat_output in graph_index.duplicate_producers
        or graph_index.producer(concat_output) is not concat
        or concat_output in {str(value) for value in model_ir.outputs}
        or not _rank4(_view(concat_tensor))
        or not _per_tensor_quantization(concat_tensor.quantization)
        or not _layout_in(
            concat_output,
            concat_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    occupied = set(str(name) for name in model_ir.tensors)
    split_plans_by_id: Dict[int, _SplitPlan] = {}
    direct_sources = []
    direct_removals = []
    canonical_inputs = []
    canonical_views = []
    reference_view: Optional[_View] = None
    post_permutation_name: Optional[str] = None

    for input_name_value in concat.inputs:
        input_name = str(input_name_value)
        if input_name in graph_index.duplicate_producers:
            return None
        producer_index = graph_index.producers.get(input_name)
        if producer_index is None or int(producer_index) >= int(concat_index):
            return None
        producer = model_ir.operators[int(producer_index)]

        if _op_type(producer) == "TRANSPOSE":
            if (
                not _typed_permutation(
                    model_ir,
                    graph_index,
                    producer,
                    _PERM_NHWC_TO_NCHW,
                )
                or str(producer.outputs[0]) != input_name
                or set(graph_index.consumer_indices(input_name)) != {int(concat_index)}
                or input_name in {str(value) for value in model_ir.outputs}
            ):
                return None
            source_name = str(producer.inputs[0])
            source_tensor = model_ir.tensors.get(source_name)
            output_tensor = model_ir.tensors.get(input_name)
            if (
                source_tensor is None
                or output_tensor is None
                or not _resolved_source(
                    model_ir,
                    graph_index,
                    name=source_name,
                    before_index=int(producer_index),
                )
                or not _rank4(_view(source_tensor))
                or not _rank4(_view(output_tensor))
                or not _per_tensor_quantization(source_tensor.quantization)
                or not _per_tensor_quantization(output_tensor.quantization)
                or not _layout_in(
                    source_name,
                    source_tensor,
                    layout_state,
                    {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
                )
                or not _layout_in(
                    input_name,
                    output_tensor,
                    layout_state,
                    {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
                )
            ):
                return None
            source_view = _view(source_tensor)
            if _permuted_view(source_view, _PERM_NHWC_TO_NCHW) != _view(output_tensor):
                return None
            canonical_name = source_name
            canonical_view = source_view
            direct_sources.append(source_name)
            direct_removals.append(producer)
            if post_permutation_name is None:
                post_permutation_name = str(producer.inputs[1])
        elif _op_type(producer) == "SPLIT":
            split_plan = split_plans_by_id.get(id(producer))
            if split_plan is None:
                split_plan = _resolve_split(
                    model_ir,
                    graph_index,
                    producer,
                    concat,
                    occupied=occupied,
                    layout_state=layout_state,
                )
                if split_plan is None:
                    return None
                split_plans_by_id[id(producer)] = split_plan
            update = next(
                (value for value in split_plan.outputs if value.name == input_name),
                None,
            )
            if update is None:
                return None
            canonical_name = input_name
            canonical_view = update.new_view
        else:
            return None

        if reference_view is None:
            reference_view = canonical_view
        elif not _compatible_nonchannel(reference_view, canonical_view):
            return None
        canonical_inputs.append(canonical_name)
        canonical_views.append(canonical_view)

    if len(split_plans_by_id) == 0 or reference_view is None:
        return None
    concat_old_view = _view(concat_tensor)
    concat_new_view = _permuted_view(
        concat_old_view,
        _PERM_NCHW_TO_NHWC,
    )
    derived_concat_view = _concat_view(canonical_views)
    if concat_new_view is None or derived_concat_view != concat_new_view:
        return None

    create_post_permutation = False
    if post_permutation_name is None:
        post_permutation_name = _find_permutation(
            model_ir,
            graph_index,
            _PERM_NHWC_TO_NCHW,
        )
        if post_permutation_name is None:
            post_permutation_name = _unique_name(
                "mixed_pre_concat_nhwc_to_nchw_perm",
                occupied,
            )
            create_post_permutation = True

    pre_permutation_name = _find_permutation(
        model_ir,
        graph_index,
        _PERM_NCHW_TO_NHWC,
    )
    create_pre_permutation = False
    if pre_permutation_name is None:
        pre_permutation_name = _unique_name(
            "mixed_pre_concat_nchw_to_nhwc_perm",
            occupied,
        )
        create_pre_permutation = True

    canonical_concat_output = _unique_name(
        f"{concat_output}_nhwc",
        occupied,
    )
    involved_operators = {id(concat): concat}
    involved_tensors = {concat_output}
    involved_tensors.update(str(value) for value in concat.inputs)
    involved_tensors.update(canonical_inputs)
    involved_tensors.update(direct_sources)
    for operator in direct_removals:
        involved_operators[id(operator)] = operator
        involved_tensors.update(str(value) for value in operator.inputs)
        involved_tensors.update(str(value) for value in operator.outputs)
    for split_plan in split_plans_by_id.values():
        involved_operators[id(split_plan.operator)] = split_plan.operator
        involved_tensors.add(split_plan.axis.source_name)
        involved_tensors.add(split_plan.input_name)
        for update in split_plan.outputs:
            involved_tensors.add(update.name)
        for alias in split_plan.aliases:
            involved_operators[id(alias.adapter)] = alias.adapter
            involved_tensors.add(alias.source_name)
            involved_tensors.add(alias.target_name)
    tensor_contracts = tuple(
        _tensor_contract(name, model_ir.tensors[name])
        for name in sorted(involved_tensors)
        if name in model_ir.tensors
    )
    operator_contracts = tuple(
        _operator_contract(operator)
        for operator in sorted(
            involved_operators.values(),
            key=lambda value: int(_operator_index(graph_index, value) or 0),
        )
    )
    return _Plan(
        concat=concat,
        original_concat_inputs=tuple(str(value) for value in concat.inputs),
        canonical_concat_inputs=tuple(canonical_inputs),
        original_concat_output=concat_output,
        canonical_concat_output=canonical_concat_output,
        concat_old_view=concat_old_view,
        concat_new_view=concat_new_view,
        direct_sources=tuple(direct_sources),
        direct_removals=_deduplicate_operators(direct_removals),
        splits=tuple(
            sorted(
                split_plans_by_id.values(),
                key=lambda value: int(
                    _operator_index(graph_index, value.operator) or 0
                ),
            )
        ),
        post_permutation_name=post_permutation_name,
        create_post_permutation=create_post_permutation,
        pre_permutation_name=pre_permutation_name,
        create_pre_permutation=create_pre_permutation,
        tensor_contracts=tensor_contracts,
        operator_contracts=operator_contracts,
        graph_inputs=tuple(str(value) for value in model_ir.inputs),
        graph_outputs=tuple(str(value) for value in model_ir.outputs),
    )


def _set_layout(
    tensor: TensorIR,
    name: str,
    layout: str,
    layout_state: Optional[LayoutState],
) -> None:
    tensor.logical_layout = str(layout)
    tensor.physical_layout = str(layout)
    if layout_state is not None:
        layout_state.set(str(name), logical=layout, physical=layout)


def _create_permutation(
    model_ir: ModelIR,
    name: str,
    values: Sequence[int],
    layout_state: Optional[LayoutState],
) -> None:
    model_ir.tensors[str(name)] = TensorIR(
        name=str(name),
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([int(value) for value in values], dtype=np.int32),
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
        plan.concat,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if (
        tuple(str(value) for value in model_ir.inputs) != plan.graph_inputs
        or tuple(str(value) for value in model_ir.outputs) != plan.graph_outputs
    ):
        return False

    removals = list(plan.direct_removals)
    removals.extend(operator for split in plan.splits for operator in split.removals)
    removal_indices = []
    for operator in _deduplicate_operators(removals):
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))
    if _operator_index(graph_index, plan.concat) is None or any(
        _operator_index(graph_index, split.operator) is None for split in plan.splits
    ):
        return False

    axis_data_by_split: Dict[int, np.ndarray] = {}
    for split in plan.splits:
        axis_tensor = model_ir.tensors.get(split.axis.source_name)
        if axis_tensor is None or axis_tensor.data is None:
            return False
        axis_data = np.asarray(axis_tensor.data)
        if (
            str(axis_tensor.dtype) != split.axis.dtype
            or str(axis_data.dtype) != split.axis.numpy_dtype
            or _freeze(axis_data) != split.axis.data_contract
        ):
            return False
        axis_data_by_split[id(split.operator)] = axis_data

    if plan.create_post_permutation:
        _create_permutation(
            model_ir,
            plan.post_permutation_name,
            _PERM_NHWC_TO_NCHW,
            layout_state,
        )
    if plan.create_pre_permutation:
        _create_permutation(
            model_ir,
            plan.pre_permutation_name,
            _PERM_NCHW_TO_NHWC,
            layout_state,
        )

    for split in plan.splits:
        axis_tensor = model_ir.tensors[split.axis.source_name]
        axis_data = axis_data_by_split[id(split.operator)]
        if split.axis.in_place:
            axis_tensor.data = np.asarray([3], dtype=axis_data.dtype).reshape(
                axis_data.shape
            )
        else:
            model_ir.tensors[split.axis.target_name] = TensorIR(
                name=split.axis.target_name,
                dtype=str(axis_tensor.dtype),
                shape=[1],
                shape_signature=[1],
                data=np.asarray([3], dtype=axis_data.dtype),
                is_variable=False,
                quantization=_clone_quantization(axis_tensor.quantization),
                logical_layout=str(axis_tensor.logical_layout),
                physical_layout=str(axis_tensor.physical_layout),
                onnx_tensor_name=axis_tensor.onnx_tensor_name,
            )
            if layout_state is not None:
                layout_state.set(
                    split.axis.target_name,
                    logical=str(axis_tensor.logical_layout),
                    physical=str(axis_tensor.physical_layout),
                )

        input_tensor = model_ir.tensors[split.input_name]
        canonical_input_tensor = TensorIR(
            name=split.canonical_input_name,
            dtype=str(input_tensor.dtype),
            shape=[int(value) for value in split.input_new_view.shape],
            shape_signature=[int(value) for value in split.input_new_view.signature],
            data=None,
            is_variable=False,
            quantization=_clone_quantization(input_tensor.quantization),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
            onnx_tensor_name=input_tensor.onnx_tensor_name,
        )
        model_ir.tensors[split.canonical_input_name] = canonical_input_tensor
        if layout_state is not None:
            layout_state.set(
                split.canonical_input_name,
                logical=LOGICAL_LAYOUT_NHWC,
                physical=LOGICAL_LAYOUT_NHWC,
            )

        split_inputs = [str(value) for value in split.operator.inputs]
        split_inputs[0] = split.axis.target_name
        split_inputs[1] = split.canonical_input_name
        _set_operator_inputs(
            model_ir=model_ir,
            op=split.operator,
            new_inputs=split_inputs,
            graph_index=graph_index,
        )
        for update in split.outputs:
            tensor = model_ir.tensors[update.name]
            tensor.shape = [int(value) for value in update.new_view.shape]
            tensor.shape_signature = [int(value) for value in update.new_view.signature]
            _set_layout(
                tensor,
                update.name,
                LOGICAL_LAYOUT_NHWC,
                layout_state,
            )
        for alias in split.aliases:
            _replace_tensor_inputs(
                model_ir,
                alias.source_name,
                alias.target_name,
                graph_index=graph_index,
            )

    concat_tensor = model_ir.tensors[plan.original_concat_output]
    model_ir.tensors[plan.canonical_concat_output] = TensorIR(
        name=plan.canonical_concat_output,
        dtype=str(concat_tensor.dtype),
        shape=[int(value) for value in plan.concat_new_view.shape],
        shape_signature=[int(value) for value in plan.concat_new_view.signature],
        data=None,
        is_variable=False,
        quantization=_clone_quantization(concat_tensor.quantization),
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
        onnx_tensor_name=concat_tensor.onnx_tensor_name,
    )
    if layout_state is not None:
        layout_state.set(
            plan.canonical_concat_output,
            logical=LOGICAL_LAYOUT_NHWC,
            physical=LOGICAL_LAYOUT_NHWC,
        )
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.concat,
        new_inputs=list(plan.canonical_concat_inputs),
        graph_index=graph_index,
    )
    options = dict(plan.concat.options)
    options["axis"] = 3
    plan.concat.options = options
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.concat,
        new_outputs=[plan.canonical_concat_output],
        graph_index=graph_index,
    )
    _set_layout(
        concat_tensor,
        plan.original_concat_output,
        LOGICAL_LAYOUT_NCHW,
        layout_state,
    )
    for source_name in plan.direct_sources:
        source_tensor = model_ir.tensors[source_name]
        _set_layout(
            source_tensor,
            source_name,
            LOGICAL_LAYOUT_NHWC,
            layout_state,
        )

    graph_index.remove_operators(removal_indices)

    for split in plan.splits:
        split_index = _operator_index(graph_index, split.operator)
        if split_index is None:
            raise RuntimeError("validated Split disappeared during indexed apply")
        graph_index.insert_operator(
            int(split_index),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[split.input_name, plan.pre_permutation_name],
                outputs=[split.canonical_input_name],
                options={},
            ),
        )
    concat_index = _operator_index(graph_index, plan.concat)
    if concat_index is None:
        raise RuntimeError("validated Concat disappeared during indexed apply")
    graph_index.insert_operator(
        int(concat_index) + 1,
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[
                plan.canonical_concat_output,
                plan.post_permutation_name,
            ],
            outputs=[plan.original_concat_output],
            options={},
        ),
    )
    return True


def optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Lift fully classified Split/direct-adapter Concat fan-in to NHWC."""

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
