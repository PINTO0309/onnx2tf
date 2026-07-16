from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
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
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    _AxisUpdate,
    _InputRewrite,
    _MetadataUpdate,
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
    _rank4_positive,
    _resolved_source,
    _tensor_contract,
    _typed_constant,
    _typed_permutation,
    _unique_name,
    _view,
)


_STATS_KEY = "optimized_split_conv_concat_transpose_bridge_to_single_post_nchw"


@dataclass(frozen=True)
class _ConcatUpdate:
    operator: OperatorIR
    original_axis: int
    new_axis: int
    original_output: str
    private_output: str
    private_shape: Tuple[int, ...]
    private_signature: Tuple[int, ...]
    permutation_name: str


@dataclass(frozen=True)
class _BridgePlan:
    pre: OperatorIR
    split: OperatorIR
    branch_adapter: OperatorIR
    concat: OperatorIR
    post_adapters: Tuple[OperatorIR, ...]
    input_rewrites: Tuple[_InputRewrite, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    axis_update: _AxisUpdate
    concat_update: _ConcatUpdate
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]


def _concat_views(
    views: Sequence[_View],
    *,
    axis: int,
) -> Optional[_View]:
    if len(views) < 2 or int(axis) not in range(4):
        return None
    dtype = views[0].dtype
    if any(
        view.dtype != dtype
        or len(view.shape) != 4
        or len(view.signature) != 4
        for view in views
    ):
        return None
    output_shape = []
    output_signature = []
    for dimension in range(4):
        shape_values = [int(view.shape[dimension]) for view in views]
        signature_values = [int(view.signature[dimension]) for view in views]
        if dimension == int(axis):
            output_shape.append(sum(shape_values))
            output_signature.append(
                sum(signature_values)
                if all(value >= 0 for value in signature_values)
                else -1
            )
            continue
        if len(set(shape_values)) != 1:
            return None
        known_signatures = {value for value in signature_values if value >= 0}
        if len(known_signatures) > 1:
            return None
        output_shape.append(shape_values[0])
        output_signature.append(
            next(iter(known_signatures)) if known_signatures else -1
        )
    return _View(
        shape=tuple(output_shape),
        signature=tuple(output_signature),
        dtype=dtype,
    )


def _reachable_before_concat(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    source_name: str,
    target_names: set[str],
    after_index: int,
    before_index: int,
) -> bool:
    if str(source_name) in target_names:
        return True
    queue = deque([str(source_name)])
    visited = {str(source_name)}
    edge_limit = max(
        1,
        sum(len(operator.inputs) + len(operator.outputs) for operator in model_ir.operators),
    )
    traversed = 0
    while queue and traversed < edge_limit:
        tensor_name = queue.popleft()
        for consumer_index in graph_index.consumer_indices(tensor_name):
            traversed += 1
            if (
                int(consumer_index) <= int(after_index)
                or int(consumer_index) >= int(before_index)
            ):
                continue
            operator = model_ir.operators[int(consumer_index)]
            for output_name in operator.outputs:
                normalized_name = str(output_name)
                if normalized_name in target_names:
                    return True
                if normalized_name and normalized_name not in visited:
                    visited.add(normalized_name)
                    queue.append(normalized_name)
    return False


def _plan_signature(plan: _BridgePlan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        id(plan.split),
        id(plan.branch_adapter),
        id(plan.concat),
        tuple(id(operator) for operator in plan.post_adapters),
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
            id(plan.concat_update.operator),
            plan.concat_update.original_axis,
            plan.concat_update.new_axis,
            plan.concat_update.original_output,
            plan.concat_update.private_output,
            plan.concat_update.private_shape,
            plan.concat_update.private_signature,
            plan.concat_update.permutation_name,
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
) -> Optional[_BridgePlan]:
    split_index = _operator_index(graph_index, split)
    if (
        split_index is None
        or _op_type(split) != "SPLIT"
        or len(split.inputs) != 2
        or len(split.outputs) < 2
        or len(set(str(value) for value in split.outputs)) != len(split.outputs)
        or not isinstance(split.options, dict)
    ):
        return None
    split_outputs = tuple(str(value) for value in split.outputs)
    axis_name = str(split.inputs[0])
    axis_resolved = _typed_constant(
        model_ir,
        graph_index,
        axis_name,
        shape=(1,),
    )
    if axis_resolved is None:
        return None
    split_axis = int(axis_resolved[1].reshape(-1)[0])
    if split_axis < 0:
        split_axis += 4
    if split_axis != 1:
        return None
    num_splits = split.options.get("numSplits")
    if num_splits is not None:
        try:
            if int(num_splits) != len(split_outputs):
                return None
        except Exception:
            return None

    graph_inputs = {str(value) for value in model_ir.inputs}
    graph_outputs = {str(value) for value in model_ir.outputs}
    public_names = graph_inputs | graph_outputs
    pre_output_name = str(split.inputs[1])
    if (
        pre_output_name in public_names
        or pre_output_name in graph_index.duplicate_producers
    ):
        return None
    pre_index = graph_index.producers.get(pre_output_name)
    if pre_index is None or int(pre_index) >= int(split_index):
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
        or graph_index.consumer_indices(pre_output_name) != [int(split_index)]
    ):
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
    pre_tensor = model_ir.tensors.get(pre_output_name)
    split_tensors = {name: model_ir.tensors.get(name) for name in split_outputs}
    if (
        source_tensor is None
        or pre_tensor is None
        or any(tensor is None for tensor in split_tensors.values())
        or bool(source_tensor.is_variable)
        or pre_tensor.data is not None
        or bool(pre_tensor.is_variable)
        or any(
            tensor.data is not None or bool(tensor.is_variable)
            for tensor in split_tensors.values()
            if tensor is not None
        )
        or not all(
            _per_tensor_quantization(tensor.quantization)
            for tensor in [source_tensor, pre_tensor, *split_tensors.values()]
            if tensor is not None
        )
    ):
        return None
    source_view = _view(source_tensor)
    pre_view = _view(pre_tensor)
    expected_pre_view = _permuted_view(source_view, _PERM_NHWC_TO_NCHW)
    if (
        not _rank4_positive(source_view)
        or not _rank4_positive(pre_view)
        or expected_pre_view is None
        or pre_view != expected_pre_view
        or _freeze(source_tensor.quantization) != _freeze(pre_tensor.quantization)
        or _layout_of(source_name, source_tensor, layout_state)
        not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}
        or _layout_of(pre_output_name, pre_tensor, layout_state)
        not in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN}
    ):
        return None
    output_count = len(split_outputs)
    static_channels = int(pre_view.shape[1])
    signature_channels = int(pre_view.signature[1])
    if static_channels % output_count != 0 or (
        signature_channels >= 0 and signature_channels % output_count != 0
    ):
        return None
    expected_split_view = _View(
        shape=(
            int(pre_view.shape[0]),
            static_channels // output_count,
            int(pre_view.shape[2]),
            int(pre_view.shape[3]),
        ),
        signature=(
            int(pre_view.signature[0]),
            signature_channels // output_count if signature_channels >= 0 else -1,
            int(pre_view.signature[2]),
            int(pre_view.signature[3]),
        ),
        dtype=pre_view.dtype,
    )
    split_new_view = _permuted_view(
        expected_split_view,
        _PERM_NCHW_TO_NHWC,
    )
    if split_new_view is None:
        return None
    for name in split_outputs:
        tensor = split_tensors[name]
        assert tensor is not None
        if (
            name in public_names
            or name in graph_index.duplicate_producers
            or graph_index.producers.get(name) != int(split_index)
            or _view(tensor) != expected_split_view
            or _layout_of(name, tensor, layout_state)
            not in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN}
        ):
            return None

    for branch_name in split_outputs:
        branch_consumers = graph_index.consumer_indices(branch_name)
        if len(branch_consumers) != 1:
            continue
        branch_adapter_index = int(branch_consumers[0])
        if branch_adapter_index <= int(split_index):
            continue
        branch_adapter = model_ir.operators[branch_adapter_index]
        if (
            not _typed_permutation(
                model_ir,
                graph_index,
                branch_adapter,
                _PERM_NCHW_TO_NHWC,
            )
            or str(branch_adapter.inputs[0]) != branch_name
        ):
            continue
        branch_output_name = str(branch_adapter.outputs[0])
        branch_output_tensor = model_ir.tensors.get(branch_output_name)
        if (
            branch_output_name in public_names
            or branch_output_name in graph_index.duplicate_producers
            or graph_index.producers.get(branch_output_name)
            != int(branch_adapter_index)
            or branch_output_tensor is None
            or branch_output_tensor.data is not None
            or bool(branch_output_tensor.is_variable)
            or _view(branch_output_tensor) != split_new_view
            or not _per_tensor_quantization(branch_output_tensor.quantization)
            or _freeze(branch_output_tensor.quantization)
            != _freeze(split_tensors[branch_name].quantization)
            or _layout_of(branch_output_name, branch_output_tensor, layout_state)
            not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}
        ):
            continue
        branch_output_slots = _consumer_slots(
            model_ir,
            graph_index,
            branch_output_name,
        )
        if len(branch_output_slots) == 0 or any(
            (_operator_index(graph_index, operator) or -1)
            <= int(branch_adapter_index)
            for operator, _ in branch_output_slots
        ):
            continue

        direct_names = tuple(name for name in split_outputs if name != branch_name)
        direct_concat: Optional[OperatorIR] = None
        direct_concat_index: Optional[int] = None
        valid_direct = True
        for direct_name in direct_names:
            consumers = graph_index.consumer_indices(direct_name)
            if len(consumers) != 1:
                valid_direct = False
                break
            consumer_index = int(consumers[0])
            consumer = model_ir.operators[consumer_index]
            if _op_type(consumer) != "CONCATENATION":
                valid_direct = False
                break
            if direct_concat is None:
                direct_concat = consumer
                direct_concat_index = consumer_index
            elif consumer is not direct_concat:
                valid_direct = False
                break
        if (
            not valid_direct
            or direct_concat is None
            or direct_concat_index is None
            or int(direct_concat_index) <= int(branch_adapter_index)
        ):
            continue
        concat = direct_concat
        concat_index = int(direct_concat_index)
        if (
            len(concat.inputs) < 2
            or len(concat.outputs) != 1
            or len(set(str(value) for value in concat.inputs)) != len(concat.inputs)
            or not isinstance(concat.options, dict)
        ):
            continue
        try:
            original_concat_axis = int(concat.options.get("axis", 0))
        except Exception:
            continue
        normalized_concat_axis = int(original_concat_axis)
        if normalized_concat_axis < 0:
            normalized_concat_axis += 4
        if normalized_concat_axis != 1:
            continue
        concat_output_name = str(concat.outputs[0])
        concat_output_tensor = model_ir.tensors.get(concat_output_name)
        if (
            concat_output_name in public_names
            or concat_output_name in graph_index.duplicate_producers
            or graph_index.producers.get(concat_output_name) != int(concat_index)
            or concat_output_tensor is None
            or concat_output_tensor.data is not None
            or bool(concat_output_tensor.is_variable)
            or not _per_tensor_quantization(concat_output_tensor.quantization)
            or not _rank4_positive(_view(concat_output_tensor))
            or _layout_of(concat_output_name, concat_output_tensor, layout_state)
            not in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN}
        ):
            continue
        if any(
            (_operator_index(graph_index, operator) or -1) <= int(concat_index)
            for operator, _ in _consumer_slots(
                model_ir,
                graph_index,
                concat_output_name,
            )
        ):
            continue

        concat_inputs = tuple(str(value) for value in concat.inputs)
        if any(concat_inputs.count(name) != 1 for name in direct_names):
            continue
        post_adapters = []
        post_input_names = []
        post_output_names = []
        valid_concat = True
        for input_name in concat_inputs:
            if input_name in direct_names:
                continue
            if input_name == branch_name or input_name in split_outputs:
                valid_concat = False
                break
            post_adapter_index = graph_index.producers.get(input_name)
            if (
                post_adapter_index is None
                or int(post_adapter_index) <= int(branch_adapter_index)
                or int(post_adapter_index) >= int(concat_index)
            ):
                valid_concat = False
                break
            post_adapter = model_ir.operators[int(post_adapter_index)]
            if (
                not _typed_permutation(
                    model_ir,
                    graph_index,
                    post_adapter,
                    _PERM_NHWC_TO_NCHW,
                )
                or str(post_adapter.outputs[0]) != input_name
                or graph_index.consumer_indices(input_name) != [int(concat_index)]
            ):
                valid_concat = False
                break
            post_input_name = str(post_adapter.inputs[0])
            post_input_tensor = model_ir.tensors.get(post_input_name)
            post_output_tensor = model_ir.tensors.get(input_name)
            if (
                post_input_name in graph_outputs
                or input_name in public_names
                or post_input_name in graph_index.duplicate_producers
                or input_name in graph_index.duplicate_producers
                or post_input_tensor is None
                or post_output_tensor is None
                or post_input_tensor.data is not None
                or post_output_tensor.data is not None
                or bool(post_input_tensor.is_variable)
                or bool(post_output_tensor.is_variable)
                or not _rank4_positive(_view(post_input_tensor))
                or _permuted_view(_view(post_input_tensor), _PERM_NHWC_TO_NCHW)
                != _view(post_output_tensor)
                or not _per_tensor_quantization(post_input_tensor.quantization)
                or not _per_tensor_quantization(post_output_tensor.quantization)
                or _freeze(post_input_tensor.quantization)
                != _freeze(post_output_tensor.quantization)
                or not _resolved_source(
                    model_ir,
                    graph_index,
                    name=post_input_name,
                    before_index=int(post_adapter_index),
                )
                or _layout_of(post_input_name, post_input_tensor, layout_state)
                not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}
                or _layout_of(input_name, post_output_tensor, layout_state)
                not in {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN}
            ):
                valid_concat = False
                break
            post_adapters.append(post_adapter)
            post_input_names.append(post_input_name)
            post_output_names.append(input_name)
        if not valid_concat or len(post_adapters) == 0:
            continue
        if not _reachable_before_concat(
            model_ir,
            graph_index,
            source_name=branch_output_name,
            target_names=set(post_input_names),
            after_index=int(branch_adapter_index),
            before_index=int(concat_index),
        ):
            continue

        old_input_views = []
        new_input_views = []
        post_output_to_input = dict(zip(post_output_names, post_input_names))
        for input_name in concat_inputs:
            if input_name in direct_names:
                direct_tensor = split_tensors[input_name]
                assert direct_tensor is not None
                old_input_views.append(_view(direct_tensor))
                new_input_views.append(split_new_view)
                continue
            post_input_name = post_output_to_input[input_name]
            post_input_tensor = model_ir.tensors[post_input_name]
            post_output_tensor = model_ir.tensors[input_name]
            old_input_views.append(_view(post_output_tensor))
            new_input_views.append(_view(post_input_tensor))
        old_concat_view = _concat_views(old_input_views, axis=1)
        new_concat_view = _concat_views(new_input_views, axis=3)
        concat_view = _view(concat_output_tensor)
        if (
            old_concat_view is None
            or new_concat_view is None
            or concat_view != old_concat_view
            or _permuted_view(concat_view, _PERM_NCHW_TO_NHWC)
            != new_concat_view
        ):
            continue

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
            continue
        axis_in_place = axis_slots == split_axis_slot
        target_axis_name = axis_name
        occupied = {str(name) for name in model_ir.tensors}
        if not axis_in_place:
            target_axis_name = _unique_name(f"{axis_name}_nhwc", occupied)
        private_output_name = _unique_name(
            f"{concat_output_name}__nhwc",
            occupied,
        )
        axis_tensor, axis_data = axis_resolved
        axis_update = _AxisUpdate(
            source_name=axis_name,
            target_name=target_axis_name,
            in_place=axis_in_place,
            dtype=str(axis_tensor.dtype),
            numpy_dtype=str(axis_data.dtype),
        )

        removal_set = {id(pre), id(branch_adapter), *(id(op) for op in post_adapters)}
        planned_inputs: Dict[int, list[str]] = {
            id(split): [target_axis_name, source_name],
            id(concat): [
                (
                    branch_name
                    if post_output_to_input.get(input_name) == branch_output_name
                    else post_output_to_input.get(input_name, input_name)
                )
                for input_name in concat_inputs
            ],
        }
        surviving_branch_consumers: Dict[int, OperatorIR] = {}
        for consumer, input_slot in branch_output_slots:
            if id(consumer) in removal_set:
                continue
            surviving_branch_consumers[id(consumer)] = consumer
            inputs = planned_inputs.setdefault(
                id(consumer),
                [str(value) for value in consumer.inputs],
            )
            inputs[int(input_slot)] = branch_name
        rewritten_operators = [
            split,
            concat,
            *surviving_branch_consumers.values(),
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
        metadata_updates = tuple(
            _MetadataUpdate(
                name=name,
                shape=split_new_view.shape,
                signature=split_new_view.signature,
            )
            for name in split_outputs
        )
        relevant_operators = [
            pre,
            split,
            branch_adapter,
            *post_adapters,
            concat,
            *surviving_branch_consumers.values(),
        ]
        relevant_operators = sorted(
            relevant_operators,
            key=lambda operator: int(_operator_index(graph_index, operator) or 0),
        )
        contract_names = set()
        for operator in relevant_operators:
            contract_names.update(
                str(value)
                for value in (*operator.inputs, *operator.outputs)
                if str(value)
            )
        tensor_contracts = []
        for name in sorted(contract_names):
            tensor = model_ir.tensors.get(name)
            if tensor is None:
                valid_concat = False
                break
            tensor_contracts.append(_tensor_contract(name, tensor))
        if not valid_concat:
            continue
        return _BridgePlan(
            pre=pre,
            split=split,
            branch_adapter=branch_adapter,
            concat=concat,
            post_adapters=tuple(post_adapters),
            input_rewrites=input_rewrites,
            metadata_updates=metadata_updates,
            axis_update=axis_update,
            concat_update=_ConcatUpdate(
                operator=concat,
                original_axis=original_concat_axis,
                new_axis=3,
                original_output=concat_output_name,
                private_output=private_output_name,
                private_shape=new_concat_view.shape,
                private_signature=new_concat_view.signature,
                permutation_name=str(pre.inputs[1]),
            ),
            removals=(pre, branch_adapter, *post_adapters),
            tensor_contracts=tuple(tensor_contracts),
            operator_contracts=tuple(
                _operator_contract(operator) for operator in relevant_operators
            ),
        )
    return None


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _BridgePlan,
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
    for rewrite in plan.input_rewrites:
        if tuple(str(value) for value in rewrite.operator.inputs) != rewrite.original_inputs:
            return False
    if any(
        _operator_index(graph_index, operator) is None for operator in plan.removals
    ):
        return False
    axis_update = plan.axis_update
    source_axis = model_ir.tensors.get(axis_update.source_name)
    if source_axis is None:
        return False
    if (
        not axis_update.in_place
        and axis_update.target_name in model_ir.tensors
    ):
        return False
    concat_update = plan.concat_update
    if (
        concat_update.private_output in model_ir.tensors
        or concat_update.operator is not plan.concat
        or list(plan.concat.outputs) != [concat_update.original_output]
        or not isinstance(plan.concat.options, dict)
    ):
        return False
    try:
        if int(plan.concat.options.get("axis", 0)) != int(
            concat_update.original_axis
        ):
            return False
    except Exception:
        return False
    public_tensor = model_ir.tensors.get(concat_update.original_output)
    if public_tensor is None:
        return False
    for metadata in plan.metadata_updates:
        if metadata.name not in model_ir.tensors:
            return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))

    numpy_dtype = np.dtype(axis_update.numpy_dtype)
    if axis_update.in_place:
        target_axis = source_axis
    else:
        target_axis = TensorIR(
            name=axis_update.target_name,
            dtype=axis_update.dtype,
            shape=[1],
            shape_signature=[1],
            data=np.asarray([3], dtype=numpy_dtype),
            is_variable=False,
            quantization=_clone_quantization(source_axis.quantization),
            logical_layout=str(source_axis.logical_layout),
            physical_layout=str(source_axis.physical_layout),
            onnx_tensor_name=source_axis.onnx_tensor_name,
        )
        model_ir.tensors[axis_update.target_name] = target_axis
    target_axis.data = np.asarray([3], dtype=numpy_dtype)
    target_axis.shape = [1]
    target_axis.shape_signature = [1]
    if layout_state is not None and not axis_update.in_place:
        layout_state.set(
            axis_update.target_name,
            logical=str(source_axis.logical_layout),
            physical=str(source_axis.physical_layout),
        )

    model_ir.tensors[concat_update.private_output] = TensorIR(
        name=concat_update.private_output,
        dtype=str(public_tensor.dtype),
        shape=[int(value) for value in concat_update.private_shape],
        shape_signature=[int(value) for value in concat_update.private_signature],
        data=None,
        is_variable=False,
        quantization=_clone_quantization(public_tensor.quantization),
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
        onnx_tensor_name=public_tensor.onnx_tensor_name,
    )
    for rewrite in plan.input_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    concat_options = dict(plan.concat.options)
    concat_options["axis"] = int(concat_update.new_axis)
    plan.concat.options = concat_options
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
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.concat,
        new_outputs=[concat_update.private_output],
        graph_index=graph_index,
    )
    graph_index.remove_operators(removal_indices)
    concat_index = _operator_index(graph_index, plan.concat)
    if concat_index is None:
        raise RuntimeError("validated Split/Conv/Concat producer disappeared")
    graph_index.insert_operator(
        int(concat_index) + 1,
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[concat_update.private_output, concat_update.permutation_name],
            outputs=[concat_update.original_output],
        ),
    )
    if layout_state is not None:
        layout_state.set(
            concat_update.private_output,
            logical=LOGICAL_LAYOUT_NHWC,
            physical=LOGICAL_LAYOUT_NHWC,
        )
    return True


def optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Move a fully classified Split/Conv/Concat bridge to NHWC."""

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
