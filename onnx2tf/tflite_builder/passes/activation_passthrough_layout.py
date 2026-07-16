from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
    logical_layout_permutation,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    _InputRewrite,
    _MetadataUpdate,
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
    _view,
)


_SWISH_STATS_KEY = "rewritten_swish_transpose_passthrough_chains"
_GELU_STATS_KEY = "rewritten_gelu_tanh_transpose_passthrough_chains"


@dataclass(frozen=True)
class _SwishPlan:
    pre: OperatorIR
    logistic: OperatorIR
    multiply: OperatorIR
    posts: Tuple[OperatorIR, ...]
    adapter_template: OperatorIR
    source_name: str
    pre_output_name: str
    logistic_output_name: str
    multiply_output_name: str
    representative_output_name: str
    pre_permutation_name: str
    permutation: Tuple[int, ...]
    inverse_permutation: Tuple[int, ...]
    preserve_legacy_boundary: bool
    source_logical_layout: str
    source_physical_layout: str
    input_rewrites: Tuple[_InputRewrite, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _inverse_permutation(permutation: Sequence[int]) -> Optional[Tuple[int, ...]]:
    normalized = tuple(int(value) for value in permutation)
    if sorted(normalized) != list(range(len(normalized))):
        return None
    result = [0] * len(normalized)
    for index, value in enumerate(normalized):
        result[int(value)] = int(index)
    return tuple(result)


def _resolved_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    rank: int,
) -> Optional[Tuple[int, ...]]:
    if (
        _op_type(operator) != "TRANSPOSE"
        or len(operator.inputs) != 2
        or len(operator.outputs) != 1
        or int(rank) < 2
    ):
        return None
    resolved = _typed_constant(
        model_ir,
        graph_index,
        str(operator.inputs[1]),
        shape=(int(rank),),
    )
    if resolved is None:
        return None
    permutation = tuple(int(value) for value in resolved[1].reshape(-1))
    return permutation if _inverse_permutation(permutation) is not None else None


def _runtime_tensor(
    tensor: Optional[TensorIR],
    *,
    rank: int,
    allow_constant: bool = False,
) -> bool:
    if tensor is None:
        return False
    view = _view(tensor)
    valid = bool(
        (tensor.data is None or bool(allow_constant))
        and not bool(tensor.is_variable)
        and len(view.shape) == int(rank)
        and len(view.signature) == int(rank)
        and all(int(value) > 0 for value in view.shape)
        and _per_tensor_quantization(tensor.quantization)
    )
    if not valid or tensor.data is None:
        return valid
    try:
        data = np.asarray(tensor.data)
        return bool(
            tuple(int(value) for value in data.shape) == view.shape
            and str(data.dtype).upper() == str(tensor.dtype).upper()
        )
    except Exception:
        return False


def _state_layouts(
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


def _layout_transition_matches(
    source_layout: str,
    target_layout: str,
    permutation: Sequence[int],
) -> bool:
    source = str(source_layout).upper()
    target = str(target_layout).upper()
    if LOGICAL_LAYOUT_UNKNOWN in {source, target}:
        return True
    expected = logical_layout_permutation(
        source_layout=source,
        target_layout=target,
    )
    return bool(
        expected is not None
        and tuple(int(value) for value in expected)
        == tuple(int(value) for value in permutation)
    )


def _layout_matches(value: str, expected: str) -> bool:
    normalized = str(value).upper()
    if str(expected).upper() == LOGICAL_LAYOUT_UNKNOWN:
        return True
    return normalized in {LOGICAL_LAYOUT_UNKNOWN, str(expected).upper()}


def _plan_signature(plan: _SwishPlan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        id(plan.logistic),
        id(plan.multiply),
        tuple(id(operator) for operator in plan.posts),
        id(plan.adapter_template),
        plan.source_name,
        plan.pre_output_name,
        plan.logistic_output_name,
        plan.multiply_output_name,
        plan.representative_output_name,
        plan.pre_permutation_name,
        plan.permutation,
        plan.inverse_permutation,
        plan.preserve_legacy_boundary,
        plan.source_logical_layout,
        plan.source_physical_layout,
        tuple(
            (id(rewrite.operator), rewrite.original_inputs, rewrite.new_inputs)
            for rewrite in plan.input_rewrites
        ),
        tuple(
            (update.name, update.shape, update.signature)
            for update in plan.metadata_updates
        ),
        tuple(id(operator) for operator in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_SwishPlan]:
    pre_index = _operator_index(graph_index, pre)
    if (
        pre_index is None
        or _op_type(pre) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
    ):
        return None
    graph_inputs = tuple(str(value) for value in model_ir.inputs)
    graph_outputs = tuple(str(value) for value in model_ir.outputs)
    public_inputs = set(graph_inputs)
    public_outputs = set(graph_outputs)
    source_name = str(pre.inputs[0])
    pre_output_name = str(pre.outputs[0])
    if (
        pre_output_name in public_inputs | public_outputs
        or pre_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(pre_output_name) != int(pre_index)
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
    if source_tensor is None or pre_tensor is None:
        return None
    rank = len(source_tensor.shape)
    if not _runtime_tensor(
        source_tensor,
        rank=rank,
        allow_constant=True,
    ) or not _runtime_tensor(
        pre_tensor,
        rank=rank,
    ):
        return None
    permutation = _resolved_permutation(
        model_ir,
        graph_index,
        pre,
        rank=rank,
    )
    if permutation is None:
        return None
    inverse_permutation = _inverse_permutation(permutation)
    if inverse_permutation is None:
        return None
    source_view = _view(source_tensor)
    pre_view = _view(pre_tensor)
    if (
        _permuted_view(source_view, permutation) != pre_view
        or _freeze(source_tensor.quantization) != _freeze(pre_tensor.quantization)
    ):
        return None
    source_logical, source_physical = _state_layouts(
        source_name,
        source_tensor,
        layout_state,
    )
    pre_logical, pre_physical = _state_layouts(
        pre_output_name,
        pre_tensor,
        layout_state,
    )
    if not _layout_transition_matches(
        source_physical,
        pre_physical,
        permutation,
    ):
        return None
    if not _layout_transition_matches(
        source_logical,
        pre_logical,
        permutation,
    ):
        return None

    pre_slots = _consumer_slots(model_ir, graph_index, pre_output_name)
    if len(pre_slots) != 2:
        return None
    logistic: Optional[OperatorIR] = None
    multiply: Optional[OperatorIR] = None
    multiply_data_slot: Optional[int] = None
    for consumer, input_slot in pre_slots:
        if (
            _op_type(consumer) == "LOGISTIC"
            and len(consumer.inputs) == 1
            and len(consumer.outputs) == 1
            and int(input_slot) == 0
        ):
            if logistic is not None:
                return None
            logistic = consumer
            continue
        if _op_type(consumer) == "MUL" and len(consumer.inputs) == 2:
            if multiply is not None:
                return None
            multiply = consumer
            multiply_data_slot = int(input_slot)
            continue
        return None
    if logistic is None or multiply is None or multiply_data_slot is None:
        return None
    logistic_index = _operator_index(graph_index, logistic)
    multiply_index = _operator_index(graph_index, multiply)
    if (
        logistic_index is None
        or multiply_index is None
        or not (int(pre_index) < int(logistic_index) < int(multiply_index))
        or len(multiply.outputs) != 1
    ):
        return None
    logistic_output_name = str(logistic.outputs[0])
    multiply_output_name = str(multiply.outputs[0])
    if (
        logistic_output_name in public_inputs | public_outputs
        or logistic_output_name in graph_index.duplicate_producers
        or multiply_output_name in public_inputs
        or multiply_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(logistic_output_name) != int(logistic_index)
        or graph_index.producers.get(multiply_output_name) != int(multiply_index)
    ):
        return None
    logistic_slots = _consumer_slots(
        model_ir,
        graph_index,
        logistic_output_name,
    )
    if len(logistic_slots) != 1 or logistic_slots[0][0] is not multiply:
        return None
    logistic_slot = int(logistic_slots[0][1])
    if (
        logistic_slot == int(multiply_data_slot)
        or str(multiply.inputs[int(multiply_data_slot)]) != pre_output_name
        or str(multiply.inputs[int(logistic_slot)]) != logistic_output_name
    ):
        return None
    logistic_tensor = model_ir.tensors.get(logistic_output_name)
    multiply_tensor = model_ir.tensors.get(multiply_output_name)
    if (
        not _runtime_tensor(logistic_tensor, rank=rank)
        or not _runtime_tensor(multiply_tensor, rank=rank)
    ):
        return None
    assert logistic_tensor is not None
    assert multiply_tensor is not None
    if (
        _view(logistic_tensor) != pre_view
        or _view(multiply_tensor) != pre_view
        or not _layout_matches(
            _layout_of(logistic_output_name, logistic_tensor, layout_state),
            pre_physical,
        )
        or not _layout_matches(
            _layout_of(multiply_output_name, multiply_tensor, layout_state),
            pre_physical,
        )
    ):
        return None

    posts = []
    legacy_slots = []
    for consumer, input_slot in _consumer_slots(
        model_ir,
        graph_index,
        multiply_output_name,
    ):
        consumer_index = _operator_index(graph_index, consumer)
        if consumer_index is None or int(consumer_index) <= int(multiply_index):
            return None
        candidate_permutation = _resolved_permutation(
            model_ir,
            graph_index,
            consumer,
            rank=rank,
        )
        if (
            int(input_slot) == 0
            and candidate_permutation == inverse_permutation
        ):
            posts.append(consumer)
        else:
            legacy_slots.append((consumer, int(input_slot)))
    posts = sorted(
        {id(operator): operator for operator in posts}.values(),
        key=lambda operator: int(_operator_index(graph_index, operator) or 0),
    )
    if len(posts) == 0:
        return None

    post_output_names = []
    public_post_names = []
    for post in posts:
        post_index = _operator_index(graph_index, post)
        if post_index is None or len(post.outputs) != 1:
            return None
        post_output_name = str(post.outputs[0])
        post_output_names.append(post_output_name)
        if post_output_name in public_outputs:
            public_post_names.append(post_output_name)
        post_tensor = model_ir.tensors.get(post_output_name)
        if (
            post_output_name in public_inputs
            or post_output_name in graph_index.duplicate_producers
            or graph_index.producers.get(post_output_name) != int(post_index)
            or not _runtime_tensor(post_tensor, rank=rank)
        ):
            return None
        assert post_tensor is not None
        if (
            _view(post_tensor) != source_view
            or _freeze(post_tensor.quantization)
            != _freeze(multiply_tensor.quantization)
            or not _layout_matches(
                _layout_of(post_output_name, post_tensor, layout_state),
                source_physical,
            )
        ):
            return None
        for post_consumer, _ in _consumer_slots(
            model_ir,
            graph_index,
            post_output_name,
        ):
            post_consumer_index = _operator_index(graph_index, post_consumer)
            if (
                post_consumer_index is None
                or int(post_consumer_index) <= int(post_index)
            ):
                return None
    if len(public_post_names) > 1:
        return None
    representative_output_name = (
        str(public_post_names[0])
        if public_post_names
        else str(post_output_names[0])
    )
    representative_post = next(
        post
        for post in posts
        if str(post.outputs[0]) == representative_output_name
    )
    preserve_legacy_boundary = bool(
        legacy_slots or multiply_output_name in public_outputs
    )
    if (
        not preserve_legacy_boundary
        and representative_output_name not in public_outputs
        and len(
            _consumer_slots(
                model_ir,
                graph_index,
                representative_output_name,
            )
        )
        == 0
    ):
        return None

    planned_inputs: Dict[int, list[str]] = {
        id(logistic): [source_name],
        id(multiply): [str(value) for value in multiply.inputs],
    }
    planned_inputs[id(multiply)][int(multiply_data_slot)] = source_name
    changed_consumers: Dict[int, OperatorIR] = {
        id(logistic): logistic,
        id(multiply): multiply,
    }
    for post_output_name in post_output_names:
        if post_output_name == representative_output_name:
            continue
        for consumer, input_slot in _consumer_slots(
            model_ir,
            graph_index,
            post_output_name,
        ):
            changed_consumers[id(consumer)] = consumer
            inputs = planned_inputs.setdefault(
                id(consumer),
                [str(value) for value in consumer.inputs],
            )
            inputs[int(input_slot)] = representative_output_name
    input_rewrites = tuple(
        _InputRewrite(
            operator=operator,
            original_inputs=tuple(str(value) for value in operator.inputs),
            new_inputs=tuple(planned_inputs[id(operator)]),
        )
        for operator in sorted(
            changed_consumers.values(),
            key=lambda candidate: int(
                _operator_index(graph_index, candidate) or 0
            ),
        )
        if tuple(str(value) for value in operator.inputs)
        != tuple(planned_inputs[id(operator)])
    )
    metadata_updates = (
        _MetadataUpdate(
            name=logistic_output_name,
            shape=source_view.shape,
            signature=source_view.signature,
        ),
        _MetadataUpdate(
            name=representative_output_name,
            shape=source_view.shape,
            signature=source_view.signature,
        ),
    )
    relevant_operators = [
        pre,
        logistic,
        multiply,
        *posts,
        *(operator for operator, _ in legacy_slots),
        *changed_consumers.values(),
    ]
    relevant_operators = sorted(
        {id(operator): operator for operator in relevant_operators}.values(),
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
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))
    return _SwishPlan(
        pre=pre,
        logistic=logistic,
        multiply=multiply,
        posts=tuple(posts),
        adapter_template=representative_post,
        source_name=source_name,
        pre_output_name=pre_output_name,
        logistic_output_name=logistic_output_name,
        multiply_output_name=multiply_output_name,
        representative_output_name=representative_output_name,
        pre_permutation_name=str(pre.inputs[1]),
        permutation=tuple(permutation),
        inverse_permutation=tuple(inverse_permutation),
        preserve_legacy_boundary=preserve_legacy_boundary,
        source_logical_layout=source_logical,
        source_physical_layout=source_physical,
        input_rewrites=input_rewrites,
        metadata_updates=metadata_updates,
        removals=(pre, *posts),
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in relevant_operators
        ),
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _SwishPlan,
    *,
    layout_state: Optional[LayoutState],
) -> bool:
    current = _resolve_candidate(
        model_ir,
        graph_index,
        plan.pre,
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
    ):
        return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))
    if (
        plan.representative_output_name not in model_ir.tensors
        or plan.logistic_output_name not in model_ir.tensors
        or plan.multiply_output_name not in model_ir.tensors
    ):
        return False

    for rewrite in plan.input_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    graph_index.remove_operators(removal_indices)
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.multiply,
        new_outputs=[plan.representative_output_name],
        graph_index=graph_index,
    )
    if plan.preserve_legacy_boundary:
        multiply_index = _operator_index(graph_index, plan.multiply)
        if multiply_index is None:
            raise RuntimeError("validated Swish multiply disappeared")
        template = plan.adapter_template
        graph_index.insert_operator(
            int(multiply_index) + 1,
            OperatorIR(
                op_type=str(template.op_type),
                inputs=[
                    plan.representative_output_name,
                    plan.pre_permutation_name,
                ],
                outputs=[plan.multiply_output_name],
                options=copy.deepcopy(template.options),
                axis_semantics=copy.deepcopy(template.axis_semantics),
                version=int(template.version),
                onnx_node_name=template.onnx_node_name,
                onnx_op_type=template.onnx_op_type,
            ),
        )
    for metadata in plan.metadata_updates:
        tensor = model_ir.tensors[metadata.name]
        tensor.shape = [int(value) for value in metadata.shape]
        tensor.shape_signature = [int(value) for value in metadata.signature]
        tensor.logical_layout = str(plan.source_logical_layout)
        tensor.physical_layout = str(plan.source_physical_layout)
        if layout_state is not None:
            layout_state.set(
                metadata.name,
                logical=plan.source_logical_layout,
                physical=plan.source_physical_layout,
            )
    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return True


def optimize_swish_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Fold fully classified transpose-wrapped pseudo-Swish islands."""

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
        return {_SWISH_STATS_KEY: 0}
    rewritten = 0
    for pre in candidates:
        if rewritten >= rewrite_limit:
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
            rewritten += 1
    return {_SWISH_STATS_KEY: int(rewritten)}


@dataclass(frozen=True)
class _GeluPlan:
    pre: OperatorIR
    chain: Tuple[OperatorIR, ...]
    posts: Tuple[OperatorIR, ...]
    adapter_template: OperatorIR
    source_name: str
    final_output_name: str
    representative_output_name: str
    pre_permutation_name: str
    permutation: Tuple[int, ...]
    inverse_permutation: Tuple[int, ...]
    preserve_legacy_boundary: bool
    source_logical_layout: str
    source_physical_layout: str
    input_rewrites: Tuple[_InputRewrite, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _gelu_plan_signature(plan: _GeluPlan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        tuple(id(operator) for operator in plan.chain),
        tuple(id(operator) for operator in plan.posts),
        id(plan.adapter_template),
        plan.source_name,
        plan.final_output_name,
        plan.representative_output_name,
        plan.pre_permutation_name,
        plan.permutation,
        plan.inverse_permutation,
        plan.preserve_legacy_boundary,
        plan.source_logical_layout,
        plan.source_physical_layout,
        tuple(
            (id(rewrite.operator), rewrite.original_inputs, rewrite.new_inputs)
            for rewrite in plan.input_rewrites
        ),
        tuple(
            (update.name, update.shape, update.signature)
            for update in plan.metadata_updates
        ),
        tuple(id(operator) for operator in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _sole_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Optional[OperatorIR]:
    slots = _consumer_slots(model_ir, graph_index, str(name))
    if len(slots) != 1:
        return None
    return slots[0][0]


def _singleton_constant_contract(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    expected_dtype: str,
    graph_inputs: set[str],
    graph_outputs: set[str],
) -> Optional[TensorIR]:
    normalized_name = str(name)
    tensor = model_ir.tensors.get(normalized_name)
    if (
        tensor is None
        or tensor.data is None
        or bool(tensor.is_variable)
        or normalized_name in graph_inputs | graph_outputs
        or normalized_name in graph_index.producers
        or normalized_name in graph_index.duplicate_producers
        or str(tensor.dtype) != str(expected_dtype)
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return None
    try:
        data = np.asarray(tensor.data)
        signature = (
            tensor.shape
            if tensor.shape_signature is None
            else tensor.shape_signature
        )
        if (
            int(data.size) != 1
            or str(data.dtype).upper() != str(tensor.dtype).upper()
            or len(tensor.shape) != len(signature)
            or any(int(value) <= 0 for value in tensor.shape)
            or int(np.prod(tensor.shape, dtype=np.int64)) != 1
        ):
            return None
    except Exception:
        return None
    return tensor


def _binary_side_name(
    operator: OperatorIR,
    data_name: str,
) -> Optional[str]:
    if len(operator.inputs) != 2 or len(operator.outputs) != 1:
        return None
    inputs = tuple(str(value) for value in operator.inputs)
    if inputs[0] == str(data_name) and inputs[1] != str(data_name):
        return inputs[1]
    if inputs[1] == str(data_name) and inputs[0] != str(data_name):
        return inputs[0]
    return None


def _resolve_gelu_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_GeluPlan]:
    pre_index = _operator_index(graph_index, pre)
    if (
        pre_index is None
        or _op_type(pre) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
    ):
        return None
    graph_inputs_tuple = tuple(str(value) for value in model_ir.inputs)
    graph_outputs_tuple = tuple(str(value) for value in model_ir.outputs)
    graph_inputs = set(graph_inputs_tuple)
    graph_outputs = set(graph_outputs_tuple)
    source_name = str(pre.inputs[0])
    pre_output_name = str(pre.outputs[0])
    if (
        pre_output_name in graph_inputs | graph_outputs
        or pre_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(pre_output_name) != int(pre_index)
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
    if source_tensor is None or pre_tensor is None:
        return None
    rank = len(source_tensor.shape)
    if not _runtime_tensor(
        source_tensor,
        rank=rank,
        allow_constant=True,
    ) or not _runtime_tensor(pre_tensor, rank=rank):
        return None
    permutation = _resolved_permutation(
        model_ir,
        graph_index,
        pre,
        rank=rank,
    )
    if permutation is None:
        return None
    inverse_permutation = _inverse_permutation(permutation)
    if inverse_permutation is None:
        return None
    source_view = _view(source_tensor)
    pre_view = _view(pre_tensor)
    if (
        _permuted_view(source_view, permutation) != pre_view
        or _freeze(source_tensor.quantization) != _freeze(pre_tensor.quantization)
    ):
        return None
    source_logical, source_physical = _state_layouts(
        source_name,
        source_tensor,
        layout_state,
    )
    pre_logical, pre_physical = _state_layouts(
        pre_output_name,
        pre_tensor,
        layout_state,
    )
    if not _layout_transition_matches(
        source_physical,
        pre_physical,
        permutation,
    ) or not _layout_transition_matches(
        source_logical,
        pre_logical,
        permutation,
    ):
        return None

    square_candidates = []
    for operator_index in sorted(
        set(graph_index.consumer_indices(pre_output_name))
    ):
        operator = model_ir.operators[int(operator_index)]
        if (
            _op_type(operator) == "MUL"
            and len(operator.inputs) == 2
            and len(operator.outputs) == 1
            and all(
                str(value) == pre_output_name for value in operator.inputs
            )
        ):
            square_candidates.append(operator)
    for square in square_candidates:
        square_index = _operator_index(graph_index, square)
        if square_index is None or int(square_index) <= int(pre_index):
            continue
        square_output = str(square.outputs[0])
        cube = _sole_consumer(model_ir, graph_index, square_output)
        if (
            cube is None
            or _op_type(cube) != "MUL"
            or len(cube.inputs) != 2
            or len(cube.outputs) != 1
            or sorted(str(value) for value in cube.inputs)
            != sorted([pre_output_name, square_output])
        ):
            continue
        cube_output = str(cube.outputs[0])
        multiply_cubic = _sole_consumer(
            model_ir,
            graph_index,
            cube_output,
        )
        if multiply_cubic is None or _op_type(multiply_cubic) != "MUL":
            continue
        cubic_constant = _binary_side_name(multiply_cubic, cube_output)
        if cubic_constant is None:
            continue
        multiply_cubic_output = str(multiply_cubic.outputs[0])
        add_residual = _sole_consumer(
            model_ir,
            graph_index,
            multiply_cubic_output,
        )
        if (
            add_residual is None
            or _op_type(add_residual) != "ADD"
            or len(add_residual.inputs) != 2
            or len(add_residual.outputs) != 1
            or sorted(str(value) for value in add_residual.inputs)
            != sorted([pre_output_name, multiply_cubic_output])
        ):
            continue
        add_residual_output = str(add_residual.outputs[0])
        multiply_scale = _sole_consumer(
            model_ir,
            graph_index,
            add_residual_output,
        )
        if multiply_scale is None or _op_type(multiply_scale) != "MUL":
            continue
        scale_constant = _binary_side_name(
            multiply_scale,
            add_residual_output,
        )
        if scale_constant is None:
            continue
        multiply_scale_output = str(multiply_scale.outputs[0])
        tanh = _sole_consumer(
            model_ir,
            graph_index,
            multiply_scale_output,
        )
        if (
            tanh is None
            or _op_type(tanh) != "TANH"
            or len(tanh.inputs) != 1
            or len(tanh.outputs) != 1
            or str(tanh.inputs[0]) != multiply_scale_output
        ):
            continue
        tanh_output = str(tanh.outputs[0])
        add_one = _sole_consumer(model_ir, graph_index, tanh_output)
        if add_one is None or _op_type(add_one) != "ADD":
            continue
        one_constant = _binary_side_name(add_one, tanh_output)
        if one_constant is None:
            continue
        add_one_output = str(add_one.outputs[0])
        multiply_residual = _sole_consumer(
            model_ir,
            graph_index,
            add_one_output,
        )
        if (
            multiply_residual is None
            or _op_type(multiply_residual) != "MUL"
            or len(multiply_residual.inputs) != 2
            or len(multiply_residual.outputs) != 1
            or sorted(str(value) for value in multiply_residual.inputs)
            != sorted([pre_output_name, add_one_output])
        ):
            continue
        multiply_residual_output = str(multiply_residual.outputs[0])
        multiply_half = _sole_consumer(
            model_ir,
            graph_index,
            multiply_residual_output,
        )
        if multiply_half is None or _op_type(multiply_half) != "MUL":
            continue
        half_constant = _binary_side_name(
            multiply_half,
            multiply_residual_output,
        )
        if half_constant is None:
            continue
        chain = (
            square,
            cube,
            multiply_cubic,
            add_residual,
            multiply_scale,
            tanh,
            add_one,
            multiply_residual,
            multiply_half,
        )
        chain_indices = [_operator_index(graph_index, operator) for operator in chain]
        if (
            any(value is None for value in chain_indices)
            or len({int(value) for value in chain_indices if value is not None})
            != len(chain_indices)
            or [int(value) for value in chain_indices if value is not None]
            != sorted(int(value) for value in chain_indices if value is not None)
            or int(chain_indices[0]) <= int(pre_index)  # type: ignore[arg-type]
        ):
            continue
        expected_pre_slots = {
            (id(square), 0),
            (id(square), 1),
            (
                id(cube),
                next(
                    index
                    for index, value in enumerate(cube.inputs)
                    if str(value) == pre_output_name
                ),
            ),
            (
                id(add_residual),
                next(
                    index
                    for index, value in enumerate(add_residual.inputs)
                    if str(value) == pre_output_name
                ),
            ),
            (
                id(multiply_residual),
                next(
                    index
                    for index, value in enumerate(multiply_residual.inputs)
                    if str(value) == pre_output_name
                ),
            ),
        }
        actual_pre_slots = {
            (id(operator), int(input_slot))
            for operator, input_slot in _consumer_slots(
                model_ir,
                graph_index,
                pre_output_name,
            )
        }
        if actual_pre_slots != expected_pre_slots:
            continue
        constant_names = (
            cubic_constant,
            scale_constant,
            one_constant,
            half_constant,
        )
        if any(
            _singleton_constant_contract(
                model_ir,
                graph_index,
                name,
                expected_dtype=pre_view.dtype,
                graph_inputs=graph_inputs,
                graph_outputs=graph_outputs,
            )
            is None
            for name in constant_names
        ):
            continue
        chain_output_names = tuple(str(operator.outputs[0]) for operator in chain)
        if len(set(chain_output_names)) != len(chain_output_names):
            continue
        valid_chain = True
        for operator, output_name in zip(chain, chain_output_names):
            operator_index = _operator_index(graph_index, operator)
            tensor = model_ir.tensors.get(output_name)
            if (
                operator_index is None
                or output_name in graph_inputs
                or (
                    output_name in graph_outputs
                    and operator is not multiply_half
                )
                or output_name in graph_index.duplicate_producers
                or graph_index.producers.get(output_name) != int(operator_index)
                or not _runtime_tensor(tensor, rank=rank)
            ):
                valid_chain = False
                break
            assert tensor is not None
            if (
                _view(tensor) != pre_view
                or not _layout_matches(
                    _layout_of(output_name, tensor, layout_state),
                    pre_physical,
                )
            ):
                valid_chain = False
                break
        if not valid_chain:
            continue
        final_output_name = chain_output_names[-1]
        final_tensor = model_ir.tensors[final_output_name]

        posts = []
        legacy_slots = []
        for consumer, input_slot in _consumer_slots(
            model_ir,
            graph_index,
            final_output_name,
        ):
            consumer_index = _operator_index(graph_index, consumer)
            if consumer_index is None or int(consumer_index) <= int(chain_indices[-1]):  # type: ignore[arg-type]
                valid_chain = False
                break
            candidate_permutation = _resolved_permutation(
                model_ir,
                graph_index,
                consumer,
                rank=rank,
            )
            if int(input_slot) == 0 and candidate_permutation == inverse_permutation:
                posts.append(consumer)
            else:
                legacy_slots.append((consumer, int(input_slot)))
        if not valid_chain:
            continue
        posts = sorted(
            {id(operator): operator for operator in posts}.values(),
            key=lambda operator: int(_operator_index(graph_index, operator) or 0),
        )
        if len(posts) == 0:
            continue
        post_output_names = []
        public_post_names = []
        for post in posts:
            post_index = _operator_index(graph_index, post)
            if post_index is None or len(post.outputs) != 1:
                valid_chain = False
                break
            post_output_name = str(post.outputs[0])
            post_output_names.append(post_output_name)
            if post_output_name in graph_outputs:
                public_post_names.append(post_output_name)
            post_tensor = model_ir.tensors.get(post_output_name)
            if (
                post_output_name in graph_inputs
                or post_output_name in graph_index.duplicate_producers
                or graph_index.producers.get(post_output_name) != int(post_index)
                or not _runtime_tensor(post_tensor, rank=rank)
            ):
                valid_chain = False
                break
            assert post_tensor is not None
            if (
                _view(post_tensor) != source_view
                or _freeze(post_tensor.quantization)
                != _freeze(final_tensor.quantization)
                or not _layout_matches(
                    _layout_of(post_output_name, post_tensor, layout_state),
                    source_physical,
                )
            ):
                valid_chain = False
                break
            for post_consumer, _ in _consumer_slots(
                model_ir,
                graph_index,
                post_output_name,
            ):
                post_consumer_index = _operator_index(graph_index, post_consumer)
                if (
                    post_consumer_index is None
                    or int(post_consumer_index) <= int(post_index)
                ):
                    valid_chain = False
                    break
            if not valid_chain:
                break
        if not valid_chain or len(public_post_names) > 1:
            continue
        representative_output_name = (
            str(public_post_names[0])
            if public_post_names
            else str(post_output_names[0])
        )
        representative_post = next(
            post
            for post in posts
            if str(post.outputs[0]) == representative_output_name
        )
        preserve_legacy_boundary = bool(
            legacy_slots or final_output_name in graph_outputs
        )
        if (
            not preserve_legacy_boundary
            and representative_output_name not in graph_outputs
            and len(
                _consumer_slots(
                    model_ir,
                    graph_index,
                    representative_output_name,
                )
            )
            == 0
        ):
            continue

        planned_inputs: Dict[int, list[str]] = {}
        changed_operators: Dict[int, OperatorIR] = {}
        for operator in (square, cube, add_residual, multiply_residual):
            inputs = [
                source_name if str(value) == pre_output_name else str(value)
                for value in operator.inputs
            ]
            planned_inputs[id(operator)] = inputs
            changed_operators[id(operator)] = operator
        for post_output_name in post_output_names:
            if post_output_name == representative_output_name:
                continue
            for consumer, input_slot in _consumer_slots(
                model_ir,
                graph_index,
                post_output_name,
            ):
                changed_operators[id(consumer)] = consumer
                inputs = planned_inputs.setdefault(
                    id(consumer),
                    [str(value) for value in consumer.inputs],
                )
                inputs[int(input_slot)] = representative_output_name
        input_rewrites = tuple(
            _InputRewrite(
                operator=operator,
                original_inputs=tuple(str(value) for value in operator.inputs),
                new_inputs=tuple(planned_inputs[id(operator)]),
            )
            for operator in sorted(
                changed_operators.values(),
                key=lambda candidate: int(
                    _operator_index(graph_index, candidate) or 0
                ),
            )
            if tuple(str(value) for value in operator.inputs)
            != tuple(planned_inputs[id(operator)])
        )
        metadata_updates = tuple(
            _MetadataUpdate(
                name=name,
                shape=source_view.shape,
                signature=source_view.signature,
            )
            for name in (*chain_output_names[:-1], representative_output_name)
        )
        relevant_operators = [
            pre,
            *chain,
            *posts,
            *(operator for operator, _ in legacy_slots),
            *changed_operators.values(),
        ]
        relevant_operators = sorted(
            {id(operator): operator for operator in relevant_operators}.values(),
            key=lambda operator: int(_operator_index(graph_index, operator) or 0),
        )
        contract_names = set(constant_names)
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
                valid_chain = False
                break
            tensor_contracts.append(_tensor_contract(name, tensor))
        if not valid_chain:
            continue
        return _GeluPlan(
            pre=pre,
            chain=chain,
            posts=tuple(posts),
            adapter_template=representative_post,
            source_name=source_name,
            final_output_name=final_output_name,
            representative_output_name=representative_output_name,
            pre_permutation_name=str(pre.inputs[1]),
            permutation=tuple(permutation),
            inverse_permutation=tuple(inverse_permutation),
            preserve_legacy_boundary=preserve_legacy_boundary,
            source_logical_layout=source_logical,
            source_physical_layout=source_physical,
            input_rewrites=input_rewrites,
            metadata_updates=metadata_updates,
            removals=(pre, *posts),
            tensor_contracts=tuple(tensor_contracts),
            operator_contracts=tuple(
                _operator_contract(operator) for operator in relevant_operators
            ),
            graph_inputs=graph_inputs_tuple,
            graph_outputs=graph_outputs_tuple,
        )
    return None


def _apply_gelu_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _GeluPlan,
    *,
    layout_state: Optional[LayoutState],
) -> bool:
    current = _resolve_gelu_candidate(
        model_ir,
        graph_index,
        plan.pre,
        layout_state=layout_state,
    )
    if current is None or _gelu_plan_signature(current) != _gelu_plan_signature(plan):
        return False
    if (
        tuple(str(value) for value in model_ir.inputs) != plan.graph_inputs
        or tuple(str(value) for value in model_ir.outputs) != plan.graph_outputs
        or any(
            tuple(str(value) for value in rewrite.operator.inputs)
            != rewrite.original_inputs
            for rewrite in plan.input_rewrites
        )
    ):
        return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))
    final = plan.chain[-1]
    if (
        plan.representative_output_name not in model_ir.tensors
        or plan.final_output_name not in model_ir.tensors
        or any(metadata.name not in model_ir.tensors for metadata in plan.metadata_updates)
    ):
        return False

    for rewrite in plan.input_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    graph_index.remove_operators(removal_indices)
    _set_operator_outputs(
        model_ir=model_ir,
        op=final,
        new_outputs=[plan.representative_output_name],
        graph_index=graph_index,
    )
    if plan.preserve_legacy_boundary:
        final_index = _operator_index(graph_index, final)
        if final_index is None:
            raise RuntimeError("validated tanh-GELU final multiply disappeared")
        template = plan.adapter_template
        graph_index.insert_operator(
            int(final_index) + 1,
            OperatorIR(
                op_type=str(template.op_type),
                inputs=[
                    plan.representative_output_name,
                    plan.pre_permutation_name,
                ],
                outputs=[plan.final_output_name],
                options=copy.deepcopy(template.options),
                axis_semantics=copy.deepcopy(template.axis_semantics),
                version=int(template.version),
                onnx_node_name=template.onnx_node_name,
                onnx_op_type=template.onnx_op_type,
            ),
        )
    for metadata in plan.metadata_updates:
        tensor = model_ir.tensors[metadata.name]
        tensor.shape = [int(value) for value in metadata.shape]
        tensor.shape_signature = [int(value) for value in metadata.signature]
        tensor.logical_layout = str(plan.source_logical_layout)
        tensor.physical_layout = str(plan.source_physical_layout)
        if layout_state is not None:
            layout_state.set(
                metadata.name,
                logical=plan.source_logical_layout,
                physical=plan.source_physical_layout,
            )
    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return True


def optimize_gelu_tanh_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Fold fully classified transpose-wrapped tanh-GELU islands."""

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
        return {_GELU_STATS_KEY: 0}
    rewritten = 0
    for pre in candidates:
        if rewritten >= rewrite_limit:
            break
        if pre is None or _operator_index(active_index, pre) is None:
            continue
        plan = _resolve_gelu_candidate(
            model_ir,
            active_index,
            pre,
            layout_state=layout_state,
        )
        if plan is None:
            continue
        if _apply_gelu_plan(
            model_ir,
            active_index,
            plan,
            layout_state=layout_state,
        ):
            rewritten += 1
    return {_GELU_STATS_KEY: int(rewritten)}
