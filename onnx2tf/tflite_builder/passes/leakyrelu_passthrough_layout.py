from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    _inverse_permutation,
    _layout_matches,
    _layout_transition_matches,
    _resolved_permutation,
    _runtime_tensor,
    _singleton_constant_contract,
    _state_layouts,
)
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    _optimize_fuse_pseudo_leakyrelu_chains,
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
    _view,
)


_REWRITE_STATS_KEY = "rewritten_leakyrelu_transpose_passthrough_chains"
_FUSION_STATS_KEY = "fused_pseudo_leakyrelu_chains"


@dataclass(frozen=True)
class _Plan:
    pre: OperatorIR
    negate: OperatorIR
    negative_relu: OperatorIR
    multiply: OperatorIR
    positive_relu: OperatorIR
    subtract: OperatorIR
    posts: Tuple[OperatorIR, ...]
    adapter_template: OperatorIR
    source_name: str
    pre_output_name: str
    negative_output_name: str
    negative_relu_output_name: str
    multiply_output_name: str
    positive_output_name: str
    subtract_output_name: str
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


def _sole_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Optional[OperatorIR]:
    slots = _consumer_slots(model_ir, graph_index, str(name))
    return slots[0][0] if len(slots) == 1 else None


def _private_runtime_output(
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
    public = {str(value) for value in (*model_ir.inputs, *model_ir.outputs)}
    return bool(
        operator_index is not None
        and str(name) not in public
        and str(name) not in graph_index.duplicate_producers
        and graph_index.producers.get(str(name)) == int(operator_index)
        and _runtime_tensor(tensor, rank=len(expected_view.shape))
        and _view(tensor) == expected_view  # type: ignore[arg-type]
        and _freeze(tensor.quantization) == expected_quantization  # type: ignore[union-attr]
        and _layout_matches(
            _layout_of(str(name), tensor, layout_state),  # type: ignore[arg-type]
            expected_layout,
        )
    )


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        id(plan.negate),
        id(plan.negative_relu),
        id(plan.multiply),
        id(plan.positive_relu),
        id(plan.subtract),
        tuple(id(operator) for operator in plan.posts),
        id(plan.adapter_template),
        plan.source_name,
        plan.pre_output_name,
        plan.negative_output_name,
        plan.negative_relu_output_name,
        plan.multiply_output_name,
        plan.positive_output_name,
        plan.subtract_output_name,
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
) -> Optional[_Plan]:
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
    if (
        not _runtime_tensor(source_tensor, rank=rank, allow_constant=True)
        or not _runtime_tensor(pre_tensor, rank=rank)
    ):
        return None
    permutation = _resolved_permutation(
        model_ir,
        graph_index,
        pre,
        rank=rank,
    )
    inverse_permutation = (
        None if permutation is None else _inverse_permutation(permutation)
    )
    if permutation is None or inverse_permutation is None:
        return None
    source_view = _view(source_tensor)
    pre_view = _view(pre_tensor)
    if (
        _permuted_view(source_view, permutation) != pre_view
        or _freeze(source_tensor.quantization)
        != _freeze(pre_tensor.quantization)
        or not _per_tensor_quantization(source_tensor.quantization)
        or not _per_tensor_quantization(pre_tensor.quantization)
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
    if (
        not _layout_transition_matches(
            source_logical,
            pre_logical,
            permutation,
        )
        or not _layout_transition_matches(
            source_physical,
            pre_physical,
            permutation,
        )
    ):
        return None

    pre_slots = _consumer_slots(model_ir, graph_index, pre_output_name)
    if len(pre_slots) != 2:
        return None
    negate: Optional[OperatorIR] = None
    positive_relu: Optional[OperatorIR] = None
    for operator, input_slot in pre_slots:
        operator_index = _operator_index(graph_index, operator)
        if (
            int(input_slot) != 0
            or operator_index is None
            or int(operator_index) <= int(pre_index)
            or len(operator.inputs) != 1
            or len(operator.outputs) != 1
            or str(operator.inputs[0]) != pre_output_name
        ):
            return None
        if _op_type(operator) == "NEG" and negate is None:
            negate = operator
        elif _op_type(operator) == "RELU" and positive_relu is None:
            positive_relu = operator
        else:
            return None
    if negate is None or positive_relu is None or negate is positive_relu:
        return None
    negate_index = _operator_index(graph_index, negate)
    positive_relu_index = _operator_index(graph_index, positive_relu)
    assert negate_index is not None and positive_relu_index is not None

    negative_output_name = str(negate.outputs[0])
    negative_relu = _sole_consumer(model_ir, graph_index, negative_output_name)
    negative_relu_index = (
        None if negative_relu is None else _operator_index(graph_index, negative_relu)
    )
    if (
        negative_relu is None
        or negative_relu_index is None
        or int(negative_relu_index) <= int(negate_index)
        or _op_type(negative_relu) != "RELU"
        or len(negative_relu.inputs) != 1
        or len(negative_relu.outputs) != 1
        or str(negative_relu.inputs[0]) != negative_output_name
    ):
        return None
    negative_relu_output_name = str(negative_relu.outputs[0])
    multiply = _sole_consumer(
        model_ir,
        graph_index,
        negative_relu_output_name,
    )
    multiply_index = None if multiply is None else _operator_index(graph_index, multiply)
    if (
        multiply is None
        or multiply_index is None
        or int(multiply_index) <= int(negative_relu_index)
        or _op_type(multiply) != "MUL"
        or len(multiply.inputs) != 2
        or len(multiply.outputs) != 1
        or list(str(value) for value in multiply.inputs).count(
            negative_relu_output_name
        )
        != 1
    ):
        return None
    alpha_name = next(
        str(value)
        for value in multiply.inputs
        if str(value) != negative_relu_output_name
    )
    if (
        _singleton_constant_contract(
            model_ir,
            graph_index,
            alpha_name,
            expected_dtype=pre_view.dtype,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
        )
        is None
    ):
        return None
    multiply_output_name = str(multiply.outputs[0])
    positive_output_name = str(positive_relu.outputs[0])
    multiply_consumer = _sole_consumer(
        model_ir,
        graph_index,
        multiply_output_name,
    )
    positive_consumer = _sole_consumer(
        model_ir,
        graph_index,
        positive_output_name,
    )
    if (
        multiply_consumer is None
        or positive_consumer is None
        or multiply_consumer is not positive_consumer
    ):
        return None
    subtract = multiply_consumer
    subtract_index = _operator_index(graph_index, subtract)
    if (
        subtract_index is None
        or int(subtract_index) <= max(
            int(multiply_index),
            int(positive_relu_index),
        )
        or _op_type(subtract) != "SUB"
        or len(subtract.inputs) != 2
        or len(subtract.outputs) != 1
        or tuple(str(value) for value in subtract.inputs)
        != (positive_output_name, multiply_output_name)
    ):
        return None
    subtract_output_name = str(subtract.outputs[0])
    chain = (
        negate,
        negative_relu,
        multiply,
        positive_relu,
        subtract,
    )
    chain_output_names = (
        negative_output_name,
        negative_relu_output_name,
        multiply_output_name,
        positive_output_name,
        subtract_output_name,
    )
    if (
        len({id(operator) for operator in chain}) != len(chain)
        or len(set(chain_output_names)) != len(chain_output_names)
        or source_name in chain_output_names
        or pre_output_name in chain_output_names
    ):
        return None
    expected_quantization = _freeze(pre_tensor.quantization)
    for operator, name in zip(chain[:-1], chain_output_names[:-1]):
        if not _private_runtime_output(
            model_ir,
            graph_index,
            operator,
            name,
            expected_view=pre_view,
            expected_quantization=expected_quantization,
            expected_layout=pre_physical,
            layout_state=layout_state,
        ):
            return None
    subtract_tensor = model_ir.tensors.get(subtract_output_name)
    if (
        subtract_output_name in graph_inputs
        or subtract_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(subtract_output_name) != int(subtract_index)
        or not _runtime_tensor(subtract_tensor, rank=rank)
        or _view(subtract_tensor) != pre_view  # type: ignore[arg-type]
        or _freeze(subtract_tensor.quantization) != expected_quantization  # type: ignore[union-attr]
        or not _layout_matches(
            _layout_of(subtract_output_name, subtract_tensor, layout_state),  # type: ignore[arg-type]
            pre_physical,
        )
    ):
        return None

    posts = []
    legacy_slots = []
    for consumer, input_slot in _consumer_slots(
        model_ir,
        graph_index,
        subtract_output_name,
    ):
        consumer_index = _operator_index(graph_index, consumer)
        if consumer_index is None or int(consumer_index) <= int(subtract_index):
            return None
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
    posts = sorted(
        {id(operator): operator for operator in posts}.values(),
        key=lambda operator: int(_operator_index(graph_index, operator) or 0),
    )
    if len(posts) == 0:
        return None
    post_output_names = []
    public_post_names = []
    observable_post_names = []
    for post in posts:
        post_index = _operator_index(graph_index, post)
        if post_index is None or len(post.outputs) != 1:
            return None
        output_name = str(post.outputs[0])
        output_tensor = model_ir.tensors.get(output_name)
        if (
            output_name in graph_inputs
            or output_name in graph_index.duplicate_producers
            or graph_index.producers.get(output_name) != int(post_index)
            or output_name in set(chain_output_names)
            or not _runtime_tensor(output_tensor, rank=rank)
            or _view(output_tensor) != source_view  # type: ignore[arg-type]
            or _freeze(output_tensor.quantization)  # type: ignore[union-attr]
            != expected_quantization
            or not _layout_matches(
                _layout_of(output_name, output_tensor, layout_state),  # type: ignore[arg-type]
                source_physical,
            )
        ):
            return None
        post_output_names.append(output_name)
        if output_name in graph_outputs:
            public_post_names.append(output_name)
        if output_name in graph_outputs or len(
            _consumer_slots(model_ir, graph_index, output_name)
        ) > 0:
            observable_post_names.append(output_name)
        for post_consumer, _ in _consumer_slots(
            model_ir,
            graph_index,
            output_name,
        ):
            post_consumer_index = _operator_index(graph_index, post_consumer)
            if post_consumer_index is None or int(post_consumer_index) <= int(post_index):
                return None
    if len(public_post_names) > 1:
        return None
    preserve_legacy_boundary = bool(
        legacy_slots or subtract_output_name in graph_outputs
    )
    if public_post_names:
        representative_output_name = public_post_names[0]
    elif observable_post_names:
        representative_output_name = observable_post_names[0]
    elif preserve_legacy_boundary:
        representative_output_name = post_output_names[0]
    else:
        return None
    adapter_template = next(
        post
        for post in posts
        if str(post.outputs[0]) == representative_output_name
    )

    planned_inputs: Dict[int, list[str]] = {
        id(negate): [source_name],
        id(positive_relu): [source_name],
    }
    changed_operators: Dict[int, OperatorIR] = {
        id(negate): negate,
        id(positive_relu): positive_relu,
    }
    for output_name in post_output_names:
        if output_name == representative_output_name:
            continue
        for consumer, input_slot in _consumer_slots(
            model_ir,
            graph_index,
            output_name,
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
        for name in (
            negative_output_name,
            negative_relu_output_name,
            multiply_output_name,
            positive_output_name,
            representative_output_name,
        )
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
    contract_names = {str(pre.inputs[1]), alpha_name}
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
        pre=pre,
        negate=negate,
        negative_relu=negative_relu,
        multiply=multiply,
        positive_relu=positive_relu,
        subtract=subtract,
        posts=tuple(posts),
        adapter_template=adapter_template,
        source_name=source_name,
        pre_output_name=pre_output_name,
        negative_output_name=negative_output_name,
        negative_relu_output_name=negative_relu_output_name,
        multiply_output_name=multiply_output_name,
        positive_output_name=positive_output_name,
        subtract_output_name=subtract_output_name,
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
        or plan.representative_output_name not in model_ir.tensors
        or plan.subtract_output_name not in model_ir.tensors
        or any(
            update.name not in model_ir.tensors
            for update in plan.metadata_updates
        )
    ):
        return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))

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
        op=plan.subtract,
        new_outputs=[plan.representative_output_name],
        graph_index=graph_index,
    )
    if plan.preserve_legacy_boundary:
        subtract_index = _operator_index(graph_index, plan.subtract)
        if subtract_index is None:
            raise RuntimeError("validated pseudo-LeakyReLU subtract disappeared")
        template = plan.adapter_template
        graph_index.insert_operator(
            int(subtract_index) + 1,
            OperatorIR(
                op_type=str(template.op_type),
                inputs=[
                    plan.representative_output_name,
                    plan.pre_permutation_name,
                ],
                outputs=[plan.subtract_output_name],
                options=copy.deepcopy(template.options),
                axis_semantics=copy.deepcopy(template.axis_semantics),
                version=int(template.version),
                onnx_node_name=template.onnx_node_name,
                onnx_op_type=template.onnx_op_type,
            ),
        )
    for update in plan.metadata_updates:
        tensor = model_ir.tensors[update.name]
        tensor.shape = [int(value) for value in update.shape]
        tensor.shape_signature = [int(value) for value in update.signature]
        tensor.logical_layout = str(plan.source_logical_layout)
        tensor.physical_layout = str(plan.source_physical_layout)
        if layout_state is not None:
            layout_state.set(
                update.name,
                logical=plan.source_logical_layout,
                physical=plan.source_physical_layout,
            )
    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return True


def optimize_leakyrelu_transpose_passthrough(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Recover source layout for fully classified pseudo-LeakyReLU islands."""

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
    rewritten = 0
    if rewrite_limit > 0:
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
    return {_REWRITE_STATS_KEY: int(rewritten)}


def optimize_leakyrelu_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Recover source layout, then run the existing pseudo-LeakyReLU fusion."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    rewrite_stats = optimize_leakyrelu_transpose_passthrough(
        model_ir,
        graph_index=active_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )
    fuse_stats = _optimize_fuse_pseudo_leakyrelu_chains(
        model_ir,
        graph_index=active_index,
        layout_state=layout_state,
    )
    return {
        _REWRITE_STATS_KEY: int(rewrite_stats.get(_REWRITE_STATS_KEY, 0)),
        _FUSION_STATS_KEY: int(fuse_stats.get(_FUSION_STATS_KEY, 0)),
    }
