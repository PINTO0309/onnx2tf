from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _clone_quantization,
    _prune_unused_tensors,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_onnx_shape,
)
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    _inverse_permutation,
    _layout_matches,
    _layout_transition_matches,
    _resolved_permutation,
    _runtime_tensor,
    _state_layouts,
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


_STATS_KEY = "rewritten_prelu_transpose_passthrough_chains"


@dataclass(frozen=True)
class _AlphaUpdate:
    original_name: str
    selected_name: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    numpy_dtype: str
    data_contract: Any
    in_place: bool


@dataclass(frozen=True)
class _AdapterPermutation:
    name: str
    rewrite_in_place: bool
    numpy_dtype: str


@dataclass(frozen=True)
class _Plan:
    pre: OperatorIR
    prelu: OperatorIR
    posts: Tuple[OperatorIR, ...]
    adapter_template: OperatorIR
    source_name: str
    pre_output_name: str
    prelu_output_name: str
    representative_output_name: str
    pre_permutation_name: str
    permutation: Tuple[int, ...]
    inverse_permutation: Tuple[int, ...]
    preserve_pre: bool
    preserve_legacy_boundary: bool
    source_logical_layout: str
    source_physical_layout: str
    alpha_update: _AlphaUpdate
    adapter_permutation: _AdapterPermutation
    input_rewrites: Tuple[_InputRewrite, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        id(plan.prelu),
        tuple(id(operator) for operator in plan.posts),
        id(plan.adapter_template),
        plan.source_name,
        plan.pre_output_name,
        plan.prelu_output_name,
        plan.representative_output_name,
        plan.pre_permutation_name,
        plan.permutation,
        plan.inverse_permutation,
        plan.preserve_pre,
        plan.preserve_legacy_boundary,
        plan.source_logical_layout,
        plan.source_physical_layout,
        (
            plan.alpha_update.original_name,
            plan.alpha_update.selected_name,
            plan.alpha_update.shape,
            plan.alpha_update.signature,
            plan.alpha_update.numpy_dtype,
            plan.alpha_update.data_contract,
            plan.alpha_update.in_place,
        ),
        (
            plan.adapter_permutation.name,
            plan.adapter_permutation.rewrite_in_place,
            plan.adapter_permutation.numpy_dtype,
        ),
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


def _numpy_dtype_matches(tensor: TensorIR, data: np.ndarray) -> bool:
    return str(np.asarray(data).dtype).upper() == str(tensor.dtype).upper()


def _broadcasts_to_source(
    candidate_shape: Tuple[int, ...],
    source_shape: Tuple[int, ...],
    source_signature: Tuple[int, ...],
) -> bool:
    static_candidate_shape = candidate_shape if candidate_shape else (1,)
    static = _broadcast_static_shapes(
        [int(value) for value in source_shape],
        [int(value) for value in static_candidate_shape],
    )
    if static != [int(value) for value in source_shape]:
        return False
    padded = (
        (1,) * (len(source_signature) - len(static_candidate_shape))
        + static_candidate_shape
    )
    if len(padded) != len(source_signature):
        return False
    return all(
        int(alpha_dim) == 1
        or int(source_dim) == -1
        or int(alpha_dim) == int(source_dim)
        for alpha_dim, source_dim in zip(padded, source_signature)
    )


def _selected_alpha(
    alpha_data: np.ndarray,
    *,
    permutation: Tuple[int, ...],
    inverse_permutation: Tuple[int, ...],
    source_shape: Tuple[int, ...],
    source_signature: Tuple[int, ...],
) -> Optional[np.ndarray]:
    candidates = []
    if int(alpha_data.ndim) == len(inverse_permutation):
        candidates.append(
            np.transpose(alpha_data, axes=inverse_permutation).astype(
                alpha_data.dtype,
                copy=False,
            )
        )
    candidates.append(np.asarray(alpha_data))
    if (
        permutation == (0, 3, 1, 2)
        and inverse_permutation == (0, 2, 3, 1)
        and int(alpha_data.ndim) == 3
    ):
        candidates.append(
            np.transpose(alpha_data, axes=(1, 2, 0)).astype(
                alpha_data.dtype,
                copy=False,
            )
        )
    for candidate in candidates:
        candidate = np.asarray(candidate)
        if _broadcasts_to_source(
            tuple(int(value) for value in candidate.shape),
            source_shape,
            source_signature,
        ):
            return candidate
    return None


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    name = str(base)
    suffix = 1
    while name in model_ir.tensors:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


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
    ) or not _runtime_tensor(pre_tensor, rank=rank):
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
        or _freeze(source_tensor.quantization) != _freeze(pre_tensor.quantization)
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
    prelu_slots = [
        (operator, int(input_slot))
        for operator, input_slot in pre_slots
        if _op_type(operator) == "PRELU"
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
        and int(input_slot) == 0
        and str(operator.inputs[0]) == pre_output_name
    ]
    if len(prelu_slots) != 1:
        return None
    prelu = prelu_slots[0][0]
    prelu_index = _operator_index(graph_index, prelu)
    if prelu_index is None or int(prelu_index) <= int(pre_index):
        return None
    prelu_output_name = str(prelu.outputs[0])
    if (
        prelu_output_name in public_inputs
        or prelu_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(prelu_output_name) != int(prelu_index)
    ):
        return None
    prelu_tensor = model_ir.tensors.get(prelu_output_name)
    if not _runtime_tensor(prelu_tensor, rank=rank):
        return None
    assert prelu_tensor is not None
    if (
        _view(prelu_tensor) != pre_view
        or _freeze(prelu_tensor.quantization) != _freeze(pre_tensor.quantization)
        or not _layout_matches(
            _layout_of(prelu_output_name, prelu_tensor, layout_state),
            pre_physical,
        )
    ):
        return None

    posts = []
    legacy_slots = []
    for consumer, input_slot in _consumer_slots(
        model_ir,
        graph_index,
        prelu_output_name,
    ):
        consumer_index = _operator_index(graph_index, consumer)
        if consumer_index is None or int(consumer_index) <= int(prelu_index):
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
            or _freeze(post_tensor.quantization) != _freeze(prelu_tensor.quantization)
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
            if post_consumer_index is None or int(post_consumer_index) <= int(post_index):
                return None
    if len(public_post_names) > 1:
        return None
    preserve_legacy_boundary = bool(
        legacy_slots or prelu_output_name in public_outputs
    )
    consumed_post_names = [
        name
        for name in post_output_names
        if _consumer_slots(model_ir, graph_index, name)
    ]
    if (
        not public_post_names
        and not consumed_post_names
        and not preserve_legacy_boundary
    ):
        return None
    representative_output_name = (
        str(public_post_names[0])
        if public_post_names
        else str(
            consumed_post_names[0]
            if consumed_post_names
            else post_output_names[0]
        )
    )
    representative_post = next(
        post
        for post in posts
        if str(post.outputs[0]) == representative_output_name
    )
    post_permutation_name = str(representative_post.inputs[1])
    post_permutation_tensor = model_ir.tensors.get(post_permutation_name)
    if post_permutation_tensor is None or post_permutation_tensor.data is None:
        return None
    post_permutation_slots = _consumer_slots(
        model_ir,
        graph_index,
        post_permutation_name,
    )
    rewrite_post_permutation = bool(
        preserve_legacy_boundary
        and len(post_permutation_slots) == 1
        and post_permutation_slots[0][0] is representative_post
        and int(post_permutation_slots[0][1]) == 1
    )
    adapter_permutation = _AdapterPermutation(
        name=(
            post_permutation_name
            if rewrite_post_permutation
            else str(pre.inputs[1])
        ),
        rewrite_in_place=rewrite_post_permutation,
        numpy_dtype=str(np.asarray(post_permutation_tensor.data).dtype),
    )
    alpha_name = str(prelu.inputs[1])
    alpha_tensor = model_ir.tensors.get(alpha_name)
    if (
        alpha_tensor is None
        or alpha_tensor.data is None
        or not isinstance(alpha_tensor.data, np.ndarray)
        or bool(alpha_tensor.is_variable)
        or alpha_name in graph_index.producers
        or alpha_name in graph_index.duplicate_producers
        or alpha_name in public_inputs | public_outputs
        or not _per_tensor_quantization(alpha_tensor.quantization)
    ):
        return None
    alpha_data = np.asarray(alpha_tensor.data)
    if (
        alpha_data.size == 0
        or not _numpy_dtype_matches(alpha_tensor, alpha_data)
        or str(alpha_tensor.dtype) != str(source_tensor.dtype)
    ):
        return None
    selected_alpha = _selected_alpha(
        alpha_data,
        permutation=tuple(permutation),
        inverse_permutation=tuple(inverse_permutation),
        source_shape=source_view.shape,
        source_signature=source_view.signature,
    )
    if selected_alpha is None:
        return None
    alpha_needs_rewrite = bool(
        selected_alpha.shape != alpha_data.shape
        or not np.array_equal(selected_alpha, alpha_data)
    )
    alpha_slots = _consumer_slots(model_ir, graph_index, alpha_name)
    alpha_in_place = bool(
        alpha_needs_rewrite
        and len(alpha_slots) == 1
        and alpha_slots[0][0] is prelu
        and int(alpha_slots[0][1]) == 1
    )
    selected_alpha_name = alpha_name
    if alpha_needs_rewrite and not alpha_in_place:
        selected_alpha_name = _unique_tensor_name(model_ir, f"{alpha_name}_nhwc")
    shape, signature = normalize_onnx_shape(
        [int(value) for value in selected_alpha.shape]
    )
    alpha_update = _AlphaUpdate(
        original_name=alpha_name,
        selected_name=selected_alpha_name,
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        numpy_dtype=str(selected_alpha.dtype),
        data_contract=_freeze(selected_alpha),
        in_place=alpha_in_place,
    )

    planned_inputs: Dict[int, list[str]] = {
        id(prelu): [source_name, selected_alpha_name],
    }
    changed_consumers: Dict[int, OperatorIR] = {id(prelu): prelu}
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
            name=representative_output_name,
            shape=source_view.shape,
            signature=source_view.signature,
        ),
    )
    preserve_pre = any(operator is not prelu for operator, _ in pre_slots)
    removable_posts = (
        tuple(post for post in posts if post is not representative_post)
        if preserve_legacy_boundary
        else tuple(posts)
    )
    removals = (*(() if preserve_pre else (pre,)), *removable_posts)
    relevant_operators = [
        pre,
        prelu,
        *posts,
        *(operator for operator, _ in pre_slots),
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
    return _Plan(
        pre=pre,
        prelu=prelu,
        posts=tuple(posts),
        adapter_template=representative_post,
        source_name=source_name,
        pre_output_name=pre_output_name,
        prelu_output_name=prelu_output_name,
        representative_output_name=representative_output_name,
        pre_permutation_name=str(pre.inputs[1]),
        permutation=tuple(permutation),
        inverse_permutation=tuple(inverse_permutation),
        preserve_pre=preserve_pre,
        preserve_legacy_boundary=preserve_legacy_boundary,
        source_logical_layout=source_logical,
        source_physical_layout=source_physical,
        alpha_update=alpha_update,
        adapter_permutation=adapter_permutation,
        input_rewrites=input_rewrites,
        metadata_updates=metadata_updates,
        removals=tuple(removals),
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in relevant_operators
        ),
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
    )


def _selected_alpha_data(model_ir: ModelIR, plan: _Plan) -> Optional[np.ndarray]:
    alpha_tensor = model_ir.tensors.get(plan.alpha_update.original_name)
    if alpha_tensor is None or not isinstance(alpha_tensor.data, np.ndarray):
        return None
    selected = _selected_alpha(
        np.asarray(alpha_tensor.data),
        permutation=plan.permutation,
        inverse_permutation=plan.inverse_permutation,
        source_shape=_view(model_ir.tensors[plan.source_name]).shape,
        source_signature=_view(model_ir.tensors[plan.source_name]).signature,
    )
    if (
        selected is None
        or str(selected.dtype) != plan.alpha_update.numpy_dtype
        or _freeze(selected) != plan.alpha_update.data_contract
    ):
        return None
    return selected


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
    ):
        return False
    selected_alpha = _selected_alpha_data(model_ir, plan)
    if selected_alpha is None:
        return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))
    if any(
        name not in model_ir.tensors
        for name in (
            plan.source_name,
            plan.prelu_output_name,
            plan.representative_output_name,
            plan.alpha_update.original_name,
        )
    ):
        return False

    alpha_tensor = model_ir.tensors[plan.alpha_update.original_name]
    alpha_needs_rewrite = bool(
        _freeze(selected_alpha) != _freeze(alpha_tensor.data)
        or tuple(int(value) for value in selected_alpha.shape)
        != tuple(int(value) for value in np.asarray(alpha_tensor.data).shape)
    )
    if alpha_needs_rewrite:
        if plan.alpha_update.in_place:
            alpha_tensor.data = np.asarray(selected_alpha)
            alpha_tensor.shape = [int(value) for value in plan.alpha_update.shape]
            alpha_tensor.shape_signature = [
                int(value) for value in plan.alpha_update.signature
            ]
        else:
            if plan.alpha_update.selected_name in model_ir.tensors:
                return False
            model_ir.tensors[plan.alpha_update.selected_name] = TensorIR(
                name=plan.alpha_update.selected_name,
                dtype=str(alpha_tensor.dtype),
                shape=[int(value) for value in plan.alpha_update.shape],
                shape_signature=[
                    int(value) for value in plan.alpha_update.signature
                ],
                data=np.asarray(selected_alpha),
                is_variable=False,
                quantization=_clone_quantization(alpha_tensor.quantization),
                logical_layout=str(alpha_tensor.logical_layout),
                physical_layout=str(alpha_tensor.physical_layout),
                onnx_tensor_name=alpha_tensor.onnx_tensor_name,
            )
            if layout_state is not None:
                layout_state.set(
                    plan.alpha_update.selected_name,
                    logical=str(alpha_tensor.logical_layout),
                    physical=str(alpha_tensor.physical_layout),
                )

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
        op=plan.prelu,
        new_outputs=[plan.representative_output_name],
        graph_index=graph_index,
    )
    if plan.preserve_legacy_boundary:
        if plan.adapter_permutation.rewrite_in_place:
            permutation_tensor = model_ir.tensors[
                plan.adapter_permutation.name
            ]
            permutation_tensor.data = np.asarray(
                plan.permutation,
                dtype=np.dtype(plan.adapter_permutation.numpy_dtype),
            )
        _set_operator_inputs(
            model_ir=model_ir,
            op=plan.adapter_template,
            new_inputs=[
                plan.representative_output_name,
                plan.adapter_permutation.name,
            ],
            graph_index=graph_index,
        )
        _set_operator_outputs(
            model_ir=model_ir,
            op=plan.adapter_template,
            new_outputs=[plan.prelu_output_name],
            graph_index=graph_index,
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
    return True


def optimize_prelu_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Fold fully classified transpose-wrapped PReLU operators."""

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
    # The former owner pruned once on every invocation, including zero-match
    # calls. Preserve that externally visible tensor-table cleanup contract.
    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {_STATS_KEY: int(rewritten)}


def run_prelu_transpose_passthrough_summary(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Run PRELU passthrough cleanup and retain prune-only evidence."""

    initial_tensor_count = len(model_ir.tensors)
    result = optimize_prelu_transpose_passthrough_chains(
        model_ir,
        layout_state=layout_state,
    )
    return {
        **result,
        "pruned_unused_tensors": max(
            0,
            int(initial_tensor_count - len(model_ir.tensors)),
        ),
    }
