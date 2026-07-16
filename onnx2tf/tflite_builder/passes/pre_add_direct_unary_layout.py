from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
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
    _typed_permutation,
    _view,
)


_STATS_KEY = "optimized_transpose_pre_add_direct_unary_nhwc_chains"
_OPTIMIZED_MARKER = "__transpose_pre_add_nhwc_optimized__"
_SKIP_FUSE_MARKER = "__skip_add_activation_fuse__"
_UNARY_TYPES = frozenset(
    {
        "RELU",
        "RELU6",
        "LOGISTIC",
        "TANH",
        "GELU",
        "HARD_SWISH",
        "LEAKY_RELU",
    }
)


@dataclass(frozen=True)
class _BranchPlan:
    kind: str
    adapter: OperatorIR
    original_input: str
    canonical_input: str
    source_name: str
    source_view: _View
    remove_adapter: bool
    unary: Optional[OperatorIR] = None


@dataclass(frozen=True)
class _Plan:
    add: OperatorIR
    original_inputs: Tuple[str, str]
    canonical_inputs: Tuple[str, str]
    add_output: str
    bridge_producer: OperatorIR
    bridge_output: str
    canonical_output: str
    old_add_output_view: _View
    old_bridge_output_view: _View
    new_output_view: _View
    branches: Tuple[_BranchPlan, _BranchPlan]
    posts: Tuple[OperatorIR, ...]
    legacy_users: Tuple[OperatorIR, ...]
    retained_post: Optional[OperatorIR]
    retained_permutation_name: Optional[str]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.add),
        plan.original_inputs,
        plan.canonical_inputs,
        plan.add_output,
        id(plan.bridge_producer),
        plan.bridge_output,
        plan.canonical_output,
        plan.old_add_output_view,
        plan.old_bridge_output_view,
        plan.new_output_view,
        tuple(
            (
                branch.kind,
                id(branch.adapter),
                branch.original_input,
                branch.canonical_input,
                branch.source_name,
                branch.source_view,
                branch.remove_adapter,
                None if branch.unary is None else id(branch.unary),
            )
            for branch in plan.branches
        ),
        tuple(id(post) for post in plan.posts),
        tuple(id(user) for user in plan.legacy_users),
        None if plan.retained_post is None else id(plan.retained_post),
        plan.retained_permutation_name,
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


def _same_quantization(*tensors: TensorIR) -> bool:
    if len(tensors) == 0:
        return True
    return bool(
        all(_per_tensor_quantization(tensor.quantization) for tensor in tensors)
        and all(
            _freeze(tensor.quantization) == _freeze(tensors[0].quantization)
            for tensor in tensors[1:]
        )
    )


def _exclusive_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    operator: OperatorIR,
) -> bool:
    slots = _consumer_slots(model_ir, graph_index, str(name))
    return bool(len(slots) > 0 and all(slot[0] is operator for slot in slots))


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


def _resolve_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    add: OperatorIR,
    input_name: str,
    *,
    old_output_view: _View,
    new_output_view: _View,
    layout_state: Optional[LayoutState],
) -> Optional[_BranchPlan]:
    add_index = _operator_index(graph_index, add)
    producer_index = graph_index.producers.get(str(input_name))
    input_tensor = model_ir.tensors.get(str(input_name))
    graph_outputs = {str(value) for value in model_ir.outputs}
    if (
        add_index is None
        or producer_index is None
        or int(producer_index) >= int(add_index)
        or str(input_name) in graph_index.duplicate_producers
        or str(input_name) in graph_outputs
        or input_tensor is None
        or not _rank4(_view(input_tensor))
        or _view(input_tensor) != old_output_view
        or not _layout_in(
            str(input_name),
            input_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    producer = model_ir.operators[int(producer_index)]
    if graph_index.producer(str(input_name)) is not producer:
        return None

    if _op_type(producer) == "TRANSPOSE":
        if (
            not _typed_permutation(
                model_ir,
                graph_index,
                producer,
                _PERM_NHWC_TO_NCHW,
            )
            or add not in graph_index.consumers_of(str(input_name))
        ):
            return None
        source_name = str(producer.inputs[0])
        source_tensor = model_ir.tensors.get(source_name)
        if (
            source_tensor is None
            or not _resolved_source(
                model_ir,
                graph_index,
                name=source_name,
                before_index=int(producer_index),
            )
            or not _rank4(_view(source_tensor))
            or _view(source_tensor) != new_output_view
            or _permuted_view(_view(source_tensor), _PERM_NHWC_TO_NCHW)
            != _view(input_tensor)
            or not _same_quantization(source_tensor, input_tensor)
            or not _layout_in(
                source_name,
                source_tensor,
                layout_state,
                {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            return None
        return _BranchPlan(
            kind="direct",
            adapter=producer,
            original_input=str(input_name),
            canonical_input=source_name,
            source_name=source_name,
            source_view=_view(source_tensor),
            remove_adapter=_exclusive_consumer(
                model_ir,
                graph_index,
                str(input_name),
                add,
            ),
        )

    if (
        _op_type(producer) not in _UNARY_TYPES
        or len(producer.inputs) != 1
        or len(producer.outputs) != 1
        or not _exclusive_consumer(
            model_ir,
            graph_index,
            str(input_name),
            add,
        )
    ):
        return None
    adapter_output = str(producer.inputs[0])
    adapter_index = graph_index.producers.get(adapter_output)
    adapter_tensor = model_ir.tensors.get(adapter_output)
    if (
        adapter_index is None
        or int(adapter_index) >= int(producer_index)
        or adapter_output in graph_index.duplicate_producers
        or adapter_output in graph_outputs
        or adapter_tensor is None
        or _view(adapter_tensor) != old_output_view
        or not _exclusive_consumer(
            model_ir,
            graph_index,
            adapter_output,
            producer,
        )
    ):
        return None
    adapter = model_ir.operators[int(adapter_index)]
    if (
        graph_index.producer(adapter_output) is not adapter
        or not _typed_permutation(
            model_ir,
            graph_index,
            adapter,
            _PERM_NHWC_TO_NCHW,
        )
    ):
        return None
    source_name = str(adapter.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    if (
        source_tensor is None
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(adapter_index),
        )
        or not _rank4(_view(source_tensor))
        or _view(source_tensor) != new_output_view
        or _permuted_view(_view(source_tensor), _PERM_NHWC_TO_NCHW)
        != _view(adapter_tensor)
        or not _same_quantization(source_tensor, adapter_tensor, input_tensor)
        or not _layout_in(
            source_name,
            source_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not _layout_in(
            adapter_output,
            adapter_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    return _BranchPlan(
        kind="unary",
        adapter=adapter,
        original_input=str(input_name),
        canonical_input=str(input_name),
        source_name=source_name,
        source_view=_view(source_tensor),
        remove_adapter=True,
        unary=producer,
    )


def _owned_retained_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    post: OperatorIR,
    posts: Sequence[OperatorIR],
) -> Optional[str]:
    name = str(post.inputs[1])
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or str(tensor.dtype) != "INT32"
        or tensor.data is None
        or np.asarray(tensor.data).dtype != np.dtype(np.int32)
        or name in {str(value) for value in model_ir.inputs}
        or name in {str(value) for value in model_ir.outputs}
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
    ):
        return None
    owners = {id(value) for value in posts}
    if any(
        id(operator) not in owners
        for operator, _ in _consumer_slots(model_ir, graph_index, name)
    ):
        return None
    return name


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    add: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    add_index = _operator_index(graph_index, add)
    options = add.options if isinstance(add.options, dict) else {}
    if (
        add_index is None
        or _op_type(add) != "ADD"
        or len(add.inputs) != 2
        or len(add.outputs) != 1
        or bool(options.get(_OPTIMIZED_MARKER, False))
    ):
        return None
    add_output = str(add.outputs[0])
    add_output_tensor = model_ir.tensors.get(add_output)
    graph_outputs = {str(value) for value in model_ir.outputs}
    if (
        add_output_tensor is None
        or add_output in graph_index.duplicate_producers
        or graph_index.producer(add_output) is not add
        or add_output in graph_outputs
        or not _rank4(_view(add_output_tensor))
        or not _per_tensor_quantization(add_output_tensor.quantization)
        or not _layout_in(
            add_output,
            add_output_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    old_add_output_view = _view(add_output_tensor)
    new_output_view = _permuted_view(
        old_add_output_view,
        _PERM_NCHW_TO_NHWC,
    )
    if new_output_view is None:
        return None

    bridge_producer = add
    bridge_output = add_output
    add_output_users = graph_index.consumers_of(add_output)
    if len(add_output_users) == 1:
        output_unary = add_output_users[0]
        if (
            _op_type(output_unary) in _UNARY_TYPES
            and len(output_unary.inputs) == 1
            and len(output_unary.outputs) == 1
            and str(output_unary.inputs[0]) == add_output
            and _operator_index(graph_index, output_unary) is not None
        ):
            candidate_bridge_output = str(output_unary.outputs[0])
            bridge_tensor = model_ir.tensors.get(candidate_bridge_output)
            if (
                candidate_bridge_output in graph_outputs
                or candidate_bridge_output in graph_index.duplicate_producers
                or graph_index.producer(candidate_bridge_output) is not output_unary
                or bridge_tensor is None
                or _view(bridge_tensor) != old_add_output_view
                or not _same_quantization(add_output_tensor, bridge_tensor)
                or not _layout_in(
                    candidate_bridge_output,
                    bridge_tensor,
                    layout_state,
                    {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
                )
            ):
                return None
            bridge_producer = output_unary
            bridge_output = candidate_bridge_output

    bridge_output_tensor = model_ir.tensors.get(bridge_output)
    if bridge_output_tensor is None:
        return None
    old_bridge_output_view = _view(bridge_output_tensor)

    posts = []
    legacy_users = []
    bridge_index = _operator_index(graph_index, bridge_producer)
    if bridge_index is None:
        return None
    for consumer_index in graph_index.consumer_indices(bridge_output):
        if int(consumer_index) <= int(bridge_index):
            return None
        consumer = model_ir.operators[int(consumer_index)]
        if (
            _typed_permutation(
                model_ir,
                graph_index,
                consumer,
                _PERM_NCHW_TO_NHWC,
            )
            and str(consumer.inputs[0]) == bridge_output
            and str(consumer.outputs[0]) not in graph_outputs
        ):
            post_tensor = model_ir.tensors.get(str(consumer.outputs[0]))
            if (
                post_tensor is None
                or _view(post_tensor) != new_output_view
                or not _same_quantization(bridge_output_tensor, post_tensor)
                or not _layout_in(
                    str(consumer.outputs[0]),
                    post_tensor,
                    layout_state,
                    {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
                )
            ):
                return None
            posts.append(consumer)
        else:
            legacy_users.append(consumer)
    if len(posts) == 0:
        return None

    branch_plans = []
    for input_name in (str(add.inputs[0]), str(add.inputs[1])):
        branch = _resolve_branch(
            model_ir,
            graph_index,
            add,
            input_name,
            old_output_view=old_add_output_view,
            new_output_view=new_output_view,
            layout_state=layout_state,
        )
        if branch is None:
            return None
        branch_plans.append(branch)
    branches = (branch_plans[0], branch_plans[1])

    retained_post = posts[0] if len(legacy_users) > 0 else None
    retained_permutation_name = None
    if retained_post is not None:
        retained_permutation_name = _owned_retained_permutation(
            model_ir,
            graph_index,
            retained_post,
            posts,
        )
        if retained_permutation_name is None:
            return None

    involved_operators = [add, bridge_producer, *posts, *legacy_users]
    involved_tensors = {add_output, bridge_output}
    for post in posts:
        involved_tensors.update(str(value) for value in post.inputs)
        involved_tensors.update(str(value) for value in post.outputs)
    for branch in branches:
        involved_operators.append(branch.adapter)
        involved_tensors.update(
            {
                branch.original_input,
                branch.canonical_input,
                branch.source_name,
            }
        )
        involved_tensors.update(str(value) for value in branch.adapter.inputs)
        if branch.unary is not None:
            involved_operators.append(branch.unary)
    involved_operators = list(_deduplicate_operators(involved_operators))
    involved_operators.sort(
        key=lambda operator: int(_operator_index(graph_index, operator) or 0)
    )
    return _Plan(
        add=add,
        original_inputs=(str(add.inputs[0]), str(add.inputs[1])),
        canonical_inputs=(
            branches[0].canonical_input,
            branches[1].canonical_input,
        ),
        add_output=add_output,
        bridge_producer=bridge_producer,
        bridge_output=bridge_output,
        canonical_output=str(posts[0].outputs[0]),
        old_add_output_view=old_add_output_view,
        old_bridge_output_view=old_bridge_output_view,
        new_output_view=new_output_view,
        branches=branches,
        posts=tuple(posts),
        legacy_users=tuple(_deduplicate_operators(legacy_users)),
        retained_post=retained_post,
        retained_permutation_name=retained_permutation_name,
        tensor_contracts=tuple(
            _tensor_contract(name, model_ir.tensors[name])
            for name in sorted(involved_tensors)
            if name in model_ir.tensors
        ),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in involved_operators
        ),
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
        plan.add,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if (
        tuple(str(value) for value in model_ir.inputs) != plan.graph_inputs
        or tuple(str(value) for value in model_ir.outputs) != plan.graph_outputs
    ):
        return False

    removals = [
        branch.adapter for branch in plan.branches if branch.remove_adapter
    ]
    removals.extend(
        post for post in plan.posts if post is not plan.retained_post
    )
    removals = list(_deduplicate_operators(removals))
    removal_indices = []
    for operator in removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))

    seen_unaries: set[int] = set()
    for branch in plan.branches:
        if branch.unary is None or id(branch.unary) in seen_unaries:
            continue
        seen_unaries.add(id(branch.unary))
        _set_operator_inputs(
            model_ir=model_ir,
            op=branch.unary,
            new_inputs=[branch.source_name],
            graph_index=graph_index,
        )
        unary_tensor = model_ir.tensors[branch.original_input]
        unary_tensor.shape = [int(value) for value in branch.source_view.shape]
        unary_tensor.shape_signature = [
            int(value) for value in branch.source_view.signature
        ]
        _set_layout(
            unary_tensor,
            branch.original_input,
            LOGICAL_LAYOUT_NHWC,
            layout_state,
        )

    for branch in plan.branches:
        _set_layout(
            model_ir.tensors[branch.source_name],
            branch.source_name,
            LOGICAL_LAYOUT_NHWC,
            layout_state,
        )

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.add,
        new_inputs=list(plan.canonical_inputs),
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.bridge_producer,
        new_outputs=[plan.canonical_output],
        graph_index=graph_index,
    )
    add_options = dict(plan.add.options) if isinstance(plan.add.options, dict) else {}
    add_options[_SKIP_FUSE_MARKER] = True
    add_options[_OPTIMIZED_MARKER] = True
    plan.add.options = add_options

    add_output_tensor = model_ir.tensors[plan.add_output]
    bridge_output_tensor = model_ir.tensors[plan.bridge_output]
    canonical_tensor = model_ir.tensors[plan.canonical_output]
    canonical_tensor.dtype = str(bridge_output_tensor.dtype)
    canonical_tensor.quantization = _clone_quantization(
        bridge_output_tensor.quantization
    )
    canonical_tensor.shape = [int(value) for value in plan.new_output_view.shape]
    canonical_tensor.shape_signature = [
        int(value) for value in plan.new_output_view.signature
    ]
    _set_layout(
        canonical_tensor,
        plan.canonical_output,
        LOGICAL_LAYOUT_NHWC,
        layout_state,
    )
    if plan.bridge_producer is not plan.add:
        add_output_tensor.shape = [
            int(value) for value in plan.new_output_view.shape
        ]
        add_output_tensor.shape_signature = [
            int(value) for value in plan.new_output_view.signature
        ]
        _set_layout(
            add_output_tensor,
            plan.add_output,
            LOGICAL_LAYOUT_NHWC,
            layout_state,
        )

    for post in plan.posts[1:]:
        _replace_tensor_inputs(
            model_ir,
            str(post.outputs[0]),
            plan.canonical_output,
            graph_index=graph_index,
        )

    if plan.retained_post is not None:
        if plan.retained_permutation_name is None:
            raise RuntimeError("validated retained Add adapter lost its permutation")
        permutation_tensor = model_ir.tensors[plan.retained_permutation_name]
        permutation_tensor.data = np.asarray(
            _PERM_NHWC_TO_NCHW,
            dtype=np.int32,
        )
        _set_operator_inputs(
            model_ir=model_ir,
            op=plan.retained_post,
            new_inputs=[plan.canonical_output, plan.retained_permutation_name],
            graph_index=graph_index,
        )
        _set_operator_outputs(
            model_ir=model_ir,
            op=plan.retained_post,
            new_outputs=[plan.bridge_output],
            graph_index=graph_index,
        )
        _set_layout(
            bridge_output_tensor,
            plan.bridge_output,
            LOGICAL_LAYOUT_NCHW,
            layout_state,
        )

    graph_index.remove_operators(removal_indices)
    return True


def optimize_transpose_pre_add_direct_unary_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Index strict direct/unary residual Add adapter elimination.

    The broader compatibility rule still owns Swish, Gather, affine, broadcast,
    PReLU, nested-Add, and direct-fallback patterns. This pass deliberately
    rejects those families before mutation so they retain their historical path.
    """

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
            for index in active_index.operator_indices_for_normalized_types({"ADD"})
        ]
    )
    rewrite_limit = (
        len(candidates) if max_rewrites is None else max(0, int(max_rewrites))
    )
    rewritten = 0
    for add in candidates:
        if rewritten >= rewrite_limit:
            break
        if add is None or _operator_index(active_index, add) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            add,
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
