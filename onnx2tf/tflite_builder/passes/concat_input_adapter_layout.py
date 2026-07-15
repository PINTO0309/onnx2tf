from __future__ import annotations

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


_STATS_KEY = "optimized_transpose_input_chains_pre_concat_to_single_post_adapter"
_UNARY_TYPES = frozenset(
    {
        "LOGISTIC",
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "ELU",
        "LEAKY_RELU",
        "TANH",
        "GELU",
        "HARD_SWISH",
        "ABS",
        "EXP",
        "NEG",
        "SQRT",
    }
)


@dataclass(frozen=True)
class _MetadataUpdate:
    name: str
    old_view: _View
    new_view: _View


@dataclass(frozen=True)
class _BranchPlan:
    kind: str
    adapter: OperatorIR
    source_name: str
    adapter_output_name: str
    canonical_input_name: str
    canonical_view: _View
    unary: Optional[OperatorIR] = None
    metadata: Optional[_MetadataUpdate] = None


@dataclass(frozen=True)
class _Plan:
    concat: OperatorIR
    original_inputs: Tuple[str, ...]
    canonical_inputs: Tuple[str, ...]
    original_output: str
    canonical_output: str
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
        id(plan.concat),
        plan.original_inputs,
        plan.canonical_inputs,
        plan.original_output,
        plan.canonical_output,
        plan.output_old_view,
        plan.output_new_view,
        tuple(
            (
                branch.kind,
                id(branch.adapter),
                branch.source_name,
                branch.adapter_output_name,
                branch.canonical_input_name,
                branch.canonical_view,
                None if branch.unary is None else id(branch.unary),
                (
                    None
                    if branch.metadata is None
                    else (
                        branch.metadata.name,
                        branch.metadata.old_view,
                        branch.metadata.new_view,
                    )
                ),
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


def _rank4(view: _View) -> bool:
    return bool(len(view.shape) == 4 and len(view.signature) == 4)


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


def _concat_view(views: Sequence[_View]) -> Optional[_View]:
    if len(views) == 0 or any(not _rank4(view) for view in views):
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


def _find_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[str]:
    target = tuple(int(value) for value in _PERM_NHWC_TO_NCHW)
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


def _reshape_values_match(
    values: Sequence[int],
    input_view: _View,
    output_view: _View,
) -> bool:
    normalized = tuple(int(value) for value in values)
    if len(normalized) != 4 or normalized.count(-1) > 1:
        return False
    if any(value == 0 or value < -1 for value in normalized):
        return False
    for value, expected in zip(normalized, output_view.shape):
        if value >= 0 and int(value) != int(expected):
            return False
    input_elements = int(np.prod(input_view.shape, dtype=np.int64))
    output_elements = int(np.prod(output_view.shape, dtype=np.int64))
    if input_elements != output_elements:
        return False
    known_elements = int(
        np.prod([value for value in normalized if value >= 0], dtype=np.int64)
    )
    return bool(
        -1 not in normalized
        or (known_elements > 0 and output_elements % known_elements == 0)
    )


def _supported_singleton_reshape(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    input_view: _View,
    output_view: _View,
) -> bool:
    if (
        _op_type(operator) != "RESHAPE"
        or len(operator.inputs) not in {1, 2}
        or len(operator.outputs) != 1
        or not _rank4(input_view)
        or not _rank4(output_view)
        or int(input_view.shape[3]) != 1
        or int(output_view.shape[1]) != 1
        or _permuted_view(input_view, _PERM_NHWC_TO_NCHW) != output_view
    ):
        return False
    if len(operator.inputs) == 2:
        shape = _typed_constant(
            model_ir,
            graph_index,
            str(operator.inputs[1]),
            shape=(4,),
        )
        return bool(
            shape is not None
            and _reshape_values_match(
                shape[1].reshape(-1),
                input_view,
                output_view,
            )
        )
    if not isinstance(operator.options, dict):
        return False
    values = operator.options.get("newShape")
    if not isinstance(values, (list, tuple)):
        return False
    return _reshape_values_match(values, input_view, output_view)


def _exclusive_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    operator: OperatorIR,
) -> bool:
    slots = _consumer_slots(model_ir, graph_index, str(name))
    return bool(len(slots) > 0 and all(slot[0] is operator for slot in slots))


def _resolve_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    concat: OperatorIR,
    input_name: str,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_BranchPlan]:
    concat_index = _operator_index(graph_index, concat)
    if concat_index is None or str(input_name) in graph_index.duplicate_producers:
        return None
    producer_index = graph_index.producers.get(str(input_name))
    if producer_index is None or int(producer_index) >= int(concat_index):
        return None
    producer = model_ir.operators[int(producer_index)]
    input_tensor = model_ir.tensors.get(str(input_name))
    graph_outputs = {str(value) for value in model_ir.outputs}
    if (
        input_tensor is None
        or str(input_name) in graph_outputs
        or graph_index.producer(str(input_name)) is not producer
        or not _rank4(_view(input_tensor))
        or not _per_tensor_quantization(input_tensor.quantization)
        or not _exclusive_consumer(
            model_ir,
            graph_index,
            str(input_name),
            concat,
        )
        or not _layout_in(
            str(input_name),
            input_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    if _op_type(producer) == "TRANSPOSE":
        if not _typed_permutation(
            model_ir,
            graph_index,
            producer,
            _PERM_NHWC_TO_NCHW,
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
            source_name=source_name,
            adapter_output_name=str(input_name),
            canonical_input_name=source_name,
            canonical_view=_view(source_tensor),
        )

    if (
        _op_type(producer) not in _UNARY_TYPES
        or len(producer.inputs) != 1
        or len(producer.outputs) != 1
    ):
        return None
    adapter_output_name = str(producer.inputs[0])
    if (
        adapter_output_name in graph_index.duplicate_producers
        or adapter_output_name in graph_outputs
        or not _exclusive_consumer(
            model_ir,
            graph_index,
            adapter_output_name,
            producer,
        )
    ):
        return None
    adapter_index = graph_index.producers.get(adapter_output_name)
    if adapter_index is None or int(adapter_index) >= int(producer_index):
        return None
    adapter = model_ir.operators[int(adapter_index)]
    if graph_index.producer(adapter_output_name) is not adapter:
        return None
    adapter_output_tensor = model_ir.tensors.get(adapter_output_name)
    if (
        adapter_output_tensor is None
        or not _rank4(_view(adapter_output_tensor))
        or _view(adapter_output_tensor) != _view(input_tensor)
        or not _same_quantization(adapter_output_tensor, input_tensor)
        or not _layout_in(
            adapter_output_name,
            adapter_output_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    source_name = str(adapter.inputs[0]) if len(adapter.inputs) > 0 else ""
    source_tensor = model_ir.tensors.get(source_name)
    if (
        not source_name
        or source_tensor is None
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(adapter_index),
        )
        or not _rank4(_view(source_tensor))
        or not _same_quantization(source_tensor, adapter_output_tensor)
        or not _layout_in(
            source_name,
            source_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    if _op_type(adapter) == "TRANSPOSE":
        if not _typed_permutation(
            model_ir,
            graph_index,
            adapter,
            _PERM_NHWC_TO_NCHW,
        ):
            return None
    elif not _supported_singleton_reshape(
        model_ir,
        graph_index,
        adapter,
        _view(source_tensor),
        _view(adapter_output_tensor),
    ):
        return None
    if _permuted_view(_view(source_tensor), _PERM_NHWC_TO_NCHW) != _view(
        adapter_output_tensor
    ):
        return None
    new_view = _permuted_view(_view(input_tensor), _PERM_NCHW_TO_NHWC)
    if new_view is None or new_view != _view(source_tensor):
        return None
    return _BranchPlan(
        kind="unary",
        adapter=adapter,
        source_name=source_name,
        adapter_output_name=adapter_output_name,
        canonical_input_name=str(input_name),
        canonical_view=new_view,
        unary=producer,
        metadata=_MetadataUpdate(
            name=str(input_name),
            old_view=_view(input_tensor),
            new_view=new_view,
        ),
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
        or len(concat.inputs) == 0
        or len(concat.outputs) != 1
        or _normalized_axis(concat) != 1
    ):
        return None
    output_name = str(concat.outputs[0])
    output_tensor = model_ir.tensors.get(output_name)
    if (
        output_tensor is None
        or output_name in graph_index.duplicate_producers
        or graph_index.producer(output_name) is not concat
        or not _rank4(_view(output_tensor))
        or not _per_tensor_quantization(output_tensor.quantization)
        or not _layout_in(
            output_name,
            output_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
        or any(
            int(consumer_index) <= int(concat_index)
            for consumer_index in graph_index.consumer_indices(output_name)
        )
    ):
        return None

    branch_cache: Dict[str, _BranchPlan] = {}
    branches = []
    for raw_input_name in concat.inputs:
        input_name = str(raw_input_name)
        branch = branch_cache.get(input_name)
        if branch is None:
            branch = _resolve_branch(
                model_ir,
                graph_index,
                concat,
                input_name,
                layout_state=layout_state,
            )
            if branch is None:
                return None
            branch_cache[input_name] = branch
        branches.append(branch)

    canonical_views = [branch.canonical_view for branch in branches]
    output_old_view = _view(output_tensor)
    output_new_view = _permuted_view(
        output_old_view,
        _PERM_NCHW_TO_NHWC,
    )
    if output_new_view is None or _concat_view(canonical_views) != output_new_view:
        return None

    post_permutation_name = next(
        (
            str(branch.adapter.inputs[1])
            for branch in branches
            if _op_type(branch.adapter) == "TRANSPOSE"
        ),
        None,
    )
    occupied = set(str(name) for name in model_ir.tensors)
    create_post_permutation = False
    if post_permutation_name is None:
        post_permutation_name = _find_permutation(model_ir, graph_index)
        if post_permutation_name is None:
            post_permutation_name = _unique_name(
                "transpose_input_chains_nhwc_to_nchw_perm",
                occupied,
            )
            create_post_permutation = True
    canonical_output = _unique_name(f"{output_name}_nhwc", occupied)

    involved_operators = {id(concat): concat}
    involved_tensors = {output_name}
    for branch in branches:
        involved_operators[id(branch.adapter)] = branch.adapter
        if branch.unary is not None:
            involved_operators[id(branch.unary)] = branch.unary
        involved_tensors.update(
            {
                branch.source_name,
                branch.adapter_output_name,
                branch.canonical_input_name,
            }
        )
        involved_tensors.update(str(value) for value in branch.adapter.inputs[1:])
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
        original_inputs=tuple(str(value) for value in concat.inputs),
        canonical_inputs=tuple(branch.canonical_input_name for branch in branches),
        original_output=output_name,
        canonical_output=canonical_output,
        output_old_view=output_old_view,
        output_new_view=output_new_view,
        branches=tuple(branches),
        post_permutation_name=str(post_permutation_name),
        create_post_permutation=create_post_permutation,
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
    layout_state: Optional[LayoutState],
) -> None:
    model_ir.tensors[str(name)] = TensorIR(
        name=str(name),
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray(_PERM_NHWC_TO_NCHW, dtype=np.int32),
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

    removals = _deduplicate_operators([branch.adapter for branch in plan.branches])
    removal_indices = []
    for operator in removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))
    if _operator_index(graph_index, plan.concat) is None:
        return False
    if any(
        branch.unary is not None and _operator_index(graph_index, branch.unary) is None
        for branch in plan.branches
    ):
        return False

    if plan.create_post_permutation:
        _create_permutation(
            model_ir,
            plan.post_permutation_name,
            layout_state,
        )

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
        if branch.metadata is None:
            raise RuntimeError("validated unary branch lost its metadata plan")
        tensor = model_ir.tensors[branch.metadata.name]
        tensor.shape = [int(value) for value in branch.metadata.new_view.shape]
        tensor.shape_signature = [
            int(value) for value in branch.metadata.new_view.signature
        ]
        _set_layout(
            tensor,
            branch.metadata.name,
            LOGICAL_LAYOUT_NHWC,
            layout_state,
        )

    for branch in plan.branches:
        source_tensor = model_ir.tensors[branch.source_name]
        _set_layout(
            source_tensor,
            branch.source_name,
            LOGICAL_LAYOUT_NHWC,
            layout_state,
        )

    output_tensor = model_ir.tensors[plan.original_output]
    model_ir.tensors[plan.canonical_output] = TensorIR(
        name=plan.canonical_output,
        dtype=str(output_tensor.dtype),
        shape=[int(value) for value in plan.output_new_view.shape],
        shape_signature=[int(value) for value in plan.output_new_view.signature],
        data=None,
        is_variable=False,
        quantization=_clone_quantization(output_tensor.quantization),
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
        onnx_tensor_name=output_tensor.onnx_tensor_name,
    )
    if layout_state is not None:
        layout_state.set(
            plan.canonical_output,
            logical=LOGICAL_LAYOUT_NHWC,
            physical=LOGICAL_LAYOUT_NHWC,
        )
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.concat,
        new_inputs=list(plan.canonical_inputs),
        graph_index=graph_index,
    )
    options = dict(plan.concat.options)
    options["axis"] = 3
    plan.concat.options = options
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.concat,
        new_outputs=[plan.canonical_output],
        graph_index=graph_index,
    )
    _set_layout(
        output_tensor,
        plan.original_output,
        LOGICAL_LAYOUT_NCHW,
        layout_state,
    )

    graph_index.remove_operators(removal_indices)
    concat_index = _operator_index(graph_index, plan.concat)
    if concat_index is None:
        raise RuntimeError("validated Concat disappeared during indexed apply")
    graph_index.insert_operator(
        int(concat_index) + 1,
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[plan.canonical_output, plan.post_permutation_name],
            outputs=[plan.original_output],
            options={},
        ),
    )
    return True


def optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Move closed direct/unary Concat input adapters behind the Concat."""

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

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {_STATS_KEY: int(rewritten)}
