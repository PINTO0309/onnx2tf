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
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
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
    _resolved_source,
    _tensor_contract,
    _typed_constant,
    _typed_permutation,
    _view,
)


_STATS_KEY = "optimized_transpose_stridedslice_pre_concat_nhwc_chains"
_MASK_OPTIONS = (
    "beginMask",
    "endMask",
    "ellipsisMask",
    "newAxisMask",
    "shrinkAxisMask",
)


@dataclass(frozen=True)
class _ConstantUpdate:
    name: str
    original_values: Tuple[int, ...]
    target_values: Tuple[int, ...]
    dtype: str
    numpy_dtype: str
    data_contract: Any


@dataclass(frozen=True)
class _SliceUpdate:
    operator: OperatorIR
    output_name: str
    original_inputs: Tuple[str, ...]
    new_inputs: Tuple[str, ...]
    old_view: _View
    new_view: _View


@dataclass(frozen=True)
class _Plan:
    seed: OperatorIR
    source_name: str
    pre_output_name: str
    pre_permutation_name: str
    slices: Tuple[_SliceUpdate, ...]
    concat: OperatorIR
    posts: Tuple[OperatorIR, ...]
    keep_post: Optional[OperatorIR]
    concat_output_name: str
    canonical_output_name: str
    concat_old_view: _View
    concat_new_view: _View
    constants: Tuple[_ConstantUpdate, ...]
    alias_names: Tuple[str, ...]
    alias_rewrites: Tuple[_InputRewrite, ...]
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.seed),
        plan.source_name,
        plan.pre_output_name,
        plan.pre_permutation_name,
        tuple(
            (
                id(update.operator),
                update.output_name,
                update.original_inputs,
                update.new_inputs,
                update.old_view,
                update.new_view,
            )
            for update in plan.slices
        ),
        id(plan.concat),
        tuple(id(operator) for operator in plan.posts),
        None if plan.keep_post is None else id(plan.keep_post),
        plan.concat_output_name,
        plan.canonical_output_name,
        plan.concat_old_view,
        plan.concat_new_view,
        tuple(
            (
                update.name,
                update.original_values,
                update.target_values,
                update.dtype,
                update.numpy_dtype,
                update.data_contract,
            )
            for update in plan.constants
        ),
        plan.alias_names,
        tuple(
            (id(rewrite.operator), rewrite.original_inputs, rewrite.new_inputs)
            for rewrite in plan.alias_rewrites
        ),
        tuple(id(operator) for operator in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _rank4(view: _View) -> bool:
    return bool(
        len(view.shape) == 4
        and len(view.signature) == 4
        and all(int(value) > 0 for value in view.shape)
    )


def _layout_in(
    name: str,
    model_ir: ModelIR,
    layout_state: Optional[LayoutState],
    allowed: set[str],
) -> bool:
    tensor = model_ir.tensors.get(str(name))
    return bool(
        tensor is not None
        and str(_layout_of(str(name), tensor, layout_state)).upper() in allowed
    )


def _same_quantization(left: Any, right: Any) -> bool:
    return bool(
        _per_tensor_quantization(left)
        and _per_tensor_quantization(right)
        and _freeze(left) == _freeze(right)
    )


def _supported_stridedslice(operator: OperatorIR) -> bool:
    if (
        _op_type(operator) != "STRIDED_SLICE"
        or len(operator.inputs) != 4
        or len(operator.outputs) != 1
        or not isinstance(operator.options, dict)
    ):
        return False
    try:
        if any(int(operator.options.get(name, 0)) != 0 for name in _MASK_OPTIONS):
            return False
    except Exception:
        return False
    return not bool(operator.options.get("offset", False))


def _concat_view(views: Sequence[_View], *, axis: int) -> Optional[_View]:
    if len(views) < 2 or int(axis) not in range(4):
        return None
    if any(not _rank4(view) for view in views):
        return None
    dtype = views[0].dtype
    if any(view.dtype != dtype for view in views[1:]):
        return None

    shape = []
    signature = []
    for dimension in range(4):
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


def _constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    slice_operator: OperatorIR,
    slice_index: int,
) -> Optional[_ConstantUpdate]:
    resolved = _typed_constant(
        model_ir,
        graph_index,
        str(name),
        shape=(4,),
    )
    slots = _consumer_slots(model_ir, graph_index, str(name))
    if (
        resolved is None
        or set(graph_index.consumer_indices(str(name))) != {int(slice_index)}
        or len(slots) == 0
        or any(operator is not slice_operator for operator, _ in slots)
    ):
        return None
    tensor, data = resolved
    original_values = tuple(int(value) for value in data.reshape(-1))
    target_values = tuple(
        int(original_values[index]) for index in _PERM_NCHW_TO_NHWC
    )
    return _ConstantUpdate(
        name=str(name),
        original_values=original_values,
        target_values=target_values,
        dtype=str(tensor.dtype),
        numpy_dtype=str(data.dtype),
        data_contract=_freeze(data),
    )


def _append_unique_operators(
    target: list[OperatorIR],
    operators: Sequence[OperatorIR],
) -> None:
    occupied = {id(operator) for operator in target}
    for operator in operators:
        if id(operator) not in occupied:
            target.append(operator)
            occupied.add(id(operator))


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    seed: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    pre_index = _operator_index(graph_index, seed)
    if (
        pre_index is None
        or not _typed_permutation(
            model_ir,
            graph_index,
            seed,
            _PERM_NHWC_TO_NCHW,
        )
    ):
        return None
    source_name = str(seed.inputs[0])
    pre_output_name = str(seed.outputs[0])
    pre_permutation_name = str(seed.inputs[1])
    source_tensor = model_ir.tensors.get(source_name)
    pre_output_tensor = model_ir.tensors.get(pre_output_name)
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    if (
        source_tensor is None
        or pre_output_tensor is None
        or pre_output_name in public_outputs
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(pre_index),
        )
        or not _rank4(_view(source_tensor))
        or not _rank4(_view(pre_output_tensor))
        or _permuted_view(_view(source_tensor), _PERM_NHWC_TO_NCHW)
        != _view(pre_output_tensor)
        or not _same_quantization(
            source_tensor.quantization,
            pre_output_tensor.quantization,
        )
        or not _layout_in(
            source_name,
            model_ir,
            layout_state,
            {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not _layout_in(
            pre_output_name,
            model_ir,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    pre_consumer_indices = graph_index.consumer_indices(pre_output_name)
    unique_pre_consumers = tuple(dict.fromkeys(pre_consumer_indices))
    if len(unique_pre_consumers) < 2:
        return None

    slice_updates = []
    constant_updates: dict[str, _ConstantUpdate] = {}
    slice_output_names = []
    slice_operators = []
    for slice_index in unique_pre_consumers:
        if int(slice_index) <= int(pre_index):
            return None
        operator = model_ir.operators[int(slice_index)]
        if (
            not _supported_stridedslice(operator)
            or str(operator.inputs[0]) != pre_output_name
            or pre_consumer_indices.count(int(slice_index)) != 1
            or len(set(str(value) for value in operator.inputs[1:])) != 3
        ):
            return None
        output_name = str(operator.outputs[0])
        output_tensor = model_ir.tensors.get(output_name)
        if (
            output_tensor is None
            or output_name in public_inputs
            or output_name in public_outputs
            or output_name in graph_index.duplicate_producers
            or graph_index.producer(output_name) is not operator
            or not _rank4(_view(output_tensor))
            or output_tensor.dtype != pre_output_tensor.dtype
            or not _same_quantization(
                output_tensor.quantization,
                pre_output_tensor.quantization,
            )
            or not _layout_in(
                output_name,
                model_ir,
                layout_state,
                {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            return None
        new_view = _permuted_view(_view(output_tensor), _PERM_NCHW_TO_NHWC)
        if new_view is None:
            return None
        resolved_slice_constants = []
        for constant_name in operator.inputs[1:]:
            update = _constant_update(
                model_ir,
                graph_index,
                name=str(constant_name),
                slice_operator=operator,
                slice_index=int(slice_index),
            )
            if update is None:
                return None
            previous = constant_updates.get(update.name)
            if previous is not None and previous != update:
                return None
            constant_updates[update.name] = update
            resolved_slice_constants.append(update)
        if any(
            int(value) == 0
            for value in resolved_slice_constants[2].original_values
        ):
            return None
        slice_updates.append(
            _SliceUpdate(
                operator=operator,
                output_name=output_name,
                original_inputs=tuple(str(value) for value in operator.inputs),
                new_inputs=(
                    source_name,
                    *(str(value) for value in operator.inputs[1:]),
                ),
                old_view=_view(output_tensor),
                new_view=new_view,
            )
        )
        slice_output_names.append(output_name)
        slice_operators.append(operator)

    if len(set(slice_output_names)) != len(slice_output_names):
        return None
    concat_indices = []
    for output_name in slice_output_names:
        consumers = graph_index.consumer_indices(output_name)
        if len(consumers) != 1:
            return None
        concat_indices.append(int(consumers[0]))
    if len(set(concat_indices)) != 1:
        return None
    concat_index = concat_indices[0]
    if concat_index <= max(int(value) for value in unique_pre_consumers):
        return None
    concat = model_ir.operators[int(concat_index)]
    concat_inputs = tuple(str(value) for value in concat.inputs)
    if (
        _op_type(concat) != "CONCATENATION"
        or len(concat.outputs) != 1
        or _normalized_axis(concat) != 1
        or len(concat_inputs) != len(slice_output_names)
        or len(set(concat_inputs)) != len(concat_inputs)
        or set(concat_inputs) != set(slice_output_names)
    ):
        return None

    concat_output_name = str(concat.outputs[0])
    concat_output_tensor = model_ir.tensors.get(concat_output_name)
    ordered_slice_views = [
        _view(model_ir.tensors[name]) for name in concat_inputs
    ]
    expected_concat_view = _concat_view(ordered_slice_views, axis=1)
    if (
        concat_output_tensor is None
        or concat_output_name in public_inputs
        or concat_output_name in graph_index.duplicate_producers
        or graph_index.producer(concat_output_name) is not concat
        or expected_concat_view is None
        or _view(concat_output_tensor) != expected_concat_view
        or not _per_tensor_quantization(concat_output_tensor.quantization)
        or any(
            not _same_quantization(
                model_ir.tensors[name].quantization,
                concat_output_tensor.quantization,
            )
            for name in concat_inputs
        )
        or not _layout_in(
            concat_output_name,
            model_ir,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    concat_new_view = _permuted_view(
        _view(concat_output_tensor),
        _PERM_NCHW_TO_NHWC,
    )
    if concat_new_view is None:
        return None

    posts = []
    legacy_operators = []
    concat_consumers = tuple(
        dict.fromkeys(graph_index.consumer_indices(concat_output_name))
    )
    if len(concat_consumers) == 0:
        return None
    for consumer_index in concat_consumers:
        if int(consumer_index) <= int(concat_index):
            return None
        operator = model_ir.operators[int(consumer_index)]
        output_name = (
            str(operator.outputs[0])
            if len(operator.outputs) == 1
            else ""
        )
        if (
            _typed_permutation(
                model_ir,
                graph_index,
                operator,
                _PERM_NCHW_TO_NHWC,
            )
            and str(operator.inputs[0]) == concat_output_name
            and output_name not in public_outputs
        ):
            output_tensor = model_ir.tensors.get(output_name)
            if (
                output_tensor is None
                or output_name in public_inputs
                or output_name in graph_index.duplicate_producers
                or graph_index.producer(output_name) is not operator
                or _view(output_tensor) != concat_new_view
                or not _same_quantization(
                    output_tensor.quantization,
                    concat_output_tensor.quantization,
                )
                or not _layout_in(
                    output_name,
                    model_ir,
                    layout_state,
                    {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
                )
            ):
                return None
            posts.append(operator)
        else:
            legacy_operators.append(operator)
    if len(posts) == 0:
        return None

    canonical_output_name = str(posts[0].outputs[0])
    keep_post = (
        posts[0]
        if legacy_operators or concat_output_name in public_outputs
        else None
    )
    keep_post_index = (
        None if keep_post is None else _operator_index(graph_index, keep_post)
    )
    if keep_post_index is not None and any(
        int(keep_post_index) >= int(_operator_index(graph_index, operator) or -1)
        for operator in legacy_operators
    ):
        return None

    alias_rewrites = []
    alias_names = []
    alias_consumers = []
    for post in posts[1:]:
        post_index = _operator_index(graph_index, post)
        if post_index is None:
            return None
        alias_name = str(post.outputs[0])
        alias_names.append(alias_name)
        for operator, _ in _consumer_slots(model_ir, graph_index, alias_name):
            consumer_index = _operator_index(graph_index, operator)
            if consumer_index is None or int(consumer_index) <= int(post_index):
                return None
            original_inputs = tuple(str(value) for value in operator.inputs)
            new_inputs = tuple(
                canonical_output_name if value == alias_name else value
                for value in original_inputs
            )
            alias_rewrites.append(
                _InputRewrite(
                    operator=operator,
                    original_inputs=original_inputs,
                    new_inputs=new_inputs,
                )
            )
            alias_consumers.append(operator)

    removals = [seed]
    removals.extend(
        post for post in posts if keep_post is None or post is not keep_post
    )

    contract_operators = [seed, *slice_operators, concat, *posts]
    _append_unique_operators(contract_operators, alias_consumers)
    _append_unique_operators(contract_operators, legacy_operators)
    contract_names = {
        source_name,
        pre_output_name,
        pre_permutation_name,
        concat_output_name,
        canonical_output_name,
    }
    for operator in contract_operators:
        contract_names.update(str(value) for value in operator.inputs)
        contract_names.update(str(value) for value in operator.outputs)
    if any(name not in model_ir.tensors for name in contract_names):
        return None

    return _Plan(
        seed=seed,
        source_name=source_name,
        pre_output_name=pre_output_name,
        pre_permutation_name=pre_permutation_name,
        slices=tuple(slice_updates),
        concat=concat,
        posts=tuple(posts),
        keep_post=keep_post,
        concat_output_name=concat_output_name,
        canonical_output_name=canonical_output_name,
        concat_old_view=_view(concat_output_tensor),
        concat_new_view=concat_new_view,
        constants=tuple(
            constant_updates[name] for name in sorted(constant_updates)
        ),
        alias_names=tuple(alias_names),
        alias_rewrites=tuple(alias_rewrites),
        removals=tuple(removals),
        tensor_contracts=tuple(
            _tensor_contract(name, model_ir.tensors[name])
            for name in sorted(contract_names)
        ),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in contract_operators
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
    tensor.shape = [int(value) for value in update.shape]
    tensor.shape_signature = [int(value) for value in update.signature]
    tensor.logical_layout = layout
    tensor.physical_layout = layout
    if layout_state is not None:
        layout_state.set(update.name, logical=layout, physical=layout)


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
        plan.seed,
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
            for rewrite in plan.alias_rewrites
        )
    ):
        return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))

    for update in plan.constants:
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
    for update in plan.constants:
        tensor = model_ir.tensors[update.name]
        data = np.asarray(tensor.data)
        tensor.data = np.asarray(
            update.target_values,
            dtype=data.dtype,
        ).reshape(data.shape)

    for update in plan.slices:
        _replace_operator_input_at(
            model_ir=model_ir,
            op=update.operator,
            input_index=0,
            new_input_name=update.new_inputs[0],
            graph_index=graph_index,
        )
        _set_metadata(
            model_ir,
            _MetadataUpdate(
                name=update.output_name,
                shape=update.new_view.shape,
                signature=update.new_view.signature,
            ),
            layout_state=layout_state,
            layout=LOGICAL_LAYOUT_NHWC,
        )

    options = dict(plan.concat.options)
    options["axis"] = 3
    plan.concat.options = options
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.concat,
        new_outputs=[plan.canonical_output_name],
        graph_index=graph_index,
    )
    old_concat_tensor = model_ir.tensors[plan.concat_output_name]
    canonical_tensor = model_ir.tensors[plan.canonical_output_name]
    canonical_tensor.dtype = str(old_concat_tensor.dtype)
    canonical_tensor.quantization = _clone_quantization(
        old_concat_tensor.quantization
    )
    _set_metadata(
        model_ir,
        _MetadataUpdate(
            name=plan.canonical_output_name,
            shape=plan.concat_new_view.shape,
            signature=plan.concat_new_view.signature,
        ),
        layout_state=layout_state,
        layout=LOGICAL_LAYOUT_NHWC,
    )

    for alias_name in plan.alias_names:
        _replace_tensor_inputs(
            model_ir,
            alias_name,
            plan.canonical_output_name,
            graph_index=graph_index,
        )

    if plan.keep_post is not None:
        _set_operator_inputs(
            model_ir=model_ir,
            op=plan.keep_post,
            new_inputs=[
                plan.canonical_output_name,
                plan.pre_permutation_name,
            ],
            graph_index=graph_index,
        )
        _set_operator_outputs(
            model_ir=model_ir,
            op=plan.keep_post,
            new_outputs=[plan.concat_output_name],
            graph_index=graph_index,
        )
        old_concat_tensor.shape = [
            int(value) for value in plan.concat_old_view.shape
        ]
        old_concat_tensor.shape_signature = [
            int(value) for value in plan.concat_old_view.signature
        ]
        old_concat_tensor.logical_layout = LOGICAL_LAYOUT_NCHW
        old_concat_tensor.physical_layout = LOGICAL_LAYOUT_NCHW
        if layout_state is not None:
            layout_state.set(
                plan.concat_output_name,
                logical=LOGICAL_LAYOUT_NCHW,
                physical=LOGICAL_LAYOUT_NCHW,
            )

    graph_index.remove_operators(removal_indices)
    return True


def optimize_transpose_stridedslice_pre_concat_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Lift a fully classified StridedSlice/Concat fan-in group to NHWC."""

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
    for seed in candidates:
        if rewritten >= rewrite_limit:
            break
        if seed is None or _operator_index(active_index, seed) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            seed,
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
