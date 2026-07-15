from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_shape_signatures,
    _broadcast_static_shapes,
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
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    _runtime_tensor,
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
    _typed_permutation,
    _unique_name,
    _view,
)


_STATS_KEY = "optimized_transpose_elementwise_concat_conv_nhwc_groups"
_PERMUTATION_NAME = "__nhwc_to_nchw_perm_rank4__"
_UNARY_TYPES = {
    "RELU",
    "RELU6",
    "LEAKY_RELU",
    "LOGISTIC",
    "TANH",
    "NEG",
    "EXP",
    "ABS",
    "SQRT",
    "GELU",
    "ELU",
}
_BINARY_TYPES = {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM"}
_LEGACY_TYPES = _UNARY_TYPES | _BINARY_TYPES


@dataclass(frozen=True)
class _Boundary:
    tensor_name: str
    replacement_name: str
    kind: str
    adapter: Optional[OperatorIR]
    old_view: _View
    new_view: _View


@dataclass(frozen=True)
class _ConstantUpdate:
    name: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    numpy_dtype: str
    data_contract: Any


@dataclass(frozen=True)
class _ConcatUpdate:
    operator: OperatorIR
    posts: Tuple[OperatorIR, ...]
    original_axis: int
    original_output: str
    canonical_output: str
    old_view: _View
    new_view: _View


@dataclass(frozen=True)
class _LegacyAdapter:
    source_name: str
    output_name: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    dtype: str
    quantization_contract: Any
    logical_layout: str
    physical_layout: str
    consumer_operators: Tuple[OperatorIR, ...]


@dataclass(frozen=True)
class _Plan:
    seed: OperatorIR
    concats: Tuple[OperatorIR, ...]
    closure: Tuple[OperatorIR, ...]
    boundaries: Tuple[_Boundary, ...]
    constant_updates: Tuple[_ConstantUpdate, ...]
    concat_updates: Tuple[_ConcatUpdate, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    main_rewrites: Tuple[_InputRewrite, ...]
    alias_rewrites: Tuple[_InputRewrite, ...]
    legacy_rewrites: Tuple[_InputRewrite, ...]
    legacy_adapters: Tuple[_LegacyAdapter, ...]
    create_permutation: bool
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.seed),
        tuple(id(operator) for operator in plan.concats),
        tuple(id(operator) for operator in plan.closure),
        tuple(
            (
                boundary.tensor_name,
                boundary.replacement_name,
                boundary.kind,
                None if boundary.adapter is None else id(boundary.adapter),
                boundary.old_view,
                boundary.new_view,
            )
            for boundary in plan.boundaries
        ),
        tuple(
            (
                update.name,
                update.shape,
                update.signature,
                update.numpy_dtype,
                update.data_contract,
            )
            for update in plan.constant_updates
        ),
        tuple(
            (
                id(update.operator),
                tuple(id(post) for post in update.posts),
                update.original_axis,
                update.original_output,
                update.canonical_output,
                update.old_view,
                update.new_view,
            )
            for update in plan.concat_updates
        ),
        tuple(
            (update.name, update.shape, update.signature)
            for update in plan.metadata_updates
        ),
        tuple(
            (
                id(rewrite.operator),
                rewrite.original_inputs,
                rewrite.new_inputs,
            )
            for rewrites in (
                plan.main_rewrites,
                plan.alias_rewrites,
                plan.legacy_rewrites,
            )
            for rewrite in rewrites
        ),
        tuple(
            (
                adapter.source_name,
                adapter.output_name,
                adapter.shape,
                adapter.signature,
                adapter.dtype,
                adapter.quantization_contract,
                adapter.logical_layout,
                adapter.physical_layout,
                tuple(id(operator) for operator in adapter.consumer_operators),
            )
            for adapter in plan.legacy_adapters
        ),
        plan.create_permutation,
        tuple(id(operator) for operator in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _rank4_runtime(tensor: Optional[TensorIR], *, allow_constant: bool = False) -> bool:
    return bool(
        _runtime_tensor(tensor, rank=4, allow_constant=allow_constant)
        and tensor is not None
        and len(tensor.shape_signature or tensor.shape) == 4
    )


def _layout_in(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
    allowed: set[str],
) -> bool:
    return str(_layout_of(name, tensor, layout_state)).upper() in allowed


def _broadcast_views(left: _View, right: _View) -> Optional[_View]:
    if str(left.dtype) != str(right.dtype):
        return None
    left_shape = left.shape if left.shape else (1,)
    right_shape = right.shape if right.shape else (1,)
    left_signature = left.signature if left.signature else (1,)
    right_signature = right.signature if right.signature else (1,)
    shape = _broadcast_static_shapes(list(left_shape), list(right_shape))
    signature = _broadcast_shape_signatures(
        list(left_signature),
        list(right_signature),
    )
    if shape is None or signature is None:
        return None
    return _View(
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        dtype=str(left.dtype),
    )


def _concat_view(views: Sequence[_View], *, axis: int) -> Optional[_View]:
    if len(views) == 0 or int(axis) not in range(4):
        return None
    dtype = str(views[0].dtype)
    if any(
        len(view.shape) != 4
        or len(view.signature) != 4
        or str(view.dtype) != dtype
        for view in views
    ):
        return None
    shape = []
    signature = []
    for dimension in range(4):
        shape_values = [int(view.shape[dimension]) for view in views]
        signature_values = [int(view.signature[dimension]) for view in views]
        if dimension == int(axis):
            shape.append(sum(shape_values))
            signature.append(
                sum(signature_values)
                if all(value >= 0 for value in signature_values)
                else -1
            )
            continue
        if len(set(shape_values)) != 1:
            return None
        known = {value for value in signature_values if value >= 0}
        if len(known) > 1:
            return None
        shape.append(shape_values[0])
        signature.append(next(iter(known)) if known else -1)
    return _View(
        shape=tuple(shape),
        signature=tuple(signature),
        dtype=dtype,
    )


def _normalized_concat_axis(operator: OperatorIR) -> Optional[Tuple[int, int]]:
    if _op_type(operator) != "CONCATENATION" or not isinstance(
        operator.options,
        dict,
    ):
        return None
    try:
        original = int(operator.options.get("axis", 1))
    except Exception:
        return None
    normalized = int(original)
    if normalized < 0:
        normalized += 4
    return original, normalized


def _post_adapters(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    concat: OperatorIR,
) -> Optional[Tuple[OperatorIR, ...]]:
    concat_index = _operator_index(graph_index, concat)
    axis = _normalized_concat_axis(concat)
    if (
        concat_index is None
        or axis is None
        or int(axis[1]) != 1
        or len(concat.inputs) == 0
        or len(concat.outputs) != 1
    ):
        return None
    graph_outputs = {str(value) for value in model_ir.outputs}
    output_name = str(concat.outputs[0])
    if output_name in graph_outputs or output_name in graph_index.duplicate_producers:
        return None
    slots = _consumer_slots(model_ir, graph_index, output_name)
    if len(slots) == 0:
        return None
    posts = []
    for post, input_slot in slots:
        post_index = _operator_index(graph_index, post)
        if (
            int(input_slot) != 0
            or post_index is None
            or int(post_index) <= int(concat_index)
            or not _typed_permutation(
                model_ir,
                graph_index,
                post,
                _PERM_NCHW_TO_NHWC,
            )
            or str(post.inputs[0]) != output_name
            or str(post.outputs[0]) in graph_outputs
        ):
            return None
        posts.append(post)
    unique = tuple(
        sorted(
            {id(post): post for post in posts}.values(),
            key=lambda post: int(_operator_index(graph_index, post) or 0),
        )
    )
    return unique if len(unique) == len(slots) else None


def _candidate_concats(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Tuple[OperatorIR, ...]:
    candidates = []
    for index in graph_index.operator_indices_for_normalized_types(
        {"CONCATENATION"}
    ):
        operator = model_ir.operators[int(index)]
        if _post_adapters(model_ir, graph_index, operator) is not None:
            candidates.append(operator)
    return tuple(candidates)


def _boundary_before(
    graph_index: ModelIRGraphIndex,
    boundary: _Boundary,
    before_index: int,
) -> bool:
    if boundary.adapter is None:
        return True
    adapter_index = _operator_index(graph_index, boundary.adapter)
    return bool(adapter_index is not None and int(adapter_index) < int(before_index))


def _resolve_boundary(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    before_index: int,
    layout_state: Optional[LayoutState],
) -> Optional[_Boundary]:
    tensor_name = str(name)
    tensor = model_ir.tensors.get(tensor_name)
    if tensor is None or tensor_name in graph_index.duplicate_producers:
        return None
    graph_inputs = {str(value) for value in model_ir.inputs}
    producer_index = graph_index.producers.get(tensor_name)
    if producer_index is None:
        if (
            tensor_name not in graph_inputs
            or not _rank4_runtime(tensor)
            or not _layout_in(
                tensor_name,
                tensor,
                layout_state,
                {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            return None
        new_view = _view(tensor)
        old_view = _permuted_view(new_view, _PERM_NHWC_TO_NCHW)
        if old_view is None:
            return None
        return _Boundary(
            tensor_name=tensor_name,
            replacement_name=tensor_name,
            kind="input",
            adapter=None,
            old_view=old_view,
            new_view=new_view,
        )

    producer = model_ir.operators[int(producer_index)]
    if (
        int(producer_index) < int(before_index)
        and _typed_permutation(
            model_ir,
            graph_index,
            producer,
            _PERM_NHWC_TO_NCHW,
        )
        and str(producer.outputs[0]) == tensor_name
    ):
        source_name = str(producer.inputs[0])
        source_tensor = model_ir.tensors.get(source_name)
        if (
            source_tensor is None
            or not _rank4_runtime(source_tensor, allow_constant=True)
            or not _rank4_runtime(tensor)
            or not _resolved_source(
                model_ir,
                graph_index,
                name=source_name,
                before_index=int(producer_index),
            )
            or _permuted_view(_view(source_tensor), _PERM_NHWC_TO_NCHW)
            != _view(tensor)
            or _freeze(source_tensor.quantization) != _freeze(tensor.quantization)
            or not _layout_in(
                source_name,
                source_tensor,
                layout_state,
                {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
            )
            or not _layout_in(
                tensor_name,
                tensor,
                layout_state,
                {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            return None
        return _Boundary(
            tensor_name=tensor_name,
            replacement_name=source_name,
            kind="pre",
            adapter=producer,
            old_view=_view(tensor),
            new_view=_view(source_tensor),
        )

    for consumer_index in graph_index.consumer_indices(tensor_name):
        if int(consumer_index) >= int(before_index):
            continue
        consumer = model_ir.operators[int(consumer_index)]
        if not _typed_permutation(
            model_ir,
            graph_index,
            consumer,
            _PERM_NCHW_TO_NHWC,
        ) or str(consumer.inputs[0]) != tensor_name:
            continue
        replacement_name = str(consumer.outputs[0])
        replacement_tensor = model_ir.tensors.get(replacement_name)
        if (
            replacement_name in {str(value) for value in model_ir.outputs}
            or replacement_name in graph_index.duplicate_producers
            or graph_index.producers.get(replacement_name) != int(consumer_index)
            or replacement_tensor is None
            or not _rank4_runtime(tensor)
            or not _rank4_runtime(replacement_tensor)
            or _permuted_view(_view(tensor), _PERM_NCHW_TO_NHWC)
            != _view(replacement_tensor)
            or _freeze(tensor.quantization)
            != _freeze(replacement_tensor.quantization)
            or not _layout_in(
                tensor_name,
                tensor,
                layout_state,
                {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
            )
            or not _layout_in(
                replacement_name,
                replacement_tensor,
                layout_state,
                {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            continue
        return _Boundary(
            tensor_name=tensor_name,
            replacement_name=replacement_name,
            kind="post",
            adapter=consumer,
            old_view=_view(tensor),
            new_view=_view(replacement_tensor),
        )
    return None


def _resolve_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[Tuple[_View, _View, Optional[_ConstantUpdate]]]:
    tensor = model_ir.tensors.get(str(name))
    public = {str(value) for value in (*model_ir.inputs, *model_ir.outputs)}
    if (
        tensor is None
        or tensor.data is None
        or not isinstance(tensor.data, np.ndarray)
        or bool(tensor.is_variable)
        or str(name) in public
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return None
    data = np.asarray(tensor.data)
    signature = tuple(
        int(value)
        for value in (tensor.shape_signature or tensor.shape)
    )
    if (
        str(data.dtype).upper() != str(tensor.dtype).upper()
        or tuple(int(value) for value in data.shape)
        != tuple(int(value) for value in tensor.shape)
        or signature != tuple(int(value) for value in tensor.shape)
    ):
        return None
    old_view = _view(tensor)
    if int(data.ndim) != 4 or int(data.size) <= 1:
        return old_view, old_view, None
    if not _layout_in(
        str(name),
        tensor,
        layout_state,
        {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
    ):
        return None
    transposed = np.transpose(data, axes=_PERM_NCHW_TO_NHWC)
    new_view = _View(
        shape=tuple(int(value) for value in transposed.shape),
        signature=tuple(int(value) for value in transposed.shape),
        dtype=str(tensor.dtype),
    )
    return (
        old_view,
        new_view,
        _ConstantUpdate(
            name=str(name),
            shape=new_view.shape,
            signature=new_view.signature,
            numpy_dtype=str(transposed.dtype),
            data_contract=_freeze(transposed),
        ),
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    seed: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    seed_index = _operator_index(graph_index, seed)
    seed_posts = _post_adapters(model_ir, graph_index, seed)
    if seed_index is None or seed_posts is None:
        return None
    graph_inputs = tuple(str(value) for value in model_ir.inputs)
    graph_outputs = tuple(str(value) for value in model_ir.outputs)
    public = set(graph_inputs) | set(graph_outputs)
    candidates = _candidate_concats(model_ir, graph_index)
    if seed not in candidates:
        return None
    target: Dict[int, OperatorIR] = {id(seed): seed}
    edge_limit = max(
        1,
        sum(
            len(operator.inputs) + len(operator.outputs)
            for operator in model_ir.operators
        ),
    )
    closure: Dict[int, OperatorIR] = {}
    boundaries: Dict[str, _Boundary] = {}
    constant_names: set[str] = set()

    for _ in range(len(candidates) + 1):
        closure = {}
        boundaries = {}
        constant_names = set()
        visiting: set[int] = set()
        traversed = 0

        def trace(name: str, *, before_index: int) -> bool:
            nonlocal traversed
            traversed += 1
            if traversed > edge_limit:
                return False
            normalized_name = str(name)
            cached = boundaries.get(normalized_name)
            if cached is not None:
                return _boundary_before(
                    graph_index,
                    cached,
                    before_index,
                )
            tensor = model_ir.tensors.get(normalized_name)
            if tensor is not None and tensor.data is not None:
                constant_names.add(normalized_name)
                return True
            boundary = _resolve_boundary(
                model_ir,
                graph_index,
                normalized_name,
                before_index=before_index,
                layout_state=layout_state,
            )
            if boundary is not None:
                boundaries[normalized_name] = boundary
                return True
            producer_index = graph_index.producers.get(normalized_name)
            if producer_index is None or int(producer_index) >= int(before_index):
                return False
            producer = model_ir.operators[int(producer_index)]
            producer_id = id(producer)
            if producer_id in closure:
                return True
            if producer_id in visiting:
                return False
            if normalized_name in public:
                return False
            if (
                _op_type(producer) not in _UNARY_TYPES | _BINARY_TYPES
                or len(producer.outputs) != 1
                or str(producer.outputs[0]) != normalized_name
                or len(producer.inputs)
                != (1 if _op_type(producer) in _UNARY_TYPES else 2)
            ):
                return False
            visiting.add(producer_id)
            closure[producer_id] = producer
            valid = all(
                trace(str(input_name), before_index=int(producer_index))
                for input_name in producer.inputs
            )
            visiting.remove(producer_id)
            return valid

        valid_trace = True
        for concat in sorted(
            target.values(),
            key=lambda operator: int(_operator_index(graph_index, operator) or 0),
        ):
            concat_index = _operator_index(graph_index, concat)
            if concat_index is None or not all(
                trace(str(input_name), before_index=int(concat_index))
                for input_name in concat.inputs
            ):
                valid_trace = False
                break
        if not valid_trace:
            return None
        expanded = False
        closure_ids = set(closure)
        for candidate_concat in candidates:
            if id(candidate_concat) in target:
                continue
            if any(
                (
                    graph_index.producer(str(input_name)) is not None
                    and id(graph_index.producer(str(input_name))) in closure_ids
                )
                for input_name in candidate_concat.inputs
            ):
                target[id(candidate_concat)] = candidate_concat
                expanded = True
                break
        if not expanded:
            break
    else:
        return None

    concats = tuple(
        sorted(
            target.values(),
            key=lambda operator: int(_operator_index(graph_index, operator) or 0),
        )
    )
    closure_ops = tuple(
        sorted(
            closure.values(),
            key=lambda operator: int(_operator_index(graph_index, operator) or 0),
        )
    )
    concat_posts: Dict[int, Tuple[OperatorIR, ...]] = {}
    for concat in concats:
        posts = _post_adapters(model_ir, graph_index, concat)
        if posts is None:
            return None
        concat_posts[id(concat)] = posts
    allowed_ids = {
        *(id(operator) for operator in closure_ops),
        *(id(operator) for operator in concats),
        *(
            id(post)
            for posts in concat_posts.values()
            for post in posts
        ),
    }

    legacy_slots: Dict[str, Tuple[Tuple[OperatorIR, int], ...]] = {}
    for operator in closure_ops:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return None
        for output_name in (str(value) for value in operator.outputs):
            if output_name in public:
                return None
            external = []
            for consumer, input_slot in _consumer_slots(
                model_ir,
                graph_index,
                output_name,
            ):
                consumer_index = _operator_index(graph_index, consumer)
                if consumer_index is None or int(consumer_index) <= int(operator_index):
                    return None
                if id(consumer) in allowed_ids:
                    continue
                if _op_type(consumer) not in _LEGACY_TYPES:
                    return None
                external.append((consumer, int(input_slot)))
            if external:
                legacy_slots[output_name] = tuple(external)

    constant_results: Dict[str, Tuple[_View, _View]] = {}
    constant_updates = []
    closure_ids = {id(operator) for operator in closure_ops}
    for name in sorted(constant_names):
        resolved = _resolve_constant(
            model_ir,
            graph_index,
            name,
            layout_state=layout_state,
        )
        if resolved is None:
            return None
        if any(
            id(consumer) not in closure_ids
            for consumer, _ in _consumer_slots(model_ir, graph_index, name)
        ):
            return None
        constant_results[name] = (resolved[0], resolved[1])
        if resolved[2] is not None:
            constant_updates.append(resolved[2])

    old_views: Dict[str, _View] = {
        name: boundary.old_view for name, boundary in boundaries.items()
    }
    new_views: Dict[str, _View] = {
        name: boundary.new_view for name, boundary in boundaries.items()
    }
    for name, (old_view, new_view) in constant_results.items():
        old_views[name] = old_view
        new_views[name] = new_view
    metadata_updates = []
    for operator in closure_ops:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return None
        input_old = [old_views.get(str(name)) for name in operator.inputs]
        input_new = [new_views.get(str(name)) for name in operator.inputs]
        if any(view is None for view in (*input_old, *input_new)):
            return None
        old_input_views = [view for view in input_old if view is not None]
        new_input_views = [view for view in input_new if view is not None]
        if _op_type(operator) in _UNARY_TYPES:
            expected_old = old_input_views[0]
            expected_new = new_input_views[0]
        else:
            expected_old = _broadcast_views(old_input_views[0], old_input_views[1])
            expected_new = _broadcast_views(new_input_views[0], new_input_views[1])
            if expected_old is None or expected_new is None:
                return None
        output_name = str(operator.outputs[0])
        output_tensor = model_ir.tensors.get(output_name)
        if (
            expected_old is None
            or expected_new is None
            or output_name in graph_index.duplicate_producers
            or graph_index.producers.get(output_name) != int(operator_index)
            or not _rank4_runtime(output_tensor)
            or output_tensor is None
            or _view(output_tensor) != expected_old
            or _permuted_view(expected_old, _PERM_NCHW_TO_NHWC)
            != expected_new
            or not _layout_in(
                output_name,
                output_tensor,
                layout_state,
                {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            return None
        output_quantization = _freeze(output_tensor.quantization)
        for input_name in operator.inputs:
            if str(input_name) in constant_results:
                continue
            input_tensor = model_ir.tensors.get(str(input_name))
            if (
                input_tensor is None
                or _freeze(input_tensor.quantization) != output_quantization
            ):
                return None
        old_views[output_name] = expected_old
        new_views[output_name] = expected_new
        metadata_updates.append(
            _MetadataUpdate(
                name=output_name,
                shape=expected_new.shape,
                signature=expected_new.signature,
            )
        )

    planned_main: Dict[int, list[str]] = {}
    main_operators: Dict[int, OperatorIR] = {}
    for operator in (*closure_ops, *concats):
        inputs = [str(value) for value in operator.inputs]
        updated = [
            boundaries[name].replacement_name if name in boundaries else name
            for name in inputs
        ]
        if updated != inputs:
            planned_main[id(operator)] = updated
            main_operators[id(operator)] = operator

    concat_updates = []
    planned_alias: Dict[int, list[str]] = {}
    alias_operators: Dict[int, OperatorIR] = {}
    for concat in concats:
        concat_index = _operator_index(graph_index, concat)
        axis = _normalized_concat_axis(concat)
        if concat_index is None or axis is None:
            return None
        input_old = [old_views.get(str(name)) for name in concat.inputs]
        input_new = [new_views.get(str(name)) for name in concat.inputs]
        if any(view is None for view in (*input_old, *input_new)):
            return None
        old_view = _concat_view(
            [view for view in input_old if view is not None],
            axis=1,
        )
        new_view = _concat_view(
            [view for view in input_new if view is not None],
            axis=3,
        )
        output_name = str(concat.outputs[0])
        output_tensor = model_ir.tensors.get(output_name)
        if (
            old_view is None
            or new_view is None
            or output_tensor is None
            or not _rank4_runtime(output_tensor)
            or output_name in public
            or output_name in graph_index.duplicate_producers
            or graph_index.producers.get(output_name) != int(concat_index)
            or _view(output_tensor) != old_view
            or _permuted_view(old_view, _PERM_NCHW_TO_NHWC) != new_view
            or not _layout_in(
                output_name,
                output_tensor,
                layout_state,
                {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            return None
        output_quantization = _freeze(output_tensor.quantization)
        if any(
            model_ir.tensors.get(str(input_name)) is None
            or _freeze(model_ir.tensors[str(input_name)].quantization)
            != output_quantization
            for input_name in concat.inputs
        ):
            return None
        posts = concat_posts[id(concat)]
        post_slots = _consumer_slots(model_ir, graph_index, output_name)
        if len(post_slots) != len(posts) or any(
            post_slots[index][0] is not post or int(post_slots[index][1]) != 0
            for index, post in enumerate(posts)
        ):
            return None
        canonical_output = str(posts[0].outputs[0])
        for post in posts:
            post_index = _operator_index(graph_index, post)
            post_output = str(post.outputs[0])
            post_tensor = model_ir.tensors.get(post_output)
            if (
                post_index is None
                or post_tensor is None
                or not _rank4_runtime(post_tensor)
                or post_output in public
                or post_output in graph_index.duplicate_producers
                or graph_index.producers.get(post_output) != int(post_index)
                or _view(post_tensor) != new_view
                or _freeze(post_tensor.quantization) != output_quantization
                or not _layout_in(
                    post_output,
                    post_tensor,
                    layout_state,
                    {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
                )
            ):
                return None
            for consumer, _ in _consumer_slots(
                model_ir,
                graph_index,
                post_output,
            ):
                consumer_index = _operator_index(graph_index, consumer)
                if consumer_index is None or int(consumer_index) <= int(post_index):
                    return None
        for alias_post in posts[1:]:
            alias_name = str(alias_post.outputs[0])
            for consumer, input_slot in _consumer_slots(
                model_ir,
                graph_index,
                alias_name,
            ):
                alias_operators[id(consumer)] = consumer
                inputs = planned_alias.setdefault(
                    id(consumer),
                    [str(value) for value in consumer.inputs],
                )
                inputs[int(input_slot)] = canonical_output
        concat_updates.append(
            _ConcatUpdate(
                operator=concat,
                posts=posts,
                original_axis=int(axis[0]),
                original_output=output_name,
                canonical_output=canonical_output,
                old_view=old_view,
                new_view=new_view,
            )
        )
        metadata_updates.extend(
            (
                _MetadataUpdate(
                    name=output_name,
                    shape=new_view.shape,
                    signature=new_view.signature,
                ),
                _MetadataUpdate(
                    name=canonical_output,
                    shape=new_view.shape,
                    signature=new_view.signature,
                ),
            )
        )

    occupied = {str(name) for name in model_ir.tensors}
    legacy_adapters = []
    planned_legacy: Dict[int, list[str]] = {}
    legacy_operators: Dict[int, OperatorIR] = {}
    for source_name in sorted(
        legacy_slots,
        key=lambda name: int(
            graph_index.producers.get(str(name), len(model_ir.operators))
        ),
    ):
        source_tensor = model_ir.tensors.get(source_name)
        if source_tensor is None:
            return None
        adapter_name = _unique_name(f"{source_name}_nchw_adapter", occupied)
        consumers = []
        for consumer, input_slot in legacy_slots[source_name]:
            consumers.append(consumer)
            legacy_operators[id(consumer)] = consumer
            inputs = planned_legacy.setdefault(
                id(consumer),
                [str(value) for value in consumer.inputs],
            )
            inputs[int(input_slot)] = adapter_name
        old_view = old_views.get(source_name)
        if old_view is None:
            return None
        legacy_adapters.append(
            _LegacyAdapter(
                source_name=source_name,
                output_name=adapter_name,
                shape=old_view.shape,
                signature=old_view.signature,
                dtype=str(source_tensor.dtype),
                quantization_contract=_freeze(source_tensor.quantization),
                logical_layout=str(source_tensor.logical_layout),
                physical_layout=str(source_tensor.physical_layout),
                consumer_operators=tuple(
                    sorted(
                        {id(operator): operator for operator in consumers}.values(),
                        key=lambda operator: int(
                            _operator_index(graph_index, operator) or 0
                        ),
                    )
                ),
            )
        )

    create_permutation = False
    if legacy_adapters:
        permutation_tensor = model_ir.tensors.get(_PERMUTATION_NAME)
        if permutation_tensor is None:
            create_permutation = True
        else:
            synthetic = OperatorIR(
                op_type="TRANSPOSE",
                inputs=[legacy_adapters[0].source_name, _PERMUTATION_NAME],
                outputs=[legacy_adapters[0].output_name],
            )
            if not _typed_permutation(
                model_ir,
                graph_index,
                synthetic,
                _PERM_NHWC_TO_NCHW,
            ):
                return None

    def rewrites(
        operators: Dict[int, OperatorIR],
        planned: Dict[int, list[str]],
    ) -> Tuple[_InputRewrite, ...]:
        return tuple(
            _InputRewrite(
                operator=operator,
                original_inputs=tuple(str(value) for value in operator.inputs),
                new_inputs=tuple(planned[id(operator)]),
            )
            for operator in sorted(
                operators.values(),
                key=lambda candidate: int(
                    _operator_index(graph_index, candidate) or 0
                ),
            )
            if tuple(str(value) for value in operator.inputs)
            != tuple(planned[id(operator)])
        )

    main_rewrites = rewrites(main_operators, planned_main)
    alias_rewrites = rewrites(alias_operators, planned_alias)
    legacy_rewrites = rewrites(legacy_operators, planned_legacy)
    all_planned = {
        id(rewrite.operator): rewrite.new_inputs
        for rewrite in (*main_rewrites, *alias_rewrites, *legacy_rewrites)
    }
    removals: Dict[int, OperatorIR] = {
        id(post): post
        for update in concat_updates
        for post in update.posts
    }
    for boundary in boundaries.values():
        if boundary.kind != "pre" or boundary.adapter is None:
            continue
        output_name = boundary.tensor_name
        if output_name in public:
            continue
        remaining = False
        for consumer, input_slot in _consumer_slots(
            model_ir,
            graph_index,
            output_name,
        ):
            planned = all_planned.get(id(consumer))
            if planned is None or str(planned[int(input_slot)]) == output_name:
                remaining = True
                break
        if not remaining:
            removals[id(boundary.adapter)] = boundary.adapter

    relevant_operators = {
        id(operator): operator
        for operator in (
            *concats,
            *closure_ops,
            *(boundary.adapter for boundary in boundaries.values() if boundary.adapter),
            *(post for update in concat_updates for post in update.posts),
            *(rewrite.operator for rewrite in main_rewrites),
            *(rewrite.operator for rewrite in alias_rewrites),
            *(rewrite.operator for rewrite in legacy_rewrites),
        )
        if operator is not None
    }
    ordered_relevant = tuple(
        sorted(
            relevant_operators.values(),
            key=lambda operator: int(_operator_index(graph_index, operator) or 0),
        )
    )
    contract_names = set()
    for operator in ordered_relevant:
        contract_names.update(
            str(value)
            for value in (*operator.inputs, *operator.outputs)
            if str(value)
        )
    if not create_permutation and legacy_adapters:
        contract_names.add(_PERMUTATION_NAME)
    tensor_contracts = []
    for name in sorted(contract_names):
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))
    return _Plan(
        seed=seed,
        concats=concats,
        closure=closure_ops,
        boundaries=tuple(boundaries[name] for name in sorted(boundaries)),
        constant_updates=tuple(constant_updates),
        concat_updates=tuple(concat_updates),
        metadata_updates=tuple(metadata_updates),
        main_rewrites=main_rewrites,
        alias_rewrites=alias_rewrites,
        legacy_rewrites=legacy_rewrites,
        legacy_adapters=tuple(legacy_adapters),
        create_permutation=create_permutation,
        removals=tuple(
            sorted(
                removals.values(),
                key=lambda operator: int(
                    _operator_index(graph_index, operator) or 0
                ),
            )
        ),
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in ordered_relevant
        ),
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
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
            for rewrite in (
                *plan.main_rewrites,
                *plan.alias_rewrites,
                *plan.legacy_rewrites,
            )
        )
        or any(
            adapter.output_name in model_ir.tensors
            for adapter in plan.legacy_adapters
        )
        or (plan.create_permutation and _PERMUTATION_NAME in model_ir.tensors)
    ):
        return False
    removal_indices = []
    for operator in plan.removals:
        operator_index = _operator_index(graph_index, operator)
        if operator_index is None:
            return False
        removal_indices.append(int(operator_index))

    if plan.create_permutation:
        model_ir.tensors[_PERMUTATION_NAME] = TensorIR(
            name=_PERMUTATION_NAME,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(_PERM_NHWC_TO_NCHW, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )
        if layout_state is not None:
            layout_state.set(_PERMUTATION_NAME)

    for update in plan.constant_updates:
        tensor = model_ir.tensors[update.name]
        transposed = np.transpose(
            np.asarray(tensor.data),
            axes=_PERM_NCHW_TO_NHWC,
        )
        if (
            str(transposed.dtype) != update.numpy_dtype
            or _freeze(transposed) != update.data_contract
        ):
            return False
        tensor.data = np.asarray(transposed)
        tensor.shape = [int(value) for value in update.shape]
        tensor.shape_signature = [int(value) for value in update.signature]
        tensor.logical_layout = LOGICAL_LAYOUT_NHWC
        tensor.physical_layout = LOGICAL_LAYOUT_NHWC
        if layout_state is not None:
            layout_state.set(
                update.name,
                logical=LOGICAL_LAYOUT_NHWC,
                physical=LOGICAL_LAYOUT_NHWC,
            )

    for rewrite in plan.main_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    metadata_by_name = {update.name: update for update in plan.metadata_updates}
    concat_ids = {id(update.operator) for update in plan.concat_updates}
    for operator in plan.closure:
        for output_name in operator.outputs:
            metadata = metadata_by_name[str(output_name)]
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

    for update in plan.concat_updates:
        if id(update.operator) not in concat_ids:
            return False
        options = dict(update.operator.options)
        options["axis"] = 3
        update.operator.options = options
        old_tensor = model_ir.tensors[update.original_output]
        old_tensor.shape = [int(value) for value in update.new_view.shape]
        old_tensor.shape_signature = [
            int(value) for value in update.new_view.signature
        ]
        old_tensor.logical_layout = LOGICAL_LAYOUT_NHWC
        old_tensor.physical_layout = LOGICAL_LAYOUT_NHWC
        if layout_state is not None:
            layout_state.set(
                update.original_output,
                logical=LOGICAL_LAYOUT_NHWC,
                physical=LOGICAL_LAYOUT_NHWC,
            )
        _set_operator_outputs(
            model_ir=model_ir,
            op=update.operator,
            new_outputs=[update.canonical_output],
            graph_index=graph_index,
        )
        canonical_tensor = model_ir.tensors[update.canonical_output]
        canonical_tensor.dtype = str(old_tensor.dtype)
        canonical_tensor.quantization = _clone_quantization(
            old_tensor.quantization
        )
        canonical_tensor.shape = [int(value) for value in update.new_view.shape]
        canonical_tensor.shape_signature = [
            int(value) for value in update.new_view.signature
        ]
        canonical_tensor.logical_layout = LOGICAL_LAYOUT_NHWC
        canonical_tensor.physical_layout = LOGICAL_LAYOUT_NHWC
        if layout_state is not None:
            layout_state.set(
                update.canonical_output,
                logical=LOGICAL_LAYOUT_NHWC,
                physical=LOGICAL_LAYOUT_NHWC,
            )

    for rewrite in plan.alias_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    graph_index.remove_operators(removal_indices)

    insertion_plans = []
    for adapter in plan.legacy_adapters:
        source_tensor = model_ir.tensors[adapter.source_name]
        if _freeze(source_tensor.quantization) != adapter.quantization_contract:
            raise RuntimeError("validated legacy tensor quantization changed")
        model_ir.tensors[adapter.output_name] = TensorIR(
            name=adapter.output_name,
            dtype=adapter.dtype,
            shape=[int(value) for value in adapter.shape],
            shape_signature=[int(value) for value in adapter.signature],
            data=None,
            is_variable=False,
            quantization=_clone_quantization(source_tensor.quantization),
            logical_layout=adapter.logical_layout,
            physical_layout=adapter.physical_layout,
        )
        if layout_state is not None:
            layout_state.set(
                adapter.output_name,
                logical=adapter.logical_layout,
                physical=adapter.physical_layout,
            )
        indices = [
            _operator_index(graph_index, operator)
            for operator in adapter.consumer_operators
        ]
        if any(index is None for index in indices):
            raise RuntimeError("validated legacy consumer disappeared")
        insertion_plans.append(
            (
                min(int(index) for index in indices if index is not None),
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[adapter.source_name, _PERMUTATION_NAME],
                    outputs=[adapter.output_name],
                ),
            )
        )
    for rewrite in plan.legacy_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    inserted = 0
    for index, operator in sorted(insertion_plans, key=lambda item: item[0]):
        graph_index.insert_operator(int(index) + inserted, operator)
        inserted += 1

    return True


def optimize_transpose_elementwise_concat_conv_nhwc_groups(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Lift a fully classified elementwise/Concat group to NHWC."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    if _PERMUTATION_NAME not in model_ir.tensors:
        model_ir.tensors[_PERMUTATION_NAME] = TensorIR(
            name=_PERMUTATION_NAME,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(_PERM_NHWC_TO_NCHW, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )
        if layout_state is not None:
            layout_state.set(_PERMUTATION_NAME)
    candidates = (
        [candidate]
        if candidate is not None
        else list(_candidate_concats(model_ir, active_index))
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
