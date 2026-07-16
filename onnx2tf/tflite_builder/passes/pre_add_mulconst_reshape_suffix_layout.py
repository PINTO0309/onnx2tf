from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
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


_STATS_KEY = "optimized_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains"
_PERM_NCW_TO_NWC = (0, 2, 1)


@dataclass(frozen=True)
class _ConstantRewrite:
    source_name: str
    target_name: str
    data: np.ndarray
    clone: bool


@dataclass(frozen=True)
class _BranchPlan:
    kind: str
    original_input: str
    canonical_input: str
    adapter: OperatorIR
    adapter_output: str
    source_name: str
    source_view: _View
    remove_adapter: bool
    mul: Optional[OperatorIR] = None
    mul_data_input_index: Optional[int] = None
    mul_side_input_index: Optional[int] = None
    constant_rewrite: Optional[_ConstantRewrite] = None


@dataclass(frozen=True)
class _Plan:
    add: OperatorIR
    original_inputs: Tuple[str, str]
    add_output: str
    add_nhwc_output: str
    old_add_view: _View
    new_add_view: _View
    branches: Tuple[_BranchPlan, _BranchPlan]
    reshape: OperatorIR
    reshape_original_inputs: Tuple[str, ...]
    reshape_original_output: str
    reshape_shape_name: str
    reshape_shape_target_name: str
    reshape_shape_data: np.ndarray
    reshape_shape_clone: bool
    reshape_options: Tuple[Tuple[str, Any], ...]
    post: OperatorIR
    suffix_output: str
    legacy_users: Tuple[OperatorIR, ...]
    adapter_permutation_name: Optional[str]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.add),
        plan.original_inputs,
        plan.add_output,
        plan.add_nhwc_output,
        plan.old_add_view,
        plan.new_add_view,
        tuple(
            (
                branch.kind,
                branch.original_input,
                branch.canonical_input,
                id(branch.adapter),
                branch.adapter_output,
                branch.source_name,
                branch.source_view,
                branch.remove_adapter,
                None if branch.mul is None else id(branch.mul),
                branch.mul_data_input_index,
                branch.mul_side_input_index,
                (
                    None
                    if branch.constant_rewrite is None
                    else (
                        branch.constant_rewrite.source_name,
                        branch.constant_rewrite.target_name,
                        _freeze(branch.constant_rewrite.data),
                        branch.constant_rewrite.clone,
                    )
                ),
            )
            for branch in plan.branches
        ),
        id(plan.reshape),
        plan.reshape_original_inputs,
        plan.reshape_original_output,
        plan.reshape_shape_name,
        plan.reshape_shape_target_name,
        _freeze(plan.reshape_shape_data),
        plan.reshape_shape_clone,
        plan.reshape_options,
        id(plan.post),
        plan.suffix_output,
        tuple(id(user) for user in plan.legacy_users),
        plan.adapter_permutation_name,
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _rank4_positive(view: _View) -> bool:
    return bool(
        len(view.shape) == 4
        and len(view.signature) == 4
        and all(int(value) > 0 for value in view.shape)
    )


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


def _unique_name(model_ir: ModelIR, base: str) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in model_ir.tensors:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


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


def _typed_int_vector(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    size: int,
) -> Optional[Tuple[TensorIR, np.ndarray]]:
    tensor = model_ir.tensors.get(str(name))
    graph_names = {str(value) for value in model_ir.inputs} | {
        str(value) for value in model_ir.outputs
    }
    if (
        tensor is None
        or tensor.data is None
        or bool(tensor.is_variable)
        or str(tensor.dtype) not in {"INT32", "INT64"}
        or str(name) in graph_names
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return None
    data = np.asarray(tensor.data)
    expected_dtype = np.dtype(np.int32 if str(tensor.dtype) == "INT32" else np.int64)
    if data.dtype != expected_dtype or int(data.size) != int(size):
        return None
    return tensor, data


def _typed_rank3_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
) -> bool:
    if (
        _op_type(operator) != "TRANSPOSE"
        or len(operator.inputs) != 2
        or len(operator.outputs) != 1
    ):
        return False
    constant = _typed_int_vector(
        model_ir,
        graph_index,
        str(operator.inputs[1]),
        size=3,
    )
    return bool(
        constant is not None
        and tuple(int(value) for value in constant[1].reshape(-1)) == _PERM_NCW_TO_NWC
    )


def _broadcast_shape(
    left: Sequence[int],
    right: Sequence[int],
) -> Optional[Tuple[int, ...]]:
    output = []
    for left_value, right_value in zip(
        reversed(tuple(int(value) for value in left)),
        reversed(tuple(int(value) for value in right)),
    ):
        if left_value == right_value or left_value == 1 or right_value == 1:
            output.append(max(left_value, right_value))
        else:
            return None
    longer = left if len(left) > len(right) else right
    output.extend(
        reversed(tuple(int(value) for value in longer[: abs(len(left) - len(right))]))
    )
    return tuple(reversed(output))


def _constant_nhwc_data(
    data: np.ndarray,
    *,
    nchw_shape: Sequence[int],
    nhwc_shape: Sequence[int],
) -> Optional[np.ndarray]:
    source = np.asarray(data)
    if int(source.size) == 1:
        return np.asarray(source)
    if source.ndim not in {3, 4}:
        return None
    source_shape = tuple(int(value) for value in source.shape)
    if _broadcast_shape(nchw_shape, source_shape) is None:
        return None

    is_nchw_channelwise = bool(
        source.ndim == 4
        and source_shape[0] == 1
        and source_shape[1] > 0
        and source_shape[2:] == (1, 1)
    )
    if is_nchw_channelwise:
        rotated = np.transpose(source, axes=_PERM_NCHW_TO_NHWC)
        return (
            np.asarray(rotated).astype(source.dtype, copy=False)
            if _broadcast_shape(nhwc_shape, rotated.shape) is not None
            else None
        )
    if _broadcast_shape(nhwc_shape, source_shape) is not None:
        return np.asarray(source)

    rotated = np.asarray(source)
    permutation = _PERM_NCHW_TO_NHWC if source.ndim == 4 else (1, 2, 0)
    attempts = 1 if source.ndim == 4 else 3
    for _ in range(attempts):
        rotated = np.transpose(rotated, axes=permutation).astype(
            source.dtype,
            copy=False,
        )
        if _broadcast_shape(nhwc_shape, rotated.shape) is not None:
            return np.asarray(rotated)
    return None


def _resolve_direct_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    add: OperatorIR,
    input_name: str,
    *,
    old_view: _View,
    new_view: _View,
    layout_state: Optional[LayoutState],
) -> Optional[_BranchPlan]:
    add_index = _operator_index(graph_index, add)
    adapter_index = graph_index.producers.get(str(input_name))
    input_tensor = model_ir.tensors.get(str(input_name))
    graph_outputs = {str(value) for value in model_ir.outputs}
    if (
        add_index is None
        or adapter_index is None
        or int(adapter_index) >= int(add_index)
        or str(input_name) in graph_index.duplicate_producers
        or str(input_name) in graph_outputs
        or input_tensor is None
        or _view(input_tensor) != old_view
        or not _layout_in(
            str(input_name),
            input_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    adapter = model_ir.operators[int(adapter_index)]
    if (
        graph_index.producer(str(input_name)) is not adapter
        or not _typed_permutation(
            model_ir,
            graph_index,
            adapter,
            _PERM_NHWC_TO_NCHW,
        )
        or str(adapter.outputs[0]) != str(input_name)
        or add not in graph_index.consumers_of(str(input_name))
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
        or _view(source_tensor) != new_view
        or _permuted_view(_view(source_tensor), _PERM_NHWC_TO_NCHW) != old_view
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
        original_input=str(input_name),
        canonical_input=source_name,
        adapter=adapter,
        adapter_output=str(input_name),
        source_name=source_name,
        source_view=_view(source_tensor),
        remove_adapter=_exclusive_consumer(
            model_ir,
            graph_index,
            str(input_name),
            add,
        ),
    )


def _resolve_mul_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    add: OperatorIR,
    input_name: str,
    *,
    old_view: _View,
    new_view: _View,
    layout_state: Optional[LayoutState],
) -> Optional[_BranchPlan]:
    add_index = _operator_index(graph_index, add)
    mul_index = graph_index.producers.get(str(input_name))
    mul_output_tensor = model_ir.tensors.get(str(input_name))
    graph_names = {str(value) for value in model_ir.inputs} | {
        str(value) for value in model_ir.outputs
    }
    if (
        add_index is None
        or mul_index is None
        or int(mul_index) >= int(add_index)
        or str(input_name) in graph_index.duplicate_producers
        or str(input_name) in graph_names
        or mul_output_tensor is None
        or _view(mul_output_tensor) != old_view
        or not _exclusive_consumer(model_ir, graph_index, str(input_name), add)
    ):
        return None
    mul = model_ir.operators[int(mul_index)]
    if (
        graph_index.producer(str(input_name)) is not mul
        or _op_type(mul) != "MUL"
        or len(mul.inputs) != 2
        or len(mul.outputs) != 1
        or str(mul.outputs[0]) != str(input_name)
    ):
        return None

    for data_input_index, side_input_index in ((0, 1), (1, 0)):
        adapter_output = str(mul.inputs[data_input_index])
        side_name = str(mul.inputs[side_input_index])
        adapter_index = graph_index.producers.get(adapter_output)
        adapter_tensor = model_ir.tensors.get(adapter_output)
        side_tensor = model_ir.tensors.get(side_name)
        if (
            adapter_index is None
            or int(adapter_index) >= int(mul_index)
            or adapter_output in graph_index.duplicate_producers
            or adapter_output in graph_names
            or adapter_tensor is None
            or side_tensor is None
            or side_tensor.data is None
            or bool(side_tensor.is_variable)
            or side_name in graph_names
            or side_name in graph_index.producers
            or side_name in graph_index.duplicate_producers
            or not _per_tensor_quantization(side_tensor.quantization)
            or _view(adapter_tensor) != old_view
            or not _exclusive_consumer(
                model_ir,
                graph_index,
                adapter_output,
                mul,
            )
        ):
            continue
        adapter = model_ir.operators[int(adapter_index)]
        if (
            graph_index.producer(adapter_output) is not adapter
            or not _typed_permutation(
                model_ir,
                graph_index,
                adapter,
                _PERM_NHWC_TO_NCHW,
            )
            or str(adapter.outputs[0]) != adapter_output
        ):
            continue
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
            or _view(source_tensor) != new_view
            or _permuted_view(_view(source_tensor), _PERM_NHWC_TO_NCHW) != old_view
            or not _same_quantization(
                source_tensor,
                adapter_tensor,
                mul_output_tensor,
            )
            or not _layout_in(
                source_name,
                source_tensor,
                layout_state,
                {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
            )
        ):
            continue
        nhwc_data = _constant_nhwc_data(
            np.asarray(side_tensor.data),
            nchw_shape=old_view.shape,
            nhwc_shape=new_view.shape,
        )
        if nhwc_data is None:
            continue
        needs_update = not np.array_equal(nhwc_data, np.asarray(side_tensor.data))
        side_shared = any(
            operator is not mul
            for operator, _ in _consumer_slots(model_ir, graph_index, side_name)
        )
        target_name = (
            _unique_name(model_ir, f"{side_name}_nhwc")
            if needs_update and side_shared
            else side_name
        )
        constant_rewrite = (
            _ConstantRewrite(
                source_name=side_name,
                target_name=target_name,
                data=np.asarray(nhwc_data),
                clone=bool(side_shared),
            )
            if needs_update
            else None
        )
        return _BranchPlan(
            kind="mul_const",
            original_input=str(input_name),
            canonical_input=str(input_name),
            adapter=adapter,
            adapter_output=adapter_output,
            source_name=source_name,
            source_view=_view(source_tensor),
            remove_adapter=True,
            mul=mul,
            mul_data_input_index=int(data_input_index),
            mul_side_input_index=int(side_input_index),
            constant_rewrite=constant_rewrite,
        )
    return None


def _resolve_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    add: OperatorIR,
    input_name: str,
    *,
    old_view: _View,
    new_view: _View,
    layout_state: Optional[LayoutState],
) -> Optional[_BranchPlan]:
    direct = _resolve_direct_branch(
        model_ir,
        graph_index,
        add,
        input_name,
        old_view=old_view,
        new_view=new_view,
        layout_state=layout_state,
    )
    if direct is not None:
        return direct
    return _resolve_mul_branch(
        model_ir,
        graph_index,
        add,
        input_name,
        old_view=old_view,
        new_view=new_view,
        layout_state=layout_state,
    )


def _swapped_reshape_options(reshape: OperatorIR) -> Tuple[Tuple[str, Any], ...]:
    options = dict(reshape.options) if isinstance(reshape.options, dict) else {}
    for key in ("newShape", "onnxRawNewShape"):
        value = options.get(key)
        if isinstance(value, list) and len(value) == 3:
            options[key] = [int(value[0]), int(value[2]), int(value[1])]
    return tuple(sorted((str(key), _freeze(value)) for key, value in options.items()))


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    add: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    add_index = _operator_index(graph_index, add)
    if (
        add_index is None
        or _op_type(add) != "ADD"
        or len(add.inputs) != 2
        or len(add.outputs) != 1
    ):
        return None
    add_output = str(add.outputs[0])
    add_tensor = model_ir.tensors.get(add_output)
    graph_outputs = {str(value) for value in model_ir.outputs}
    if (
        add_tensor is None
        or add_output in graph_outputs
        or add_output in graph_index.duplicate_producers
        or graph_index.producer(add_output) is not add
        or not _rank4_positive(_view(add_tensor))
        or not _per_tensor_quantization(add_tensor.quantization)
        or not _layout_in(
            add_output,
            add_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    old_view = _view(add_tensor)
    new_view = _permuted_view(old_view, _PERM_NCHW_TO_NHWC)
    if new_view is None:
        return None

    suffix: Optional[Tuple[OperatorIR, OperatorIR]] = None
    legacy_users = []
    for consumer_index in graph_index.consumer_indices(add_output):
        if int(consumer_index) <= int(add_index):
            return None
        consumer = model_ir.operators[int(consumer_index)]
        if (
            _op_type(consumer) == "RESHAPE"
            and len(consumer.inputs) >= 2
            and len(consumer.outputs) == 1
            and str(consumer.inputs[0]) == add_output
        ):
            reshape_output = str(consumer.outputs[0])
            reshape_users = graph_index.consumers_of(reshape_output)
            if len(reshape_users) == 1:
                post = reshape_users[0]
                if (
                    _typed_rank3_permutation(model_ir, graph_index, post)
                    and str(post.inputs[0]) == reshape_output
                    and str(post.outputs[0]) not in graph_outputs
                ):
                    if suffix is not None:
                        return None
                    suffix = (consumer, post)
                    continue
        legacy_users.append(consumer)
    if suffix is None:
        return None
    reshape, post = suffix
    reshape_index = _operator_index(graph_index, reshape)
    post_index = _operator_index(graph_index, post)
    if (
        reshape_index is None
        or post_index is None
        or int(reshape_index) <= int(add_index)
        or int(post_index) <= int(reshape_index)
    ):
        return None

    reshape_output = str(reshape.outputs[0])
    suffix_output = str(post.outputs[0])
    reshape_tensor = model_ir.tensors.get(reshape_output)
    suffix_tensor = model_ir.tensors.get(suffix_output)
    expected_reshape_view = _View(
        shape=(
            old_view.shape[0],
            old_view.shape[1],
            old_view.shape[2] * old_view.shape[3],
        ),
        signature=(
            old_view.signature[0],
            old_view.signature[1],
            (
                old_view.signature[2] * old_view.signature[3]
                if old_view.signature[2] > 0 and old_view.signature[3] > 0
                else -1
            ),
        ),
        dtype=old_view.dtype,
    )
    expected_suffix_view = _View(
        shape=(
            expected_reshape_view.shape[0],
            expected_reshape_view.shape[2],
            expected_reshape_view.shape[1],
        ),
        signature=(
            expected_reshape_view.signature[0],
            expected_reshape_view.signature[2],
            expected_reshape_view.signature[1],
        ),
        dtype=old_view.dtype,
    )
    if (
        reshape_tensor is None
        or suffix_tensor is None
        or _view(reshape_tensor) != expected_reshape_view
        or _view(suffix_tensor) != expected_suffix_view
        or not _same_quantization(add_tensor, reshape_tensor, suffix_tensor)
    ):
        return None

    shape_name = str(reshape.inputs[1])
    shape_constant = _typed_int_vector(
        model_ir,
        graph_index,
        shape_name,
        size=3,
    )
    if shape_constant is None:
        return None
    shape_tensor, shape_data = shape_constant
    if (
        tuple(int(value) for value in shape_data.reshape(-1))
        != expected_reshape_view.shape
    ):
        return None
    swapped_shape_data = np.asarray(shape_data).reshape(-1).copy()
    swapped_shape_data[[1, 2]] = swapped_shape_data[[2, 1]]
    swapped_shape_data = swapped_shape_data.reshape(shape_data.shape).astype(
        shape_data.dtype,
        copy=False,
    )
    shape_shared = any(
        operator is not reshape
        for operator, _ in _consumer_slots(model_ir, graph_index, shape_name)
    )
    shape_target_name = (
        _unique_name(model_ir, f"{shape_name}_nhwc") if shape_shared else shape_name
    )

    branches = []
    for input_name in (str(add.inputs[0]), str(add.inputs[1])):
        branch = _resolve_branch(
            model_ir,
            graph_index,
            add,
            input_name,
            old_view=old_view,
            new_view=new_view,
            layout_state=layout_state,
        )
        if branch is None:
            return None
        branches.append(branch)
    branch_pair = (branches[0], branches[1])

    add_nhwc_output = (
        _unique_name(model_ir, f"{add_output}_nhwc")
        if len(legacy_users) > 0
        else add_output
    )
    adapter_permutation_name = (
        _unique_name(model_ir, f"{add_output}_adapter_perm")
        if len(legacy_users) > 0
        else None
    )

    involved_operators = [add, reshape, post, *legacy_users]
    involved_tensors = {
        add_output,
        reshape_output,
        suffix_output,
        shape_name,
        str(post.inputs[1]),
    }
    for branch in branch_pair:
        involved_operators.append(branch.adapter)
        involved_tensors.update(
            {
                branch.original_input,
                branch.adapter_output,
                branch.source_name,
            }
        )
        involved_tensors.update(str(value) for value in branch.adapter.inputs)
        if branch.mul is not None:
            involved_operators.append(branch.mul)
            involved_tensors.update(str(value) for value in branch.mul.inputs)
            involved_tensors.update(str(value) for value in branch.mul.outputs)
    involved_operators = list(_deduplicate_operators(involved_operators))
    involved_operators.sort(
        key=lambda operator: int(_operator_index(graph_index, operator) or 0)
    )
    return _Plan(
        add=add,
        original_inputs=(str(add.inputs[0]), str(add.inputs[1])),
        add_output=add_output,
        add_nhwc_output=add_nhwc_output,
        old_add_view=old_view,
        new_add_view=new_view,
        branches=branch_pair,
        reshape=reshape,
        reshape_original_inputs=tuple(str(value) for value in reshape.inputs),
        reshape_original_output=reshape_output,
        reshape_shape_name=shape_name,
        reshape_shape_target_name=shape_target_name,
        reshape_shape_data=np.asarray(swapped_shape_data),
        reshape_shape_clone=bool(shape_shared),
        reshape_options=_swapped_reshape_options(reshape),
        post=post,
        suffix_output=suffix_output,
        legacy_users=_deduplicate_operators(legacy_users),
        adapter_permutation_name=adapter_permutation_name,
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


def _copy_tensor_with_data(
    source: TensorIR,
    *,
    name: str,
    data: np.ndarray,
) -> TensorIR:
    return TensorIR(
        name=str(name),
        dtype=str(source.dtype),
        shape=[int(value) for value in data.shape],
        shape_signature=[int(value) for value in data.shape],
        data=np.asarray(data),
        is_variable=False,
        quantization=_clone_quantization(source.quantization),
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

    removals = [plan.post]
    for branch in plan.branches:
        if branch.remove_adapter:
            removals.append(branch.adapter)
    removals = list(_deduplicate_operators(removals))
    removal_indices = []
    for operator in removals:
        index = _operator_index(graph_index, operator)
        if index is None:
            return False
        removal_indices.append(int(index))

    for branch in plan.branches:
        if branch.mul is None:
            continue
        data_input_index = int(branch.mul_data_input_index or 0)
        side_input_index = int(branch.mul_side_input_index or 0)
        mul_inputs = [str(value) for value in branch.mul.inputs]
        side_name = mul_inputs[side_input_index]
        rewrite = branch.constant_rewrite
        if rewrite is not None:
            side_tensor = model_ir.tensors[rewrite.source_name]
            if rewrite.clone:
                model_ir.tensors[rewrite.target_name] = _copy_tensor_with_data(
                    side_tensor,
                    name=rewrite.target_name,
                    data=rewrite.data,
                )
                if layout_state is not None:
                    layout_state.set(
                        rewrite.target_name,
                        logical=LOGICAL_LAYOUT_UNKNOWN,
                        physical=LOGICAL_LAYOUT_UNKNOWN,
                    )
            else:
                side_tensor.data = np.asarray(rewrite.data)
                side_tensor.shape = [int(value) for value in rewrite.data.shape]
                side_tensor.shape_signature = [
                    int(value) for value in rewrite.data.shape
                ]
            side_name = rewrite.target_name
        mul_inputs[data_input_index] = branch.source_name
        mul_inputs[side_input_index] = side_name
        _set_operator_inputs(
            model_ir=model_ir,
            op=branch.mul,
            new_inputs=mul_inputs,
            graph_index=graph_index,
        )
        mul_output_tensor = model_ir.tensors[branch.original_input]
        mul_output_tensor.shape = [int(value) for value in plan.new_add_view.shape]
        mul_output_tensor.shape_signature = [
            int(value) for value in plan.new_add_view.signature
        ]
        _set_layout(
            mul_output_tensor,
            branch.original_input,
            LOGICAL_LAYOUT_NHWC,
            layout_state,
        )

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.add,
        new_inputs=[branch.canonical_input for branch in plan.branches],
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.add,
        new_outputs=[plan.add_nhwc_output],
        graph_index=graph_index,
    )

    old_add_tensor = model_ir.tensors[plan.add_output]
    if plan.add_nhwc_output != plan.add_output:
        model_ir.tensors[plan.add_nhwc_output] = TensorIR(
            name=plan.add_nhwc_output,
            dtype=str(old_add_tensor.dtype),
            shape=[int(value) for value in old_add_tensor.shape],
            shape_signature=(
                [int(value) for value in old_add_tensor.shape_signature]
                if old_add_tensor.shape_signature is not None
                else [int(value) for value in old_add_tensor.shape]
            ),
            data=None,
            is_variable=False,
            quantization=_clone_quantization(old_add_tensor.quantization),
        )
    add_nhwc_tensor = model_ir.tensors[plan.add_nhwc_output]
    add_nhwc_tensor.shape = [int(value) for value in plan.new_add_view.shape]
    add_nhwc_tensor.shape_signature = [
        int(value) for value in plan.new_add_view.signature
    ]
    _set_layout(
        add_nhwc_tensor,
        plan.add_nhwc_output,
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

    reshape_inputs = [str(value) for value in plan.reshape.inputs]
    reshape_inputs[0] = plan.add_nhwc_output
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.reshape,
        new_inputs=reshape_inputs,
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.reshape,
        new_outputs=[plan.suffix_output],
        graph_index=graph_index,
    )
    if plan.reshape_shape_clone:
        source_shape_tensor = model_ir.tensors[plan.reshape_shape_name]
        model_ir.tensors[plan.reshape_shape_target_name] = _copy_tensor_with_data(
            source_shape_tensor,
            name=plan.reshape_shape_target_name,
            data=plan.reshape_shape_data,
        )
        if layout_state is not None:
            layout_state.set(
                plan.reshape_shape_target_name,
                logical=LOGICAL_LAYOUT_UNKNOWN,
                physical=LOGICAL_LAYOUT_UNKNOWN,
            )
        reshape_inputs = [str(value) for value in plan.reshape.inputs]
        reshape_inputs[1] = plan.reshape_shape_target_name
        _set_operator_inputs(
            model_ir=model_ir,
            op=plan.reshape,
            new_inputs=reshape_inputs,
            graph_index=graph_index,
        )
    else:
        shape_tensor = model_ir.tensors[plan.reshape_shape_name]
        shape_tensor.data = np.asarray(plan.reshape_shape_data)
        shape_tensor.shape = [int(value) for value in plan.reshape_shape_data.shape]
        shape_tensor.shape_signature = [
            int(value) for value in plan.reshape_shape_data.shape
        ]
    if isinstance(plan.reshape.options, dict):
        options = dict(plan.reshape.options)
        for key in ("newShape", "onnxRawNewShape"):
            value = options.get(key)
            if isinstance(value, list) and len(value) == 3:
                options[key] = [int(value[0]), int(value[2]), int(value[1])]
        plan.reshape.options = options

    add_index_before_removal = _operator_index(graph_index, plan.add)
    if add_index_before_removal is None:
        return False
    removed_before_add = sum(
        1 for index in removal_indices if int(index) < int(add_index_before_removal)
    )
    new_add_index = int(add_index_before_removal) - int(removed_before_add)
    graph_index.remove_operators(removal_indices)

    if len(plan.legacy_users) > 0:
        if plan.adapter_permutation_name is None:
            raise RuntimeError("validated legacy Add boundary lost its permutation")
        model_ir.tensors[plan.adapter_permutation_name] = TensorIR(
            name=plan.adapter_permutation_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(_PERM_NHWC_TO_NCHW, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )
        if layout_state is not None:
            layout_state.set(
                plan.adapter_permutation_name,
                logical=LOGICAL_LAYOUT_UNKNOWN,
                physical=LOGICAL_LAYOUT_UNKNOWN,
            )
        graph_index.insert_operator(
            int(new_add_index) + 1,
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[plan.add_nhwc_output, plan.adapter_permutation_name],
                outputs=[plan.add_output],
                options={},
            ),
        )
        _set_layout(
            old_add_tensor,
            plan.add_output,
            LOGICAL_LAYOUT_NCHW,
            layout_state,
        )
    return True


def optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Index strict pre-Add rank-4 to rank-3 reshape suffix recovery."""

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
    pending = tuple(operator for operator in candidates if operator is not None)
    while rewritten < rewrite_limit:
        changed = False
        for add in pending:
            if rewritten >= rewrite_limit:
                break
            if _operator_index(active_index, add) is None:
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
                changed = True
        if not changed:
            break

    return {_STATS_KEY: int(rewritten)}
