from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _set_operator_outputs
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    _View,
    _consumer_slots,
    _freeze,
    _layout_of,
    _op_type,
    _operator_contract,
    _operator_index,
    _per_tensor_quantization,
    _resolved_source,
    _tensor_contract,
    _view,
)


_STATS_KEY = (
    "optimized_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains"
)
_PERM_HAD = (1, 0, 2)
_PERM_NAHD_TO_NHAD = (0, 2, 1, 3)


@dataclass(frozen=True)
class _Plan:
    reshape: OperatorIR
    source_name: str
    source_view: _View
    reshape_output: str
    reshape_original_view: _View
    reshape_target_view: _View
    reshape_shape_name: str
    reshape_shape_data: np.ndarray
    transpose: OperatorIR
    transpose_output: str
    transpose_original_view: _View
    transpose_permutation_name: str
    transpose_permutation_data: np.ndarray
    tail: OperatorIR
    tail_output: str
    tail_view: _View
    tail_shape_name: str
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.reshape),
        plan.source_name,
        plan.source_view,
        plan.reshape_output,
        plan.reshape_original_view,
        plan.reshape_target_view,
        plan.reshape_shape_name,
        _freeze(plan.reshape_shape_data),
        id(plan.transpose),
        plan.transpose_output,
        plan.transpose_original_view,
        plan.transpose_permutation_name,
        _freeze(plan.transpose_permutation_data),
        id(plan.tail),
        plan.tail_output,
        plan.tail_view,
        plan.tail_shape_name,
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _positive_view(view: _View, rank: int) -> bool:
    return bool(
        len(view.shape) == int(rank)
        and len(view.signature) == int(rank)
        and all(int(value) > 0 for value in view.shape)
        and all(int(value) > 0 for value in view.signature)
    )


def _layout_is_unknown(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
) -> bool:
    return (
        str(_layout_of(str(name), tensor, layout_state)).upper()
        == LOGICAL_LAYOUT_UNKNOWN
    )


def _same_quantization(*tensors: TensorIR) -> bool:
    return bool(
        len(tensors) > 0
        and all(_per_tensor_quantization(tensor.quantization) for tensor in tensors)
        and all(
            _freeze(tensor.quantization) == _freeze(tensors[0].quantization)
            for tensor in tensors[1:]
        )
    )


def _exclusive_slot_consumers(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    expected: Tuple[Tuple[OperatorIR, int], ...],
) -> bool:
    actual = _consumer_slots(model_ir, graph_index, str(name))
    return sorted((id(operator), int(slot)) for operator, slot in actual) == sorted(
        (id(operator), int(slot)) for operator, slot in expected
    )


def _typed_vector(
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
        or tuple(int(value) for value in tensor.shape) != (int(size),)
        or (
            tensor.shape_signature is not None
            and tuple(int(value) for value in tensor.shape_signature)
            != (int(size),)
        )
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return None
    data = np.asarray(tensor.data)
    expected_dtype = np.dtype(
        np.int32 if str(tensor.dtype) == "INT32" else np.int64
    )
    if data.dtype != expected_dtype or tuple(int(value) for value in data.shape) != (
        int(size),
    ):
        return None
    return tensor, data


def _typed_mutable_vector(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    size: int,
    owner: OperatorIR,
    input_slot: int,
) -> Optional[Tuple[TensorIR, np.ndarray]]:
    resolved = _typed_vector(model_ir, graph_index, str(name), size=int(size))
    if resolved is None or not _exclusive_slot_consumers(
        model_ir,
        graph_index,
        str(name),
        ((owner, int(input_slot)),),
    ):
        return None
    return resolved


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    reshape: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    reshape_index = _operator_index(graph_index, reshape)
    if (
        reshape_index is None
        or _op_type(reshape) != "RESHAPE"
        or len(reshape.inputs) < 2
        or len(reshape.outputs) != 1
    ):
        return None
    source_name = str(reshape.inputs[0])
    reshape_output = str(reshape.outputs[0])
    graph_outputs = {str(value) for value in model_ir.outputs}
    if reshape_output in graph_outputs or reshape_output in graph_index.duplicate_producers:
        return None
    transpose_users = graph_index.consumers_of(reshape_output)
    if len(transpose_users) != 1:
        return None
    transpose = transpose_users[0]
    transpose_index = _operator_index(graph_index, transpose)
    if (
        transpose_index is None
        or int(transpose_index) <= int(reshape_index)
        or _op_type(transpose) != "TRANSPOSE"
        or len(transpose.inputs) != 2
        or len(transpose.outputs) != 1
        or str(transpose.inputs[0]) != reshape_output
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            reshape_output,
            ((transpose, 0),),
        )
    ):
        return None
    transpose_output = str(transpose.outputs[0])
    if (
        transpose_output in graph_outputs
        or transpose_output in graph_index.duplicate_producers
    ):
        return None
    tail_users = graph_index.consumers_of(transpose_output)
    if len(tail_users) != 1:
        return None
    tail = tail_users[0]
    tail_index = _operator_index(graph_index, tail)
    if (
        tail_index is None
        or int(tail_index) <= int(transpose_index)
        or _op_type(tail) != "RESHAPE"
        or len(tail.inputs) < 2
        or len(tail.outputs) != 1
        or str(tail.inputs[0]) != transpose_output
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            transpose_output,
            ((tail, 0),),
        )
    ):
        return None
    tail_output = str(tail.outputs[0])
    if (
        tail_output in graph_index.duplicate_producers
        or graph_index.producer(tail_output) is not tail
    ):
        return None

    source_tensor = model_ir.tensors.get(source_name)
    reshape_tensor = model_ir.tensors.get(reshape_output)
    transpose_tensor = model_ir.tensors.get(transpose_output)
    tail_tensor = model_ir.tensors.get(tail_output)
    if any(
        tensor is None
        for tensor in (source_tensor, reshape_tensor, transpose_tensor, tail_tensor)
    ):
        return None
    assert source_tensor is not None
    assert reshape_tensor is not None
    assert transpose_tensor is not None
    assert tail_tensor is not None

    source_view = _view(source_tensor)
    reshape_view = _view(reshape_tensor)
    transpose_view = _view(transpose_tensor)
    tail_view = _view(tail_tensor)
    if (
        not _positive_view(source_view, 3)
        or not _positive_view(reshape_view, 3)
        or not _positive_view(transpose_view, 3)
        or not _positive_view(tail_view, 4)
    ):
        return None
    a, singleton, channels = (int(value) for value in source_view.shape)
    signature_a, signature_singleton, signature_channels = (
        int(value) for value in source_view.signature
    )
    reshape_a, heads, depth = (int(value) for value in reshape_view.shape)
    reshape_signature_a, signature_heads, signature_depth = (
        int(value) for value in reshape_view.signature
    )
    expected_transpose_view = _View(
        shape=(heads, a, depth),
        signature=(signature_heads, signature_a, signature_depth),
        dtype=source_view.dtype,
    )
    expected_tail_view = _View(
        shape=(1, heads, a, depth),
        signature=(1, signature_heads, signature_a, signature_depth),
        dtype=source_view.dtype,
    )
    if (
        singleton != 1
        or signature_singleton != 1
        or reshape_a != a
        or reshape_signature_a != signature_a
        or heads * depth != channels
        or signature_heads * signature_depth != signature_channels
        or transpose_view != expected_transpose_view
        or tail_view != expected_tail_view
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(reshape_index),
        )
        or not _same_quantization(
            source_tensor,
            reshape_tensor,
            transpose_tensor,
            tail_tensor,
        )
        or not all(
            _layout_is_unknown(name, tensor, layout_state)
            for name, tensor in (
                (source_name, source_tensor),
                (reshape_output, reshape_tensor),
                (transpose_output, transpose_tensor),
                (tail_output, tail_tensor),
            )
        )
    ):
        return None

    shape_name = str(reshape.inputs[1])
    shape_constant = _typed_mutable_vector(
        model_ir,
        graph_index,
        shape_name,
        size=3,
        owner=reshape,
        input_slot=1,
    )
    if shape_constant is None:
        return None
    _, shape_data = shape_constant
    if tuple(int(value) for value in shape_data.reshape(-1)) != reshape_view.shape:
        return None
    target_reshape_view = _View(
        shape=(1, a, heads, depth),
        signature=(1, signature_a, signature_heads, signature_depth),
        dtype=reshape_view.dtype,
    )
    target_shape_data = np.asarray(
        target_reshape_view.shape,
        dtype=shape_data.dtype,
    )

    permutation_name = str(transpose.inputs[1])
    permutation_constant = _typed_mutable_vector(
        model_ir,
        graph_index,
        permutation_name,
        size=3,
        owner=transpose,
        input_slot=1,
    )
    if permutation_constant is None:
        return None
    _, permutation_data = permutation_constant
    if tuple(int(value) for value in permutation_data.reshape(-1)) != _PERM_HAD:
        return None
    target_permutation_data = np.asarray(
        _PERM_NAHD_TO_NHAD,
        dtype=permutation_data.dtype,
    )

    tail_shape_name = str(tail.inputs[1])
    tail_shape_constant = _typed_vector(
        model_ir,
        graph_index,
        tail_shape_name,
        size=4,
    )
    if (
        tail_shape_constant is None
        or tuple(int(value) for value in tail_shape_constant[1].reshape(-1))
        != tail_view.shape
    ):
        return None

    involved_names = {
        source_name,
        shape_name,
        reshape_output,
        permutation_name,
        transpose_output,
        tail_shape_name,
        tail_output,
    }
    return _Plan(
        reshape=reshape,
        source_name=source_name,
        source_view=source_view,
        reshape_output=reshape_output,
        reshape_original_view=reshape_view,
        reshape_target_view=target_reshape_view,
        reshape_shape_name=shape_name,
        reshape_shape_data=np.asarray(target_shape_data),
        transpose=transpose,
        transpose_output=transpose_output,
        transpose_original_view=transpose_view,
        transpose_permutation_name=permutation_name,
        transpose_permutation_data=np.asarray(target_permutation_data),
        tail=tail,
        tail_output=tail_output,
        tail_view=tail_view,
        tail_shape_name=tail_shape_name,
        tensor_contracts=tuple(
            _tensor_contract(name, model_ir.tensors[name])
            for name in sorted(involved_names)
            if name in model_ir.tensors
        ),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in (reshape, transpose, tail)
        ),
        graph_inputs=tuple(str(value) for value in model_ir.inputs),
        graph_outputs=tuple(str(value) for value in model_ir.outputs),
    )


def _set_unknown_layout(
    tensor: TensorIR,
    name: str,
    layout_state: Optional[LayoutState],
) -> None:
    tensor.logical_layout = LOGICAL_LAYOUT_UNKNOWN
    tensor.physical_layout = LOGICAL_LAYOUT_UNKNOWN
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
        plan.reshape,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if (
        tuple(str(value) for value in model_ir.inputs) != plan.graph_inputs
        or tuple(str(value) for value in model_ir.outputs) != plan.graph_outputs
    ):
        return False
    tail_index = _operator_index(graph_index, plan.tail)
    if tail_index is None:
        return False

    shape_tensor = model_ir.tensors[plan.reshape_shape_name]
    shape_tensor.data = np.asarray(plan.reshape_shape_data)
    shape_tensor.shape = [int(value) for value in plan.reshape_shape_data.shape]
    shape_tensor.shape_signature = [
        int(value) for value in plan.reshape_shape_data.shape
    ]
    if isinstance(plan.reshape.options, dict):
        options = dict(plan.reshape.options)
        for key in ("newShape", "onnxRawNewShape"):
            if isinstance(options.get(key), list):
                options[key] = [
                    int(value) for value in plan.reshape_target_view.shape
                ]
        plan.reshape.options = options

    permutation_tensor = model_ir.tensors[plan.transpose_permutation_name]
    permutation_tensor.data = np.asarray(plan.transpose_permutation_data)
    permutation_tensor.shape = [
        int(value) for value in plan.transpose_permutation_data.shape
    ]
    permutation_tensor.shape_signature = [
        int(value) for value in plan.transpose_permutation_data.shape
    ]

    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.transpose,
        new_outputs=[plan.tail_output],
        graph_index=graph_index,
    )

    reshape_tensor = model_ir.tensors[plan.reshape_output]
    reshape_tensor.shape = [
        int(value) for value in plan.reshape_target_view.shape
    ]
    reshape_tensor.shape_signature = [
        int(value) for value in plan.reshape_target_view.signature
    ]
    _set_unknown_layout(
        reshape_tensor,
        plan.reshape_output,
        layout_state,
    )
    tail_tensor = model_ir.tensors[plan.tail_output]
    tail_tensor.shape = [int(value) for value in plan.tail_view.shape]
    tail_tensor.shape_signature = [int(value) for value in plan.tail_view.signature]
    _set_unknown_layout(tail_tensor, plan.tail_output, layout_state)

    graph_index.remove_operator(int(tail_index))
    return True


def optimize_attention_qkv_had_reshape_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Index the strict static QKV `[1,0,2]` rank-adapter family."""

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
                {"RESHAPE"}
            )
        ]
    )
    rewrite_limit = (
        len(candidates) if max_rewrites is None else max(0, int(max_rewrites))
    )
    rewritten = 0
    pending = tuple(operator for operator in candidates if operator is not None)
    for reshape in pending:
        if rewritten >= rewrite_limit:
            break
        if _operator_index(active_index, reshape) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            reshape,
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
