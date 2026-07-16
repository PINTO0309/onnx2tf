from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NCW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_NWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
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
    _view,
)


_STATS_KEY = "optimized_transpose_reshape_transpose_to_flatten_hw_nhwc_chains"
_PERM_NCW_TO_NWC = (0, 2, 1)


@dataclass(frozen=True)
class _Plan:
    pre: OperatorIR
    source_name: str
    pre_output: str
    source_view: _View
    pre_view: _View
    reshape: OperatorIR
    reshape_original_inputs: Tuple[str, ...]
    reshape_output: str
    reshape_view: _View
    shape_name: str
    target_shape_data: np.ndarray
    post: OperatorIR
    post_output: str
    post_view: _View
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        plan.source_name,
        plan.pre_output,
        plan.source_view,
        plan.pre_view,
        id(plan.reshape),
        plan.reshape_original_inputs,
        plan.reshape_output,
        plan.reshape_view,
        plan.shape_name,
        _freeze(plan.target_shape_data),
        id(plan.post),
        plan.post_output,
        plan.post_view,
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


def _layout_in(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
    allowed: set[str],
) -> bool:
    return str(_layout_of(str(name), tensor, layout_state)).upper() in allowed


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


def _typed_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    expected: Tuple[int, ...],
) -> bool:
    if (
        _op_type(operator) != "TRANSPOSE"
        or len(operator.inputs) != 2
        or len(operator.outputs) != 1
    ):
        return False
    resolved = _typed_vector(
        model_ir,
        graph_index,
        str(operator.inputs[1]),
        size=len(expected),
    )
    return bool(
        resolved is not None
        and tuple(int(value) for value in resolved[1].reshape(-1)) == expected
    )


def _typed_mutable_shape(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    reshape: OperatorIR,
) -> Optional[Tuple[TensorIR, np.ndarray]]:
    if len(reshape.inputs) < 2:
        return None
    name = str(reshape.inputs[1])
    resolved = _typed_vector(model_ir, graph_index, name, size=3)
    if resolved is None or not _exclusive_slot_consumers(
        model_ir,
        graph_index,
        name,
        ((reshape, 1),),
    ):
        return None
    return resolved


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
        or not _typed_permutation(
            model_ir,
            graph_index,
            pre,
            tuple(_PERM_NHWC_TO_NCHW),
        )
    ):
        return None
    source_name = str(pre.inputs[0])
    pre_output = str(pre.outputs[0])
    graph_outputs = {str(value) for value in model_ir.outputs}
    if pre_output in graph_outputs or pre_output in graph_index.duplicate_producers:
        return None
    reshape_users = graph_index.consumers_of(pre_output)
    if len(reshape_users) != 1:
        return None
    reshape = reshape_users[0]
    reshape_index = _operator_index(graph_index, reshape)
    if (
        reshape_index is None
        or int(reshape_index) <= int(pre_index)
        or _op_type(reshape) != "RESHAPE"
        or len(reshape.inputs) < 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != pre_output
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            pre_output,
            ((reshape, 0),),
        )
    ):
        return None
    reshape_output = str(reshape.outputs[0])
    if (
        reshape_output in graph_outputs
        or reshape_output in graph_index.duplicate_producers
    ):
        return None
    post_users = graph_index.consumers_of(reshape_output)
    if len(post_users) != 1:
        return None
    post = post_users[0]
    post_index = _operator_index(graph_index, post)
    if (
        post_index is None
        or int(post_index) <= int(reshape_index)
        or not _typed_permutation(
            model_ir,
            graph_index,
            post,
            _PERM_NCW_TO_NWC,
        )
        or str(post.inputs[0]) != reshape_output
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            reshape_output,
            ((post, 0),),
        )
    ):
        return None
    post_output = str(post.outputs[0])
    if (
        post_output in graph_index.duplicate_producers
        or graph_index.producer(post_output) is not post
    ):
        return None

    source_tensor = model_ir.tensors.get(source_name)
    pre_tensor = model_ir.tensors.get(pre_output)
    reshape_tensor = model_ir.tensors.get(reshape_output)
    post_tensor = model_ir.tensors.get(post_output)
    if any(
        tensor is None
        for tensor in (source_tensor, pre_tensor, reshape_tensor, post_tensor)
    ):
        return None
    assert source_tensor is not None
    assert pre_tensor is not None
    assert reshape_tensor is not None
    assert post_tensor is not None

    source_view = _view(source_tensor)
    pre_view = _permuted_view(source_view, _PERM_NHWC_TO_NCHW)
    reshape_view = _view(reshape_tensor)
    post_view = _view(post_tensor)
    if (
        pre_view is None
        or not _positive_view(source_view, 4)
        or not _positive_view(pre_view, 4)
        or not _positive_view(reshape_view, 3)
        or not _positive_view(post_view, 3)
    ):
        return None
    n, h, w, c = (int(value) for value in source_view.shape)
    ns, hs, ws, cs = (int(value) for value in source_view.signature)
    hw = int(h) * int(w)
    hw_signature = int(hs) * int(ws)
    expected_reshape_view = _View(
        shape=(n, c, hw),
        signature=(ns, cs, hw_signature),
        dtype=source_view.dtype,
    )
    expected_post_view = _View(
        shape=(n, hw, c),
        signature=(ns, hw_signature, cs),
        dtype=source_view.dtype,
    )
    if (
        not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(pre_index),
        )
        or _view(pre_tensor) != pre_view
        or reshape_view != expected_reshape_view
        or post_view != expected_post_view
        or not _same_quantization(
            source_tensor,
            pre_tensor,
            reshape_tensor,
            post_tensor,
        )
        or not _layout_in(
            source_name,
            source_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not _layout_in(
            pre_output,
            pre_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not _layout_in(
            reshape_output,
            reshape_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCW, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not _layout_in(
            post_output,
            post_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NWC, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    shape_name = str(reshape.inputs[1])
    shape_constant = _typed_mutable_shape(model_ir, graph_index, reshape)
    if shape_constant is None:
        return None
    _, shape_data = shape_constant
    if tuple(int(value) for value in shape_data.reshape(-1)) != reshape_view.shape:
        return None
    target_shape_data = np.asarray(
        expected_post_view.shape,
        dtype=shape_data.dtype,
    ).reshape(shape_data.shape)
    involved_names = {
        source_name,
        str(pre.inputs[1]),
        pre_output,
        shape_name,
        reshape_output,
        str(post.inputs[1]),
        post_output,
    }
    return _Plan(
        pre=pre,
        source_name=source_name,
        pre_output=pre_output,
        source_view=source_view,
        pre_view=pre_view,
        reshape=reshape,
        reshape_original_inputs=tuple(str(value) for value in reshape.inputs),
        reshape_output=reshape_output,
        reshape_view=reshape_view,
        shape_name=shape_name,
        target_shape_data=np.asarray(target_shape_data),
        post=post,
        post_output=post_output,
        post_view=post_view,
        tensor_contracts=tuple(
            _tensor_contract(name, model_ir.tensors[name])
            for name in sorted(involved_names)
            if name in model_ir.tensors
        ),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in (pre, reshape, post)
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
        plan.pre,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if (
        tuple(str(value) for value in model_ir.inputs) != plan.graph_inputs
        or tuple(str(value) for value in model_ir.outputs) != plan.graph_outputs
    ):
        return False
    pre_index = _operator_index(graph_index, plan.pre)
    post_index = _operator_index(graph_index, plan.post)
    if pre_index is None or post_index is None:
        return False

    shape_tensor = model_ir.tensors[plan.shape_name]
    shape_tensor.data = np.asarray(plan.target_shape_data)
    shape_tensor.shape = [int(value) for value in plan.target_shape_data.shape]
    shape_tensor.shape_signature = [
        int(value) for value in plan.target_shape_data.shape
    ]
    if isinstance(plan.reshape.options, dict):
        options = dict(plan.reshape.options)
        for key in ("newShape", "onnxRawNewShape"):
            if isinstance(options.get(key), list):
                options[key] = [int(value) for value in plan.post_view.shape]
        plan.reshape.options = options

    reshape_inputs = [str(value) for value in plan.reshape.inputs]
    reshape_inputs[0] = plan.source_name
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.reshape,
        new_inputs=reshape_inputs,
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.reshape,
        new_outputs=[plan.post_output],
        graph_index=graph_index,
    )
    _set_layout(
        model_ir.tensors[plan.source_name],
        plan.source_name,
        LOGICAL_LAYOUT_NHWC,
        layout_state,
    )
    _set_layout(
        model_ir.tensors[plan.post_output],
        plan.post_output,
        LOGICAL_LAYOUT_NWC,
        layout_state,
    )

    graph_index.remove_operators((int(pre_index), int(post_index)))
    return True


def optimize_transpose_flatten_hw_reshape_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Index strict static NHWC flatten-HW reshape suffixes."""

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
    pending = tuple(operator for operator in candidates if operator is not None)
    for pre in pending:
        if rewritten >= rewrite_limit:
            break
        if _operator_index(active_index, pre) is None:
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

    return {_STATS_KEY: int(rewritten)}
