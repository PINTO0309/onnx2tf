from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _set_operator_inputs
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
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


_STATS_KEY = "optimized_transpose_reshape_transpose_to_expanddims_nhwc_chains"
_PERM_NA_BHW_TO_NA_HWB = (0, 1, 3, 4, 2)
_PERM_NH_WAB_TO_NA_HWB = (0, 3, 1, 2, 4)


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
    reshape_original_view: _View
    reshape_target_view: _View
    reshape_shape_name: str
    reshape_shape_data: np.ndarray
    reshape_options: Tuple[Tuple[str, Any], ...]
    post: OperatorIR
    post_output: str
    post_view: _View
    post_permutation_name: str
    post_permutation_data: np.ndarray
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
        plan.reshape_original_view,
        plan.reshape_target_view,
        plan.reshape_shape_name,
        _freeze(plan.reshape_shape_data),
        plan.reshape_options,
        id(plan.post),
        plan.post_output,
        plan.post_view,
        plan.post_permutation_name,
        _freeze(plan.post_permutation_data),
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


def _typed_mutable_vector(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    *,
    size: int,
    owner: OperatorIR,
    input_slot: int,
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
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            str(name),
            ((owner, int(input_slot)),),
        )
    ):
        return None
    data = np.asarray(tensor.data)
    expected_dtype = np.dtype(
        np.int32 if str(tensor.dtype) == "INT32" else np.int64
    )
    if data.dtype != expected_dtype or int(data.size) != int(size):
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
    name = str(operator.inputs[1])
    tensor = model_ir.tensors.get(name)
    graph_names = {str(value) for value in model_ir.inputs} | {
        str(value) for value in model_ir.outputs
    }
    if (
        tensor is None
        or tensor.data is None
        or bool(tensor.is_variable)
        or str(tensor.dtype) not in {"INT32", "INT64"}
        or name in graph_names
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or tuple(int(value) for value in tensor.shape) != (len(expected),)
        or (
            tensor.shape_signature is not None
            and tuple(int(value) for value in tensor.shape_signature)
            != (len(expected),)
        )
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return False
    data = np.asarray(tensor.data)
    expected_dtype = np.dtype(
        np.int32 if str(tensor.dtype) == "INT32" else np.int64
    )
    return bool(
        data.dtype == expected_dtype
        and tuple(int(value) for value in data.shape) == (len(expected),)
        and tuple(int(value) for value in data.reshape(-1)) == expected
    )


def _target_options(
    reshape: OperatorIR,
    target_shape: Tuple[int, ...],
) -> Tuple[Tuple[str, Any], ...]:
    options = dict(reshape.options) if isinstance(reshape.options, dict) else {}
    for key in ("newShape", "onnxRawNewShape"):
        if isinstance(options.get(key), list):
            options[key] = [int(value) for value in target_shape]
    return tuple(sorted((str(key), _freeze(value)) for key, value in options.items()))


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
            _PERM_NA_BHW_TO_NA_HWB,
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
        or not _rank4_positive(source_view)
        or _view(pre_tensor) != pre_view
        or len(reshape_view.shape) != 5
        or len(reshape_view.signature) != 5
        or len(post_view.shape) != 5
        or len(post_view.signature) != 5
        or not all(int(value) > 0 for value in reshape_view.shape)
        or not all(int(value) > 0 for value in post_view.shape)
    ):
        return None
    n, h, w, c = (int(value) for value in source_view.shape)
    rn, anchors, values_per_anchor, rh, rw = (
        int(value) for value in reshape_view.shape
    )
    expected_post_shape = (n, anchors, h, w, values_per_anchor)
    expected_reshape_signature = (
        int(source_view.signature[0]),
        int(anchors),
        int(values_per_anchor),
        int(source_view.signature[1]),
        int(source_view.signature[2]),
    )
    expected_post_signature = (
        int(source_view.signature[0]),
        int(anchors),
        int(source_view.signature[1]),
        int(source_view.signature[2]),
        int(values_per_anchor),
    )
    if (
        int(anchors) <= 1
        or int(values_per_anchor) <= 0
        or (rn, rh, rw) != (n, h, w)
        or int(c) != int(anchors) * int(values_per_anchor)
        or tuple(int(value) for value in post_view.shape) != expected_post_shape
        or tuple(int(value) for value in reshape_view.signature)
        != expected_reshape_signature
        or tuple(int(value) for value in post_view.signature)
        != expected_post_signature
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(pre_index),
        )
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
    ):
        return None

    target_shape = (n, h, w, anchors, values_per_anchor)
    target_signature = (
        int(source_view.signature[0]),
        int(source_view.signature[1]),
        int(source_view.signature[2]),
        int(anchors),
        int(values_per_anchor),
    )
    target_view = _View(
        shape=target_shape,
        signature=target_signature,
        dtype=reshape_view.dtype,
    )
    shape_name = str(reshape.inputs[1])
    shape_constant = _typed_mutable_vector(
        model_ir,
        graph_index,
        shape_name,
        size=5,
        owner=reshape,
        input_slot=1,
    )
    if shape_constant is None:
        return None
    _, shape_data = shape_constant
    if (
        tuple(int(value) for value in shape_data.reshape(-1))
        != tuple(int(value) for value in reshape_view.shape)
        or tuple(int(value) for value in shape_data.reshape(-1)) == target_shape
    ):
        return None
    target_shape_data = np.asarray(target_shape, dtype=shape_data.dtype).reshape(
        shape_data.shape
    )

    post_permutation_name = str(post.inputs[1])
    post_permutation = _typed_mutable_vector(
        model_ir,
        graph_index,
        post_permutation_name,
        size=5,
        owner=post,
        input_slot=1,
    )
    if post_permutation is None:
        return None
    _, post_permutation_data = post_permutation
    if (
        tuple(int(value) for value in post_permutation_data.reshape(-1))
        != _PERM_NA_BHW_TO_NA_HWB
    ):
        return None
    target_post_permutation_data = np.asarray(
        _PERM_NH_WAB_TO_NA_HWB,
        dtype=post_permutation_data.dtype,
    ).reshape(post_permutation_data.shape)

    involved_names = {
        source_name,
        pre_output,
        str(pre.inputs[1]),
        reshape_output,
        shape_name,
        post_output,
        post_permutation_name,
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
        reshape_original_view=reshape_view,
        reshape_target_view=target_view,
        reshape_shape_name=shape_name,
        reshape_shape_data=np.asarray(target_shape_data),
        reshape_options=_target_options(reshape, target_shape),
        post=post,
        post_output=post_output,
        post_view=post_view,
        post_permutation_name=post_permutation_name,
        post_permutation_data=np.asarray(target_post_permutation_data),
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
    if pre_index is None:
        return False

    reshape_inputs = [str(value) for value in plan.reshape.inputs]
    reshape_inputs[0] = plan.source_name
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.reshape,
        new_inputs=reshape_inputs,
        graph_index=graph_index,
    )
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

    reshape_tensor = model_ir.tensors[plan.reshape_output]
    reshape_tensor.shape = [
        int(value) for value in plan.reshape_target_view.shape
    ]
    reshape_tensor.shape_signature = [
        int(value) for value in plan.reshape_target_view.signature
    ]
    _set_layout(
        reshape_tensor,
        plan.reshape_output,
        LOGICAL_LAYOUT_UNKNOWN,
        layout_state,
    )
    _set_layout(
        model_ir.tensors[plan.source_name],
        plan.source_name,
        LOGICAL_LAYOUT_NHWC,
        layout_state,
    )

    post_permutation_tensor = model_ir.tensors[plan.post_permutation_name]
    post_permutation_tensor.data = np.asarray(plan.post_permutation_data)
    post_permutation_tensor.shape = [
        int(value) for value in plan.post_permutation_data.shape
    ]
    post_permutation_tensor.shape_signature = [
        int(value) for value in plan.post_permutation_data.shape
    ]

    graph_index.remove_operator(int(pre_index))
    return True


def optimize_transpose_factorized_expanddims_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Index strict factorized rank-4 to rank-5 detection-head reshapes."""

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
