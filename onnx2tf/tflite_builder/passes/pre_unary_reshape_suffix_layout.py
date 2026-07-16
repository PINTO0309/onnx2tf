from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCW,
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NWC,
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
    _typed_permutation,
    _view,
)


_STATS_KEY = "optimized_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains"
_PERM_NCW_TO_NWC = (0, 2, 1)


@dataclass(frozen=True)
class _Plan:
    pre: OperatorIR
    source_name: str
    pre_output: str
    source_view: _View
    nchw_view: _View
    logistic: OperatorIR
    logistic_output: str
    mul: OperatorIR
    mul_data_input_index: int
    mul_logistic_input_index: int
    mul_output: str
    reshape: OperatorIR
    reshape_original_inputs: Tuple[str, ...]
    reshape_output: str
    reshape_view: _View
    shape_name: str
    swapped_shape_data: np.ndarray
    swapped_options: Tuple[Tuple[str, Any], ...]
    post: OperatorIR
    suffix_output: str
    suffix_view: _View
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
        plan.nchw_view,
        id(plan.logistic),
        plan.logistic_output,
        id(plan.mul),
        plan.mul_data_input_index,
        plan.mul_logistic_input_index,
        plan.mul_output,
        id(plan.reshape),
        plan.reshape_original_inputs,
        plan.reshape_output,
        plan.reshape_view,
        plan.shape_name,
        _freeze(plan.swapped_shape_data),
        plan.swapped_options,
        id(plan.post),
        plan.suffix_output,
        plan.suffix_view,
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


def _typed_shape_vector(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    reshape: OperatorIR,
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
            ((reshape, 1),),
        )
    ):
        return None
    data = np.asarray(tensor.data)
    expected_dtype = np.dtype(
        np.int32 if str(tensor.dtype) == "INT32" else np.int64
    )
    if data.dtype != expected_dtype or int(data.size) != 3:
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
    permutation_name = str(operator.inputs[1])
    tensor = model_ir.tensors.get(permutation_name)
    graph_names = {str(value) for value in model_ir.inputs} | {
        str(value) for value in model_ir.outputs
    }
    if (
        tensor is None
        or tensor.data is None
        or bool(tensor.is_variable)
        or str(tensor.dtype) not in {"INT32", "INT64"}
        or permutation_name in graph_names
        or permutation_name in graph_index.producers
        or permutation_name in graph_index.duplicate_producers
        or not _per_tensor_quantization(tensor.quantization)
    ):
        return False
    data = np.asarray(tensor.data)
    expected_dtype = np.dtype(
        np.int32 if str(tensor.dtype) == "INT32" else np.int64
    )
    return bool(
        data.dtype == expected_dtype
        and int(data.size) == 3
        and tuple(int(value) for value in data.reshape(-1)) == _PERM_NCW_TO_NWC
    )


def _swapped_options(reshape: OperatorIR) -> Tuple[Tuple[str, Any], ...]:
    options = dict(reshape.options) if isinstance(reshape.options, dict) else {}
    for key in ("newShape", "onnxRawNewShape"):
        value = options.get(key)
        if isinstance(value, list) and len(value) == 3:
            options[key] = [int(value[0]), int(value[2]), int(value[1])]
    return tuple(sorted((str(key), _freeze(value)) for key, value in options.items()))


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    mul: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    mul_index = _operator_index(graph_index, mul)
    if (
        mul_index is None
        or _op_type(mul) != "MUL"
        or len(mul.inputs) != 2
        or len(mul.outputs) != 1
    ):
        return None

    logistic: Optional[OperatorIR] = None
    logistic_index: Optional[int] = None
    logistic_output = ""
    source_name = ""
    data_input_index: Optional[int] = None
    logistic_input_index: Optional[int] = None
    for candidate_logistic_input_index in (0, 1):
        candidate_data_input_index = 1 - int(candidate_logistic_input_index)
        candidate_logistic_output = str(
            mul.inputs[int(candidate_logistic_input_index)]
        )
        candidate_source_name = str(mul.inputs[int(candidate_data_input_index)])
        candidate_logistic_index = graph_index.producers.get(
            candidate_logistic_output
        )
        if (
            candidate_logistic_index is None
            or int(candidate_logistic_index) >= int(mul_index)
            or candidate_logistic_output in graph_index.duplicate_producers
        ):
            continue
        candidate_logistic = model_ir.operators[int(candidate_logistic_index)]
        if (
            graph_index.producer(candidate_logistic_output)
            is not candidate_logistic
            or _op_type(candidate_logistic) != "LOGISTIC"
            or len(candidate_logistic.inputs) != 1
            or len(candidate_logistic.outputs) != 1
            or str(candidate_logistic.inputs[0]) != candidate_source_name
            or str(candidate_logistic.outputs[0]) != candidate_logistic_output
            or not _exclusive_slot_consumers(
                model_ir,
                graph_index,
                candidate_logistic_output,
                ((mul, int(candidate_logistic_input_index)),),
            )
        ):
            continue
        logistic = candidate_logistic
        logistic_index = int(candidate_logistic_index)
        logistic_output = candidate_logistic_output
        source_name = candidate_source_name
        data_input_index = int(candidate_data_input_index)
        logistic_input_index = int(candidate_logistic_input_index)
        break
    if (
        logistic is None
        or logistic_index is None
        or data_input_index is None
        or logistic_input_index is None
    ):
        return None

    graph_outputs = {str(value) for value in model_ir.outputs}
    mul_output = str(mul.outputs[0])
    pre_index = graph_index.producers.get(source_name)
    if (
        pre_index is None
        or int(pre_index) >= int(logistic_index)
        or int(pre_index) >= int(mul_index)
        or source_name in graph_index.duplicate_producers
        or source_name in graph_outputs
        or logistic_output in graph_outputs
        or mul_output in graph_outputs
    ):
        return None
    pre = model_ir.operators[int(pre_index)]
    if (
        graph_index.producer(source_name) is not pre
        or not _typed_permutation(
            model_ir,
            graph_index,
            pre,
            _PERM_NHWC_TO_NCHW,
        )
        or str(pre.outputs[0]) != source_name
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            source_name,
            ((logistic, 0), (mul, int(data_input_index))),
        )
    ):
        return None
    pre_source_name = str(pre.inputs[0])
    if pre_source_name in graph_outputs:
        return None

    reshape_users = graph_index.consumers_of(mul_output)
    if len(reshape_users) != 1:
        return None
    reshape = reshape_users[0]
    reshape_index = _operator_index(graph_index, reshape)
    if (
        reshape_index is None
        or int(reshape_index) <= int(mul_index)
        or _op_type(reshape) != "RESHAPE"
        or len(reshape.inputs) < 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != mul_output
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            mul_output,
            ((reshape, 0),),
        )
    ):
        return None
    reshape_output = str(reshape.outputs[0])
    if reshape_output in graph_outputs:
        return None
    post_users = graph_index.consumers_of(reshape_output)
    if len(post_users) != 1:
        return None
    post = post_users[0]
    post_index = _operator_index(graph_index, post)
    if (
        post_index is None
        or int(post_index) <= int(reshape_index)
        or not _typed_rank3_permutation(model_ir, graph_index, post)
        or str(post.inputs[0]) != reshape_output
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            reshape_output,
            ((post, 0),),
        )
    ):
        return None
    suffix_output = str(post.outputs[0])

    source_tensor = model_ir.tensors.get(pre_source_name)
    pre_tensor = model_ir.tensors.get(source_name)
    logistic_tensor = model_ir.tensors.get(logistic_output)
    mul_tensor = model_ir.tensors.get(mul_output)
    reshape_tensor = model_ir.tensors.get(reshape_output)
    suffix_tensor = model_ir.tensors.get(suffix_output)
    if any(
        tensor is None
        for tensor in (
            source_tensor,
            pre_tensor,
            logistic_tensor,
            mul_tensor,
            reshape_tensor,
            suffix_tensor,
        )
    ):
        return None
    assert source_tensor is not None
    assert pre_tensor is not None
    assert logistic_tensor is not None
    assert mul_tensor is not None
    assert reshape_tensor is not None
    assert suffix_tensor is not None

    source_view = _view(source_tensor)
    nchw_view = _permuted_view(source_view, _PERM_NHWC_TO_NCHW)
    if nchw_view is None or not _rank4_positive(source_view):
        return None
    hw_shape = int(nchw_view.shape[2]) * int(nchw_view.shape[3])
    hw_signature = (
        int(nchw_view.signature[2]) * int(nchw_view.signature[3])
        if int(nchw_view.signature[2]) > 0
        and int(nchw_view.signature[3]) > 0
        else -1
    )
    reshape_view = _View(
        shape=(nchw_view.shape[0], nchw_view.shape[1], hw_shape),
        signature=(
            nchw_view.signature[0],
            nchw_view.signature[1],
            hw_signature,
        ),
        dtype=nchw_view.dtype,
    )
    suffix_view = _View(
        shape=(reshape_view.shape[0], reshape_view.shape[2], reshape_view.shape[1]),
        signature=(
            reshape_view.signature[0],
            reshape_view.signature[2],
            reshape_view.signature[1],
        ),
        dtype=reshape_view.dtype,
    )
    if (
        not _resolved_source(
            model_ir,
            graph_index,
            name=pre_source_name,
            before_index=int(pre_index),
        )
        or _view(pre_tensor) != nchw_view
        or _view(logistic_tensor) != nchw_view
        or _view(mul_tensor) != nchw_view
        or _view(reshape_tensor) != reshape_view
        or _view(suffix_tensor) != suffix_view
        or not _same_quantization(
            source_tensor,
            pre_tensor,
            logistic_tensor,
            mul_tensor,
            reshape_tensor,
            suffix_tensor,
        )
        or not _layout_in(
            pre_source_name,
            source_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not all(
            _layout_in(
                name,
                tensor,
                layout_state,
                {LOGICAL_LAYOUT_NCHW, LOGICAL_LAYOUT_UNKNOWN},
            )
            for name, tensor in (
                (source_name, pre_tensor),
                (logistic_output, logistic_tensor),
                (mul_output, mul_tensor),
            )
        )
        or not _layout_in(
            reshape_output,
            reshape_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NCW, LOGICAL_LAYOUT_UNKNOWN},
        )
        or not _layout_in(
            suffix_output,
            suffix_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NWC, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None

    shape_name = str(reshape.inputs[1])
    shape_constant = _typed_shape_vector(
        model_ir,
        graph_index,
        shape_name,
        reshape,
    )
    if shape_constant is None:
        return None
    _, shape_data = shape_constant
    if tuple(int(value) for value in shape_data.reshape(-1)) != reshape_view.shape:
        return None
    swapped_shape_data = np.asarray(shape_data).reshape(-1).copy()
    swapped_shape_data[[1, 2]] = swapped_shape_data[[2, 1]]
    swapped_shape_data = swapped_shape_data.reshape(shape_data.shape).astype(
        shape_data.dtype,
        copy=False,
    )

    involved_operators = (pre, logistic, mul, reshape, post)
    involved_names = {
        pre_source_name,
        source_name,
        str(pre.inputs[1]),
        logistic_output,
        mul_output,
        reshape_output,
        shape_name,
        suffix_output,
        str(post.inputs[1]),
    }
    return _Plan(
        pre=pre,
        source_name=pre_source_name,
        pre_output=source_name,
        source_view=source_view,
        nchw_view=nchw_view,
        logistic=logistic,
        logistic_output=logistic_output,
        mul=mul,
        mul_data_input_index=int(data_input_index),
        mul_logistic_input_index=int(logistic_input_index),
        mul_output=mul_output,
        reshape=reshape,
        reshape_original_inputs=tuple(str(value) for value in reshape.inputs),
        reshape_output=reshape_output,
        reshape_view=reshape_view,
        shape_name=shape_name,
        swapped_shape_data=np.asarray(swapped_shape_data),
        swapped_options=_swapped_options(reshape),
        post=post,
        suffix_output=suffix_output,
        suffix_view=suffix_view,
        tensor_contracts=tuple(
            _tensor_contract(name, model_ir.tensors[name])
            for name in sorted(involved_names)
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
        plan.mul,
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

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.logistic,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.mul,
        input_index=int(plan.mul_data_input_index),
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )

    for name in (plan.logistic_output, plan.mul_output):
        tensor = model_ir.tensors[name]
        tensor.shape = [int(value) for value in plan.source_view.shape]
        tensor.shape_signature = [int(value) for value in plan.source_view.signature]
        _set_layout(
            tensor,
            name,
            LOGICAL_LAYOUT_NHWC,
            layout_state,
        )
    _set_layout(
        model_ir.tensors[plan.source_name],
        plan.source_name,
        LOGICAL_LAYOUT_NHWC,
        layout_state,
    )

    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.reshape,
        new_outputs=[plan.suffix_output],
        graph_index=graph_index,
    )
    shape_tensor = model_ir.tensors[plan.shape_name]
    shape_tensor.data = np.asarray(plan.swapped_shape_data)
    shape_tensor.shape = [int(value) for value in plan.swapped_shape_data.shape]
    shape_tensor.shape_signature = [
        int(value) for value in plan.swapped_shape_data.shape
    ]
    if isinstance(plan.reshape.options, dict):
        options = dict(plan.reshape.options)
        for key in ("newShape", "onnxRawNewShape"):
            value = options.get(key)
            if isinstance(value, list) and len(value) == 3:
                options[key] = [int(value[0]), int(value[2]), int(value[1])]
        plan.reshape.options = options

    suffix_tensor = model_ir.tensors[plan.suffix_output]
    suffix_tensor.dtype = str(plan.suffix_view.dtype)
    suffix_tensor.quantization = _clone_quantization(
        model_ir.tensors[plan.reshape_output].quantization
    )
    suffix_tensor.shape = [int(value) for value in plan.suffix_view.shape]
    suffix_tensor.shape_signature = [
        int(value) for value in plan.suffix_view.signature
    ]
    _set_layout(
        suffix_tensor,
        plan.suffix_output,
        LOGICAL_LAYOUT_NWC,
        layout_state,
    )

    graph_index.remove_operators((int(pre_index), int(post_index)))
    return True


def optimize_transpose_pre_swish_reshape_transpose_suffix_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Index the strict Swish rank-4 to rank-3 reshape suffix family."""

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
            for index in active_index.operator_indices_for_normalized_types({"MUL"})
        ]
    )
    rewrite_limit = (
        len(candidates) if max_rewrites is None else max(0, int(max_rewrites))
    )
    rewritten = 0
    pending = tuple(operator for operator in candidates if operator is not None)
    while rewritten < rewrite_limit:
        changed = False
        for mul in pending:
            if rewritten >= rewrite_limit:
                break
            if _operator_index(active_index, mul) is None:
                continue
            plan = _resolve_candidate(
                model_ir,
                active_index,
                mul,
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
