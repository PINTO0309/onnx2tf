from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_per_tensor_quantization,
    _is_same_per_tensor_quantization,
    _normalize_squeeze_axes_for_rank,
    _prune_unused_tensors,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_STATS_KEY = (
    "optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_chains"
)
_UNARY_OPS = {
    "RELU",
    "RELU6",
    "RELU_0_TO_1",
    "LEAKY_RELU",
    "LOGISTIC",
    "TANH",
    "GELU",
    "ABS",
    "NEG",
    "SQRT",
    "EXP",
    "CAST",
    "FLOOR",
    "CEIL",
    "ROUND",
    "HARD_SWISH",
}


@dataclass(frozen=True)
class _TensorContract:
    tensor: TensorIR
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _RewritePlan:
    pre: OperatorIR
    squeeze: OperatorIR
    unary: OperatorIR
    expand: OperatorIR
    post: OperatorIR
    source_name: str
    output_name: str
    output_tensor: TensorIR
    output_dtype: str
    output_quantization: Any


def _tensor_contract(
    model_ir: ModelIR,
    tensor_name: str,
    rank: int,
) -> Optional[_TensorContract]:
    tensor = model_ir.tensors.get(str(tensor_name))
    if tensor is None:
        return None
    try:
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
    except (TypeError, ValueError):
        return None
    if len(shape) != int(rank) or len(signature) != int(rank):
        return None
    if any(int(value) <= 0 for value in shape) or any(
        int(signature_value) == 0
        or int(signature_value) < -1
        or (int(signature_value) > 0 and int(signature_value) != int(shape_value))
        for shape_value, signature_value in zip(shape, signature)
    ):
        return None
    return _TensorContract(
        tensor=tensor,
        shape=shape,
        signature=signature,
    )


def _constant_vector(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    size: int,
    public_inputs: set[str],
) -> Optional[Tuple[int, ...]]:
    name = str(tensor_name)
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or name in public_inputs
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or str(tensor.dtype) not in {"INT32", "INT64"}
        or list(tensor.shape) != [int(size)]
        or (
            tensor.shape_signature is not None
            and [int(value) for value in tensor.shape_signature] != [int(size)]
        )
    ):
        return None
    try:
        data = np.asarray(tensor.data)
        expected_dtype = np.int32 if str(tensor.dtype) == "INT32" else np.int64
        if data.dtype != np.dtype(expected_dtype) or data.size != int(size):
            return None
        return tuple(int(value) for value in data.reshape(-1).tolist())
    except Exception:
        return None


def _producer_is_valid(
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    expected_index: int,
) -> bool:
    name = str(tensor_name)
    return name not in graph_index.duplicate_producers and graph_index.producers.get(
        name
    ) == int(expected_index)


def _quantization_contract(contracts: Tuple[_TensorContract, ...]) -> bool:
    quantizations = [contract.tensor.quantization for contract in contracts]
    if all(quantization is None for quantization in quantizations):
        return True
    if not all(
        _is_per_tensor_quantization(quantization)
        for quantization in quantizations
    ):
        return False
    anchor = quantizations[0]
    return all(
        _is_same_per_tensor_quantization(anchor, quantization)
        for quantization in quantizations[1:]
    )


def _squeeze_axis(
    squeeze: OperatorIR,
    pre_contract: _TensorContract,
    squeeze_contract: _TensorContract,
) -> Optional[int]:
    options = dict(squeeze.options) if isinstance(squeeze.options, dict) else {}
    if "squeezeDims" in options:
        try:
            raw_axes = np.asarray(options.get("squeezeDims", []), dtype=np.int64)
            normalized_axes = _normalize_squeeze_axes_for_rank(
                [int(value) for value in raw_axes.reshape(-1).tolist()],
                4,
            )
        except Exception:
            return None
        if normalized_axes is None or len(normalized_axes) != 1:
            return None
        candidates = [int(normalized_axes[0])]
    else:
        candidates = [
            axis
            for axis in range(4)
            if int(pre_contract.shape[axis]) == 1
            and int(pre_contract.signature[axis]) == 1
        ]

    matches = []
    for axis in candidates:
        if (
            int(pre_contract.shape[axis]) != 1
            or int(pre_contract.signature[axis]) != 1
        ):
            continue
        expected_shape = tuple(
            value
            for index, value in enumerate(pre_contract.shape)
            if int(index) != int(axis)
        )
        expected_signature = tuple(
            value
            for index, value in enumerate(pre_contract.signature)
            if int(index) != int(axis)
        )
        if (
            squeeze_contract.shape == expected_shape
            and squeeze_contract.signature == expected_signature
        ):
            matches.append(int(axis))
    if len(matches) != 1:
        return None
    return int(matches[0])


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_RewritePlan]:
    pre = model_ir.operators[int(pre_index)]
    if len(pre.inputs) != 2 or len(pre.outputs) != 1:
        return None
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    if (
        _constant_vector(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NHWC_TO_NCHW
    ):
        return None

    source_name = str(pre.inputs[0])
    pre_output_name = str(pre.outputs[0])
    if (
        not source_name
        or not pre_output_name
        or pre_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, pre_output_name, pre_index)
    ):
        return None
    source_contract = _tensor_contract(model_ir, source_name, 4)
    pre_contract = _tensor_contract(model_ir, pre_output_name, 4)
    if source_contract is None or pre_contract is None:
        return None
    if (
        pre_contract.shape
        != tuple(source_contract.shape[index] for index in _PERM_NHWC_TO_NCHW)
        or pre_contract.signature
        != tuple(source_contract.signature[index] for index in _PERM_NHWC_TO_NCHW)
    ):
        return None

    source_producer = graph_index.producers.get(source_name)
    if source_name in graph_index.duplicate_producers:
        return None
    if source_producer is not None:
        if source_name in public_inputs or int(source_producer) >= int(pre_index):
            return None
    elif source_name not in public_inputs and source_contract.tensor.data is None:
        return None

    pre_users = graph_index.consumer_indices(pre_output_name)
    if len(pre_users) != 1:
        return None
    squeeze_index = int(pre_users[0])
    if squeeze_index <= int(pre_index):
        return None
    squeeze = model_ir.operators[squeeze_index]
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or len(squeeze.outputs) != 1
        or str(squeeze.inputs[0]) != pre_output_name
    ):
        return None
    squeeze_output_name = str(squeeze.outputs[0])
    if (
        not squeeze_output_name
        or squeeze_output_name in public_inputs | public_outputs
        or not _producer_is_valid(
            graph_index,
            squeeze_output_name,
            squeeze_index,
        )
    ):
        return None
    squeeze_contract = _tensor_contract(model_ir, squeeze_output_name, 3)
    if squeeze_contract is None:
        return None
    squeeze_axis = _squeeze_axis(squeeze, pre_contract, squeeze_contract)
    if squeeze_axis is None:
        return None

    squeeze_users = graph_index.consumer_indices(squeeze_output_name)
    if len(squeeze_users) != 1:
        return None
    unary_index = int(squeeze_users[0])
    if unary_index <= squeeze_index:
        return None
    unary = model_ir.operators[unary_index]
    if (
        str(unary.op_type) not in _UNARY_OPS
        or len(unary.inputs) != 1
        or len(unary.outputs) != 1
        or str(unary.inputs[0]) != squeeze_output_name
    ):
        return None
    unary_output_name = str(unary.outputs[0])
    if (
        not unary_output_name
        or unary_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, unary_output_name, unary_index)
    ):
        return None
    unary_contract = _tensor_contract(model_ir, unary_output_name, 3)
    if (
        unary_contract is None
        or unary_contract.shape != squeeze_contract.shape
        or unary_contract.signature != squeeze_contract.signature
    ):
        return None

    unary_users = graph_index.consumer_indices(unary_output_name)
    if len(unary_users) != 1:
        return None
    expand_index = int(unary_users[0])
    if expand_index <= unary_index:
        return None
    expand = model_ir.operators[expand_index]
    if (
        str(expand.op_type) != "EXPAND_DIMS"
        or len(expand.inputs) != 2
        or len(expand.outputs) != 1
        or str(expand.inputs[0]) != unary_output_name
    ):
        return None
    expand_axis = _constant_vector(
        model_ir,
        graph_index,
        str(expand.inputs[1]),
        1,
        public_inputs,
    )
    if expand_axis is None:
        return None
    normalized_expand_axis = int(expand_axis[0])
    if normalized_expand_axis < 0:
        normalized_expand_axis += 4
    if normalized_expand_axis != int(squeeze_axis):
        return None

    expand_output_name = str(expand.outputs[0])
    if (
        not expand_output_name
        or expand_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, expand_output_name, expand_index)
    ):
        return None
    expand_contract = _tensor_contract(model_ir, expand_output_name, 4)
    if (
        expand_contract is None
        or expand_contract.shape != pre_contract.shape
        or expand_contract.signature != pre_contract.signature
    ):
        return None

    expand_users = graph_index.consumer_indices(expand_output_name)
    if len(expand_users) != 1:
        return None
    post_index = int(expand_users[0])
    if post_index <= expand_index:
        return None
    post = model_ir.operators[post_index]
    if (
        str(post.op_type) != "TRANSPOSE"
        or len(post.inputs) != 2
        or len(post.outputs) != 1
        or str(post.inputs[0]) != expand_output_name
        or _constant_vector(
            model_ir,
            graph_index,
            str(post.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NCHW_TO_NHWC
    ):
        return None
    output_name = str(post.outputs[0])
    if (
        not output_name
        or output_name in public_inputs
        or not _producer_is_valid(graph_index, output_name, post_index)
    ):
        return None
    output_contract = _tensor_contract(model_ir, output_name, 4)
    if (
        output_contract is None
        or output_contract.shape != source_contract.shape
        or output_contract.signature != source_contract.signature
    ):
        return None
    if any(
        int(consumer_index) <= int(post_index)
        for consumer_index in graph_index.consumer_indices(output_name)
    ):
        return None

    data_names = (
        source_name,
        pre_output_name,
        squeeze_output_name,
        unary_output_name,
        expand_output_name,
        output_name,
    )
    if len(set(data_names)) != len(data_names):
        return None

    input_group = (source_contract, pre_contract, squeeze_contract)
    output_group = (unary_contract, expand_contract)
    if (
        len({str(contract.tensor.dtype) for contract in input_group}) != 1
        or len({str(contract.tensor.dtype) for contract in output_group}) != 1
        or not _quantization_contract(input_group)
        or not _quantization_contract(output_group)
    ):
        return None
    if str(unary.op_type) != "CAST" and (
        str(source_contract.tensor.dtype) != str(unary_contract.tensor.dtype)
        or not _quantization_contract(input_group + output_group)
    ):
        return None

    try:
        output_quantization = _clone_quantization(
            unary_contract.tensor.quantization
        )
    except Exception:
        return None
    return _RewritePlan(
        pre=pre,
        squeeze=squeeze,
        unary=unary,
        expand=expand,
        post=post,
        source_name=source_name,
        output_name=output_name,
        output_tensor=output_contract.tensor,
        output_dtype=str(unary_contract.tensor.dtype),
        output_quantization=output_quantization,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _RewritePlan,
) -> bool:
    indices = [
        graph_index.operator_index(operator)
        for operator in (
            plan.pre,
            plan.squeeze,
            plan.unary,
            plan.expand,
            plan.post,
        )
    ]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != 5:
        return False

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.unary,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.unary,
        new_outputs=[plan.output_name],
        graph_index=graph_index,
    )
    plan.output_tensor.dtype = str(plan.output_dtype)
    plan.output_tensor.quantization = plan.output_quantization
    graph_index.remove_operators(
        [resolved[index] for index in (0, 1, 3, 4)]
    )
    return True


def _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    required = {"TRANSPOSE": 2, "SQUEEZE": 1, "EXPAND_DIMS": 1}
    has_unary = False
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required and required[op_type] > 0:
            required[op_type] -= 1
        if op_type in _UNARY_OPS:
            has_unary = True
    if not has_unary or any(count > 0 for count in required.values()):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {_STATS_KEY: 0}

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    candidates = [
        model_ir.operators[index]
        for index in active_index.operator_indices("TRANSPOSE")
    ]
    rewritten = 0
    for pre in candidates:
        pre_index = active_index.operator_index(pre)
        if pre_index is None:
            continue
        plan = _resolve_candidate(model_ir, active_index, pre_index)
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
