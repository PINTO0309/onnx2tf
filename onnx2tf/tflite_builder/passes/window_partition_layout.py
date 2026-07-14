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
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_TARGET_PERM = (0, 1, 3, 2, 4, 5)
_STATS_KEY = "optimized_window_partition_reshape_transpose_to_space_to_depth_chains"
_REVERSE_STATS_KEY = (
    "optimized_window_reverse_reshape_transpose_to_depth_to_space_chains"
)


@dataclass(frozen=True)
class _ConstantVector:
    tensor: TensorIR
    values: Tuple[int, ...]


@dataclass(frozen=True)
class _RewritePlan:
    reshape1: OperatorIR
    transpose: OperatorIR
    transpose_tensor: TensorIR
    transpose_shape: Tuple[int, ...]
    transpose_signature: Tuple[int, ...]
    block_size: int
    reshape2: OperatorIR
    reshape2_tensor: TensorIR
    reshape2_signature: Tuple[int, ...]
    reshape2_options: Optional[Dict[str, Any]]
    reshape2_shape_tensor: Optional[TensorIR]
    reshape2_shape_data: Optional[np.ndarray]
    input_name: str


@dataclass(frozen=True)
class _ReverseRewritePlan:
    reshape1: OperatorIR
    reshape1_tensor: TensorIR
    reshape1_shape: Tuple[int, ...]
    reshape1_signature: Tuple[int, ...]
    reshape1_options: Optional[Dict[str, Any]]
    reshape1_shape_tensor: TensorIR
    reshape1_shape_data: np.ndarray
    reshape1_shape_clone_name: Optional[str]
    reshape1_shape_clone: Optional[TensorIR]
    transpose: OperatorIR
    block_size: int
    reshape2: OperatorIR
    output_name: str


def _shape_contract(
    tensor: Optional[TensorIR],
    rank: int,
) -> Optional[tuple[list[int], list[int]]]:
    if tensor is None:
        return None
    try:
        shape = [int(value) for value in tensor.shape]
        signature = (
            list(shape)
            if tensor.shape_signature is None
            else [int(value) for value in tensor.shape_signature]
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
    return shape, signature


def _constant_vector(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    size: int,
    model_inputs: set[str],
) -> Optional[_ConstantVector]:
    name = str(tensor_name)
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or name in model_inputs
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
        if data.size != int(size) or data.dtype != np.dtype(expected_dtype):
            return None
        values = tuple(int(value) for value in data.reshape(-1).tolist())
    except Exception:
        return None
    return _ConstantVector(tensor=tensor, values=values)


def _quantization_contract(tensors: Tuple[TensorIR, ...]) -> bool:
    quantizations = [tensor.quantization for tensor in tensors]
    if all(quantization is None for quantization in quantizations):
        return True
    if not all(
        _is_per_tensor_quantization(quantization) for quantization in quantizations
    ):
        return False
    anchor = quantizations[0]
    return all(
        _is_same_per_tensor_quantization(anchor, quantization)
        for quantization in quantizations[1:]
    )


def _producer_is_valid(
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    expected_index: int,
) -> bool:
    name = str(tensor_name)
    return name not in graph_index.duplicate_producers and graph_index.producers.get(
        name
    ) == int(expected_index)


def _candidate_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    reshape1_index: int,
) -> Optional[_RewritePlan]:
    reshape1 = model_ir.operators[int(reshape1_index)]
    if len(reshape1.inputs) != 2 or len(reshape1.outputs) != 1:
        return None
    input_name = str(reshape1.inputs[0])
    reshape1_name = str(reshape1.outputs[0])
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    if (
        not input_name
        or not reshape1_name
        or reshape1_name in public_inputs | public_outputs
        or not _producer_is_valid(
            graph_index,
            reshape1_name,
            int(reshape1_index),
        )
    ):
        return None

    input_tensor = model_ir.tensors.get(input_name)
    reshape1_tensor = model_ir.tensors.get(reshape1_name)
    input_contract = _shape_contract(input_tensor, 4)
    reshape1_contract = _shape_contract(reshape1_tensor, 6)
    if input_contract is None or reshape1_contract is None:
        return None
    input_shape, input_signature = input_contract
    reshape1_shape, reshape1_signature = reshape1_contract
    n, h, w, channels = input_shape
    n1, out_h, block_h, out_w, block_w, channels1 = reshape1_shape
    if (
        n1 != n
        or channels1 != channels
        or block_h <= 1
        or block_h != block_w
        or out_h * block_h != h
        or out_w * block_h != w
    ):
        return None
    block_size = int(block_h)
    expected_reshape1_signature = [
        int(input_signature[0]),
        -1 if int(input_signature[1]) < 0 else int(out_h),
        block_size,
        -1 if int(input_signature[2]) < 0 else int(out_w),
        block_size,
        int(input_signature[3]),
    ]
    if reshape1_signature != expected_reshape1_signature:
        return None
    reshape1_shape_vector = _constant_vector(
        model_ir,
        graph_index,
        str(reshape1.inputs[1]),
        6,
        public_inputs,
    )
    if (
        reshape1_shape_vector is None
        or list(reshape1_shape_vector.values) != reshape1_shape
    ):
        return None

    input_producer = graph_index.producers.get(input_name)
    if input_name in graph_index.duplicate_producers:
        return None
    if input_producer is not None:
        if input_name in public_inputs or int(input_producer) >= int(reshape1_index):
            return None
    elif input_name not in public_inputs and input_tensor.data is None:
        return None

    reshape1_users = graph_index.consumer_indices(reshape1_name)
    if len(reshape1_users) != 1:
        return None
    transpose_index = int(reshape1_users[0])
    if transpose_index <= int(reshape1_index):
        return None
    transpose = model_ir.operators[transpose_index]
    if (
        str(transpose.op_type) != "TRANSPOSE"
        or len(transpose.inputs) != 2
        or len(transpose.outputs) != 1
        or str(transpose.inputs[0]) != reshape1_name
    ):
        return None
    permutation = _constant_vector(
        model_ir,
        graph_index,
        str(transpose.inputs[1]),
        6,
        public_inputs,
    )
    if permutation is None or permutation.values != _TARGET_PERM:
        return None

    transpose_name = str(transpose.outputs[0])
    if (
        not transpose_name
        or transpose_name in public_inputs | public_outputs
        or not _producer_is_valid(
            graph_index,
            transpose_name,
            transpose_index,
        )
    ):
        return None
    transpose_tensor = model_ir.tensors.get(transpose_name)
    transpose_contract = _shape_contract(transpose_tensor, 6)
    expected_transpose_shape = [
        n,
        out_h,
        out_w,
        block_size,
        block_size,
        channels,
    ]
    expected_transpose_signature = [
        expected_reshape1_signature[index] for index in _TARGET_PERM
    ]
    if (
        transpose_contract is None
        or transpose_contract[0] != expected_transpose_shape
        or transpose_contract[1] != expected_transpose_signature
    ):
        return None

    transpose_users = graph_index.consumer_indices(transpose_name)
    if len(transpose_users) != 1:
        return None
    reshape2_index = int(transpose_users[0])
    if reshape2_index <= transpose_index:
        return None
    reshape2 = model_ir.operators[reshape2_index]
    if (
        str(reshape2.op_type) != "RESHAPE"
        or len(reshape2.inputs) != 2
        or len(reshape2.outputs) != 1
        or str(reshape2.inputs[0]) != transpose_name
    ):
        return None
    reshape2_name = str(reshape2.outputs[0])
    if (
        not reshape2_name
        or reshape2_name in public_inputs
        or not _producer_is_valid(
            graph_index,
            reshape2_name,
            reshape2_index,
        )
    ):
        return None
    reshape2_tensor = model_ir.tensors.get(reshape2_name)
    reshape2_contract = _shape_contract(reshape2_tensor, 3)
    expected_reshape2_shape = [
        n * out_h * out_w,
        block_size * block_size,
        channels,
    ]
    flattened_dynamic = any(
        int(expected_transpose_signature[index]) < 0 for index in (0, 1, 2)
    )
    expected_reshape2_signature = [
        -1 if flattened_dynamic else int(expected_reshape2_shape[0]),
        int(expected_reshape2_shape[1]),
        -1 if int(input_signature[3]) < 0 else int(channels),
    ]
    if (
        reshape2_contract is None
        or reshape2_contract[0] != expected_reshape2_shape
        or reshape2_contract[1] != expected_reshape2_signature
        or expected_reshape2_signature.count(-1) > 1
    ):
        return None
    reshape2_shape_vector = _constant_vector(
        model_ir,
        graph_index,
        str(reshape2.inputs[1]),
        3,
        public_inputs,
    )
    if (
        reshape2_shape_vector is None
        or list(reshape2_shape_vector.values) != expected_reshape2_shape
    ):
        return None

    tensors = (
        input_tensor,
        reshape1_tensor,
        transpose_tensor,
        reshape2_tensor,
    )
    if len(
        {str(tensor.dtype) for tensor in tensors}
    ) != 1 or not _quantization_contract(tensors):
        return None

    reshape2_options: Optional[Dict[str, Any]] = None
    reshape2_shape_data: Optional[np.ndarray] = None
    reshape2_shape_tensor: Optional[TensorIR] = None
    if expected_reshape2_signature != expected_reshape2_shape:
        shape_name = str(reshape2.inputs[1])
        if (
            not isinstance(reshape2.options, dict)
            or shape_name in public_outputs
            or graph_index.consumer_indices(shape_name) != [reshape2_index]
        ):
            return None
        reshape2_options = dict(reshape2.options)
        reshape2_options["newShape"] = list(expected_reshape2_signature)
        if "onnxRawNewShape" in reshape2_options:
            reshape2_options["onnxRawNewShape"] = list(expected_reshape2_signature)
        reshape2_options["preserveDynamicShape"] = True
        reshape2_shape_tensor = reshape2_shape_vector.tensor
        reshape2_shape_data = np.asarray(
            expected_reshape2_signature,
            dtype=np.asarray(reshape2_shape_vector.tensor.data).dtype,
        )

    space_to_depth_shape = (
        n,
        out_h,
        out_w,
        block_size * block_size * channels,
    )
    space_to_depth_signature = (
        int(input_signature[0]),
        -1 if int(input_signature[1]) < 0 else int(out_h),
        -1 if int(input_signature[2]) < 0 else int(out_w),
        (-1 if int(input_signature[3]) < 0 else int(space_to_depth_shape[3])),
    )
    return _RewritePlan(
        reshape1=reshape1,
        transpose=transpose,
        transpose_tensor=transpose_tensor,
        transpose_shape=space_to_depth_shape,
        transpose_signature=space_to_depth_signature,
        block_size=block_size,
        reshape2=reshape2,
        reshape2_tensor=reshape2_tensor,
        reshape2_signature=tuple(expected_reshape2_signature),
        reshape2_options=reshape2_options,
        reshape2_shape_tensor=reshape2_shape_tensor,
        reshape2_shape_data=reshape2_shape_data,
        input_name=input_name,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _RewritePlan,
) -> bool:
    reshape1_index = graph_index.operator_index(plan.reshape1)
    transpose_index = graph_index.operator_index(plan.transpose)
    reshape2_index = graph_index.operator_index(plan.reshape2)
    if (
        reshape1_index is None
        or transpose_index is None
        or reshape2_index is None
        or not int(reshape1_index) < int(transpose_index) < int(reshape2_index)
    ):
        return False
    graph_index.replace_operator_type(transpose_index, "SPACE_TO_DEPTH")
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.transpose,
        new_inputs=[plan.input_name],
        graph_index=graph_index,
    )
    plan.transpose.options = {"blockSize": int(plan.block_size)}
    plan.transpose_tensor.shape = list(plan.transpose_shape)
    plan.transpose_tensor.shape_signature = list(plan.transpose_signature)
    plan.reshape2_tensor.shape_signature = list(plan.reshape2_signature)
    if (
        plan.reshape2_options is not None
        and plan.reshape2_shape_tensor is not None
        and plan.reshape2_shape_data is not None
    ):
        plan.reshape2.options = dict(plan.reshape2_options)
        plan.reshape2_shape_tensor.data = np.asarray(plan.reshape2_shape_data)
    graph_index.remove_operator(reshape1_index)
    return True


def _optimize_window_partition_reshape_transpose_to_space_to_depth_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    required = {"RESHAPE": 2, "TRANSPOSE": 1}
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required and required[op_type] > 0:
            required[op_type] -= 1
    if any(count > 0 for count in required.values()):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {_STATS_KEY: 0}
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    plans = [
        plan
        for reshape_index in active_index.operator_indices("RESHAPE")
        if (
            plan := _candidate_plan(
                model_ir,
                active_index,
                int(reshape_index),
            )
        )
        is not None
    ]
    rewritten = sum(bool(_apply_plan(model_ir, active_index, plan)) for plan in plans)
    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    name = str(base)
    serial = 1
    while name in model_ir.tensors:
        name = f"{base}_{serial}"
        serial += 1
    return name


def _reverse_candidate_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    reshape1_index: int,
) -> Optional[_ReverseRewritePlan]:
    reshape1 = model_ir.operators[int(reshape1_index)]
    if len(reshape1.inputs) != 2 or len(reshape1.outputs) != 1:
        return None
    input_name = str(reshape1.inputs[0])
    reshape1_name = str(reshape1.outputs[0])
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    if (
        not input_name
        or not reshape1_name
        or reshape1_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, reshape1_name, reshape1_index)
    ):
        return None

    input_tensor = model_ir.tensors.get(input_name)
    reshape1_tensor = model_ir.tensors.get(reshape1_name)
    input_contract = _shape_contract(input_tensor, 3)
    reshape1_contract = _shape_contract(reshape1_tensor, 6)
    if input_contract is None or reshape1_contract is None:
        return None
    input_shape, input_signature = input_contract
    reshape1_shape, reshape1_signature = reshape1_contract
    nw, window_size_squared, channels = input_shape
    n, out_h, out_w, block_h, block_w, reshape_channels = reshape1_shape
    if (
        reshape_channels != channels
        or block_h <= 1
        or block_h != block_w
        or window_size_squared != block_h * block_h
        or nw != n * out_h * out_w
    ):
        return None
    block_size = int(block_h)
    flattened_dynamic = any(int(value) < 0 for value in reshape1_signature[:3])
    expected_input_signature = [
        -1 if flattened_dynamic else int(nw),
        int(window_size_squared),
        int(reshape1_signature[5]),
    ]
    if (
        reshape1_signature[3:5] != [block_size, block_size]
        or input_signature != expected_input_signature
    ):
        return None

    reshape1_shape_name = str(reshape1.inputs[1])
    reshape1_shape_vector = _constant_vector(
        model_ir,
        graph_index,
        reshape1_shape_name,
        6,
        public_inputs,
    )
    if (
        reshape1_shape_name in public_outputs
        or reshape1_shape_vector is None
        or list(reshape1_shape_vector.values) != reshape1_shape
    ):
        return None

    input_producer = graph_index.producers.get(input_name)
    if input_name in graph_index.duplicate_producers:
        return None
    if input_producer is not None:
        if input_name in public_inputs or int(input_producer) >= int(reshape1_index):
            return None
    elif input_name not in public_inputs and input_tensor.data is None:
        return None

    reshape1_users = graph_index.consumer_indices(reshape1_name)
    if len(reshape1_users) != 1:
        return None
    transpose_index = int(reshape1_users[0])
    if transpose_index <= int(reshape1_index):
        return None
    transpose = model_ir.operators[transpose_index]
    if (
        str(transpose.op_type) != "TRANSPOSE"
        or len(transpose.inputs) != 2
        or len(transpose.outputs) != 1
        or str(transpose.inputs[0]) != reshape1_name
    ):
        return None
    permutation = _constant_vector(
        model_ir,
        graph_index,
        str(transpose.inputs[1]),
        6,
        public_inputs,
    )
    if permutation is None or permutation.values != _TARGET_PERM:
        return None

    transpose_name = str(transpose.outputs[0])
    if (
        not transpose_name
        or transpose_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, transpose_name, transpose_index)
    ):
        return None
    transpose_tensor = model_ir.tensors.get(transpose_name)
    transpose_contract = _shape_contract(transpose_tensor, 6)
    expected_transpose_shape = [
        n,
        out_h,
        block_size,
        out_w,
        block_size,
        channels,
    ]
    expected_transpose_signature = [
        reshape1_signature[index] for index in _TARGET_PERM
    ]
    if (
        transpose_contract is None
        or transpose_contract[0] != expected_transpose_shape
        or transpose_contract[1] != expected_transpose_signature
    ):
        return None

    transpose_users = graph_index.consumer_indices(transpose_name)
    if len(transpose_users) != 1:
        return None
    reshape2_index = int(transpose_users[0])
    if reshape2_index <= transpose_index:
        return None
    reshape2 = model_ir.operators[reshape2_index]
    if (
        str(reshape2.op_type) != "RESHAPE"
        or len(reshape2.inputs) != 2
        or len(reshape2.outputs) != 1
        or str(reshape2.inputs[0]) != transpose_name
    ):
        return None
    output_name = str(reshape2.outputs[0])
    if (
        not output_name
        or output_name in public_inputs
        or not _producer_is_valid(graph_index, output_name, reshape2_index)
    ):
        return None
    output_tensor = model_ir.tensors.get(output_name)
    output_contract = _shape_contract(output_tensor, 4)
    expected_output_shape = [
        n,
        out_h * block_size,
        out_w * block_size,
        channels,
    ]
    expected_output_signature = [
        int(reshape1_signature[0]),
        -1 if int(reshape1_signature[1]) < 0 else int(expected_output_shape[1]),
        -1 if int(reshape1_signature[2]) < 0 else int(expected_output_shape[2]),
        int(reshape1_signature[5]),
    ]
    if (
        output_contract is None
        or output_contract[0] != expected_output_shape
        or output_contract[1] != expected_output_signature
    ):
        return None
    reshape2_shape_vector = _constant_vector(
        model_ir,
        graph_index,
        str(reshape2.inputs[1]),
        4,
        public_inputs,
    )
    if (
        reshape2_shape_vector is None
        or list(reshape2_shape_vector.values) != expected_output_shape
    ):
        return None

    tensors = (input_tensor, reshape1_tensor, transpose_tensor, output_tensor)
    if len({str(tensor.dtype) for tensor in tensors}) != 1 or not _quantization_contract(
        tensors
    ):
        return None

    new_reshape1_shape = [
        n,
        out_h,
        out_w,
        block_size * block_size * channels,
    ]
    new_reshape1_signature = [
        int(reshape1_signature[0]),
        int(reshape1_signature[1]),
        int(reshape1_signature[2]),
        -1 if int(reshape1_signature[5]) < 0 else int(new_reshape1_shape[3]),
    ]
    if new_reshape1_signature.count(-1) > 1:
        return None

    reshape1_options: Optional[Dict[str, Any]] = None
    if isinstance(reshape1.options, dict):
        reshape1_options = dict(reshape1.options)
        for key in ("newShape", "onnxRawNewShape"):
            if isinstance(reshape1_options.get(key), list):
                reshape1_options[key] = list(new_reshape1_signature)
        if new_reshape1_signature != new_reshape1_shape:
            reshape1_options["newShape"] = list(new_reshape1_signature)
            if "onnxRawNewShape" in reshape1_options:
                reshape1_options["onnxRawNewShape"] = list(new_reshape1_signature)
            reshape1_options["preserveDynamicShape"] = True
    elif new_reshape1_signature != new_reshape1_shape:
        return None

    reshape1_shape_data = np.asarray(
        new_reshape1_signature,
        dtype=np.asarray(reshape1_shape_vector.tensor.data).dtype,
    )
    reshape1_shape_clone_name: Optional[str] = None
    reshape1_shape_clone: Optional[TensorIR] = None
    if set(graph_index.consumer_indices(reshape1_shape_name)) != {
        int(reshape1_index)
    }:
        reshape1_shape_clone_name = _unique_tensor_name(
            model_ir,
            f"{reshape1_shape_name}_d2s_shape",
        )
        try:
            cloned_quantization = _clone_quantization(
                reshape1_shape_vector.tensor.quantization
            )
        except Exception:
            return None
        reshape1_shape_clone = TensorIR(
            name=reshape1_shape_clone_name,
            dtype=str(reshape1_shape_vector.tensor.dtype),
            shape=[len(new_reshape1_signature)],
            shape_signature=[len(new_reshape1_signature)],
            data=np.asarray(reshape1_shape_data),
            is_variable=False,
            quantization=cloned_quantization,
        )

    return _ReverseRewritePlan(
        reshape1=reshape1,
        reshape1_tensor=reshape1_tensor,
        reshape1_shape=tuple(new_reshape1_shape),
        reshape1_signature=tuple(new_reshape1_signature),
        reshape1_options=reshape1_options,
        reshape1_shape_tensor=reshape1_shape_vector.tensor,
        reshape1_shape_data=reshape1_shape_data,
        reshape1_shape_clone_name=reshape1_shape_clone_name,
        reshape1_shape_clone=reshape1_shape_clone,
        transpose=transpose,
        block_size=block_size,
        reshape2=reshape2,
        output_name=output_name,
    )


def _apply_reverse_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _ReverseRewritePlan,
) -> bool:
    reshape1_index = graph_index.operator_index(plan.reshape1)
    transpose_index = graph_index.operator_index(plan.transpose)
    reshape2_index = graph_index.operator_index(plan.reshape2)
    if (
        reshape1_index is None
        or transpose_index is None
        or reshape2_index is None
        or not int(reshape1_index) < int(transpose_index) < int(reshape2_index)
    ):
        return False

    if plan.reshape1_shape_clone_name is not None:
        if (
            plan.reshape1_shape_clone is None
            or plan.reshape1_shape_clone_name in model_ir.tensors
        ):
            return False
        model_ir.tensors[plan.reshape1_shape_clone_name] = plan.reshape1_shape_clone
        _replace_operator_input_at(
            model_ir=model_ir,
            op=plan.reshape1,
            input_index=1,
            new_input_name=plan.reshape1_shape_clone_name,
            graph_index=graph_index,
        )
    else:
        plan.reshape1_shape_tensor.data = np.asarray(plan.reshape1_shape_data)
        plan.reshape1_shape_tensor.shape = [len(plan.reshape1_signature)]
        plan.reshape1_shape_tensor.shape_signature = [len(plan.reshape1_signature)]

    if plan.reshape1_options is not None:
        plan.reshape1.options = dict(plan.reshape1_options)
    plan.reshape1_tensor.shape = list(plan.reshape1_shape)
    plan.reshape1_tensor.shape_signature = list(plan.reshape1_signature)

    graph_index.replace_operator_type(transpose_index, "DEPTH_TO_SPACE")
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.transpose,
        new_inputs=[str(plan.reshape1.outputs[0])],
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.transpose,
        new_outputs=[plan.output_name],
        graph_index=graph_index,
    )
    plan.transpose.options = {"blockSize": int(plan.block_size)}
    graph_index.remove_operator(reshape2_index)
    return True


def _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    required = {"RESHAPE": 2, "TRANSPOSE": 1}
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required and required[op_type] > 0:
            required[op_type] -= 1
    if any(count > 0 for count in required.values()):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {_REVERSE_STATS_KEY: 0}

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    reshape_candidates = [
        model_ir.operators[index]
        for index in active_index.operator_indices("RESHAPE")
    ]
    rewritten = 0
    for reshape in reshape_candidates:
        reshape_index = active_index.operator_index(reshape)
        if reshape_index is None:
            continue
        plan = _reverse_candidate_plan(model_ir, active_index, reshape_index)
        if plan is not None and _apply_reverse_plan(model_ir, active_index, plan):
            rewritten += 1

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {_REVERSE_STATS_KEY: int(rewritten)}
