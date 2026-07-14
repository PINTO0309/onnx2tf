from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _read_transpose_perm
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


@dataclass(frozen=True)
class _RepairPlan:
    concat: OperatorIR
    concat_options: Dict[str, object]
    tensor_shapes: Tuple[Tuple[TensorIR, Tuple[int, ...]], ...]


def _shape(tensor: Optional[TensorIR]) -> Optional[list[int]]:
    if tensor is None:
        return None
    try:
        shape = [int(value) for value in tensor.shape]
        if tensor.shape_signature is not None:
            [int(value) for value in tensor.shape_signature]
        return shape
    except (TypeError, ValueError):
        return None


def _producer(
    model_ir: ModelIR,
    index: ModelIRGraphIndex,
    tensor_name: str,
) -> Optional[tuple[int, OperatorIR]]:
    name = str(tensor_name)
    producer_index = index.producers.get(name)
    if producer_index is None or name in index.duplicate_producers:
        return None
    operator = model_ir.operators[int(producer_index)]
    if len(operator.outputs) != 1 or str(operator.outputs[0]) != name:
        return None
    return int(producer_index), operator


def _candidate_plan(
    model_ir: ModelIR,
    index: ModelIRGraphIndex,
    conv_index: int,
) -> Optional[_RepairPlan]:
    conv = model_ir.operators[int(conv_index)]
    conv_type = str(conv.op_type)
    minimum_inputs = 3 if conv_type == "TRANSPOSE_CONV" else 2
    if len(conv.inputs) < minimum_inputs or len(conv.outputs) != 1:
        return None
    data_index = 2 if conv_type == "TRANSPOSE_CONV" else 0
    filter_name = str(conv.inputs[1])

    post_indices: list[int] = []
    post_names: list[str] = []
    current_name = str(conv.inputs[data_index])
    transpose_index: Optional[int] = None
    transpose: Optional[OperatorIR] = None
    while True:
        match = _producer(model_ir, index, current_name)
        if match is None:
            return None
        candidate_index, candidate = match
        if (
            str(candidate.op_type) == "TRANSPOSE"
            and len(candidate.inputs) >= 2
            and _read_transpose_perm(model_ir, candidate) == [0, 2, 3, 1]
        ):
            perm_name = str(candidate.inputs[1])
            if perm_name in index.producers or perm_name in index.duplicate_producers:
                return None
            transpose_index, transpose = candidate_index, candidate
            break
        if (
            str(candidate.op_type) not in {"PAD", "CAST", "SUB"}
            or len(candidate.inputs) < 1
            or str(candidate.inputs[0]) == current_name
        ):
            return None
        post_indices.append(candidate_index)
        post_names.append(current_name)
        current_name = str(candidate.inputs[0])
    if transpose is None or transpose_index is None:
        return None

    pre_indices: list[int] = []
    pre_names: list[str] = []
    concat_name = str(transpose.inputs[0])
    while True:
        match = _producer(model_ir, index, concat_name)
        if match is None:
            return None
        candidate_index, candidate = match
        if str(candidate.op_type) == "CONCATENATION":
            concat_index, concat = candidate_index, candidate
            break
        if (
            str(candidate.op_type)
            not in {"RELU", "RELU6", "QUANTIZE", "DEQUANTIZE", "CAST"}
            or len(candidate.inputs) != 1
        ):
            return None
        pre_indices.append(candidate_index)
        pre_names.append(concat_name)
        concat_name = str(candidate.inputs[0])
    if len(concat.inputs) < 2 or not isinstance(concat.options, dict):
        return None
    try:
        if int(concat.options.get("axis", 1)) == 1:
            return None
    except (TypeError, ValueError):
        return None

    forward_pre = list(reversed(pre_indices))
    forward_post = list(reversed(post_indices))
    ordered = (
        [concat_index] + forward_pre + [transpose_index] + forward_post + [conv_index]
    )
    if ordered != sorted(ordered) or len(ordered) != len(set(ordered)):
        return None
    chain_names = (
        [concat_name]
        + list(reversed(pre_names))
        + [str(transpose.outputs[0])]
        + list(reversed(post_names))
    )
    for position, tensor_name in enumerate(chain_names):
        expected_consumer = ordered[position + 1]
        if index.consumer_indices(tensor_name) != [
            int(expected_consumer)
        ] or tensor_name in {
            str(value) for value in model_ir.inputs + model_ir.outputs
        }:
            return None

    input_shapes = [_shape(model_ir.tensors.get(str(name))) for name in concat.inputs]
    filter_tensor = model_ir.tensors.get(filter_name)
    filter_shape = _shape(filter_tensor)
    transpose_name = str(transpose.outputs[0])
    transpose_tensor = model_ir.tensors.get(transpose_name)
    conv_output = model_ir.tensors.get(str(conv.outputs[0]))
    if (
        any(shape is None or len(shape) != 4 for shape in input_shapes)
        or filter_tensor is None
        or filter_shape is None
        or len(filter_shape) != 4
        or not isinstance(filter_tensor.data, np.ndarray)
        or list(filter_tensor.data.shape) != filter_shape
        or filter_name in index.producers
        or filter_name in index.duplicate_producers
        or transpose_tensor is None
        or conv_output is None
    ):
        return None
    concrete = [shape for shape in input_shapes if shape is not None]
    reference = concrete[0]
    if any(value <= 0 for shape in concrete for value in shape) or any(
        any(shape[axis] != reference[axis] for axis in (0, 2, 3))
        for shape in concrete[1:]
    ):
        return None
    channels = sum(shape[1] for shape in concrete)
    transpose_shape = _shape(transpose_tensor)
    if (
        channels <= 0
        or filter_shape[3] != channels
        or transpose_shape is None
        or len(transpose_shape) != 4
        or transpose_shape[3] == channels
    ):
        return None

    concat_shape = tuple(reference[:1] + [channels] + reference[2:])
    nhwc_shape = (concat_shape[0], concat_shape[2], concat_shape[3], concat_shape[1])
    tensors: list[tuple[TensorIR, Tuple[int, ...]]] = []
    for name in [concat_name] + pre_names:
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensors.append((tensor, concat_shape))
    tensors.append((transpose_tensor, nhwc_shape))
    if conv_type == "CONV_2D" and not post_indices:
        tensors.append(
            (
                conv_output,
                (nhwc_shape[0], nhwc_shape[1], nhwc_shape[2], filter_shape[0]),
            )
        )
    options = dict(concat.options)
    options["axis"] = 1
    return _RepairPlan(concat, options, tuple(tensors))


def _apply_plan(plan: _RepairPlan) -> None:
    plan.concat.options = dict(plan.concat_options)
    for tensor, shape in plan.tensor_shapes:
        tensor.shape = list(shape)
        tensor.shape_signature = list(shape)


def _repair_nchw_concat_transpose_conv_axes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    stats_key = "repaired_nchw_concat_transpose_conv_axes"
    if not any(
        str(op.op_type) == "CONCATENATION" for op in model_ir.operators
    ) or not any(
        str(op.op_type) in {"CONV_2D", "TRANSPOSE_CONV"} for op in model_ir.operators
    ):
        return {stats_key: 0}
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    repaired = 0
    for conv_index in active_index.operator_indices_for_types(
        {"CONV_2D", "TRANSPOSE_CONV"}
    ):
        plan = _candidate_plan(model_ir, active_index, int(conv_index))
        if plan is not None:
            _apply_plan(plan)
            repaired += 1
    if repaired and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {stats_key: repaired}
