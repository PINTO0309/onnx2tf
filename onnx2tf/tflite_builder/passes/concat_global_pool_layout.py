from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _read_const_ints_from_tensor
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


@dataclass(frozen=True)
class _RepairPlan:
    concat: OperatorIR
    concat_options: Dict[str, object]
    reshape: OperatorIR
    reshape_options: Dict[str, object]
    tensor_shapes: Tuple[Tuple[TensorIR, Tuple[int, ...]], ...]
    shape_tensor: TensorIR
    shape_data: np.ndarray


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
    op_type: str,
) -> Optional[tuple[int, OperatorIR]]:
    name = str(tensor_name)
    producer_index = index.producers.get(name)
    if producer_index is None or name in index.duplicate_producers:
        return None
    operator = model_ir.operators[int(producer_index)]
    if (
        str(operator.op_type) != str(op_type)
        or len(operator.outputs) != 1
        or str(operator.outputs[0]) != name
    ):
        return None
    return int(producer_index), operator


def _normalized_global_axes(tensor: Optional[TensorIR]) -> Optional[set[int]]:
    axes = _read_const_ints_from_tensor(tensor)
    if axes is None or len(axes) != 2:
        return None
    normalized = {int(axis) + 4 if int(axis) < 0 else int(axis) for axis in axes}
    return normalized if normalized == {2, 3} else None


def _tensor_depends_on(
    model_ir: ModelIR,
    index: ModelIRGraphIndex,
    *,
    tensor_name: str,
    ancestor_name: str,
) -> bool:
    """Return whether one tensor is downstream of another in the indexed graph."""

    target = str(ancestor_name)
    pending = [str(tensor_name)]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in visited:
            continue
        visited.add(current)
        producer_index = index.producers.get(current)
        if producer_index is None or current in index.duplicate_producers:
            continue
        producer = model_ir.operators[int(producer_index)]
        pending.extend(str(name) for name in producer.inputs)
    return False


def _is_guarded_concat_fanout(
    model_ir: ModelIR,
    index: ModelIRGraphIndex,
    *,
    concat_name: str,
    pool_index: int,
    gate_name: str,
) -> bool:
    """Accept only self-gating MUL fanout in addition to the global pool."""

    consumers = index.consumer_indices(str(concat_name))
    if int(pool_index) not in consumers:
        return False
    for consumer_index in consumers:
        if int(consumer_index) == int(pool_index):
            continue
        consumer = model_ir.operators[int(consumer_index)]
        if str(consumer.op_type) != "MUL" or str(concat_name) not in {
            str(name) for name in consumer.inputs
        }:
            return False
        gate_inputs = [
            str(name) for name in consumer.inputs if str(name) != str(concat_name)
        ]
        if not gate_inputs or not any(
            _tensor_depends_on(
                model_ir,
                index,
                tensor_name=gate_input,
                ancestor_name=gate_name,
            )
            for gate_input in gate_inputs
        ):
            return False
    return True


def _candidate_plan(
    model_ir: ModelIR,
    index: ModelIRGraphIndex,
    conv_index: int,
) -> Optional[_RepairPlan]:
    conv = model_ir.operators[int(conv_index)]
    if len(conv.inputs) < 2 or len(conv.outputs) != 1:
        return None
    reshape_match = _producer(model_ir, index, str(conv.inputs[0]), "RESHAPE")
    if reshape_match is None:
        return None
    reshape_index, reshape = reshape_match
    if len(reshape.inputs) != 2 or not isinstance(reshape.options, dict):
        return None
    pool_match = _producer(model_ir, index, str(reshape.inputs[0]), "MEAN")
    if pool_match is None:
        return None
    pool_index, pool = pool_match
    if (
        len(pool.inputs) != 2
        or not isinstance(pool.options, dict)
        or not bool(pool.options.get("keepDims", True))
        or _normalized_global_axes(model_ir.tensors.get(str(pool.inputs[1]))) is None
        or str(pool.inputs[1]) in index.producers
        or str(pool.inputs[1]) in index.duplicate_producers
    ):
        return None
    concat_match = _producer(
        model_ir,
        index,
        str(pool.inputs[0]),
        "CONCATENATION",
    )
    if concat_match is None:
        return None
    concat_index, concat = concat_match
    if len(concat.inputs) < 2 or not isinstance(concat.options, dict):
        return None
    try:
        if int(concat.options.get("axis", 1)) == 1:
            return None
    except (TypeError, ValueError):
        return None
    if not int(concat_index) < int(pool_index) < int(reshape_index) < int(conv_index):
        return None

    concat_name = str(concat.outputs[0])
    pool_name = str(pool.outputs[0])
    reshape_name = str(reshape.outputs[0])
    if (
        not _is_guarded_concat_fanout(
            model_ir,
            index,
            concat_name=concat_name,
            pool_index=int(pool_index),
            gate_name=str(conv.outputs[0]),
        )
        or index.consumer_indices(pool_name) != [int(reshape_index)]
        or index.consumer_indices(reshape_name) != [int(conv_index)]
        or any(
            name in {str(value) for value in model_ir.inputs + model_ir.outputs}
            for name in (concat_name, pool_name, reshape_name)
        )
    ):
        return None

    concat_tensor = model_ir.tensors.get(concat_name)
    pool_tensor = model_ir.tensors.get(pool_name)
    reshape_tensor = model_ir.tensors.get(reshape_name)
    input_shapes = [_shape(model_ir.tensors.get(str(name))) for name in concat.inputs]
    filter_name = str(conv.inputs[1])
    filter_tensor = model_ir.tensors.get(filter_name)
    filter_shape = _shape(filter_tensor)
    if (
        concat_tensor is None
        or pool_tensor is None
        or reshape_tensor is None
        or any(shape is None or len(shape) != 4 for shape in input_shapes)
        or filter_shape is None
        or len(filter_shape) != 4
        or filter_tensor is None
        or not isinstance(filter_tensor.data, np.ndarray)
        or list(np.asarray(filter_tensor.data).shape) != filter_shape
        or filter_name in index.producers
        or filter_name in index.duplicate_producers
    ):
        return None
    concrete_shapes = [shape for shape in input_shapes if shape is not None]
    reference = concrete_shapes[0]
    if any(value <= 0 for shape in concrete_shapes for value in shape) or any(
        any(shape[axis] != reference[axis] for axis in (0, 2, 3))
        for shape in concrete_shapes[1:]
    ):
        return None
    channels = sum(shape[1] for shape in concrete_shapes)
    reshape_target = (reference[0], 1, 1, channels)
    if (
        channels <= 0
        or filter_shape[3] != channels
        or _shape(reshape_tensor) == list(reshape_target)
    ):
        return None

    shape_name = str(reshape.inputs[1])
    shape_tensor = model_ir.tensors.get(shape_name)
    public = {str(value) for value in model_ir.inputs + model_ir.outputs}
    if (
        shape_tensor is None
        or shape_tensor.data is None
        or shape_name in public
        or shape_name in index.producers
        or shape_name in index.duplicate_producers
        or index.consumer_indices(shape_name) != [int(reshape_index)]
    ):
        return None
    try:
        current_shape_data = np.asarray(shape_tensor.data)
        if current_shape_data.size != 4 or not np.issubdtype(
            current_shape_data.dtype, np.integer
        ):
            return None
        shape_data = np.asarray(reshape_target, dtype=current_shape_data.dtype)
    except Exception:
        return None
    concat_shape = tuple(reference[:1] + [channels] + reference[2:])
    pool_shape = (reference[0], channels, 1, 1)
    concat_options = dict(concat.options)
    concat_options["axis"] = 1
    reshape_options = dict(reshape.options)
    reshape_options["newShape"] = list(reshape_target)
    return _RepairPlan(
        concat=concat,
        concat_options=concat_options,
        reshape=reshape,
        reshape_options=reshape_options,
        tensor_shapes=(
            (concat_tensor, concat_shape),
            (pool_tensor, pool_shape),
            (reshape_tensor, reshape_target),
        ),
        shape_tensor=shape_tensor,
        shape_data=shape_data,
    )


def _apply_plan(plan: _RepairPlan) -> None:
    plan.concat.options = dict(plan.concat_options)
    plan.reshape.options = dict(plan.reshape_options)
    for tensor, shape in plan.tensor_shapes:
        tensor.shape = list(shape)
        tensor.shape_signature = list(shape)
    plan.shape_tensor.data = np.asarray(plan.shape_data)
    plan.shape_tensor.shape = [4]
    plan.shape_tensor.shape_signature = [4]


def _repair_nchw_concat_global_pool_conv_axes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    stats_key = "repaired_nchw_concat_global_pool_conv_axes"
    required = {"CONV_2D", "RESHAPE", "MEAN", "CONCATENATION"}
    for operator in model_ir.operators:
        required.discard(str(operator.op_type))
        if not required:
            break
    if required:
        return {stats_key: 0}
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    repaired = 0
    for conv_index in active_index.operator_indices("CONV_2D"):
        plan = _candidate_plan(model_ir, active_index, int(conv_index))
        if plan is not None:
            _apply_plan(plan)
            repaired += 1
    if repaired and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {stats_key: repaired}
