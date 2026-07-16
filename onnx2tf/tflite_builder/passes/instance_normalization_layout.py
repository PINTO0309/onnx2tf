from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _read_const_ints_from_tensor,
    _read_transpose_perm,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)


@dataclass(frozen=True)
class _Mutation:
    tensor: TensorIR
    data: Optional[np.ndarray] = None
    shape: Optional[Tuple[int, ...]] = None
    signature: Optional[Tuple[int, ...]] = None

    @property
    def changed(self) -> bool:
        return (
            self.data is not None
            or self.shape is not None
            or self.signature is not None
        )


def _shape(values: object) -> Optional[list[int]]:
    if values is None:
        return None
    try:
        return [int(value) for value in list(values)]
    except (TypeError, ValueError):
        return None


def _shape_plan(
    tensor: Optional[TensorIR], target: Sequence[int]
) -> Optional[_Mutation]:
    if tensor is None:
        return None
    current = _shape(tensor.shape)
    signature = (
        _shape(tensor.shape_signature) if tensor.shape_signature is not None else None
    )
    if current is None or (tensor.shape_signature is not None and signature is None):
        return None
    normalized = tuple(int(value) for value in target)
    return _Mutation(
        tensor=tensor,
        shape=normalized if tuple(current) != normalized else None,
        signature=(
            normalized
            if signature is not None and tuple(signature) != normalized
            else None
        ),
    )


def _axes_plan(tensor: Optional[TensorIR], axes: Sequence[int]) -> Optional[_Mutation]:
    current = _read_const_ints_from_tensor(tensor)
    if tensor is None or current is None:
        return None
    desired = [int(value) for value in axes]
    if current == desired:
        return _Mutation(tensor)
    try:
        data = np.asarray(tensor.data)
        if not np.issubdtype(data.dtype, np.integer):
            return None
        replacement = np.asarray(desired, dtype=data.dtype)
    except (TypeError, ValueError):
        return None
    metadata = (len(desired),)
    return _Mutation(tensor, replacement, metadata, metadata)


def _constant_plan(
    tensor: Optional[TensorIR],
    target: Sequence[int],
) -> Optional[_Mutation]:
    if tensor is None or tensor.data is None:
        return None
    normalized = tuple(int(value) for value in target)
    if not normalized or any(value <= 0 for value in normalized):
        return None
    try:
        data = np.asarray(tensor.data)
        if data.size <= 1 or int(np.prod(normalized, dtype=np.int64)) != int(data.size):
            return None
        replacement = np.asarray(data.reshape(normalized), dtype=data.dtype)
    except (TypeError, ValueError, OverflowError):
        return None
    metadata = _shape_plan(tensor, normalized)
    if metadata is None:
        return None
    return _Mutation(
        tensor,
        replacement if list(data.shape) != list(normalized) else None,
        metadata.shape,
        metadata.signature,
    )


def _apply(plan: _Mutation) -> None:
    if plan.data is not None:
        plan.tensor.data = np.asarray(plan.data)
    if plan.shape is not None:
        plan.tensor.shape = list(plan.shape)
    if plan.signature is not None:
        plan.tensor.shape_signature = list(plan.signature)


def _op(model_ir: ModelIR, index: Optional[int]) -> Optional[OperatorIR]:
    if index is None or not 0 <= int(index) < len(model_ir.operators):
        return None
    return model_ir.operators[int(index)]


def _sole_consumer(index: ModelIRGraphIndex, name: str) -> Optional[int]:
    users = index.consumer_indices(name)
    return int(users[0]) if len(users) == 1 else None


def _singleton_constant(
    model_ir: ModelIR,
    index: ModelIRGraphIndex,
    name: str,
    expected: Optional[float] = None,
) -> bool:
    tensor = model_ir.tensors.get(name)
    if tensor is None or tensor.data is None or name in index.producers:
        return False
    try:
        data = np.asarray(tensor.data, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return False
    return bool(
        data.size == 1
        and np.isfinite(data[0])
        and (expected is None or float(data[0]) == float(expected))
    )


def _constant_change_is_private(
    model_ir: ModelIR,
    index: ModelIRGraphIndex,
    name: str,
    expected_users: Sequence[int],
    plan: _Mutation,
) -> bool:
    if not plan.changed:
        return True
    public = {str(value) for value in model_ir.inputs + model_ir.outputs}
    return bool(
        name not in public
        and name not in index.producers
        and name not in index.duplicate_producers
        and index.consumer_indices(name) == [int(value) for value in expected_users]
    )


def _candidate_plans(
    model_ir: ModelIR,
    index: ModelIRGraphIndex,
    mean1_index: int,
) -> Optional[list[_Mutation]]:
    mean1 = model_ir.operators[int(mean1_index)]
    if (
        str(getattr(mean1, "onnx_op_type", "")) != "InstanceNormalization"
        or len(mean1.inputs) != 2
        or len(mean1.outputs) != 1
        or not bool(mean1.options.get("keepDims", True))
    ):
        return None
    x_name, axes1_name = (str(value) for value in mean1.inputs)
    mean_name = str(mean1.outputs[0])
    x_tensor = model_ir.tensors.get(x_name)
    x_shape = _shape(x_tensor.shape if x_tensor is not None else None)
    if x_tensor is None or x_shape is None or len(x_shape) not in {3, 4, 5}:
        return None
    layout = normalize_logical_layout(x_tensor.logical_layout)
    if is_channel_first_logical_layout(layout):
        channel_axis = 1
    elif is_channel_last_logical_layout(layout):
        channel_axis = len(x_shape) - 1
    else:
        return None
    channel_size = int(x_shape[channel_axis])
    if (
        channel_size <= 0
        or _read_const_ints_from_tensor(model_ir.tensors.get(axes1_name)) is None
    ):
        return None

    sub_matches = [
        user
        for user in sorted(set(index.consumer_indices(x_name)))
        if int(user) > int(mean1_index)
        and str(model_ir.operators[int(user)].op_type) == "SUB"
        and model_ir.operators[int(user)].inputs == [x_name, mean_name]
        and len(model_ir.operators[int(user)].outputs) == 1
    ]
    if len(sub_matches) != 1:
        return None
    sub_index = int(sub_matches[0])
    centered = str(model_ir.operators[sub_index].outputs[0])
    centered_users = sorted(set(index.consumer_indices(centered)))
    square_matches = [
        user
        for user in centered_users
        if str(model_ir.operators[user].op_type) == "MUL"
        and model_ir.operators[user].inputs == [centered, centered]
        and len(model_ir.operators[user].outputs) == 1
    ]
    norm_matches = [
        user
        for user in centered_users
        if str(model_ir.operators[user].op_type) == "MUL"
        and len(model_ir.operators[user].inputs) == 2
        and len(model_ir.operators[user].outputs) == 1
        and centered in model_ir.operators[user].inputs
        and model_ir.operators[user].inputs != [centered, centered]
    ]
    if len(square_matches) != 1 or len(norm_matches) != 1:
        return None
    square_index, norm_index = int(square_matches[0]), int(norm_matches[0])
    if centered_users != sorted({square_index, norm_index}):
        return None
    squared = str(model_ir.operators[square_index].outputs[0])

    mean2_index = _sole_consumer(index, squared)
    mean2 = _op(model_ir, mean2_index)
    if (
        mean2 is None
        or str(mean2.op_type) != "MEAN"
        or len(mean2.inputs) != 2
        or str(mean2.inputs[0]) != squared
        or len(mean2.outputs) != 1
        or not bool(mean2.options.get("keepDims", True))
    ):
        return None
    axes2_name = str(mean2.inputs[1])
    if _read_const_ints_from_tensor(model_ir.tensors.get(axes2_name)) is None:
        return None
    variance = str(mean2.outputs[0])
    add_index = _sole_consumer(index, variance)
    add = _op(model_ir, add_index)
    if (
        add is None
        or str(add.op_type) != "ADD"
        or len(add.inputs) != 2
        or len(add.outputs) != 1
        or variance not in add.inputs
    ):
        return None
    epsilon = str(add.inputs[0] if str(add.inputs[1]) == variance else add.inputs[1])
    if not _singleton_constant(model_ir, index, epsilon):
        return None
    variance_epsilon = str(add.outputs[0])
    sqrt_index = _sole_consumer(index, variance_epsilon)
    sqrt = _op(model_ir, sqrt_index)
    if (
        sqrt is None
        or str(sqrt.op_type) != "SQRT"
        or sqrt.inputs != [variance_epsilon]
        or len(sqrt.outputs) != 1
    ):
        return None
    std = str(sqrt.outputs[0])
    div_index = _sole_consumer(index, std)
    div = _op(model_ir, div_index)
    if (
        div is None
        or str(div.op_type) != "DIV"
        or len(div.inputs) != 2
        or len(div.outputs) != 1
        or str(div.inputs[1]) != std
    ):
        return None
    one = str(div.inputs[0])
    if not _singleton_constant(model_ir, index, one, 1.0):
        return None
    inverse_std = str(div.outputs[0])
    norm = model_ir.operators[norm_index]
    if set(str(value) for value in norm.inputs) != {centered, inverse_std}:
        return None
    normalized = str(norm.outputs[0])
    scale_index = _sole_consumer(index, normalized)
    scale = _op(model_ir, scale_index)
    if (
        scale is None
        or str(scale.op_type) != "MUL"
        or len(scale.inputs) != 2
        or len(scale.outputs) != 1
        or normalized not in scale.inputs
    ):
        return None
    scale_name = str(
        scale.inputs[0] if str(scale.inputs[1]) == normalized else scale.inputs[1]
    )
    scaled = str(scale.outputs[0])

    next_index = _sole_consumer(index, scaled)
    next_op = _op(model_ir, next_index)
    post_index: Optional[int] = None
    bias_data = scaled
    bias_shape = [1] * len(x_shape)
    bias_shape[channel_axis] = channel_size
    if next_op is None:
        return None
    if str(next_op.op_type) != "ADD":
        post_index, post = int(next_index), next_op
        if (
            str(post.op_type) != "TRANSPOSE"
            or len(post.inputs) < 2
            or str(post.inputs[0]) != scaled
            or len(post.outputs) != 1
        ):
            return None
        perm_name = str(post.inputs[1])
        permutation = _read_transpose_perm(model_ir, post)
        if (
            permutation is None
            or sorted(permutation) != list(range(len(x_shape)))
            or perm_name in index.producers
        ):
            return None
        bias_data = str(post.outputs[0])
        next_index = _sole_consumer(index, bias_data)
        next_op = _op(model_ir, next_index)
        inverse = [0] * len(x_shape)
        for new_axis, old_axis in enumerate(permutation):
            inverse[int(old_axis)] = int(new_axis)
        bias_shape = [1] * len(x_shape)
        bias_shape[inverse[channel_axis]] = channel_size
    bias_index, bias = next_index, next_op
    if (
        bias is None
        or str(bias.op_type) != "ADD"
        or len(bias.inputs) != 2
        or len(bias.outputs) != 1
        or bias_data not in bias.inputs
    ):
        return None
    bias_name = str(
        bias.inputs[0] if str(bias.inputs[1]) == bias_data else bias.inputs[1]
    )
    if scale_name == bias_name:
        return None

    produced = [
        (mean1_index, mean_name),
        (sub_index, centered),
        (square_index, squared),
        (int(mean2_index), variance),
        (int(add_index), variance_epsilon),
        (int(sqrt_index), std),
        (int(div_index), inverse_std),
        (norm_index, normalized),
        (int(scale_index), scaled),
    ] + ([(int(post_index), bias_data)] if post_index is not None else [])
    public = {str(value) for value in model_ir.inputs + model_ir.outputs}
    if any(
        name in public
        or name in index.duplicate_producers
        or index.producers.get(name) != int(producer)
        for producer, name in produced
    ):
        return None
    order = [producer for producer, _ in produced] + [int(bias_index)]
    if order != sorted(order) or len(order) != len(set(order)):
        return None
    expected_consumers = {
        mean_name: [sub_index],
        centered: [square_index, square_index, norm_index],
        squared: [int(mean2_index)],
        variance: [int(add_index)],
        variance_epsilon: [int(sqrt_index)],
        std: [int(div_index)],
        inverse_std: [norm_index],
        normalized: [int(scale_index)],
        scaled: [int(post_index if post_index is not None else bias_index)],
    }
    if post_index is not None:
        expected_consumers[bias_data] = [int(bias_index)]
    if any(
        index.consumer_indices(name) != users
        for name, users in expected_consumers.items()
    ):
        return None

    desired_axes = [axis for axis in range(1, len(x_shape)) if axis != channel_axis]
    reduced = list(x_shape)
    for axis in desired_axes:
        reduced[axis] = 1
    broadcast = [1] * len(x_shape)
    broadcast[channel_axis] = channel_size
    axes_plans: list[tuple[str, _Mutation, list[int]]] = []
    for name in dict.fromkeys([axes1_name, axes2_name]):
        plan = _axes_plan(model_ir.tensors.get(name), desired_axes)
        if plan is None:
            return None
        users = ([mean1_index] if name == axes1_name else []) + (
            [int(mean2_index)] if name == axes2_name else []
        )
        axes_plans.append((name, plan, users))
    shape_targets = {
        mean_name: reduced,
        variance: reduced,
        variance_epsilon: reduced,
        std: reduced,
        inverse_std: reduced,
        centered: x_shape,
        squared: x_shape,
        normalized: x_shape,
        scaled: x_shape,
    }
    shape_plans = [
        _shape_plan(model_ir.tensors.get(name), target)
        for name, target in shape_targets.items()
    ]
    scale_plan = _constant_plan(model_ir.tensors.get(scale_name), broadcast)
    bias_plan = _constant_plan(model_ir.tensors.get(bias_name), bias_shape)
    if (
        any(plan is None for plan in shape_plans)
        or scale_plan is None
        or bias_plan is None
    ):
        return None
    if (
        any(
            not _constant_change_is_private(model_ir, index, name, users, plan)
            for name, plan, users in axes_plans
        )
        or not _constant_change_is_private(
            model_ir, index, scale_name, [int(scale_index)], scale_plan
        )
        or not _constant_change_is_private(
            model_ir, index, bias_name, [int(bias_index)], bias_plan
        )
    ):
        return None
    return (
        [plan for _, plan, _ in axes_plans]
        + [plan for plan in shape_plans if plan is not None]
        + [scale_plan, bias_plan]
    )


def _repair_decomposed_instance_normalization_layouts(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    stats_key = "repaired_decomposed_instance_normalization_layouts"
    if not any(
        str(op.op_type) == "MEAN"
        and str(getattr(op, "onnx_op_type", "")) == "InstanceNormalization"
        for op in model_ir.operators
    ):
        return {stats_key: 0}
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    repaired = 0
    for mean_index in active_index.operator_indices("MEAN"):
        plans = _candidate_plans(model_ir, active_index, int(mean_index))
        if plans and any(plan.changed for plan in plans):
            for plan in plans:
                _apply(plan)
            repaired += 1
    if repaired and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {stats_key: repaired}
