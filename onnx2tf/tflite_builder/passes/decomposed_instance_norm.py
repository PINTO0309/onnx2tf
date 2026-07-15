from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import AbstractSet, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _normalize_squeeze_axes_for_rank,
    _replace_operator_input_at,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _TensorContract,
    _constant_vector,
    _producer_is_valid,
    _tensor_contract,
    _unique_tensor_name,
)


FLOAT_DTYPES = {
    "FLOAT16": np.dtype(np.float16),
    "FLOAT32": np.dtype(np.float32),
}


@dataclass(frozen=True)
class TensorMetadataUpdate:
    contract: _TensorContract
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class ConstantUse:
    operator: OperatorIR
    input_index: int


@dataclass(frozen=True)
class ConstantUpdate:
    tensor: TensorIR
    data: np.ndarray
    uses: Tuple[ConstantUse, ...]
    clone_name: Optional[str]
    clone: Optional[TensorIR]


@dataclass(frozen=True)
class DecomposedInstanceNormCore:
    """Side-effect-free match of one strict rank-4 InstanceNorm decomposition."""

    ordered_ops: Tuple[OperatorIR, ...]
    x_name: str
    x: _TensorContract
    mean1: OperatorIR
    mean1_contract: _TensorContract
    mean1_axes_name: str
    sub: OperatorIR
    sub_x_input_index: int
    centered: _TensorContract
    square: OperatorIR
    squared: _TensorContract
    mean2: OperatorIR
    mean2_contract: _TensorContract
    mean2_axes_name: str
    add_epsilon: OperatorIR
    add_epsilon_contract: _TensorContract
    epsilon_name: str
    sqrt: OperatorIR
    sqrt_contract: _TensorContract
    div: OperatorIR
    div_contract: _TensorContract
    one_name: str
    norm: OperatorIR
    normalized: _TensorContract
    scale: OperatorIR
    scaled: _TensorContract
    scale_name: str
    scale_input_index: int


def tensor_contract_exact(
    model_ir: ModelIR,
    name: str,
    rank: int,
    shape: Sequence[int],
    signature: Sequence[int],
) -> Optional[_TensorContract]:
    contract = _tensor_contract(model_ir, str(name), int(rank))
    if (
        contract is None
        or contract.shape != tuple(int(value) for value in shape)
        or contract.signature != tuple(int(value) for value in signature)
    ):
        return None
    return contract


def normalized_axes(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    rank: int,
    size: int,
    public_inputs: AbstractSet[str],
) -> Optional[Tuple[int, ...]]:
    values = _constant_vector(
        model_ir,
        graph_index,
        str(name),
        int(size),
        public_inputs,
    )
    if values is None:
        return None
    normalized = _normalize_squeeze_axes_for_rank(
        [int(value) for value in values],
        int(rank),
    )
    if normalized is None or len(normalized) != int(size):
        return None
    return tuple(int(value) for value in normalized)


def float_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    dtype: str,
    *,
    shape: Optional[Tuple[int, ...]] = None,
    value: Optional[float] = None,
    nonnegative: bool = False,
) -> Optional[np.ndarray]:
    tensor = model_ir.tensors.get(str(name))
    expected_dtype = FLOAT_DTYPES.get(str(dtype))
    if (
        tensor is None
        or tensor.data is None
        or expected_dtype is None
        or str(tensor.dtype) != str(dtype)
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or tensor.quantization is not None
    ):
        return None
    try:
        data = np.asarray(tensor.data)
        tensor_shape = tuple(int(item) for item in tensor.shape)
        signature = (
            tensor_shape
            if tensor.shape_signature is None
            else tuple(int(item) for item in tensor.shape_signature)
        )
    except Exception:
        return None
    if (
        data.dtype != expected_dtype
        or data.shape != tensor_shape
        or signature != tensor_shape
        or not np.all(np.isfinite(data))
        or (shape is not None and tensor_shape != tuple(shape))
    ):
        return None
    if value is not None and (
        data.size != 1 or float(data.reshape(-1)[0]) != float(value)
    ):
        return None
    if nonnegative and (data.size != 1 or float(data.reshape(-1)[0]) < 0.0):
        return None
    return data


def binary_other_input(
    operator: OperatorIR,
    data_name: str,
) -> Optional[Tuple[str, int]]:
    if len(operator.inputs) != 2:
        return None
    matches = [
        index
        for index, name in enumerate(operator.inputs)
        if str(name) == str(data_name)
    ]
    if len(matches) != 1:
        return None
    data_index = int(matches[0])
    return str(operator.inputs[1 - data_index]), 1 - data_index


def sole_consumer(
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Optional[Tuple[int, OperatorIR]]:
    users = graph_index.consumer_indices(str(name))
    if len(users) != 1:
        return None
    index = int(users[0])
    return index, graph_index.model_ir.operators[index]


def plan_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    data: np.ndarray,
    uses: Sequence[ConstantUse],
    suffix: str,
    public_names: AbstractSet[str],
) -> Optional[ConstantUpdate]:
    name = str(tensor_name)
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or name in public_names
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or not uses
    ):
        return None
    resolved_indices = []
    for use in uses:
        operator_index = graph_index.operator_index(use.operator)
        if (
            operator_index is None
            or int(use.input_index) < 0
            or int(use.input_index) >= len(use.operator.inputs)
            or str(use.operator.inputs[int(use.input_index)]) != name
        ):
            return None
        resolved_indices.append(int(operator_index))
    try:
        replacement = np.asarray(data, dtype=np.asarray(tensor.data).dtype)
    except Exception:
        return None
    clone_name: Optional[str] = None
    clone: Optional[TensorIR] = None
    if Counter(graph_index.consumer_indices(name)) != Counter(resolved_indices):
        clone_name = _unique_tensor_name(model_ir, f"{name}_{suffix}")
        try:
            quantization = _clone_quantization(tensor.quantization)
        except Exception:
            return None
        clone = TensorIR(
            name=clone_name,
            dtype=str(tensor.dtype),
            shape=[int(value) for value in replacement.shape],
            shape_signature=[int(value) for value in replacement.shape],
            data=np.asarray(replacement),
            is_variable=False,
            quantization=quantization,
            logical_layout=str(tensor.logical_layout),
            physical_layout=str(tensor.physical_layout),
            onnx_tensor_name=tensor.onnx_tensor_name,
        )
    return ConstantUpdate(
        tensor=tensor,
        data=np.asarray(replacement),
        uses=tuple(uses),
        clone_name=clone_name,
        clone=clone,
    )


def apply_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    update: ConstantUpdate,
) -> bool:
    target = update.tensor
    if update.clone_name is not None:
        if update.clone is None or update.clone_name in model_ir.tensors:
            return False
        model_ir.tensors[update.clone_name] = update.clone
        target = update.clone
        for use in update.uses:
            _replace_operator_input_at(
                model_ir=model_ir,
                op=use.operator,
                input_index=use.input_index,
                new_input_name=update.clone_name,
                graph_index=graph_index,
            )
    target.data = np.asarray(update.data)
    target.shape = [int(value) for value in update.data.shape]
    target.shape_signature = [int(value) for value in update.data.shape]
    return True


def constant_is_private_and_unquantized(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    public_names: AbstractSet[str],
) -> bool:
    tensor = model_ir.tensors.get(str(name))
    return bool(
        tensor is not None
        and tensor.data is not None
        and str(name) not in public_names
        and str(name) not in graph_index.producers
        and str(name) not in graph_index.duplicate_producers
        and tensor.quantization is None
    )


def _coefficient_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    dtype: str,
    channel_count: int,
) -> Optional[np.ndarray]:
    data = float_constant(model_ir, graph_index, name, dtype)
    if data is None:
        return None
    shape = tuple(int(value) for value in data.shape)
    if data.size == 1 or shape == (1, 1, 1, int(channel_count)):
        return np.asarray(data)
    if shape != (1, int(channel_count), 1, 1):
        return None
    return np.transpose(data, (0, 2, 3, 1))


def plan_nhwc_coefficient_updates(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    coefficient_uses: Sequence[Tuple[str, OperatorIR, int]],
    dtype: str,
    channel_count: int,
    public_names: AbstractSet[str],
) -> Optional[Tuple[ConstantUpdate, ...]]:
    """Plan scalar or channelwise coefficient updates as one transaction."""

    grouped_uses: dict[str, list[ConstantUse]] = {}
    for coefficient_name, operator, input_index in coefficient_uses:
        grouped_uses.setdefault(str(coefficient_name), []).append(
            ConstantUse(operator, int(input_index))
        )
    updates = []
    for coefficient_name, uses in grouped_uses.items():
        if not constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            coefficient_name,
            public_names,
        ):
            return None
        replacement = _coefficient_replacement(
            model_ir,
            graph_index,
            name=coefficient_name,
            dtype=dtype,
            channel_count=int(channel_count),
        )
        if replacement is None:
            return None
        current = np.asarray(model_ir.tensors[coefficient_name].data)
        if current.shape == replacement.shape and np.array_equal(
            current,
            replacement,
        ):
            continue
        update = plan_constant_update(
            model_ir,
            graph_index,
            coefficient_name,
            replacement,
            tuple(uses),
            "nhwc_coefficient",
            public_names,
        )
        if update is None:
            return None
        updates.append(update)
    return tuple(updates)


def plan_nhwc_instance_norm_constant_updates(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    core: DecomposedInstanceNormCore,
    bias_name: str,
    bias_operator: OperatorIR,
    bias_input_index: int,
    channel_count: int,
    public_names: AbstractSet[str],
    additional_coefficient_uses: Sequence[
        Tuple[str, OperatorIR, int]
    ] = (),
) -> Optional[Tuple[ConstantUpdate, ...]]:
    """Plan all Mean-axis and affine updates for one NHWC rewrite."""

    axes_uses: dict[str, list[ConstantUse]] = {}
    axes_uses.setdefault(core.mean1_axes_name, []).append(
        ConstantUse(core.mean1, 1)
    )
    axes_uses.setdefault(core.mean2_axes_name, []).append(
        ConstantUse(core.mean2, 1)
    )
    updates = []
    for axes_name, uses in axes_uses.items():
        if not constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            axes_name,
            public_names,
        ):
            return None
        update = plan_constant_update(
            model_ir,
            graph_index,
            axes_name,
            np.asarray(
                [1, 2],
                dtype=np.asarray(model_ir.tensors[axes_name].data).dtype,
            ),
            tuple(uses),
            "nhwc_axes",
            public_names,
        )
        if update is None:
            return None
        updates.append(update)

    coefficient_updates = plan_nhwc_coefficient_updates(
        model_ir,
        graph_index,
        coefficient_uses=(
            (core.scale_name, core.scale, core.scale_input_index),
            (str(bias_name), bias_operator, int(bias_input_index)),
            *additional_coefficient_uses,
        ),
        dtype=str(core.x.tensor.dtype),
        channel_count=int(channel_count),
        public_names=public_names,
    )
    if coefficient_updates is None:
        return None
    return tuple(updates) + coefficient_updates


def match_decomposed_instance_norm_core(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    x_name: str,
    x: _TensorContract,
    public_inputs: AbstractSet[str],
    public_outputs: AbstractSet[str],
    allow_commuted_sub: bool = False,
) -> Optional[DecomposedInstanceNormCore]:
    """Match the NCHW decomposition without mutating the graph or constants."""

    public_names = set(public_inputs) | set(public_outputs)
    x_name = str(x_name)
    dtype = str(x.tensor.dtype)
    if (
        str(x.tensor.name) != x_name
        or len(x.shape) != 4
        or x_name in public_names
        or dtype not in FLOAT_DTYPES
        or x.tensor.quantization is not None
    ):
        return None

    x_users = sorted(set(graph_index.consumer_indices(x_name)))
    if len(x_users) != 2:
        return None
    mean_matches = [
        index
        for index in x_users
        if str(model_ir.operators[index].op_type) == "MEAN"
        and len(model_ir.operators[index].inputs) == 2
        and str(model_ir.operators[index].inputs[0]) == x_name
    ]
    sub_matches = [
        index
        for index in x_users
        if str(model_ir.operators[index].op_type) == "SUB"
        and len(model_ir.operators[index].inputs) == 2
        and len(model_ir.operators[index].outputs) == 1
    ]
    if len(mean_matches) != 1 or len(sub_matches) != 1:
        return None
    mean1_index = int(mean_matches[0])
    sub_index = int(sub_matches[0])
    mean1 = model_ir.operators[mean1_index]
    sub = model_ir.operators[sub_index]
    mean1_name = str(mean1.outputs[0]) if len(mean1.outputs) == 1 else ""
    reduced_shape = (x.shape[0], x.shape[1], 1, 1)
    reduced_signature = (x.signature[0], x.signature[1], 1, 1)
    mean1_contract = tensor_contract_exact(
        model_ir,
        mean1_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    mean1_axes_name = str(mean1.inputs[1])
    sub_inputs = [str(value) for value in sub.inputs]
    expected_sub_inputs = [x_name, mean1_name]
    if allow_commuted_sub:
        valid_sub_inputs = Counter(sub_inputs) == Counter(expected_sub_inputs)
    else:
        valid_sub_inputs = sub_inputs == expected_sub_inputs
    mean1_options = dict(mean1.options) if isinstance(mean1.options, dict) else {}
    if (
        len(mean1.outputs) != 1
        or not bool(mean1_options.get("keepDims", False))
        or normalized_axes(
            model_ir,
            graph_index,
            mean1_axes_name,
            4,
            2,
            public_inputs,
        )
        not in {(2, 3), (3, 2)}
        or mean1_axes_name in public_outputs
        or mean1_contract is None
        or mean1_name in public_names
        or not _producer_is_valid(graph_index, mean1_name, mean1_index)
        or graph_index.consumer_indices(mean1_name) != [sub_index]
        or not valid_sub_inputs
    ):
        return None
    sub_x_input_index = sub_inputs.index(x_name)
    centered_name = str(sub.outputs[0])
    centered = tensor_contract_exact(
        model_ir,
        centered_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        centered is None
        or centered_name in public_names
        or not _producer_is_valid(graph_index, centered_name, sub_index)
    ):
        return None

    centered_user_indices = sorted(set(graph_index.consumer_indices(centered_name)))
    square_matches = [
        index
        for index in centered_user_indices
        if str(model_ir.operators[index].op_type) == "MUL"
        and [str(value) for value in model_ir.operators[index].inputs]
        == [centered_name, centered_name]
        and len(model_ir.operators[index].outputs) == 1
    ]
    norm_matches = [
        index
        for index in centered_user_indices
        if str(model_ir.operators[index].op_type) == "MUL"
        and len(model_ir.operators[index].inputs) == 2
        and len(model_ir.operators[index].outputs) == 1
        and Counter(str(value) for value in model_ir.operators[index].inputs)[
            centered_name
        ]
        == 1
    ]
    if len(square_matches) != 1 or len(norm_matches) != 1:
        return None
    square_index = int(square_matches[0])
    norm_index = int(norm_matches[0])
    if Counter(graph_index.consumer_indices(centered_name)) != Counter(
        [square_index, square_index, norm_index]
    ):
        return None
    square = model_ir.operators[square_index]
    squared_name = str(square.outputs[0])
    squared = tensor_contract_exact(
        model_ir,
        squared_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        squared is None
        or squared_name in public_names
        or not _producer_is_valid(graph_index, squared_name, square_index)
    ):
        return None

    mean2_match = sole_consumer(graph_index, squared_name)
    if mean2_match is None:
        return None
    mean2_index, mean2 = mean2_match
    mean2_name = str(mean2.outputs[0]) if len(mean2.outputs) == 1 else ""
    mean2_contract = tensor_contract_exact(
        model_ir,
        mean2_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    mean2_axes_name = str(mean2.inputs[1]) if len(mean2.inputs) == 2 else ""
    mean2_options = dict(mean2.options) if isinstance(mean2.options, dict) else {}
    if (
        str(mean2.op_type) != "MEAN"
        or len(mean2.inputs) != 2
        or len(mean2.outputs) != 1
        or str(mean2.inputs[0]) != squared_name
        or not bool(mean2_options.get("keepDims", False))
        or normalized_axes(
            model_ir,
            graph_index,
            mean2_axes_name,
            4,
            2,
            public_inputs,
        )
        not in {(2, 3), (3, 2)}
        or mean2_axes_name in public_outputs
        or mean2_contract is None
        or mean2_name in public_names
        or not _producer_is_valid(graph_index, mean2_name, mean2_index)
    ):
        return None

    add_epsilon_match = sole_consumer(graph_index, mean2_name)
    if add_epsilon_match is None:
        return None
    add_epsilon_index, add_epsilon = add_epsilon_match
    epsilon_match = binary_other_input(add_epsilon, mean2_name)
    add_epsilon_name = (
        str(add_epsilon.outputs[0]) if len(add_epsilon.outputs) == 1 else ""
    )
    add_epsilon_contract = tensor_contract_exact(
        model_ir,
        add_epsilon_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    if (
        str(add_epsilon.op_type) != "ADD"
        or epsilon_match is None
        or len(add_epsilon.outputs) != 1
        or add_epsilon_contract is None
        or add_epsilon_name in public_names
        or not _producer_is_valid(
            graph_index,
            add_epsilon_name,
            add_epsilon_index,
        )
    ):
        return None
    epsilon_name = epsilon_match[0]

    sqrt_match = sole_consumer(graph_index, add_epsilon_name)
    if sqrt_match is None:
        return None
    sqrt_index, sqrt = sqrt_match
    sqrt_name = str(sqrt.outputs[0]) if len(sqrt.outputs) == 1 else ""
    sqrt_contract = tensor_contract_exact(
        model_ir,
        sqrt_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    if (
        str(sqrt.op_type) != "SQRT"
        or [str(value) for value in sqrt.inputs] != [add_epsilon_name]
        or len(sqrt.outputs) != 1
        or sqrt_contract is None
        or sqrt_name in public_names
        or not _producer_is_valid(graph_index, sqrt_name, sqrt_index)
    ):
        return None

    div_match = sole_consumer(graph_index, sqrt_name)
    if div_match is None:
        return None
    div_index, div = div_match
    div_name = str(div.outputs[0]) if len(div.outputs) == 1 else ""
    div_contract = tensor_contract_exact(
        model_ir,
        div_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    if (
        str(div.op_type) != "DIV"
        or len(div.inputs) != 2
        or len(div.outputs) != 1
        or str(div.inputs[1]) != sqrt_name
        or div_contract is None
        or div_name in public_names
        or not _producer_is_valid(graph_index, div_name, div_index)
    ):
        return None
    one_name = str(div.inputs[0])

    norm = model_ir.operators[norm_index]
    norm_other = binary_other_input(norm, centered_name)
    normalized_name = str(norm.outputs[0])
    normalized = tensor_contract_exact(
        model_ir,
        normalized_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        norm_other is None
        or norm_other[0] != div_name
        or graph_index.consumer_indices(div_name) != [norm_index]
        or normalized is None
        or normalized_name in public_names
        or not _producer_is_valid(graph_index, normalized_name, norm_index)
    ):
        return None

    scale_match = sole_consumer(graph_index, normalized_name)
    if scale_match is None:
        return None
    scale_index, scale = scale_match
    scale_constant_match = binary_other_input(scale, normalized_name)
    scaled_name = str(scale.outputs[0]) if len(scale.outputs) == 1 else ""
    scaled = tensor_contract_exact(
        model_ir,
        scaled_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        str(scale.op_type) != "MUL"
        or scale_constant_match is None
        or len(scale.outputs) != 1
        or scaled is None
        or scaled_name in public_names
        or not _producer_is_valid(graph_index, scaled_name, scale_index)
    ):
        return None
    scale_name, scale_input_index = scale_constant_match

    ordered_ops = (
        mean1,
        sub,
        square,
        mean2,
        add_epsilon,
        sqrt,
        div,
        norm,
        scale,
    )
    ordered_indices = [graph_index.operator_index(op) for op in ordered_ops]
    contracts = (
        x,
        mean1_contract,
        centered,
        squared,
        mean2_contract,
        add_epsilon_contract,
        sqrt_contract,
        div_contract,
        normalized,
        scaled,
    )
    if (
        any(index is None for index in ordered_indices)
        or [int(index) for index in ordered_indices if index is not None]
        != sorted(int(index) for index in ordered_indices if index is not None)
        or len({id(op) for op in ordered_ops}) != len(ordered_ops)
        or any(str(contract.tensor.dtype) != dtype for contract in contracts)
        or any(contract.tensor.quantization is not None for contract in contracts)
        or float_constant(
            model_ir,
            graph_index,
            epsilon_name,
            dtype,
            nonnegative=True,
        )
        is None
        or float_constant(
            model_ir,
            graph_index,
            one_name,
            dtype,
            value=1.0,
        )
        is None
    ):
        return None

    return DecomposedInstanceNormCore(
        ordered_ops=ordered_ops,
        x_name=x_name,
        x=x,
        mean1=mean1,
        mean1_contract=mean1_contract,
        mean1_axes_name=mean1_axes_name,
        sub=sub,
        sub_x_input_index=sub_x_input_index,
        centered=centered,
        square=square,
        squared=squared,
        mean2=mean2,
        mean2_contract=mean2_contract,
        mean2_axes_name=mean2_axes_name,
        add_epsilon=add_epsilon,
        add_epsilon_contract=add_epsilon_contract,
        epsilon_name=epsilon_name,
        sqrt=sqrt,
        sqrt_contract=sqrt_contract,
        div=div,
        div_contract=div_contract,
        one_name=one_name,
        norm=norm,
        normalized=normalized,
        scale=scale,
        scaled=scaled,
        scale_name=scale_name,
        scale_input_index=scale_input_index,
    )
