from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _valid_source,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _TensorContract,
    _constant_vector,
    _producer_is_valid,
    _tensor_contract,
)
from onnx2tf.tflite_builder.passes.decomposed_instance_norm import (
    FLOAT_DTYPES,
    ConstantUpdate,
    ConstantUse,
    TensorMetadataUpdate,
    apply_constant_update,
    binary_other_input,
    constant_is_private_and_unquantized,
    float_constant,
    normalized_axes,
    plan_constant_update,
    plan_nhwc_coefficient_updates,
    sole_consumer,
    tensor_contract_exact,
)


_STATS_KEY = (
    "optimized_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)


@dataclass(frozen=True)
class _DualStatPath:
    ordered_ops: Tuple[OperatorIR, ...]
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
    factor: OperatorIR
    factor_contract: _TensorContract
    factor_name: str
    add_epsilon: OperatorIR
    add_epsilon_contract: _TensorContract
    epsilon_name: str
    sqrt: OperatorIR
    sqrt_contract: _TensorContract
    div: OperatorIR
    div_contract: _TensorContract
    scale: OperatorIR
    scaled: _TensorContract
    scale_name: str
    scale_input_index: int


@dataclass(frozen=True)
class _CoefficientUsePlan:
    operator: OperatorIR
    input_index: int
    coefficient_name: str
    bypass_name: Optional[str]
    bypass_reshape: Optional[OperatorIR]


@dataclass(frozen=True)
class _DualStatsPlan:
    involved_ops: Tuple[OperatorIR, ...]
    pre: OperatorIR
    source_name: str
    spatial: _DualStatPath
    global_path: _DualStatPath
    blend_add: OperatorIR
    blend_add_contract: _TensorContract
    blend_mul: OperatorIR
    blend_mul_contract: _TensorContract
    gamma: _CoefficientUsePlan
    blend_bias: OperatorIR
    inst_output: _TensorContract
    beta: _CoefficientUsePlan
    residual_pre: Optional[OperatorIR]
    residual_source_name: Optional[str]
    tail_add: Optional[OperatorIR]
    tail_add_contract: Optional[_TensorContract]
    tail_add_residual_input_index: Optional[int]
    post: OperatorIR
    post_output_name: str
    output_owner: OperatorIR
    output_owner_old_name: str
    constant_updates: Tuple[ConstantUpdate, ...]
    metadata_updates: Tuple[TensorMetadataUpdate, ...]
    channel_last_names: Tuple[str, ...]


def _permuted(values: Tuple[int, ...], permutation: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(int(values[index]) for index in permutation)


def _axes_equivalent(
    actual: Optional[Tuple[int, ...]],
    expected: Tuple[int, ...],
) -> bool:
    return bool(
        actual is not None
        and len(actual) == len(expected)
        and sorted(int(value) for value in actual)
        == sorted(int(value) for value in expected)
    )


def _valid_permutation_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    expected: Tuple[int, ...],
    public_inputs: set[str],
    public_names: set[str],
) -> bool:
    return bool(
        _constant_vector(
            model_ir,
            graph_index,
            str(name),
            len(expected),
            public_inputs,
        )
        == expected
        and constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            str(name),
            public_names,
        )
    )


def _valid_scalar_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    dtype: str,
    public_names: set[str],
    *,
    nonnegative: bool,
) -> bool:
    data = float_constant(
        model_ir,
        graph_index,
        str(name),
        str(dtype),
        nonnegative=bool(nonnegative),
    )
    return bool(
        data is not None
        and int(data.size) == 1
        and str(name) not in public_names
    )


def _match_path(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    x_name: str,
    x: _TensorContract,
    mean1: OperatorIR,
    sub: OperatorIR,
    expected_axes: Tuple[int, ...],
    reduced_shape: Tuple[int, ...],
    reduced_signature: Tuple[int, ...],
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_DualStatPath]:
    public_names = public_inputs | public_outputs
    dtype = str(x.tensor.dtype)
    mean1_index = graph_index.operator_index(mean1)
    sub_index = graph_index.operator_index(sub)
    mean1_name = str(mean1.outputs[0]) if len(mean1.outputs) == 1 else ""
    mean1_axes_name = str(mean1.inputs[1]) if len(mean1.inputs) == 2 else ""
    mean1_contract = tensor_contract_exact(
        model_ir,
        mean1_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    sub_inputs = [str(value) for value in sub.inputs]
    if (
        mean1_index is None
        or sub_index is None
        or str(mean1.op_type) != "MEAN"
        or len(mean1.inputs) != 2
        or len(mean1.outputs) != 1
        or str(mean1.inputs[0]) != x_name
        or not bool(
            (dict(mean1.options) if isinstance(mean1.options, dict) else {}).get(
                "keepDims",
                False,
            )
        )
        or not _axes_equivalent(
            normalized_axes(
                model_ir,
                graph_index,
                mean1_axes_name,
                4,
                len(expected_axes),
                public_inputs,
            ),
            expected_axes,
        )
        or not constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            mean1_axes_name,
            public_names,
        )
        or mean1_contract is None
        or mean1_name in public_names
        or not _producer_is_valid(graph_index, mean1_name, mean1_index)
        or graph_index.consumer_indices(mean1_name) != [sub_index]
        or str(sub.op_type) != "SUB"
        or len(sub.inputs) != 2
        or len(sub.outputs) != 1
        or Counter(sub_inputs) != Counter([x_name, mean1_name])
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

    centered_users = graph_index.consumer_indices(centered_name)
    square_matches = [
        index
        for index in sorted(set(centered_users))
        if str(model_ir.operators[index].op_type) == "MUL"
        and [str(value) for value in model_ir.operators[index].inputs]
        == [centered_name, centered_name]
        and len(model_ir.operators[index].outputs) == 1
    ]
    div_matches = [
        index
        for index in sorted(set(centered_users))
        if str(model_ir.operators[index].op_type) == "DIV"
        and len(model_ir.operators[index].inputs) == 2
        and str(model_ir.operators[index].inputs[0]) == centered_name
        and len(model_ir.operators[index].outputs) == 1
    ]
    if len(square_matches) != 1 or len(div_matches) != 1:
        return None
    square_index = int(square_matches[0])
    div_index = int(div_matches[0])
    if Counter(centered_users) != Counter([square_index, square_index, div_index]):
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
    mean2_axes_name = str(mean2.inputs[1]) if len(mean2.inputs) == 2 else ""
    mean2_contract = tensor_contract_exact(
        model_ir,
        mean2_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    if (
        str(mean2.op_type) != "MEAN"
        or len(mean2.inputs) != 2
        or len(mean2.outputs) != 1
        or str(mean2.inputs[0]) != squared_name
        or not bool(
            (dict(mean2.options) if isinstance(mean2.options, dict) else {}).get(
                "keepDims",
                False,
            )
        )
        or not _axes_equivalent(
            normalized_axes(
                model_ir,
                graph_index,
                mean2_axes_name,
                4,
                len(expected_axes),
                public_inputs,
            ),
            expected_axes,
        )
        or not constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            mean2_axes_name,
            public_names,
        )
        or mean2_contract is None
        or mean2_name in public_names
        or not _producer_is_valid(graph_index, mean2_name, mean2_index)
    ):
        return None

    factor_match = sole_consumer(graph_index, mean2_name)
    if factor_match is None:
        return None
    factor_index, factor = factor_match
    factor_constant_match = binary_other_input(factor, mean2_name)
    factor_output_name = str(factor.outputs[0]) if len(factor.outputs) == 1 else ""
    factor_contract = tensor_contract_exact(
        model_ir,
        factor_output_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    if (
        str(factor.op_type) != "MUL"
        or factor_constant_match is None
        or len(factor.outputs) != 1
        or factor_contract is None
        or factor_output_name in public_names
        or not _producer_is_valid(graph_index, factor_output_name, factor_index)
    ):
        return None
    factor_name = factor_constant_match[0]

    add_epsilon_match = sole_consumer(graph_index, factor_output_name)
    if add_epsilon_match is None:
        return None
    add_epsilon_index, add_epsilon = add_epsilon_match
    epsilon_match = binary_other_input(add_epsilon, factor_output_name)
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

    div = model_ir.operators[div_index]
    div_name = str(div.outputs[0])
    div_contract = tensor_contract_exact(
        model_ir,
        div_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        [str(value) for value in div.inputs] != [centered_name, sqrt_name]
        or graph_index.consumer_indices(sqrt_name) != [div_index]
        or div_contract is None
        or div_name in public_names
        or not _producer_is_valid(graph_index, div_name, div_index)
    ):
        return None

    scale_match = sole_consumer(graph_index, div_name)
    if scale_match is None:
        return None
    scale_index, scale = scale_match
    scale_constant_match = binary_other_input(scale, div_name)
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
        factor,
        add_epsilon,
        sqrt,
        div,
        scale,
    )
    ordered_indices = [graph_index.operator_index(op) for op in ordered_ops]
    contracts = (
        mean1_contract,
        centered,
        squared,
        mean2_contract,
        factor_contract,
        add_epsilon_contract,
        sqrt_contract,
        div_contract,
        scaled,
    )
    if (
        any(index is None for index in ordered_indices)
        or [int(index) for index in ordered_indices if index is not None]
        != sorted(int(index) for index in ordered_indices if index is not None)
        or len({id(op) for op in ordered_ops}) != len(ordered_ops)
        or any(str(contract.tensor.dtype) != dtype for contract in contracts)
        or any(contract.tensor.quantization is not None for contract in contracts)
        or not _valid_scalar_constant(
            model_ir,
            graph_index,
            factor_name,
            dtype,
            public_names,
            nonnegative=True,
        )
        or not _valid_scalar_constant(
            model_ir,
            graph_index,
            epsilon_name,
            dtype,
            public_names,
            nonnegative=True,
        )
    ):
        return None

    return _DualStatPath(
        ordered_ops=ordered_ops,
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
        factor=factor,
        factor_contract=factor_contract,
        factor_name=factor_name,
        add_epsilon=add_epsilon,
        add_epsilon_contract=add_epsilon_contract,
        epsilon_name=epsilon_name,
        sqrt=sqrt,
        sqrt_contract=sqrt_contract,
        div=div,
        div_contract=div_contract,
        scale=scale,
        scaled=scaled,
        scale_name=scale_name,
        scale_input_index=scale_input_index,
    )


def _resolve_coefficient_use(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    operator: OperatorIR,
    input_index: int,
    coefficient_name: str,
    dtype: str,
    channel_count: int,
    channel_signature: int,
    public_inputs: set[str],
    public_names: set[str],
) -> Optional[_CoefficientUsePlan]:
    producer_index = graph_index.producers.get(str(coefficient_name))
    operator_index = graph_index.operator_index(operator)
    if producer_index is None:
        if str(coefficient_name) in graph_index.duplicate_producers:
            return None
        return _CoefficientUsePlan(
            operator=operator,
            input_index=int(input_index),
            coefficient_name=str(coefficient_name),
            bypass_name=None,
            bypass_reshape=None,
        )
    if str(coefficient_name) in graph_index.duplicate_producers:
        return None
    reshape = model_ir.operators[int(producer_index)]
    reshape_output = _tensor_contract(model_ir, str(coefficient_name), 4)
    if (
        str(reshape.op_type) != "RESHAPE"
        or len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.outputs[0]) != str(coefficient_name)
        or reshape_output is None
        or reshape_output.shape != (1, int(channel_count), 1, 1)
        or reshape_output.signature[0] != 1
        or reshape_output.signature[1] not in {
            int(channel_count),
            int(channel_signature),
        }
        or reshape_output.signature[2:] != (1, 1)
        or str(reshape_output.tensor.dtype) != str(dtype)
        or reshape_output.tensor.quantization is not None
        or str(coefficient_name) in public_names
        or not _producer_is_valid(
            graph_index,
            str(coefficient_name),
            int(producer_index),
        )
        or graph_index.consumer_indices(str(coefficient_name))
        != [operator_index]
        or operator_index is None
        or int(producer_index) >= int(operator_index)
    ):
        return None
    shape_name = str(reshape.inputs[1])
    if (
        _constant_vector(
            model_ir,
            graph_index,
            shape_name,
            4,
            public_inputs,
        )
        != (1, int(channel_count), 1, 1)
        or not constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            shape_name,
            public_names,
        )
    ):
        return None
    source_name = str(reshape.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    if source_tensor is None:
        return None
    try:
        source_rank = len(source_tensor.shape)
    except (TypeError, ValueError):
        return None
    source = _tensor_contract(model_ir, source_name, source_rank)
    if (
        source is None
        or source_rank not in {1, 2}
        or source.shape
        not in {(int(channel_count),), (1, int(channel_count))}
        or source.signature[-1]
        not in {int(channel_count), int(channel_signature)}
        or str(source.tensor.dtype) != str(dtype)
        or source.tensor.quantization is not None
        or not _valid_source(
            graph_index,
            source,
            source_name,
            int(producer_index),
            public_inputs,
        )
    ):
        return None
    return _CoefficientUsePlan(
        operator=operator,
        input_index=int(input_index),
        coefficient_name=str(coefficient_name),
        bypass_name=source_name,
        bypass_reshape=reshape,
    )


def _plan_spatial_axes_updates(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    spatial: _DualStatPath,
    public_names: set[str],
) -> Optional[Tuple[ConstantUpdate, ...]]:
    uses_by_name: dict[str, list[ConstantUse]] = {}
    uses_by_name.setdefault(spatial.mean1_axes_name, []).append(
        ConstantUse(spatial.mean1, 1)
    )
    uses_by_name.setdefault(spatial.mean2_axes_name, []).append(
        ConstantUse(spatial.mean2, 1)
    )
    updates = []
    for axes_name, uses in uses_by_name.items():
        tensor = model_ir.tensors.get(axes_name)
        if (
            tensor is None
            or not constant_is_private_and_unquantized(
                model_ir,
                graph_index,
                axes_name,
                public_names,
            )
        ):
            return None
        update = plan_constant_update(
            model_ir,
            graph_index,
            axes_name,
            np.asarray([1, 2], dtype=np.asarray(tensor.data).dtype),
            tuple(uses),
            "nhwc_axes",
            public_names,
        )
        if update is None:
            return None
        updates.append(update)
    return tuple(updates)


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_DualStatsPlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    public_names = public_inputs | public_outputs
    pre = model_ir.operators[int(pre_index)]
    if (
        str(pre.op_type) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or not _valid_permutation_constant(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            _PERM_NHWC_TO_NCHW,
            public_inputs,
            public_names,
        )
    ):
        return None
    source_name = str(pre.inputs[0])
    x_name = str(pre.outputs[0])
    source = _tensor_contract(model_ir, source_name, 4)
    if source is None:
        return None
    x = tensor_contract_exact(
        model_ir,
        x_name,
        4,
        _permuted(source.shape, _PERM_NHWC_TO_NCHW),
        _permuted(source.signature, _PERM_NHWC_TO_NCHW),
    )
    if (
        x is None
        or source_name in public_outputs
        or x_name in public_names
        or str(source.tensor.dtype) not in FLOAT_DTYPES
        or source.tensor.quantization is not None
        or not _valid_source(
            graph_index,
            source,
            source_name,
            int(pre_index),
            public_inputs,
        )
        or not _producer_is_valid(graph_index, x_name, int(pre_index))
    ):
        return None

    x_user_indices = graph_index.consumer_indices(x_name)
    if len(x_user_indices) != 4 or len(set(x_user_indices)) != 4:
        return None
    means = []
    subs = []
    for index in sorted(x_user_indices):
        operator = model_ir.operators[index]
        if str(operator.op_type) == "MEAN":
            means.append(operator)
        elif str(operator.op_type) == "SUB":
            subs.append(operator)
    if len(means) != 2 or len(subs) != 2:
        return None

    mean_by_axes: dict[Tuple[int, ...], OperatorIR] = {}
    for mean in means:
        if len(mean.inputs) != 2:
            return None
        axes_name = str(mean.inputs[1])
        for expected in ((2, 3), (1, 2, 3)):
            axes = normalized_axes(
                model_ir,
                graph_index,
                axes_name,
                4,
                len(expected),
                public_inputs,
            )
            if _axes_equivalent(axes, expected):
                if expected in mean_by_axes:
                    return None
                mean_by_axes[expected] = mean
    if set(mean_by_axes) != {(2, 3), (1, 2, 3)}:
        return None

    sub_by_axes: dict[Tuple[int, ...], OperatorIR] = {}
    for expected, mean in mean_by_axes.items():
        mean_name = str(mean.outputs[0]) if len(mean.outputs) == 1 else ""
        matches = [
            sub
            for sub in subs
            if len(sub.inputs) == 2
            and Counter(str(value) for value in sub.inputs)
            == Counter([x_name, mean_name])
        ]
        if len(matches) != 1:
            return None
        sub_by_axes[expected] = matches[0]
    if len({id(value) for value in sub_by_axes.values()}) != 2:
        return None

    spatial_shape = (x.shape[0], x.shape[1], 1, 1)
    spatial_signature = (x.signature[0], x.signature[1], 1, 1)
    global_shape = (x.shape[0], 1, 1, 1)
    global_signature = (x.signature[0], 1, 1, 1)
    spatial = _match_path(
        model_ir,
        graph_index,
        x_name=x_name,
        x=x,
        mean1=mean_by_axes[(2, 3)],
        sub=sub_by_axes[(2, 3)],
        expected_axes=(2, 3),
        reduced_shape=spatial_shape,
        reduced_signature=spatial_signature,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    global_path = _match_path(
        model_ir,
        graph_index,
        x_name=x_name,
        x=x,
        mean1=mean_by_axes[(1, 2, 3)],
        sub=sub_by_axes[(1, 2, 3)],
        expected_axes=(1, 2, 3),
        reduced_shape=global_shape,
        reduced_signature=global_signature,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if spatial is None or global_path is None:
        return None

    spatial_blend = sole_consumer(graph_index, str(spatial.scaled.tensor.name))
    global_blend = sole_consumer(graph_index, str(global_path.scaled.tensor.name))
    if (
        spatial_blend is None
        or global_blend is None
        or spatial_blend[0] != global_blend[0]
    ):
        return None
    blend_add_index, blend_add = spatial_blend
    blend_add_name = str(blend_add.outputs[0]) if len(blend_add.outputs) == 1 else ""
    blend_add_contract = tensor_contract_exact(
        model_ir,
        blend_add_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        str(blend_add.op_type) != "ADD"
        or len(blend_add.inputs) != 2
        or len(blend_add.outputs) != 1
        or Counter(str(value) for value in blend_add.inputs)
        != Counter(
            [str(spatial.scaled.tensor.name), str(global_path.scaled.tensor.name)]
        )
        or blend_add_contract is None
        or blend_add_name in public_names
        or not _producer_is_valid(graph_index, blend_add_name, blend_add_index)
    ):
        return None

    blend_mul_match = sole_consumer(graph_index, blend_add_name)
    if blend_mul_match is None:
        return None
    blend_mul_index, blend_mul = blend_mul_match
    gamma_match = binary_other_input(blend_mul, blend_add_name)
    blend_mul_name = str(blend_mul.outputs[0]) if len(blend_mul.outputs) == 1 else ""
    blend_mul_contract = tensor_contract_exact(
        model_ir,
        blend_mul_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        str(blend_mul.op_type) != "MUL"
        or gamma_match is None
        or len(blend_mul.outputs) != 1
        or blend_mul_contract is None
        or blend_mul_name in public_names
        or not _producer_is_valid(graph_index, blend_mul_name, blend_mul_index)
    ):
        return None
    gamma = _resolve_coefficient_use(
        model_ir,
        graph_index,
        operator=blend_mul,
        input_index=gamma_match[1],
        coefficient_name=gamma_match[0],
        dtype=str(x.tensor.dtype),
        channel_count=int(x.shape[1]),
        channel_signature=int(x.signature[1]),
        public_inputs=public_inputs,
        public_names=public_names,
    )
    if gamma is None:
        return None

    blend_bias_match = sole_consumer(graph_index, blend_mul_name)
    if blend_bias_match is None:
        return None
    blend_bias_index, blend_bias = blend_bias_match
    beta_match = binary_other_input(blend_bias, blend_mul_name)
    inst_output_name = (
        str(blend_bias.outputs[0]) if len(blend_bias.outputs) == 1 else ""
    )
    inst_output = tensor_contract_exact(
        model_ir,
        inst_output_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        str(blend_bias.op_type) != "ADD"
        or beta_match is None
        or len(blend_bias.outputs) != 1
        or inst_output is None
        or inst_output_name in public_names
        or not _producer_is_valid(graph_index, inst_output_name, blend_bias_index)
    ):
        return None
    beta = _resolve_coefficient_use(
        model_ir,
        graph_index,
        operator=blend_bias,
        input_index=beta_match[1],
        coefficient_name=beta_match[0],
        dtype=str(x.tensor.dtype),
        channel_count=int(x.shape[1]),
        channel_signature=int(x.signature[1]),
        public_inputs=public_inputs,
        public_names=public_names,
    )
    if beta is None:
        return None

    inst_consumer = sole_consumer(graph_index, inst_output_name)
    if inst_consumer is None:
        return None
    inst_consumer_index, inst_consumer_op = inst_consumer
    residual_pre: Optional[OperatorIR] = None
    residual_source_name: Optional[str] = None
    tail_add: Optional[OperatorIR] = None
    tail_add_contract: Optional[_TensorContract] = None
    tail_add_residual_input_index: Optional[int] = None
    output_owner = blend_bias
    output_owner_old_name = inst_output_name
    if str(inst_consumer_op.op_type) == "TRANSPOSE":
        post = inst_consumer_op
        post_index = int(inst_consumer_index)
    elif str(inst_consumer_op.op_type) == "ADD":
        tail_add = inst_consumer_op
        tail_add_index = int(inst_consumer_index)
        residual_match = binary_other_input(tail_add, inst_output_name)
        tail_add_name = str(tail_add.outputs[0]) if len(tail_add.outputs) == 1 else ""
        tail_add_contract = tensor_contract_exact(
            model_ir,
            tail_add_name,
            4,
            x.shape,
            x.signature,
        )
        if (
            residual_match is None
            or len(tail_add.outputs) != 1
            or tail_add_contract is None
            or tail_add_name in public_names
            or not _producer_is_valid(graph_index, tail_add_name, tail_add_index)
        ):
            return None
        residual_name, tail_add_residual_input_index = residual_match
        residual_pre_index = graph_index.producers.get(residual_name)
        if (
            residual_pre_index is None
            or residual_name in graph_index.duplicate_producers
        ):
            return None
        residual_pre = model_ir.operators[int(residual_pre_index)]
        residual_source_name = (
            str(residual_pre.inputs[0]) if len(residual_pre.inputs) == 2 else ""
        )
        residual_source = tensor_contract_exact(
            model_ir,
            residual_source_name,
            4,
            source.shape,
            source.signature,
        )
        residual_output = tensor_contract_exact(
            model_ir,
            residual_name,
            4,
            x.shape,
            x.signature,
        )
        if (
            str(residual_pre.op_type) != "TRANSPOSE"
            or len(residual_pre.inputs) != 2
            or len(residual_pre.outputs) != 1
            or str(residual_pre.outputs[0]) != residual_name
            or not _valid_permutation_constant(
                model_ir,
                graph_index,
                str(residual_pre.inputs[1]),
                _PERM_NHWC_TO_NCHW,
                public_inputs,
                public_names,
            )
            or residual_source is None
            or residual_output is None
            or residual_source_name in public_outputs
            or residual_name in public_names
            or str(residual_source.tensor.dtype) != str(x.tensor.dtype)
            or residual_source.tensor.quantization is not None
            or residual_output.tensor.quantization is not None
            or graph_index.consumer_indices(residual_name) != [tail_add_index]
            or int(residual_pre_index) >= int(tail_add_index)
            or not _valid_source(
                graph_index,
                residual_source,
                residual_source_name,
                int(residual_pre_index),
                public_inputs,
            )
        ):
            return None
        post_match = sole_consumer(graph_index, tail_add_name)
        if post_match is None:
            return None
        post_index, post = post_match
        output_owner = tail_add
        output_owner_old_name = tail_add_name
    else:
        return None

    post_output_name = str(post.outputs[0]) if len(post.outputs) == 1 else ""
    post_output = tensor_contract_exact(
        model_ir,
        post_output_name,
        4,
        source.shape,
        source.signature,
    )
    if (
        str(post.op_type) != "TRANSPOSE"
        or len(post.inputs) != 2
        or len(post.outputs) != 1
        or str(post.inputs[0]) != output_owner_old_name
        or not _valid_permutation_constant(
            model_ir,
            graph_index,
            str(post.inputs[1]),
            _PERM_NCHW_TO_NHWC,
            public_inputs,
            public_names,
        )
        or post_output is None
        or post_output_name in public_names
        or not _producer_is_valid(graph_index, post_output_name, post_index)
        or any(
            int(index) <= int(post_index)
            for index in graph_index.consumer_indices(post_output_name)
        )
    ):
        return None

    involved_ops = (
        pre,
        *spatial.ordered_ops,
        *global_path.ordered_ops,
        blend_add,
        blend_mul,
        blend_bias,
        *((gamma.bypass_reshape,) if gamma.bypass_reshape is not None else ()),
        *((beta.bypass_reshape,) if beta.bypass_reshape is not None else ()),
        *((residual_pre,) if residual_pre is not None else ()),
        *((tail_add,) if tail_add is not None else ()),
        post,
    )
    involved_indices = [graph_index.operator_index(op) for op in involved_ops]
    dependency_tail_indices = [
        graph_index.operator_index(spatial.scale),
        graph_index.operator_index(global_path.scale),
        blend_add_index,
        blend_mul_index,
        blend_bias_index,
        *(([graph_index.operator_index(tail_add)]) if tail_add is not None else []),
        post_index,
    ]
    dtype = str(x.tensor.dtype)
    contracts = (
        source,
        x,
        spatial.scaled,
        global_path.scaled,
        blend_add_contract,
        blend_mul_contract,
        inst_output,
        post_output,
        *((tail_add_contract,) if tail_add_contract is not None else ()),
    )
    if (
        any(index is None for index in involved_indices)
        or len({id(op) for op in involved_ops}) != len(involved_ops)
        or any(index is None for index in dependency_tail_indices)
        or [int(index) for index in dependency_tail_indices if index is not None]
        != sorted(int(index) for index in dependency_tail_indices if index is not None)
        or int(pre_index)
        >= min(
            int(graph_index.operator_index(spatial.mean1) or -1),
            int(graph_index.operator_index(global_path.mean1) or -1),
        )
        or any(str(contract.tensor.dtype) != dtype for contract in contracts)
        or any(contract.tensor.quantization is not None for contract in contracts)
    ):
        return None

    axes_updates = _plan_spatial_axes_updates(
        model_ir,
        graph_index,
        spatial,
        public_names,
    )
    if axes_updates is None:
        return None
    coefficient_uses = [
        (spatial.scale_name, spatial.scale, spatial.scale_input_index),
        (global_path.scale_name, global_path.scale, global_path.scale_input_index),
    ]
    for coefficient in (gamma, beta):
        if coefficient.bypass_name is None:
            coefficient_uses.append(
                (
                    coefficient.coefficient_name,
                    coefficient.operator,
                    coefficient.input_index,
                )
            )
    coefficient_updates = plan_nhwc_coefficient_updates(
        model_ir,
        graph_index,
        coefficient_uses=tuple(coefficient_uses),
        dtype=dtype,
        channel_count=int(x.shape[1]),
        public_names=public_names,
    )
    if coefficient_updates is None:
        return None

    path_contracts = (
        spatial.mean1_contract,
        spatial.centered,
        spatial.squared,
        spatial.mean2_contract,
        spatial.factor_contract,
        spatial.add_epsilon_contract,
        spatial.sqrt_contract,
        spatial.div_contract,
        spatial.scaled,
        global_path.mean1_contract,
        global_path.centered,
        global_path.squared,
        global_path.mean2_contract,
        global_path.factor_contract,
        global_path.add_epsilon_contract,
        global_path.sqrt_contract,
        global_path.div_contract,
        global_path.scaled,
        blend_add_contract,
        blend_mul_contract,
        inst_output,
        *((tail_add_contract,) if tail_add_contract is not None else ()),
    )
    metadata_updates = tuple(
        TensorMetadataUpdate(
            contract,
            _permuted(contract.shape, _PERM_NCHW_TO_NHWC),
            _permuted(contract.signature, _PERM_NCHW_TO_NHWC),
        )
        for contract in path_contracts
    )
    channel_last_names = tuple(
        dict.fromkeys(
            [
                source_name,
                post_output_name,
                *(([residual_source_name]) if residual_source_name else []),
                *(contract.tensor.name for contract in path_contracts),
            ]
        )
    )
    return _DualStatsPlan(
        involved_ops=involved_ops,
        pre=pre,
        source_name=source_name,
        spatial=spatial,
        global_path=global_path,
        blend_add=blend_add,
        blend_add_contract=blend_add_contract,
        blend_mul=blend_mul,
        blend_mul_contract=blend_mul_contract,
        gamma=gamma,
        blend_bias=blend_bias,
        inst_output=inst_output,
        beta=beta,
        residual_pre=residual_pre,
        residual_source_name=residual_source_name,
        tail_add=tail_add,
        tail_add_contract=tail_add_contract,
        tail_add_residual_input_index=tail_add_residual_input_index,
        post=post,
        post_output_name=post_output_name,
        output_owner=output_owner,
        output_owner_old_name=output_owner_old_name,
        constant_updates=axes_updates + coefficient_updates,
        metadata_updates=metadata_updates,
        channel_last_names=channel_last_names,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _DualStatsPlan,
) -> bool:
    involved_indices = [graph_index.operator_index(op) for op in plan.involved_ops]
    remove_ops = [plan.pre, plan.post]
    if plan.residual_pre is not None:
        remove_ops.append(plan.residual_pre)
    for coefficient in (plan.gamma, plan.beta):
        if coefficient.bypass_reshape is not None:
            remove_ops.append(coefficient.bypass_reshape)
    remove_indices = [graph_index.operator_index(op) for op in remove_ops]
    clone_names = [
        update.clone_name
        for update in plan.constant_updates
        if update.clone_name is not None
    ]
    if (
        any(index is None for index in involved_indices)
        or len({int(index) for index in involved_indices if index is not None})
        != len(involved_indices)
        or any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or len(clone_names) != len(set(clone_names))
        or any(name in model_ir.tensors for name in clone_names)
        or any(
            update.clone_name is not None and update.clone is None
            for update in plan.constant_updates
        )
        or str(plan.output_owner.outputs[0]) != plan.output_owner_old_name
        or str(plan.post.outputs[0]) != plan.post_output_name
    ):
        return False
    for coefficient in (plan.gamma, plan.beta):
        if (
            coefficient.bypass_name is not None
            and str(coefficient.operator.inputs[coefficient.input_index])
            != coefficient.coefficient_name
        ):
            return False
    if plan.tail_add is not None and (
        plan.residual_source_name is None
        or plan.residual_pre is None
        or plan.tail_add_residual_input_index is None
        or int(plan.tail_add_residual_input_index) < 0
        or int(plan.tail_add_residual_input_index) >= len(plan.tail_add.inputs)
        or str(plan.tail_add.inputs[int(plan.tail_add_residual_input_index)])
        != str(plan.residual_pre.outputs[0])
    ):
        return False

    for update in plan.constant_updates:
        if not apply_constant_update(model_ir, graph_index, update):
            return False
    for coefficient in (plan.gamma, plan.beta):
        if coefficient.bypass_name is not None:
            _replace_operator_input_at(
                model_ir=model_ir,
                op=coefficient.operator,
                input_index=coefficient.input_index,
                new_input_name=coefficient.bypass_name,
                graph_index=graph_index,
            )
    for path in (plan.spatial, plan.global_path):
        _replace_operator_input_at(
            model_ir=model_ir,
            op=path.mean1,
            input_index=0,
            new_input_name=plan.source_name,
            graph_index=graph_index,
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=path.sub,
            input_index=path.sub_x_input_index,
            new_input_name=plan.source_name,
            graph_index=graph_index,
        )
    if plan.tail_add is not None:
        _replace_operator_input_at(
            model_ir=model_ir,
            op=plan.tail_add,
            input_index=int(plan.tail_add_residual_input_index or 0),
            new_input_name=str(plan.residual_source_name),
            graph_index=graph_index,
        )
    for update in plan.metadata_updates:
        update.contract.tensor.shape = list(update.shape)
        update.contract.tensor.shape_signature = list(update.signature)
    graph_index.remove_operators(
        [int(index) for index in remove_indices if index is not None]
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.output_owner,
        new_outputs=[plan.post_output_name],
        graph_index=graph_index,
    )
    hints = {
        str(value)
        for value in model_ir.metadata.get(
            "assume_channel_last_layout_tensor_names",
            [],
        )
        if str(value)
    }
    hints.update(plan.channel_last_names)
    model_ir.metadata["assume_channel_last_layout_tensor_names"] = sorted(hints)
    return True


def optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Lift a strict dual-stat normalization block and optional residual to NHWC."""

    if candidate is None:
        counts = Counter(str(operator.op_type) for operator in model_ir.operators)
        required = {
            "TRANSPOSE": 2,
            "MEAN": 4,
            "SUB": 2,
            "MUL": 7,
            "ADD": 4,
            "SQRT": 2,
            "DIV": 2,
        }
        if any(counts[op_type] < minimum for op_type, minimum in required.items()):
            return {_STATS_KEY: 0}
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
            for index in active_index.operator_indices("TRANSPOSE")
        ]
    )
    rewritten = 0
    for pre in candidates:
        if rewritten >= max(0, int(max_rewrites)):
            break
        pre_index = active_index.operator_index(pre)
        if pre_index is None:
            continue
        plan = _resolve_candidate(model_ir, active_index, pre_index)
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1
    if rewritten:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
