from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _normalize_squeeze_axes_for_rank,
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _valid_source,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _TensorContract,
    _constant_vector,
    _producer_is_valid,
    _tensor_contract,
    _unique_tensor_name,
)


_STATS_KEY = (
    "optimized_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains"
)
_SIDE_STATS_KEY = (
    "optimized_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains"
)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_PERM_CHW_TO_HWC = (1, 2, 0)
_FLOAT_DTYPES = {
    "FLOAT16": np.dtype(np.float16),
    "FLOAT32": np.dtype(np.float32),
}
_SIDE_ADAPTER_PERM_NAME = "__instancenorm_tail_nhwc_to_nchw_perm_rank4__"


@dataclass(frozen=True)
class _TensorMetadataUpdate:
    contract: _TensorContract
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _ConstantUse:
    operator: OperatorIR
    input_index: int


@dataclass(frozen=True)
class _ConstantUpdate:
    tensor: TensorIR
    data: np.ndarray
    uses: Tuple[_ConstantUse, ...]
    clone_name: Optional[str]
    clone: Optional[TensorIR]


@dataclass(frozen=True)
class _InstanceNormPrepostPlan:
    ordered_ops: Tuple[OperatorIR, ...]
    pre: OperatorIR
    squeeze: OperatorIR
    reshape: OperatorIR
    reshape_options: Dict[str, object]
    source_name: str
    mean1: OperatorIR
    sub: OperatorIR
    add_bias: OperatorIR
    post: OperatorIR
    post_output_name: str
    tail_mode: str
    side_squeeze: Optional[OperatorIR]
    side_adapter_permutation: Optional[TensorIR]
    constant_updates: Tuple[_ConstantUpdate, ...]
    metadata_updates: Tuple[_TensorMetadataUpdate, ...]
    channel_last_names: Tuple[str, ...]


def _contract(
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


def _normalized_axes(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    rank: int,
    size: int,
    public_inputs: set[str],
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


def _squeeze_axis_zero(
    squeeze: OperatorIR,
    source: _TensorContract,
    output: _TensorContract,
) -> bool:
    options = dict(squeeze.options) if isinstance(squeeze.options, dict) else {}
    if "squeezeDims" in options:
        try:
            values = [
                int(value)
                for value in np.asarray(options["squeezeDims"])
                .reshape(-1)
                .tolist()
            ]
        except Exception:
            return False
        axes = _normalize_squeeze_axes_for_rank(values, 4)
        if axes != [0]:
            return False
    elif sum(int(value) == 1 for value in source.shape) != 1:
        return False
    return bool(
        source.shape[0] == 1
        and source.signature[0] == 1
        and output.shape == source.shape[1:]
        and output.signature == source.signature[1:]
    )


def _float_constant(
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
    expected_dtype = _FLOAT_DTYPES.get(str(dtype))
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
    if nonnegative and (
        data.size != 1 or float(data.reshape(-1)[0]) < 0.0
    ):
        return None
    return data


def _plan_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    data: np.ndarray,
    uses: Sequence[_ConstantUse],
    suffix: str,
    public_names: set[str],
) -> Optional[_ConstantUpdate]:
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
    return _ConstantUpdate(
        tensor=tensor,
        data=np.asarray(replacement),
        uses=tuple(uses),
        clone_name=clone_name,
        clone=clone,
    )


def _binary_other_input(
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


def _sole_consumer(
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Optional[Tuple[int, OperatorIR]]:
    users = graph_index.consumer_indices(str(name))
    if len(users) != 1:
        return None
    index = int(users[0])
    return index, graph_index.model_ir.operators[index]


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
    *,
    tail_mode: str,
) -> Optional[_InstanceNormPrepostPlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    public_names = public_inputs | public_outputs
    pre = model_ir.operators[int(pre_index)]
    if (
        len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or _constant_vector(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NHWC_TO_NCHW
        or str(pre.inputs[1]) in public_outputs
    ):
        return None
    source_name = str(pre.inputs[0])
    pre_output_name = str(pre.outputs[0])
    source = _tensor_contract(model_ir, source_name, 4)
    if source is None:
        return None
    pre_output = _contract(
        model_ir,
        pre_output_name,
        4,
        tuple(source.shape[index] for index in _PERM_NHWC_TO_NCHW),
        tuple(source.signature[index] for index in _PERM_NHWC_TO_NCHW),
    )
    if (
        pre_output is None
        or source_name in public_outputs
        or pre_output_name in public_names
        or not _valid_source(
            graph_index,
            source,
            source_name,
            int(pre_index),
            public_inputs,
        )
        or not _producer_is_valid(graph_index, pre_output_name, int(pre_index))
    ):
        return None

    squeeze_match = _sole_consumer(graph_index, pre_output_name)
    if squeeze_match is None:
        return None
    squeeze_index, squeeze = squeeze_match
    squeeze_output_name = (
        str(squeeze.outputs[0]) if len(squeeze.outputs) == 1 else ""
    )
    squeeze_output = _tensor_contract(model_ir, squeeze_output_name, 3)
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or len(squeeze.outputs) != 1
        or str(squeeze.inputs[0]) != pre_output_name
        or squeeze_output is None
        or squeeze_output_name in public_names
        or not _producer_is_valid(
            graph_index,
            squeeze_output_name,
            squeeze_index,
        )
        or not _squeeze_axis_zero(squeeze, pre_output, squeeze_output)
    ):
        return None

    reshape_match = _sole_consumer(graph_index, squeeze_output_name)
    if reshape_match is None:
        return None
    reshape_index, reshape = reshape_match
    x_name = str(reshape.outputs[0]) if len(reshape.outputs) == 1 else ""
    x = _contract(
        model_ir,
        x_name,
        4,
        pre_output.shape,
        pre_output.signature,
    )
    if (
        str(reshape.op_type) != "RESHAPE"
        or len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != squeeze_output_name
        or x is None
        or x_name in public_names
        or not _producer_is_valid(graph_index, x_name, reshape_index)
    ):
        return None
    reshape_shape_name = str(reshape.inputs[1])
    if _constant_vector(
        model_ir,
        graph_index,
        reshape_shape_name,
        4,
        public_inputs,
    ) != x.shape or reshape_shape_name in public_outputs:
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
    ]
    if len(mean_matches) != 1 or len(sub_matches) != 1:
        return None
    mean1_index = int(mean_matches[0])
    sub_index = int(sub_matches[0])
    mean1 = model_ir.operators[mean1_index]
    sub = model_ir.operators[sub_index]
    mean1_options = dict(mean1.options) if isinstance(mean1.options, dict) else {}
    mean1_name = str(mean1.outputs[0]) if len(mean1.outputs) == 1 else ""
    reduced_shape = (
        pre_output.shape[0],
        pre_output.shape[1],
        1,
        1,
    )
    reduced_signature = (
        pre_output.signature[0],
        pre_output.signature[1],
        1,
        1,
    )
    mean1_contract = _contract(
        model_ir,
        mean1_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    mean1_axes_name = str(mean1.inputs[1])
    if (
        len(mean1.outputs) != 1
        or not bool(mean1_options.get("keepDims", False))
        or _normalized_axes(
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
        or [str(value) for value in sub.inputs] != [x_name, mean1_name]
        or len(sub.outputs) != 1
    ):
        return None
    centered_name = str(sub.outputs[0])
    centered = _contract(
        model_ir,
        centered_name,
        4,
        pre_output.shape,
        pre_output.signature,
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
    squared = _contract(
        model_ir,
        squared_name,
        4,
        pre_output.shape,
        pre_output.signature,
    )
    if (
        squared is None
        or squared_name in public_names
        or not _producer_is_valid(graph_index, squared_name, square_index)
    ):
        return None

    mean2_match = _sole_consumer(graph_index, squared_name)
    if mean2_match is None:
        return None
    mean2_index, mean2 = mean2_match
    mean2_name = str(mean2.outputs[0]) if len(mean2.outputs) == 1 else ""
    mean2_contract = _contract(
        model_ir,
        mean2_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    mean2_options = dict(mean2.options) if isinstance(mean2.options, dict) else {}
    mean2_axes_name = str(mean2.inputs[1]) if len(mean2.inputs) == 2 else ""
    if (
        str(mean2.op_type) != "MEAN"
        or len(mean2.inputs) != 2
        or len(mean2.outputs) != 1
        or str(mean2.inputs[0]) != squared_name
        or not bool(mean2_options.get("keepDims", False))
        or _normalized_axes(
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

    add_epsilon_match = _sole_consumer(graph_index, mean2_name)
    if add_epsilon_match is None:
        return None
    add_epsilon_index, add_epsilon = add_epsilon_match
    epsilon_match = _binary_other_input(add_epsilon, mean2_name)
    add_epsilon_name = (
        str(add_epsilon.outputs[0]) if len(add_epsilon.outputs) == 1 else ""
    )
    add_epsilon_contract = _contract(
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

    sqrt_match = _sole_consumer(graph_index, add_epsilon_name)
    if sqrt_match is None:
        return None
    sqrt_index, sqrt = sqrt_match
    sqrt_name = str(sqrt.outputs[0]) if len(sqrt.outputs) == 1 else ""
    sqrt_contract = _contract(
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

    div_match = _sole_consumer(graph_index, sqrt_name)
    if div_match is None:
        return None
    div_index, div = div_match
    div_name = str(div.outputs[0]) if len(div.outputs) == 1 else ""
    div_contract = _contract(
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
    norm_other = _binary_other_input(norm, centered_name)
    normalized_name = str(norm.outputs[0])
    normalized = _contract(
        model_ir,
        normalized_name,
        4,
        pre_output.shape,
        pre_output.signature,
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

    scale_match = _sole_consumer(graph_index, normalized_name)
    if scale_match is None:
        return None
    scale_index, scale = scale_match
    scale_constant_match = _binary_other_input(scale, normalized_name)
    scaled_name = str(scale.outputs[0]) if len(scale.outputs) == 1 else ""
    scaled = _contract(
        model_ir,
        scaled_name,
        4,
        pre_output.shape,
        pre_output.signature,
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

    bias_match = _sole_consumer(graph_index, scaled_name)
    if bias_match is None:
        return None
    bias_index, add_bias = bias_match
    bias_constant_match = _binary_other_input(add_bias, scaled_name)
    inst_output_name = (
        str(add_bias.outputs[0]) if len(add_bias.outputs) == 1 else ""
    )
    inst_output = _contract(
        model_ir,
        inst_output_name,
        4,
        pre_output.shape,
        pre_output.signature,
    )
    if (
        str(add_bias.op_type) != "ADD"
        or bias_constant_match is None
        or len(add_bias.outputs) != 1
        or inst_output is None
        or inst_output_name in public_names
        or not _producer_is_valid(graph_index, inst_output_name, bias_index)
    ):
        return None
    bias_name, bias_input_index = bias_constant_match
    if bias_name == scale_name:
        return None

    if tail_mode not in {"direct", "side_squeeze"}:
        return None
    inst_output_users = sorted(
        set(graph_index.consumer_indices(inst_output_name))
    )
    post_matches = []
    side_matches = []
    for user_index in inst_output_users:
        user = model_ir.operators[int(user_index)]
        if (
            str(user.op_type) == "TRANSPOSE"
            and len(user.inputs) == 2
            and len(user.outputs) == 1
            and str(user.inputs[0]) == inst_output_name
            and _constant_vector(
                model_ir,
                graph_index,
                str(user.inputs[1]),
                4,
                public_inputs,
            )
            == _PERM_NCHW_TO_NHWC
            and str(user.inputs[1]) not in public_outputs
        ):
            post_matches.append((int(user_index), user))
        if (
            str(user.op_type) == "SQUEEZE"
            and len(user.inputs) == 1
            and len(user.outputs) == 1
            and str(user.inputs[0]) == inst_output_name
        ):
            side_matches.append((int(user_index), user))
    if len(post_matches) != 1:
        return None
    post_index, post = post_matches[0]
    side_squeeze: Optional[OperatorIR] = None
    side_contract: Optional[_TensorContract] = None
    side_adapter_permutation: Optional[TensorIR] = None
    if tail_mode == "direct":
        if inst_output_users != [post_index]:
            return None
    else:
        if len(side_matches) != 1:
            return None
        side_index, side_squeeze = side_matches[0]
        if inst_output_users != sorted([post_index, side_index]):
            return None
        side_output_name = str(side_squeeze.outputs[0])
        side_contract = _contract(
            model_ir,
            side_output_name,
            3,
            inst_output.shape[1:],
            inst_output.signature[1:],
        )
        if (
            side_contract is None
            or side_output_name in public_inputs
            or not _producer_is_valid(
                graph_index,
                side_output_name,
                side_index,
            )
            or not _squeeze_axis_zero(
                side_squeeze,
                inst_output,
                side_contract,
            )
            or any(
                int(consumer_index) <= side_index
                for consumer_index in graph_index.consumer_indices(
                    side_output_name
                )
            )
        ):
            return None
        existing_adapter_perm = model_ir.tensors.get(_SIDE_ADAPTER_PERM_NAME)
        if existing_adapter_perm is None:
            side_adapter_permutation = TensorIR(
                name=_SIDE_ADAPTER_PERM_NAME,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray(_PERM_NHWC_TO_NCHW, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
        elif (
            _constant_vector(
                model_ir,
                graph_index,
                _SIDE_ADAPTER_PERM_NAME,
                4,
                public_inputs,
            )
            != _PERM_NHWC_TO_NCHW
            or _SIDE_ADAPTER_PERM_NAME in public_outputs
            or existing_adapter_perm.quantization is not None
        ):
            return None
    post_output_name = str(post.outputs[0]) if len(post.outputs) == 1 else ""
    post_output = _contract(
        model_ir,
        post_output_name,
        4,
        source.shape,
        source.signature,
    )
    if (
        post_output is None
        or post_output_name in public_names
        or not _producer_is_valid(graph_index, post_output_name, post_index)
        or any(
            int(consumer_index) <= post_index
            for consumer_index in graph_index.consumer_indices(post_output_name)
        )
    ):
        return None

    core_ops = (
        pre,
        squeeze,
        reshape,
        mean1,
        sub,
        square,
        mean2,
        add_epsilon,
        sqrt,
        div,
        norm,
        scale,
        add_bias,
    )
    tail_ops = [post]
    if side_squeeze is not None:
        tail_ops.append(side_squeeze)
    ordered_ops = core_ops + tuple(
        sorted(
            tail_ops,
            key=lambda operator: int(
                graph_index.operator_index(operator)
                if graph_index.operator_index(operator) is not None
                else len(model_ir.operators)
            ),
        )
    )
    ordered_indices = [
        graph_index.operator_index(operator) for operator in ordered_ops
    ]
    if (
        any(index is None for index in ordered_indices)
        or [int(index) for index in ordered_indices if index is not None]
        != sorted(int(index) for index in ordered_indices if index is not None)
        or len({id(operator) for operator in ordered_ops}) != len(ordered_ops)
    ):
        return None

    contracts = (
        source,
        pre_output,
        squeeze_output,
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
        inst_output,
        post_output,
    ) + ((side_contract,) if side_contract is not None else ())
    dtype = str(source.tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(str(contract.tensor.dtype) != dtype for contract in contracts)
        or any(contract.tensor.quantization is not None for contract in contracts)
    ):
        return None
    channel_count = int(pre_output.shape[1])
    old_coefficient_shape = (1, channel_count, 1, 1)
    scale_data = _float_constant(
        model_ir,
        graph_index,
        scale_name,
        dtype,
        shape=old_coefficient_shape,
    )
    bias_data = _float_constant(
        model_ir,
        graph_index,
        bias_name,
        dtype,
        shape=old_coefficient_shape,
    )
    if (
        _float_constant(
            model_ir,
            graph_index,
            epsilon_name,
            dtype,
            nonnegative=True,
        )
        is None
        or _float_constant(
            model_ir,
            graph_index,
            one_name,
            dtype,
            value=1.0,
        )
        is None
        or scale_data is None
        or bias_data is None
    ):
        return None

    constant_updates = []
    reshape_update = _plan_constant_update(
        model_ir,
        graph_index,
        reshape_shape_name,
        np.asarray(source.shape, dtype=np.asarray(
            model_ir.tensors[reshape_shape_name].data
        ).dtype),
        (_ConstantUse(reshape, 1),),
        "nhwc_shape",
        public_names,
    )
    if reshape_update is None:
        return None
    constant_updates.append(reshape_update)
    if mean1_axes_name == mean2_axes_name:
        axes_updates = (
            (
                mean1_axes_name,
                (_ConstantUse(mean1, 1), _ConstantUse(mean2, 1)),
            ),
        )
    else:
        axes_updates = (
            (mean1_axes_name, (_ConstantUse(mean1, 1),)),
            (mean2_axes_name, (_ConstantUse(mean2, 1),)),
        )
    for axes_name, uses in axes_updates:
        update = _plan_constant_update(
            model_ir,
            graph_index,
            axes_name,
            np.asarray(
                [1, 2],
                dtype=np.asarray(model_ir.tensors[axes_name].data).dtype,
            ),
            uses,
            "nhwc_axes",
            public_names,
        )
        if update is None:
            return None
        constant_updates.append(update)
    for name, data, use, suffix in (
        (
            scale_name,
            np.transpose(scale_data, _PERM_NCHW_TO_NHWC),
            _ConstantUse(scale, scale_input_index),
            "nhwc_scale",
        ),
        (
            bias_name,
            np.transpose(bias_data, _PERM_NCHW_TO_NHWC),
            _ConstantUse(add_bias, bias_input_index),
            "nhwc_bias",
        ),
    ):
        update = _plan_constant_update(
            model_ir,
            graph_index,
            name,
            data,
            (use,),
            suffix,
            public_names,
        )
        if update is None:
            return None
        constant_updates.append(update)

    full_contracts = (
        x,
        centered,
        squared,
        normalized,
        scaled,
    ) + ((inst_output,) if tail_mode == "direct" else ())
    reduced_contracts = (
        mean1_contract,
        mean2_contract,
        add_epsilon_contract,
        sqrt_contract,
        div_contract,
    )
    metadata_updates = [
        _TensorMetadataUpdate(
            squeeze_output,
            tuple(squeeze_output.shape[index] for index in _PERM_CHW_TO_HWC),
            tuple(
                squeeze_output.signature[index]
                for index in _PERM_CHW_TO_HWC
            ),
        )
    ]
    metadata_updates.extend(
        _TensorMetadataUpdate(
            contract,
            tuple(contract.shape[index] for index in _PERM_NCHW_TO_NHWC),
            tuple(
                contract.signature[index]
                for index in _PERM_NCHW_TO_NHWC
            ),
        )
        for contract in full_contracts + reduced_contracts
    )
    channel_last_names = tuple(
        dict.fromkeys(
            [
                squeeze_output_name,
                *(contract.tensor.name for contract in full_contracts),
                *(contract.tensor.name for contract in reduced_contracts),
                post_output_name,
            ]
        )
    )
    reshape_options = (
        dict(reshape.options) if isinstance(reshape.options, dict) else {}
    )
    reshape_options["newShape"] = [int(value) for value in source.shape]
    return _InstanceNormPrepostPlan(
        ordered_ops=ordered_ops,
        pre=pre,
        squeeze=squeeze,
        reshape=reshape,
        reshape_options=reshape_options,
        source_name=source_name,
        mean1=mean1,
        sub=sub,
        add_bias=add_bias,
        post=post,
        post_output_name=post_output_name,
        tail_mode=tail_mode,
        side_squeeze=side_squeeze,
        side_adapter_permutation=side_adapter_permutation,
        constant_updates=tuple(constant_updates),
        metadata_updates=tuple(metadata_updates),
        channel_last_names=channel_last_names,
    )


def _apply_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    update: _ConstantUpdate,
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


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _InstanceNormPrepostPlan,
) -> bool:
    indices = [graph_index.operator_index(operator) for operator in plan.ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != len(resolved):
        return False
    clone_names = [
        update.clone_name
        for update in plan.constant_updates
        if update.clone_name is not None
    ]
    if (
        len(clone_names) != len(set(clone_names))
        or any(name in model_ir.tensors for name in clone_names)
        or any(
            update.clone_name is not None and update.clone is None
            for update in plan.constant_updates
        )
        or (
            plan.tail_mode == "side_squeeze"
            and plan.side_squeeze is None
        )
        or (
            plan.side_adapter_permutation is not None
            and _SIDE_ADAPTER_PERM_NAME in model_ir.tensors
        )
    ):
        return False
    for update in plan.constant_updates:
        if not _apply_constant_update(model_ir, graph_index, update):
            return False

    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.mean1,
        input_index=0,
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.squeeze,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    plan.reshape.options = dict(plan.reshape_options)
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.sub,
        input_index=0,
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )
    for update in plan.metadata_updates:
        update.contract.tensor.shape = list(update.shape)
        update.contract.tensor.shape_signature = list(update.signature)
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.add_bias,
        new_outputs=[plan.post_output_name],
        graph_index=graph_index,
    )
    remove_indices = [
        graph_index.operator_index(plan.pre),
        graph_index.operator_index(plan.post),
    ]
    if any(index is None for index in remove_indices):
        return False
    graph_index.remove_operators(
        [int(index) for index in remove_indices if index is not None]
    )
    if plan.tail_mode == "side_squeeze":
        assert plan.side_squeeze is not None
        if plan.side_adapter_permutation is not None:
            model_ir.tensors[_SIDE_ADAPTER_PERM_NAME] = (
                plan.side_adapter_permutation
            )
        side_index = graph_index.operator_index(plan.side_squeeze)
        if side_index is None:
            return False
        graph_index.insert_operator(
            int(side_index),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[plan.post_output_name, _SIDE_ADAPTER_PERM_NAME],
                outputs=[str(plan.side_squeeze.inputs[0])],
            ),
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


def _run_indexed_instance_norm_prepost_tail(
    model_ir: ModelIR,
    *,
    tail_mode: str,
    stats_key: str,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    if candidate is None:
        counts = Counter(str(operator.op_type) for operator in model_ir.operators)
        required = {
            "TRANSPOSE": 2,
            "SQUEEZE": 2 if tail_mode == "side_squeeze" else 1,
            "RESHAPE": 1,
            "MEAN": 2,
            "SUB": 1,
            "MUL": 3,
            "ADD": 2,
            "SQRT": 1,
            "DIV": 1,
        }
        if any(
            counts[op_type] < minimum
            for op_type, minimum in required.items()
        ):
            return {stats_key: 0}
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
        plan = _resolve_candidate(
            model_ir,
            active_index,
            pre_index,
            tail_mode=tail_mode,
        )
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1
    if rewritten:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {stats_key: int(rewritten)}


def _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _run_indexed_instance_norm_prepost_tail(
        model_ir,
        tail_mode="direct",
        stats_key=_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _run_indexed_instance_norm_prepost_tail(
        model_ir,
        tail_mode="side_squeeze",
        stats_key=_SIDE_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )
