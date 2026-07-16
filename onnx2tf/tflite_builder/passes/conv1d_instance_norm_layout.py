from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _ConstantUpdate,
    _TensorContract,
    _UNARY_OPS,
    _apply_constant_update,
    _constant_vector,
    _plan_constant_update,
    _producer_is_valid,
    _quantization_contract,
    _squeeze_axis,
    _tensor_contract,
)


_STATS_KEY = (
    "optimized_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains"
)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)


@dataclass(frozen=True)
class _InstanceNormRewritePlan:
    ordered_ops: Tuple[OperatorIR, ...]
    squeeze: OperatorIR
    squeeze_options: Dict[str, Any]
    squeeze_tensor: TensorIR
    reshape2: OperatorIR
    reshape2_options: Dict[str, Any]
    reshape2_shape_update: Optional[_ConstantUpdate]
    reshape2_tensor: TensorIR
    unary_tensor: TensorIR
    expand: OperatorIR
    expand_axis_update: _ConstantUpdate
    expand_tensor: TensorIR
    source_name: str
    post_output_name: str
    rank3_shape: Tuple[int, ...]
    rank3_signature: Tuple[int, ...]
    rank4_shape: Tuple[int, ...]
    rank4_signature: Tuple[int, ...]


@dataclass(frozen=True)
class _FlatInstanceNormPrefixPlan:
    public_inputs: frozenset[str]
    public_outputs: frozenset[str]
    ordered_ops: Tuple[OperatorIR, ...]
    ordered_indices: Tuple[int, ...]
    pre: OperatorIR
    squeeze: OperatorIR
    reshape2: OperatorIR
    source_name: str
    reshape2_name: str
    source_contract: _TensorContract
    pre_contract: _TensorContract
    squeeze_contract: _TensorContract
    reshape2_contract: _TensorContract
    data_contracts: Tuple[_TensorContract, ...]
    data_names: Tuple[str, ...]
    n: int
    c: int
    w: int
    n_signature: int
    c_signature: int
    w_signature: int


def _produced_contract(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    operator_index: int,
    rank: int,
    public_inputs: set[str],
    public_outputs: set[str],
    *,
    allow_public_output: bool = False,
) -> Optional[Tuple[str, _TensorContract]]:
    if len(operator.outputs) != 1:
        return None
    name = str(operator.outputs[0])
    if (
        not name
        or name in public_inputs
        or (not allow_public_output and name in public_outputs)
        or not _producer_is_valid(graph_index, name, operator_index)
    ):
        return None
    contract = _tensor_contract(model_ir, name, rank)
    if contract is None:
        return None
    return name, contract


def _constant_scalar(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    public_inputs: set[str],
) -> Optional[float]:
    name = str(tensor_name)
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or name in public_inputs
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or str(tensor.dtype) not in {"FLOAT16", "FLOAT32", "FLOAT64"}
    ):
        return None
    try:
        data = np.asarray(tensor.data)
        if data.size != 1 or data.dtype.kind != "f":
            return None
        value = float(data.reshape(-1)[0])
    except Exception:
        return None
    return value if np.isfinite(value) else None


def _binary_other_input(operator: OperatorIR, tensor_name: str) -> Optional[str]:
    if len(operator.inputs) != 2:
        return None
    inputs = [str(value) for value in operator.inputs]
    if inputs.count(str(tensor_name)) != 1:
        return None
    return inputs[1] if inputs[0] == str(tensor_name) else inputs[0]


def _unique_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    op_type: str,
) -> Optional[Tuple[int, OperatorIR]]:
    candidates = []
    for index in sorted(set(graph_index.consumer_indices(str(tensor_name)))):
        operator = model_ir.operators[int(index)]
        if str(operator.op_type) == str(op_type):
            candidates.append((int(index), operator))
    return candidates[0] if len(candidates) == 1 else None


def _mean_contract(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    operator_index: int,
    input_name: str,
    expected_shape: Tuple[int, ...],
    expected_signature: Tuple[int, ...],
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[Tuple[str, _TensorContract]]:
    if (
        len(operator.inputs) != 2
        or str(operator.inputs[0]) != str(input_name)
        or _constant_vector(
            model_ir,
            graph_index,
            str(operator.inputs[1]),
            1,
            public_inputs,
        )
        != (2,)
    ):
        return None
    options = dict(operator.options) if isinstance(operator.options, dict) else {}
    if not bool(options.get("keepDims", options.get("keep_dims", False))):
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        operator,
        operator_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    name, contract = produced
    if contract.shape != expected_shape or contract.signature != expected_signature:
        return None
    return name, contract


def _resolve_flattened_instance_norm_prefix(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_FlatInstanceNormPrefixPlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
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
    ):
        return None

    source_name = str(pre.inputs[0])
    if not source_name:
        return None
    source_contract = _tensor_contract(model_ir, source_name, 4)
    if (
        source_contract is None
        or source_contract.shape[1] != 1
        or source_contract.signature[1] != 1
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

    produced = _produced_contract(
        model_ir,
        graph_index,
        pre,
        pre_index,
        4,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    pre_output_name, pre_contract = produced
    if (
        pre_contract.shape
        != tuple(source_contract.shape[index] for index in _PERM_NHWC_TO_NCHW)
        or pre_contract.signature
        != tuple(source_contract.signature[index] for index in _PERM_NHWC_TO_NCHW)
    ):
        return None

    pre_users = graph_index.consumer_indices(pre_output_name)
    if len(pre_users) != 1 or int(pre_users[0]) <= int(pre_index):
        return None
    squeeze_index = int(pre_users[0])
    squeeze = model_ir.operators[squeeze_index]
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or str(squeeze.inputs[0]) != pre_output_name
    ):
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        squeeze,
        squeeze_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    squeeze_output_name, squeeze_contract = produced
    squeeze_axis = _squeeze_axis(squeeze, pre_contract, squeeze_contract)
    if squeeze_axis != 2:
        return None

    squeeze_users = graph_index.consumer_indices(squeeze_output_name)
    if len(squeeze_users) != 1 or int(squeeze_users[0]) <= squeeze_index:
        return None
    reshape1_index = int(squeeze_users[0])
    reshape1 = model_ir.operators[reshape1_index]
    if (
        str(reshape1.op_type) != "RESHAPE"
        or len(reshape1.inputs) != 2
        or str(reshape1.inputs[0]) != squeeze_output_name
    ):
        return None

    n, c, _, w = pre_contract.shape
    n_sig, c_sig, _, w_sig = pre_contract.signature
    flat_shape = (int(n), 1, int(c) * int(w))
    flat_signature = (
        int(n_sig),
        1,
        -1 if int(c_sig) == -1 or int(w_sig) == -1 else int(c) * int(w),
    )
    if (
        _constant_vector(
            model_ir,
            graph_index,
            str(reshape1.inputs[1]),
            3,
            public_inputs,
        )
        != flat_shape
    ):
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        reshape1,
        reshape1_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    flat_name, flat_contract = produced
    if flat_contract.shape != flat_shape or flat_contract.signature != flat_signature:
        return None
    reshape1_options = (
        dict(reshape1.options) if isinstance(reshape1.options, dict) else {}
    )
    for key in ("newShape", "onnxRawNewShape"):
        value = reshape1_options.get(key)
        if isinstance(value, list) and tuple(int(item) for item in value) != flat_shape:
            return None

    flat_users = sorted(set(graph_index.consumer_indices(flat_name)))
    mean1_match = _unique_consumer(model_ir, graph_index, flat_name, "MEAN")
    if mean1_match is None:
        return None
    mean1_index, mean1 = mean1_match
    mean_shape = (int(n), 1, 1)
    mean_signature = (int(n_sig), 1, 1)
    mean1_result = _mean_contract(
        model_ir,
        graph_index,
        mean1,
        mean1_index,
        flat_name,
        mean_shape,
        mean_signature,
        public_inputs,
        public_outputs,
    )
    if mean1_result is None:
        return None
    mean1_name, mean1_contract = mean1_result

    sub_match = _unique_consumer(model_ir, graph_index, flat_name, "SUB")
    if sub_match is None:
        return None
    sub_index, sub = sub_match
    if [str(value) for value in sub.inputs] != [flat_name, mean1_name]:
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        sub,
        sub_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    centered_name, centered_contract = produced
    if (
        centered_contract.shape != flat_shape
        or centered_contract.signature != flat_signature
        or flat_users != sorted([mean1_index, sub_index])
        or graph_index.consumer_indices(mean1_name) != [sub_index]
    ):
        return None

    centered_users = graph_index.consumer_indices(centered_name)
    square_candidates = []
    for index in sorted(set(centered_users)):
        operator = model_ir.operators[int(index)]
        if (
            str(operator.op_type) == "MUL"
            and [str(value) for value in operator.inputs]
            == [centered_name, centered_name]
        ):
            square_candidates.append((int(index), operator))
    if len(square_candidates) != 1:
        return None
    square_index, square = square_candidates[0]
    produced = _produced_contract(
        model_ir,
        graph_index,
        square,
        square_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    square_name, square_contract = produced
    if square_contract.shape != flat_shape or square_contract.signature != flat_signature:
        return None

    mean2_users = graph_index.consumer_indices(square_name)
    if len(mean2_users) != 1:
        return None
    mean2_index = int(mean2_users[0])
    mean2 = model_ir.operators[mean2_index]
    if str(mean2.op_type) != "MEAN":
        return None
    mean2_result = _mean_contract(
        model_ir,
        graph_index,
        mean2,
        mean2_index,
        square_name,
        mean_shape,
        mean_signature,
        public_inputs,
        public_outputs,
    )
    if mean2_result is None:
        return None
    mean2_name, mean2_contract = mean2_result

    add_users = graph_index.consumer_indices(mean2_name)
    if len(add_users) != 1:
        return None
    add_epsilon_index = int(add_users[0])
    add_epsilon = model_ir.operators[add_epsilon_index]
    if str(add_epsilon.op_type) != "ADD":
        return None
    epsilon_name = _binary_other_input(add_epsilon, mean2_name)
    epsilon = (
        None
        if epsilon_name is None
        else _constant_scalar(model_ir, graph_index, epsilon_name, public_inputs)
    )
    if epsilon is None or epsilon < 0.0:
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        add_epsilon,
        add_epsilon_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    variance_name, variance_contract = produced
    if variance_contract.shape != mean_shape or variance_contract.signature != mean_signature:
        return None

    sqrt_users = graph_index.consumer_indices(variance_name)
    if len(sqrt_users) != 1:
        return None
    sqrt_index = int(sqrt_users[0])
    sqrt = model_ir.operators[sqrt_index]
    if (
        str(sqrt.op_type) != "SQRT"
        or len(sqrt.inputs) != 1
        or str(sqrt.inputs[0]) != variance_name
    ):
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        sqrt,
        sqrt_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    standard_deviation_name, standard_deviation_contract = produced
    if (
        standard_deviation_contract.shape != mean_shape
        or standard_deviation_contract.signature != mean_signature
    ):
        return None

    div_users = graph_index.consumer_indices(standard_deviation_name)
    if len(div_users) != 1:
        return None
    div_index = int(div_users[0])
    div = model_ir.operators[div_index]
    if (
        str(div.op_type) != "DIV"
        or len(div.inputs) != 2
        or str(div.inputs[1]) != standard_deviation_name
    ):
        return None
    one = _constant_scalar(
        model_ir,
        graph_index,
        str(div.inputs[0]),
        public_inputs,
    )
    if one != 1.0:
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        div,
        div_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    inverse_name, inverse_contract = produced
    if inverse_contract.shape != mean_shape or inverse_contract.signature != mean_signature:
        return None

    norm_candidates = []
    for index in sorted(set(centered_users)):
        operator = model_ir.operators[int(index)]
        if str(operator.op_type) != "MUL" or len(operator.inputs) != 2:
            continue
        if Counter(str(value) for value in operator.inputs) == Counter(
            (centered_name, inverse_name)
        ):
            norm_candidates.append((int(index), operator))
    if len(norm_candidates) != 1:
        return None
    norm_index, norm = norm_candidates[0]
    if Counter(centered_users) != Counter(
        {square_index: 2, norm_index: 1}
    ) or graph_index.consumer_indices(inverse_name) != [norm_index]:
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        norm,
        norm_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    norm_name, norm_contract = produced
    if norm_contract.shape != flat_shape or norm_contract.signature != flat_signature:
        return None

    scale_users = graph_index.consumer_indices(norm_name)
    if len(scale_users) != 1:
        return None
    scale_index = int(scale_users[0])
    scale = model_ir.operators[scale_index]
    if str(scale.op_type) != "MUL":
        return None
    gamma_name = _binary_other_input(scale, norm_name)
    gamma = (
        None
        if gamma_name is None
        else _constant_scalar(model_ir, graph_index, gamma_name, public_inputs)
    )
    if gamma is None:
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        scale,
        scale_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    scaled_name, scaled_contract = produced
    if scaled_contract.shape != flat_shape or scaled_contract.signature != flat_signature:
        return None

    bias_users = graph_index.consumer_indices(scaled_name)
    if len(bias_users) != 1:
        return None
    bias_index = int(bias_users[0])
    bias = model_ir.operators[bias_index]
    if str(bias.op_type) != "ADD":
        return None
    beta_name = _binary_other_input(bias, scaled_name)
    beta = (
        None
        if beta_name is None
        else _constant_scalar(model_ir, graph_index, beta_name, public_inputs)
    )
    if beta is None:
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        bias,
        bias_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    biased_name, biased_contract = produced
    if biased_contract.shape != flat_shape or biased_contract.signature != flat_signature:
        return None

    reshape2_users = graph_index.consumer_indices(biased_name)
    if len(reshape2_users) != 1:
        return None
    reshape2_index = int(reshape2_users[0])
    reshape2 = model_ir.operators[reshape2_index]
    if (
        str(reshape2.op_type) != "RESHAPE"
        or len(reshape2.inputs) != 2
        or str(reshape2.inputs[0]) != biased_name
        or _constant_vector(
            model_ir,
            graph_index,
            str(reshape2.inputs[1]),
            3,
            public_inputs,
        )
        != squeeze_contract.shape
    ):
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        reshape2,
        reshape2_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    reshape2_name, reshape2_contract = produced
    if (
        reshape2_contract.shape != squeeze_contract.shape
        or reshape2_contract.signature != squeeze_contract.signature
    ):
        return None

    ordered_indices = (
        pre_index,
        squeeze_index,
        reshape1_index,
        mean1_index,
        sub_index,
        square_index,
        mean2_index,
        add_epsilon_index,
        sqrt_index,
        div_index,
        norm_index,
        scale_index,
        bias_index,
        reshape2_index,
    )
    if (
        list(ordered_indices) != sorted(ordered_indices)
        or len(set(ordered_indices)) != len(ordered_indices)
    ):
        return None

    data_contracts = (
        source_contract,
        pre_contract,
        squeeze_contract,
        flat_contract,
        mean1_contract,
        centered_contract,
        square_contract,
        mean2_contract,
        variance_contract,
        standard_deviation_contract,
        inverse_contract,
        norm_contract,
        scaled_contract,
        biased_contract,
        reshape2_contract,
    )
    if (
        len({str(contract.tensor.dtype) for contract in data_contracts}) != 1
        or str(source_contract.tensor.dtype) not in {"FLOAT16", "FLOAT32"}
        or any(contract.tensor.quantization is not None for contract in data_contracts)
    ):
        return None

    data_names = (
        source_name,
        pre_output_name,
        squeeze_output_name,
        flat_name,
        mean1_name,
        centered_name,
        square_name,
        mean2_name,
        variance_name,
        standard_deviation_name,
        inverse_name,
        norm_name,
        scaled_name,
        biased_name,
        reshape2_name,
    )
    if len(data_names) != len(set(data_names)):
        return None

    return _FlatInstanceNormPrefixPlan(
        public_inputs=frozenset(public_inputs),
        public_outputs=frozenset(public_outputs),
        ordered_ops=tuple(model_ir.operators[index] for index in ordered_indices),
        ordered_indices=tuple(int(index) for index in ordered_indices),
        pre=pre,
        squeeze=squeeze,
        reshape2=reshape2,
        source_name=source_name,
        reshape2_name=reshape2_name,
        source_contract=source_contract,
        pre_contract=pre_contract,
        squeeze_contract=squeeze_contract,
        reshape2_contract=reshape2_contract,
        data_contracts=data_contracts,
        data_names=data_names,
        n=int(n),
        c=int(c),
        w=int(w),
        n_signature=int(n_sig),
        c_signature=int(c_sig),
        w_signature=int(w_sig),
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_InstanceNormRewritePlan]:
    prefix = _resolve_flattened_instance_norm_prefix(
        model_ir,
        graph_index,
        pre_index,
    )
    if prefix is None:
        return None

    public_inputs = set(prefix.public_inputs)
    public_outputs = set(prefix.public_outputs)
    reshape2_name = prefix.reshape2_name
    reshape2_contract = prefix.reshape2_contract
    source_contract = prefix.source_contract
    pre_contract = prefix.pre_contract

    unary_users = graph_index.consumer_indices(reshape2_name)
    if len(unary_users) != 1:
        return None
    unary_index = int(unary_users[0])
    unary = model_ir.operators[unary_index]
    if (
        str(unary.op_type) not in _UNARY_OPS
        or len(unary.inputs) != 1
        or str(unary.inputs[0]) != reshape2_name
    ):
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        unary,
        unary_index,
        3,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    unary_name, unary_contract = produced
    if unary_contract.shape != reshape2_contract.shape or unary_contract.signature != reshape2_contract.signature:
        return None

    expand_users = graph_index.consumer_indices(unary_name)
    if len(expand_users) != 1:
        return None
    expand_index = int(expand_users[0])
    expand = model_ir.operators[expand_index]
    if (
        str(expand.op_type) != "EXPAND_DIMS"
        or len(expand.inputs) != 2
        or str(expand.inputs[0]) != unary_name
        or _constant_vector(
            model_ir,
            graph_index,
            str(expand.inputs[1]),
            1,
            public_inputs,
        )
        != (2,)
    ):
        return None
    produced = _produced_contract(
        model_ir,
        graph_index,
        expand,
        expand_index,
        4,
        public_inputs,
        public_outputs,
    )
    if produced is None:
        return None
    expand_name, expand_contract = produced
    if (
        expand_contract.shape != pre_contract.shape
        or expand_contract.signature != pre_contract.signature
    ):
        return None

    post_users = graph_index.consumer_indices(expand_name)
    if len(post_users) != 1:
        return None
    post_index = int(post_users[0])
    post = model_ir.operators[post_index]
    if (
        str(post.op_type) != "TRANSPOSE"
        or len(post.inputs) != 2
        or str(post.inputs[0]) != expand_name
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
    produced = _produced_contract(
        model_ir,
        graph_index,
        post,
        post_index,
        4,
        public_inputs,
        public_outputs,
        allow_public_output=True,
    )
    if produced is None:
        return None
    post_name, post_contract = produced
    if (
        post_contract.shape != source_contract.shape
        or post_contract.signature != source_contract.signature
        or any(
            int(index) <= post_index
            for index in graph_index.consumer_indices(post_name)
        )
    ):
        return None

    ordered_indices = prefix.ordered_indices + (
        unary_index,
        expand_index,
        post_index,
    )
    if list(ordered_indices) != sorted(ordered_indices) or len(set(ordered_indices)) != len(ordered_indices):
        return None

    output_group = (unary_contract, expand_contract)
    if (
        len({str(contract.tensor.dtype) for contract in output_group}) != 1
        or not _quantization_contract(output_group)
    ):
        return None
    if str(unary.op_type) != "CAST" and (
        str(unary_contract.tensor.dtype) != str(source_contract.tensor.dtype)
        or unary_contract.tensor.quantization is not None
    ):
        return None

    data_names = list(prefix.data_names) + [
        unary_name,
        expand_name,
        post_name,
    ]
    if len(data_names) != len(set(data_names)):
        return None

    rank3_shape = (prefix.n, prefix.w, prefix.c)
    rank3_signature = (
        prefix.n_signature,
        prefix.w_signature,
        prefix.c_signature,
    )
    if rank3_signature.count(-1) > 1:
        return None
    rank4_shape = source_contract.shape
    rank4_signature = source_contract.signature
    reshape2 = prefix.reshape2
    squeeze_contract = prefix.squeeze_contract
    reshape2_options = (
        dict(reshape2.options) if isinstance(reshape2.options, dict) else {}
    )
    for key in ("newShape", "onnxRawNewShape"):
        value = reshape2_options.get(key)
        if isinstance(value, list):
            if tuple(int(item) for item in value) != squeeze_contract.shape:
                return None
            reshape2_options[key] = list(rank3_signature)
    if rank3_signature != rank3_shape:
        reshape2_options["newShape"] = list(rank3_signature)
        if "onnxRawNewShape" in reshape2_options:
            reshape2_options["onnxRawNewShape"] = list(rank3_signature)
        reshape2_options["preserveDynamicShape"] = True

    reshape2_target = (
        rank3_signature if rank3_signature != rank3_shape else rank3_shape
    )
    reshape2_shape_update = None
    if tuple(int(value) for value in squeeze_contract.shape) != reshape2_target:
        reshape2_shape_update = _plan_constant_update(
            model_ir,
            graph_index,
            reshape2,
            prefix.ordered_indices[-1],
            1,
            reshape2_target,
            "nhwc_shape",
            public_outputs,
        )
        if reshape2_shape_update is None:
            return None
    expand_axis_update = _plan_constant_update(
        model_ir,
        graph_index,
        expand,
        expand_index,
        1,
        (1,),
        "nhwc_axis",
        public_outputs,
    )
    if expand_axis_update is None:
        return None

    squeeze = prefix.squeeze
    squeeze_options = (
        dict(squeeze.options) if isinstance(squeeze.options, dict) else {}
    )
    squeeze_options["squeezeDims"] = [1]
    return _InstanceNormRewritePlan(
        ordered_ops=tuple(model_ir.operators[index] for index in ordered_indices),
        squeeze=squeeze,
        squeeze_options=squeeze_options,
        squeeze_tensor=squeeze_contract.tensor,
        reshape2=reshape2,
        reshape2_options=reshape2_options,
        reshape2_shape_update=reshape2_shape_update,
        reshape2_tensor=reshape2_contract.tensor,
        unary_tensor=unary_contract.tensor,
        expand=expand,
        expand_axis_update=expand_axis_update,
        expand_tensor=expand_contract.tensor,
        source_name=prefix.source_name,
        post_output_name=post_name,
        rank3_shape=rank3_shape,
        rank3_signature=rank3_signature,
        rank4_shape=rank4_shape,
        rank4_signature=rank4_signature,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _InstanceNormRewritePlan,
) -> bool:
    indices = [graph_index.operator_index(operator) for operator in plan.ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != len(plan.ordered_ops):
        return False
    remove_indices = [resolved[0], resolved[-1]]
    if (
        plan.reshape2_shape_update is not None
        and not _apply_constant_update(
            model_ir,
            graph_index,
            plan.reshape2_shape_update,
        )
    ):
        return False
    if not _apply_constant_update(
        model_ir,
        graph_index,
        plan.expand_axis_update,
    ):
        return False

    plan.squeeze.options = dict(plan.squeeze_options)
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.squeeze,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    plan.squeeze_tensor.shape = list(plan.rank3_shape)
    plan.squeeze_tensor.shape_signature = list(plan.rank3_signature)
    plan.reshape2.options = dict(plan.reshape2_options)
    plan.reshape2_tensor.shape = list(plan.rank3_shape)
    plan.reshape2_tensor.shape_signature = list(plan.rank3_signature)
    plan.unary_tensor.shape = list(plan.rank3_shape)
    plan.unary_tensor.shape_signature = list(plan.rank3_signature)
    plan.expand_tensor.shape = list(plan.rank4_shape)
    plan.expand_tensor.shape_signature = list(plan.rank4_signature)
    _replace_tensor_inputs(
        model_ir,
        plan.post_output_name,
        str(plan.expand.outputs[0]),
        graph_index=graph_index,
    )
    model_ir.outputs = [
        str(plan.expand.outputs[0])
        if str(output_name) == plan.post_output_name
        else str(output_name)
        for output_name in model_ir.outputs
    ]
    graph_index.remove_operators(remove_indices)
    return True


def _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    required = {
        "TRANSPOSE": 2,
        "SQUEEZE": 1,
        "RESHAPE": 2,
        "MEAN": 2,
        "SUB": 1,
        "MUL": 3,
        "ADD": 2,
        "SQRT": 1,
        "DIV": 1,
        "EXPAND_DIMS": 1,
    }
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
