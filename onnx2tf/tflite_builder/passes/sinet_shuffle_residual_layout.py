from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _replace_operator_input_at,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    _FLOAT_DTYPES,
    _NCHW_TO_NHWC,
    _NHWC_TO_NCHW,
    _has_exact_producer,
    _permute,
    _plain_binary,
    _tensor_contract,
    _typed_permutation,
    _unique_planned_name,
)


_STATS_KEY = "optimized_sinet_shuffle_residual_transpose_chains"
_POSTMUL_STATS_KEY = (
    "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains"
)
_LATE_STATS_KEY = (
    "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains"
)


@dataclass(frozen=True)
class _ConstantUse:
    operator: OperatorIR
    input_index: int


@dataclass(frozen=True)
class _ConstantPlan:
    name: str
    tensor: TensorIR
    data: np.ndarray
    clone_name: Optional[str]
    uses: Tuple[_ConstantUse, ...]


@dataclass(frozen=True)
class _MetadataUpdate:
    name: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    logical_layout: str
    physical_layout: str


@dataclass(frozen=True)
class _ResidualPrefixPlan:
    pre_a: OperatorIR
    pre_x: OperatorIR
    pre_y: OperatorIR
    add0: OperatorIR
    mul1: OperatorIR
    add1: OperatorIR
    prelu1: OperatorIR
    post1: OperatorIR
    concat2: OperatorIR
    add0_inputs: Tuple[str, str]
    concat2_inputs: Tuple[str, str]
    prelu1_output_name: str
    post1_output_name: str
    concat2_output_name: str
    dtype: str
    concat_nchw_shape: Tuple[int, ...]
    concat_nchw_signature: Tuple[int, ...]
    concat_nhwc_shape: Tuple[int, ...]
    concat_nhwc_signature: Tuple[int, ...]
    tensor_names: Tuple[str, ...]
    constant_roles: Tuple[Tuple[str, np.ndarray, OperatorIR, int], ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]


@dataclass(frozen=True)
class _ShuffleResidualPlan:
    root: OperatorIR
    pre_a: OperatorIR
    pre_x: OperatorIR
    pre_y: OperatorIR
    add0: OperatorIR
    mul1: OperatorIR
    add1: OperatorIR
    prelu1: OperatorIR
    post1: OperatorIR
    concat2: OperatorIR
    mul2: OperatorIR
    add2: OperatorIR
    prelu2: OperatorIR
    post2: OperatorIR
    add0_inputs: Tuple[str, str]
    concat2_inputs: Tuple[str, str]
    prelu1_output_name: str
    post1_output_name: str
    prelu2_output_name: str
    post2_output_name: str
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]


@dataclass(frozen=True)
class _PostMulTailPlan:
    root: OperatorIR
    prefix: _ResidualPrefixPlan
    mul2: OperatorIR
    post2: OperatorIR
    add2: OperatorIR
    prelu2: OperatorIR
    mul2_output_name: str
    post2_output_name: str
    add2_output_name: str
    prelu2_output_name: str
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]


@dataclass(frozen=True)
class _LateResidualPlan:
    root: OperatorIR
    downstream: OperatorIR
    pre_x: OperatorIR
    pre_y: OperatorIR
    add0: OperatorIR
    mul1: OperatorIR
    add1: OperatorIR
    prelu1: OperatorIR
    add0_inputs: Tuple[str, str]
    prelu1_output_name: str
    post1_output_name: str
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]


def _producer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    op_type: str,
) -> Optional[Tuple[int, OperatorIR]]:
    index = graph_index.producers.get(str(name))
    if (
        index is None
        or not _has_exact_producer(graph_index, str(name), int(index))
    ):
        return None
    operator = model_ir.operators[int(index)]
    if (
        str(operator.op_type) != str(op_type)
        or len(operator.outputs) != 1
        or str(operator.outputs[0]) != str(name)
    ):
        return None
    return int(index), operator


def _no_fused_activation(operator: OperatorIR) -> bool:
    return str(operator.options.get("fusedActivationFunction", "NONE")).upper() in {
        "",
        "NONE",
    }


def _plain_prelu(operator: OperatorIR) -> bool:
    return bool(
        str(operator.op_type) == "PRELU"
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
        and _no_fused_activation(operator)
    )


def _plain_concat(operator: OperatorIR) -> bool:
    try:
        axis = int(operator.options.get("axis", 1))
    except (TypeError, ValueError):
        return False
    return bool(
        str(operator.op_type) == "CONCATENATION"
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
        and axis == 1
        and _no_fused_activation(operator)
    )


def _resolved_source(
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    adapter_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> bool:
    source_name = str(name)
    if source_name in public_outputs or source_name in graph_index.duplicate_producers:
        return False
    producer_index = graph_index.producers.get(source_name)
    return bool(
        (producer_index is None and source_name in public_inputs)
        or (
            producer_index is not None
            and int(producer_index) < int(adapter_index)
        )
    )


def _concat_signature(
    lhs: Tuple[int, ...],
    rhs: Tuple[int, ...],
    *,
    axis: int,
) -> Optional[Tuple[int, ...]]:
    if len(lhs) != len(rhs):
        return None
    result = []
    for index, (lhs_value, rhs_value) in enumerate(zip(lhs, rhs)):
        if index == int(axis):
            result.append(
                -1
                if int(lhs_value) < 0 or int(rhs_value) < 0
                else int(lhs_value) + int(rhs_value)
            )
            continue
        if int(lhs_value) == int(rhs_value):
            result.append(int(lhs_value))
        elif int(lhs_value) < 0 or int(rhs_value) < 0:
            result.append(-1)
        else:
            return None
    return tuple(result)


def _constant_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    dtype: str,
    target_shape: Tuple[int, ...],
    public_names: set[str],
) -> Optional[np.ndarray]:
    tensor = model_ir.tensors.get(str(name))
    expected_dtype = _FLOAT_DTYPES.get(str(dtype))
    if tensor is None or tensor.data is None or expected_dtype is None:
        return None
    try:
        data = np.asarray(tensor.data)
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
    except (TypeError, ValueError):
        return None
    if (
        str(name) in public_names
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or str(tensor.dtype) != str(dtype)
        or tensor.is_variable
        or tensor.quantization is not None
        or data.dtype != expected_dtype
        or tuple(int(value) for value in data.shape) != shape
        or signature != shape
        or not np.all(np.isfinite(data))
        or len(target_shape) != 4
    ):
        return None
    if int(data.size) == 1:
        return np.asarray(data)
    channels = int(target_shape[3])
    if data.ndim != 4 or channels <= 0:
        return None
    if shape == (1, 1, 1, channels):
        return np.asarray(data)
    if shape == (1, channels, 1, 1):
        return np.transpose(data, _NCHW_TO_NHWC).astype(
            expected_dtype,
            copy=False,
        )
    return None


def _late_constant_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    dtype: str,
    old_nchw_shape: Tuple[int, ...],
    target_nhwc_shape: Tuple[int, ...],
    public_names: set[str],
) -> Optional[np.ndarray]:
    validated = _constant_replacement(
        model_ir,
        graph_index,
        name=str(name),
        dtype=str(dtype),
        target_shape=target_nhwc_shape,
        public_names=public_names,
    )
    if validated is None:
        return None
    tensor = model_ir.tensors[str(name)]
    data = np.asarray(tensor.data)
    if int(data.size) == 1:
        return np.asarray(data)
    if data.ndim != 4:
        return None
    rotated = np.transpose(data, _NCHW_TO_NHWC).astype(
        _FLOAT_DTYPES[str(dtype)],
        copy=False,
    )
    try:
        old_broadcast = tuple(
            int(value)
            for value in np.broadcast_shapes(data.shape, old_nchw_shape)
        )
        new_broadcast = tuple(
            int(value)
            for value in np.broadcast_shapes(
                rotated.shape,
                target_nhwc_shape,
            )
        )
    except (TypeError, ValueError):
        return None
    if (
        old_broadcast != tuple(int(value) for value in old_nchw_shape)
        or new_broadcast != tuple(int(value) for value in target_nhwc_shape)
    ):
        return None
    return np.asarray(rotated)


def _plan_constants(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    roles: Tuple[Tuple[str, np.ndarray, OperatorIR, int], ...],
) -> Optional[Tuple[_ConstantPlan, ...]]:
    grouped: dict[str, list[Tuple[np.ndarray, _ConstantUse]]] = {}
    order = []
    for name, data, operator, input_index in roles:
        normalized_name = str(name)
        if normalized_name not in grouped:
            grouped[normalized_name] = []
            order.append(normalized_name)
        grouped[normalized_name].append(
            (np.asarray(data), _ConstantUse(operator, int(input_index)))
        )

    reserved_names = {str(name) for name in model_ir.tensors}
    plans = []
    for name in order:
        entries = grouped[name]
        reference = entries[0][0]
        if any(
            data.dtype != reference.dtype
            or data.shape != reference.shape
            or not np.array_equal(data, reference)
            for data, _ in entries[1:]
        ):
            return None
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        uses = tuple(use for _, use in entries)
        use_indices = []
        for use in uses:
            operator_index = graph_index.operator_index(use.operator)
            if operator_index is None:
                return None
            use_indices.append(int(operator_index))
        original = np.asarray(tensor.data)
        changed = bool(
            original.dtype != reference.dtype
            or original.shape != reference.shape
            or not np.array_equal(original, reference)
        )
        clone_name = None
        if changed and Counter(graph_index.consumer_indices(name)) != Counter(
            use_indices
        ):
            clone_name = _unique_planned_name(
                f"{name}_nhwc",
                reserved_names,
            )
        plans.append(
            _ConstantPlan(
                name=name,
                tensor=tensor,
                data=np.asarray(reference),
                clone_name=clone_name,
                uses=uses,
            )
        )
    return tuple(plans)


def _metadata_update(
    name: str,
    canonical: TensorIR,
) -> _MetadataUpdate:
    return _MetadataUpdate(
        name=str(name),
        shape=tuple(int(value) for value in canonical.shape),
        signature=tuple(
            int(value)
            for value in (
                canonical.shape
                if canonical.shape_signature is None
                else canonical.shape_signature
            )
        ),
        logical_layout=str(canonical.logical_layout),
        physical_layout=str(canonical.physical_layout),
    )


def _resolve_prefix(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    concat2: OperatorIR,
    *,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_ResidualPrefixPlan]:
    concat2_index = graph_index.operator_index(concat2)
    public_names = public_inputs | public_outputs
    if (
        concat2_index is None
        or not _plain_concat(concat2)
        or not _has_exact_producer(
            graph_index,
            str(concat2.outputs[0]),
            int(concat2_index),
        )
    ):
        return None
    concat2_output_name = str(concat2.outputs[0])
    if concat2_output_name in public_names:
        return None

    concat_roles = []
    for input_index, input_name in enumerate(concat2.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        if (
            producer_index is None
            or not _has_exact_producer(
                graph_index,
                str(input_name),
                int(producer_index),
            )
        ):
            return None
        producer = model_ir.operators[int(producer_index)]
        if (
            len(producer.outputs) != 1
            or str(producer.outputs[0]) != str(input_name)
        ):
            return None
        if (
            str(producer.op_type) == "TRANSPOSE"
            and _typed_permutation(
                model_ir,
                graph_index,
                producer,
                _NHWC_TO_NCHW,
                public_names,
            )
        ):
            concat_roles.append(
                ("pre_a", int(input_index), int(producer_index), producer)
            )
        elif str(producer.op_type) == "PRELU" and _plain_prelu(producer):
            concat_roles.append(
                ("prelu1", int(input_index), int(producer_index), producer)
            )
        else:
            return None
    if [role[0] for role in concat_roles].count("pre_a") != 1 or [
        role[0] for role in concat_roles
    ].count("prelu1") != 1:
        return None

    _, pre_a_input_index, pre_a_index, pre_a = next(
        role for role in concat_roles if role[0] == "pre_a"
    )
    _, prelu1_input_index, prelu1_index, prelu1 = next(
        role for role in concat_roles if role[0] == "prelu1"
    )
    pre_a_output_name = str(pre_a.outputs[0])
    source_a_name = str(pre_a.inputs[0])
    prelu1_output_name = str(prelu1.outputs[0])
    if (
        int(pre_a_index) >= int(concat2_index)
        or int(prelu1_index) >= int(concat2_index)
        or graph_index.consumer_indices(pre_a_output_name)
        != [int(concat2_index)]
        or not _resolved_source(
            graph_index,
            name=source_a_name,
            adapter_index=int(pre_a_index),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
    ):
        return None

    prelu1_users = graph_index.consumer_indices(prelu1_output_name)
    post1_matches = []
    for consumer_index in sorted(set(prelu1_users)):
        if int(consumer_index) == int(concat2_index):
            continue
        consumer = model_ir.operators[int(consumer_index)]
        if (
            str(consumer.op_type) == "TRANSPOSE"
            and len(consumer.inputs) == 2
            and len(consumer.outputs) == 1
            and str(consumer.inputs[0]) == prelu1_output_name
            and _typed_permutation(
                model_ir,
                graph_index,
                consumer,
                _NCHW_TO_NHWC,
                public_names,
            )
        ):
            post1_matches.append((int(consumer_index), consumer))
    if len(post1_matches) != 1:
        return None
    post1_index, post1 = post1_matches[0]
    post1_output_name = str(post1.outputs[0])
    if (
        Counter(prelu1_users)
        != Counter((int(concat2_index), int(post1_index)))
        or int(prelu1_index) >= min(int(concat2_index), int(post1_index))
        or any(
            int(index) <= int(post1_index)
            for index in graph_index.consumer_indices(post1_output_name)
        )
    ):
        return None

    add1_output_name = str(prelu1.inputs[0])
    add1_match = _producer(model_ir, graph_index, add1_output_name, "ADD")
    if add1_match is None:
        return None
    add1_index, add1 = add1_match
    if (
        not _plain_binary(add1, "ADD")
        or int(add1_index) >= int(prelu1_index)
        or graph_index.consumer_indices(add1_output_name)
        != [int(prelu1_index)]
    ):
        return None
    add1_constants = [
        index
        for index, name in enumerate(add1.inputs)
        if (tensor := model_ir.tensors.get(str(name))) is not None
        and tensor.data is not None
    ]
    if len(add1_constants) != 1:
        return None
    add1_constant_index = int(add1_constants[0])
    add1_constant_name = str(add1.inputs[add1_constant_index])
    mul1_output_name = str(add1.inputs[1 - add1_constant_index])

    mul1_match = _producer(model_ir, graph_index, mul1_output_name, "MUL")
    if mul1_match is None:
        return None
    mul1_index, mul1 = mul1_match
    if (
        not _plain_binary(mul1, "MUL")
        or int(mul1_index) >= int(add1_index)
        or graph_index.consumer_indices(mul1_output_name)
        != [int(add1_index)]
    ):
        return None
    mul1_constants = [
        index
        for index, name in enumerate(mul1.inputs)
        if (tensor := model_ir.tensors.get(str(name))) is not None
        and tensor.data is not None
    ]
    if len(mul1_constants) != 1:
        return None
    mul1_constant_index = int(mul1_constants[0])
    mul1_constant_name = str(mul1.inputs[mul1_constant_index])
    add0_output_name = str(mul1.inputs[1 - mul1_constant_index])

    add0_match = _producer(model_ir, graph_index, add0_output_name, "ADD")
    if add0_match is None:
        return None
    add0_index, add0 = add0_match
    if (
        not _plain_binary(add0, "ADD")
        or int(add0_index) >= int(mul1_index)
        or graph_index.consumer_indices(add0_output_name)
        != [int(mul1_index)]
    ):
        return None

    pre_add_roles = []
    for input_index, input_name in enumerate(add0.inputs):
        pre_match = _producer(
            model_ir,
            graph_index,
            str(input_name),
            "TRANSPOSE",
        )
        if pre_match is None:
            return None
        pre_index, pre = pre_match
        if (
            not _typed_permutation(
                model_ir,
                graph_index,
                pre,
                _NHWC_TO_NCHW,
                public_names,
            )
            or int(pre_index) >= int(add0_index)
            or graph_index.consumer_indices(str(input_name))
            != [int(add0_index)]
            or not _resolved_source(
                graph_index,
                name=str(pre.inputs[0]),
                adapter_index=int(pre_index),
                public_inputs=public_inputs,
                public_outputs=public_outputs,
            )
        ):
            return None
        pre_add_roles.append(
            (int(input_index), int(pre_index), pre, str(pre.inputs[0]))
        )
    if len(pre_add_roles) != 2 or pre_add_roles[0][2] is pre_add_roles[1][2]:
        return None
    _, _, pre_x, source_x_name = pre_add_roles[0]
    _, _, pre_y, source_y_name = pre_add_roles[1]

    operators = (
        pre_a,
        pre_x,
        pre_y,
        add0,
        mul1,
        add1,
        prelu1,
        post1,
        concat2,
    )
    operator_indices = [
        graph_index.operator_index(operator) for operator in operators
    ]
    if any(index is None for index in operator_indices) or len(
        {int(index) for index in operator_indices if index is not None}
    ) != len(operators):
        return None

    tensor_names = (
        source_a_name,
        source_x_name,
        source_y_name,
        pre_a_output_name,
        str(pre_x.outputs[0]),
        str(pre_y.outputs[0]),
        add0_output_name,
        mul1_output_name,
        add1_output_name,
        prelu1_output_name,
        post1_output_name,
        concat2_output_name,
    )
    if (
        len(set(tensor_names)) != len(tensor_names)
        or any(name in public_names for name in tensor_names[3:])
        or any(name in graph_index.duplicate_producers for name in tensor_names)
    ):
        return None
    contracts = {
        name: _tensor_contract(model_ir, name, 4) for name in tensor_names
    }
    if any(contract is None for contract in contracts.values()):
        return None
    source_a = contracts[source_a_name]
    source_x = contracts[source_x_name]
    source_y = contracts[source_y_name]
    assert source_a is not None
    assert source_x is not None
    assert source_y is not None
    dtype = str(source_x.tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(
            str(contract.tensor.dtype) != dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in contracts.values()
            if contract is not None
        )
        or source_y.shape != source_x.shape
        or source_y.signature != source_x.signature
        or contracts[str(pre_x.outputs[0])].shape
        != _permute(source_x.shape, _NHWC_TO_NCHW)
        or contracts[str(pre_y.outputs[0])].shape
        != _permute(source_y.shape, _NHWC_TO_NCHW)
        or contracts[str(pre_x.outputs[0])].signature
        != _permute(source_x.signature, _NHWC_TO_NCHW)
        or contracts[str(pre_y.outputs[0])].signature
        != _permute(source_y.signature, _NHWC_TO_NCHW)
        or contracts[pre_a_output_name].shape
        != _permute(source_a.shape, _NHWC_TO_NCHW)
        or contracts[pre_a_output_name].signature
        != _permute(source_a.signature, _NHWC_TO_NCHW)
    ):
        return None

    stage1_nchw_shape = contracts[str(pre_x.outputs[0])].shape
    stage1_nchw_signature = contracts[str(pre_x.outputs[0])].signature
    for name in (
        add0_output_name,
        mul1_output_name,
        add1_output_name,
        prelu1_output_name,
    ):
        if (
            contracts[name].shape != stage1_nchw_shape
            or contracts[name].signature != stage1_nchw_signature
        ):
            return None
    if (
        contracts[post1_output_name].shape != source_x.shape
        or contracts[post1_output_name].signature != source_x.signature
    ):
        return None

    concat_input_contracts = [contracts[str(name)] for name in concat2.inputs]
    expected_concat_shape = _concat_signature(
        concat_input_contracts[0].shape,
        concat_input_contracts[1].shape,
        axis=1,
    )
    expected_concat_signature = _concat_signature(
        concat_input_contracts[0].signature,
        concat_input_contracts[1].signature,
        axis=1,
    )
    if (
        expected_concat_shape is None
        or expected_concat_signature is None
        or contracts[concat2_output_name].shape != expected_concat_shape
        or contracts[concat2_output_name].signature
        != expected_concat_signature
    ):
        return None

    constant_roles = []
    for name, operator, input_index in (
        (mul1_constant_name, mul1, mul1_constant_index),
        (add1_constant_name, add1, add1_constant_index),
        (str(prelu1.inputs[1]), prelu1, 1),
    ):
        replacement = _constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=dtype,
            target_shape=source_x.shape,
            public_names=public_names,
        )
        if replacement is None:
            return None
        constant_roles.append(
            (str(name), replacement, operator, int(input_index))
        )

    post1_tensor = contracts[post1_output_name].tensor
    new_concat_inputs = [str(name) for name in concat2.inputs]
    new_concat_inputs[int(pre_a_input_index)] = source_a_name
    new_concat_inputs[int(prelu1_input_index)] = post1_output_name
    return _ResidualPrefixPlan(
        pre_a=pre_a,
        pre_x=pre_x,
        pre_y=pre_y,
        add0=add0,
        mul1=mul1,
        add1=add1,
        prelu1=prelu1,
        post1=post1,
        concat2=concat2,
        add0_inputs=(source_x_name, source_y_name),
        concat2_inputs=(
            str(new_concat_inputs[0]),
            str(new_concat_inputs[1]),
        ),
        prelu1_output_name=prelu1_output_name,
        post1_output_name=post1_output_name,
        concat2_output_name=concat2_output_name,
        dtype=dtype,
        concat_nchw_shape=contracts[concat2_output_name].shape,
        concat_nchw_signature=contracts[concat2_output_name].signature,
        concat_nhwc_shape=_permute(
            contracts[concat2_output_name].shape,
            _NCHW_TO_NHWC,
        ),
        concat_nhwc_signature=_permute(
            contracts[concat2_output_name].signature,
            _NCHW_TO_NHWC,
        ),
        tensor_names=tensor_names,
        constant_roles=tuple(constant_roles),
        metadata_updates=tuple(
            _metadata_update(name, post1_tensor)
            for name in (
                add0_output_name,
                mul1_output_name,
                add1_output_name,
            )
        ),
        remove_operators=(pre_a, pre_x, pre_y, post1),
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_ShuffleResidualPlan]:
    post2_index = graph_index.operator_index(root)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    if (
        post2_index is None
        or str(root.op_type) != "TRANSPOSE"
        or len(root.inputs) != 2
        or len(root.outputs) != 1
        or not _typed_permutation(
            model_ir,
            graph_index,
            root,
            _NCHW_TO_NHWC,
            public_names,
        )
    ):
        return None
    post2 = root
    prelu2_output_name = str(post2.inputs[0])
    post2_output_name = str(post2.outputs[0])
    prelu2_match = _producer(
        model_ir,
        graph_index,
        prelu2_output_name,
        "PRELU",
    )
    if (
        prelu2_match is None
        or post2_output_name in public_names
        or prelu2_output_name in public_names
        or graph_index.consumer_indices(prelu2_output_name)
        != [int(post2_index)]
        or any(
            int(index) <= int(post2_index)
            for index in graph_index.consumer_indices(post2_output_name)
        )
    ):
        return None
    prelu2_index, prelu2 = prelu2_match
    if not _plain_prelu(prelu2) or int(prelu2_index) >= int(post2_index):
        return None

    add2_output_name = str(prelu2.inputs[0])
    add2_match = _producer(model_ir, graph_index, add2_output_name, "ADD")
    if add2_match is None:
        return None
    add2_index, add2 = add2_match
    if (
        not _plain_binary(add2, "ADD")
        or int(add2_index) >= int(prelu2_index)
        or graph_index.consumer_indices(add2_output_name)
        != [int(prelu2_index)]
    ):
        return None
    add2_constants = [
        index
        for index, name in enumerate(add2.inputs)
        if (tensor := model_ir.tensors.get(str(name))) is not None
        and tensor.data is not None
    ]
    if len(add2_constants) != 1:
        return None
    add2_constant_index = int(add2_constants[0])
    add2_constant_name = str(add2.inputs[add2_constant_index])
    mul2_output_name = str(add2.inputs[1 - add2_constant_index])

    mul2_match = _producer(model_ir, graph_index, mul2_output_name, "MUL")
    if mul2_match is None:
        return None
    mul2_index, mul2 = mul2_match
    if (
        not _plain_binary(mul2, "MUL")
        or int(mul2_index) >= int(add2_index)
        or graph_index.consumer_indices(mul2_output_name)
        != [int(add2_index)]
    ):
        return None
    mul2_constants = [
        index
        for index, name in enumerate(mul2.inputs)
        if (tensor := model_ir.tensors.get(str(name))) is not None
        and tensor.data is not None
    ]
    if len(mul2_constants) != 1:
        return None
    mul2_constant_index = int(mul2_constants[0])
    mul2_constant_name = str(mul2.inputs[mul2_constant_index])
    concat2_output_name = str(mul2.inputs[1 - mul2_constant_index])

    concat2_match = _producer(
        model_ir,
        graph_index,
        concat2_output_name,
        "CONCATENATION",
    )
    if concat2_match is None:
        return None
    concat2_index, concat2 = concat2_match
    if (
        int(concat2_index) >= int(mul2_index)
        or graph_index.consumer_indices(concat2_output_name)
        != [int(mul2_index)]
    ):
        return None
    prefix = _resolve_prefix(
        model_ir,
        graph_index,
        concat2,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if prefix is None:
        return None

    tail_operators = (mul2, add2, prelu2, post2)
    prefix_operators = (
        prefix.pre_a,
        prefix.pre_x,
        prefix.pre_y,
        prefix.add0,
        prefix.mul1,
        prefix.add1,
        prefix.prelu1,
        prefix.post1,
        prefix.concat2,
    )
    if len({id(operator) for operator in (*prefix_operators, *tail_operators)}) != 13:
        return None
    tail_names = (
        mul2_output_name,
        add2_output_name,
        prelu2_output_name,
        post2_output_name,
    )
    if (
        len(set((*prefix.tensor_names, *tail_names)))
        != len(prefix.tensor_names) + len(tail_names)
        or any(name in public_names for name in tail_names)
        or any(name in graph_index.duplicate_producers for name in tail_names)
    ):
        return None
    tail_contracts = {
        name: _tensor_contract(model_ir, name, 4) for name in tail_names
    }
    if any(contract is None for contract in tail_contracts.values()):
        return None
    if (
        any(
            str(contract.tensor.dtype) != prefix.dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in tail_contracts.values()
            if contract is not None
        )
        or tail_contracts[mul2_output_name].shape
        != prefix.concat_nchw_shape
        or tail_contracts[mul2_output_name].signature
        != prefix.concat_nchw_signature
        or tail_contracts[add2_output_name].shape
        != prefix.concat_nchw_shape
        or tail_contracts[add2_output_name].signature
        != prefix.concat_nchw_signature
        or tail_contracts[prelu2_output_name].shape
        != prefix.concat_nchw_shape
        or tail_contracts[prelu2_output_name].signature
        != prefix.concat_nchw_signature
        or tail_contracts[post2_output_name].shape
        != prefix.concat_nhwc_shape
        or tail_contracts[post2_output_name].signature
        != prefix.concat_nhwc_signature
    ):
        return None

    constant_roles = list(prefix.constant_roles)
    for name, operator, input_index in (
        (mul2_constant_name, mul2, mul2_constant_index),
        (add2_constant_name, add2, add2_constant_index),
        (str(prelu2.inputs[1]), prelu2, 1),
    ):
        replacement = _constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=prefix.dtype,
            target_shape=prefix.concat_nhwc_shape,
            public_names=public_names,
        )
        if replacement is None:
            return None
        constant_roles.append(
            (str(name), replacement, operator, int(input_index))
        )
    constant_plans = _plan_constants(
        model_ir,
        graph_index,
        tuple(constant_roles),
    )
    if constant_plans is None:
        return None

    post2_tensor = tail_contracts[post2_output_name].tensor
    return _ShuffleResidualPlan(
        root=root,
        pre_a=prefix.pre_a,
        pre_x=prefix.pre_x,
        pre_y=prefix.pre_y,
        add0=prefix.add0,
        mul1=prefix.mul1,
        add1=prefix.add1,
        prelu1=prefix.prelu1,
        post1=prefix.post1,
        concat2=prefix.concat2,
        mul2=mul2,
        add2=add2,
        prelu2=prelu2,
        post2=post2,
        add0_inputs=prefix.add0_inputs,
        concat2_inputs=prefix.concat2_inputs,
        prelu1_output_name=prefix.prelu1_output_name,
        post1_output_name=prefix.post1_output_name,
        prelu2_output_name=prelu2_output_name,
        post2_output_name=post2_output_name,
        constant_plans=constant_plans,
        metadata_updates=prefix.metadata_updates
        + tuple(
            _metadata_update(name, post2_tensor)
            for name in (
                prefix.concat2_output_name,
                mul2_output_name,
                add2_output_name,
            )
        ),
        remove_operators=prefix.remove_operators + (post2,),
    )


def _resolve_late_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_LateResidualPlan]:
    root_index = graph_index.operator_index(root)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    if (
        root_index is None
        or str(root.op_type) != "TRANSPOSE"
        or len(root.inputs) != 2
        or len(root.outputs) != 1
        or not _typed_permutation(
            model_ir,
            graph_index,
            root,
            _NCHW_TO_NHWC,
            public_names,
        )
    ):
        return None

    prelu1_output_name = str(root.inputs[0])
    post1_output_name = str(root.outputs[0])
    if post1_output_name in public_names:
        return None
    downstream_indices = graph_index.consumer_indices(post1_output_name)
    if len(downstream_indices) != 1:
        return None
    downstream_index = int(downstream_indices[0])
    downstream = model_ir.operators[downstream_index]
    if (
        downstream_index <= int(root_index)
        or str(downstream.op_type) not in {"CONV_2D", "DEPTHWISE_CONV_2D"}
        or not downstream.inputs
        or str(downstream.inputs[0]) != post1_output_name
        or [str(name) for name in downstream.inputs].count(post1_output_name)
        != 1
    ):
        return None

    prelu1_match = _producer(
        model_ir,
        graph_index,
        prelu1_output_name,
        "PRELU",
    )
    if prelu1_match is None:
        return None
    prelu1_index, prelu1 = prelu1_match
    prelu1_users = graph_index.consumer_indices(prelu1_output_name)
    legacy_indices = [
        int(index)
        for index in prelu1_users
        if int(index) != int(root_index)
    ]
    if (
        not _plain_prelu(prelu1)
        or int(prelu1_index) >= int(root_index)
        or Counter(prelu1_users).get(int(root_index), 0) != 1
        or not legacy_indices
        or any(int(index) <= int(root_index) for index in legacy_indices)
    ):
        return None

    add1_output_name = str(prelu1.inputs[0])
    add1_match = _producer(model_ir, graph_index, add1_output_name, "ADD")
    if add1_match is None:
        return None
    add1_index, add1 = add1_match
    if (
        not _plain_binary(add1, "ADD")
        or int(add1_index) >= int(prelu1_index)
        or graph_index.consumer_indices(add1_output_name)
        != [int(prelu1_index)]
    ):
        return None
    add1_constants = [
        index
        for index, name in enumerate(add1.inputs)
        if (tensor := model_ir.tensors.get(str(name))) is not None
        and tensor.data is not None
    ]
    if len(add1_constants) != 1:
        return None
    add1_constant_index = int(add1_constants[0])
    add1_constant_name = str(add1.inputs[add1_constant_index])
    mul1_output_name = str(add1.inputs[1 - add1_constant_index])

    mul1_match = _producer(model_ir, graph_index, mul1_output_name, "MUL")
    if mul1_match is None:
        return None
    mul1_index, mul1 = mul1_match
    if (
        not _plain_binary(mul1, "MUL")
        or int(mul1_index) >= int(add1_index)
        or graph_index.consumer_indices(mul1_output_name)
        != [int(add1_index)]
    ):
        return None
    mul1_constants = [
        index
        for index, name in enumerate(mul1.inputs)
        if (tensor := model_ir.tensors.get(str(name))) is not None
        and tensor.data is not None
    ]
    if len(mul1_constants) != 1:
        return None
    mul1_constant_index = int(mul1_constants[0])
    mul1_constant_name = str(mul1.inputs[mul1_constant_index])
    add0_output_name = str(mul1.inputs[1 - mul1_constant_index])

    add0_match = _producer(model_ir, graph_index, add0_output_name, "ADD")
    if add0_match is None:
        return None
    add0_index, add0 = add0_match
    if (
        not _plain_binary(add0, "ADD")
        or int(add0_index) >= int(mul1_index)
        or graph_index.consumer_indices(add0_output_name)
        != [int(mul1_index)]
    ):
        return None

    pre_add_roles = []
    concat_backed_inputs = 0
    for input_index, input_name in enumerate(add0.inputs):
        pre_match = _producer(
            model_ir,
            graph_index,
            str(input_name),
            "TRANSPOSE",
        )
        if pre_match is None:
            return None
        pre_index, pre = pre_match
        if (
            not _typed_permutation(
                model_ir,
                graph_index,
                pre,
                _NHWC_TO_NCHW,
                public_names,
            )
            or int(pre_index) >= int(add0_index)
            or graph_index.consumer_indices(str(input_name))
            != [int(add0_index)]
            or not _resolved_source(
                graph_index,
                name=str(pre.inputs[0]),
                adapter_index=int(pre_index),
                public_inputs=public_inputs,
                public_outputs=public_outputs,
            )
        ):
            return None
        source_name = str(pre.inputs[0])
        source_producer_index = graph_index.producers.get(source_name)
        if source_producer_index is not None:
            source_producer = model_ir.operators[int(source_producer_index)]
            try:
                concat_axis = int(source_producer.options.get("axis", -1))
            except (TypeError, ValueError):
                concat_axis = -1
            if (
                str(source_producer.op_type) == "CONCATENATION"
                and concat_axis == 3
            ):
                concat_backed_inputs += 1
        pre_add_roles.append(
            (int(input_index), int(pre_index), pre, source_name)
        )
    if (
        len(pre_add_roles) != 2
        or pre_add_roles[0][2] is pre_add_roles[1][2]
        or concat_backed_inputs != 1
    ):
        return None
    _, _, pre_x, source_x_name = pre_add_roles[0]
    _, _, pre_y, source_y_name = pre_add_roles[1]

    operators = (
        pre_x,
        pre_y,
        add0,
        mul1,
        add1,
        prelu1,
        root,
        downstream,
    )
    operator_indices = [
        graph_index.operator_index(operator) for operator in operators
    ]
    if any(index is None for index in operator_indices) or len(
        {int(index) for index in operator_indices if index is not None}
    ) != len(operators):
        return None

    tensor_names = (
        source_x_name,
        source_y_name,
        str(pre_x.outputs[0]),
        str(pre_y.outputs[0]),
        add0_output_name,
        mul1_output_name,
        add1_output_name,
        prelu1_output_name,
        post1_output_name,
    )
    if (
        len(set(tensor_names)) != len(tensor_names)
        or any(name in public_names for name in tensor_names[2:7])
        or any(name in graph_index.duplicate_producers for name in tensor_names)
    ):
        return None
    contracts = {
        name: _tensor_contract(model_ir, name, 4) for name in tensor_names
    }
    if any(contract is None for contract in contracts.values()):
        return None
    source_x = contracts[source_x_name]
    source_y = contracts[source_y_name]
    assert source_x is not None
    assert source_y is not None
    dtype = str(source_x.tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(
            str(contract.tensor.dtype) != dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in contracts.values()
            if contract is not None
        )
        or source_y.shape != source_x.shape
        or source_y.signature != source_x.signature
        or contracts[str(pre_x.outputs[0])].shape
        != _permute(source_x.shape, _NHWC_TO_NCHW)
        or contracts[str(pre_y.outputs[0])].shape
        != _permute(source_y.shape, _NHWC_TO_NCHW)
        or contracts[str(pre_x.outputs[0])].signature
        != _permute(source_x.signature, _NHWC_TO_NCHW)
        or contracts[str(pre_y.outputs[0])].signature
        != _permute(source_y.signature, _NHWC_TO_NCHW)
    ):
        return None

    stage1_nchw_shape = contracts[str(pre_x.outputs[0])].shape
    stage1_nchw_signature = contracts[str(pre_x.outputs[0])].signature
    for name in (
        add0_output_name,
        mul1_output_name,
        add1_output_name,
        prelu1_output_name,
    ):
        if (
            contracts[name].shape != stage1_nchw_shape
            or contracts[name].signature != stage1_nchw_signature
        ):
            return None
    if (
        contracts[post1_output_name].shape != source_x.shape
        or contracts[post1_output_name].signature != source_x.signature
    ):
        return None

    constant_roles = []
    for name, operator, input_index in (
        (mul1_constant_name, mul1, mul1_constant_index),
        (add1_constant_name, add1, add1_constant_index),
        (str(prelu1.inputs[1]), prelu1, 1),
    ):
        replacement = _late_constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=dtype,
            old_nchw_shape=stage1_nchw_shape,
            target_nhwc_shape=source_x.shape,
            public_names=public_names,
        )
        if replacement is None:
            return None
        constant_roles.append(
            (str(name), replacement, operator, int(input_index))
        )
    root_perm_name = str(root.inputs[1])
    root_perm = model_ir.tensors[root_perm_name]
    root_perm_dtype = np.asarray(root_perm.data).dtype
    constant_roles.append(
        (
            root_perm_name,
            np.asarray(_NHWC_TO_NCHW, dtype=root_perm_dtype),
            root,
            1,
        )
    )
    constant_plans = _plan_constants(
        model_ir,
        graph_index,
        tuple(constant_roles),
    )
    if constant_plans is None:
        return None

    post1_tensor = contracts[post1_output_name].tensor
    return _LateResidualPlan(
        root=root,
        downstream=downstream,
        pre_x=pre_x,
        pre_y=pre_y,
        add0=add0,
        mul1=mul1,
        add1=add1,
        prelu1=prelu1,
        add0_inputs=(source_x_name, source_y_name),
        prelu1_output_name=prelu1_output_name,
        post1_output_name=post1_output_name,
        constant_plans=constant_plans,
        metadata_updates=tuple(
            _metadata_update(name, post1_tensor)
            for name in (
                add0_output_name,
                mul1_output_name,
                add1_output_name,
            )
        ),
        remove_operators=(pre_x, pre_y),
    )


def _constant_plans_equal(
    expected: Tuple[_ConstantPlan, ...],
    actual: Tuple[_ConstantPlan, ...],
) -> bool:
    if len(expected) != len(actual):
        return False
    for lhs, rhs in zip(expected, actual):
        if (
            lhs.name != rhs.name
            or lhs.tensor is not rhs.tensor
            or lhs.clone_name != rhs.clone_name
            or len(lhs.uses) != len(rhs.uses)
            or any(
                left.operator is not right.operator
                or left.input_index != right.input_index
                for left, right in zip(lhs.uses, rhs.uses)
            )
            or lhs.data.dtype != rhs.data.dtype
            or lhs.data.shape != rhs.data.shape
            or not np.array_equal(lhs.data, rhs.data)
        ):
            return False
    return True


def _late_plans_equal(
    expected: _LateResidualPlan,
    actual: _LateResidualPlan,
) -> bool:
    operator_fields = (
        "root",
        "downstream",
        "pre_x",
        "pre_y",
        "add0",
        "mul1",
        "add1",
        "prelu1",
    )
    return bool(
        all(
            getattr(expected, field) is getattr(actual, field)
            for field in operator_fields
        )
        and expected.add0_inputs == actual.add0_inputs
        and expected.prelu1_output_name == actual.prelu1_output_name
        and expected.post1_output_name == actual.post1_output_name
        and expected.metadata_updates == actual.metadata_updates
        and len(expected.remove_operators) == len(actual.remove_operators)
        and all(
            left is right
            for left, right in zip(
                expected.remove_operators,
                actual.remove_operators,
            )
        )
        and _constant_plans_equal(
            expected.constant_plans,
            actual.constant_plans,
        )
    )


def _apply_late_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _LateResidualPlan,
) -> bool:
    current = _resolve_late_candidate(model_ir, graph_index, plan.root)
    if current is None or not _late_plans_equal(plan, current):
        return False
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    add0_index = graph_index.operator_index(plan.add0)
    prelu1_index = graph_index.operator_index(plan.prelu1)
    root_index = graph_index.operator_index(plan.root)
    if (
        any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or any(
            constant.clone_name is not None
            and constant.clone_name in model_ir.tensors
            for constant in plan.constant_plans
        )
        or any(
            update.name not in model_ir.tensors
            for update in plan.metadata_updates
        )
        or any(
            index is None for index in (add0_index, prelu1_index, root_index)
        )
    ):
        return False

    for constant in plan.constant_plans:
        target = constant.tensor
        if constant.clone_name is not None:
            target = TensorIR(
                name=str(constant.clone_name),
                dtype=str(constant.tensor.dtype),
                shape=[int(value) for value in constant.data.shape],
                shape_signature=[int(value) for value in constant.data.shape],
                data=np.asarray(constant.data),
                is_variable=False,
                quantization=None,
                logical_layout=str(constant.tensor.logical_layout),
                physical_layout=str(constant.tensor.physical_layout),
                onnx_tensor_name=constant.tensor.onnx_tensor_name,
            )
            model_ir.tensors[str(constant.clone_name)] = target
            for use in constant.uses:
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=use.operator,
                    input_index=int(use.input_index),
                    new_input_name=str(constant.clone_name),
                    graph_index=graph_index,
                )
        target.data = np.asarray(constant.data)
        target.shape = [int(value) for value in constant.data.shape]
        target.shape_signature = [int(value) for value in constant.data.shape]

    graph_index.replace_operator_inputs(int(add0_index), plan.add0_inputs)
    graph_index.replace_operator_outputs(
        int(root_index),
        [plan.prelu1_output_name],
    )
    graph_index.replace_operator_outputs(
        int(prelu1_index),
        [plan.post1_output_name],
    )
    graph_index.replace_operator_inputs(
        int(root_index),
        [plan.post1_output_name, str(plan.root.inputs[1])],
    )
    for update in plan.metadata_updates:
        tensor = model_ir.tensors[update.name]
        tensor.shape = [int(value) for value in update.shape]
        tensor.shape_signature = [int(value) for value in update.signature]
        tensor.logical_layout = str(update.logical_layout)
        tensor.physical_layout = str(update.physical_layout)
    graph_index.remove_operators([int(index) for index in remove_indices])
    return True


def _plans_equal(
    expected: _ShuffleResidualPlan,
    actual: _ShuffleResidualPlan,
) -> bool:
    operator_fields = (
        "root",
        "pre_a",
        "pre_x",
        "pre_y",
        "add0",
        "mul1",
        "add1",
        "prelu1",
        "post1",
        "concat2",
        "mul2",
        "add2",
        "prelu2",
        "post2",
    )
    return bool(
        all(
            getattr(expected, field) is getattr(actual, field)
            for field in operator_fields
        )
        and expected.add0_inputs == actual.add0_inputs
        and expected.concat2_inputs == actual.concat2_inputs
        and expected.prelu1_output_name == actual.prelu1_output_name
        and expected.post1_output_name == actual.post1_output_name
        and expected.prelu2_output_name == actual.prelu2_output_name
        and expected.post2_output_name == actual.post2_output_name
        and expected.metadata_updates == actual.metadata_updates
        and len(expected.remove_operators) == len(actual.remove_operators)
        and all(
            left is right
            for left, right in zip(
                expected.remove_operators,
                actual.remove_operators,
            )
        )
        and _constant_plans_equal(
            expected.constant_plans,
            actual.constant_plans,
        )
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _ShuffleResidualPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.root)
    if current is None or not _plans_equal(plan, current):
        return False
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    add0_index = graph_index.operator_index(plan.add0)
    concat2_index = graph_index.operator_index(plan.concat2)
    prelu1_index = graph_index.operator_index(plan.prelu1)
    prelu2_index = graph_index.operator_index(plan.prelu2)
    if (
        any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or any(
            constant.clone_name is not None
            and constant.clone_name in model_ir.tensors
            for constant in plan.constant_plans
        )
        or any(
            update.name not in model_ir.tensors
            for update in plan.metadata_updates
        )
        or any(
            index is None
            for index in (
                add0_index,
                concat2_index,
                prelu1_index,
                prelu2_index,
            )
        )
    ):
        return False

    for constant in plan.constant_plans:
        target = constant.tensor
        if constant.clone_name is not None:
            target = TensorIR(
                name=str(constant.clone_name),
                dtype=str(constant.tensor.dtype),
                shape=[int(value) for value in constant.data.shape],
                shape_signature=[int(value) for value in constant.data.shape],
                data=np.asarray(constant.data),
                is_variable=False,
                quantization=None,
                logical_layout=str(constant.tensor.logical_layout),
                physical_layout=str(constant.tensor.physical_layout),
                onnx_tensor_name=constant.tensor.onnx_tensor_name,
            )
            model_ir.tensors[str(constant.clone_name)] = target
            for use in constant.uses:
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=use.operator,
                    input_index=int(use.input_index),
                    new_input_name=str(constant.clone_name),
                    graph_index=graph_index,
                )
        target.data = np.asarray(constant.data)
        target.shape = [int(value) for value in constant.data.shape]
        target.shape_signature = [int(value) for value in constant.data.shape]

    graph_index.replace_operator_inputs(int(add0_index), plan.add0_inputs)
    graph_index.replace_operator_inputs(int(concat2_index), plan.concat2_inputs)
    plan.concat2.options["axis"] = 3
    graph_index.replace_operator_outputs(
        int(prelu1_index),
        [plan.post1_output_name],
    )
    graph_index.replace_operator_outputs(
        int(prelu2_index),
        [plan.post2_output_name],
    )

    for update in plan.metadata_updates:
        tensor = model_ir.tensors[update.name]
        tensor.shape = [int(value) for value in update.shape]
        tensor.shape_signature = [int(value) for value in update.signature]
        tensor.logical_layout = str(update.logical_layout)
        tensor.physical_layout = str(update.physical_layout)

    graph_index.remove_operators([int(index) for index in remove_indices])
    return True


def _resolve_postmul_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_PostMulTailPlan]:
    post2_index = graph_index.operator_index(root)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    if (
        post2_index is None
        or str(root.op_type) != "TRANSPOSE"
        or len(root.inputs) != 2
        or len(root.outputs) != 1
        or not _typed_permutation(
            model_ir,
            graph_index,
            root,
            _NCHW_TO_NHWC,
            public_names,
        )
    ):
        return None
    post2 = root
    mul2_output_name = str(post2.inputs[0])
    post2_output_name = str(post2.outputs[0])
    mul2_match = _producer(model_ir, graph_index, mul2_output_name, "MUL")
    if (
        mul2_match is None
        or mul2_output_name in public_names
        or post2_output_name in public_names
        or graph_index.consumer_indices(mul2_output_name)
        != [int(post2_index)]
    ):
        return None
    mul2_index, mul2 = mul2_match
    if not _plain_binary(mul2, "MUL") or int(mul2_index) >= int(post2_index):
        return None

    add2_indices = graph_index.consumer_indices(post2_output_name)
    if len(add2_indices) != 1 or int(add2_indices[0]) <= int(post2_index):
        return None
    add2_index = int(add2_indices[0])
    add2 = model_ir.operators[add2_index]
    if (
        not _plain_binary(add2, "ADD")
        or str(post2_output_name) not in [str(name) for name in add2.inputs]
    ):
        return None
    add2_matches = [
        index
        for index, name in enumerate(add2.inputs)
        if str(name) == post2_output_name
    ]
    if len(add2_matches) != 1:
        return None
    add2_data_index = int(add2_matches[0])
    add2_constant_index = 1 - add2_data_index
    add2_constant_name = str(add2.inputs[add2_constant_index])
    add2_output_name = str(add2.outputs[0])
    if (
        add2_output_name in public_names
        or not _has_exact_producer(
            graph_index,
            add2_output_name,
            add2_index,
        )
    ):
        return None

    prelu2_indices = graph_index.consumer_indices(add2_output_name)
    if len(prelu2_indices) != 1 or int(prelu2_indices[0]) <= int(add2_index):
        return None
    prelu2_index = int(prelu2_indices[0])
    prelu2 = model_ir.operators[prelu2_index]
    if (
        not _plain_prelu(prelu2)
        or str(prelu2.inputs[0]) != add2_output_name
    ):
        return None
    prelu2_output_name = str(prelu2.outputs[0])
    if (
        not _has_exact_producer(
            graph_index,
            prelu2_output_name,
            prelu2_index,
        )
        or any(
            int(index) <= int(prelu2_index)
            for index in graph_index.consumer_indices(prelu2_output_name)
        )
    ):
        return None

    mul2_constants = [
        index
        for index, name in enumerate(mul2.inputs)
        if (tensor := model_ir.tensors.get(str(name))) is not None
        and tensor.data is not None
    ]
    if len(mul2_constants) != 1:
        return None
    mul2_constant_index = int(mul2_constants[0])
    mul2_constant_name = str(mul2.inputs[mul2_constant_index])
    concat2_output_name = str(mul2.inputs[1 - mul2_constant_index])
    concat2_match = _producer(
        model_ir,
        graph_index,
        concat2_output_name,
        "CONCATENATION",
    )
    if concat2_match is None:
        return None
    concat2_index, concat2 = concat2_match
    if (
        int(concat2_index) >= int(mul2_index)
        or graph_index.consumer_indices(concat2_output_name)
        != [int(mul2_index)]
    ):
        return None
    prefix = _resolve_prefix(
        model_ir,
        graph_index,
        concat2,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if prefix is None:
        return None

    tail_operators = (mul2, post2, add2, prelu2)
    prefix_operators = (
        prefix.pre_a,
        prefix.pre_x,
        prefix.pre_y,
        prefix.add0,
        prefix.mul1,
        prefix.add1,
        prefix.prelu1,
        prefix.post1,
        prefix.concat2,
    )
    if len({id(operator) for operator in (*prefix_operators, *tail_operators)}) != 13:
        return None
    tail_names = (
        mul2_output_name,
        post2_output_name,
        add2_output_name,
        prelu2_output_name,
    )
    if (
        len(set((*prefix.tensor_names, *tail_names)))
        != len(prefix.tensor_names) + len(tail_names)
        or any(
            name in public_names
            for name in (mul2_output_name, post2_output_name, add2_output_name)
        )
        or any(name in graph_index.duplicate_producers for name in tail_names)
    ):
        return None
    tail_contracts = {
        name: _tensor_contract(model_ir, name, 4) for name in tail_names
    }
    if any(contract is None for contract in tail_contracts.values()):
        return None
    if (
        any(
            str(contract.tensor.dtype) != prefix.dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in tail_contracts.values()
            if contract is not None
        )
        or tail_contracts[mul2_output_name].shape
        != prefix.concat_nchw_shape
        or tail_contracts[mul2_output_name].signature
        != prefix.concat_nchw_signature
        or tail_contracts[post2_output_name].shape
        != prefix.concat_nhwc_shape
        or tail_contracts[post2_output_name].signature
        != prefix.concat_nhwc_signature
        or tail_contracts[add2_output_name].shape
        != prefix.concat_nhwc_shape
        or tail_contracts[add2_output_name].signature
        != prefix.concat_nhwc_signature
        or tail_contracts[prelu2_output_name].shape
        != prefix.concat_nhwc_shape
        or tail_contracts[prelu2_output_name].signature
        != prefix.concat_nhwc_signature
    ):
        return None

    constant_roles = list(prefix.constant_roles)
    for name, operator, input_index in (
        (mul2_constant_name, mul2, mul2_constant_index),
        (add2_constant_name, add2, add2_constant_index),
        (str(prelu2.inputs[1]), prelu2, 1),
    ):
        replacement = _constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=prefix.dtype,
            target_shape=prefix.concat_nhwc_shape,
            public_names=public_names,
        )
        if replacement is None:
            return None
        constant_roles.append(
            (str(name), replacement, operator, int(input_index))
        )
    constant_plans = _plan_constants(
        model_ir,
        graph_index,
        tuple(constant_roles),
    )
    if constant_plans is None:
        return None

    return _PostMulTailPlan(
        root=root,
        prefix=prefix,
        mul2=mul2,
        post2=post2,
        add2=add2,
        prelu2=prelu2,
        mul2_output_name=mul2_output_name,
        post2_output_name=post2_output_name,
        add2_output_name=add2_output_name,
        prelu2_output_name=prelu2_output_name,
        constant_plans=constant_plans,
        metadata_updates=prefix.metadata_updates
        + (
            _metadata_update(
                prefix.concat2_output_name,
                tail_contracts[post2_output_name].tensor,
            ),
        ),
        remove_operators=prefix.remove_operators + (post2,),
    )


def _prefixes_equal(
    expected: _ResidualPrefixPlan,
    actual: _ResidualPrefixPlan,
) -> bool:
    operator_fields = (
        "pre_a",
        "pre_x",
        "pre_y",
        "add0",
        "mul1",
        "add1",
        "prelu1",
        "post1",
        "concat2",
    )
    return bool(
        all(
            getattr(expected, field) is getattr(actual, field)
            for field in operator_fields
        )
        and expected.add0_inputs == actual.add0_inputs
        and expected.concat2_inputs == actual.concat2_inputs
        and expected.prelu1_output_name == actual.prelu1_output_name
        and expected.post1_output_name == actual.post1_output_name
        and expected.concat2_output_name == actual.concat2_output_name
        and expected.dtype == actual.dtype
        and expected.concat_nchw_shape == actual.concat_nchw_shape
        and expected.concat_nchw_signature == actual.concat_nchw_signature
        and expected.concat_nhwc_shape == actual.concat_nhwc_shape
        and expected.concat_nhwc_signature == actual.concat_nhwc_signature
        and expected.tensor_names == actual.tensor_names
        and expected.metadata_updates == actual.metadata_updates
        and len(expected.remove_operators) == len(actual.remove_operators)
        and all(
            left is right
            for left, right in zip(
                expected.remove_operators,
                actual.remove_operators,
            )
        )
    )


def _postmul_plans_equal(
    expected: _PostMulTailPlan,
    actual: _PostMulTailPlan,
) -> bool:
    return bool(
        expected.root is actual.root
        and expected.mul2 is actual.mul2
        and expected.post2 is actual.post2
        and expected.add2 is actual.add2
        and expected.prelu2 is actual.prelu2
        and expected.mul2_output_name == actual.mul2_output_name
        and expected.post2_output_name == actual.post2_output_name
        and expected.add2_output_name == actual.add2_output_name
        and expected.prelu2_output_name == actual.prelu2_output_name
        and expected.metadata_updates == actual.metadata_updates
        and _prefixes_equal(expected.prefix, actual.prefix)
        and _constant_plans_equal(
            expected.constant_plans,
            actual.constant_plans,
        )
        and len(expected.remove_operators) == len(actual.remove_operators)
        and all(
            left is right
            for left, right in zip(
                expected.remove_operators,
                actual.remove_operators,
            )
        )
    )


def _apply_postmul_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _PostMulTailPlan,
) -> bool:
    current = _resolve_postmul_candidate(model_ir, graph_index, plan.root)
    if current is None or not _postmul_plans_equal(plan, current):
        return False
    prefix = plan.prefix
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    add0_index = graph_index.operator_index(prefix.add0)
    concat2_index = graph_index.operator_index(prefix.concat2)
    prelu1_index = graph_index.operator_index(prefix.prelu1)
    mul2_index = graph_index.operator_index(plan.mul2)
    if (
        any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or any(
            constant.clone_name is not None
            and constant.clone_name in model_ir.tensors
            for constant in plan.constant_plans
        )
        or any(
            update.name not in model_ir.tensors
            for update in plan.metadata_updates
        )
        or any(
            index is None
            for index in (
                add0_index,
                concat2_index,
                prelu1_index,
                mul2_index,
            )
        )
    ):
        return False

    for constant in plan.constant_plans:
        target = constant.tensor
        if constant.clone_name is not None:
            target = TensorIR(
                name=str(constant.clone_name),
                dtype=str(constant.tensor.dtype),
                shape=[int(value) for value in constant.data.shape],
                shape_signature=[int(value) for value in constant.data.shape],
                data=np.asarray(constant.data),
                is_variable=False,
                quantization=None,
                logical_layout=str(constant.tensor.logical_layout),
                physical_layout=str(constant.tensor.physical_layout),
                onnx_tensor_name=constant.tensor.onnx_tensor_name,
            )
            model_ir.tensors[str(constant.clone_name)] = target
            for use in constant.uses:
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=use.operator,
                    input_index=int(use.input_index),
                    new_input_name=str(constant.clone_name),
                    graph_index=graph_index,
                )
        target.data = np.asarray(constant.data)
        target.shape = [int(value) for value in constant.data.shape]
        target.shape_signature = [int(value) for value in constant.data.shape]

    graph_index.replace_operator_inputs(int(add0_index), prefix.add0_inputs)
    graph_index.replace_operator_inputs(
        int(concat2_index),
        prefix.concat2_inputs,
    )
    prefix.concat2.options["axis"] = 3
    graph_index.replace_operator_outputs(
        int(prelu1_index),
        [prefix.post1_output_name],
    )
    graph_index.replace_operator_outputs(
        int(mul2_index),
        [plan.post2_output_name],
    )
    for update in plan.metadata_updates:
        tensor = model_ir.tensors[update.name]
        tensor.shape = [int(value) for value in update.shape]
        tensor.shape_signature = [int(value) for value in update.signature]
        tensor.logical_layout = str(update.logical_layout)
        tensor.physical_layout = str(update.physical_layout)
    graph_index.remove_operators([int(index) for index in remove_indices])
    return True


def optimize_sinet_shuffle_residual_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift strict two-stage SiNet Shuffle residual islands to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 5,
        "ADD": 3,
        "MUL": 2,
        "PRELU": 2,
        "CONCATENATION": 1,
    }
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if all(value == 0 for value in required_counts.values()):
            break
    if rewrite_limit == 0 or any(
        value > 0 for value in required_counts.values()
    ):
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
    for root in candidates:
        if rewritten >= rewrite_limit or root is None:
            break
        if active_index.operator_index(root) is None:
            continue
        plan = _resolve_candidate(model_ir, active_index, root)
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}


def optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift the strict SiNet post-MUL adapter variant to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 5,
        "ADD": 3,
        "MUL": 2,
        "PRELU": 2,
        "CONCATENATION": 1,
    }
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if all(value == 0 for value in required_counts.values()):
            break
    if rewrite_limit == 0 or any(
        value > 0 for value in required_counts.values()
    ):
        return {_POSTMUL_STATS_KEY: 0}

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
    for root in candidates:
        if rewritten >= rewrite_limit or root is None:
            break
        if active_index.operator_index(root) is None:
            continue
        plan = _resolve_postmul_candidate(model_ir, active_index, root)
        if plan is not None and _apply_postmul_plan(
            model_ir,
            active_index,
            plan,
        ):
            rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_POSTMUL_STATS_KEY: int(rewritten)}


def optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a strict fan-out residual affine island to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 3,
        "ADD": 2,
        "MUL": 1,
        "PRELU": 1,
        "CONCATENATION": 1,
    }
    has_downstream = False
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
            has_downstream = True
        if has_downstream and all(
            value == 0 for value in required_counts.values()
        ):
            break
    if (
        rewrite_limit == 0
        or not has_downstream
        or any(value > 0 for value in required_counts.values())
    ):
        return {_LATE_STATS_KEY: 0}

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
    for root in candidates:
        if rewritten >= rewrite_limit or root is None:
            break
        if active_index.operator_index(root) is None:
            continue
        plan = _resolve_late_candidate(model_ir, active_index, root)
        if plan is not None and _apply_late_plan(
            model_ir,
            active_index,
            plan,
        ):
            rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_LATE_STATS_KEY: int(rewritten)}
