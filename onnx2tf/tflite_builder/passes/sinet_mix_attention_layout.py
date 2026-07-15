from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

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
    _data_and_constant_inputs,
    _permute,
    _plain_binary,
    _tensor_contract,
    _typed_permutation,
)
from onnx2tf.tflite_builder.passes.sinet_concat_resize_layout import (
    _InputRewrite,
    _layout_allows,
)
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    _ConstantPlan,
    _MetadataUpdate,
    _apply_constant_plans,
    _apply_metadata_updates,
    _constant_plans_equal,
    _plan_constants,
    _producer,
    _resolved_source,
)


_STATS_KEY = "optimized_sinet_mix_attention_double_logistic_nhwc_chains"
_NCKHW_TO_NHWCK = (0, 3, 4, 1, 2)


@dataclass(frozen=True)
class _AdapterMatch:
    operator: OperatorIR
    source_name: str
    output_name: str


@dataclass(frozen=True)
class _ResidualMatch:
    operator: OperatorIR
    adapters: Tuple[_AdapterMatch, ...]
    canonical_name: str
    output_name: str
    add_mode: bool


@dataclass(frozen=True)
class _TailMatch:
    root: OperatorIR
    post_conv: OperatorIR
    add3: OperatorIR
    add2: OperatorIR
    mul1: OperatorIR
    source: OperatorIR
    mul0: OperatorIR
    sub: OperatorIR
    gate1: OperatorIR
    gate2: OperatorIR
    branch: _AdapterMatch
    residual: _ResidualMatch
    source_name: str
    gate1_name: str
    gate2_name: str
    sub_name: str
    mul0_name: str
    add2_name: str
    mul1_name: str
    add3_name: str
    post_output_name: str
    sub_constant_name: str


@dataclass(frozen=True)
class _AttentionMatch:
    mean_ca: OperatorIR
    ca_pre: OperatorIR
    ca_conv0: OperatorIR
    ca_conv2: OperatorIR
    ca_post: OperatorIR
    mean_sa: OperatorIR
    max_sa: OperatorIR
    concat_sa: OperatorIR
    mirror_sa: OperatorIR
    sa_pre: OperatorIR
    sa_conv: OperatorIR
    sa_reshape: OperatorIR
    add_attention: OperatorIR
    unsqueeze_source: OperatorIR
    unsqueeze_attention: OperatorIR
    concat_pa: OperatorIR
    reshape_pa: OperatorIR
    mirror_pa: OperatorIR
    pa_pre: OperatorIR
    pa_conv: OperatorIR
    pa_post: OperatorIR
    ca_output_name: str
    sa_output_name: str
    attention_name: str
    mean_ca_name: str
    mean_sa_name: str
    max_sa_name: str
    concat_sa_name: str
    mirror_sa_name: str
    unsqueeze_source_name: str
    unsqueeze_attention_name: str
    concat_pa_name: str
    reshape_pa_name: str
    mirror_pa_name: str
    pa_conv_name: str


@dataclass(frozen=True)
class _OptionsUpdate:
    operator: OperatorIR
    options: dict[str, object]


@dataclass(frozen=True)
class _MixAttentionPlan:
    root: OperatorIR
    tail: _TailMatch
    attention: _AttentionMatch
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    input_rewrites: Tuple[_InputRewrite, ...]
    options_updates: Tuple[_OptionsUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]


def _operator_index(
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
) -> Optional[int]:
    value = graph_index.operator_index(operator)
    return None if value is None else int(value)


def _data_contract(
    model_ir: ModelIR,
    name: str,
    *,
    rank: int,
    dtype: Optional[str] = None,
    layout: Optional[str] = None,
) -> Optional[Tuple[TensorIR, Tuple[int, ...], Tuple[int, ...]]]:
    contract = _tensor_contract(model_ir, str(name), int(rank))
    if contract is None:
        return None
    tensor = contract.tensor
    normalized_dtype = str(tensor.dtype)
    if (
        normalized_dtype not in _FLOAT_DTYPES
        or (dtype is not None and normalized_dtype != str(dtype))
        or tensor.is_variable
        or tensor.quantization is not None
        or (layout is not None and not _layout_allows(tensor, str(layout)))
    ):
        return None
    return tensor, contract.shape, contract.signature


def _exact_contract(
    model_ir: ModelIR,
    name: str,
    *,
    dtype: str,
    shape: Sequence[int],
    signature: Sequence[int],
    layout: Optional[str] = None,
) -> Optional[TensorIR]:
    contract = _data_contract(
        model_ir,
        str(name),
        rank=len(tuple(shape)),
        dtype=str(dtype),
        layout=layout,
    )
    if contract is None:
        return None
    tensor, actual_shape, actual_signature = contract
    if actual_shape != tuple(shape) or actual_signature != tuple(signature):
        return None
    return tensor


def _plain_unary(operator: OperatorIR, op_type: str) -> bool:
    return bool(
        str(operator.op_type) == str(op_type)
        and len(operator.inputs) == 1
        and len(operator.outputs) == 1
        and str(
            operator.options.get("fusedActivationFunction", "NONE")
        ).upper()
        == "NONE"
    )


def _plain_concat(operator: OperatorIR, *, axis: int, rank: int) -> bool:
    if (
        str(operator.op_type) != "CONCATENATION"
        or len(operator.inputs) != 2
        or len(operator.outputs) != 1
        or str(
            operator.options.get("fusedActivationFunction", "NONE")
        ).upper()
        != "NONE"
    ):
        return False
    try:
        value = int(operator.options.get("axis", int(axis)))
    except (TypeError, ValueError):
        return False
    if value < 0:
        value += int(rank)
    return value == int(axis)


def _plain_reduce(
    operator: OperatorIR,
    op_type: str,
    *,
    keep_dims: bool,
) -> bool:
    return bool(
        str(operator.op_type) == str(op_type)
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
        and bool(operator.options.get("keepDims", False)) is bool(keep_dims)
    )


def _plain_reshape(operator: OperatorIR) -> bool:
    return bool(
        str(operator.op_type) == "RESHAPE"
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
    )


def _plain_mirror_pad(operator: OperatorIR) -> bool:
    return bool(
        str(operator.op_type) == "MIRROR_PAD"
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
    )


def _plain_conv(operator: OperatorIR) -> bool:
    return bool(
        str(operator.op_type) == "CONV_2D"
        and len(operator.inputs) >= 1
        and len(operator.outputs) == 1
    )


def _exact_consumers(
    graph_index: ModelIRGraphIndex,
    name: str,
    operators: Sequence[OperatorIR],
) -> bool:
    indices = [_operator_index(graph_index, operator) for operator in operators]
    return bool(
        all(index is not None for index in indices)
        and Counter(graph_index.consumer_indices(str(name)))
        == Counter(int(index) for index in indices if index is not None)
    )


def _sole_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    producer_index: int,
    op_type: str,
    public_names: set[str],
) -> Optional[Tuple[int, OperatorIR]]:
    consumers = graph_index.consumer_indices(str(name))
    if (
        str(name) in public_names
        or str(name) in graph_index.duplicate_producers
        or graph_index.producers.get(str(name)) != int(producer_index)
        or len(consumers) != 1
    ):
        return None
    consumer_index = int(consumers[0])
    if consumer_index <= int(producer_index):
        return None
    consumer = model_ir.operators[consumer_index]
    if (
        str(consumer.op_type) != str(op_type)
        or len(consumer.outputs) != 1
        or str(consumer.outputs[0]) in graph_index.duplicate_producers
        or graph_index.producers.get(str(consumer.outputs[0]))
        != consumer_index
    ):
        return None
    return consumer_index, consumer


def _producer_role(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    op_type: str,
) -> Optional[Tuple[int, str, int, OperatorIR]]:
    matches = []
    for input_index, input_name in enumerate(operator.inputs):
        match = _producer(
            model_ir,
            graph_index,
            str(input_name),
            str(op_type),
        )
        if match is not None:
            matches.append(
                (int(input_index), str(input_name), int(match[0]), match[1])
            )
    return matches[0] if len(matches) == 1 else None


def _resolve_adapter(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    output_name: str,
    consumers: Sequence[OperatorIR],
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_AdapterMatch]:
    public_names = public_inputs | public_outputs
    match = _producer(model_ir, graph_index, str(output_name), "TRANSPOSE")
    if match is None:
        return None
    adapter_index, adapter = match
    if (
        str(output_name) in public_names
        or not _exact_consumers(
            graph_index,
            str(output_name),
            tuple(consumers),
        )
        or not _typed_permutation(
            model_ir,
            graph_index,
            adapter,
            _NHWC_TO_NCHW,
            public_names,
        )
    ):
        return None
    source_name = str(adapter.inputs[0])
    if not _resolved_source(
        graph_index,
        name=source_name,
        adapter_index=int(adapter_index),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    ):
        return None
    return _AdapterMatch(adapter, source_name, str(output_name))


def _singleton_float_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    dtype: str,
    public_names: set[str],
) -> bool:
    tensor = model_ir.tensors.get(str(name))
    expected_dtype = _FLOAT_DTYPES.get(str(dtype))
    if tensor is None or tensor.data is None or expected_dtype is None:
        return False
    try:
        data = np.asarray(tensor.data)
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
    except (TypeError, ValueError):
        return False
    return bool(
        str(name) not in public_names
        and str(name) not in graph_index.producers
        and str(name) not in graph_index.duplicate_producers
        and str(tensor.dtype) == str(dtype)
        and not tensor.is_variable
        and tensor.quantization is None
        and data.dtype == expected_dtype
        and int(data.size) == 1
        and shape in {(), (1,)}
        and signature == shape
        and np.all(np.isfinite(data))
    )


def _resolve_tail(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
    *,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_TailMatch]:
    public_names = public_inputs | public_outputs
    root_index = _operator_index(graph_index, root)
    if (
        root_index is None
        or str(root.op_type) != "TRANSPOSE"
        or len(root.outputs) != 1
        or str(root.outputs[0]) in public_names
        or graph_index.producers.get(str(root.outputs[0])) != root_index
        or str(root.outputs[0]) in graph_index.duplicate_producers
        or not _typed_permutation(
            model_ir,
            graph_index,
            root,
            _NCHW_TO_NHWC,
            public_names,
        )
    ):
        return None
    add3_name = str(root.inputs[0])
    add3_match = _producer(model_ir, graph_index, add3_name, "ADD")
    if add3_match is None or add3_name in public_names:
        return None
    add3_index, add3 = add3_match
    if not _plain_binary(add3, "ADD") or add3_index >= root_index:
        return None
    post_output_name = str(root.outputs[0])
    post_match = _sole_consumer(
        model_ir,
        graph_index,
        name=post_output_name,
        producer_index=root_index,
        op_type="CONV_2D",
        public_names=public_names,
    )
    if post_match is None:
        return None
    post_index, post_conv = post_match
    if not _plain_conv(post_conv) or str(post_conv.inputs[0]) != post_output_name:
        return None

    add2_role = _producer_role(model_ir, graph_index, add3, "ADD")
    mul1_role = _producer_role(model_ir, graph_index, add3, "MUL")
    if (
        add2_role is None
        or mul1_role is None
        or add2_role[0] == mul1_role[0]
        or add2_role[2] >= add3_index
        or mul1_role[2] >= add3_index
    ):
        return None
    _, add2_name, add2_index, add2 = add2_role
    _, mul1_name, mul1_index, mul1 = mul1_role
    if (
        not _plain_binary(add2, "ADD")
        or not _plain_binary(mul1, "MUL")
        or not _exact_consumers(graph_index, add2_name, (add3,))
        or not _exact_consumers(graph_index, mul1_name, (add3,))
    ):
        return None

    source_role = _producer_role(model_ir, graph_index, add2, "ADD")
    mul0_role = _producer_role(model_ir, graph_index, add2, "MUL")
    if (
        source_role is None
        or mul0_role is None
        or source_role[0] == mul0_role[0]
        or source_role[2] >= add2_index
        or mul0_role[2] >= add2_index
    ):
        return None
    _, source_name, source_index, source = source_role
    _, mul0_name, mul0_index, mul0 = mul0_role
    if (
        not _plain_binary(source, "ADD")
        or not _plain_binary(mul0, "MUL")
        or not _exact_consumers(graph_index, mul0_name, (add2,))
        or source_name in public_names
    ):
        return None

    sub_role = _producer_role(model_ir, graph_index, mul1, "SUB")
    if sub_role is None or sub_role[2] >= mul1_index:
        return None
    _, sub_name, sub_index, sub = sub_role
    if (
        not _plain_binary(sub, "SUB")
        or not _exact_consumers(graph_index, sub_name, (mul1,))
    ):
        return None

    gate2_inputs = [str(name) for name in sub.inputs]
    gate2_candidates = []
    for input_name in gate2_inputs:
        match = _producer(model_ir, graph_index, input_name, "LOGISTIC")
        if match is not None:
            gate2_candidates.append((input_name, match[0], match[1]))
    if len(gate2_candidates) != 1:
        return None
    gate2_name, gate2_index, gate2 = gate2_candidates[0]
    if not _plain_unary(gate2, "LOGISTIC") or gate2_index >= sub_index:
        return None
    sub_inputs = _data_and_constant_inputs(model_ir, sub)
    if sub_inputs is None or sub_inputs[1] != gate2_name:
        return None
    sub_constant_name = str(sub_inputs[3])

    gate1_name = str(gate2.inputs[0])
    gate1_match = _producer(model_ir, graph_index, gate1_name, "LOGISTIC")
    if gate1_match is None:
        return None
    gate1_index, gate1 = gate1_match
    if (
        not _plain_unary(gate1, "LOGISTIC")
        or gate1_index >= gate2_index
        or not _exact_consumers(graph_index, gate1_name, (gate2,))
        or not _exact_consumers(graph_index, gate2_name, (mul0, sub))
    ):
        return None

    residual_candidates = []
    for input_name in mul1.inputs:
        if str(input_name) == sub_name:
            continue
        if str(input_name) in source.inputs:
            residual_candidates.append(str(input_name))
    if len(residual_candidates) != 1:
        return None
    residual_name = residual_candidates[0]
    branch_candidates = [
        str(name) for name in source.inputs if str(name) != residual_name
    ]
    if len(branch_candidates) != 1:
        return None
    branch_name = branch_candidates[0]
    if Counter(str(name) for name in mul0.inputs) != Counter(
        (branch_name, gate2_name)
    ):
        return None

    branch = _resolve_adapter(
        model_ir,
        graph_index,
        output_name=branch_name,
        consumers=(source, mul0),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if branch is None:
        return None

    residual_match = _producer(
        model_ir,
        graph_index,
        residual_name,
        "ADD",
    )
    if residual_match is not None:
        residual_index, residual_operator = residual_match
        if (
            not _plain_binary(residual_operator, "ADD")
            or residual_index >= source_index
            or not _exact_consumers(
                graph_index,
                residual_name,
                (source, mul1),
            )
        ):
            return None
        adapters = []
        for input_name in residual_operator.inputs:
            adapter = _resolve_adapter(
                model_ir,
                graph_index,
                output_name=str(input_name),
                consumers=(residual_operator,),
                public_inputs=public_inputs,
                public_outputs=public_outputs,
            )
            if adapter is None:
                return None
            adapters.append(adapter)
        residual = _ResidualMatch(
            operator=residual_operator,
            adapters=tuple(adapters),
            canonical_name=residual_name,
            output_name=residual_name,
            add_mode=True,
        )
    else:
        adapter = _resolve_adapter(
            model_ir,
            graph_index,
            output_name=residual_name,
            consumers=(source, mul1),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
        if adapter is None:
            return None
        residual = _ResidualMatch(
            operator=adapter.operator,
            adapters=(adapter,),
            canonical_name=adapter.source_name,
            output_name=residual_name,
            add_mode=False,
        )
    if not _singleton_float_constant(
        model_ir,
        graph_index,
        name=sub_constant_name,
        dtype=str(model_ir.tensors[post_output_name].dtype),
        public_names=public_names,
    ):
        return None
    return _TailMatch(
        root=root,
        post_conv=post_conv,
        add3=add3,
        add2=add2,
        mul1=mul1,
        source=source,
        mul0=mul0,
        sub=sub,
        gate1=gate1,
        gate2=gate2,
        branch=branch,
        residual=residual,
        source_name=source_name,
        gate1_name=gate1_name,
        gate2_name=gate2_name,
        sub_name=sub_name,
        mul0_name=mul0_name,
        add2_name=add2_name,
        mul1_name=mul1_name,
        add3_name=add3_name,
        post_output_name=post_output_name,
        sub_constant_name=sub_constant_name,
    )


def _resolve_ca_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    mean: OperatorIR,
    *,
    public_names: set[str],
) -> Optional[Tuple[OperatorIR, OperatorIR, OperatorIR, OperatorIR]]:
    mean_index = _operator_index(graph_index, mean)
    if mean_index is None or not _plain_reduce(mean, "MEAN", keep_dims=True):
        return None
    mean_name = str(mean.outputs[0])
    pre_match = _sole_consumer(
        model_ir,
        graph_index,
        name=mean_name,
        producer_index=mean_index,
        op_type="TRANSPOSE",
        public_names=public_names,
    )
    if pre_match is None:
        return None
    pre_index, pre = pre_match
    if not _typed_permutation(
        model_ir,
        graph_index,
        pre,
        _NCHW_TO_NHWC,
        public_names,
    ):
        return None
    conv0_match = _sole_consumer(
        model_ir,
        graph_index,
        name=str(pre.outputs[0]),
        producer_index=pre_index,
        op_type="CONV_2D",
        public_names=public_names,
    )
    if conv0_match is None:
        return None
    conv0_index, conv0 = conv0_match
    if not _plain_conv(conv0) or str(conv0.inputs[0]) != str(pre.outputs[0]):
        return None
    conv2_match = _sole_consumer(
        model_ir,
        graph_index,
        name=str(conv0.outputs[0]),
        producer_index=conv0_index,
        op_type="CONV_2D",
        public_names=public_names,
    )
    if conv2_match is None:
        return None
    conv2_index, conv2 = conv2_match
    if not _plain_conv(conv2) or str(conv2.inputs[0]) != str(conv0.outputs[0]):
        return None
    post_match = _sole_consumer(
        model_ir,
        graph_index,
        name=str(conv2.outputs[0]),
        producer_index=conv2_index,
        op_type="TRANSPOSE",
        public_names=public_names,
    )
    if post_match is None:
        return None
    _, post = post_match
    if not _typed_permutation(
        model_ir,
        graph_index,
        post,
        _NHWC_TO_NCHW,
        public_names,
    ):
        return None
    return pre, conv0, conv2, post


def _resolve_attention(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tail: _TailMatch,
    *,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_AttentionMatch]:
    del public_inputs
    public_names = public_outputs | {str(name) for name in model_ir.inputs}
    source_index = _operator_index(graph_index, tail.source)
    if source_index is None:
        return None
    source_users = graph_index.consumer_indices(tail.source_name)
    unique_users = sorted(set(source_users))
    if len(source_users) != 5 or len(unique_users) != 5:
        return None
    operators = [model_ir.operators[index] for index in unique_users]
    means = [operator for operator in operators if str(operator.op_type) == "MEAN"]
    maxima = [
        operator
        for operator in operators
        if str(operator.op_type) == "REDUCE_MAX"
    ]
    reshapes = [
        operator for operator in operators if str(operator.op_type) == "RESHAPE"
    ]
    if (
        len(means) != 2
        or len(maxima) != 1
        or len(reshapes) != 1
        or tail.add2 not in operators
        or any(index <= source_index for index in unique_users)
    ):
        return None
    max_sa = maxima[0]
    unsqueeze_source = reshapes[0]
    if (
        not _plain_reduce(max_sa, "REDUCE_MAX", keep_dims=True)
        or not _plain_reshape(unsqueeze_source)
        or str(max_sa.inputs[0]) != tail.source_name
        or str(unsqueeze_source.inputs[0]) != tail.source_name
        or any(str(mean.inputs[0]) != tail.source_name for mean in means)
    ):
        return None

    ca_matches = []
    for mean in means:
        match = _resolve_ca_branch(
            model_ir,
            graph_index,
            mean,
            public_names=public_names,
        )
        if match is not None:
            ca_matches.append((mean, match))
    if len(ca_matches) != 1:
        return None
    mean_ca, (ca_pre, ca_conv0, ca_conv2, ca_post) = ca_matches[0]
    mean_sa = next((mean for mean in means if mean is not mean_ca), None)
    if mean_sa is None or not _plain_reduce(mean_sa, "MEAN", keep_dims=True):
        return None

    mean_sa_index = _operator_index(graph_index, mean_sa)
    max_sa_index = _operator_index(graph_index, max_sa)
    if mean_sa_index is None or max_sa_index is None:
        return None
    mean_sa_name = str(mean_sa.outputs[0])
    max_sa_name = str(max_sa.outputs[0])
    mean_sa_users = graph_index.consumer_indices(mean_sa_name)
    max_sa_users = graph_index.consumer_indices(max_sa_name)
    if (
        len(mean_sa_users) != 1
        or mean_sa_users != max_sa_users
        or mean_sa_name in public_names
        or max_sa_name in public_names
    ):
        return None
    concat_sa_index = int(mean_sa_users[0])
    if concat_sa_index <= max(mean_sa_index, max_sa_index):
        return None
    concat_sa = model_ir.operators[concat_sa_index]
    if (
        not _plain_concat(concat_sa, axis=1, rank=4)
        or Counter(str(name) for name in concat_sa.inputs)
        != Counter((mean_sa_name, max_sa_name))
    ):
        return None
    concat_sa_name = str(concat_sa.outputs[0])

    mirror_sa_match = _sole_consumer(
        model_ir,
        graph_index,
        name=concat_sa_name,
        producer_index=concat_sa_index,
        op_type="MIRROR_PAD",
        public_names=public_names,
    )
    if mirror_sa_match is None:
        return None
    mirror_sa_index, mirror_sa = mirror_sa_match
    if not _plain_mirror_pad(mirror_sa) or str(mirror_sa.inputs[0]) != concat_sa_name:
        return None
    mirror_sa_name = str(mirror_sa.outputs[0])
    sa_pre_match = _sole_consumer(
        model_ir,
        graph_index,
        name=mirror_sa_name,
        producer_index=mirror_sa_index,
        op_type="TRANSPOSE",
        public_names=public_names,
    )
    if sa_pre_match is None:
        return None
    sa_pre_index, sa_pre = sa_pre_match
    if not _typed_permutation(
        model_ir,
        graph_index,
        sa_pre,
        _NCHW_TO_NHWC,
        public_names,
    ):
        return None
    sa_conv_match = _sole_consumer(
        model_ir,
        graph_index,
        name=str(sa_pre.outputs[0]),
        producer_index=sa_pre_index,
        op_type="CONV_2D",
        public_names=public_names,
    )
    if sa_conv_match is None:
        return None
    sa_conv_index, sa_conv = sa_conv_match
    if not _plain_conv(sa_conv) or str(sa_conv.inputs[0]) != str(sa_pre.outputs[0]):
        return None
    sa_reshape_match = _sole_consumer(
        model_ir,
        graph_index,
        name=str(sa_conv.outputs[0]),
        producer_index=sa_conv_index,
        op_type="RESHAPE",
        public_names=public_names,
    )
    if sa_reshape_match is None:
        return None
    sa_reshape_index, sa_reshape = sa_reshape_match
    if not _plain_reshape(sa_reshape):
        return None
    sa_output_name = str(sa_reshape.outputs[0])
    attention_match = _sole_consumer(
        model_ir,
        graph_index,
        name=sa_output_name,
        producer_index=sa_reshape_index,
        op_type="ADD",
        public_names=public_names,
    )
    if attention_match is None:
        return None
    attention_index, add_attention = attention_match
    if not _plain_binary(add_attention, "ADD"):
        return None
    ca_output_name = str(ca_post.outputs[0])
    if (
        Counter(str(name) for name in add_attention.inputs)
        != Counter((ca_output_name, sa_output_name))
        or not _exact_consumers(
            graph_index,
            ca_output_name,
            (add_attention,),
        )
    ):
        return None
    attention_name = str(add_attention.outputs[0])
    unsqueeze_attention_match = _sole_consumer(
        model_ir,
        graph_index,
        name=attention_name,
        producer_index=attention_index,
        op_type="RESHAPE",
        public_names=public_names,
    )
    if unsqueeze_attention_match is None:
        return None
    unsqueeze_attention_index, unsqueeze_attention = unsqueeze_attention_match
    if not _plain_reshape(unsqueeze_attention):
        return None

    unsqueeze_source_index = _operator_index(graph_index, unsqueeze_source)
    if unsqueeze_source_index is None:
        return None
    unsqueeze_source_name = str(unsqueeze_source.outputs[0])
    unsqueeze_attention_name = str(unsqueeze_attention.outputs[0])
    source_unsqueeze_users = graph_index.consumer_indices(unsqueeze_source_name)
    attention_unsqueeze_users = graph_index.consumer_indices(
        unsqueeze_attention_name
    )
    if (
        len(source_unsqueeze_users) != 1
        or source_unsqueeze_users != attention_unsqueeze_users
    ):
        return None
    concat_pa_index = int(source_unsqueeze_users[0])
    if concat_pa_index <= max(
        unsqueeze_source_index,
        unsqueeze_attention_index,
    ):
        return None
    concat_pa = model_ir.operators[concat_pa_index]
    if (
        not _plain_concat(concat_pa, axis=2, rank=5)
        or Counter(str(name) for name in concat_pa.inputs)
        != Counter((unsqueeze_source_name, unsqueeze_attention_name))
    ):
        return None
    concat_pa_name = str(concat_pa.outputs[0])
    reshape_pa_match = _sole_consumer(
        model_ir,
        graph_index,
        name=concat_pa_name,
        producer_index=concat_pa_index,
        op_type="RESHAPE",
        public_names=public_names,
    )
    if reshape_pa_match is None:
        return None
    reshape_pa_index, reshape_pa = reshape_pa_match
    if not _plain_reshape(reshape_pa):
        return None
    reshape_pa_name = str(reshape_pa.outputs[0])
    mirror_pa_match = _sole_consumer(
        model_ir,
        graph_index,
        name=reshape_pa_name,
        producer_index=reshape_pa_index,
        op_type="MIRROR_PAD",
        public_names=public_names,
    )
    if mirror_pa_match is None:
        return None
    mirror_pa_index, mirror_pa = mirror_pa_match
    if not _plain_mirror_pad(mirror_pa):
        return None
    mirror_pa_name = str(mirror_pa.outputs[0])
    pa_pre_match = _sole_consumer(
        model_ir,
        graph_index,
        name=mirror_pa_name,
        producer_index=mirror_pa_index,
        op_type="TRANSPOSE",
        public_names=public_names,
    )
    if pa_pre_match is None:
        return None
    pa_pre_index, pa_pre = pa_pre_match
    if not _typed_permutation(
        model_ir,
        graph_index,
        pa_pre,
        _NCHW_TO_NHWC,
        public_names,
    ):
        return None
    pa_conv_match = _sole_consumer(
        model_ir,
        graph_index,
        name=str(pa_pre.outputs[0]),
        producer_index=pa_pre_index,
        op_type="CONV_2D",
        public_names=public_names,
    )
    if pa_conv_match is None:
        return None
    pa_conv_index, pa_conv = pa_conv_match
    if not _plain_conv(pa_conv):
        return None
    pa_conv_name = str(pa_conv.outputs[0])
    pa_post_match = _sole_consumer(
        model_ir,
        graph_index,
        name=pa_conv_name,
        producer_index=pa_conv_index,
        op_type="TRANSPOSE",
        public_names=public_names,
    )
    if pa_post_match is None:
        return None
    _, pa_post = pa_post_match
    if (
        not _typed_permutation(
            model_ir,
            graph_index,
            pa_post,
            _NHWC_TO_NCHW,
            public_names,
        )
        or str(tail.gate1.inputs[0]) != str(pa_post.outputs[0])
        or not _exact_consumers(
            graph_index,
            str(pa_post.outputs[0]),
            (tail.gate1,),
        )
    ):
        return None
    return _AttentionMatch(
        mean_ca=mean_ca,
        ca_pre=ca_pre,
        ca_conv0=ca_conv0,
        ca_conv2=ca_conv2,
        ca_post=ca_post,
        mean_sa=mean_sa,
        max_sa=max_sa,
        concat_sa=concat_sa,
        mirror_sa=mirror_sa,
        sa_pre=sa_pre,
        sa_conv=sa_conv,
        sa_reshape=sa_reshape,
        add_attention=add_attention,
        unsqueeze_source=unsqueeze_source,
        unsqueeze_attention=unsqueeze_attention,
        concat_pa=concat_pa,
        reshape_pa=reshape_pa,
        mirror_pa=mirror_pa,
        pa_pre=pa_pre,
        pa_conv=pa_conv,
        pa_post=pa_post,
        ca_output_name=ca_output_name,
        sa_output_name=sa_output_name,
        attention_name=attention_name,
        mean_ca_name=str(mean_ca.outputs[0]),
        mean_sa_name=mean_sa_name,
        max_sa_name=max_sa_name,
        concat_sa_name=concat_sa_name,
        mirror_sa_name=mirror_sa_name,
        unsqueeze_source_name=unsqueeze_source_name,
        unsqueeze_attention_name=unsqueeze_attention_name,
        concat_pa_name=concat_pa_name,
        reshape_pa_name=reshape_pa_name,
        mirror_pa_name=mirror_pa_name,
        pa_conv_name=pa_conv_name,
    )


def _int_tensor(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    public_names: set[str],
) -> Optional[Tuple[TensorIR, np.ndarray, np.dtype]]:
    tensor = model_ir.tensors.get(str(name))
    if tensor is None or tensor.data is None:
        return None
    expected_dtype = {
        "INT32": np.dtype(np.int32),
        "INT64": np.dtype(np.int64),
    }.get(str(tensor.dtype))
    if expected_dtype is None:
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
        or tensor.is_variable
        or tensor.quantization is not None
        or data.dtype != expected_dtype
        or tuple(int(value) for value in data.shape) != shape
        or signature != shape
    ):
        return None
    return tensor, data, expected_dtype


def _axis_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    public_names: set[str],
) -> Optional[Tuple[str, np.ndarray]]:
    if len(input_shape) != 4 or len(operator.inputs) != 2:
        return None
    name = str(operator.inputs[1])
    resolved = _int_tensor(
        model_ir,
        graph_index,
        name=name,
        public_names=public_names,
    )
    if resolved is None:
        return None
    _, data, dtype = resolved
    if data.ndim != 1 or int(data.size) == 0:
        return None
    axes = []
    for raw_value in data.tolist():
        axis = int(raw_value)
        if axis < 0:
            axis += 4
        if axis < 0 or axis >= 4 or axis in axes:
            return None
        axes.append(axis)
    expected_output = list(input_shape)
    for axis in axes:
        expected_output[axis] = 1
    if tuple(expected_output) != tuple(output_shape):
        return None
    mapped = tuple(int(_NHWC_TO_NCHW[axis]) for axis in axes)
    return name, np.asarray(mapped, dtype=dtype)


def _pad_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    public_names: set[str],
) -> Optional[Tuple[str, np.ndarray]]:
    if len(input_shape) != 4 or len(operator.inputs) != 2:
        return None
    name = str(operator.inputs[1])
    resolved = _int_tensor(
        model_ir,
        graph_index,
        name=name,
        public_names=public_names,
    )
    if resolved is None:
        return None
    _, data, dtype = resolved
    if data.shape != (4, 2) or np.any(data < 0):
        return None
    expected_output = tuple(
        int(value) + int(pair[0]) + int(pair[1])
        for value, pair in zip(input_shape, data.tolist())
    )
    if expected_output != tuple(output_shape):
        return None
    mapped = np.asarray(
        [data[0], data[2], data[3], data[1]],
        dtype=dtype,
    )
    return name, mapped


def _reshape_options(
    operator: OperatorIR,
    *,
    old_shape: Tuple[int, ...],
    new_shape: Tuple[int, ...],
) -> Optional[dict[str, object]]:
    if not isinstance(operator.options, dict):
        return None
    options = dict(operator.options)
    for key in ("newShape", "onnxRawNewShape"):
        value = options.get(key)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            return None
        try:
            normalized = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
        if normalized != tuple(old_shape):
            return None
        options[key] = [int(item) for item in new_shape]
    return options


def _shape_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    old_shape: Tuple[int, ...],
    new_shape: Tuple[int, ...],
    public_names: set[str],
) -> Optional[Tuple[str, np.ndarray, dict[str, object]]]:
    if len(operator.inputs) != 2:
        return None
    name = str(operator.inputs[1])
    resolved = _int_tensor(
        model_ir,
        graph_index,
        name=name,
        public_names=public_names,
    )
    if resolved is None:
        return None
    _, data, dtype = resolved
    if (
        data.ndim != 1
        or tuple(int(value) for value in data.tolist()) != tuple(old_shape)
    ):
        return None
    options = _reshape_options(
        operator,
        old_shape=old_shape,
        new_shape=new_shape,
    )
    if options is None:
        return None
    return name, np.asarray(new_shape, dtype=dtype), options


def _validate_shape_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    output_shape: Tuple[int, ...],
    public_names: set[str],
) -> bool:
    if len(operator.inputs) != 2:
        return False
    resolved = _int_tensor(
        model_ir,
        graph_index,
        name=str(operator.inputs[1]),
        public_names=public_names,
    )
    if resolved is None:
        return False
    _, data, _ = resolved
    return bool(
        data.ndim == 1
        and tuple(int(value) for value in data.tolist())
        == tuple(output_shape)
        and _reshape_options(
            operator,
            old_shape=output_shape,
            new_shape=output_shape,
        )
        is not None
    )


def _pad_signature(
    signature: Tuple[int, ...],
    pads: np.ndarray,
) -> Tuple[int, ...]:
    return tuple(
        -1
        if int(value) < 0
        else int(value) + int(pair[0]) + int(pair[1])
        for value, pair in zip(signature, pads.tolist())
    )


def _rewrite_inputs(
    operator: OperatorIR,
    replacements: dict[str, str],
) -> Tuple[_InputRewrite, ...]:
    rewrites = []
    for input_index, input_name in enumerate(operator.inputs):
        old_name = str(input_name)
        new_name = replacements.get(old_name)
        if new_name is not None and str(new_name) != old_name:
            rewrites.append(
                _InputRewrite(
                    operator=operator,
                    input_index=int(input_index),
                    old_name=old_name,
                    new_name=str(new_name),
                )
            )
    return tuple(rewrites)


def _metadata_update(
    model_ir: ModelIR,
    name: str,
    *,
    shape: Tuple[int, ...],
    signature: Tuple[int, ...],
    layout_tensor: Optional[TensorIR] = None,
) -> Optional[_MetadataUpdate]:
    tensor = model_ir.tensors.get(str(name))
    if tensor is None:
        return None
    layout_source = tensor if layout_tensor is None else layout_tensor
    return _MetadataUpdate(
        name=str(name),
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        logical_layout=str(layout_source.logical_layout),
        physical_layout=str(layout_source.physical_layout),
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_MixAttentionPlan]:
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    tail = _resolve_tail(
        model_ir,
        graph_index,
        root,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if tail is None:
        return None
    attention = _resolve_attention(
        model_ir,
        graph_index,
        tail,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if attention is None:
        return None

    canonical = _data_contract(
        model_ir,
        tail.post_output_name,
        rank=4,
        layout="NHWC",
    )
    if canonical is None:
        return None
    canonical_tensor, nhwc_shape, nhwc_signature = canonical
    dtype = str(canonical_tensor.dtype)
    nchw_shape = _permute(nhwc_shape, _NHWC_TO_NCHW)
    nchw_signature = _permute(nhwc_signature, _NHWC_TO_NCHW)
    batch, height, width, channels = nhwc_shape
    sig_batch, sig_height, sig_width, sig_channels = nhwc_signature

    exact_rank4 = [
        (tail.branch.source_name, nhwc_shape, nhwc_signature, "NHWC"),
        (tail.branch.output_name, nchw_shape, nchw_signature, "NCHW"),
        (tail.source_name, nchw_shape, nchw_signature, "NCHW"),
        (tail.gate1_name, nchw_shape, nchw_signature, "NCHW"),
        (tail.gate2_name, nchw_shape, nchw_signature, "NCHW"),
        (tail.sub_name, nchw_shape, nchw_signature, "NCHW"),
        (tail.mul0_name, nchw_shape, nchw_signature, "NCHW"),
        (tail.add2_name, nchw_shape, nchw_signature, "NCHW"),
        (tail.mul1_name, nchw_shape, nchw_signature, "NCHW"),
        (tail.add3_name, nchw_shape, nchw_signature, "NCHW"),
    ]
    if tail.residual.add_mode:
        exact_rank4.append(
            (
                tail.residual.output_name,
                nchw_shape,
                nchw_signature,
                "NCHW",
            )
        )
        for adapter in tail.residual.adapters:
            exact_rank4.extend(
                (
                    (
                        adapter.source_name,
                        nhwc_shape,
                        nhwc_signature,
                        "NHWC",
                    ),
                    (
                        adapter.output_name,
                        nchw_shape,
                        nchw_signature,
                        "NCHW",
                    ),
                )
            )
    else:
        adapter = tail.residual.adapters[0]
        exact_rank4.extend(
            (
                (
                    adapter.source_name,
                    nhwc_shape,
                    nhwc_signature,
                    "NHWC",
                ),
                (
                    adapter.output_name,
                    nchw_shape,
                    nchw_signature,
                    "NCHW",
                ),
            )
        )
    if any(
        _exact_contract(
            model_ir,
            name,
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout=layout,
        )
        is None
        for name, shape, signature, layout in exact_rank4
    ):
        return None

    ca_nchw_shape = (batch, channels, 1, 1)
    ca_nchw_signature = (sig_batch, sig_channels, 1, 1)
    ca_nhwc_shape = (batch, 1, 1, channels)
    ca_nhwc_signature = (sig_batch, 1, 1, sig_channels)
    ca_pre_name = str(attention.ca_pre.outputs[0])
    ca_conv0_name = str(attention.ca_conv0.outputs[0])
    ca_conv0_contract = _data_contract(
        model_ir,
        ca_conv0_name,
        rank=4,
        dtype=dtype,
        layout="NHWC",
    )
    if ca_conv0_contract is None:
        return None
    _, ca_conv0_shape, ca_conv0_signature = ca_conv0_contract
    if (
        ca_conv0_shape[0] != batch
        or ca_conv0_shape[1:3] != (1, 1)
        or ca_conv0_signature[0] != sig_batch
        or ca_conv0_signature[1:3] != (1, 1)
    ):
        return None
    ca_contracts = (
        (
            attention.mean_ca_name,
            ca_nchw_shape,
            ca_nchw_signature,
            "NCHW",
        ),
        (ca_pre_name, ca_nhwc_shape, ca_nhwc_signature, "NHWC"),
        (
            str(attention.ca_conv2.outputs[0]),
            ca_nhwc_shape,
            ca_nhwc_signature,
            "NHWC",
        ),
        (
            attention.ca_output_name,
            ca_nchw_shape,
            ca_nchw_signature,
            "NCHW",
        ),
    )
    if any(
        _exact_contract(
            model_ir,
            name,
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout=layout,
        )
        is None
        for name, shape, signature, layout in ca_contracts
    ):
        return None

    sa_single_shape = (batch, 1, height, width)
    sa_single_signature = (sig_batch, 1, sig_height, sig_width)
    sa_concat_shape = (batch, 2, height, width)
    sa_concat_signature = (sig_batch, 2, sig_height, sig_width)
    mirror_sa_contract = _data_contract(
        model_ir,
        attention.mirror_sa_name,
        rank=4,
        dtype=dtype,
        layout="NCHW",
    )
    if mirror_sa_contract is None:
        return None
    _, mirror_sa_shape, mirror_sa_signature = mirror_sa_contract
    sa_pre_shape = _permute(mirror_sa_shape, _NCHW_TO_NHWC)
    sa_pre_signature = _permute(mirror_sa_signature, _NCHW_TO_NHWC)
    sa_conv_name = str(attention.sa_conv.outputs[0])
    sa_contracts = (
        (
            attention.mean_sa_name,
            sa_single_shape,
            sa_single_signature,
            "NCHW",
        ),
        (
            attention.max_sa_name,
            sa_single_shape,
            sa_single_signature,
            "NCHW",
        ),
        (
            attention.concat_sa_name,
            sa_concat_shape,
            sa_concat_signature,
            "NCHW",
        ),
        (
            str(attention.sa_pre.outputs[0]),
            sa_pre_shape,
            sa_pre_signature,
            "NHWC",
        ),
        (
            sa_conv_name,
            (batch, height, width, 1),
            (sig_batch, sig_height, sig_width, 1),
            "NHWC",
        ),
        (
            attention.sa_output_name,
            sa_single_shape,
            sa_single_signature,
            "NCHW",
        ),
        (
            attention.attention_name,
            nchw_shape,
            nchw_signature,
            "NCHW",
        ),
    )
    if any(
        _exact_contract(
            model_ir,
            name,
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout=layout,
        )
        is None
        for name, shape, signature, layout in sa_contracts
    ):
        return None

    unsqueeze_shape = (batch, channels, 1, height, width)
    unsqueeze_signature = (
        sig_batch,
        sig_channels,
        1,
        sig_height,
        sig_width,
    )
    concat_pa_shape = (batch, channels, 2, height, width)
    concat_pa_signature = (
        sig_batch,
        sig_channels,
        2,
        sig_height,
        sig_width,
    )
    doubled_channels = 2 * channels
    doubled_signature = -1 if sig_channels < 0 else 2 * sig_channels
    reshape_pa_shape = (batch, doubled_channels, height, width)
    reshape_pa_signature = (
        sig_batch,
        doubled_signature,
        sig_height,
        sig_width,
    )
    mirror_pa_contract = _data_contract(
        model_ir,
        attention.mirror_pa_name,
        rank=4,
        dtype=dtype,
        layout="NCHW",
    )
    if mirror_pa_contract is None:
        return None
    _, mirror_pa_shape, mirror_pa_signature = mirror_pa_contract
    pa_pre_shape = _permute(mirror_pa_shape, _NCHW_TO_NHWC)
    pa_pre_signature = _permute(mirror_pa_signature, _NCHW_TO_NHWC)
    pa_post_name = str(attention.pa_post.outputs[0])
    pa_contracts = (
        (
            attention.unsqueeze_source_name,
            unsqueeze_shape,
            unsqueeze_signature,
            None,
        ),
        (
            attention.unsqueeze_attention_name,
            unsqueeze_shape,
            unsqueeze_signature,
            None,
        ),
        (
            attention.concat_pa_name,
            concat_pa_shape,
            concat_pa_signature,
            None,
        ),
        (
            attention.reshape_pa_name,
            reshape_pa_shape,
            reshape_pa_signature,
            "NCHW",
        ),
        (
            str(attention.pa_pre.outputs[0]),
            pa_pre_shape,
            pa_pre_signature,
            "NHWC",
        ),
        (
            attention.pa_conv_name,
            nhwc_shape,
            nhwc_signature,
            "NHWC",
        ),
        (pa_post_name, nchw_shape, nchw_signature, "NCHW"),
    )
    if any(
        _exact_contract(
            model_ir,
            name,
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout=layout,
        )
        is None
        for name, shape, signature, layout in pa_contracts
    ):
        return None

    axis_roles = []
    for operator, output_name in (
        (attention.mean_ca, attention.mean_ca_name),
        (attention.mean_sa, attention.mean_sa_name),
        (attention.max_sa, attention.max_sa_name),
    ):
        output = model_ir.tensors[output_name]
        replacement = _axis_replacement(
            model_ir,
            graph_index,
            operator,
            input_shape=nchw_shape,
            output_shape=tuple(int(value) for value in output.shape),
            public_names=public_names,
        )
        if replacement is None:
            return None
        axis_roles.append((replacement[0], replacement[1], operator, 1))

    sa_pad = _pad_replacement(
        model_ir,
        graph_index,
        attention.mirror_sa,
        input_shape=sa_concat_shape,
        output_shape=mirror_sa_shape,
        public_names=public_names,
    )
    pa_pad = _pad_replacement(
        model_ir,
        graph_index,
        attention.mirror_pa,
        input_shape=reshape_pa_shape,
        output_shape=mirror_pa_shape,
        public_names=public_names,
    )
    if sa_pad is None or pa_pad is None:
        return None
    sa_pad_old = np.asarray(model_ir.tensors[sa_pad[0]].data)
    pa_pad_old = np.asarray(model_ir.tensors[pa_pad[0]].data)
    if (
        mirror_sa_signature != _pad_signature(
            sa_concat_signature,
            sa_pad_old,
        )
        or mirror_pa_signature != _pad_signature(
            reshape_pa_signature,
            pa_pad_old,
        )
    ):
        return None

    unsqueeze_nhwc_shape = _permute(unsqueeze_shape, _NCKHW_TO_NHWCK)
    unsqueeze_nhwc_signature = _permute(
        unsqueeze_signature,
        _NCKHW_TO_NHWCK,
    )
    concat_pa_nhwc_shape = _permute(concat_pa_shape, _NCKHW_TO_NHWCK)
    concat_pa_nhwc_signature = _permute(
        concat_pa_signature,
        _NCKHW_TO_NHWCK,
    )
    reshape_pa_nhwc_shape = _permute(reshape_pa_shape, _NCHW_TO_NHWC)
    shape_roles = []
    options_updates = []
    for operator, old_shape, new_shape in (
        (
            attention.unsqueeze_source,
            unsqueeze_shape,
            unsqueeze_nhwc_shape,
        ),
        (
            attention.unsqueeze_attention,
            unsqueeze_shape,
            unsqueeze_nhwc_shape,
        ),
        (
            attention.reshape_pa,
            reshape_pa_shape,
            reshape_pa_nhwc_shape,
        ),
    ):
        replacement = _shape_replacement(
            model_ir,
            graph_index,
            operator,
            old_shape=old_shape,
            new_shape=new_shape,
            public_names=public_names,
        )
        if replacement is None:
            return None
        shape_roles.append((replacement[0], replacement[1], operator, 1))
        options_updates.append(_OptionsUpdate(operator, replacement[2]))
    if not _validate_shape_constant(
        model_ir,
        graph_index,
        attention.sa_reshape,
        output_shape=sa_single_shape,
        public_names=public_names,
    ):
        return None

    constant_plans = _plan_constants(
        model_ir,
        graph_index,
        tuple(
            axis_roles
            + [
                (sa_pad[0], sa_pad[1], attention.mirror_sa, 1),
                (pa_pad[0], pa_pad[1], attention.mirror_pa, 1),
            ]
            + shape_roles
        ),
    )
    if constant_plans is None:
        return None
    concat_sa_options = dict(attention.concat_sa.options)
    concat_sa_options["axis"] = 3
    concat_pa_options = dict(attention.concat_pa.options)
    concat_pa_options["axis"] = 4
    options_updates.extend(
        (
            _OptionsUpdate(attention.concat_sa, concat_sa_options),
            _OptionsUpdate(attention.concat_pa, concat_pa_options),
        )
    )

    input_rewrites = []
    if tail.residual.add_mode:
        residual_replacements = {
            adapter.output_name: adapter.source_name
            for adapter in tail.residual.adapters
        }
        input_rewrites.extend(
            _rewrite_inputs(tail.residual.operator, residual_replacements)
        )
    input_rewrites.extend(
        _rewrite_inputs(
            tail.source,
            {
                tail.branch.output_name: tail.branch.source_name,
                tail.residual.output_name: tail.residual.canonical_name,
            },
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            tail.mul0,
            {tail.branch.output_name: tail.branch.source_name},
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            tail.mul1,
            {tail.residual.output_name: tail.residual.canonical_name},
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            attention.ca_conv0,
            {str(attention.ca_pre.outputs[0]): attention.mean_ca_name},
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            attention.sa_conv,
            {str(attention.sa_pre.outputs[0]): attention.mirror_sa_name},
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            attention.add_attention,
            {
                attention.ca_output_name: str(attention.ca_conv2.outputs[0]),
                attention.sa_output_name: sa_conv_name,
            },
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            attention.pa_conv,
            {str(attention.pa_pre.outputs[0]): attention.mirror_pa_name},
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            tail.gate1,
            {pa_post_name: attention.pa_conv_name},
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            tail.post_conv,
            {tail.post_output_name: tail.add3_name},
        )
    )

    rank4_metadata = (
        *((tail.residual.output_name,) if tail.residual.add_mode else ()),
        tail.source_name,
        attention.mean_ca_name,
        attention.mean_sa_name,
        attention.max_sa_name,
        attention.concat_sa_name,
        attention.mirror_sa_name,
        attention.attention_name,
        attention.reshape_pa_name,
        attention.mirror_pa_name,
        tail.gate1_name,
        tail.gate2_name,
        tail.mul0_name,
        tail.sub_name,
        tail.add2_name,
        tail.mul1_name,
        tail.add3_name,
    )
    metadata_updates = []
    for name in rank4_metadata:
        contract = _data_contract(
            model_ir,
            name,
            rank=4,
            dtype=dtype,
        )
        if contract is None:
            return None
        _, old_shape, old_signature = contract
        update = _metadata_update(
            model_ir,
            name,
            shape=_permute(old_shape, _NCHW_TO_NHWC),
            signature=_permute(old_signature, _NCHW_TO_NHWC),
            layout_tensor=canonical_tensor,
        )
        if update is None:
            return None
        metadata_updates.append(update)
    metadata_updates.extend(
        (
            _MetadataUpdate(
                name=attention.unsqueeze_source_name,
                shape=unsqueeze_nhwc_shape,
                signature=unsqueeze_nhwc_signature,
                logical_layout="UNKNOWN",
                physical_layout="UNKNOWN",
            ),
            _MetadataUpdate(
                name=attention.unsqueeze_attention_name,
                shape=unsqueeze_nhwc_shape,
                signature=unsqueeze_nhwc_signature,
                logical_layout="UNKNOWN",
                physical_layout="UNKNOWN",
            ),
            _MetadataUpdate(
                name=attention.concat_pa_name,
                shape=concat_pa_nhwc_shape,
                signature=concat_pa_nhwc_signature,
                logical_layout="UNKNOWN",
                physical_layout="UNKNOWN",
            ),
        )
    )

    remove_operators = (
        tail.branch.operator,
        *(adapter.operator for adapter in tail.residual.adapters),
        attention.ca_pre,
        attention.ca_post,
        attention.sa_pre,
        attention.sa_reshape,
        attention.pa_pre,
        attention.pa_post,
        tail.root,
    )
    if len({id(operator) for operator in remove_operators}) != len(
        remove_operators
    ):
        return None
    return _MixAttentionPlan(
        root=root,
        tail=tail,
        attention=attention,
        constant_plans=constant_plans,
        metadata_updates=tuple(metadata_updates),
        input_rewrites=tuple(input_rewrites),
        options_updates=tuple(options_updates),
        remove_operators=tuple(remove_operators),
    )


def _tail_equal(expected: _TailMatch, actual: _TailMatch) -> bool:
    operator_fields = (
        "root",
        "post_conv",
        "add3",
        "add2",
        "mul1",
        "source",
        "mul0",
        "sub",
        "gate1",
        "gate2",
    )
    value_fields = (
        "source_name",
        "gate1_name",
        "gate2_name",
        "sub_name",
        "mul0_name",
        "add2_name",
        "mul1_name",
        "add3_name",
        "post_output_name",
        "sub_constant_name",
    )
    return bool(
        all(
            getattr(expected, name) is getattr(actual, name)
            for name in operator_fields
        )
        and all(
            getattr(expected, name) == getattr(actual, name)
            for name in value_fields
        )
        and expected.branch.operator is actual.branch.operator
        and expected.branch.source_name == actual.branch.source_name
        and expected.branch.output_name == actual.branch.output_name
        and expected.residual.operator is actual.residual.operator
        and expected.residual.canonical_name
        == actual.residual.canonical_name
        and expected.residual.output_name == actual.residual.output_name
        and expected.residual.add_mode == actual.residual.add_mode
        and len(expected.residual.adapters)
        == len(actual.residual.adapters)
        and all(
            lhs.operator is rhs.operator
            and lhs.source_name == rhs.source_name
            and lhs.output_name == rhs.output_name
            for lhs, rhs in zip(
                expected.residual.adapters,
                actual.residual.adapters,
            )
        )
    )


def _attention_equal(
    expected: _AttentionMatch,
    actual: _AttentionMatch,
) -> bool:
    operator_fields = (
        "mean_ca",
        "ca_pre",
        "ca_conv0",
        "ca_conv2",
        "ca_post",
        "mean_sa",
        "max_sa",
        "concat_sa",
        "mirror_sa",
        "sa_pre",
        "sa_conv",
        "sa_reshape",
        "add_attention",
        "unsqueeze_source",
        "unsqueeze_attention",
        "concat_pa",
        "reshape_pa",
        "mirror_pa",
        "pa_pre",
        "pa_conv",
        "pa_post",
    )
    value_fields = (
        "ca_output_name",
        "sa_output_name",
        "attention_name",
        "mean_ca_name",
        "mean_sa_name",
        "max_sa_name",
        "concat_sa_name",
        "mirror_sa_name",
        "unsqueeze_source_name",
        "unsqueeze_attention_name",
        "concat_pa_name",
        "reshape_pa_name",
        "mirror_pa_name",
        "pa_conv_name",
    )
    return bool(
        all(
            getattr(expected, name) is getattr(actual, name)
            for name in operator_fields
        )
        and all(
            getattr(expected, name) == getattr(actual, name)
            for name in value_fields
        )
    )


def _plans_equal(
    expected: _MixAttentionPlan,
    actual: _MixAttentionPlan,
) -> bool:
    return bool(
        expected.root is actual.root
        and _tail_equal(expected.tail, actual.tail)
        and _attention_equal(expected.attention, actual.attention)
        and expected.metadata_updates == actual.metadata_updates
        and len(expected.input_rewrites) == len(actual.input_rewrites)
        and all(
            lhs.operator is rhs.operator
            and lhs.input_index == rhs.input_index
            and lhs.old_name == rhs.old_name
            and lhs.new_name == rhs.new_name
            for lhs, rhs in zip(
                expected.input_rewrites,
                actual.input_rewrites,
            )
        )
        and len(expected.options_updates) == len(actual.options_updates)
        and all(
            lhs.operator is rhs.operator and lhs.options == rhs.options
            for lhs, rhs in zip(
                expected.options_updates,
                actual.options_updates,
            )
        )
        and len(expected.remove_operators) == len(actual.remove_operators)
        and all(
            lhs is rhs
            for lhs, rhs in zip(
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
    plan: _MixAttentionPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.root)
    if current is None or not _plans_equal(plan, current):
        return False
    remove_indices = [
        _operator_index(graph_index, operator)
        for operator in plan.remove_operators
    ]
    mutation_operators = (
        *(rewrite.operator for rewrite in plan.input_rewrites),
        *(update.operator for update in plan.options_updates),
    )
    mutation_indices = [
        _operator_index(graph_index, operator)
        for operator in mutation_operators
    ]
    if (
        any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or any(index is None for index in mutation_indices)
        or any(
            rewrite.input_index < 0
            or rewrite.input_index >= len(rewrite.operator.inputs)
            or str(rewrite.operator.inputs[rewrite.input_index])
            != rewrite.old_name
            for rewrite in plan.input_rewrites
        )
        or any(
            constant.clone_name is not None
            and constant.clone_name in model_ir.tensors
            for constant in plan.constant_plans
        )
        or any(
            update.name not in model_ir.tensors
            for update in plan.metadata_updates
        )
    ):
        return False

    _apply_constant_plans(model_ir, graph_index, plan.constant_plans)
    for rewrite in plan.input_rewrites:
        _replace_operator_input_at(
            model_ir=model_ir,
            op=rewrite.operator,
            input_index=rewrite.input_index,
            new_input_name=rewrite.new_name,
            graph_index=graph_index,
        )
    for update in plan.options_updates:
        update.operator.options = dict(update.options)
    _apply_metadata_updates(model_ir, plan.metadata_updates)
    graph_index.remove_operators([int(index) for index in remove_indices])
    return True


def optimize_sinet_mix_attention_double_logistic_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a strict SiNet double-Logistic mix-attention island to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 8,
        "MEAN": 2,
        "REDUCE_MAX": 1,
        "MIRROR_PAD": 2,
        "RESHAPE": 4,
        "CONCATENATION": 2,
        "CONV_2D": 5,
        "LOGISTIC": 2,
        "SUB": 1,
        "MUL": 2,
        "ADD": 4,
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
