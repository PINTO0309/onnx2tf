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
    _NCHW_TO_NHWC,
    _NHWC_TO_NCHW,
    _permute,
    _plain_binary,
    _typed_permutation,
)
from onnx2tf.tflite_builder.passes.sinet_concat_resize_layout import (
    _InputRewrite,
)
from onnx2tf.tflite_builder.passes.sinet_mix_attention_layout import (
    _NCKHW_TO_NHWCK,
    _OptionsUpdate,
    _axis_replacement,
    _data_contract,
    _exact_contract,
    _exact_consumers,
    _metadata_update,
    _operator_index,
    _pad_replacement,
    _pad_signature,
    _plain_concat,
    _plain_conv,
    _plain_mirror_pad,
    _plain_reduce,
    _plain_reshape,
    _plain_unary,
    _rewrite_inputs,
    _shape_replacement,
    _sole_consumer,
    _validate_shape_constant,
)
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    _ConstantPlan,
    _MetadataUpdate,
    _apply_constant_plans,
    _apply_metadata_updates,
    _constant_plans_equal,
    _plan_constants,
    _producer,
)


_STATS_KEY = "optimized_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"


@dataclass(frozen=True)
class _IslandMatch:
    root: OperatorIR
    mean: OperatorIR
    maximum: OperatorIR
    unsqueeze_source: OperatorIR
    concat_sa: OperatorIR
    mirror_sa: OperatorIR
    sa_pre: OperatorIR
    sa_conv: OperatorIR
    sa_reshape: OperatorIR
    ca_pre: OperatorIR
    add_attention: OperatorIR
    unsqueeze_attention: OperatorIR
    concat_pa: OperatorIR
    reshape_pa: OperatorIR
    mirror_pa: OperatorIR
    pa_pre: OperatorIR
    pa_conv: OperatorIR
    gate_pre: Optional[OperatorIR]
    gate: OperatorIR
    mul: OperatorIR
    source_nhwc_name: str
    source_nchw_name: str
    mean_name: str
    maximum_name: str
    concat_sa_name: str
    mirror_sa_name: str
    sa_pre_name: str
    sa_conv_name: str
    sa_nchw_name: str
    ca_nhwc_name: str
    ca_nchw_name: str
    attention_name: str
    unsqueeze_source_name: str
    unsqueeze_attention_name: str
    concat_pa_name: str
    reshape_pa_name: str
    mirror_pa_name: str
    pa_pre_name: str
    pa_conv_name: str
    gate_input_name: str
    gate_name: str
    mul_output_name: str
    mul_uses_source_nchw: bool


@dataclass(frozen=True)
class _SaPaPlan:
    root: OperatorIR
    island: _IslandMatch
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    input_rewrites: Tuple[_InputRewrite, ...]
    options_updates: Tuple[_OptionsUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]
    legacy_nhwc_output_name: Optional[str]
    legacy_perm_name: Optional[str]


def _resolved_input(
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    consumer_index: int,
    public_inputs: set[str],
) -> bool:
    source_name = str(name)
    if source_name in graph_index.duplicate_producers:
        return False
    producer_index = graph_index.producers.get(source_name)
    return bool(
        (producer_index is None and source_name in public_inputs)
        or (
            producer_index is not None
            and int(producer_index) < int(consumer_index)
        )
    )


def _resolve_island(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
    *,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_IslandMatch]:
    public_names = public_inputs | public_outputs
    root_index = _operator_index(graph_index, root)
    if (
        root_index is None
        or str(root.op_type) != "TRANSPOSE"
        or len(root.outputs) != 1
        or not _typed_permutation(
            model_ir,
            graph_index,
            root,
            _NHWC_TO_NCHW,
            public_names,
        )
    ):
        return None
    source_nhwc_name = str(root.inputs[0])
    source_nchw_name = str(root.outputs[0])
    if (
        source_nchw_name in public_names
        or source_nchw_name in graph_index.duplicate_producers
        or graph_index.producers.get(source_nchw_name) != root_index
        or not _resolved_input(
            graph_index,
            name=source_nhwc_name,
            consumer_index=root_index,
            public_inputs=public_inputs,
        )
    ):
        return None

    source_users = graph_index.consumer_indices(source_nchw_name)
    if (
        len(source_users) not in {3, 4}
        or len(set(source_users)) != len(source_users)
        or any(index <= root_index for index in source_users)
    ):
        return None
    user_operators = [model_ir.operators[index] for index in source_users]
    means = [operator for operator in user_operators if str(operator.op_type) == "MEAN"]
    maxima = [
        operator
        for operator in user_operators
        if str(operator.op_type) == "REDUCE_MAX"
    ]
    reshapes = [
        operator
        for operator in user_operators
        if str(operator.op_type) == "RESHAPE"
    ]
    if len(means) != 1 or len(maxima) != 1 or len(reshapes) != 1:
        return None
    mean = means[0]
    maximum = maxima[0]
    unsqueeze_source = reshapes[0]
    if (
        not _plain_reduce(mean, "MEAN", keep_dims=True)
        or not _plain_reduce(maximum, "REDUCE_MAX", keep_dims=True)
        or not _plain_reshape(unsqueeze_source)
        or any(
            str(operator.inputs[0]) != source_nchw_name
            for operator in (mean, maximum, unsqueeze_source)
        )
    ):
        return None

    mean_index = _operator_index(graph_index, mean)
    maximum_index = _operator_index(graph_index, maximum)
    unsqueeze_source_index = _operator_index(graph_index, unsqueeze_source)
    if mean_index is None or maximum_index is None or unsqueeze_source_index is None:
        return None
    mean_name = str(mean.outputs[0])
    maximum_name = str(maximum.outputs[0])
    mean_users = graph_index.consumer_indices(mean_name)
    maximum_users = graph_index.consumer_indices(maximum_name)
    if (
        mean_name in public_names
        or maximum_name in public_names
        or len(mean_users) != 1
        or mean_users != maximum_users
    ):
        return None
    concat_sa_index = int(mean_users[0])
    if concat_sa_index <= max(mean_index, maximum_index):
        return None
    concat_sa = model_ir.operators[concat_sa_index]
    if (
        not _plain_concat(concat_sa, axis=1, rank=4)
        or Counter(str(name) for name in concat_sa.inputs)
        != Counter((mean_name, maximum_name))
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
    if not _plain_mirror_pad(mirror_sa):
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
    sa_pre_name = str(sa_pre.outputs[0])
    sa_conv_match = _sole_consumer(
        model_ir,
        graph_index,
        name=sa_pre_name,
        producer_index=sa_pre_index,
        op_type="CONV_2D",
        public_names=public_names,
    )
    if sa_conv_match is None:
        return None
    sa_conv_index, sa_conv = sa_conv_match
    if not _plain_conv(sa_conv) or str(sa_conv.inputs[0]) != sa_pre_name:
        return None
    sa_conv_name = str(sa_conv.outputs[0])
    sa_reshape_match = _sole_consumer(
        model_ir,
        graph_index,
        name=sa_conv_name,
        producer_index=sa_conv_index,
        op_type="RESHAPE",
        public_names=public_names,
    )
    if sa_reshape_match is None:
        return None
    sa_reshape_index, sa_reshape = sa_reshape_match
    if not _plain_reshape(sa_reshape):
        return None
    sa_nchw_name = str(sa_reshape.outputs[0])
    attention_match = _sole_consumer(
        model_ir,
        graph_index,
        name=sa_nchw_name,
        producer_index=sa_reshape_index,
        op_type="ADD",
        public_names=public_names,
    )
    if attention_match is None:
        return None
    attention_index, add_attention = attention_match
    if not _plain_binary(add_attention, "ADD"):
        return None
    other_inputs = [
        str(name) for name in add_attention.inputs if str(name) != sa_nchw_name
    ]
    if len(other_inputs) != 1:
        return None
    ca_nchw_name = other_inputs[0]
    ca_match = _producer(model_ir, graph_index, ca_nchw_name, "TRANSPOSE")
    if ca_match is None:
        return None
    ca_pre_index, ca_pre = ca_match
    if (
        ca_pre_index >= attention_index
        or ca_nchw_name in public_names
        or not _typed_permutation(
            model_ir,
            graph_index,
            ca_pre,
            _NHWC_TO_NCHW,
            public_names,
        )
        or not _exact_consumers(graph_index, ca_nchw_name, (add_attention,))
    ):
        return None
    ca_nhwc_name = str(ca_pre.inputs[0])
    if not _resolved_input(
        graph_index,
        name=ca_nhwc_name,
        consumer_index=ca_pre_index,
        public_inputs=public_inputs,
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
    unsqueeze_source_name = str(unsqueeze_source.outputs[0])
    unsqueeze_attention_name = str(unsqueeze_attention.outputs[0])
    source_unsqueeze_users = graph_index.consumer_indices(unsqueeze_source_name)
    attention_unsqueeze_users = graph_index.consumer_indices(
        unsqueeze_attention_name
    )
    if (
        unsqueeze_source_name in public_names
        or unsqueeze_attention_name in public_names
        or len(source_unsqueeze_users) != 1
        or source_unsqueeze_users != attention_unsqueeze_users
    ):
        return None
    concat_pa_index = int(source_unsqueeze_users[0])
    if concat_pa_index <= max(unsqueeze_source_index, unsqueeze_attention_index):
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
    pa_pre_name = str(pa_pre.outputs[0])
    pa_conv_match = _sole_consumer(
        model_ir,
        graph_index,
        name=pa_pre_name,
        producer_index=pa_pre_index,
        op_type="CONV_2D",
        public_names=public_names,
    )
    if pa_conv_match is None:
        return None
    pa_conv_index, pa_conv = pa_conv_match
    if not _plain_conv(pa_conv) or str(pa_conv.inputs[0]) != pa_pre_name:
        return None
    pa_conv_name = str(pa_conv.outputs[0])
    pa_users = graph_index.consumer_indices(pa_conv_name)
    if len(pa_users) != 1:
        return None
    next_index = int(pa_users[0])
    if next_index <= pa_conv_index:
        return None
    next_operator = model_ir.operators[next_index]
    gate_pre: Optional[OperatorIR] = None
    gate_input_name = pa_conv_name
    if str(next_operator.op_type) == "TRANSPOSE":
        gate_pre = next_operator
        if not _typed_permutation(
            model_ir,
            graph_index,
            gate_pre,
            _NHWC_TO_NCHW,
            public_names,
        ):
            return None
        gate_input_name = str(gate_pre.outputs[0])
        gate_pre_users = graph_index.consumer_indices(gate_input_name)
        if (
            gate_input_name in public_names
            or len(gate_pre_users) != 1
            or gate_pre_users[0] <= next_index
        ):
            return None
        next_index = int(gate_pre_users[0])
        next_operator = model_ir.operators[next_index]
    gate = next_operator
    if (
        not _plain_unary(gate, "LOGISTIC")
        or str(gate.inputs[0]) != gate_input_name
    ):
        return None
    gate_name = str(gate.outputs[0])
    gate_match = _sole_consumer(
        model_ir,
        graph_index,
        name=gate_name,
        producer_index=next_index,
        op_type="MUL",
        public_names=public_names,
    )
    if gate_match is None:
        return None
    mul_index, mul = gate_match
    if not _plain_binary(mul, "MUL"):
        return None
    mul_output_name = str(mul.outputs[0])
    if (
        mul_output_name in graph_index.duplicate_producers
        or graph_index.producers.get(mul_output_name) != mul_index
    ):
        return None
    mul_inputs = Counter(str(name) for name in mul.inputs)
    uses_source_nhwc = mul_inputs == Counter((gate_name, source_nhwc_name))
    uses_source_nchw = mul_inputs == Counter((gate_name, source_nchw_name))
    if uses_source_nhwc == uses_source_nchw:
        return None
    if bool(gate_pre is not None) != bool(uses_source_nchw):
        return None
    expected_source_users = (mean, maximum, unsqueeze_source)
    if uses_source_nchw:
        expected_source_users = (*expected_source_users, mul)
    if not _exact_consumers(
        graph_index,
        source_nchw_name,
        expected_source_users,
    ):
        return None

    return _IslandMatch(
        root=root,
        mean=mean,
        maximum=maximum,
        unsqueeze_source=unsqueeze_source,
        concat_sa=concat_sa,
        mirror_sa=mirror_sa,
        sa_pre=sa_pre,
        sa_conv=sa_conv,
        sa_reshape=sa_reshape,
        ca_pre=ca_pre,
        add_attention=add_attention,
        unsqueeze_attention=unsqueeze_attention,
        concat_pa=concat_pa,
        reshape_pa=reshape_pa,
        mirror_pa=mirror_pa,
        pa_pre=pa_pre,
        pa_conv=pa_conv,
        gate_pre=gate_pre,
        gate=gate,
        mul=mul,
        source_nhwc_name=source_nhwc_name,
        source_nchw_name=source_nchw_name,
        mean_name=mean_name,
        maximum_name=maximum_name,
        concat_sa_name=concat_sa_name,
        mirror_sa_name=mirror_sa_name,
        sa_pre_name=sa_pre_name,
        sa_conv_name=sa_conv_name,
        sa_nchw_name=sa_nchw_name,
        ca_nhwc_name=ca_nhwc_name,
        ca_nchw_name=ca_nchw_name,
        attention_name=attention_name,
        unsqueeze_source_name=unsqueeze_source_name,
        unsqueeze_attention_name=unsqueeze_attention_name,
        concat_pa_name=concat_pa_name,
        reshape_pa_name=reshape_pa_name,
        mirror_pa_name=mirror_pa_name,
        pa_pre_name=pa_pre_name,
        pa_conv_name=pa_conv_name,
        gate_input_name=gate_input_name,
        gate_name=gate_name,
        mul_output_name=mul_output_name,
        mul_uses_source_nchw=bool(uses_source_nchw),
    )


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    candidate = str(base)
    suffix = 0
    while candidate in model_ir.tensors:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_SaPaPlan]:
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    island = _resolve_island(
        model_ir,
        graph_index,
        root,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if island is None:
        return None

    source = _data_contract(
        model_ir,
        island.source_nhwc_name,
        rank=4,
        layout="NHWC",
    )
    if source is None:
        return None
    source_tensor, nhwc_shape, nhwc_signature = source
    dtype = str(source_tensor.dtype)
    if nhwc_shape[3] != 1 or nhwc_signature[3] not in {1, -1}:
        return None
    nchw_shape = _permute(nhwc_shape, _NHWC_TO_NCHW)
    nchw_signature = _permute(nhwc_signature, _NHWC_TO_NCHW)
    batch, height, width, channels = nhwc_shape
    sig_batch, sig_height, sig_width, sig_channels = nhwc_signature
    if _exact_contract(
        model_ir,
        island.source_nchw_name,
        dtype=dtype,
        shape=nchw_shape,
        signature=nchw_signature,
        layout="NCHW",
    ) is None:
        return None

    single_nchw_shape = (batch, 1, height, width)
    single_nchw_signature = (sig_batch, 1, sig_height, sig_width)
    concat_sa_shape = (batch, 2, height, width)
    concat_sa_signature = (sig_batch, 2, sig_height, sig_width)
    mirror_sa_contract = _data_contract(
        model_ir,
        island.mirror_sa_name,
        rank=4,
        dtype=dtype,
        layout="NCHW",
    )
    if mirror_sa_contract is None:
        return None
    _, mirror_sa_shape, mirror_sa_signature = mirror_sa_contract
    sa_pre_shape = _permute(mirror_sa_shape, _NCHW_TO_NHWC)
    sa_pre_signature = _permute(mirror_sa_signature, _NCHW_TO_NHWC)
    sa_contracts = (
        (island.mean_name, single_nchw_shape, single_nchw_signature, "NCHW"),
        (
            island.maximum_name,
            single_nchw_shape,
            single_nchw_signature,
            "NCHW",
        ),
        (island.concat_sa_name, concat_sa_shape, concat_sa_signature, "NCHW"),
        (island.sa_pre_name, sa_pre_shape, sa_pre_signature, "NHWC"),
        (island.sa_conv_name, nhwc_shape, nhwc_signature, "NHWC"),
        (island.sa_nchw_name, nchw_shape, nchw_signature, "NCHW"),
        (island.ca_nhwc_name, nhwc_shape, nhwc_signature, "NHWC"),
        (island.ca_nchw_name, nchw_shape, nchw_signature, "NCHW"),
        (island.attention_name, nchw_shape, nchw_signature, "NCHW"),
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
        island.mirror_pa_name,
        rank=4,
        dtype=dtype,
        layout="NCHW",
    )
    if mirror_pa_contract is None:
        return None
    _, mirror_pa_shape, mirror_pa_signature = mirror_pa_contract
    pa_pre_shape = _permute(mirror_pa_shape, _NCHW_TO_NHWC)
    pa_pre_signature = _permute(mirror_pa_signature, _NCHW_TO_NHWC)
    rank5_contracts = (
        (
            island.unsqueeze_source_name,
            unsqueeze_shape,
            unsqueeze_signature,
            None,
        ),
        (
            island.unsqueeze_attention_name,
            unsqueeze_shape,
            unsqueeze_signature,
            None,
        ),
        (island.concat_pa_name, concat_pa_shape, concat_pa_signature, None),
        (
            island.reshape_pa_name,
            reshape_pa_shape,
            reshape_pa_signature,
            "NCHW",
        ),
        (island.pa_pre_name, pa_pre_shape, pa_pre_signature, "NHWC"),
        (island.pa_conv_name, nhwc_shape, nhwc_signature, "NHWC"),
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
        for name, shape, signature, layout in rank5_contracts
    ):
        return None

    original_gate_shape = nchw_shape if island.gate_pre is not None else nhwc_shape
    original_gate_signature = (
        nchw_signature if island.gate_pre is not None else nhwc_signature
    )
    original_gate_layout = "NCHW" if island.gate_pre is not None else "NHWC"
    gate_contracts = (
        (
            island.gate_input_name,
            original_gate_shape,
            original_gate_signature,
            original_gate_layout,
        ),
        (
            island.gate_name,
            original_gate_shape,
            original_gate_signature,
            original_gate_layout,
        ),
        (
            island.mul_output_name,
            original_gate_shape,
            original_gate_signature,
            original_gate_layout,
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
        for name, shape, signature, layout in gate_contracts
    ):
        return None

    axis_roles = []
    for operator, output_name in (
        (island.mean, island.mean_name),
        (island.maximum, island.maximum_name),
    ):
        replacement = _axis_replacement(
            model_ir,
            graph_index,
            operator,
            input_shape=nchw_shape,
            output_shape=tuple(model_ir.tensors[output_name].shape),
            public_names=public_names,
        )
        if replacement is None:
            return None
        axis_roles.append((replacement[0], replacement[1], operator, 1))

    sa_pad = _pad_replacement(
        model_ir,
        graph_index,
        island.mirror_sa,
        input_shape=concat_sa_shape,
        output_shape=mirror_sa_shape,
        public_names=public_names,
    )
    pa_pad = _pad_replacement(
        model_ir,
        graph_index,
        island.mirror_pa,
        input_shape=reshape_pa_shape,
        output_shape=mirror_pa_shape,
        public_names=public_names,
    )
    if sa_pad is None or pa_pad is None:
        return None
    if (
        mirror_sa_signature
        != _pad_signature(
            concat_sa_signature,
            np.asarray(model_ir.tensors[sa_pad[0]].data),
        )
        or mirror_pa_signature
        != _pad_signature(
            reshape_pa_signature,
            np.asarray(model_ir.tensors[pa_pad[0]].data),
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
        (island.unsqueeze_source, unsqueeze_shape, unsqueeze_nhwc_shape),
        (
            island.unsqueeze_attention,
            unsqueeze_shape,
            unsqueeze_nhwc_shape,
        ),
        (island.reshape_pa, reshape_pa_shape, reshape_pa_nhwc_shape),
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
        island.sa_reshape,
        output_shape=nchw_shape,
        public_names=public_names,
    ):
        return None

    constant_plans = _plan_constants(
        model_ir,
        graph_index,
        tuple(
            axis_roles
            + [
                (sa_pad[0], sa_pad[1], island.mirror_sa, 1),
                (pa_pad[0], pa_pad[1], island.mirror_pa, 1),
            ]
            + shape_roles
        ),
    )
    if constant_plans is None:
        return None
    concat_sa_options = dict(island.concat_sa.options)
    concat_sa_options["axis"] = 3
    concat_pa_options = dict(island.concat_pa.options)
    concat_pa_options["axis"] = 4
    options_updates.extend(
        (
            _OptionsUpdate(island.concat_sa, concat_sa_options),
            _OptionsUpdate(island.concat_pa, concat_pa_options),
        )
    )

    input_rewrites = []
    for operator in (island.mean, island.maximum, island.unsqueeze_source):
        input_rewrites.extend(
            _rewrite_inputs(
                operator,
                {island.source_nchw_name: island.source_nhwc_name},
            )
        )
    input_rewrites.extend(
        _rewrite_inputs(
            island.sa_conv,
            {island.sa_pre_name: island.mirror_sa_name},
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            island.add_attention,
            {
                island.sa_nchw_name: island.sa_conv_name,
                island.ca_nchw_name: island.ca_nhwc_name,
            },
        )
    )
    input_rewrites.extend(
        _rewrite_inputs(
            island.pa_conv,
            {island.pa_pre_name: island.mirror_pa_name},
        )
    )
    if island.gate_pre is not None:
        input_rewrites.extend(
            _rewrite_inputs(
                island.gate,
                {island.gate_input_name: island.pa_conv_name},
            )
        )
    if island.mul_uses_source_nchw:
        input_rewrites.extend(
            _rewrite_inputs(
                island.mul,
                {island.source_nchw_name: island.source_nhwc_name},
            )
        )

    metadata_updates = []
    rank4_nchw_names = (
        island.mean_name,
        island.maximum_name,
        island.concat_sa_name,
        island.mirror_sa_name,
        island.attention_name,
        island.reshape_pa_name,
        island.mirror_pa_name,
        *((island.gate_name,) if island.gate_pre is not None else ()),
    )
    for name in rank4_nchw_names:
        contract = _data_contract(model_ir, name, rank=4, dtype=dtype)
        if contract is None:
            return None
        _, old_shape, old_signature = contract
        update = _metadata_update(
            model_ir,
            name,
            shape=_permute(old_shape, _NCHW_TO_NHWC),
            signature=_permute(old_signature, _NCHW_TO_NHWC),
            layout_tensor=source_tensor,
        )
        if update is None:
            return None
        metadata_updates.append(update)
    metadata_updates.extend(
        (
            _MetadataUpdate(
                name=island.unsqueeze_source_name,
                shape=unsqueeze_nhwc_shape,
                signature=unsqueeze_nhwc_signature,
                logical_layout="UNKNOWN",
                physical_layout="UNKNOWN",
            ),
            _MetadataUpdate(
                name=island.unsqueeze_attention_name,
                shape=unsqueeze_nhwc_shape,
                signature=unsqueeze_nhwc_signature,
                logical_layout="UNKNOWN",
                physical_layout="UNKNOWN",
            ),
            _MetadataUpdate(
                name=island.concat_pa_name,
                shape=concat_pa_nhwc_shape,
                signature=concat_pa_nhwc_signature,
                logical_layout="UNKNOWN",
                physical_layout="UNKNOWN",
            ),
        )
    )

    legacy_nhwc_output_name = None
    legacy_perm_name = None
    if island.mul_uses_source_nchw:
        mul_index = _operator_index(graph_index, island.mul)
        if mul_index is None or any(
            consumer_index <= mul_index
            for consumer_index in graph_index.consumer_indices(
                island.mul_output_name
            )
        ):
            return None
        legacy_nhwc_output_name = _unique_tensor_name(
            model_ir,
            f"{island.mul_output_name}_nhwc",
        )
        legacy_perm_name = _unique_tensor_name(
            model_ir,
            f"{island.mul_output_name}_perm_nhwc_to_nchw",
        )

    remove_operators = (
        island.root,
        island.ca_pre,
        island.sa_pre,
        island.sa_reshape,
        island.pa_pre,
        *((island.gate_pre,) if island.gate_pre is not None else ()),
    )
    if len({id(operator) for operator in remove_operators}) != len(
        remove_operators
    ):
        return None
    return _SaPaPlan(
        root=root,
        island=island,
        constant_plans=constant_plans,
        metadata_updates=tuple(metadata_updates),
        input_rewrites=tuple(input_rewrites),
        options_updates=tuple(options_updates),
        remove_operators=tuple(remove_operators),
        legacy_nhwc_output_name=legacy_nhwc_output_name,
        legacy_perm_name=legacy_perm_name,
    )


def _islands_equal(expected: _IslandMatch, actual: _IslandMatch) -> bool:
    operator_fields = (
        "root",
        "mean",
        "maximum",
        "unsqueeze_source",
        "concat_sa",
        "mirror_sa",
        "sa_pre",
        "sa_conv",
        "sa_reshape",
        "ca_pre",
        "add_attention",
        "unsqueeze_attention",
        "concat_pa",
        "reshape_pa",
        "mirror_pa",
        "pa_pre",
        "pa_conv",
        "gate_pre",
        "gate",
        "mul",
    )
    value_fields = tuple(
        field
        for field in _IslandMatch.__dataclass_fields__
        if field not in operator_fields
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


def _plans_equal(expected: _SaPaPlan, actual: _SaPaPlan) -> bool:
    return bool(
        expected.root is actual.root
        and _islands_equal(expected.island, actual.island)
        and expected.metadata_updates == actual.metadata_updates
        and expected.legacy_nhwc_output_name == actual.legacy_nhwc_output_name
        and expected.legacy_perm_name == actual.legacy_perm_name
        and len(expected.input_rewrites) == len(actual.input_rewrites)
        and all(
            lhs.operator is rhs.operator
            and lhs.input_index == rhs.input_index
            and lhs.old_name == rhs.old_name
            and lhs.new_name == rhs.new_name
            for lhs, rhs in zip(expected.input_rewrites, actual.input_rewrites)
        )
        and len(expected.options_updates) == len(actual.options_updates)
        and all(
            lhs.operator is rhs.operator and lhs.options == rhs.options
            for lhs, rhs in zip(expected.options_updates, actual.options_updates)
        )
        and len(expected.remove_operators) == len(actual.remove_operators)
        and all(
            lhs is rhs
            for lhs, rhs in zip(expected.remove_operators, actual.remove_operators)
        )
        and _constant_plans_equal(
            expected.constant_plans,
            actual.constant_plans,
        )
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _SaPaPlan,
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
    legacy_requested = plan.legacy_nhwc_output_name is not None
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
        or (
            legacy_requested
            and (
                plan.legacy_perm_name is None
                or plan.legacy_nhwc_output_name in model_ir.tensors
                or plan.legacy_perm_name in model_ir.tensors
                or _operator_index(graph_index, plan.island.mul) is None
            )
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

    if legacy_requested:
        legacy_tensor = model_ir.tensors[plan.island.mul_output_name]
        model_ir.tensors[str(plan.legacy_nhwc_output_name)] = TensorIR(
            name=str(plan.legacy_nhwc_output_name),
            dtype=str(legacy_tensor.dtype),
            shape=[int(value) for value in _permute(legacy_tensor.shape, _NCHW_TO_NHWC)],
            shape_signature=[
                int(value)
                for value in _permute(
                    legacy_tensor.shape_signature
                    if legacy_tensor.shape_signature is not None
                    else legacy_tensor.shape,
                    _NCHW_TO_NHWC,
                )
            ],
            data=None,
            is_variable=False,
            quantization=None,
            logical_layout="NHWC",
            physical_layout="NHWC",
        )
        model_ir.tensors[str(plan.legacy_perm_name)] = TensorIR(
            name=str(plan.legacy_perm_name),
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(_NHWC_TO_NCHW, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )
        mul_index = _operator_index(graph_index, plan.island.mul)
        assert mul_index is not None
        graph_index.replace_operator_outputs(
            mul_index,
            [str(plan.legacy_nhwc_output_name)],
        )

    graph_index.remove_operators([int(index) for index in remove_indices])
    if legacy_requested:
        mul_index = _operator_index(graph_index, plan.island.mul)
        if mul_index is None:
            raise AssertionError("MUL disappeared after transactional preflight")
        graph_index.insert_operator(
            int(mul_index) + 1,
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[
                    str(plan.legacy_nhwc_output_name),
                    str(plan.legacy_perm_name),
                ],
                outputs=[plan.island.mul_output_name],
                options={},
            ),
        )
    return True


def optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a strict SiNet SA/PA MirrorPad attention island to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 4,
        "MEAN": 1,
        "REDUCE_MAX": 1,
        "RESHAPE": 4,
        "CONCATENATION": 2,
        "MIRROR_PAD": 2,
        "CONV_2D": 2,
        "ADD": 1,
        "LOGISTIC": 1,
        "MUL": 1,
    }
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if all(value == 0 for value in required_counts.values()):
            break
    if rewrite_limit == 0 or any(value > 0 for value in required_counts.values()):
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
