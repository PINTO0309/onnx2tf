from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from typing import Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _replace_operator_input_at,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    _FLOAT_DTYPES,
    _NCHW_TO_NHWC,
    _NHWC_TO_NCHW,
    _data_and_constant_inputs,
    _has_exact_producer,
    _permute,
    _plain_binary,
    _tensor_contract,
    _typed_permutation,
)
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    _ConstantPlan,
    _MetadataUpdate,
    _apply_constant_plans,
    _apply_metadata_updates,
    _concat_signature,
    _constant_plans_equal,
    _constant_replacement,
    _late_constant_replacement,
    _metadata_update,
    _plan_constants,
    _plain_concat,
    _plain_prelu,
    _producer,
    _resolved_source,
)


_STATS_KEY = "optimized_sinet_deep_skip_concat_resize_affine_tail_chains"


@dataclass(frozen=True)
class _TailMatch:
    root: OperatorIR
    add2: OperatorIR
    prelu2: OperatorIR
    mul2: OperatorIR
    concat2: OperatorIR
    skip_pre: OperatorIR
    prelu1: OperatorIR
    skip_source_name: str
    skip_nchw_name: str
    concat2_output_name: str
    mul2_output_name: str
    post_output_name: str
    add2_output_name: str
    prelu2_output_name: str
    concat2_skip_index: int
    mul2_constant_name: str
    mul2_constant_index: int
    add2_constant_name: str
    add2_constant_index: int
    add2_data_index: int


@dataclass(frozen=True)
class _StageMatch:
    prelu1: OperatorIR
    add1: OperatorIR
    mul1: OperatorIR
    add0: OperatorIR
    concat1: OperatorIR
    pre_source_name: str
    concat1_output_name: str
    add0_output_name: str
    mul1_output_name: str
    add1_output_name: str
    prelu1_output_name: str
    add0_pre_index: int
    mul1_constant_name: str
    mul1_constant_index: int
    add1_constant_name: str
    add1_constant_index: int
    pre_post_candidates: Tuple[OperatorIR, ...]


@dataclass(frozen=True)
class _BranchMatch:
    concat1: OperatorIR
    pre_a: OperatorIR
    addb: OperatorIR
    mulb: OperatorIR
    pre_b: OperatorIR
    resize: OperatorIR
    source_a_name: str
    source_b_name: str
    pre_a_output_name: str
    pre_b_output_name: str
    mulb_output_name: str
    addb_output_name: str
    concat1_output_name: str
    concat1_pre_a_index: int
    mulb_constant_name: str
    mulb_constant_index: int
    mulb_data_index: int
    addb_constant_name: str
    addb_constant_index: int


@dataclass(frozen=True)
class _DeepSkipPlan:
    root: OperatorIR
    tail: _TailMatch
    stage: _StageMatch
    branch: _BranchMatch
    pre_post: Optional[OperatorIR]
    pre_canonical_name: str
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]


def _one_later_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    producer_index: int,
    op_type: str,
) -> Optional[Tuple[int, OperatorIR]]:
    indices = graph_index.consumer_indices(str(name))
    if len(indices) != 1 or int(indices[0]) <= int(producer_index):
        return None
    index = int(indices[0])
    operator = model_ir.operators[index]
    if str(operator.op_type) != str(op_type):
        return None
    return index, operator


def _resolve_tail(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
    *,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_TailMatch]:
    root_index = graph_index.operator_index(root)
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
    mul2_output_name = str(root.inputs[0])
    post_output_name = str(root.outputs[0])
    if {mul2_output_name, post_output_name} & public_names:
        return None

    add2_match = _one_later_consumer(
        model_ir,
        graph_index,
        name=post_output_name,
        producer_index=int(root_index),
        op_type="ADD",
    )
    if add2_match is None:
        return None
    add2_index, add2 = add2_match
    add2_inputs = _data_and_constant_inputs(model_ir, add2)
    if add2_inputs is None or not _plain_binary(add2, "ADD"):
        return None
    (
        add2_data_index,
        add2_data_name,
        add2_constant_index,
        add2_constant_name,
    ) = add2_inputs
    add2_output_name = str(add2.outputs[0])
    if add2_data_name != post_output_name or add2_output_name in public_names:
        return None
    prelu2_match = _one_later_consumer(
        model_ir,
        graph_index,
        name=add2_output_name,
        producer_index=int(add2_index),
        op_type="PRELU",
    )
    if prelu2_match is None:
        return None
    prelu2_index, prelu2 = prelu2_match
    prelu2_output_name = str(prelu2.outputs[0])
    if (
        not _plain_prelu(prelu2)
        or str(prelu2.inputs[0]) != add2_output_name
        or not _has_exact_producer(
            graph_index,
            prelu2_output_name,
            int(prelu2_index),
        )
        or any(
            int(index) <= int(prelu2_index)
            for index in graph_index.consumer_indices(prelu2_output_name)
        )
    ):
        return None

    mul2_match = _producer(
        model_ir,
        graph_index,
        mul2_output_name,
        "MUL",
    )
    if mul2_match is None:
        return None
    mul2_index, mul2 = mul2_match
    mul2_inputs = _data_and_constant_inputs(model_ir, mul2)
    if (
        mul2_inputs is None
        or not _plain_binary(mul2, "MUL")
        or int(mul2_index) >= int(root_index)
        or graph_index.consumer_indices(mul2_output_name)
        != [int(root_index)]
    ):
        return None
    (
        _,
        concat2_output_name,
        mul2_constant_index,
        mul2_constant_name,
    ) = mul2_inputs
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
        not _plain_concat(concat2)
        or int(concat2_index) >= int(mul2_index)
        or graph_index.consumer_indices(concat2_output_name)
        != [int(mul2_index)]
        or concat2_output_name in public_names
    ):
        return None

    roles = []
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
            str(producer.op_type) == "TRANSPOSE"
            and _typed_permutation(
                model_ir,
                graph_index,
                producer,
                _NHWC_TO_NCHW,
                public_names,
            )
        ):
            roles.append(("skip", int(input_index), int(producer_index), producer))
        elif str(producer.op_type) == "PRELU" and _plain_prelu(producer):
            roles.append(("prelu1", int(input_index), int(producer_index), producer))
        else:
            return None
    if [role[0] for role in roles].count("skip") != 1 or [
        role[0] for role in roles
    ].count("prelu1") != 1:
        return None
    _, concat2_skip_index, skip_index, skip_pre = next(
        role for role in roles if role[0] == "skip"
    )
    _, _, prelu1_index, prelu1 = next(
        role for role in roles if role[0] == "prelu1"
    )
    skip_nchw_name = str(skip_pre.outputs[0])
    skip_source_name = str(skip_pre.inputs[0])
    prelu1_output_name = str(prelu1.outputs[0])
    if (
        int(skip_index) >= int(concat2_index)
        or int(prelu1_index) >= int(concat2_index)
        or graph_index.consumer_indices(skip_nchw_name)
        != [int(concat2_index)]
        or graph_index.consumer_indices(prelu1_output_name)
        != [int(concat2_index)]
        or {skip_nchw_name, prelu1_output_name} & public_names
        or not _resolved_source(
            graph_index,
            name=skip_source_name,
            adapter_index=int(skip_index),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
    ):
        return None
    return _TailMatch(
        root=root,
        add2=add2,
        prelu2=prelu2,
        mul2=mul2,
        concat2=concat2,
        skip_pre=skip_pre,
        prelu1=prelu1,
        skip_source_name=skip_source_name,
        skip_nchw_name=skip_nchw_name,
        concat2_output_name=concat2_output_name,
        mul2_output_name=mul2_output_name,
        post_output_name=post_output_name,
        add2_output_name=add2_output_name,
        prelu2_output_name=prelu2_output_name,
        concat2_skip_index=int(concat2_skip_index),
        mul2_constant_name=mul2_constant_name,
        mul2_constant_index=int(mul2_constant_index),
        add2_constant_name=add2_constant_name,
        add2_constant_index=int(add2_constant_index),
        add2_data_index=int(add2_data_index),
    )


def _resolve_stage(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    prelu1: OperatorIR,
    *,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_StageMatch]:
    public_names = public_inputs | public_outputs
    prelu1_index = graph_index.operator_index(prelu1)
    if prelu1_index is None or not _plain_prelu(prelu1):
        return None
    prelu1_output_name = str(prelu1.outputs[0])
    add1_output_name = str(prelu1.inputs[0])
    add1_match = _producer(model_ir, graph_index, add1_output_name, "ADD")
    if add1_match is None:
        return None
    add1_index, add1 = add1_match
    add1_inputs = _data_and_constant_inputs(model_ir, add1)
    if (
        add1_inputs is None
        or not _plain_binary(add1, "ADD")
        or int(add1_index) >= int(prelu1_index)
        or graph_index.consumer_indices(add1_output_name)
        != [int(prelu1_index)]
    ):
        return None
    (
        _,
        mul1_output_name,
        add1_constant_index,
        add1_constant_name,
    ) = add1_inputs
    mul1_match = _producer(model_ir, graph_index, mul1_output_name, "MUL")
    if mul1_match is None:
        return None
    mul1_index, mul1 = mul1_match
    mul1_inputs = _data_and_constant_inputs(model_ir, mul1)
    if (
        mul1_inputs is None
        or not _plain_binary(mul1, "MUL")
        or int(mul1_index) >= int(add1_index)
        or graph_index.consumer_indices(mul1_output_name)
        != [int(add1_index)]
    ):
        return None
    (
        _,
        add0_output_name,
        mul1_constant_index,
        mul1_constant_name,
    ) = mul1_inputs
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

    concat_roles = []
    for input_index, input_name in enumerate(add0.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        producer = (
            None
            if producer_index is None
            else model_ir.operators[int(producer_index)]
        )
        if (
            producer is not None
            and _plain_concat(producer)
            and str(producer.outputs[0]) == str(input_name)
        ):
            concat_roles.append(
                (int(input_index), int(producer_index), producer)
            )
    if len(concat_roles) != 1:
        return None
    _, concat1_index, concat1 = concat_roles[0]
    concat1_output_name = str(concat1.outputs[0])
    add0_pre_index = 1 - int(concat_roles[0][0])
    pre_source_name = str(add0.inputs[add0_pre_index])
    if (
        int(concat1_index) >= int(add0_index)
        or graph_index.consumer_indices(concat1_output_name)
        != [int(add0_index)]
        or concat1_output_name in public_names
        or (tensor := model_ir.tensors.get(pre_source_name)) is None
        or tensor.data is not None
        or not _resolved_source(
            graph_index,
            name=pre_source_name,
            adapter_index=int(add0_index),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
    ):
        return None

    pre_post_candidates = []
    for consumer_index in sorted(set(graph_index.consumer_indices(pre_source_name))):
        if int(consumer_index) == int(add0_index):
            continue
        consumer = model_ir.operators[int(consumer_index)]
        if (
            int(consumer_index) < int(add0_index)
            and str(consumer.op_type) == "TRANSPOSE"
            and len(consumer.inputs) == 2
            and len(consumer.outputs) == 1
            and str(consumer.inputs[0]) == pre_source_name
            and str(consumer.outputs[0]) not in public_names
            and _typed_permutation(
                model_ir,
                graph_index,
                consumer,
                _NCHW_TO_NHWC,
                public_names,
            )
        ):
            pre_post_candidates.append(consumer)
    return _StageMatch(
        prelu1=prelu1,
        add1=add1,
        mul1=mul1,
        add0=add0,
        concat1=concat1,
        pre_source_name=pre_source_name,
        concat1_output_name=concat1_output_name,
        add0_output_name=add0_output_name,
        mul1_output_name=mul1_output_name,
        add1_output_name=add1_output_name,
        prelu1_output_name=prelu1_output_name,
        add0_pre_index=int(add0_pre_index),
        mul1_constant_name=mul1_constant_name,
        mul1_constant_index=int(mul1_constant_index),
        add1_constant_name=add1_constant_name,
        add1_constant_index=int(add1_constant_index),
        pre_post_candidates=tuple(pre_post_candidates),
    )


def _resolve_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    concat1: OperatorIR,
    *,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_BranchMatch]:
    concat1_index = graph_index.operator_index(concat1)
    public_names = public_inputs | public_outputs
    if concat1_index is None or not _plain_concat(concat1):
        return None
    roles = []
    for input_index, input_name in enumerate(concat1.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        if producer_index is None:
            return None
        producer = model_ir.operators[int(producer_index)]
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
            roles.append(("pre_a", int(input_index), int(producer_index), producer))
        elif _plain_binary(producer, "ADD"):
            roles.append(("addb", int(input_index), int(producer_index), producer))
        else:
            return None
    if [role[0] for role in roles].count("pre_a") != 1 or [
        role[0] for role in roles
    ].count("addb") != 1:
        return None
    _, concat1_pre_a_index, pre_a_index, pre_a = next(
        role for role in roles if role[0] == "pre_a"
    )
    _, _, addb_index, addb = next(
        role for role in roles if role[0] == "addb"
    )
    source_a_name = str(pre_a.inputs[0])
    pre_a_output_name = str(pre_a.outputs[0])
    addb_output_name = str(addb.outputs[0])
    if (
        max(int(pre_a_index), int(addb_index)) >= int(concat1_index)
        or graph_index.consumer_indices(pre_a_output_name)
        != [int(concat1_index)]
        or graph_index.consumer_indices(addb_output_name)
        != [int(concat1_index)]
        or {pre_a_output_name, addb_output_name} & public_names
        or not _resolved_source(
            graph_index,
            name=source_a_name,
            adapter_index=int(pre_a_index),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
    ):
        return None

    addb_inputs = _data_and_constant_inputs(model_ir, addb)
    if addb_inputs is None:
        return None
    (
        _,
        mulb_output_name,
        addb_constant_index,
        addb_constant_name,
    ) = addb_inputs
    mulb_match = _producer(model_ir, graph_index, mulb_output_name, "MUL")
    if mulb_match is None:
        return None
    mulb_index, mulb = mulb_match
    mulb_inputs = _data_and_constant_inputs(model_ir, mulb)
    if (
        mulb_inputs is None
        or not _plain_binary(mulb, "MUL")
        or int(mulb_index) >= int(addb_index)
        or graph_index.consumer_indices(mulb_output_name)
        != [int(addb_index)]
    ):
        return None
    (
        mulb_data_index,
        pre_b_output_name,
        mulb_constant_index,
        mulb_constant_name,
    ) = mulb_inputs
    pre_b_match = _producer(
        model_ir,
        graph_index,
        pre_b_output_name,
        "TRANSPOSE",
    )
    if pre_b_match is None:
        return None
    pre_b_index, pre_b = pre_b_match
    source_b_name = str(pre_b.inputs[0])
    if (
        not _typed_permutation(
            model_ir,
            graph_index,
            pre_b,
            _NHWC_TO_NCHW,
            public_names,
        )
        or int(pre_b_index) >= int(mulb_index)
        or graph_index.consumer_indices(pre_b_output_name)
        != [int(mulb_index)]
        or pre_b_output_name in public_names
    ):
        return None
    resize_match = _producer(
        model_ir,
        graph_index,
        source_b_name,
        "RESIZE_BILINEAR",
    ) or _producer(
        model_ir,
        graph_index,
        source_b_name,
        "RESIZE_NEAREST_NEIGHBOR",
    )
    if resize_match is None:
        return None
    resize_index, resize = resize_match
    if int(resize_index) >= int(pre_b_index):
        return None
    return _BranchMatch(
        concat1=concat1,
        pre_a=pre_a,
        addb=addb,
        mulb=mulb,
        pre_b=pre_b,
        resize=resize,
        source_a_name=source_a_name,
        source_b_name=source_b_name,
        pre_a_output_name=pre_a_output_name,
        pre_b_output_name=pre_b_output_name,
        mulb_output_name=mulb_output_name,
        addb_output_name=addb_output_name,
        concat1_output_name=str(concat1.outputs[0]),
        concat1_pre_a_index=int(concat1_pre_a_index),
        mulb_constant_name=mulb_constant_name,
        mulb_constant_index=int(mulb_constant_index),
        mulb_data_index=int(mulb_data_index),
        addb_constant_name=addb_constant_name,
        addb_constant_index=int(addb_constant_index),
    )


def _channel_last(tensor: object) -> bool:
    return bool(
        str(getattr(tensor, "logical_layout", "")).upper() == "NHWC"
        or str(getattr(tensor, "physical_layout", "")).upper() == "NHWC"
    )


def _metadata_for_shape(
    *,
    name: str,
    shape: Tuple[int, ...],
    signature: Tuple[int, ...],
    canonical: object,
) -> _MetadataUpdate:
    return _MetadataUpdate(
        name=str(name),
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        logical_layout=str(getattr(canonical, "logical_layout", "NHWC")),
        physical_layout=str(getattr(canonical, "physical_layout", "NHWC")),
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_DeepSkipPlan]:
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
    stage = _resolve_stage(
        model_ir,
        graph_index,
        tail.prelu1,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if stage is None:
        return None
    branch = _resolve_branch(
        model_ir,
        graph_index,
        stage.concat1,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if branch is None:
        return None

    operators = (
        tail.root,
        tail.add2,
        tail.prelu2,
        tail.mul2,
        tail.concat2,
        tail.skip_pre,
        tail.prelu1,
        stage.add1,
        stage.mul1,
        stage.add0,
        stage.concat1,
        branch.pre_a,
        branch.addb,
        branch.mulb,
        branch.pre_b,
        branch.resize,
    )
    if len({id(operator) for operator in operators}) != len(operators):
        return None

    tensor_names = (
        tail.skip_source_name,
        tail.skip_nchw_name,
        tail.concat2_output_name,
        tail.mul2_output_name,
        tail.post_output_name,
        tail.add2_output_name,
        tail.prelu2_output_name,
        stage.pre_source_name,
        stage.concat1_output_name,
        stage.add0_output_name,
        stage.mul1_output_name,
        stage.add1_output_name,
        stage.prelu1_output_name,
        branch.source_a_name,
        branch.source_b_name,
        branch.pre_a_output_name,
        branch.pre_b_output_name,
        branch.mulb_output_name,
        branch.addb_output_name,
    )
    if (
        len(set(tensor_names)) != len(tensor_names)
        or any(name in graph_index.duplicate_producers for name in tensor_names)
    ):
        return None
    contracts = {
        name: _tensor_contract(model_ir, name, 4) for name in tensor_names
    }
    if any(contract is None for contract in contracts.values()):
        return None
    dtype = str(contracts[tail.post_output_name].tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(
            str(contract.tensor.dtype) != dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in contracts.values()
            if contract is not None
        )
    ):
        return None

    skip_source = contracts[tail.skip_source_name]
    skip_nchw = contracts[tail.skip_nchw_name]
    source_a = contracts[branch.source_a_name]
    pre_a_output = contracts[branch.pre_a_output_name]
    source_b = contracts[branch.source_b_name]
    pre_b_output = contracts[branch.pre_b_output_name]
    assert all(
        value is not None
        for value in (
            skip_source,
            skip_nchw,
            source_a,
            pre_a_output,
            source_b,
            pre_b_output,
        )
    )
    if (
        skip_nchw.shape != _permute(skip_source.shape, _NHWC_TO_NCHW)
        or skip_nchw.signature
        != _permute(skip_source.signature, _NHWC_TO_NCHW)
        or pre_a_output.shape != _permute(source_a.shape, _NHWC_TO_NCHW)
        or pre_a_output.signature
        != _permute(source_a.signature, _NHWC_TO_NCHW)
        or pre_b_output.shape != _permute(source_b.shape, _NHWC_TO_NCHW)
        or pre_b_output.signature
        != _permute(source_b.signature, _NHWC_TO_NCHW)
    ):
        return None
    for name in (branch.mulb_output_name, branch.addb_output_name):
        if (
            contracts[name].shape != pre_b_output.shape
            or contracts[name].signature != pre_b_output.signature
        ):
            return None

    concat1_nchw_inputs = [contracts[str(name)] for name in stage.concat1.inputs]
    concat1_nchw_shape = _concat_signature(
        concat1_nchw_inputs[0].shape,
        concat1_nchw_inputs[1].shape,
        axis=1,
    )
    concat1_nchw_signature = _concat_signature(
        concat1_nchw_inputs[0].signature,
        concat1_nchw_inputs[1].signature,
        axis=1,
    )
    concat1_nhwc_inputs = [
        source_a if index == branch.concat1_pre_a_index else source_b
        for index in range(2)
    ]
    concat1_nhwc_shape = _concat_signature(
        concat1_nhwc_inputs[0].shape,
        concat1_nhwc_inputs[1].shape,
        axis=3,
    )
    concat1_nhwc_signature = _concat_signature(
        concat1_nhwc_inputs[0].signature,
        concat1_nhwc_inputs[1].signature,
        axis=3,
    )
    if (
        concat1_nchw_shape is None
        or concat1_nchw_signature is None
        or concat1_nhwc_shape is None
        or concat1_nhwc_signature is None
        or contracts[stage.concat1_output_name].shape != concat1_nchw_shape
        or contracts[stage.concat1_output_name].signature
        != concat1_nchw_signature
    ):
        return None

    pre_source = contracts[stage.pre_source_name]
    assert pre_source is not None
    pre_post = None
    pre_canonical_name = stage.pre_source_name
    if (
        pre_source.shape != concat1_nhwc_shape
        or pre_source.signature != concat1_nhwc_signature
        or not _channel_last(pre_source.tensor)
    ):
        candidates = []
        add0_index = graph_index.operator_index(stage.add0)
        for candidate in stage.pre_post_candidates:
            candidate_index = graph_index.operator_index(candidate)
            output_name = str(candidate.outputs[0])
            output_contract = _tensor_contract(model_ir, output_name, 4)
            if (
                candidate_index is not None
                and add0_index is not None
                and output_contract is not None
                and output_name not in public_names
                and output_name not in graph_index.duplicate_producers
                and output_contract.tensor.data is None
                and output_contract.tensor.quantization is None
                and str(output_contract.tensor.dtype) == dtype
                and output_contract.shape == concat1_nhwc_shape
                and output_contract.signature == concat1_nhwc_signature
                and pre_source.shape
                == _permute(output_contract.shape, _NHWC_TO_NCHW)
                and pre_source.signature
                == _permute(output_contract.signature, _NHWC_TO_NCHW)
                and Counter(graph_index.consumer_indices(stage.pre_source_name))
                == Counter((int(candidate_index), int(add0_index)))
            ):
                candidates.append((candidate, output_name))
        if len(candidates) != 1:
            return None
        pre_post, pre_canonical_name = candidates[0]
    pre_canonical = _tensor_contract(model_ir, pre_canonical_name, 4)
    if pre_canonical is None or not _channel_last(pre_canonical.tensor):
        return None

    for name in (
        stage.add0_output_name,
        stage.mul1_output_name,
        stage.add1_output_name,
        stage.prelu1_output_name,
    ):
        if (
            contracts[name].shape != concat1_nchw_shape
            or contracts[name].signature != concat1_nchw_signature
        ):
            return None

    concat2_nchw_inputs = [contracts[str(name)] for name in tail.concat2.inputs]
    concat2_nchw_shape = _concat_signature(
        concat2_nchw_inputs[0].shape,
        concat2_nchw_inputs[1].shape,
        axis=1,
    )
    concat2_nchw_signature = _concat_signature(
        concat2_nchw_inputs[0].signature,
        concat2_nchw_inputs[1].signature,
        axis=1,
    )
    concat2_nhwc_inputs = [
        skip_source
        if index == tail.concat2_skip_index
        else pre_canonical
        for index in range(2)
    ]
    concat2_nhwc_shape = _concat_signature(
        concat2_nhwc_inputs[0].shape,
        concat2_nhwc_inputs[1].shape,
        axis=3,
    )
    concat2_nhwc_signature = _concat_signature(
        concat2_nhwc_inputs[0].signature,
        concat2_nhwc_inputs[1].signature,
        axis=3,
    )
    if (
        concat2_nchw_shape is None
        or concat2_nchw_signature is None
        or concat2_nhwc_shape is None
        or concat2_nhwc_signature is None
        or contracts[tail.concat2_output_name].shape != concat2_nchw_shape
        or contracts[tail.concat2_output_name].signature
        != concat2_nchw_signature
    ):
        return None
    post = contracts[tail.post_output_name]
    assert post is not None
    if (
        post.shape != concat2_nhwc_shape
        or post.signature != concat2_nhwc_signature
        or not _channel_last(post.tensor)
    ):
        return None
    for name in (tail.mul2_output_name,):
        if (
            contracts[name].shape != concat2_nchw_shape
            or contracts[name].signature != concat2_nchw_signature
        ):
            return None
    for name in (tail.add2_output_name, tail.prelu2_output_name):
        if (
            contracts[name].shape != post.shape
            or contracts[name].signature != post.signature
        ):
            return None

    constant_roles = []
    late_roles = (
        (
            branch.mulb_constant_name,
            branch.mulb,
            branch.mulb_constant_index,
            pre_b_output.shape,
            source_b.shape,
        ),
        (
            branch.addb_constant_name,
            branch.addb,
            branch.addb_constant_index,
            pre_b_output.shape,
            source_b.shape,
        ),
        (
            stage.mul1_constant_name,
            stage.mul1,
            stage.mul1_constant_index,
            concat1_nchw_shape,
            pre_canonical.shape,
        ),
        (
            stage.add1_constant_name,
            stage.add1,
            stage.add1_constant_index,
            concat1_nchw_shape,
            pre_canonical.shape,
        ),
        (
            str(stage.prelu1.inputs[1]),
            stage.prelu1,
            1,
            concat1_nchw_shape,
            pre_canonical.shape,
        ),
        (
            tail.mul2_constant_name,
            tail.mul2,
            tail.mul2_constant_index,
            concat2_nchw_shape,
            post.shape,
        ),
    )
    for name, operator, input_index, old_shape, target_shape in late_roles:
        replacement = _late_constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=dtype,
            old_nchw_shape=old_shape,
            target_nhwc_shape=target_shape,
            public_names=public_names,
        )
        if replacement is None:
            return None
        constant_roles.append(
            (str(name), replacement, operator, int(input_index))
        )
    for name, operator, input_index in (
        (
            tail.add2_constant_name,
            tail.add2,
            tail.add2_constant_index,
        ),
        (str(tail.prelu2.inputs[1]), tail.prelu2, 1),
    ):
        replacement = _constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=dtype,
            target_shape=post.shape,
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

    metadata_updates = tuple(
        _metadata_update(name, source_b.tensor)
        for name in (branch.mulb_output_name, branch.addb_output_name)
    ) + tuple(
        _metadata_for_shape(
            name=name,
            shape=concat1_nhwc_shape,
            signature=concat1_nhwc_signature,
            canonical=pre_canonical.tensor,
        )
        for name in (
            stage.concat1_output_name,
            stage.add0_output_name,
            stage.mul1_output_name,
            stage.add1_output_name,
            stage.prelu1_output_name,
        )
    ) + tuple(
        _metadata_update(name, post.tensor)
        for name in (tail.concat2_output_name, tail.mul2_output_name)
    )
    return _DeepSkipPlan(
        root=root,
        tail=tail,
        stage=stage,
        branch=branch,
        pre_post=pre_post,
        pre_canonical_name=pre_canonical_name,
        constant_plans=constant_plans,
        metadata_updates=metadata_updates,
        remove_operators=(
            tail.skip_pre,
            branch.pre_a,
            branch.pre_b,
            root,
        ),
    )


def _matches_equal(expected: object, actual: object) -> bool:
    if type(expected) is not type(actual):
        return False
    for field in fields(expected):
        lhs = getattr(expected, field.name)
        rhs = getattr(actual, field.name)
        if isinstance(lhs, OperatorIR):
            if lhs is not rhs:
                return False
        elif (
            isinstance(lhs, tuple)
            and lhs
            and all(isinstance(value, OperatorIR) for value in lhs)
        ):
            if len(lhs) != len(rhs) or any(
                left is not right for left, right in zip(lhs, rhs)
            ):
                return False
        elif lhs != rhs:
            return False
    return True


def _plans_equal(expected: _DeepSkipPlan, actual: _DeepSkipPlan) -> bool:
    return bool(
        expected.root is actual.root
        and _matches_equal(expected.tail, actual.tail)
        and _matches_equal(expected.stage, actual.stage)
        and _matches_equal(expected.branch, actual.branch)
        and expected.pre_post is actual.pre_post
        and expected.pre_canonical_name == actual.pre_canonical_name
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
    plan: _DeepSkipPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.root)
    if current is None or not _plans_equal(plan, current):
        return False
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    mutation_operators = (
        plan.branch.mulb,
        plan.stage.concat1,
        plan.stage.add0,
        plan.tail.concat2,
        plan.tail.add2,
    )
    mutation_indices = [
        graph_index.operator_index(operator)
        for operator in mutation_operators
    ]
    if (
        any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or any(index is None for index in mutation_indices)
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
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.branch.mulb,
        input_index=plan.branch.mulb_data_index,
        new_input_name=plan.branch.source_b_name,
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.stage.concat1,
        input_index=plan.branch.concat1_pre_a_index,
        new_input_name=plan.branch.source_a_name,
        graph_index=graph_index,
    )
    plan.stage.concat1.options["axis"] = 3
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.stage.add0,
        input_index=plan.stage.add0_pre_index,
        new_input_name=plan.pre_canonical_name,
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.tail.concat2,
        input_index=plan.tail.concat2_skip_index,
        new_input_name=plan.tail.skip_source_name,
        graph_index=graph_index,
    )
    plan.tail.concat2.options["axis"] = 3
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.tail.add2,
        input_index=plan.tail.add2_data_index,
        new_input_name=plan.tail.mul2_output_name,
        graph_index=graph_index,
    )
    _apply_metadata_updates(model_ir, plan.metadata_updates)
    graph_index.remove_operators([int(index) for index in remove_indices])
    return True


def optimize_sinet_deep_skip_concat_resize_affine_tail_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a complete deep-skip Resize/affine residual island to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 4,
        "ADD": 4,
        "MUL": 3,
        "PRELU": 2,
        "CONCATENATION": 2,
    }
    has_resize = False
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if op_type in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}:
            has_resize = True
        if has_resize and all(
            value == 0 for value in required_counts.values()
        ):
            break
    if (
        rewrite_limit == 0
        or not has_resize
        or any(value > 0 for value in required_counts.values())
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
