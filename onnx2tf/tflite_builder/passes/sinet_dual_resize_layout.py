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
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    _ConstantPlan,
    _MetadataUpdate,
    _apply_constant_plans,
    _apply_metadata_updates,
    _concat_signature,
    _constant_plans_equal,
    _late_constant_replacement,
    _metadata_update,
    _plan_constants,
    _plain_concat,
    _plain_prelu,
    _producer,
    _resolved_source,
)


_DIRECT_STATS_KEY = "optimized_sinet_dual_resize_affine_transpose_chains"
_SIBLING_STATS_KEY = (
    "optimized_sinet_deep_skip_dual_resize_affine_transpose_chains"
)


@dataclass(frozen=True)
class _BranchMatch:
    resize: OperatorIR
    pre: OperatorIR
    mul: OperatorIR
    add: OperatorIR
    resize_output_name: str
    pre_output_name: str
    mul_output_name: str
    add_output_name: str
    mul_data_index: int
    mul_constant_name: str
    mul_constant_index: int
    add_constant_name: str
    add_constant_index: int


@dataclass(frozen=True)
class _ResidualMatch:
    mode: str
    adapter: OperatorIR
    nchw_name: str
    nhwc_name: str
    remove_adapter: bool


@dataclass(frozen=True)
class _InputRewrite:
    operator: OperatorIR
    input_index: int
    old_name: str
    new_name: str


@dataclass(frozen=True)
class _DualResizePlan:
    root: OperatorIR
    post_adapters: Tuple[OperatorIR, ...]
    legacy_consumers: Tuple[OperatorIR, ...]
    branches: Tuple[_BranchMatch, _BranchMatch]
    residual: _ResidualMatch
    concat: OperatorIR
    add0: OperatorIR
    mul2: OperatorIR
    add2: OperatorIR
    prelu2: OperatorIR
    concat_output_name: str
    add0_output_name: str
    mul2_output_name: str
    add2_output_name: str
    prelu2_output_name: str
    post_output_name: str
    add0_inputs: Tuple[str, str]
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    input_rewrites: Tuple[_InputRewrite, ...]
    remove_operators: Tuple[OperatorIR, ...]
    insert_legacy_adapter: bool
    legacy_adapter_perm_name: Optional[str]


def _layout_allows(tensor: TensorIR, expected: str) -> bool:
    expected_name = str(expected).upper()
    for value in (tensor.logical_layout, tensor.physical_layout):
        normalized = str(value).strip().upper()
        if normalized not in {"", "UNKNOWN", expected_name}:
            return False
    return True


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


def _resolve_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    add_output_name: str,
    concat_index: int,
    public_names: set[str],
) -> Optional[_BranchMatch]:
    add_match = _producer(
        model_ir,
        graph_index,
        str(add_output_name),
        "ADD",
    )
    if add_match is None:
        return None
    add_index, add = add_match
    add_inputs = _data_and_constant_inputs(model_ir, add)
    if (
        add_inputs is None
        or not _plain_binary(add, "ADD")
        or int(add_index) >= int(concat_index)
        or graph_index.consumer_indices(str(add_output_name))
        != [int(concat_index)]
    ):
        return None
    (
        _,
        mul_output_name,
        add_constant_index,
        add_constant_name,
    ) = add_inputs

    mul_match = _producer(
        model_ir,
        graph_index,
        str(mul_output_name),
        "MUL",
    )
    if mul_match is None:
        return None
    mul_index, mul = mul_match
    mul_inputs = _data_and_constant_inputs(model_ir, mul)
    if (
        mul_inputs is None
        or not _plain_binary(mul, "MUL")
        or int(mul_index) >= int(add_index)
        or graph_index.consumer_indices(str(mul_output_name))
        != [int(add_index)]
    ):
        return None
    (
        mul_data_index,
        pre_output_name,
        mul_constant_index,
        mul_constant_name,
    ) = mul_inputs

    pre_match = _producer(
        model_ir,
        graph_index,
        str(pre_output_name),
        "TRANSPOSE",
    )
    if pre_match is None:
        return None
    pre_index, pre = pre_match
    if (
        int(pre_index) >= int(mul_index)
        or str(pre_output_name) in public_names
        or graph_index.consumer_indices(str(pre_output_name))
        != [int(mul_index)]
        or not _typed_permutation(
            model_ir,
            graph_index,
            pre,
            _NHWC_TO_NCHW,
            public_names,
        )
    ):
        return None
    resize_output_name = str(pre.inputs[0])
    resize_index = graph_index.producers.get(resize_output_name)
    if resize_index is None or int(resize_index) >= int(pre_index):
        return None
    resize = model_ir.operators[int(resize_index)]
    if (
        str(resize.op_type)
        not in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}
        or len(resize.outputs) != 1
        or str(resize.outputs[0]) != resize_output_name
        or resize_output_name in graph_index.duplicate_producers
    ):
        return None
    return _BranchMatch(
        resize=resize,
        pre=pre,
        mul=mul,
        add=add,
        resize_output_name=resize_output_name,
        pre_output_name=str(pre_output_name),
        mul_output_name=str(mul_output_name),
        add_output_name=str(add_output_name),
        mul_data_index=int(mul_data_index),
        mul_constant_name=str(mul_constant_name),
        mul_constant_index=int(mul_constant_index),
        add_constant_name=str(add_constant_name),
        add_constant_index=int(add_constant_index),
    )


def _resolve_residual(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    nchw_name: str,
    add0_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_ResidualMatch]:
    public_names = public_inputs | public_outputs
    producer_index = graph_index.producers.get(str(nchw_name))
    if producer_index is not None:
        direct = model_ir.operators[int(producer_index)]
        if (
            int(producer_index) < int(add0_index)
            and str(direct.op_type) == "TRANSPOSE"
            and len(direct.outputs) == 1
            and str(direct.outputs[0]) == str(nchw_name)
            and str(nchw_name) not in public_names
            and graph_index.consumer_indices(str(nchw_name))
            == [int(add0_index)]
            and _typed_permutation(
                model_ir,
                graph_index,
                direct,
                _NHWC_TO_NCHW,
                public_names,
            )
            and _resolved_source(
                graph_index,
                name=str(direct.inputs[0]),
                adapter_index=int(producer_index),
                public_inputs=public_inputs,
                public_outputs=public_outputs,
            )
        ):
            return _ResidualMatch(
                mode="direct",
                adapter=direct,
                nchw_name=str(nchw_name),
                nhwc_name=str(direct.inputs[0]),
                remove_adapter=True,
            )

    sibling_matches = []
    for consumer_index in sorted(
        set(graph_index.consumer_indices(str(nchw_name)))
    ):
        if int(consumer_index) == int(add0_index):
            continue
        sibling = model_ir.operators[int(consumer_index)]
        if (
            int(consumer_index) < int(add0_index)
            and str(sibling.op_type) == "TRANSPOSE"
            and len(sibling.inputs) == 2
            and str(sibling.inputs[0]) == str(nchw_name)
            and len(sibling.outputs) == 1
            and str(sibling.outputs[0]) not in public_names
            and _typed_permutation(
                model_ir,
                graph_index,
                sibling,
                _NCHW_TO_NHWC,
                public_names,
            )
        ):
            sibling_matches.append((int(consumer_index), sibling))
    if len(sibling_matches) != 1:
        return None
    sibling_index, sibling = sibling_matches[0]
    if (
        Counter(graph_index.consumer_indices(str(nchw_name)))
        != Counter((int(sibling_index), int(add0_index)))
        or not _resolved_source(
            graph_index,
            name=str(nchw_name),
            adapter_index=int(sibling_index),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
    ):
        return None
    return _ResidualMatch(
        mode="sibling",
        adapter=sibling,
        nchw_name=str(nchw_name),
        nhwc_name=str(sibling.outputs[0]),
        remove_adapter=False,
    )


def _input_rewrites(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    old_name: str,
    new_name: str,
    excluded: set[int],
) -> Tuple[_InputRewrite, ...]:
    rewrites = []
    for operator_index in sorted(
        set(graph_index.consumer_indices(str(old_name)))
    ):
        if int(operator_index) in excluded:
            continue
        operator = model_ir.operators[int(operator_index)]
        for input_index, input_name in enumerate(operator.inputs):
            if str(input_name) == str(old_name):
                rewrites.append(
                    _InputRewrite(
                        operator=operator,
                        input_index=int(input_index),
                        old_name=str(old_name),
                        new_name=str(new_name),
                    )
                )
    return tuple(rewrites)


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
    *,
    residual_mode: str,
) -> Optional[_DualResizePlan]:
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    root_index = graph_index.operator_index(root)
    if (
        root_index is None
        or str(root.op_type) != "TRANSPOSE"
        or len(root.outputs) != 1
        or not _typed_permutation(
            model_ir,
            graph_index,
            root,
            _NCHW_TO_NHWC,
            public_names,
        )
        or str(root.outputs[0]) in public_names
    ):
        return None

    prelu2_output_name = str(root.inputs[0])
    prelu2_match = _producer(
        model_ir,
        graph_index,
        prelu2_output_name,
        "PRELU",
    )
    if prelu2_match is None or prelu2_output_name in public_names:
        return None
    prelu2_index, prelu2 = prelu2_match
    if not _plain_prelu(prelu2) or int(prelu2_index) >= int(root_index):
        return None

    post_adapters = []
    legacy_indices = []
    for consumer_index in sorted(
        set(graph_index.consumer_indices(prelu2_output_name))
    ):
        consumer = model_ir.operators[int(consumer_index)]
        if (
            int(consumer_index) > int(prelu2_index)
            and str(consumer.op_type) == "TRANSPOSE"
            and len(consumer.outputs) == 1
            and str(consumer.outputs[0]) not in public_names
            and _typed_permutation(
                model_ir,
                graph_index,
                consumer,
                _NCHW_TO_NHWC,
                public_names,
            )
        ):
            post_adapters.append((int(consumer_index), consumer))
        else:
            legacy_indices.append(int(consumer_index))
    if (
        not post_adapters
        or post_adapters[0][1] is not root
        or any(int(index) <= int(root_index) for index in legacy_indices)
    ):
        return None
    for _, adapter in post_adapters:
        output_name = str(adapter.outputs[0])
        if not graph_index.consumer_indices(output_name):
            return None
    post_output_name = str(root.outputs[0])

    add2_output_name = str(prelu2.inputs[0])
    add2_match = _producer(
        model_ir,
        graph_index,
        add2_output_name,
        "ADD",
    )
    if add2_match is None:
        return None
    add2_index, add2 = add2_match
    add2_inputs = _data_and_constant_inputs(model_ir, add2)
    if (
        add2_inputs is None
        or not _plain_binary(add2, "ADD")
        or int(add2_index) >= int(prelu2_index)
        or graph_index.consumer_indices(add2_output_name)
        != [int(prelu2_index)]
    ):
        return None
    (
        _,
        mul2_output_name,
        add2_constant_index,
        add2_constant_name,
    ) = add2_inputs

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
        or int(mul2_index) >= int(add2_index)
        or graph_index.consumer_indices(mul2_output_name)
        != [int(add2_index)]
    ):
        return None
    (
        _,
        add0_output_name,
        mul2_constant_index,
        mul2_constant_name,
    ) = mul2_inputs

    add0_match = _producer(
        model_ir,
        graph_index,
        add0_output_name,
        "ADD",
    )
    if add0_match is None:
        return None
    add0_index, add0 = add0_match
    if (
        not _plain_binary(add0, "ADD")
        or int(add0_index) >= int(mul2_index)
        or graph_index.consumer_indices(add0_output_name)
        != [int(mul2_index)]
    ):
        return None

    concat_roles = []
    residual_names = []
    for input_index, input_name in enumerate(add0.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        producer = (
            None
            if producer_index is None
            else model_ir.operators[int(producer_index)]
        )
        if producer is not None and _plain_concat(producer):
            concat_roles.append(
                (int(input_index), int(producer_index), producer)
            )
        else:
            residual_names.append((int(input_index), str(input_name)))
    if len(concat_roles) != 1 or len(residual_names) != 1:
        return None
    concat_input_index, concat_index, concat = concat_roles[0]
    residual_input_index, residual_nchw_name = residual_names[0]
    concat_output_name = str(concat.outputs[0])
    if (
        int(concat_index) >= int(add0_index)
        or graph_index.consumer_indices(concat_output_name)
        != [int(add0_index)]
        or concat_output_name in public_names
    ):
        return None

    residual = _resolve_residual(
        model_ir,
        graph_index,
        nchw_name=residual_nchw_name,
        add0_index=int(add0_index),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if residual is None or str(residual.mode) != str(residual_mode):
        return None

    branch_matches = []
    for input_name in concat.inputs:
        branch = _resolve_branch(
            model_ir,
            graph_index,
            add_output_name=str(input_name),
            concat_index=int(concat_index),
            public_names=public_names,
        )
        if branch is None:
            return None
        branch_matches.append(branch)
    if len(branch_matches) != 2:
        return None
    branches = (branch_matches[0], branch_matches[1])

    tensor_names = [
        residual.nchw_name,
        residual.nhwc_name,
        concat_output_name,
        add0_output_name,
        mul2_output_name,
        add2_output_name,
        prelu2_output_name,
    ]
    for branch in branches:
        tensor_names.extend(
            (
                branch.resize_output_name,
                branch.pre_output_name,
                branch.mul_output_name,
                branch.add_output_name,
            )
        )
    tensor_names.extend(str(adapter.outputs[0]) for _, adapter in post_adapters)
    if (
        len(set(tensor_names)) != len(tensor_names)
        or any(name in graph_index.duplicate_producers for name in tensor_names)
        or any(name in public_names for name in tensor_names[2:-1])
    ):
        return None
    contracts = {
        name: _tensor_contract(model_ir, name, 4) for name in tensor_names
    }
    if any(contract is None for contract in contracts.values()):
        return None
    post = contracts[post_output_name]
    residual_nchw = contracts[residual.nchw_name]
    residual_nhwc = contracts[residual.nhwc_name]
    assert post is not None and residual_nchw is not None
    assert residual_nhwc is not None
    dtype = str(post.tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(
            str(contract.tensor.dtype) != dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in contracts.values()
            if contract is not None
        )
        or not _layout_allows(post.tensor, "NHWC")
        or not _layout_allows(residual_nchw.tensor, "NCHW")
        or not _layout_allows(residual_nhwc.tensor, "NHWC")
        or residual_nchw.shape
        != _permute(residual_nhwc.shape, _NHWC_TO_NCHW)
        or residual_nchw.signature
        != _permute(residual_nhwc.signature, _NHWC_TO_NCHW)
        or residual_nhwc.shape != post.shape
        or residual_nhwc.signature != post.signature
    ):
        return None

    branch_nchw_contracts = []
    branch_nhwc_contracts = []
    for branch in branches:
        resize_output = contracts[branch.resize_output_name]
        pre_output = contracts[branch.pre_output_name]
        mul_output = contracts[branch.mul_output_name]
        add_output = contracts[branch.add_output_name]
        assert resize_output is not None and pre_output is not None
        assert mul_output is not None and add_output is not None
        if (
            not _layout_allows(resize_output.tensor, "NHWC")
            or not _layout_allows(pre_output.tensor, "NCHW")
            or resize_output.shape
            != _permute(pre_output.shape, _NCHW_TO_NHWC)
            or resize_output.signature
            != _permute(pre_output.signature, _NCHW_TO_NHWC)
            or mul_output.shape != pre_output.shape
            or mul_output.signature != pre_output.signature
            or add_output.shape != pre_output.shape
            or add_output.signature != pre_output.signature
        ):
            return None
        branch_nchw_contracts.append(add_output)
        branch_nhwc_contracts.append(resize_output)

    concat_nchw_shape = _concat_signature(
        branch_nchw_contracts[0].shape,
        branch_nchw_contracts[1].shape,
        axis=1,
    )
    concat_nchw_signature = _concat_signature(
        branch_nchw_contracts[0].signature,
        branch_nchw_contracts[1].signature,
        axis=1,
    )
    concat_nhwc_shape = _concat_signature(
        branch_nhwc_contracts[0].shape,
        branch_nhwc_contracts[1].shape,
        axis=3,
    )
    concat_nhwc_signature = _concat_signature(
        branch_nhwc_contracts[0].signature,
        branch_nhwc_contracts[1].signature,
        axis=3,
    )
    concat_contract = contracts[concat_output_name]
    if (
        concat_nchw_shape is None
        or concat_nchw_signature is None
        or concat_nhwc_shape is None
        or concat_nhwc_signature is None
        or concat_contract is None
        or concat_contract.shape != concat_nchw_shape
        or concat_contract.signature != concat_nchw_signature
        or residual_nchw.shape != concat_nchw_shape
        or residual_nchw.signature != concat_nchw_signature
        or post.shape != concat_nhwc_shape
        or post.signature != concat_nhwc_signature
    ):
        return None
    for name in (
        add0_output_name,
        mul2_output_name,
        add2_output_name,
        prelu2_output_name,
    ):
        contract = contracts[name]
        assert contract is not None
        if (
            contract.shape != concat_nchw_shape
            or contract.signature != concat_nchw_signature
            or not _layout_allows(contract.tensor, "NCHW")
        ):
            return None
    for _, adapter in post_adapters[1:]:
        alias = contracts[str(adapter.outputs[0])]
        assert alias is not None
        if alias.shape != post.shape or alias.signature != post.signature:
            return None

    constant_roles = []
    for branch, target in zip(branches, branch_nhwc_contracts):
        old = contracts[branch.pre_output_name]
        assert old is not None
        for name, operator, input_index in (
            (
                branch.mul_constant_name,
                branch.mul,
                branch.mul_constant_index,
            ),
            (
                branch.add_constant_name,
                branch.add,
                branch.add_constant_index,
            ),
        ):
            replacement = _late_constant_replacement(
                model_ir,
                graph_index,
                name=str(name),
                dtype=dtype,
                old_nchw_shape=old.shape,
                target_nhwc_shape=target.shape,
                public_names=public_names,
            )
            if replacement is None:
                return None
            constant_roles.append(
                (str(name), replacement, operator, int(input_index))
            )
    for name, operator, input_index in (
        (mul2_constant_name, mul2, mul2_constant_index),
        (add2_constant_name, add2, add2_constant_index),
        (str(prelu2.inputs[1]), prelu2, 1),
    ):
        replacement = _late_constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=dtype,
            old_nchw_shape=concat_nchw_shape,
            target_nhwc_shape=post.shape,
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

    add0_inputs = [str(name) for name in add0.inputs]
    add0_inputs[int(residual_input_index)] = residual.nhwc_name
    add0_inputs[int(concat_input_index)] = concat_output_name
    post_indices = {int(index) for index, _ in post_adapters}
    rewrites = []
    for _, adapter in post_adapters[1:]:
        rewrites.extend(
            _input_rewrites(
                model_ir,
                graph_index,
                old_name=str(adapter.outputs[0]),
                new_name=post_output_name,
                excluded=post_indices,
            )
        )
    insert_legacy_adapter = bool(
        residual.mode == "direct" and legacy_indices
    )
    if residual.mode == "sibling":
        rewrites.extend(
            _input_rewrites(
                model_ir,
                graph_index,
                old_name=prelu2_output_name,
                new_name=post_output_name,
                excluded=post_indices,
            )
        )
    metadata_updates = tuple(
        _metadata_update(name, target.tensor)
        for branch, target in zip(branches, branch_nhwc_contracts)
        for name in (branch.mul_output_name, branch.add_output_name)
    ) + tuple(
        _metadata_update(name, post.tensor)
        for name in (
            concat_output_name,
            add0_output_name,
            mul2_output_name,
            add2_output_name,
        )
    )
    remove_operators = tuple(branch.pre for branch in branches)
    if residual.remove_adapter:
        remove_operators += (residual.adapter,)
    remove_operators += tuple(adapter for _, adapter in post_adapters)
    legacy_consumers = tuple(
        model_ir.operators[int(index)] for index in legacy_indices
    )
    return _DualResizePlan(
        root=root,
        post_adapters=tuple(adapter for _, adapter in post_adapters),
        legacy_consumers=legacy_consumers,
        branches=branches,
        residual=residual,
        concat=concat,
        add0=add0,
        mul2=mul2,
        add2=add2,
        prelu2=prelu2,
        concat_output_name=concat_output_name,
        add0_output_name=add0_output_name,
        mul2_output_name=mul2_output_name,
        add2_output_name=add2_output_name,
        prelu2_output_name=prelu2_output_name,
        post_output_name=post_output_name,
        add0_inputs=(str(add0_inputs[0]), str(add0_inputs[1])),
        constant_plans=constant_plans,
        metadata_updates=metadata_updates,
        input_rewrites=tuple(rewrites),
        remove_operators=remove_operators,
        insert_legacy_adapter=insert_legacy_adapter,
        legacy_adapter_perm_name=(
            str(residual.adapter.inputs[1])
            if insert_legacy_adapter
            else None
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


def _plans_equal(
    expected: _DualResizePlan,
    actual: _DualResizePlan,
) -> bool:
    return bool(
        expected.root is actual.root
        and len(expected.post_adapters) == len(actual.post_adapters)
        and all(
            lhs is rhs
            for lhs, rhs in zip(
                expected.post_adapters,
                actual.post_adapters,
            )
        )
        and len(expected.legacy_consumers) == len(actual.legacy_consumers)
        and all(
            lhs is rhs
            for lhs, rhs in zip(
                expected.legacy_consumers,
                actual.legacy_consumers,
            )
        )
        and all(
            _matches_equal(lhs, rhs)
            for lhs, rhs in zip(expected.branches, actual.branches)
        )
        and _matches_equal(expected.residual, actual.residual)
        and all(
            getattr(expected, name) is getattr(actual, name)
            for name in ("concat", "add0", "mul2", "add2", "prelu2")
        )
        and expected.concat_output_name == actual.concat_output_name
        and expected.add0_output_name == actual.add0_output_name
        and expected.mul2_output_name == actual.mul2_output_name
        and expected.add2_output_name == actual.add2_output_name
        and expected.prelu2_output_name == actual.prelu2_output_name
        and expected.post_output_name == actual.post_output_name
        and expected.add0_inputs == actual.add0_inputs
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
        and len(expected.remove_operators) == len(actual.remove_operators)
        and all(
            lhs is rhs
            for lhs, rhs in zip(
                expected.remove_operators,
                actual.remove_operators,
            )
        )
        and expected.insert_legacy_adapter == actual.insert_legacy_adapter
        and expected.legacy_adapter_perm_name
        == actual.legacy_adapter_perm_name
        and _constant_plans_equal(
            expected.constant_plans,
            actual.constant_plans,
        )
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _DualResizePlan,
) -> bool:
    current = _resolve_candidate(
        model_ir,
        graph_index,
        plan.root,
        residual_mode=plan.residual.mode,
    )
    if current is None or not _plans_equal(plan, current):
        return False
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    mutation_operators = (
        *(branch.mul for branch in plan.branches),
        plan.concat,
        plan.add0,
        plan.prelu2,
        *(rewrite.operator for rewrite in plan.input_rewrites),
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
            plan.insert_legacy_adapter
            and (
                not plan.legacy_consumers
                or plan.legacy_adapter_perm_name not in model_ir.tensors
                or any(
                    graph_index.operator_index(operator) is None
                    for operator in plan.legacy_consumers
                )
            )
        )
    ):
        return False

    _apply_constant_plans(model_ir, graph_index, plan.constant_plans)
    for branch in plan.branches:
        _replace_operator_input_at(
            model_ir=model_ir,
            op=branch.mul,
            input_index=branch.mul_data_index,
            new_input_name=branch.resize_output_name,
            graph_index=graph_index,
        )
    plan.concat.options["axis"] = 3
    add0_index = graph_index.operator_index(plan.add0)
    prelu2_index = graph_index.operator_index(plan.prelu2)
    assert add0_index is not None and prelu2_index is not None
    graph_index.replace_operator_inputs(int(add0_index), plan.add0_inputs)
    graph_index.replace_operator_outputs(
        int(prelu2_index),
        [plan.post_output_name],
    )
    for rewrite in plan.input_rewrites:
        _replace_operator_input_at(
            model_ir=model_ir,
            op=rewrite.operator,
            input_index=rewrite.input_index,
            new_input_name=rewrite.new_name,
            graph_index=graph_index,
        )
    _apply_metadata_updates(model_ir, plan.metadata_updates)
    graph_index.remove_operators([int(index) for index in remove_indices])

    if plan.insert_legacy_adapter:
        live_legacy_indices = [
            graph_index.operator_index(operator)
            for operator in plan.legacy_consumers
        ]
        if any(index is None for index in live_legacy_indices):
            raise AssertionError("legacy consumer disappeared after preflight")
        insert_index = min(
            int(index) for index in live_legacy_indices if index is not None
        )
        graph_index.insert_operator(
            int(insert_index),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[
                    plan.post_output_name,
                    str(plan.legacy_adapter_perm_name),
                ],
                outputs=[plan.prelu2_output_name],
                options={},
            ),
        )
    return True


def _optimize(
    model_ir: ModelIR,
    *,
    residual_mode: str,
    stats_key: str,
    graph_index: Optional[ModelIRGraphIndex],
    layout_state: Optional[LayoutState],
    max_rewrites: int,
    candidate: Optional[OperatorIR],
) -> dict[str, int]:
    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 4 if residual_mode == "direct" else 3,
        "ADD": 4,
        "MUL": 3,
        "PRELU": 1,
        "CONCATENATION": 1,
    }
    resize_count = 0
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if op_type in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}:
            resize_count += 1
        if resize_count >= 2 and all(
            value == 0 for value in required_counts.values()
        ):
            break
    if (
        rewrite_limit == 0
        or resize_count < 2
        or any(value > 0 for value in required_counts.values())
    ):
        return {str(stats_key): 0}

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
        plan = _resolve_candidate(
            model_ir,
            active_index,
            root,
            residual_mode=str(residual_mode),
        )
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {str(stats_key): int(rewritten)}


def optimize_sinet_dual_resize_affine_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a direct-adapter dual-Resize residual island to NHWC."""

    return _optimize(
        model_ir,
        residual_mode="direct",
        stats_key=_DIRECT_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def optimize_sinet_deep_skip_dual_resize_affine_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a sibling-adapter dual-Resize residual island to NHWC."""

    return _optimize(
        model_ir,
        residual_mode="sibling",
        stats_key=_SIBLING_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )
