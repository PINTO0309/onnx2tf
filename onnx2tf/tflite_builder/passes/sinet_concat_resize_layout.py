from __future__ import annotations

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


_STATS_KEY = "optimized_sinet_concat_resize_affine_transpose_chains"


@dataclass(frozen=True)
class _AdapterMatch:
    pre: OperatorIR
    source_name: str
    nchw_name: str


@dataclass(frozen=True)
class _AffineBranchMatch:
    resize: OperatorIR
    pre: OperatorIR
    mul: OperatorIR
    add: OperatorIR
    source_name: str
    pre_output_name: str
    mul_output_name: str
    add_output_name: str
    mul_data_index: int
    mul_constant_name: str
    mul_constant_index: int
    add_constant_name: str
    add_constant_index: int


@dataclass(frozen=True)
class _InputRewrite:
    operator: OperatorIR
    input_index: int
    old_name: str
    new_name: str


@dataclass(frozen=True)
class _ConcatResizePlan:
    root: OperatorIR
    post_adapters: Tuple[OperatorIR, ...]
    legacy_consumers: Tuple[OperatorIR, ...]
    residual: _AdapterMatch
    plain_branch: _AdapterMatch
    affine_branch: _AffineBranchMatch
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
    concat_inputs: Tuple[str, str]
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


def _resolve_adapter(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    nchw_name: str,
    consumer_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_AdapterMatch]:
    public_names = public_inputs | public_outputs
    pre_match = _producer(
        model_ir,
        graph_index,
        str(nchw_name),
        "TRANSPOSE",
    )
    if pre_match is None:
        return None
    pre_index, pre = pre_match
    if (
        int(pre_index) >= int(consumer_index)
        or str(nchw_name) in public_names
        or graph_index.consumer_indices(str(nchw_name))
        != [int(consumer_index)]
        or not _typed_permutation(
            model_ir,
            graph_index,
            pre,
            _NHWC_TO_NCHW,
            public_names,
        )
    ):
        return None
    source_name = str(pre.inputs[0])
    if not _resolved_source(
        graph_index,
        name=source_name,
        adapter_index=int(pre_index),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    ):
        return None
    return _AdapterMatch(
        pre=pre,
        source_name=source_name,
        nchw_name=str(nchw_name),
    )


def _resolve_affine_branch(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    add_output_name: str,
    concat_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_AffineBranchMatch]:
    public_names = public_inputs | public_outputs
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
        or str(add_output_name) in public_names
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

    adapter = _resolve_adapter(
        model_ir,
        graph_index,
        nchw_name=str(pre_output_name),
        consumer_index=int(mul_index),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if adapter is None:
        return None
    pre_index = graph_index.operator_index(adapter.pre)
    resize_index = graph_index.producers.get(adapter.source_name)
    if (
        pre_index is None
        or resize_index is None
        or adapter.source_name in graph_index.duplicate_producers
        or int(resize_index) >= int(pre_index)
    ):
        return None
    resize = model_ir.operators[int(resize_index)]
    if (
        str(resize.op_type)
        not in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}
        or len(resize.outputs) != 1
        or str(resize.outputs[0]) != adapter.source_name
    ):
        return None
    return _AffineBranchMatch(
        resize=resize,
        pre=adapter.pre,
        mul=mul,
        add=add,
        source_name=adapter.source_name,
        pre_output_name=str(pre_output_name),
        mul_output_name=str(mul_output_name),
        add_output_name=str(add_output_name),
        mul_data_index=int(mul_data_index),
        mul_constant_name=str(mul_constant_name),
        mul_constant_index=int(mul_constant_index),
        add_constant_name=str(add_constant_name),
        add_constant_index=int(add_constant_index),
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
) -> Optional[_ConcatResizePlan]:
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    root_index = graph_index.operator_index(root)
    if (
        root_index is None
        or str(root.op_type) != "TRANSPOSE"
        or len(root.outputs) != 1
        or str(root.outputs[0]) in public_names
        or not _typed_permutation(
            model_ir,
            graph_index,
            root,
            _NCHW_TO_NHWC,
            public_names,
        )
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
            and str(consumer.inputs[0]) == prelu2_output_name
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
        or any(int(index) <= int(prelu2_index) for index in legacy_indices)
    ):
        return None
    for adapter_index, adapter in post_adapters:
        if any(
            int(consumer_index) <= int(adapter_index)
            for consumer_index in graph_index.consumer_indices(
                str(adapter.outputs[0])
            )
        ):
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
    residual_roles = []
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
            residual_roles.append((int(input_index), str(input_name)))
    if len(concat_roles) != 1 or len(residual_roles) != 1:
        return None
    concat_input_index, concat_index, concat = concat_roles[0]
    residual_input_index, residual_nchw_name = residual_roles[0]
    concat_output_name = str(concat.outputs[0])
    if (
        int(concat_index) >= int(add0_index)
        or concat_output_name in public_names
        or graph_index.consumer_indices(concat_output_name)
        != [int(add0_index)]
    ):
        return None

    residual = _resolve_adapter(
        model_ir,
        graph_index,
        nchw_name=residual_nchw_name,
        consumer_index=int(add0_index),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if residual is None:
        return None

    plain_roles = []
    affine_roles = []
    for input_index, input_name in enumerate(concat.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        producer = (
            None
            if producer_index is None
            else model_ir.operators[int(producer_index)]
        )
        if producer is not None and str(producer.op_type) == "ADD":
            affine = _resolve_affine_branch(
                model_ir,
                graph_index,
                add_output_name=str(input_name),
                concat_index=int(concat_index),
                public_inputs=public_inputs,
                public_outputs=public_outputs,
            )
            if affine is None:
                return None
            affine_roles.append((int(input_index), affine))
        else:
            plain = _resolve_adapter(
                model_ir,
                graph_index,
                nchw_name=str(input_name),
                consumer_index=int(concat_index),
                public_inputs=public_inputs,
                public_outputs=public_outputs,
            )
            if plain is None:
                return None
            plain_roles.append((int(input_index), plain))
    if len(plain_roles) != 1 or len(affine_roles) != 1:
        return None
    plain_input_index, plain_branch = plain_roles[0]
    affine_input_index, affine_branch = affine_roles[0]

    source_names = (
        residual.source_name,
        plain_branch.source_name,
        affine_branch.source_name,
    )
    private_names = (
        residual.nchw_name,
        plain_branch.nchw_name,
        affine_branch.pre_output_name,
        affine_branch.mul_output_name,
        affine_branch.add_output_name,
        concat_output_name,
        add0_output_name,
        mul2_output_name,
        add2_output_name,
        prelu2_output_name,
        *(str(adapter.outputs[0]) for _, adapter in post_adapters),
    )
    tensor_names = source_names + private_names
    if (
        len(set(tensor_names)) != len(tensor_names)
        or any(name in graph_index.duplicate_producers for name in tensor_names)
        or any(name in public_names for name in private_names)
    ):
        return None
    contracts = {
        name: _tensor_contract(model_ir, name, 4) for name in tensor_names
    }
    if any(contract is None for contract in contracts.values()):
        return None
    post = contracts[post_output_name]
    assert post is not None
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
    ):
        return None

    for source_name, nchw_name in (
        (residual.source_name, residual.nchw_name),
        (plain_branch.source_name, plain_branch.nchw_name),
        (affine_branch.source_name, affine_branch.pre_output_name),
    ):
        source = contracts[source_name]
        nchw = contracts[nchw_name]
        assert source is not None and nchw is not None
        if (
            not _layout_allows(source.tensor, "NHWC")
            or not _layout_allows(nchw.tensor, "NCHW")
            or nchw.shape != _permute(source.shape, _NHWC_TO_NCHW)
            or nchw.signature
            != _permute(source.signature, _NHWC_TO_NCHW)
        ):
            return None

    affine_source = contracts[affine_branch.source_name]
    affine_pre = contracts[affine_branch.pre_output_name]
    affine_mul = contracts[affine_branch.mul_output_name]
    affine_add = contracts[affine_branch.add_output_name]
    assert affine_source is not None and affine_pre is not None
    assert affine_mul is not None and affine_add is not None
    if (
        affine_mul.shape != affine_pre.shape
        or affine_mul.signature != affine_pre.signature
        or affine_add.shape != affine_pre.shape
        or affine_add.signature != affine_pre.signature
        or not _layout_allows(affine_mul.tensor, "NCHW")
        or not _layout_allows(affine_add.tensor, "NCHW")
    ):
        return None

    plain_nchw = contracts[plain_branch.nchw_name]
    plain_source = contracts[plain_branch.source_name]
    residual_nchw = contracts[residual.nchw_name]
    residual_source = contracts[residual.source_name]
    concat_contract = contracts[concat_output_name]
    assert plain_nchw is not None and plain_source is not None
    assert residual_nchw is not None and residual_source is not None
    assert concat_contract is not None
    concat_nchw_shape = _concat_signature(
        plain_nchw.shape,
        affine_add.shape,
        axis=1,
    )
    concat_nchw_signature = _concat_signature(
        plain_nchw.signature,
        affine_add.signature,
        axis=1,
    )
    concat_nhwc_shape = _concat_signature(
        plain_source.shape,
        affine_source.shape,
        axis=3,
    )
    concat_nhwc_signature = _concat_signature(
        plain_source.signature,
        affine_source.signature,
        axis=3,
    )
    if (
        concat_nchw_shape is None
        or concat_nchw_signature is None
        or concat_nhwc_shape is None
        or concat_nhwc_signature is None
        or concat_contract.shape != concat_nchw_shape
        or concat_contract.signature != concat_nchw_signature
        or residual_nchw.shape != concat_nchw_shape
        or residual_nchw.signature != concat_nchw_signature
        or residual_source.shape != concat_nhwc_shape
        or residual_source.signature != concat_nhwc_signature
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
    for name, operator, input_index in (
        (
            affine_branch.mul_constant_name,
            affine_branch.mul,
            affine_branch.mul_constant_index,
        ),
        (
            affine_branch.add_constant_name,
            affine_branch.add,
            affine_branch.add_constant_index,
        ),
    ):
        replacement = _late_constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=dtype,
            old_nchw_shape=affine_pre.shape,
            target_nhwc_shape=affine_source.shape,
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

    concat_inputs = [str(name) for name in concat.inputs]
    concat_inputs[int(plain_input_index)] = plain_branch.source_name
    concat_inputs[int(affine_input_index)] = affine_branch.add_output_name
    add0_inputs = [str(name) for name in add0.inputs]
    add0_inputs[int(residual_input_index)] = residual.source_name
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
    metadata_updates = (
        _metadata_update(affine_branch.mul_output_name, affine_source.tensor),
        _metadata_update(affine_branch.add_output_name, affine_source.tensor),
        *(
            _metadata_update(name, post.tensor)
            for name in (
                concat_output_name,
                add0_output_name,
                mul2_output_name,
                add2_output_name,
            )
        ),
    )
    legacy_consumers = tuple(
        model_ir.operators[int(index)] for index in legacy_indices
    )
    remove_operators = (
        residual.pre,
        plain_branch.pre,
        affine_branch.pre,
        *(adapter for _, adapter in post_adapters),
    )
    return _ConcatResizePlan(
        root=root,
        post_adapters=tuple(adapter for _, adapter in post_adapters),
        legacy_consumers=legacy_consumers,
        residual=residual,
        plain_branch=plain_branch,
        affine_branch=affine_branch,
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
        concat_inputs=(str(concat_inputs[0]), str(concat_inputs[1])),
        add0_inputs=(str(add0_inputs[0]), str(add0_inputs[1])),
        constant_plans=constant_plans,
        metadata_updates=tuple(metadata_updates),
        input_rewrites=tuple(rewrites),
        remove_operators=tuple(remove_operators),
        insert_legacy_adapter=bool(legacy_consumers),
        legacy_adapter_perm_name=(
            str(residual.pre.inputs[1]) if legacy_consumers else None
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
        elif lhs != rhs:
            return False
    return True


def _plans_equal(
    expected: _ConcatResizePlan,
    actual: _ConcatResizePlan,
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
        and _matches_equal(expected.residual, actual.residual)
        and _matches_equal(expected.plain_branch, actual.plain_branch)
        and _matches_equal(expected.affine_branch, actual.affine_branch)
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
        and expected.concat_inputs == actual.concat_inputs
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
    plan: _ConcatResizePlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.root)
    if current is None or not _plans_equal(plan, current):
        return False
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    mutation_operators = (
        plan.affine_branch.mul,
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
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.affine_branch.mul,
        input_index=plan.affine_branch.mul_data_index,
        new_input_name=plan.affine_branch.source_name,
        graph_index=graph_index,
    )
    concat_index = graph_index.operator_index(plan.concat)
    add0_index = graph_index.operator_index(plan.add0)
    prelu2_index = graph_index.operator_index(plan.prelu2)
    assert concat_index is not None
    assert add0_index is not None
    assert prelu2_index is not None
    graph_index.replace_operator_inputs(int(concat_index), plan.concat_inputs)
    plan.concat.options["axis"] = 3
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


def optimize_sinet_concat_resize_affine_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a strict Concat/Resize affine residual island to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 4,
        "ADD": 3,
        "MUL": 2,
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
        if resize_count >= 1 and all(
            value == 0 for value in required_counts.values()
        ):
            break
    if (
        rewrite_limit == 0
        or resize_count < 1
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
