from __future__ import annotations

from dataclasses import dataclass
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
    _permute,
    _plain_binary,
    _tensor_contract,
    _typed_permutation,
)
from onnx2tf.tflite_builder.passes.sinet_concat_resize_layout import (
    _AdapterMatch,
    _AffineBranchMatch,
    _InputRewrite,
    _input_rewrites,
    _layout_allows,
    _matches_equal,
    _resolve_adapter,
    _resolve_affine_branch,
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
)


_STATS_KEY = (
    "optimized_sinet_concat_resize_affine_tail_concat_transpose_chains"
)


@dataclass(frozen=True)
class _TailConcatPlan:
    root: OperatorIR
    post_adapters: Tuple[OperatorIR, ...]
    legacy_consumers: Tuple[OperatorIR, ...]
    skip: _AdapterMatch
    residual: _AdapterMatch
    plain_branch: _AdapterMatch
    affine_branch: _AffineBranchMatch
    concat1: OperatorIR
    add0: OperatorIR
    mul1: OperatorIR
    add1: OperatorIR
    prelu1: OperatorIR
    concat2: OperatorIR
    mul2: OperatorIR
    add2: OperatorIR
    prelu2: OperatorIR
    concat1_output_name: str
    add0_output_name: str
    mul1_output_name: str
    add1_output_name: str
    prelu1_output_name: str
    concat2_output_name: str
    mul2_output_name: str
    add2_output_name: str
    prelu2_output_name: str
    post_output_name: str
    concat1_inputs: Tuple[str, str]
    add0_inputs: Tuple[str, str]
    concat2_inputs: Tuple[str, str]
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    input_rewrites: Tuple[_InputRewrite, ...]
    remove_operators: Tuple[OperatorIR, ...]
    insert_legacy_adapter: bool
    legacy_adapter_perm_name: Optional[str]


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_TailConcatPlan]:
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
        or concat2_output_name in public_names
        or graph_index.consumer_indices(concat2_output_name)
        != [int(mul2_index)]
    ):
        return None

    skip_roles = []
    prelu1_roles = []
    for input_index, input_name in enumerate(concat2.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        producer = (
            None
            if producer_index is None
            else model_ir.operators[int(producer_index)]
        )
        if producer is not None and str(producer.op_type) == "PRELU":
            match = _producer(
                model_ir,
                graph_index,
                str(input_name),
                "PRELU",
            )
            if match is None:
                return None
            prelu1_index, prelu1 = match
            if (
                not _plain_prelu(prelu1)
                or int(prelu1_index) >= int(concat2_index)
                or str(input_name) in public_names
                or graph_index.consumer_indices(str(input_name))
                != [int(concat2_index)]
            ):
                return None
            prelu1_roles.append((int(input_index), int(prelu1_index), prelu1))
        else:
            skip = _resolve_adapter(
                model_ir,
                graph_index,
                nchw_name=str(input_name),
                consumer_index=int(concat2_index),
                public_inputs=public_inputs,
                public_outputs=public_outputs,
            )
            if skip is None:
                return None
            skip_roles.append((int(input_index), skip))
    if len(skip_roles) != 1 or len(prelu1_roles) != 1:
        return None
    skip_input_index, skip = skip_roles[0]
    prelu1_input_index, prelu1_index, prelu1 = prelu1_roles[0]
    prelu1_output_name = str(prelu1.outputs[0])

    add1_output_name = str(prelu1.inputs[0])
    add1_match = _producer(
        model_ir,
        graph_index,
        add1_output_name,
        "ADD",
    )
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

    mul1_match = _producer(
        model_ir,
        graph_index,
        mul1_output_name,
        "MUL",
    )
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
        or int(add0_index) >= int(mul1_index)
        or graph_index.consumer_indices(add0_output_name)
        != [int(mul1_index)]
    ):
        return None

    concat1_roles = []
    residual_roles = []
    for input_index, input_name in enumerate(add0.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        producer = (
            None
            if producer_index is None
            else model_ir.operators[int(producer_index)]
        )
        if producer is not None and _plain_concat(producer):
            concat1_roles.append(
                (int(input_index), int(producer_index), producer)
            )
        else:
            residual_roles.append((int(input_index), str(input_name)))
    if len(concat1_roles) != 1 or len(residual_roles) != 1:
        return None
    concat1_input_index, concat1_index, concat1 = concat1_roles[0]
    residual_input_index, residual_nchw_name = residual_roles[0]
    concat1_output_name = str(concat1.outputs[0])
    if (
        int(concat1_index) >= int(add0_index)
        or concat1_output_name in public_names
        or graph_index.consumer_indices(concat1_output_name)
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
    for input_index, input_name in enumerate(concat1.inputs):
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
                concat_index=int(concat1_index),
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
                consumer_index=int(concat1_index),
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
        skip.source_name,
        residual.source_name,
        plain_branch.source_name,
        affine_branch.source_name,
    )
    private_names = (
        skip.nchw_name,
        residual.nchw_name,
        plain_branch.nchw_name,
        affine_branch.pre_output_name,
        affine_branch.mul_output_name,
        affine_branch.add_output_name,
        concat1_output_name,
        add0_output_name,
        mul1_output_name,
        add1_output_name,
        prelu1_output_name,
        concat2_output_name,
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
        (skip.source_name, skip.nchw_name),
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
    concat1_contract = contracts[concat1_output_name]
    assert plain_nchw is not None and plain_source is not None
    assert residual_nchw is not None and residual_source is not None
    assert concat1_contract is not None
    concat1_nchw_shape = _concat_signature(
        plain_nchw.shape,
        affine_add.shape,
        axis=1,
    )
    concat1_nchw_signature = _concat_signature(
        plain_nchw.signature,
        affine_add.signature,
        axis=1,
    )
    concat1_nhwc_shape = _concat_signature(
        plain_source.shape,
        affine_source.shape,
        axis=3,
    )
    concat1_nhwc_signature = _concat_signature(
        plain_source.signature,
        affine_source.signature,
        axis=3,
    )
    if (
        concat1_nchw_shape is None
        or concat1_nchw_signature is None
        or concat1_nhwc_shape is None
        or concat1_nhwc_signature is None
        or concat1_contract.shape != concat1_nchw_shape
        or concat1_contract.signature != concat1_nchw_signature
        or residual_nchw.shape != concat1_nchw_shape
        or residual_nchw.signature != concat1_nchw_signature
        or residual_source.shape != concat1_nhwc_shape
        or residual_source.signature != concat1_nhwc_signature
    ):
        return None
    for name in (
        add0_output_name,
        mul1_output_name,
        add1_output_name,
        prelu1_output_name,
    ):
        contract = contracts[name]
        assert contract is not None
        if (
            contract.shape != concat1_nchw_shape
            or contract.signature != concat1_nchw_signature
            or not _layout_allows(contract.tensor, "NCHW")
        ):
            return None

    skip_nchw = contracts[skip.nchw_name]
    skip_source = contracts[skip.source_name]
    prelu1_contract = contracts[prelu1_output_name]
    concat2_contract = contracts[concat2_output_name]
    assert skip_nchw is not None and skip_source is not None
    assert prelu1_contract is not None and concat2_contract is not None
    concat2_nchw_shape = _concat_signature(
        skip_nchw.shape,
        prelu1_contract.shape,
        axis=1,
    )
    concat2_nchw_signature = _concat_signature(
        skip_nchw.signature,
        prelu1_contract.signature,
        axis=1,
    )
    concat2_nhwc_shape = _concat_signature(
        skip_source.shape,
        residual_source.shape,
        axis=3,
    )
    concat2_nhwc_signature = _concat_signature(
        skip_source.signature,
        residual_source.signature,
        axis=3,
    )
    if (
        concat2_nchw_shape is None
        or concat2_nchw_signature is None
        or concat2_nhwc_shape is None
        or concat2_nhwc_signature is None
        or concat2_contract.shape != concat2_nchw_shape
        or concat2_contract.signature != concat2_nchw_signature
        or post.shape != concat2_nhwc_shape
        or post.signature != concat2_nhwc_signature
    ):
        return None
    for name in (mul2_output_name, add2_output_name, prelu2_output_name):
        contract = contracts[name]
        assert contract is not None
        if (
            contract.shape != concat2_nchw_shape
            or contract.signature != concat2_nchw_signature
            or not _layout_allows(contract.tensor, "NCHW")
        ):
            return None
    for _, adapter in post_adapters[1:]:
        alias = contracts[str(adapter.outputs[0])]
        assert alias is not None
        if alias.shape != post.shape or alias.signature != post.signature:
            return None

    constant_roles = []
    constant_groups = (
        (
            affine_pre.shape,
            affine_source.shape,
            (
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
            ),
        ),
        (
            concat1_nchw_shape,
            residual_source.shape,
            (
                (mul1_constant_name, mul1, mul1_constant_index),
                (add1_constant_name, add1, add1_constant_index),
                (str(prelu1.inputs[1]), prelu1, 1),
            ),
        ),
        (
            concat2_nchw_shape,
            post.shape,
            (
                (mul2_constant_name, mul2, mul2_constant_index),
                (add2_constant_name, add2, add2_constant_index),
                (str(prelu2.inputs[1]), prelu2, 1),
            ),
        ),
    )
    for old_shape, target_shape, roles in constant_groups:
        for name, operator, input_index in roles:
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
    constant_plans = _plan_constants(
        model_ir,
        graph_index,
        tuple(constant_roles),
    )
    if constant_plans is None:
        return None

    concat1_inputs = [str(name) for name in concat1.inputs]
    concat1_inputs[int(plain_input_index)] = plain_branch.source_name
    concat1_inputs[int(affine_input_index)] = affine_branch.add_output_name
    add0_inputs = [str(name) for name in add0.inputs]
    add0_inputs[int(residual_input_index)] = residual.source_name
    add0_inputs[int(concat1_input_index)] = concat1_output_name
    concat2_inputs = [str(name) for name in concat2.inputs]
    concat2_inputs[int(skip_input_index)] = skip.source_name
    concat2_inputs[int(prelu1_input_index)] = prelu1_output_name
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
            _metadata_update(name, residual_source.tensor)
            for name in (
                concat1_output_name,
                add0_output_name,
                mul1_output_name,
                add1_output_name,
                prelu1_output_name,
            )
        ),
        *(
            _metadata_update(name, post.tensor)
            for name in (
                concat2_output_name,
                mul2_output_name,
                add2_output_name,
            )
        ),
    )
    legacy_consumers = tuple(
        model_ir.operators[int(index)] for index in legacy_indices
    )
    remove_operators = (
        skip.pre,
        residual.pre,
        plain_branch.pre,
        affine_branch.pre,
        *(adapter for _, adapter in post_adapters),
    )
    return _TailConcatPlan(
        root=root,
        post_adapters=tuple(adapter for _, adapter in post_adapters),
        legacy_consumers=legacy_consumers,
        skip=skip,
        residual=residual,
        plain_branch=plain_branch,
        affine_branch=affine_branch,
        concat1=concat1,
        add0=add0,
        mul1=mul1,
        add1=add1,
        prelu1=prelu1,
        concat2=concat2,
        mul2=mul2,
        add2=add2,
        prelu2=prelu2,
        concat1_output_name=concat1_output_name,
        add0_output_name=add0_output_name,
        mul1_output_name=mul1_output_name,
        add1_output_name=add1_output_name,
        prelu1_output_name=prelu1_output_name,
        concat2_output_name=concat2_output_name,
        mul2_output_name=mul2_output_name,
        add2_output_name=add2_output_name,
        prelu2_output_name=prelu2_output_name,
        post_output_name=post_output_name,
        concat1_inputs=(str(concat1_inputs[0]), str(concat1_inputs[1])),
        add0_inputs=(str(add0_inputs[0]), str(add0_inputs[1])),
        concat2_inputs=(str(concat2_inputs[0]), str(concat2_inputs[1])),
        constant_plans=constant_plans,
        metadata_updates=tuple(metadata_updates),
        input_rewrites=tuple(rewrites),
        remove_operators=tuple(remove_operators),
        insert_legacy_adapter=bool(legacy_consumers),
        legacy_adapter_perm_name=(
            str(skip.pre.inputs[1]) if legacy_consumers else None
        ),
    )


def _plans_equal(
    expected: _TailConcatPlan,
    actual: _TailConcatPlan,
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
        and _matches_equal(expected.skip, actual.skip)
        and _matches_equal(expected.residual, actual.residual)
        and _matches_equal(expected.plain_branch, actual.plain_branch)
        and _matches_equal(expected.affine_branch, actual.affine_branch)
        and all(
            getattr(expected, name) is getattr(actual, name)
            for name in (
                "concat1",
                "add0",
                "mul1",
                "add1",
                "prelu1",
                "concat2",
                "mul2",
                "add2",
                "prelu2",
            )
        )
        and all(
            getattr(expected, name) == getattr(actual, name)
            for name in (
                "concat1_output_name",
                "add0_output_name",
                "mul1_output_name",
                "add1_output_name",
                "prelu1_output_name",
                "concat2_output_name",
                "mul2_output_name",
                "add2_output_name",
                "prelu2_output_name",
                "post_output_name",
                "concat1_inputs",
                "add0_inputs",
                "concat2_inputs",
                "metadata_updates",
                "insert_legacy_adapter",
                "legacy_adapter_perm_name",
            )
        )
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
        and _constant_plans_equal(
            expected.constant_plans,
            actual.constant_plans,
        )
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _TailConcatPlan,
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
        plan.concat1,
        plan.add0,
        plan.concat2,
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
    concat1_index = graph_index.operator_index(plan.concat1)
    add0_index = graph_index.operator_index(plan.add0)
    concat2_index = graph_index.operator_index(plan.concat2)
    prelu2_index = graph_index.operator_index(plan.prelu2)
    assert concat1_index is not None
    assert add0_index is not None
    assert concat2_index is not None
    assert prelu2_index is not None
    graph_index.replace_operator_inputs(
        int(concat1_index),
        plan.concat1_inputs,
    )
    plan.concat1.options["axis"] = 3
    graph_index.replace_operator_inputs(int(add0_index), plan.add0_inputs)
    graph_index.replace_operator_inputs(
        int(concat2_index),
        plan.concat2_inputs,
    )
    plan.concat2.options["axis"] = 3
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


def optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a strict two-Concat SiNet affine tail to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 5,
        "ADD": 4,
        "MUL": 3,
        "PRELU": 2,
        "CONCATENATION": 2,
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
