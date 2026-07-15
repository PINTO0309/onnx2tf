from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _prune_unused_tensors
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
    _constant_plans_equal,
    _late_constant_replacement,
    _metadata_update,
    _plan_constants,
    _plain_prelu,
    _producer,
    _resolved_source,
)


_STATS_KEY = "optimized_sinet_shared_post_prelu_transpose_fanout_chains"


@dataclass(frozen=True)
class _InputMatch:
    pre: OperatorIR
    source_name: str
    nchw_name: str
    concat_backed: bool


@dataclass(frozen=True)
class _SharedPostPlan:
    root: OperatorIR
    inputs: Tuple[_InputMatch, _InputMatch]
    add0: OperatorIR
    mul1: OperatorIR
    add1: OperatorIR
    prelu1: OperatorIR
    post_consumers: Tuple[OperatorIR, ...]
    add0_output_name: str
    mul1_output_name: str
    add1_output_name: str
    prelu1_output_name: str
    post_output_name: str
    add0_inputs: Tuple[str, str]
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]


def _layout_allows(tensor: TensorIR, expected: str) -> bool:
    expected_name = str(expected).upper()
    for value in (tensor.logical_layout, tensor.physical_layout):
        normalized = str(value).strip().upper()
        if normalized not in {"", "UNKNOWN", expected_name}:
            return False
    return True


def _channel_last_concat(operator: OperatorIR) -> bool:
    try:
        axis = int(operator.options.get("axis", -1))
    except (TypeError, ValueError):
        return False
    return bool(
        str(operator.op_type) == "CONCATENATION"
        and len(operator.inputs) >= 2
        and len(operator.outputs) == 1
        and axis == 3
        and str(
            operator.options.get("fusedActivationFunction", "NONE")
        ).upper()
        == "NONE"
    )


def _resolve_input(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    nchw_name: str,
    add0_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_InputMatch]:
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
        int(pre_index) >= int(add0_index)
        or str(nchw_name) in public_names
        or graph_index.consumer_indices(str(nchw_name))
        != [int(add0_index)]
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
    source_producer_index = graph_index.producers.get(source_name)
    source_producer = (
        None
        if source_producer_index is None
        else model_ir.operators[int(source_producer_index)]
    )
    return _InputMatch(
        pre=pre,
        source_name=source_name,
        nchw_name=str(nchw_name),
        concat_backed=bool(
            source_producer is not None
            and _channel_last_concat(source_producer)
        ),
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_SharedPostPlan]:
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
    prelu1_output_name = str(root.inputs[0])
    post_output_name = str(root.outputs[0])
    post_indices = sorted(
        set(graph_index.consumer_indices(post_output_name))
    )
    if len(post_indices) < 2:
        return None
    post_consumers = []
    saw_conv = False
    saw_add = False
    for consumer_index in post_indices:
        if int(consumer_index) <= int(root_index):
            return None
        consumer = model_ir.operators[int(consumer_index)]
        op_type = str(consumer.op_type)
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
            if (
                not consumer.inputs
                or str(consumer.inputs[0]) != post_output_name
                or [str(name) for name in consumer.inputs].count(
                    post_output_name
                )
                != 1
            ):
                return None
            saw_conv = True
        elif op_type == "ADD":
            if (
                not _plain_binary(consumer, "ADD")
                or post_output_name
                not in {str(name) for name in consumer.inputs}
            ):
                return None
            saw_add = True
        else:
            return None
        post_consumers.append(consumer)
    if not saw_conv or not saw_add:
        return None

    prelu1_match = _producer(
        model_ir,
        graph_index,
        prelu1_output_name,
        "PRELU",
    )
    if prelu1_match is None or prelu1_output_name in public_names:
        return None
    prelu1_index, prelu1 = prelu1_match
    if (
        not _plain_prelu(prelu1)
        or int(prelu1_index) >= int(root_index)
        or graph_index.consumer_indices(prelu1_output_name)
        != [int(root_index)]
    ):
        return None

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

    input_matches = []
    for input_name in add0.inputs:
        matched = _resolve_input(
            model_ir,
            graph_index,
            nchw_name=str(input_name),
            add0_index=int(add0_index),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
        if matched is None:
            return None
        input_matches.append(matched)
    if (
        len(input_matches) != 2
        or sum(int(match.concat_backed) for match in input_matches) != 1
    ):
        return None
    inputs = (input_matches[0], input_matches[1])

    source_names = (
        inputs[0].source_name,
        inputs[1].source_name,
    )
    private_names = (
        inputs[0].nchw_name,
        inputs[1].nchw_name,
        add0_output_name,
        mul1_output_name,
        add1_output_name,
        prelu1_output_name,
        post_output_name,
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
    for matched in inputs:
        source = contracts[matched.source_name]
        nchw = contracts[matched.nchw_name]
        assert source is not None and nchw is not None
        if (
            not _layout_allows(source.tensor, "NHWC")
            or not _layout_allows(nchw.tensor, "NCHW")
            or nchw.shape != _permute(source.shape, _NHWC_TO_NCHW)
            or nchw.signature
            != _permute(source.signature, _NHWC_TO_NCHW)
            or source.shape != post.shape
            or source.signature != post.signature
        ):
            return None
    stage_shape = contracts[inputs[0].nchw_name].shape
    stage_signature = contracts[inputs[0].nchw_name].signature
    for name in (
        inputs[1].nchw_name,
        add0_output_name,
        mul1_output_name,
        add1_output_name,
        prelu1_output_name,
    ):
        contract = contracts[name]
        assert contract is not None
        if (
            contract.shape != stage_shape
            or contract.signature != stage_signature
            or not _layout_allows(contract.tensor, "NCHW")
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
            old_nchw_shape=stage_shape,
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
    return _SharedPostPlan(
        root=root,
        inputs=inputs,
        add0=add0,
        mul1=mul1,
        add1=add1,
        prelu1=prelu1,
        post_consumers=tuple(post_consumers),
        add0_output_name=add0_output_name,
        mul1_output_name=mul1_output_name,
        add1_output_name=add1_output_name,
        prelu1_output_name=prelu1_output_name,
        post_output_name=post_output_name,
        add0_inputs=(inputs[0].source_name, inputs[1].source_name),
        constant_plans=constant_plans,
        metadata_updates=tuple(
            _metadata_update(name, post.tensor)
            for name in (
                add0_output_name,
                mul1_output_name,
                add1_output_name,
            )
        ),
        remove_operators=(inputs[0].pre, inputs[1].pre, root),
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
    expected: _SharedPostPlan,
    actual: _SharedPostPlan,
) -> bool:
    return bool(
        expected.root is actual.root
        and all(
            _matches_equal(lhs, rhs)
            for lhs, rhs in zip(expected.inputs, actual.inputs)
        )
        and all(
            getattr(expected, name) is getattr(actual, name)
            for name in ("add0", "mul1", "add1", "prelu1")
        )
        and len(expected.post_consumers) == len(actual.post_consumers)
        and all(
            lhs is rhs
            for lhs, rhs in zip(
                expected.post_consumers,
                actual.post_consumers,
            )
        )
        and expected.add0_output_name == actual.add0_output_name
        and expected.mul1_output_name == actual.mul1_output_name
        and expected.add1_output_name == actual.add1_output_name
        and expected.prelu1_output_name == actual.prelu1_output_name
        and expected.post_output_name == actual.post_output_name
        and expected.add0_inputs == actual.add0_inputs
        and expected.metadata_updates == actual.metadata_updates
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
    plan: _SharedPostPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.root)
    if current is None or not _plans_equal(plan, current):
        return False
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    add0_index = graph_index.operator_index(plan.add0)
    prelu1_index = graph_index.operator_index(plan.prelu1)
    if (
        any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or add0_index is None
        or prelu1_index is None
        or any(
            graph_index.operator_index(operator) is None
            for operator in plan.post_consumers
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
    graph_index.replace_operator_inputs(int(add0_index), plan.add0_inputs)
    graph_index.replace_operator_outputs(
        int(prelu1_index),
        [plan.post_output_name],
    )
    _apply_metadata_updates(model_ir, plan.metadata_updates)
    graph_index.remove_operators([int(index) for index in remove_indices])
    return True


def optimize_sinet_shared_post_prelu_transpose_fanout_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a strict shared-post SiNet affine/PReLU island to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 3,
        "ADD": 3,
        "MUL": 1,
        "PRELU": 1,
        "CONCATENATION": 1,
    }
    has_conv = False
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
            has_conv = True
        if has_conv and all(
            value == 0 for value in required_counts.values()
        ):
            break
    if (
        rewrite_limit == 0
        or not has_conv
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
