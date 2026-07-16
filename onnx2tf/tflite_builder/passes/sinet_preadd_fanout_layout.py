from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _prune_unused_tensors
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    _FLOAT_DTYPES,
    _NCHW_TO_NHWC,
    _NHWC_TO_NCHW,
    _permute,
    _tensor_contract,
    _typed_permutation,
)
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    _ConstantPlan,
    _LateAffineTailMatch,
    _MetadataUpdate,
    _apply_constant_plans,
    _apply_metadata_updates,
    _constant_plans_equal,
    _late_constant_replacement,
    _metadata_update,
    _plan_constants,
    _resolve_late_affine_tail,
    _resolved_source,
)


_STATS_KEY = (
    "optimized_sinet_deep_skip_pre_add_concat_prelu_fanout_chains"
)


@dataclass(frozen=True)
class _PreAddFanoutPlan:
    root: OperatorIR
    tail: _LateAffineTailMatch
    concat_pre: OperatorIR
    sibling_post: OperatorIR
    concat_source_name: str
    direct_source_name: str
    sibling_output_name: str
    add0_inputs: Tuple[str, str]
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    remove_operators: Tuple[OperatorIR, ...]


def _channel_last(tensor: object) -> bool:
    return bool(
        str(getattr(tensor, "logical_layout", "")).upper() == "NHWC"
        or str(getattr(tensor, "physical_layout", "")).upper() == "NHWC"
    )


def _tail_equal(
    expected: _LateAffineTailMatch,
    actual: _LateAffineTailMatch,
) -> bool:
    operator_fields = (
        "root",
        "downstream",
        "prelu1",
        "add1",
        "mul1",
        "add0",
    )
    value_fields = (
        "prelu1_output_name",
        "post1_output_name",
        "add1_output_name",
        "mul1_output_name",
        "add0_output_name",
        "mul1_constant_name",
        "mul1_constant_index",
        "add1_constant_name",
        "add1_constant_index",
    )
    return bool(
        all(
            getattr(expected, field) is getattr(actual, field)
            for field in operator_fields
        )
        and all(
            getattr(expected, field) == getattr(actual, field)
            for field in value_fields
        )
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_PreAddFanoutPlan]:
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    tail = _resolve_late_affine_tail(
        model_ir,
        graph_index,
        root,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if tail is None:
        return None
    add0_index = graph_index.operator_index(tail.add0)
    if add0_index is None:
        return None

    concat_roles = []
    direct_roles = []
    for input_index, input_name in enumerate(tail.add0.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        producer = (
            None
            if producer_index is None
            else model_ir.operators[int(producer_index)]
        )
        if (
            producer is not None
            and str(producer.op_type) == "TRANSPOSE"
            and _typed_permutation(
                model_ir,
                graph_index,
                producer,
                _NHWC_TO_NCHW,
                public_names,
            )
        ):
            concat_source_name = str(producer.inputs[0])
            source_producer_index = graph_index.producers.get(
                concat_source_name
            )
            source_producer = (
                None
                if source_producer_index is None
                else model_ir.operators[int(source_producer_index)]
            )
            try:
                concat_axis = int(
                    source_producer.options.get("axis", -1)
                    if source_producer is not None
                    else -1
                )
            except (TypeError, ValueError):
                concat_axis = -1
            if (
                source_producer is not None
                and str(source_producer.op_type) == "CONCATENATION"
                and concat_axis == 3
            ):
                concat_roles.append(
                    (
                        int(input_index),
                        int(producer_index),
                        producer,
                        concat_source_name,
                    )
                )
                continue

        source_name = str(input_name)
        sibling_matches = []
        for consumer_index in sorted(
            set(graph_index.consumer_indices(source_name))
        ):
            if int(consumer_index) == int(add0_index):
                continue
            consumer = model_ir.operators[int(consumer_index)]
            if (
                int(consumer_index) < int(add0_index)
                and str(consumer.op_type) == "TRANSPOSE"
                and len(consumer.inputs) == 2
                and len(consumer.outputs) == 1
                and str(consumer.inputs[0]) == source_name
                and str(consumer.outputs[0]) not in public_names
                and _typed_permutation(
                    model_ir,
                    graph_index,
                    consumer,
                    _NCHW_TO_NHWC,
                    public_names,
                )
            ):
                sibling_matches.append((int(consumer_index), consumer))
        if len(sibling_matches) == 1:
            sibling_index, sibling = sibling_matches[0]
            if Counter(graph_index.consumer_indices(source_name)) == Counter(
                (int(add0_index), int(sibling_index))
            ):
                direct_roles.append(
                    (
                        int(input_index),
                        sibling_index,
                        sibling,
                        source_name,
                        str(sibling.outputs[0]),
                    )
                )
    if len(concat_roles) != 1 or len(direct_roles) != 1:
        return None
    (
        concat_input_index,
        concat_pre_index,
        concat_pre,
        concat_source_name,
    ) = concat_roles[0]
    (
        direct_input_index,
        sibling_post_index,
        sibling_post,
        direct_source_name,
        sibling_output_name,
    ) = direct_roles[0]
    concat_pre_output_name = str(concat_pre.outputs[0])
    if (
        concat_input_index == direct_input_index
        or int(concat_pre_index) >= int(add0_index)
        or int(sibling_post_index) >= int(add0_index)
        or graph_index.consumer_indices(concat_pre_output_name)
        != [int(add0_index)]
        or concat_pre_output_name in public_names
        or not _resolved_source(
            graph_index,
            name=concat_source_name,
            adapter_index=int(concat_pre_index),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
        or not _resolved_source(
            graph_index,
            name=direct_source_name,
            adapter_index=int(sibling_post_index),
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
    ):
        return None

    tensor_names = (
        concat_source_name,
        concat_pre_output_name,
        direct_source_name,
        sibling_output_name,
        tail.add0_output_name,
        tail.mul1_output_name,
        tail.add1_output_name,
        tail.prelu1_output_name,
        tail.post1_output_name,
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
    post = contracts[tail.post1_output_name]
    concat_source = contracts[concat_source_name]
    concat_pre_output = contracts[concat_pre_output_name]
    direct_source = contracts[direct_source_name]
    sibling_output = contracts[sibling_output_name]
    assert all(
        value is not None
        for value in (
            post,
            concat_source,
            concat_pre_output,
            direct_source,
            sibling_output,
        )
    )
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
        or not _channel_last(post.tensor)
        or not _channel_last(concat_source.tensor)
        or not _channel_last(sibling_output.tensor)
        or concat_source.shape != post.shape
        or concat_source.signature != post.signature
        or sibling_output.shape != post.shape
        or sibling_output.signature != post.signature
        or concat_pre_output.shape
        != _permute(concat_source.shape, _NHWC_TO_NCHW)
        or concat_pre_output.signature
        != _permute(concat_source.signature, _NHWC_TO_NCHW)
        or direct_source.shape
        != _permute(sibling_output.shape, _NHWC_TO_NCHW)
        or direct_source.signature
        != _permute(sibling_output.signature, _NHWC_TO_NCHW)
    ):
        return None
    stage_nchw_shape = concat_pre_output.shape
    stage_nchw_signature = concat_pre_output.signature
    for name in (
        tail.add0_output_name,
        tail.mul1_output_name,
        tail.add1_output_name,
        tail.prelu1_output_name,
    ):
        if (
            contracts[name].shape != stage_nchw_shape
            or contracts[name].signature != stage_nchw_signature
        ):
            return None

    constant_roles = []
    for name, operator, input_index in (
        (
            tail.mul1_constant_name,
            tail.mul1,
            tail.mul1_constant_index,
        ),
        (
            tail.add1_constant_name,
            tail.add1,
            tail.add1_constant_index,
        ),
        (str(tail.prelu1.inputs[1]), tail.prelu1, 1),
    ):
        replacement = _late_constant_replacement(
            model_ir,
            graph_index,
            name=str(name),
            dtype=dtype,
            old_nchw_shape=stage_nchw_shape,
            target_nhwc_shape=post.shape,
            public_names=public_names,
        )
        if replacement is None:
            return None
        constant_roles.append(
            (str(name), replacement, operator, int(input_index))
        )
    root_perm_name = str(root.inputs[1])
    root_perm = model_ir.tensors[root_perm_name]
    constant_roles.append(
        (
            root_perm_name,
            np.asarray(
                _NHWC_TO_NCHW,
                dtype=np.asarray(root_perm.data).dtype,
            ),
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

    add0_inputs = [str(name) for name in tail.add0.inputs]
    add0_inputs[int(concat_input_index)] = concat_source_name
    add0_inputs[int(direct_input_index)] = sibling_output_name
    return _PreAddFanoutPlan(
        root=root,
        tail=tail,
        concat_pre=concat_pre,
        sibling_post=sibling_post,
        concat_source_name=concat_source_name,
        direct_source_name=direct_source_name,
        sibling_output_name=sibling_output_name,
        add0_inputs=(str(add0_inputs[0]), str(add0_inputs[1])),
        constant_plans=constant_plans,
        metadata_updates=tuple(
            _metadata_update(name, post.tensor)
            for name in (
                tail.add0_output_name,
                tail.mul1_output_name,
                tail.add1_output_name,
            )
        ),
        remove_operators=(concat_pre,),
    )


def _plans_equal(
    expected: _PreAddFanoutPlan,
    actual: _PreAddFanoutPlan,
) -> bool:
    return bool(
        expected.root is actual.root
        and _tail_equal(expected.tail, actual.tail)
        and expected.concat_pre is actual.concat_pre
        and expected.sibling_post is actual.sibling_post
        and expected.concat_source_name == actual.concat_source_name
        and expected.direct_source_name == actual.direct_source_name
        and expected.sibling_output_name == actual.sibling_output_name
        and expected.add0_inputs == actual.add0_inputs
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
    plan: _PreAddFanoutPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.root)
    if current is None or not _plans_equal(plan, current):
        return False
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    add0_index = graph_index.operator_index(plan.tail.add0)
    prelu1_index = graph_index.operator_index(plan.tail.prelu1)
    root_index = graph_index.operator_index(plan.root)
    if (
        any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or any(
            index is None for index in (add0_index, prelu1_index, root_index)
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
        int(root_index),
        [plan.tail.prelu1_output_name],
    )
    graph_index.replace_operator_outputs(
        int(prelu1_index),
        [plan.tail.post1_output_name],
    )
    graph_index.replace_operator_inputs(
        int(root_index),
        [plan.tail.post1_output_name, str(plan.root.inputs[1])],
    )
    _apply_metadata_updates(model_ir, plan.metadata_updates)
    graph_index.remove_operators([int(index) for index in remove_indices])
    return True


def optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a strict pre-ADD Concat/PReLU fan-out island to NHWC."""

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
