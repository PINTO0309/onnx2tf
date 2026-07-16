from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _valid_source,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _TensorContract,
    _constant_vector,
    _producer_is_valid,
    _tensor_contract,
)
from onnx2tf.tflite_builder.passes.decomposed_instance_norm import (
    FLOAT_DTYPES,
    ConstantUpdate,
    DecomposedInstanceNormCore,
    TensorMetadataUpdate,
    apply_constant_update,
    binary_other_input,
    constant_is_private_and_unquantized,
    match_decomposed_instance_norm_core,
    plan_nhwc_instance_norm_constant_updates,
    sole_consumer,
    tensor_contract_exact,
)


_STATS_KEY = (
    "optimized_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains"
)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)


@dataclass(frozen=True)
class _ResidualMulConcatPlan:
    involved_ops: Tuple[OperatorIR, ...]
    pre: OperatorIR
    source_name: str
    core: DecomposedInstanceNormCore
    add_bias: OperatorIR
    inst_output: _TensorContract
    post: OperatorIR
    post_output_name: str
    tail_add: OperatorIR
    tail_add_post_input_index: int
    residual_name: str
    add_output: _TensorContract
    tail_muls: Tuple[OperatorIR, OperatorIR]
    tail_mul_outputs: Tuple[_TensorContract, _TensorContract]
    concat: OperatorIR
    concat_output_name: str
    concat_options: Dict[str, object]
    tail_post: OperatorIR
    tail_post_output_name: str
    constant_updates: Tuple[ConstantUpdate, ...]
    metadata_updates: Tuple[TensorMetadataUpdate, ...]
    channel_last_names: Tuple[str, ...]


def _valid_permutation_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    expected: Tuple[int, ...],
    public_inputs: set[str],
    public_names: set[str],
) -> bool:
    return bool(
        _constant_vector(
            model_ir,
            graph_index,
            str(name),
            len(expected),
            public_inputs,
        )
        == expected
        and constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            str(name),
            public_names,
        )
    )


def _permuted(values: Tuple[int, ...], permutation: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(int(values[index]) for index in permutation)


def _concat_contract(
    model_ir: ModelIR,
    name: str,
    left: _TensorContract,
    right: _TensorContract,
) -> Optional[_TensorContract]:
    if any(
        int(left.shape[index]) != int(right.shape[index])
        or int(left.signature[index]) != int(right.signature[index])
        for index in (0, 2, 3)
    ):
        return None
    shape = list(left.shape)
    shape[1] = int(left.shape[1]) + int(right.shape[1])
    signature = list(left.signature)
    signature[1] = (
        -1
        if int(left.signature[1]) == -1 or int(right.signature[1]) == -1
        else int(left.signature[1]) + int(right.signature[1])
    )
    return tensor_contract_exact(
        model_ir,
        str(name),
        4,
        tuple(shape),
        tuple(signature),
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_ResidualMulConcatPlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    public_names = public_inputs | public_outputs
    pre = model_ir.operators[int(pre_index)]
    if (
        str(pre.op_type) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or not _valid_permutation_constant(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            _PERM_NHWC_TO_NCHW,
            public_inputs,
            public_names,
        )
    ):
        return None

    source_name = str(pre.inputs[0])
    x_name = str(pre.outputs[0])
    source = _tensor_contract(model_ir, source_name, 4)
    if source is None:
        return None
    x = tensor_contract_exact(
        model_ir,
        x_name,
        4,
        _permuted(source.shape, _PERM_NHWC_TO_NCHW),
        _permuted(source.signature, _PERM_NHWC_TO_NCHW),
    )
    if (
        x is None
        or source_name in public_outputs
        or x_name in public_names
        or not _valid_source(
            graph_index,
            source,
            source_name,
            int(pre_index),
            public_inputs,
        )
        or not _producer_is_valid(graph_index, x_name, int(pre_index))
    ):
        return None

    core = match_decomposed_instance_norm_core(
        model_ir,
        graph_index,
        x_name=x_name,
        x=x,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
        allow_commuted_sub=True,
    )
    if core is None:
        return None

    bias_match = sole_consumer(graph_index, str(core.scaled.tensor.name))
    if bias_match is None:
        return None
    bias_index, add_bias = bias_match
    bias_constant_match = binary_other_input(
        add_bias,
        str(core.scaled.tensor.name),
    )
    inst_output_name = str(add_bias.outputs[0]) if len(add_bias.outputs) == 1 else ""
    inst_output = tensor_contract_exact(
        model_ir,
        inst_output_name,
        4,
        x.shape,
        x.signature,
    )
    if (
        str(add_bias.op_type) != "ADD"
        or bias_constant_match is None
        or len(add_bias.outputs) != 1
        or inst_output is None
        or inst_output_name in public_names
        or not _producer_is_valid(graph_index, inst_output_name, bias_index)
    ):
        return None
    bias_name, bias_input_index = bias_constant_match

    post_match = sole_consumer(graph_index, inst_output_name)
    if post_match is None:
        return None
    post_index, post = post_match
    post_output_name = str(post.outputs[0]) if len(post.outputs) == 1 else ""
    post_output = tensor_contract_exact(
        model_ir,
        post_output_name,
        4,
        source.shape,
        source.signature,
    )
    if (
        str(post.op_type) != "TRANSPOSE"
        or len(post.inputs) != 2
        or len(post.outputs) != 1
        or str(post.inputs[0]) != inst_output_name
        or not _valid_permutation_constant(
            model_ir,
            graph_index,
            str(post.inputs[1]),
            _PERM_NCHW_TO_NHWC,
            public_inputs,
            public_names,
        )
        or post_output is None
        or post_output_name in public_names
        or not _producer_is_valid(graph_index, post_output_name, post_index)
    ):
        return None

    tail_add_match = sole_consumer(graph_index, post_output_name)
    if tail_add_match is None:
        return None
    tail_add_index, tail_add = tail_add_match
    residual_match = binary_other_input(tail_add, post_output_name)
    add_output_name = str(tail_add.outputs[0]) if len(tail_add.outputs) == 1 else ""
    add_output = tensor_contract_exact(
        model_ir,
        add_output_name,
        4,
        source.shape,
        source.signature,
    )
    if (
        str(tail_add.op_type) != "ADD"
        or residual_match is None
        or len(tail_add.outputs) != 1
        or add_output is None
        or add_output_name in public_names
        or not _producer_is_valid(graph_index, add_output_name, tail_add_index)
    ):
        return None
    residual_name, tail_add_residual_input_index = residual_match
    residual = tensor_contract_exact(
        model_ir,
        residual_name,
        4,
        source.shape,
        source.signature,
    )
    if residual is None or not _valid_source(
        graph_index,
        residual,
        residual_name,
        tail_add_index,
        public_inputs,
    ):
        return None

    tail_mul_indices = graph_index.consumer_indices(add_output_name)
    if len(tail_mul_indices) != 2 or len(set(tail_mul_indices)) != 2:
        return None
    tail_muls = tuple(
        model_ir.operators[index] for index in sorted(tail_mul_indices)
    )
    tail_mul_outputs = []
    tail_coefficients = []
    concat_index: Optional[int] = None
    for tail_mul_index, tail_mul in zip(sorted(tail_mul_indices), tail_muls):
        coefficient_match = binary_other_input(tail_mul, add_output_name)
        output_name = str(tail_mul.outputs[0]) if len(tail_mul.outputs) == 1 else ""
        output = tensor_contract_exact(
            model_ir,
            output_name,
            4,
            x.shape,
            x.signature,
        )
        if (
            str(tail_mul.op_type) != "MUL"
            or coefficient_match is None
            or len(tail_mul.outputs) != 1
            or output is None
            or output_name in public_names
            or not _producer_is_valid(graph_index, output_name, tail_mul_index)
        ):
            return None
        concat_match = sole_consumer(graph_index, output_name)
        if concat_match is None:
            return None
        current_concat_index, current_concat = concat_match
        if str(current_concat.op_type) != "CONCATENATION":
            return None
        if concat_index is None:
            concat_index = int(current_concat_index)
        elif int(concat_index) != int(current_concat_index):
            return None
        tail_mul_outputs.append(output)
        tail_coefficients.append(
            (coefficient_match[0], tail_mul, coefficient_match[1])
        )
    if concat_index is None:
        return None

    concat = model_ir.operators[int(concat_index)]
    concat_inputs = [str(value) for value in concat.inputs]
    tail_output_names = [
        str(contract.tensor.name) for contract in tail_mul_outputs
    ]
    concat_options = dict(concat.options) if isinstance(concat.options, dict) else {}
    try:
        concat_axis = int(concat_options.get("axis", 1))
    except (TypeError, ValueError):
        return None
    if concat_axis < 0:
        concat_axis += 4
    concat_output_name = str(concat.outputs[0]) if len(concat.outputs) == 1 else ""
    concat_output = _concat_contract(
        model_ir,
        concat_output_name,
        tail_mul_outputs[0],
        tail_mul_outputs[1],
    )
    if (
        len(concat.inputs) != 2
        or len(concat.outputs) != 1
        or Counter(concat_inputs) != Counter(tail_output_names)
        or concat_axis != 1
        or concat_output is None
        or concat_output_name in public_names
        or not _producer_is_valid(graph_index, concat_output_name, concat_index)
    ):
        return None

    tail_post_match = sole_consumer(graph_index, concat_output_name)
    if tail_post_match is None:
        return None
    tail_post_index, tail_post = tail_post_match
    tail_post_output_name = (
        str(tail_post.outputs[0]) if len(tail_post.outputs) == 1 else ""
    )
    tail_post_output = tensor_contract_exact(
        model_ir,
        tail_post_output_name,
        4,
        _permuted(concat_output.shape, _PERM_NCHW_TO_NHWC),
        _permuted(concat_output.signature, _PERM_NCHW_TO_NHWC),
    )
    if (
        str(tail_post.op_type) != "TRANSPOSE"
        or len(tail_post.inputs) != 2
        or len(tail_post.outputs) != 1
        or str(tail_post.inputs[0]) != concat_output_name
        or not _valid_permutation_constant(
            model_ir,
            graph_index,
            str(tail_post.inputs[1]),
            _PERM_NCHW_TO_NHWC,
            public_inputs,
            public_names,
        )
        or tail_post_output is None
        or tail_post_output_name in public_inputs
        or not _producer_is_valid(
            graph_index,
            tail_post_output_name,
            tail_post_index,
        )
        or any(
            int(index) <= int(tail_post_index)
            for index in graph_index.consumer_indices(tail_post_output_name)
        )
    ):
        return None

    involved_ops = (
        pre,
        *core.ordered_ops,
        add_bias,
        post,
        tail_add,
        *tail_muls,
        concat,
        tail_post,
    )
    involved_indices = [graph_index.operator_index(op) for op in involved_ops]
    dtype = str(source.tensor.dtype)
    contracts = (
        source,
        x,
        core.mean1_contract,
        core.centered,
        core.squared,
        core.mean2_contract,
        core.add_epsilon_contract,
        core.sqrt_contract,
        core.div_contract,
        core.normalized,
        core.scaled,
        inst_output,
        post_output,
        residual,
        add_output,
        *tail_mul_outputs,
        concat_output,
        tail_post_output,
    )
    if (
        any(index is None for index in involved_indices)
        or [int(index) for index in involved_indices if index is not None]
        != sorted(int(index) for index in involved_indices if index is not None)
        or len({id(op) for op in involved_ops}) != len(involved_ops)
        or dtype not in FLOAT_DTYPES
        or any(str(contract.tensor.dtype) != dtype for contract in contracts)
        or any(contract.tensor.quantization is not None for contract in contracts)
    ):
        return None

    constant_updates = plan_nhwc_instance_norm_constant_updates(
        model_ir,
        graph_index,
        core=core,
        bias_name=bias_name,
        bias_operator=add_bias,
        bias_input_index=bias_input_index,
        channel_count=int(x.shape[1]),
        public_names=public_names,
        additional_coefficient_uses=tuple(tail_coefficients),
    )
    if constant_updates is None:
        return None

    full_contracts = (
        core.centered,
        core.squared,
        core.normalized,
        core.scaled,
        inst_output,
        *tail_mul_outputs,
        concat_output,
    )
    reduced_contracts = (
        core.mean1_contract,
        core.mean2_contract,
        core.add_epsilon_contract,
        core.sqrt_contract,
        core.div_contract,
    )
    metadata_updates = tuple(
        TensorMetadataUpdate(
            contract,
            _permuted(contract.shape, _PERM_NCHW_TO_NHWC),
            _permuted(contract.signature, _PERM_NCHW_TO_NHWC),
        )
        for contract in full_contracts + reduced_contracts
    )
    concat_options["axis"] = 3
    channel_last_names = tuple(
        dict.fromkeys(
            [
                source_name,
                residual_name,
                add_output_name,
                tail_post_output_name,
                *(contract.tensor.name for contract in full_contracts),
                *(contract.tensor.name for contract in reduced_contracts),
            ]
        )
    )
    return _ResidualMulConcatPlan(
        involved_ops=involved_ops,
        pre=pre,
        source_name=source_name,
        core=core,
        add_bias=add_bias,
        inst_output=inst_output,
        post=post,
        post_output_name=post_output_name,
        tail_add=tail_add,
        tail_add_post_input_index=1 - int(tail_add_residual_input_index),
        residual_name=residual_name,
        add_output=add_output,
        tail_muls=(tail_muls[0], tail_muls[1]),
        tail_mul_outputs=(tail_mul_outputs[0], tail_mul_outputs[1]),
        concat=concat,
        concat_output_name=concat_output_name,
        concat_options=concat_options,
        tail_post=tail_post,
        tail_post_output_name=tail_post_output_name,
        constant_updates=constant_updates,
        metadata_updates=metadata_updates,
        channel_last_names=channel_last_names,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _ResidualMulConcatPlan,
) -> bool:
    involved_indices = [graph_index.operator_index(op) for op in plan.involved_ops]
    remove_ops = (plan.pre, plan.post, plan.tail_post)
    remove_indices = [graph_index.operator_index(op) for op in remove_ops]
    clone_names = [
        update.clone_name
        for update in plan.constant_updates
        if update.clone_name is not None
    ]
    if (
        any(index is None for index in involved_indices)
        or len({int(index) for index in involved_indices if index is not None})
        != len(involved_indices)
        or any(index is None for index in remove_indices)
        or len(clone_names) != len(set(clone_names))
        or any(name in model_ir.tensors for name in clone_names)
        or any(
            update.clone_name is not None and update.clone is None
            for update in plan.constant_updates
        )
        or int(plan.tail_add_post_input_index) < 0
        or int(plan.tail_add_post_input_index) >= len(plan.tail_add.inputs)
        or str(plan.tail_add.inputs[int(plan.tail_add_post_input_index)])
        != plan.post_output_name
        or str(plan.concat.outputs[0]) != plan.concat_output_name
        or str(plan.tail_post.outputs[0]) != plan.tail_post_output_name
    ):
        return False

    for update in plan.constant_updates:
        if not apply_constant_update(model_ir, graph_index, update):
            return False
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.core.mean1,
        input_index=0,
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.core.sub,
        input_index=plan.core.sub_x_input_index,
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.tail_add,
        input_index=plan.tail_add_post_input_index,
        new_input_name=str(plan.inst_output.tensor.name),
        graph_index=graph_index,
    )
    for update in plan.metadata_updates:
        update.contract.tensor.shape = list(update.shape)
        update.contract.tensor.shape_signature = list(update.signature)
    plan.concat.options = dict(plan.concat_options)
    graph_index.remove_operators(
        [int(index) for index in remove_indices if index is not None]
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.concat,
        new_outputs=[plan.tail_post_output_name],
        graph_index=graph_index,
    )
    hints = {
        str(value)
        for value in model_ir.metadata.get(
            "assume_channel_last_layout_tensor_names",
            [],
        )
        if str(value)
    }
    hints.update(plan.channel_last_names)
    model_ir.metadata["assume_channel_last_layout_tensor_names"] = sorted(hints)
    return True


def optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Lift a strict decomposed-InstanceNorm residual-MUL/CONCAT tail to NHWC."""

    if candidate is None:
        counts = Counter(str(operator.op_type) for operator in model_ir.operators)
        required = {
            "TRANSPOSE": 3,
            "MEAN": 2,
            "SUB": 1,
            "MUL": 5,
            "ADD": 3,
            "SQRT": 1,
            "DIV": 1,
            "CONCATENATION": 1,
        }
        if any(counts[op_type] < minimum for op_type, minimum in required.items()):
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
    for pre in candidates:
        if rewritten >= max(0, int(max_rewrites)):
            break
        pre_index = active_index.operator_index(pre)
        if pre_index is None:
            continue
        plan = _resolve_candidate(model_ir, active_index, pre_index)
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1
    if rewritten:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
