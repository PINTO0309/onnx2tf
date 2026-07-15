from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _replace_operator_input_at,
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
    ConstantUse,
    DecomposedInstanceNormCore,
    TensorMetadataUpdate,
    apply_constant_update,
    binary_other_input,
    float_constant,
    match_decomposed_instance_norm_core,
    plan_constant_update,
    sole_consumer,
    tensor_contract_exact,
)


_STATS_KEY = "optimized_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)


@dataclass(frozen=True)
class _PostBiasPlan:
    ordered_ops: Tuple[OperatorIR, ...]
    pre: OperatorIR
    source_name: str
    core: DecomposedInstanceNormCore
    post: OperatorIR
    post_output_name: str
    add_bias: OperatorIR
    add_bias_data_input_index: int
    constant_updates: Tuple[ConstantUpdate, ...]
    metadata_updates: Tuple[TensorMetadataUpdate, ...]
    channel_last_names: Tuple[str, ...]


def _constant_is_private_and_unquantized(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    public_names: set[str],
) -> bool:
    tensor = model_ir.tensors.get(str(name))
    return bool(
        tensor is not None
        and tensor.data is not None
        and str(name) not in public_names
        and str(name) not in graph_index.producers
        and str(name) not in graph_index.duplicate_producers
        and tensor.quantization is None
    )


def _coefficient_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    dtype: str,
    channel_count: int,
) -> Optional[np.ndarray]:
    data = float_constant(model_ir, graph_index, name, dtype)
    if data is None:
        return None
    shape = tuple(int(value) for value in data.shape)
    if data.size == 1 or shape == (1, 1, 1, int(channel_count)):
        return np.asarray(data)
    if shape != (1, int(channel_count), 1, 1):
        return None
    return np.transpose(data, _PERM_NCHW_TO_NHWC)


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_PostBiasPlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    public_names = public_inputs | public_outputs
    pre = model_ir.operators[int(pre_index)]
    if (
        str(pre.op_type) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or _constant_vector(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NHWC_TO_NCHW
        or not _constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            public_names,
        )
    ):
        return None

    source_name = str(pre.inputs[0])
    pre_output_name = str(pre.outputs[0])
    source = _tensor_contract(model_ir, source_name, 4)
    if source is None:
        return None
    pre_output = tensor_contract_exact(
        model_ir,
        pre_output_name,
        4,
        tuple(source.shape[index] for index in _PERM_NHWC_TO_NCHW),
        tuple(source.signature[index] for index in _PERM_NHWC_TO_NCHW),
    )
    if (
        pre_output is None
        or source_name in public_outputs
        or pre_output_name in public_names
        or not _valid_source(
            graph_index,
            source,
            source_name,
            int(pre_index),
            public_inputs,
        )
        or not _producer_is_valid(graph_index, pre_output_name, int(pre_index))
    ):
        return None

    core = match_decomposed_instance_norm_core(
        model_ir,
        graph_index,
        x_name=pre_output_name,
        x=pre_output,
        public_inputs=public_inputs,
        public_outputs=public_outputs,
        allow_commuted_sub=True,
    )
    if core is None:
        return None

    post_match = sole_consumer(graph_index, str(core.scaled.tensor.name))
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
        or str(post.inputs[0]) != str(core.scaled.tensor.name)
        or _constant_vector(
            model_ir,
            graph_index,
            str(post.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NCHW_TO_NHWC
        or not _constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            str(post.inputs[1]),
            public_names,
        )
        or post_output is None
        or post_output_name in public_names
        or not _producer_is_valid(graph_index, post_output_name, post_index)
    ):
        return None

    bias_match = sole_consumer(graph_index, post_output_name)
    if bias_match is None:
        return None
    bias_index, add_bias = bias_match
    bias_constant_match = binary_other_input(add_bias, post_output_name)
    output_name = str(add_bias.outputs[0]) if len(add_bias.outputs) == 1 else ""
    output = tensor_contract_exact(
        model_ir,
        output_name,
        4,
        source.shape,
        source.signature,
    )
    if (
        str(add_bias.op_type) != "ADD"
        or bias_constant_match is None
        or len(add_bias.outputs) != 1
        or output is None
        or output_name in public_inputs
        or not _producer_is_valid(graph_index, output_name, bias_index)
        or any(
            int(consumer_index) <= int(bias_index)
            for consumer_index in graph_index.consumer_indices(output_name)
        )
    ):
        return None
    bias_name, bias_input_index = bias_constant_match

    ordered_ops = (pre, *core.ordered_ops, post, add_bias)
    ordered_indices = [graph_index.operator_index(op) for op in ordered_ops]
    dtype = str(source.tensor.dtype)
    contracts: Tuple[_TensorContract, ...] = (
        source,
        pre_output,
        core.mean1_contract,
        core.centered,
        core.squared,
        core.mean2_contract,
        core.add_epsilon_contract,
        core.sqrt_contract,
        core.div_contract,
        core.normalized,
        core.scaled,
        post_output,
        output,
    )
    if (
        any(index is None for index in ordered_indices)
        or [int(index) for index in ordered_indices if index is not None]
        != sorted(int(index) for index in ordered_indices if index is not None)
        or len({id(op) for op in ordered_ops}) != len(ordered_ops)
        or dtype not in FLOAT_DTYPES
        or any(str(contract.tensor.dtype) != dtype for contract in contracts)
        or any(contract.tensor.quantization is not None for contract in contracts)
    ):
        return None

    axes_uses: Dict[str, list[ConstantUse]] = {}
    axes_uses.setdefault(core.mean1_axes_name, []).append(ConstantUse(core.mean1, 1))
    axes_uses.setdefault(core.mean2_axes_name, []).append(ConstantUse(core.mean2, 1))
    constant_updates = []
    for axes_name, uses in axes_uses.items():
        if not _constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            axes_name,
            public_names,
        ):
            return None
        update = plan_constant_update(
            model_ir,
            graph_index,
            axes_name,
            np.asarray(
                [1, 2],
                dtype=np.asarray(model_ir.tensors[axes_name].data).dtype,
            ),
            tuple(uses),
            "nhwc_axes",
            public_names,
        )
        if update is None:
            return None
        constant_updates.append(update)

    coefficient_uses: Dict[str, list[ConstantUse]] = {}
    coefficient_uses.setdefault(core.scale_name, []).append(
        ConstantUse(core.scale, core.scale_input_index)
    )
    coefficient_uses.setdefault(bias_name, []).append(
        ConstantUse(add_bias, bias_input_index)
    )
    channel_count = int(pre_output.shape[1])
    for coefficient_name, uses in coefficient_uses.items():
        if not _constant_is_private_and_unquantized(
            model_ir,
            graph_index,
            coefficient_name,
            public_names,
        ):
            return None
        replacement = _coefficient_replacement(
            model_ir,
            graph_index,
            name=coefficient_name,
            dtype=dtype,
            channel_count=channel_count,
        )
        if replacement is None:
            return None
        current = np.asarray(model_ir.tensors[coefficient_name].data)
        if current.shape == replacement.shape and np.array_equal(
            current,
            replacement,
        ):
            continue
        update = plan_constant_update(
            model_ir,
            graph_index,
            coefficient_name,
            replacement,
            tuple(uses),
            "nhwc_coefficient",
            public_names,
        )
        if update is None:
            return None
        constant_updates.append(update)

    full_contracts = (
        core.centered,
        core.squared,
        core.normalized,
        core.scaled,
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
            tuple(contract.shape[index] for index in _PERM_NCHW_TO_NHWC),
            tuple(contract.signature[index] for index in _PERM_NCHW_TO_NHWC),
        )
        for contract in full_contracts + reduced_contracts
    )
    channel_last_names = tuple(
        dict.fromkeys(
            [
                source_name,
                *(contract.tensor.name for contract in full_contracts),
                *(contract.tensor.name for contract in reduced_contracts),
                output_name,
            ]
        )
    )
    return _PostBiasPlan(
        ordered_ops=ordered_ops,
        pre=pre,
        source_name=source_name,
        core=core,
        post=post,
        post_output_name=post_output_name,
        add_bias=add_bias,
        add_bias_data_input_index=1 - int(bias_input_index),
        constant_updates=tuple(constant_updates),
        metadata_updates=metadata_updates,
        channel_last_names=channel_last_names,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _PostBiasPlan,
) -> bool:
    indices = [graph_index.operator_index(op) for op in plan.ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    clone_names = [
        update.clone_name
        for update in plan.constant_updates
        if update.clone_name is not None
    ]
    if (
        resolved != sorted(resolved)
        or len(set(resolved)) != len(resolved)
        or len(clone_names) != len(set(clone_names))
        or any(name in model_ir.tensors for name in clone_names)
        or any(
            update.clone_name is not None and update.clone is None
            for update in plan.constant_updates
        )
        or int(plan.add_bias_data_input_index) < 0
        or int(plan.add_bias_data_input_index) >= len(plan.add_bias.inputs)
        or str(plan.add_bias.inputs[int(plan.add_bias_data_input_index)])
        != plan.post_output_name
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
        op=plan.add_bias,
        input_index=plan.add_bias_data_input_index,
        new_input_name=str(plan.core.scaled.tensor.name),
        graph_index=graph_index,
    )
    for update in plan.metadata_updates:
        update.contract.tensor.shape = list(update.shape)
        update.contract.tensor.shape_signature = list(update.signature)
    remove_indices = [
        graph_index.operator_index(plan.pre),
        graph_index.operator_index(plan.post),
    ]
    if any(index is None for index in remove_indices):
        return False
    graph_index.remove_operators(
        [int(index) for index in remove_indices if index is not None]
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


def optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Eliminate a strict NCHW InstanceNorm bridge before NHWC bias ADD."""

    if candidate is None:
        counts = Counter(str(operator.op_type) for operator in model_ir.operators)
        required = {
            "TRANSPOSE": 2,
            "MEAN": 2,
            "SUB": 1,
            "MUL": 3,
            "ADD": 2,
            "SQRT": 1,
            "DIV": 1,
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
