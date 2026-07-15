from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _valid_source,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _TensorContract,
    _constant_vector,
    _producer_is_valid,
    _tensor_contract,
    _unique_tensor_name,
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
    "optimized_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains"
)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_ADAPTER_PERM_NAME = "__instancenorm_residual_add_nhwc_to_nchw_perm_rank4__"


@dataclass(frozen=True)
class _ResidualAddPlan:
    involved_ops: Tuple[OperatorIR, ...]
    main_pre: OperatorIR
    main_source_name: str
    residual_pre: OperatorIR
    residual_source_name: str
    core: DecomposedInstanceNormCore
    add_bias: OperatorIR
    inst_output: _TensorContract
    tail_add: OperatorIR
    tail_add_residual_input_index: int
    old_add_output_name: str
    new_add_output: TensorIR
    adapter_permutation: Optional[TensorIR]
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


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_ResidualAddPlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    public_names = public_inputs | public_outputs
    main_pre = model_ir.operators[int(pre_index)]
    if (
        str(main_pre.op_type) != "TRANSPOSE"
        or len(main_pre.inputs) != 2
        or len(main_pre.outputs) != 1
        or not _valid_permutation_constant(
            model_ir,
            graph_index,
            str(main_pre.inputs[1]),
            _PERM_NHWC_TO_NCHW,
            public_inputs,
            public_names,
        )
    ):
        return None

    main_source_name = str(main_pre.inputs[0])
    main_pre_output_name = str(main_pre.outputs[0])
    main_source = _tensor_contract(model_ir, main_source_name, 4)
    if main_source is None:
        return None
    main_pre_output = tensor_contract_exact(
        model_ir,
        main_pre_output_name,
        4,
        tuple(main_source.shape[index] for index in _PERM_NHWC_TO_NCHW),
        tuple(main_source.signature[index] for index in _PERM_NHWC_TO_NCHW),
    )
    if (
        main_pre_output is None
        or main_source_name in public_outputs
        or main_pre_output_name in public_names
        or not _valid_source(
            graph_index,
            main_source,
            main_source_name,
            int(pre_index),
            public_inputs,
        )
        or not _producer_is_valid(
            graph_index,
            main_pre_output_name,
            int(pre_index),
        )
    ):
        return None

    core = match_decomposed_instance_norm_core(
        model_ir,
        graph_index,
        x_name=main_pre_output_name,
        x=main_pre_output,
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
    inst_output_name = (
        str(add_bias.outputs[0]) if len(add_bias.outputs) == 1 else ""
    )
    inst_output = tensor_contract_exact(
        model_ir,
        inst_output_name,
        4,
        main_pre_output.shape,
        main_pre_output.signature,
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

    tail_match = sole_consumer(graph_index, inst_output_name)
    if tail_match is None:
        return None
    tail_index, tail_add = tail_match
    residual_match = binary_other_input(tail_add, inst_output_name)
    old_add_output_name = (
        str(tail_add.outputs[0]) if len(tail_add.outputs) == 1 else ""
    )
    old_add_output = tensor_contract_exact(
        model_ir,
        old_add_output_name,
        4,
        main_pre_output.shape,
        main_pre_output.signature,
    )
    if (
        str(tail_add.op_type) != "ADD"
        or residual_match is None
        or len(tail_add.outputs) != 1
        or old_add_output is None
        or old_add_output_name in public_names
        or not _producer_is_valid(graph_index, old_add_output_name, tail_index)
    ):
        return None
    residual_input_name, residual_input_index = residual_match
    residual_pre_index_value = graph_index.producers.get(residual_input_name)
    if (
        residual_pre_index_value is None
        or residual_input_name in graph_index.duplicate_producers
    ):
        return None
    residual_pre_index = int(residual_pre_index_value)
    residual_pre = model_ir.operators[residual_pre_index]
    residual_source_name = (
        str(residual_pre.inputs[0]) if len(residual_pre.inputs) == 2 else ""
    )
    residual_source = tensor_contract_exact(
        model_ir,
        residual_source_name,
        4,
        main_source.shape,
        main_source.signature,
    )
    residual_pre_output = tensor_contract_exact(
        model_ir,
        residual_input_name,
        4,
        main_pre_output.shape,
        main_pre_output.signature,
    )
    if (
        str(residual_pre.op_type) != "TRANSPOSE"
        or len(residual_pre.inputs) != 2
        or len(residual_pre.outputs) != 1
        or str(residual_pre.outputs[0]) != residual_input_name
        or not _valid_permutation_constant(
            model_ir,
            graph_index,
            str(residual_pre.inputs[1]),
            _PERM_NHWC_TO_NCHW,
            public_inputs,
            public_names,
        )
        or residual_source is None
        or residual_pre_output is None
        or residual_source_name in public_outputs
        or residual_input_name in public_names
        or graph_index.consumer_indices(residual_input_name) != [tail_index]
        or not _producer_is_valid(
            graph_index,
            residual_input_name,
            residual_pre_index,
        )
        or not _valid_source(
            graph_index,
            residual_source,
            residual_source_name,
            residual_pre_index,
            public_inputs,
        )
    ):
        return None

    downstream_indices = graph_index.consumer_indices(old_add_output_name)
    involved_ops = (
        main_pre,
        residual_pre,
        *core.ordered_ops,
        add_bias,
        tail_add,
    )
    involved_indices = [graph_index.operator_index(op) for op in involved_ops]
    core_indices = [graph_index.operator_index(op) for op in core.ordered_ops]
    dtype = str(main_source.tensor.dtype)
    contracts: Tuple[_TensorContract, ...] = (
        main_source,
        main_pre_output,
        residual_source,
        residual_pre_output,
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
        old_add_output,
    )
    if (
        not downstream_indices
        or any(int(index) <= int(tail_index) for index in downstream_indices)
        or any(index is None for index in involved_indices)
        or len({id(op) for op in involved_ops}) != len(involved_ops)
        or int(pre_index) >= int(core_indices[0])
        or int(core_indices[-1]) >= int(bias_index)
        or int(bias_index) >= int(tail_index)
        or int(residual_pre_index) >= int(tail_index)
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
        channel_count=int(main_pre_output.shape[1]),
        public_names=public_names,
    )
    if constant_updates is None:
        return None

    adapter_permutation: Optional[TensorIR] = None
    existing_adapter_permutation = model_ir.tensors.get(_ADAPTER_PERM_NAME)
    if existing_adapter_permutation is None:
        adapter_permutation = TensorIR(
            name=_ADAPTER_PERM_NAME,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(_PERM_NHWC_TO_NCHW, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )
    elif not _valid_permutation_constant(
        model_ir,
        graph_index,
        _ADAPTER_PERM_NAME,
        _PERM_NHWC_TO_NCHW,
        public_inputs,
        public_names,
    ):
        return None

    new_add_output_name = _unique_tensor_name(
        model_ir,
        f"{old_add_output_name}_nhwc",
    )
    try:
        new_quantization = _clone_quantization(old_add_output.tensor.quantization)
    except Exception:
        return None
    new_add_output = TensorIR(
        name=new_add_output_name,
        dtype=dtype,
        shape=[
            int(old_add_output.shape[index]) for index in _PERM_NCHW_TO_NHWC
        ],
        shape_signature=[
            int(old_add_output.signature[index])
            for index in _PERM_NCHW_TO_NHWC
        ],
        data=None,
        is_variable=False,
        quantization=new_quantization,
        logical_layout=str(old_add_output.tensor.logical_layout),
        physical_layout=str(old_add_output.tensor.physical_layout),
        onnx_tensor_name=old_add_output.tensor.onnx_tensor_name,
    )
    full_contracts = (
        core.centered,
        core.squared,
        core.normalized,
        core.scaled,
        inst_output,
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
            tuple(
                contract.signature[index] for index in _PERM_NCHW_TO_NHWC
            ),
        )
        for contract in full_contracts + reduced_contracts
    )
    channel_last_names = tuple(
        dict.fromkeys(
            [
                main_source_name,
                residual_source_name,
                *(contract.tensor.name for contract in full_contracts),
                *(contract.tensor.name for contract in reduced_contracts),
                new_add_output_name,
            ]
        )
    )
    return _ResidualAddPlan(
        involved_ops=involved_ops,
        main_pre=main_pre,
        main_source_name=main_source_name,
        residual_pre=residual_pre,
        residual_source_name=residual_source_name,
        core=core,
        add_bias=add_bias,
        inst_output=inst_output,
        tail_add=tail_add,
        tail_add_residual_input_index=residual_input_index,
        old_add_output_name=old_add_output_name,
        new_add_output=new_add_output,
        adapter_permutation=adapter_permutation,
        constant_updates=constant_updates,
        metadata_updates=metadata_updates,
        channel_last_names=channel_last_names,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _ResidualAddPlan,
) -> bool:
    involved_indices = [graph_index.operator_index(op) for op in plan.involved_ops]
    clone_names = [
        update.clone_name
        for update in plan.constant_updates
        if update.clone_name is not None
    ]
    main_pre_index = graph_index.operator_index(plan.main_pre)
    residual_pre_index = graph_index.operator_index(plan.residual_pre)
    if (
        any(index is None for index in involved_indices)
        or len({int(index) for index in involved_indices if index is not None})
        != len(involved_indices)
        or main_pre_index is None
        or residual_pre_index is None
        or len(clone_names) != len(set(clone_names))
        or any(name in model_ir.tensors for name in clone_names)
        or any(
            update.clone_name is not None and update.clone is None
            for update in plan.constant_updates
        )
        or plan.new_add_output.name in model_ir.tensors
        or (
            plan.adapter_permutation is not None
            and _ADAPTER_PERM_NAME in model_ir.tensors
        )
        or int(plan.tail_add_residual_input_index) < 0
        or int(plan.tail_add_residual_input_index) >= len(plan.tail_add.inputs)
        or str(plan.tail_add.inputs[int(plan.tail_add_residual_input_index)])
        != str(plan.residual_pre.outputs[0])
        or str(plan.tail_add.outputs[0]) != plan.old_add_output_name
        or not graph_index.consumer_indices(plan.old_add_output_name)
    ):
        return False

    for update in plan.constant_updates:
        if not apply_constant_update(model_ir, graph_index, update):
            return False
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.core.mean1,
        input_index=0,
        new_input_name=plan.main_source_name,
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.core.sub,
        input_index=plan.core.sub_x_input_index,
        new_input_name=plan.main_source_name,
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.tail_add,
        input_index=plan.tail_add_residual_input_index,
        new_input_name=plan.residual_source_name,
        graph_index=graph_index,
    )
    for update in plan.metadata_updates:
        update.contract.tensor.shape = list(update.shape)
        update.contract.tensor.shape_signature = list(update.signature)
    if plan.adapter_permutation is not None:
        model_ir.tensors[_ADAPTER_PERM_NAME] = plan.adapter_permutation
    model_ir.tensors[plan.new_add_output.name] = plan.new_add_output
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.tail_add,
        new_outputs=[plan.new_add_output.name],
        graph_index=graph_index,
    )
    graph_index.remove_operators([main_pre_index, residual_pre_index])
    downstream_indices = graph_index.consumer_indices(plan.old_add_output_name)
    if not downstream_indices:
        return False
    graph_index.insert_operator(
        min(int(index) for index in downstream_indices),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[plan.new_add_output.name, _ADAPTER_PERM_NAME],
            outputs=[plan.old_add_output_name],
        ),
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


def optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Lift strict dual-pre-Transpose InstanceNorm residual ADD to NHWC."""

    if candidate is None:
        counts = Counter(str(operator.op_type) for operator in model_ir.operators)
        required = {
            "TRANSPOSE": 2,
            "MEAN": 2,
            "SUB": 1,
            "MUL": 3,
            "ADD": 3,
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
