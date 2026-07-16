from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _TensorContract,
    _UNARY_OPS,
    _constant_vector,
    _producer_is_valid,
    _quantization_contract,
    _squeeze_axis,
    _tensor_contract,
)


_STATS_KEY = "optimized_transpose_squeeze_unary_batchmatmul_nhwc_chains"
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_SWAP_LAST_TWO = (0, 2, 1)


@dataclass(frozen=True)
class _TensorMetadataUpdate:
    contract: _TensorContract
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _BatchMatMulRewritePlan:
    ordered_ops: Tuple[OperatorIR, ...]
    pre: OperatorIR
    squeeze: OperatorIR
    squeeze_options: Dict[str, object]
    source_name: str
    metadata_updates: Tuple[_TensorMetadataUpdate, ...]
    matmul: OperatorIR
    matmul_options: Dict[str, object]


def _any_rank_contract(
    model_ir: ModelIR,
    tensor_name: str,
    *,
    minimum_rank: int,
) -> Optional[_TensorContract]:
    tensor = model_ir.tensors.get(str(tensor_name))
    if tensor is None:
        return None
    try:
        rank = len(tensor.shape)
    except (TypeError, ValueError):
        return None
    if rank < int(minimum_rank):
        return None
    return _tensor_contract(model_ir, str(tensor_name), rank)


def _valid_source(
    graph_index: ModelIRGraphIndex,
    contract: _TensorContract,
    tensor_name: str,
    consumer_index: int,
    public_inputs: set[str],
) -> bool:
    name = str(tensor_name)
    if name in graph_index.duplicate_producers:
        return False
    producer = graph_index.producers.get(name)
    if producer is not None:
        return name not in public_inputs and int(producer) < int(consumer_index)
    return name in public_inputs or contract.tensor.data is not None


def _broadcast_shape(
    left: Tuple[int, ...],
    right: Tuple[int, ...],
) -> Optional[Tuple[int, ...]]:
    result = []
    max_rank = max(len(left), len(right))
    padded_left = (1,) * (max_rank - len(left)) + tuple(left)
    padded_right = (1,) * (max_rank - len(right)) + tuple(right)
    for left_dim, right_dim in zip(padded_left, padded_right):
        if int(left_dim) == int(right_dim):
            result.append(int(left_dim))
        elif int(left_dim) == 1:
            result.append(int(right_dim))
        elif int(right_dim) == 1:
            result.append(int(left_dim))
        else:
            return None
    return tuple(result)


def _expected_batch_matmul_shape(
    lhs_shape: Tuple[int, ...],
    rhs_shape: Tuple[int, ...],
    *,
    adj_x: bool,
    adj_y: bool,
) -> Optional[Tuple[int, ...]]:
    lhs_rows, lhs_depth = (
        (lhs_shape[-1], lhs_shape[-2])
        if adj_x
        else (lhs_shape[-2], lhs_shape[-1])
    )
    rhs_depth, rhs_columns = (
        (rhs_shape[-1], rhs_shape[-2])
        if adj_y
        else (rhs_shape[-2], rhs_shape[-1])
    )
    if int(lhs_depth) != int(rhs_depth):
        return None
    batch_shape = _broadcast_shape(lhs_shape[:-2], rhs_shape[:-2])
    if batch_shape is None:
        return None
    return batch_shape + (int(lhs_rows), int(rhs_columns))


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_BatchMatMulRewritePlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    pre = model_ir.operators[int(pre_index)]
    if (
        len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or _constant_vector(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NHWC_TO_NCHW
    ):
        return None

    source_name = str(pre.inputs[0])
    pre_output_name = str(pre.outputs[0])
    if (
        not source_name
        or not pre_output_name
        or pre_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, pre_output_name, pre_index)
    ):
        return None
    source_contract = _tensor_contract(model_ir, source_name, 4)
    pre_contract = _tensor_contract(model_ir, pre_output_name, 4)
    if (
        source_contract is None
        or pre_contract is None
        or pre_contract.shape
        != tuple(source_contract.shape[index] for index in _PERM_NHWC_TO_NCHW)
        or pre_contract.signature
        != tuple(source_contract.signature[index] for index in _PERM_NHWC_TO_NCHW)
        or not _valid_source(
            graph_index,
            source_contract,
            source_name,
            int(pre_index),
            public_inputs,
        )
        or str(source_contract.tensor.dtype) != str(pre_contract.tensor.dtype)
        or not _quantization_contract((source_contract, pre_contract))
    ):
        return None

    pre_users = graph_index.consumer_indices(pre_output_name)
    if len(pre_users) != 1 or int(pre_users[0]) <= int(pre_index):
        return None
    squeeze_index = int(pre_users[0])
    squeeze = model_ir.operators[squeeze_index]
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or len(squeeze.outputs) != 1
        or str(squeeze.inputs[0]) != pre_output_name
        or "squeezeDims"
        not in (squeeze.options if isinstance(squeeze.options, dict) else {})
    ):
        return None
    squeeze_output_name = str(squeeze.outputs[0])
    if (
        not squeeze_output_name
        or squeeze_output_name in public_inputs | public_outputs
        or not _producer_is_valid(
            graph_index,
            squeeze_output_name,
            squeeze_index,
        )
    ):
        return None
    squeeze_contract = _tensor_contract(model_ir, squeeze_output_name, 3)
    squeeze_axis = (
        None
        if squeeze_contract is None
        else _squeeze_axis(squeeze, pre_contract, squeeze_contract)
    )
    if (
        squeeze_contract is None
        or squeeze_axis not in {1, 2, 3}
        or str(squeeze_contract.tensor.dtype) != str(pre_contract.tensor.dtype)
        or not _quantization_contract((pre_contract, squeeze_contract))
    ):
        return None

    old_dim_indices = tuple(
        int(_PERM_NHWC_TO_NCHW[index])
        for index in range(4)
        if index != int(squeeze_axis)
    )
    source_axis = int(_PERM_NHWC_TO_NCHW[int(squeeze_axis)])
    new_dim_indices = tuple(index for index in range(4) if index != source_axis)
    if old_dim_indices == new_dim_indices:
        metadata_permutation: Optional[Tuple[int, ...]] = None
    elif old_dim_indices == (
        new_dim_indices[0],
        new_dim_indices[2],
        new_dim_indices[1],
    ):
        metadata_permutation = _PERM_SWAP_LAST_TWO
    else:
        return None
    new_shape = tuple(source_contract.shape[index] for index in new_dim_indices)
    new_signature = tuple(
        source_contract.signature[index] for index in new_dim_indices
    )

    chain = []
    metadata_contracts = [squeeze_contract]
    current_name = squeeze_output_name
    previous_contract = squeeze_contract
    previous_index = squeeze_index
    while True:
        users = graph_index.consumer_indices(current_name)
        if len(users) != 1:
            break
        unary_index = int(users[0])
        unary = model_ir.operators[unary_index]
        if str(unary.op_type) not in _UNARY_OPS:
            break
        if (
            len(unary.inputs) != 1
            or len(unary.outputs) != 1
            or str(unary.inputs[0]) != current_name
            or unary_index <= previous_index
        ):
            return None
        unary_output_name = str(unary.outputs[0])
        if (
            not unary_output_name
            or unary_output_name in public_inputs | public_outputs
            or not _producer_is_valid(
                graph_index,
                unary_output_name,
                unary_index,
            )
        ):
            return None
        unary_contract = _tensor_contract(model_ir, unary_output_name, 3)
        if (
            unary_contract is None
            or unary_contract.shape != previous_contract.shape
            or unary_contract.signature != previous_contract.signature
            or not _quantization_contract((unary_contract,))
        ):
            return None
        if str(unary.op_type) != "CAST" and (
            str(unary_contract.tensor.dtype) != str(previous_contract.tensor.dtype)
            or not _quantization_contract((previous_contract, unary_contract))
        ):
            return None
        chain.append(unary)
        metadata_contracts.append(unary_contract)
        current_name = unary_output_name
        previous_contract = unary_contract
        previous_index = unary_index

    tail_users = graph_index.consumer_indices(current_name)
    if len(tail_users) != 1 or int(tail_users[0]) <= previous_index:
        return None
    matmul_index = int(tail_users[0])
    matmul = model_ir.operators[matmul_index]
    if (
        str(matmul.op_type) != "BATCH_MATMUL"
        or len(matmul.inputs) != 2
        or len(matmul.outputs) != 1
        or str(matmul.inputs[0]) != current_name
        or Counter(str(value) for value in matmul.inputs)[current_name] != 1
    ):
        return None

    rhs_name = str(matmul.inputs[1])
    output_name = str(matmul.outputs[0])
    rhs_contract = _any_rank_contract(model_ir, rhs_name, minimum_rank=2)
    output_contract = _any_rank_contract(model_ir, output_name, minimum_rank=2)
    if (
        rhs_contract is None
        or output_contract is None
        or output_name in public_inputs
        or not _valid_source(
            graph_index,
            rhs_contract,
            rhs_name,
            matmul_index,
            public_inputs,
        )
        or not _producer_is_valid(graph_index, output_name, matmul_index)
        or any(
            int(index) <= matmul_index
            for index in graph_index.consumer_indices(output_name)
        )
        or len(
            {
                str(previous_contract.tensor.dtype),
                str(rhs_contract.tensor.dtype),
                str(output_contract.tensor.dtype),
            }
        )
        != 1
        or not _quantization_contract(
            (previous_contract, rhs_contract, output_contract)
        )
    ):
        return None
    options = dict(matmul.options) if isinstance(matmul.options, dict) else {}
    old_adj_x = bool(options.get("adjX", False))
    adj_y = bool(options.get("adjY", False))
    expected_output_shape = _expected_batch_matmul_shape(
        previous_contract.shape,
        rhs_contract.shape,
        adj_x=old_adj_x,
        adj_y=adj_y,
    )
    if expected_output_shape is None or output_contract.shape != expected_output_shape:
        return None
    new_adj_x = old_adj_x if metadata_permutation is None else not old_adj_x
    new_effective_shape = _expected_batch_matmul_shape(
        new_shape,
        rhs_contract.shape,
        adj_x=new_adj_x,
        adj_y=adj_y,
    )
    if new_effective_shape != expected_output_shape:
        return None

    names = (
        source_name,
        pre_output_name,
        squeeze_output_name,
        *(str(operator.outputs[0]) for operator in chain),
        rhs_name,
        output_name,
    )
    if len(names) != len(set(names)):
        return None
    ordered_ops = (pre, squeeze, *chain, matmul)
    ordered_indices = tuple(
        graph_index.operator_index(operator) for operator in ordered_ops
    )
    if (
        any(index is None for index in ordered_indices)
        or [int(index) for index in ordered_indices if index is not None]
        != sorted(int(index) for index in ordered_indices if index is not None)
    ):
        return None

    squeeze_options = (
        dict(squeeze.options) if isinstance(squeeze.options, dict) else {}
    )
    squeeze_options["squeezeDims"] = [source_axis]
    options["adjX"] = bool(new_adj_x)
    metadata_updates = tuple(
        _TensorMetadataUpdate(
            contract=contract,
            shape=(
                new_shape
                if metadata_permutation is not None
                else contract.shape
            ),
            signature=(
                new_signature
                if metadata_permutation is not None
                else contract.signature
            ),
        )
        for contract in metadata_contracts
    )
    return _BatchMatMulRewritePlan(
        ordered_ops=ordered_ops,
        pre=pre,
        squeeze=squeeze,
        squeeze_options=squeeze_options,
        source_name=source_name,
        metadata_updates=metadata_updates,
        matmul=matmul,
        matmul_options=options,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _BatchMatMulRewritePlan,
) -> bool:
    indices = [graph_index.operator_index(operator) for operator in plan.ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != len(resolved):
        return False
    pre_index = graph_index.operator_index(plan.pre)
    if pre_index is None:
        return False

    plan.squeeze.options = dict(plan.squeeze_options)
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.squeeze,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    for update in plan.metadata_updates:
        update.contract.tensor.shape = list(update.shape)
        update.contract.tensor.shape_signature = list(update.signature)
    plan.matmul.options = dict(plan.matmul_options)
    graph_index.remove_operator(int(pre_index))
    return True


def _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    counts = Counter(str(operator.op_type) for operator in model_ir.operators)
    if not all(counts[op_type] for op_type in ("TRANSPOSE", "SQUEEZE", "BATCH_MATMUL")):
        return {_STATS_KEY: 0}
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    candidates = [
        model_ir.operators[index]
        for index in active_index.operator_indices("TRANSPOSE")
    ]
    rewritten = 0
    for pre in candidates:
        pre_index = active_index.operator_index(pre)
        if pre_index is None:
            continue
        plan = _resolve_candidate(model_ir, active_index, int(pre_index))
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1
    if rewritten:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
