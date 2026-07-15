from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _normalize_squeeze_axes_for_rank,
    _prune_unused_tensors,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _valid_source,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _ConstantUpdate,
    _TensorContract,
    _apply_constant_update,
    _constant_vector,
    _plan_constant_update,
    _producer_is_valid,
    _quantization_contract,
    _tensor_contract,
)


_STATS_KEY = "optimized_transpose_squeeze_mean_squeeze_terminal_nhwc_chains"
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_RANK3_SWAP = (0, 2, 1)


@dataclass(frozen=True)
class _TensorMetadataUpdate:
    contract: _TensorContract
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _TerminalMeanRewritePlan:
    ordered_ops: Tuple[OperatorIR, ...]
    transpose: OperatorIR
    squeeze1: OperatorIR
    squeeze1_options: Dict[str, object]
    source_name: str
    mean: OperatorIR
    mean_axis_update: _ConstantUpdate
    squeeze2: OperatorIR
    squeeze2_options: Dict[str, object]
    metadata_updates: Tuple[_TensorMetadataUpdate, ...]


def _normalized_axes(
    operator: OperatorIR,
    rank: int,
) -> Optional[Tuple[int, ...]]:
    options = dict(operator.options) if isinstance(operator.options, dict) else {}
    if "squeezeDims" not in options:
        return None
    raw_axes = options.get("squeezeDims")
    try:
        axes = [
            int(value)
            for value in np.asarray(raw_axes).reshape(-1).tolist()
        ]
    except Exception:
        return None
    normalized = _normalize_squeeze_axes_for_rank(axes, int(rank))
    if normalized is None:
        return None
    return tuple(int(value) for value in normalized)


def _expected_squeeze(
    source: _TensorContract,
    axis: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return (
        tuple(
            value
            for index, value in enumerate(source.shape)
            if int(index) != int(axis)
        ),
        tuple(
            value
            for index, value in enumerate(source.signature)
            if int(index) != int(axis)
        ),
    )


def _expected_keepdims_mean(
    source: _TensorContract,
    axis: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    shape = list(source.shape)
    signature = list(source.signature)
    shape[int(axis)] = 1
    signature[int(axis)] = 1
    return tuple(shape), tuple(signature)


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    transpose_index: int,
) -> Optional[_TerminalMeanRewritePlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    transpose = model_ir.operators[int(transpose_index)]
    if (
        len(transpose.inputs) != 2
        or len(transpose.outputs) != 1
        or _constant_vector(
            model_ir,
            graph_index,
            str(transpose.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NHWC_TO_NCHW
        or str(transpose.inputs[1]) in public_outputs
    ):
        return None

    source_name = str(transpose.inputs[0])
    transpose_output_name = str(transpose.outputs[0])
    if (
        not source_name
        or not transpose_output_name
        or transpose_output_name in public_inputs | public_outputs
        or not _producer_is_valid(
            graph_index,
            transpose_output_name,
            int(transpose_index),
        )
    ):
        return None
    source_contract = _tensor_contract(model_ir, source_name, 4)
    transpose_contract = _tensor_contract(model_ir, transpose_output_name, 4)
    if (
        source_contract is None
        or transpose_contract is None
        or transpose_contract.shape
        != tuple(source_contract.shape[index] for index in _PERM_NHWC_TO_NCHW)
        or transpose_contract.signature
        != tuple(source_contract.signature[index] for index in _PERM_NHWC_TO_NCHW)
        or not _valid_source(
            graph_index,
            source_contract,
            source_name,
            int(transpose_index),
            public_inputs,
        )
    ):
        return None

    transpose_users = graph_index.consumer_indices(transpose_output_name)
    if len(transpose_users) != 1 or int(transpose_users[0]) <= int(transpose_index):
        return None
    squeeze1_index = int(transpose_users[0])
    squeeze1 = model_ir.operators[squeeze1_index]
    squeeze1_output_name = (
        str(squeeze1.outputs[0]) if len(squeeze1.outputs) == 1 else ""
    )
    if (
        str(squeeze1.op_type) != "SQUEEZE"
        or len(squeeze1.inputs) != 1
        or len(squeeze1.outputs) != 1
        or str(squeeze1.inputs[0]) != transpose_output_name
        or _normalized_axes(squeeze1, 4) != (2,)
        or not squeeze1_output_name
        or squeeze1_output_name in public_inputs | public_outputs
        or not _producer_is_valid(
            graph_index,
            squeeze1_output_name,
            squeeze1_index,
        )
    ):
        return None
    squeeze1_contract = _tensor_contract(model_ir, squeeze1_output_name, 3)
    expected_squeeze1 = _expected_squeeze(transpose_contract, 2)
    if (
        squeeze1_contract is None
        or transpose_contract.shape[2] != 1
        or transpose_contract.signature[2] != 1
        or squeeze1_contract.shape != expected_squeeze1[0]
        or squeeze1_contract.signature != expected_squeeze1[1]
    ):
        return None

    squeeze1_users = graph_index.consumer_indices(squeeze1_output_name)
    if len(squeeze1_users) != 1 or int(squeeze1_users[0]) <= squeeze1_index:
        return None
    mean_index = int(squeeze1_users[0])
    mean = model_ir.operators[mean_index]
    mean_output_name = str(mean.outputs[0]) if len(mean.outputs) == 1 else ""
    mean_options = dict(mean.options) if isinstance(mean.options, dict) else {}
    if (
        str(mean.op_type) != "MEAN"
        or len(mean.inputs) != 2
        or len(mean.outputs) != 1
        or str(mean.inputs[0]) != squeeze1_output_name
        or not bool(mean_options.get("keepDims", False))
        or not mean_output_name
        or mean_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, mean_output_name, mean_index)
    ):
        return None
    mean_axis_name = str(mean.inputs[1])
    mean_axis_values = _constant_vector(
        model_ir,
        graph_index,
        mean_axis_name,
        1,
        public_inputs,
    )
    if (
        mean_axis_values is None
        or _normalize_squeeze_axes_for_rank(list(mean_axis_values), 3) != [1]
        or mean_axis_name in public_outputs
    ):
        return None
    mean_contract = _tensor_contract(model_ir, mean_output_name, 3)
    expected_mean = _expected_keepdims_mean(squeeze1_contract, 1)
    if (
        mean_contract is None
        or mean_contract.shape != expected_mean[0]
        or mean_contract.signature != expected_mean[1]
    ):
        return None

    mean_users = graph_index.consumer_indices(mean_output_name)
    if len(mean_users) != 1 or int(mean_users[0]) <= mean_index:
        return None
    squeeze2_index = int(mean_users[0])
    squeeze2 = model_ir.operators[squeeze2_index]
    output_name = str(squeeze2.outputs[0]) if len(squeeze2.outputs) == 1 else ""
    if (
        str(squeeze2.op_type) != "SQUEEZE"
        or len(squeeze2.inputs) != 1
        or len(squeeze2.outputs) != 1
        or str(squeeze2.inputs[0]) != mean_output_name
        or _normalized_axes(squeeze2, 3) != (1,)
        or not output_name
        or output_name in public_inputs
        or not _producer_is_valid(graph_index, output_name, squeeze2_index)
    ):
        return None
    output_contract = _tensor_contract(model_ir, output_name, 2)
    expected_output = _expected_squeeze(mean_contract, 1)
    if (
        output_contract is None
        or mean_contract.shape[1] != 1
        or mean_contract.signature[1] != 1
        or output_contract.shape != expected_output[0]
        or output_contract.signature != expected_output[1]
        or any(
            int(consumer_index) <= squeeze2_index
            for consumer_index in graph_index.consumer_indices(output_name)
        )
    ):
        return None

    data_contracts = (
        source_contract,
        transpose_contract,
        squeeze1_contract,
        mean_contract,
        output_contract,
    )
    if (
        len({str(contract.tensor.dtype) for contract in data_contracts}) != 1
        or not _quantization_contract(
            (source_contract, transpose_contract, squeeze1_contract)
        )
        or not _quantization_contract((mean_contract, output_contract))
        or any(
            not _quantization_contract((contract,))
            for contract in data_contracts
        )
    ):
        return None

    mean_axis_update = _plan_constant_update(
        model_ir,
        graph_index,
        mean,
        mean_index,
        1,
        (2,),
        "nhwc_axis",
        public_outputs,
    )
    if mean_axis_update is None:
        return None

    squeeze1_options = dict(squeeze1.options)
    squeeze1_options["squeezeDims"] = [1]
    squeeze2_options = dict(squeeze2.options)
    squeeze2_options["squeezeDims"] = [2]
    new_squeeze1_shape = (
        source_contract.shape[0],
        source_contract.shape[2],
        source_contract.shape[3],
    )
    new_squeeze1_signature = (
        source_contract.signature[0],
        source_contract.signature[2],
        source_contract.signature[3],
    )
    new_mean_shape = list(new_squeeze1_shape)
    new_mean_signature = list(new_squeeze1_signature)
    new_mean_shape[2] = 1
    new_mean_signature[2] = 1
    if (
        new_squeeze1_shape
        != tuple(squeeze1_contract.shape[index] for index in _RANK3_SWAP)
        or new_squeeze1_signature
        != tuple(squeeze1_contract.signature[index] for index in _RANK3_SWAP)
        or tuple(new_mean_shape)
        != tuple(mean_contract.shape[index] for index in _RANK3_SWAP)
        or tuple(new_mean_signature)
        != tuple(mean_contract.signature[index] for index in _RANK3_SWAP)
        or _expected_squeeze(
            _TensorContract(
                tensor=mean_contract.tensor,
                shape=tuple(new_mean_shape),
                signature=tuple(new_mean_signature),
            ),
            2,
        )
        != (output_contract.shape, output_contract.signature)
    ):
        return None

    ordered_ops = (transpose, squeeze1, mean, squeeze2)
    ordered_indices = tuple(
        graph_index.operator_index(operator) for operator in ordered_ops
    )
    if (
        any(index is None for index in ordered_indices)
        or [int(index) for index in ordered_indices if index is not None]
        != sorted(int(index) for index in ordered_indices if index is not None)
    ):
        return None
    return _TerminalMeanRewritePlan(
        ordered_ops=ordered_ops,
        transpose=transpose,
        squeeze1=squeeze1,
        squeeze1_options=squeeze1_options,
        source_name=source_name,
        mean=mean,
        mean_axis_update=mean_axis_update,
        squeeze2=squeeze2,
        squeeze2_options=squeeze2_options,
        metadata_updates=(
            _TensorMetadataUpdate(
                squeeze1_contract,
                new_squeeze1_shape,
                new_squeeze1_signature,
            ),
            _TensorMetadataUpdate(
                mean_contract,
                tuple(new_mean_shape),
                tuple(new_mean_signature),
            ),
        ),
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _TerminalMeanRewritePlan,
) -> bool:
    indices = [graph_index.operator_index(operator) for operator in plan.ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != len(resolved):
        return False
    transpose_index = graph_index.operator_index(plan.transpose)
    if transpose_index is None:
        return False
    update = plan.mean_axis_update
    if (
        update.clone_name is not None
        and (
            update.clone is None
            or update.clone_name in model_ir.tensors
        )
    ):
        return False
    if not _apply_constant_update(model_ir, graph_index, update):
        return False

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.squeeze1,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    plan.squeeze1.options = dict(plan.squeeze1_options)
    plan.squeeze2.options = dict(plan.squeeze2_options)
    for metadata_update in plan.metadata_updates:
        metadata_update.contract.tensor.shape = list(metadata_update.shape)
        metadata_update.contract.tensor.shape_signature = list(
            metadata_update.signature
        )
    graph_index.remove_operator(int(transpose_index))
    return True


def _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    counts = Counter(str(operator.op_type) for operator in model_ir.operators)
    if (
        counts["TRANSPOSE"] < 1
        or counts["SQUEEZE"] < 2
        or counts["MEAN"] < 1
    ):
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
    for transpose in candidates:
        transpose_index = active_index.operator_index(transpose)
        if transpose_index is None:
            continue
        plan = _resolve_candidate(model_ir, active_index, transpose_index)
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1
    if rewritten:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
