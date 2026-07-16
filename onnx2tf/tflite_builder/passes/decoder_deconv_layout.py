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
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _any_rank_contract,
    _broadcast_shape,
    _expected_batch_matmul_shape,
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
    _unique_tensor_name,
)


_STATS_KEY = (
    "optimized_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input"
)
_PERM_EXPANDED_NCHW_TO_NHWC = (0, 2, 3, 1)


@dataclass(frozen=True)
class _BiasUpdate:
    operator: OperatorIR
    input_index: int
    tensor: TensorIR
    data: np.ndarray
    clone_name: Optional[str]
    clone: Optional[TensorIR]


@dataclass(frozen=True)
class _TensorMetadataUpdate:
    contract: _TensorContract
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _DecoderRewritePlan:
    ordered_ops: Tuple[OperatorIR, ...]
    matmul: OperatorIR
    matmul_inputs: Tuple[str, str]
    matmul_options: Dict[str, object]
    add: OperatorIR
    bias_update: _BiasUpdate
    expand: OperatorIR
    expand_axis_update: _ConstantUpdate
    transpose: OperatorIR
    deconvolution: OperatorIR
    metadata_updates: Tuple[_TensorMetadataUpdate, ...]


def _plan_bias_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    add: OperatorIR,
    add_index: int,
    input_index: int,
    length: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_BiasUpdate]:
    name = str(add.inputs[int(input_index)])
    tensor = model_ir.tensors.get(name)
    contract = _any_rank_contract(model_ir, name, minimum_rank=0)
    if (
        tensor is None
        or contract is None
        or tensor.data is None
        or name in public_inputs | public_outputs
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or str(tensor.dtype) not in {"FLOAT16", "FLOAT32", "FLOAT64"}
        or not _quantization_contract((contract,))
    ):
        return None
    try:
        data = np.asarray(tensor.data)
        expected_dtype = {
            "FLOAT16": np.float16,
            "FLOAT32": np.float32,
            "FLOAT64": np.float64,
        }[str(tensor.dtype)]
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
        if (
            data.dtype != np.dtype(expected_dtype)
            or data.size != int(length)
            or data.shape != shape
            or signature != shape
            or _broadcast_shape(shape, (1, 1, int(length)))
            != (1, 1, int(length))
        ):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    reshaped = np.asarray(data).reshape(1, int(length), 1)
    clone_name: Optional[str] = None
    clone: Optional[TensorIR] = None
    if set(graph_index.consumer_indices(name)) != {int(add_index)}:
        clone_name = _unique_tensor_name(model_ir, f"{name}_nlc")
        try:
            quantization = _clone_quantization(tensor.quantization)
        except Exception:
            return None
        clone = TensorIR(
            name=clone_name,
            dtype=str(tensor.dtype),
            shape=[1, int(length), 1],
            shape_signature=[1, int(length), 1],
            data=np.asarray(reshaped),
            is_variable=False,
            quantization=quantization,
        )
    return _BiasUpdate(
        operator=add,
        input_index=int(input_index),
        tensor=tensor,
        data=np.asarray(reshaped),
        clone_name=clone_name,
        clone=clone,
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    transpose_index: int,
) -> Optional[_DecoderRewritePlan]:
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
        != _PERM_EXPANDED_NCHW_TO_NHWC
    ):
        return None
    expand_output_name = str(transpose.inputs[0])
    transpose_output_name = str(transpose.outputs[0])
    if (
        not expand_output_name
        or not transpose_output_name
        or expand_output_name in public_inputs | public_outputs
        or transpose_output_name in public_inputs | public_outputs
        or not _producer_is_valid(
            graph_index,
            transpose_output_name,
            transpose_index,
        )
    ):
        return None
    transpose_users = graph_index.consumer_indices(transpose_output_name)
    if len(transpose_users) != 1 or int(transpose_users[0]) <= int(transpose_index):
        return None
    deconvolution_index = int(transpose_users[0])
    deconvolution = model_ir.operators[deconvolution_index]
    if (
        str(deconvolution.op_type) != "TRANSPOSE_CONV"
        or len(deconvolution.inputs) < 3
        or str(deconvolution.inputs[2]) != transpose_output_name
        or Counter(str(value) for value in deconvolution.inputs)[
            transpose_output_name
        ]
        != 1
    ):
        return None

    expand_index = graph_index.producers.get(expand_output_name)
    if (
        expand_index is None
        or expand_output_name in graph_index.duplicate_producers
        or graph_index.consumer_indices(expand_output_name)
        != [int(transpose_index)]
    ):
        return None
    expand = model_ir.operators[int(expand_index)]
    expand_axis = (
        _constant_vector(
            model_ir,
            graph_index,
            str(expand.inputs[1]),
            1,
            public_inputs,
        )
        if len(expand.inputs) == 2
        else None
    )
    if (
        str(expand.op_type) != "EXPAND_DIMS"
        or len(expand.inputs) != 2
        or len(expand.outputs) != 1
        or str(expand.outputs[0]) != expand_output_name
        or int(expand_index) >= int(transpose_index)
        or expand_axis not in {(2,), (-2,)}
    ):
        return None

    add_output_name = str(expand.inputs[0])
    add_index = graph_index.producers.get(add_output_name)
    if (
        add_index is None
        or add_output_name in graph_index.duplicate_producers
        or add_output_name in public_inputs | public_outputs
        or graph_index.consumer_indices(add_output_name) != [int(expand_index)]
    ):
        return None
    add = model_ir.operators[int(add_index)]
    if (
        str(add.op_type) != "ADD"
        or len(add.inputs) != 2
        or len(add.outputs) != 1
        or str(add.outputs[0]) != add_output_name
        or int(add_index) >= int(expand_index)
    ):
        return None

    matmul_matches = []
    for input_index, input_name in enumerate(add.inputs):
        producer_index = graph_index.producers.get(str(input_name))
        if (
            producer_index is not None
            and str(model_ir.operators[int(producer_index)].op_type)
            == "BATCH_MATMUL"
        ):
            matmul_matches.append((int(input_index), int(producer_index)))
    if len(matmul_matches) != 1:
        return None
    matmul_input_index, matmul_index = matmul_matches[0]
    bias_input_index = 1 - matmul_input_index
    matmul = model_ir.operators[matmul_index]
    matmul_output_name = str(add.inputs[matmul_input_index])
    if (
        len(matmul.inputs) != 2
        or len(matmul.outputs) != 1
        or str(matmul.outputs[0]) != matmul_output_name
        or matmul_output_name in public_inputs | public_outputs
        or matmul_output_name in graph_index.duplicate_producers
        or matmul_index >= int(add_index)
        or graph_index.consumer_indices(matmul_output_name) != [int(add_index)]
    ):
        return None

    lhs_name, rhs_name = (str(value) for value in matmul.inputs)
    lhs_contract = _any_rank_contract(model_ir, lhs_name, minimum_rank=2)
    rhs_contract = _any_rank_contract(model_ir, rhs_name, minimum_rank=2)
    matmul_contract = _tensor_contract(model_ir, matmul_output_name, 3)
    add_contract = _tensor_contract(model_ir, add_output_name, 3)
    expand_contract = _tensor_contract(model_ir, expand_output_name, 4)
    transpose_contract = _tensor_contract(model_ir, transpose_output_name, 4)
    if any(
        contract is None
        for contract in (
            lhs_contract,
            rhs_contract,
            matmul_contract,
            add_contract,
            expand_contract,
            transpose_contract,
        )
    ):
        return None
    assert lhs_contract is not None
    assert rhs_contract is not None
    assert matmul_contract is not None
    assert add_contract is not None
    assert expand_contract is not None
    assert transpose_contract is not None
    if (
        not _valid_source(
            graph_index,
            lhs_contract,
            lhs_name,
            matmul_index,
            public_inputs,
        )
        or not _valid_source(
            graph_index,
            rhs_contract,
            rhs_name,
            matmul_index,
            public_inputs,
        )
        or matmul_contract.shape != add_contract.shape
        or matmul_contract.signature != add_contract.signature
        or expand_contract.shape
        != (
            add_contract.shape[0],
            add_contract.shape[1],
            1,
            add_contract.shape[2],
        )
        or expand_contract.signature
        != (
            add_contract.signature[0],
            add_contract.signature[1],
            1,
            add_contract.signature[2],
        )
        or transpose_contract.shape
        != tuple(
            expand_contract.shape[index]
            for index in _PERM_EXPANDED_NCHW_TO_NHWC
        )
        or transpose_contract.signature
        != tuple(
            expand_contract.signature[index]
            for index in _PERM_EXPANDED_NCHW_TO_NHWC
        )
        or str(transpose_contract.tensor.dtype)
        != str(expand_contract.tensor.dtype)
        or not _quantization_contract((expand_contract, transpose_contract))
    ):
        return None
    data_contracts = (
        lhs_contract,
        rhs_contract,
        matmul_contract,
        add_contract,
        expand_contract,
        transpose_contract,
    )
    if (
        len({str(contract.tensor.dtype) for contract in data_contracts}) != 1
        or any(not _quantization_contract((contract,)) for contract in data_contracts)
    ):
        return None
    options = dict(matmul.options) if isinstance(matmul.options, dict) else {}
    old_adj_x = bool(options.get("adjX", False))
    old_adj_y = bool(options.get("adjY", False))
    expected_old_shape = _expected_batch_matmul_shape(
        lhs_contract.shape,
        rhs_contract.shape,
        adj_x=old_adj_x,
        adj_y=old_adj_y,
    )
    new_adj_x = not old_adj_y
    new_adj_y = not old_adj_x
    expected_new_shape = _expected_batch_matmul_shape(
        rhs_contract.shape,
        lhs_contract.shape,
        adj_x=new_adj_x,
        adj_y=new_adj_y,
    )
    new_shape = (
        matmul_contract.shape[0],
        matmul_contract.shape[2],
        matmul_contract.shape[1],
    )
    new_signature = (
        matmul_contract.signature[0],
        matmul_contract.signature[2],
        matmul_contract.signature[1],
    )
    if (
        expected_old_shape != matmul_contract.shape
        or expected_new_shape != new_shape
        or new_shape != (
            transpose_contract.shape[0],
            transpose_contract.shape[2],
            transpose_contract.shape[3],
        )
    ):
        return None
    length = int(matmul_contract.shape[2])
    bias_update = _plan_bias_update(
        model_ir,
        graph_index,
        add,
        int(add_index),
        bias_input_index,
        length,
        public_inputs,
        public_outputs,
    )
    if (
        bias_update is None
        or str(bias_update.tensor.dtype) != str(add_contract.tensor.dtype)
    ):
        return None
    expand_axis_update = _plan_constant_update(
        model_ir,
        graph_index,
        expand,
        int(expand_index),
        1,
        (1,),
        "nhwc_axis",
        public_outputs,
    )
    if expand_axis_update is None:
        return None

    options["adjX"] = bool(new_adj_x)
    options["adjY"] = bool(new_adj_y)
    metadata_updates = (
        _TensorMetadataUpdate(matmul_contract, new_shape, new_signature),
        _TensorMetadataUpdate(add_contract, new_shape, new_signature),
        _TensorMetadataUpdate(
            expand_contract,
            transpose_contract.shape,
            transpose_contract.signature,
        ),
    )
    ordered_ops = (matmul, add, expand, transpose, deconvolution)
    ordered_indices = tuple(
        graph_index.operator_index(operator) for operator in ordered_ops
    )
    if (
        any(index is None for index in ordered_indices)
        or [int(index) for index in ordered_indices if index is not None]
        != sorted(int(index) for index in ordered_indices if index is not None)
    ):
        return None
    return _DecoderRewritePlan(
        ordered_ops=ordered_ops,
        matmul=matmul,
        matmul_inputs=(rhs_name, lhs_name),
        matmul_options=options,
        add=add,
        bias_update=bias_update,
        expand=expand,
        expand_axis_update=expand_axis_update,
        transpose=transpose,
        deconvolution=deconvolution,
        metadata_updates=metadata_updates,
    )


def _apply_bias_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    update: _BiasUpdate,
) -> bool:
    if update.clone_name is not None:
        if update.clone is None or update.clone_name in model_ir.tensors:
            return False
        model_ir.tensors[update.clone_name] = update.clone
        _replace_operator_input_at(
            model_ir=model_ir,
            op=update.operator,
            input_index=update.input_index,
            new_input_name=update.clone_name,
            graph_index=graph_index,
        )
        return True
    update.tensor.data = np.asarray(update.data)
    update.tensor.shape = [int(value) for value in update.data.shape]
    update.tensor.shape_signature = [int(value) for value in update.data.shape]
    return True


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _DecoderRewritePlan,
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
    clone_name_values = [
        name
        for name in (
            plan.bias_update.clone_name,
            plan.expand_axis_update.clone_name,
        )
        if name is not None
    ]
    if (
        len(clone_name_values) != len(set(clone_name_values))
        or any(name in model_ir.tensors for name in clone_name_values)
        or (
            plan.bias_update.clone_name is not None
            and plan.bias_update.clone is None
        )
        or (
            plan.expand_axis_update.clone_name is not None
            and plan.expand_axis_update.clone is None
        )
    ):
        return False
    if not _apply_bias_update(model_ir, graph_index, plan.bias_update):
        return False
    if not _apply_constant_update(
        model_ir,
        graph_index,
        plan.expand_axis_update,
    ):
        return False

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.matmul,
        new_inputs=list(plan.matmul_inputs),
        graph_index=graph_index,
    )
    plan.matmul.options = dict(plan.matmul_options)
    for update in plan.metadata_updates:
        update.contract.tensor.shape = list(update.shape)
        update.contract.tensor.shape_signature = list(update.signature)
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.deconvolution,
        input_index=2,
        new_input_name=str(plan.expand.outputs[0]),
        graph_index=graph_index,
    )
    graph_index.remove_operator(int(transpose_index))
    return True


def _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    counts = Counter(str(operator.op_type) for operator in model_ir.operators)
    required = ("BATCH_MATMUL", "ADD", "EXPAND_DIMS", "TRANSPOSE", "TRANSPOSE_CONV")
    if not all(counts[op_type] for op_type in required):
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
        plan = _resolve_candidate(
            model_ir,
            active_index,
            int(transpose_index),
        )
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1
    if rewritten:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
