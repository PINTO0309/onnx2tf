from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_instance_norm_layout import (
    _FlatInstanceNormPrefixPlan,
    _resolve_flattened_instance_norm_prefix,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _ConstantUpdate,
    _TensorContract,
    _apply_constant_update,
    _constant_vector,
    _plan_constant_update,
    _producer_is_valid,
    _quantization_contract,
    _squeeze_axis,
    _tensor_contract,
    _unique_tensor_name,
)


_STATS_KEY = "optimized_tencoder_add_expand_transpose_conv_nhwc_chains"
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_PERM_NCW_TO_NWC = (0, 2, 1)


@dataclass(frozen=True)
class _LHSBranchPlan:
    kind: str
    pre: OperatorIR
    squeeze: Optional[OperatorIR]
    source_name: str
    source_contract: _TensorContract
    output_name: str
    output_contract: _TensorContract
    n: int
    w: int
    c: int
    n_signature: int
    w_signature: int
    c_signature: int


@dataclass(frozen=True)
class _ScaleConstantUpdate:
    operator: OperatorIR
    input_index: int
    tensor: TensorIR
    data: np.ndarray
    clone_name: Optional[str]
    clone: Optional[TensorIR]


@dataclass(frozen=True)
class _GateBranchPlan:
    prefix: _FlatInstanceNormPrefixPlan
    slice0: OperatorIR
    slice1: OperatorIR
    logistic: OperatorIR
    gate: OperatorIR
    scale: OperatorIR
    slice0_contract: _TensorContract
    slice1_contract: _TensorContract
    logistic_contract: _TensorContract
    gate_contract: _TensorContract
    scale_contract: _TensorContract
    scale_constant_update: Optional[_ScaleConstantUpdate]
    n: int
    w: int
    c: int
    c2: int
    n_signature: int
    w_signature: int
    c_signature: int
    c2_signature: int


@dataclass(frozen=True)
class _TensorMetadataUpdate:
    tensor: TensorIR
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _TencoderRewritePlan:
    ordered_ops: Tuple[OperatorIR, ...]
    removed_ops: Tuple[OperatorIR, ...]
    lhs: _LHSBranchPlan
    rhs: _GateBranchPlan
    add: OperatorIR
    add_contract: _TensorContract
    expand: OperatorIR
    expand_contract: _TensorContract
    post: OperatorIR
    post_output_name: str
    int_updates: Tuple[_ConstantUpdate, ...]
    metadata_updates: Tuple[_TensorMetadataUpdate, ...]
    rhs_squeeze_options: dict
    lhs_squeeze_options: Optional[dict]
    reshape2_options: dict
    side_users: Tuple[OperatorIR, ...]
    bridge_permutation: Optional[TensorIR]
    bridge_tensor: Optional[TensorIR]
    bridge: Optional[OperatorIR]


def _data_contract(
    model_ir: ModelIR,
    tensor_name: str,
    rank: int,
) -> Optional[_TensorContract]:
    return _tensor_contract(model_ir, tensor_name, rank)


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


def _same_unquantized_float(contracts: Tuple[_TensorContract, ...]) -> bool:
    return bool(
        contracts
        and len({str(contract.tensor.dtype) for contract in contracts}) == 1
        and str(contracts[0].tensor.dtype) in {"FLOAT16", "FLOAT32"}
        and _quantization_contract(contracts)
        and all(contract.tensor.quantization is None for contract in contracts)
    )


def _resolve_simple_lhs(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    output_name: str,
    add_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_LHSBranchPlan]:
    name = str(output_name)
    squeeze_index = graph_index.producers.get(name)
    if squeeze_index is None or name in graph_index.duplicate_producers:
        return None
    squeeze = model_ir.operators[int(squeeze_index)]
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or len(squeeze.outputs) != 1
        or str(squeeze.outputs[0]) != name
        or int(squeeze_index) >= int(add_index)
        or name in public_inputs | public_outputs
        or graph_index.consumer_indices(name) != [int(add_index)]
    ):
        return None

    pre_output_name = str(squeeze.inputs[0])
    pre_index = graph_index.producers.get(pre_output_name)
    if pre_index is None or pre_output_name in graph_index.duplicate_producers:
        return None
    pre = model_ir.operators[int(pre_index)]
    if (
        str(pre.op_type) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or str(pre.outputs[0]) != pre_output_name
        or _constant_vector(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NHWC_TO_NCHW
        or int(pre_index) >= int(squeeze_index)
        or pre_output_name in public_inputs | public_outputs
        or graph_index.consumer_indices(pre_output_name) != [int(squeeze_index)]
    ):
        return None

    source_name = str(pre.inputs[0])
    source_contract = _data_contract(model_ir, source_name, 4)
    pre_contract = _data_contract(model_ir, pre_output_name, 4)
    output_contract = _data_contract(model_ir, name, 3)
    if (
        source_contract is None
        or pre_contract is None
        or output_contract is None
        or source_contract.shape[1] != 1
        or pre_contract.shape
        != tuple(source_contract.shape[index] for index in _PERM_NHWC_TO_NCHW)
        or pre_contract.signature
        != tuple(
            source_contract.signature[index] for index in _PERM_NHWC_TO_NCHW
        )
        or _squeeze_axis(squeeze, pre_contract, output_contract) != 2
        or not _valid_source(
            graph_index,
            source_contract,
            source_name,
            int(pre_index),
            public_inputs,
        )
        or not _same_unquantized_float(
            (source_contract, pre_contract, output_contract)
        )
    ):
        return None
    n, _, w, c = source_contract.shape
    n_signature, _, w_signature, c_signature = source_contract.signature
    if (
        output_contract.shape != (n, c, w)
        or output_contract.signature != (n_signature, c_signature, w_signature)
    ):
        return None
    return _LHSBranchPlan(
        kind="simple",
        pre=pre,
        squeeze=squeeze,
        source_name=source_name,
        source_contract=source_contract,
        output_name=name,
        output_contract=output_contract,
        n=int(n),
        w=int(w),
        c=int(c),
        n_signature=int(n_signature),
        w_signature=int(w_signature),
        c_signature=int(c_signature),
    )


def _resolve_legacy_lhs(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    output_name: str,
    add_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_LHSBranchPlan]:
    name = str(output_name)
    pre_index = graph_index.producers.get(name)
    if pre_index is None or name in graph_index.duplicate_producers:
        return None
    pre = model_ir.operators[int(pre_index)]
    if (
        str(pre.op_type) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or str(pre.outputs[0]) != name
        or _constant_vector(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            3,
            public_inputs,
        )
        != _PERM_NCW_TO_NWC
        or int(pre_index) >= int(add_index)
        or name in public_inputs | public_outputs
        or graph_index.consumer_indices(name) != [int(add_index)]
    ):
        return None
    source_name = str(pre.inputs[0])
    source_contract = _data_contract(model_ir, source_name, 3)
    output_contract = _data_contract(model_ir, name, 3)
    if (
        source_contract is None
        or output_contract is None
        or output_contract.shape
        != tuple(source_contract.shape[index] for index in _PERM_NCW_TO_NWC)
        or output_contract.signature
        != tuple(source_contract.signature[index] for index in _PERM_NCW_TO_NWC)
        or not _valid_source(
            graph_index,
            source_contract,
            source_name,
            int(pre_index),
            public_inputs,
        )
        or not _same_unquantized_float((source_contract, output_contract))
    ):
        return None
    n, w, c = source_contract.shape
    n_signature, w_signature, c_signature = source_contract.signature
    return _LHSBranchPlan(
        kind="legacy_rank3",
        pre=pre,
        squeeze=None,
        source_name=source_name,
        source_contract=source_contract,
        output_name=name,
        output_contract=output_contract,
        n=int(n),
        w=int(w),
        c=int(c),
        n_signature=int(n_signature),
        w_signature=int(w_signature),
        c_signature=int(c_signature),
    )


def _resolve_lhs(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    output_name: str,
    add_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_LHSBranchPlan]:
    return _resolve_simple_lhs(
        model_ir,
        graph_index,
        output_name,
        add_index,
        public_inputs,
        public_outputs,
    ) or _resolve_legacy_lhs(
        model_ir,
        graph_index,
        output_name,
        add_index,
        public_inputs,
        public_outputs,
    )


def _plan_scale_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    operator_index: int,
    input_index: int,
    channel_count: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[Tuple[TensorIR, Optional[_ScaleConstantUpdate]]]:
    name = str(operator.inputs[int(input_index)])
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or name in public_inputs | public_outputs
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or str(tensor.dtype) not in {"FLOAT16", "FLOAT32"}
        or tensor.quantization is not None
    ):
        return None
    try:
        data = np.asarray(tensor.data)
        expected_dtype = np.float16 if str(tensor.dtype) == "FLOAT16" else np.float32
        if (
            data.dtype != np.dtype(expected_dtype)
            or tuple(int(value) for value in data.shape)
            not in {(int(channel_count), 1), (1, int(channel_count))}
            or tuple(int(value) for value in tensor.shape) != data.shape
            or (
                tensor.shape_signature is not None
                and tuple(int(value) for value in tensor.shape_signature) != data.shape
            )
        ):
            return None
    except (TypeError, ValueError):
        return None
    target = data.T.copy() if data.shape == (int(channel_count), 1) else data.copy()
    if np.array_equal(target, data) and target.shape == data.shape:
        return tensor, None

    clone_name: Optional[str] = None
    clone: Optional[TensorIR] = None
    if set(graph_index.consumer_indices(name)) != {int(operator_index)}:
        clone_name = _unique_tensor_name(model_ir, f"{name}_nhwc")
        clone = TensorIR(
            name=clone_name,
            dtype=str(tensor.dtype),
            shape=[int(value) for value in target.shape],
            shape_signature=[int(value) for value in target.shape],
            data=np.asarray(target),
            is_variable=False,
            quantization=None,
        )
    return tensor, _ScaleConstantUpdate(
        operator=operator,
        input_index=int(input_index),
        tensor=tensor,
        data=np.asarray(target),
        clone_name=clone_name,
        clone=clone,
    )


def _resolve_gate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    output_name: str,
    add_index: int,
    prefix_pre_by_reshape2: Dict[int, OperatorIR],
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_GateBranchPlan]:
    name = str(output_name)
    scale_index = graph_index.producers.get(name)
    if scale_index is None or name in graph_index.duplicate_producers:
        return None
    scale = model_ir.operators[int(scale_index)]
    if (
        str(scale.op_type) != "MUL"
        or len(scale.inputs) != 2
        or len(scale.outputs) != 1
        or str(scale.outputs[0]) != name
        or int(scale_index) >= int(add_index)
        or name in public_inputs | public_outputs
        or graph_index.consumer_indices(name) != [int(add_index)]
    ):
        return None

    constant_inputs = []
    for input_index, input_name in enumerate(scale.inputs):
        tensor = model_ir.tensors.get(str(input_name))
        if (
            tensor is not None
            and tensor.data is not None
            and str(input_name) not in graph_index.producers
            and str(input_name) not in graph_index.duplicate_producers
        ):
            try:
                if np.asarray(tensor.data).ndim == 2:
                    constant_inputs.append((int(input_index), str(input_name)))
            except Exception:
                return None
    if len(constant_inputs) != 1:
        return None
    scale_constant_input, _ = constant_inputs[0]
    gate_name = str(scale.inputs[1 - scale_constant_input])
    gate_index = graph_index.producers.get(gate_name)
    if gate_index is None or gate_name in graph_index.duplicate_producers:
        return None
    gate = model_ir.operators[int(gate_index)]
    if (
        str(gate.op_type) != "MUL"
        or len(gate.inputs) != 2
        or len(gate.outputs) != 1
        or str(gate.outputs[0]) != gate_name
        or int(gate_index) >= int(scale_index)
        or gate_name in public_inputs | public_outputs
        or graph_index.consumer_indices(gate_name) != [int(scale_index)]
    ):
        return None

    logistic_inputs = []
    for gate_input in gate.inputs:
        producer_index = graph_index.producers.get(str(gate_input))
        if (
            producer_index is not None
            and str(model_ir.operators[int(producer_index)].op_type) == "LOGISTIC"
        ):
            logistic_inputs.append((str(gate_input), int(producer_index)))
    if len(logistic_inputs) != 1:
        return None
    logistic_name, logistic_index = logistic_inputs[0]
    slice0_name = next(
        (str(value) for value in gate.inputs if str(value) != logistic_name),
        "",
    )
    slice0_index = graph_index.producers.get(slice0_name)
    logistic = model_ir.operators[int(logistic_index)]
    if slice0_index is None or slice0_name in graph_index.duplicate_producers:
        return None
    slice0 = model_ir.operators[int(slice0_index)]
    if (
        len(logistic.inputs) != 1
        or len(logistic.outputs) != 1
        or str(logistic.outputs[0]) != logistic_name
        or logistic_name in public_inputs | public_outputs
        or graph_index.consumer_indices(logistic_name) != [int(gate_index)]
        or str(slice0.op_type) != "SLICE"
        or len(slice0.inputs) != 3
        or len(slice0.outputs) != 1
        or str(slice0.outputs[0]) != slice0_name
        or slice0_name in public_inputs | public_outputs
        or graph_index.consumer_indices(slice0_name) != [int(gate_index)]
    ):
        return None

    slice1_name = str(logistic.inputs[0])
    slice1_index = graph_index.producers.get(slice1_name)
    if slice1_index is None or slice1_name in graph_index.duplicate_producers:
        return None
    slice1 = model_ir.operators[int(slice1_index)]
    if (
        str(slice1.op_type) != "SLICE"
        or len(slice1.inputs) != 3
        or len(slice1.outputs) != 1
        or str(slice1.outputs[0]) != slice1_name
        or slice1_name in public_inputs | public_outputs
        or graph_index.consumer_indices(slice1_name) != [int(logistic_index)]
        or str(slice0.inputs[0]) != str(slice1.inputs[0])
    ):
        return None
    split_source_name = str(slice0.inputs[0])
    reshape2_index = graph_index.producers.get(split_source_name)
    if (
        reshape2_index is None
        or split_source_name in graph_index.duplicate_producers
        or Counter(graph_index.consumer_indices(split_source_name))
        != Counter((int(slice0_index), int(slice1_index)))
    ):
        return None
    reshape2 = model_ir.operators[int(reshape2_index)]
    prefix_pre = prefix_pre_by_reshape2.get(id(reshape2))
    if prefix_pre is None:
        return None
    prefix_pre_index = graph_index.operator_index(prefix_pre)
    if prefix_pre_index is None:
        return None
    prefix = _resolve_flattened_instance_norm_prefix(
        model_ir,
        graph_index,
        int(prefix_pre_index),
    )
    if prefix is None or prefix.reshape2 is not reshape2:
        return None
    if prefix.c % 2 != 0:
        return None
    n, c2, w = prefix.reshape2_contract.shape
    n_signature, c2_signature, w_signature = prefix.reshape2_contract.signature
    if c2_signature == -1 or (n_signature, w_signature).count(-1) > 1:
        return None
    c = int(c2 // 2)
    c_signature = int(c)
    old_shape = (int(n), int(c), int(w))
    old_signature = (
        int(n_signature),
        int(c_signature),
        int(w_signature),
    )
    slice0_contract = _data_contract(model_ir, slice0_name, 3)
    slice1_contract = _data_contract(model_ir, slice1_name, 3)
    logistic_contract = _data_contract(model_ir, logistic_name, 3)
    gate_contract = _data_contract(model_ir, gate_name, 3)
    scale_contract = _data_contract(model_ir, name, 3)
    contracts = (
        slice0_contract,
        slice1_contract,
        logistic_contract,
        gate_contract,
        scale_contract,
    )
    if (
        any(contract is None for contract in contracts)
        or any(contract.shape != old_shape for contract in contracts if contract)
        or any(
            contract.signature != old_signature
            for contract in contracts
            if contract
        )
        or not _same_unquantized_float(
            prefix.data_contracts
            + tuple(contract for contract in contracts if contract is not None)
        )
        or _constant_vector(
            model_ir,
            graph_index,
            str(slice0.inputs[1]),
            3,
            public_inputs,
        )
        != (0, 0, 0)
        or _constant_vector(
            model_ir,
            graph_index,
            str(slice0.inputs[2]),
            3,
            public_inputs,
        )
        != old_shape
        or _constant_vector(
            model_ir,
            graph_index,
            str(slice1.inputs[1]),
            3,
            public_inputs,
        )
        != (0, c, 0)
        or _constant_vector(
            model_ir,
            graph_index,
            str(slice1.inputs[2]),
            3,
            public_inputs,
        )
        != old_shape
    ):
        return None
    scale_update_result = _plan_scale_constant_update(
        model_ir,
        graph_index,
        scale,
        int(scale_index),
        scale_constant_input,
        c,
        public_inputs,
        public_outputs,
    )
    if scale_update_result is None:
        return None
    scale_tensor, scale_update = scale_update_result
    if str(scale_tensor.dtype) != str(scale_contract.tensor.dtype):
        return None
    if not (
        int(reshape2_index) < int(slice0_index) < int(gate_index)
        and int(reshape2_index) < int(slice1_index) < int(logistic_index)
        and int(logistic_index) < int(gate_index) < int(scale_index) < int(add_index)
    ):
        return None
    data_names = prefix.data_names + (
        slice0_name,
        slice1_name,
        logistic_name,
        gate_name,
        name,
    )
    if len(data_names) != len(set(data_names)):
        return None
    return _GateBranchPlan(
        prefix=prefix,
        slice0=slice0,
        slice1=slice1,
        logistic=logistic,
        gate=gate,
        scale=scale,
        slice0_contract=slice0_contract,
        slice1_contract=slice1_contract,
        logistic_contract=logistic_contract,
        gate_contract=gate_contract,
        scale_contract=scale_contract,
        scale_constant_update=scale_update,
        n=int(n),
        w=int(w),
        c=int(c),
        c2=int(c2),
        n_signature=int(n_signature),
        w_signature=int(w_signature),
        c_signature=int(c_signature),
        c2_signature=int(c2_signature),
    )


def _plan_int_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    operator_index: int,
    input_index: int,
    current: Tuple[int, ...],
    target: Tuple[int, ...],
    suffix: str,
    public_outputs: set[str],
) -> Tuple[bool, Optional[_ConstantUpdate]]:
    if current == target:
        return True, None
    update = _plan_constant_update(
        model_ir,
        graph_index,
        operator,
        operator_index,
        input_index,
        target,
        suffix,
        public_outputs,
    )
    return update is not None, update


def _reserve_planned_name(base: str, reserved: set[str]) -> str:
    name = str(base)
    serial = 1
    while name in reserved:
        name = f"{base}_{serial}"
        serial += 1
    reserved.add(name)
    return name


def _reserve_constant_clone_names(
    model_ir: ModelIR,
    updates: Tuple[_ConstantUpdate, ...],
) -> Tuple[Tuple[_ConstantUpdate, ...], set[str]]:
    reserved = set(model_ir.tensors)
    normalized = []
    for update in updates:
        if update.clone_name is None:
            normalized.append(update)
            continue
        if update.clone is None:
            return (), set()
        clone_name = _reserve_planned_name(update.clone_name, reserved)
        update.clone.name = clone_name
        normalized.append(
            _ConstantUpdate(
                operator=update.operator,
                input_index=update.input_index,
                tensor=update.tensor,
                data=update.data,
                clone_name=clone_name,
                clone=update.clone,
            )
        )
    return tuple(normalized), reserved


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    post_index: int,
    prefix_pre_by_reshape2: Dict[int, OperatorIR],
) -> Optional[_TencoderRewritePlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    post = model_ir.operators[int(post_index)]
    if (
        len(post.inputs) != 2
        or len(post.outputs) != 1
        or _constant_vector(
            model_ir,
            graph_index,
            str(post.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NCHW_TO_NHWC
    ):
        return None
    post_input_name = str(post.inputs[0])
    post_output_name = str(post.outputs[0])
    if (
        not post_output_name
        or post_output_name in public_inputs
        or not _producer_is_valid(graph_index, post_output_name, post_index)
    ):
        return None
    post_contract = _data_contract(model_ir, post_output_name, 4)
    post_users = graph_index.consumer_indices(post_output_name)
    if (
        post_contract is None
        or not post_users
        or any(int(index) <= int(post_index) for index in post_users)
        or any(
            str(model_ir.operators[int(index)].op_type) != "CONV_2D"
            for index in post_users
        )
    ):
        return None

    expand_index = graph_index.producers.get(post_input_name)
    if expand_index is None or post_input_name in graph_index.duplicate_producers:
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
        or str(expand.outputs[0]) != post_input_name
        or expand_axis not in {(2,), (-2,)}
        or int(expand_index) >= int(post_index)
        or post_input_name in public_inputs | public_outputs
        or graph_index.consumer_indices(post_input_name) != [int(post_index)]
    ):
        return None
    add_output_name = str(expand.inputs[0])
    add_index = graph_index.producers.get(add_output_name)
    if add_index is None or add_output_name in graph_index.duplicate_producers:
        return None
    add = model_ir.operators[int(add_index)]
    add_users = graph_index.consumer_indices(add_output_name)
    if (
        str(add.op_type) != "ADD"
        or len(add.inputs) != 2
        or len(add.outputs) != 1
        or str(add.outputs[0]) != add_output_name
        or int(add_index) >= int(expand_index)
        or add_output_name in public_inputs | public_outputs
        or Counter(add_users)[int(expand_index)] != 1
    ):
        return None
    side_indices = tuple(
        sorted({int(index) for index in add_users if int(index) != int(expand_index)})
    )
    if (
        any(index <= int(add_index) for index in side_indices)
        or (not side_indices and post_output_name in public_outputs)
    ):
        return None

    lhs = _resolve_lhs(
        model_ir,
        graph_index,
        str(add.inputs[0]),
        int(add_index),
        public_inputs,
        public_outputs,
    )
    rhs = _resolve_gate(
        model_ir,
        graph_index,
        str(add.inputs[1]),
        int(add_index),
        prefix_pre_by_reshape2,
        public_inputs,
        public_outputs,
    )
    if lhs is None or rhs is None:
        lhs = _resolve_lhs(
            model_ir,
            graph_index,
            str(add.inputs[1]),
            int(add_index),
            public_inputs,
            public_outputs,
        )
        rhs = _resolve_gate(
            model_ir,
            graph_index,
            str(add.inputs[0]),
            int(add_index),
            prefix_pre_by_reshape2,
            public_inputs,
            public_outputs,
        )
    if (
        lhs is None
        or rhs is None
        or (lhs.n, lhs.w, lhs.c) != (rhs.n, rhs.w, rhs.c)
        or (
            lhs.n_signature,
            lhs.w_signature,
            lhs.c_signature,
        )
        != (
            rhs.n_signature,
            rhs.w_signature,
            rhs.c_signature,
        )
    ):
        return None

    n, w, c = lhs.n, lhs.w, lhs.c
    n_signature = lhs.n_signature
    w_signature = lhs.w_signature
    c_signature = lhs.c_signature
    old_rank3_signature = (n_signature, c_signature, w_signature)
    old_rank4_signature = (n_signature, c_signature, 1, w_signature)
    output_signature = (n_signature, 1, w_signature, c_signature)
    add_contract = _data_contract(model_ir, add_output_name, 3)
    expand_contract = _data_contract(model_ir, post_input_name, 4)
    if (
        add_contract is None
        or expand_contract is None
        or add_contract.shape != (n, c, w)
        or add_contract.signature != old_rank3_signature
        or expand_contract.shape != (n, c, 1, w)
        or expand_contract.signature != old_rank4_signature
        or post_contract.shape != (n, 1, w, c)
        or post_contract.signature != output_signature
        or not _same_unquantized_float(
            (
                lhs.output_contract,
                rhs.scale_contract,
                add_contract,
                expand_contract,
                post_contract,
            )
        )
    ):
        return None

    reshape2_target = (n_signature, w_signature, rhs.c2_signature)
    if reshape2_target == (n, w, rhs.c2):
        reshape2_target = (n, w, rhs.c2)
    slice_size_target = (n_signature, w_signature, c_signature)
    if slice_size_target == (n, w, c):
        slice_size_target = (n, w, c)
    int_updates = []
    requests = (
        (
            rhs.prefix.reshape2,
            rhs.prefix.ordered_indices[-1],
            1,
            (n, rhs.c2, w),
            reshape2_target,
            "nhwc_shape",
        ),
        (
            rhs.slice0,
            graph_index.operator_index(rhs.slice0),
            2,
            (n, c, w),
            slice_size_target,
            "nhwc_size",
        ),
        (
            rhs.slice1,
            graph_index.operator_index(rhs.slice1),
            1,
            (0, c, 0),
            (0, 0, c),
            "nhwc_begin",
        ),
        (
            rhs.slice1,
            graph_index.operator_index(rhs.slice1),
            2,
            (n, c, w),
            slice_size_target,
            "nhwc_size",
        ),
        (
            expand,
            int(expand_index),
            1,
            expand_axis,
            (1,),
            "nhwc_axis",
        ),
    )
    for operator, operator_index, input_index, current, target, suffix in requests:
        if operator_index is None:
            return None
        ok, update = _plan_int_update(
            model_ir,
            graph_index,
            operator,
            int(operator_index),
            int(input_index),
            tuple(int(value) for value in current),
            tuple(int(value) for value in target),
            str(suffix),
            public_outputs,
        )
        if not ok:
            return None
        if update is not None:
            int_updates.append(update)

    int_updates_tuple, reserved_names = _reserve_constant_clone_names(
        model_ir,
        tuple(int_updates),
    )
    if not reserved_names:
        return None
    scale_update = rhs.scale_constant_update
    if scale_update is not None and scale_update.clone_name is not None:
        if scale_update.clone is None:
            return None
        clone_name = _reserve_planned_name(
            scale_update.clone_name,
            reserved_names,
        )
        scale_update.clone.name = clone_name
        scale_update = _ScaleConstantUpdate(
            operator=scale_update.operator,
            input_index=scale_update.input_index,
            tensor=scale_update.tensor,
            data=scale_update.data,
            clone_name=clone_name,
            clone=scale_update.clone,
        )
        rhs = replace(rhs, scale_constant_update=scale_update)

    rank3_shape = (n, w, c)
    rank3_signature = (n_signature, w_signature, c_signature)
    rhs_rank3_shape = (n, w, rhs.c2)
    rhs_rank3_signature = (n_signature, w_signature, rhs.c2_signature)
    rank4_shape = (n, 1, w, c)
    rank4_signature = (n_signature, 1, w_signature, c_signature)
    metadata_updates = [
        _TensorMetadataUpdate(
            rhs.prefix.squeeze_contract.tensor,
            rhs_rank3_shape,
            rhs_rank3_signature,
        ),
        _TensorMetadataUpdate(
            rhs.prefix.reshape2_contract.tensor,
            rhs_rank3_shape,
            rhs_rank3_signature,
        ),
        *(
            _TensorMetadataUpdate(
                contract.tensor,
                rank3_shape,
                rank3_signature,
            )
            for contract in (
                rhs.slice0_contract,
                rhs.slice1_contract,
                rhs.logistic_contract,
                rhs.gate_contract,
                rhs.scale_contract,
                add_contract,
            )
        ),
        _TensorMetadataUpdate(
            expand_contract.tensor,
            rank4_shape,
            rank4_signature,
        ),
    ]
    lhs_squeeze_options = None
    if lhs.squeeze is not None:
        lhs_squeeze_options = (
            dict(lhs.squeeze.options) if isinstance(lhs.squeeze.options, dict) else {}
        )
        lhs_squeeze_options["squeezeDims"] = [1]
        metadata_updates.append(
            _TensorMetadataUpdate(
                lhs.output_contract.tensor,
                rank3_shape,
                rank3_signature,
            )
        )
    rhs_squeeze_options = (
        dict(rhs.prefix.squeeze.options)
        if isinstance(rhs.prefix.squeeze.options, dict)
        else {}
    )
    rhs_squeeze_options["squeezeDims"] = [1]
    reshape2_options = (
        dict(rhs.prefix.reshape2.options)
        if isinstance(rhs.prefix.reshape2.options, dict)
        else {}
    )
    for key in ("newShape", "onnxRawNewShape"):
        if isinstance(reshape2_options.get(key), list):
            reshape2_options[key] = list(rhs_rank3_signature)
    if rhs_rank3_signature != rhs_rank3_shape:
        reshape2_options["newShape"] = list(rhs_rank3_signature)
        if "onnxRawNewShape" in reshape2_options:
            reshape2_options["onnxRawNewShape"] = list(rhs_rank3_signature)
        reshape2_options["preserveDynamicShape"] = True

    side_users = tuple(model_ir.operators[index] for index in side_indices)
    bridge_permutation = None
    bridge_tensor = None
    bridge = None
    if side_users:
        permutation_name = _reserve_planned_name(
            f"{add_output_name}_legacy_ncw_perm",
            reserved_names,
        )
        bridge_name = _reserve_planned_name(
            f"{add_output_name}_legacy_ncw",
            reserved_names,
        )
        try:
            bridge_quantization = _clone_quantization(add_contract.tensor.quantization)
        except Exception:
            return None
        bridge_permutation = TensorIR(
            name=permutation_name,
            dtype="INT32",
            shape=[3],
            shape_signature=[3],
            data=np.asarray(_PERM_NCW_TO_NWC, dtype=np.int32),
            is_variable=False,
        )
        bridge_tensor = TensorIR(
            name=bridge_name,
            dtype=str(add_contract.tensor.dtype),
            shape=[n, c, w],
            shape_signature=list(old_rank3_signature),
            quantization=bridge_quantization,
        )
        bridge = OperatorIR(
            op_type="TRANSPOSE",
            inputs=[add_output_name, permutation_name],
            outputs=[bridge_name],
            options={},
        )

    target_ops = [
        lhs.pre,
        rhs.prefix.pre,
        rhs.prefix.squeeze,
        rhs.prefix.reshape2,
        rhs.slice0,
        rhs.slice1,
        rhs.logistic,
        rhs.gate,
        rhs.scale,
        add,
        expand,
        post,
        *side_users,
    ]
    if lhs.squeeze is not None:
        target_ops.append(lhs.squeeze)
    target_ops = list({id(operator): operator for operator in target_ops}.values())
    ordered_ops = tuple(
        sorted(
            target_ops,
            key=lambda operator: int(graph_index.operator_index(operator) or -1),
        )
    )
    if any(graph_index.operator_index(operator) is None for operator in ordered_ops):
        return None
    removed_ops = (lhs.pre, rhs.prefix.pre, post)
    if len({id(operator) for operator in removed_ops}) != len(removed_ops):
        return None
    return _TencoderRewritePlan(
        ordered_ops=ordered_ops,
        removed_ops=removed_ops,
        lhs=lhs,
        rhs=rhs,
        add=add,
        add_contract=add_contract,
        expand=expand,
        expand_contract=expand_contract,
        post=post,
        post_output_name=post_output_name,
        int_updates=int_updates_tuple,
        metadata_updates=tuple(metadata_updates),
        rhs_squeeze_options=rhs_squeeze_options,
        lhs_squeeze_options=lhs_squeeze_options,
        reshape2_options=reshape2_options,
        side_users=side_users,
        bridge_permutation=bridge_permutation,
        bridge_tensor=bridge_tensor,
        bridge=bridge,
    )


def _apply_scale_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    update: _ScaleConstantUpdate,
) -> None:
    if update.clone_name is not None:
        model_ir.tensors[update.clone_name] = update.clone
        _replace_operator_input_at(
            model_ir=model_ir,
            op=update.operator,
            input_index=update.input_index,
            new_input_name=update.clone_name,
            graph_index=graph_index,
        )
        return
    update.tensor.data = np.asarray(update.data)
    update.tensor.shape = [int(value) for value in update.data.shape]
    update.tensor.shape_signature = [int(value) for value in update.data.shape]


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _TencoderRewritePlan,
) -> bool:
    indices = [graph_index.operator_index(operator) for operator in plan.ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != len(resolved):
        return False
    clone_names = {
        update.clone_name
        for update in plan.int_updates
        if update.clone_name is not None
    }
    scale_update = plan.rhs.scale_constant_update
    if scale_update is not None and scale_update.clone_name is not None:
        clone_names.add(scale_update.clone_name)
    if any(name in model_ir.tensors for name in clone_names):
        return False
    if (
        plan.bridge_permutation is not None
        and plan.bridge_permutation.name in model_ir.tensors
    ) or (plan.bridge_tensor is not None and plan.bridge_tensor.name in model_ir.tensors):
        return False

    for update in plan.int_updates:
        if not _apply_constant_update(model_ir, graph_index, update):
            return False
    if scale_update is not None:
        _apply_scale_update(model_ir, graph_index, scale_update)

    if plan.lhs.squeeze is not None:
        plan.lhs.squeeze.options = dict(plan.lhs_squeeze_options or {})
        _set_operator_inputs(
            model_ir=model_ir,
            op=plan.lhs.squeeze,
            new_inputs=[plan.lhs.source_name],
            graph_index=graph_index,
        )
    else:
        for input_index, input_name in enumerate(plan.add.inputs):
            if str(input_name) == plan.lhs.output_name:
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=plan.add,
                    input_index=int(input_index),
                    new_input_name=plan.lhs.source_name,
                    graph_index=graph_index,
                )

    plan.rhs.prefix.squeeze.options = dict(plan.rhs_squeeze_options)
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.rhs.prefix.squeeze,
        new_inputs=[plan.rhs.prefix.source_name],
        graph_index=graph_index,
    )
    plan.rhs.prefix.reshape2.options = dict(plan.reshape2_options)
    for update in plan.metadata_updates:
        update.tensor.shape = list(update.shape)
        update.tensor.shape_signature = list(update.signature)

    if plan.bridge is not None:
        if plan.bridge_permutation is None or plan.bridge_tensor is None:
            return False
        model_ir.tensors[plan.bridge_permutation.name] = plan.bridge_permutation
        model_ir.tensors[plan.bridge_tensor.name] = plan.bridge_tensor
        side_indices = [
            graph_index.operator_index(operator) for operator in plan.side_users
        ]
        if any(index is None for index in side_indices):
            return False
        graph_index.insert_operator(
            min(int(index) for index in side_indices if index is not None),
            plan.bridge,
        )
        for user in plan.side_users:
            for input_index, input_name in enumerate(list(user.inputs)):
                if str(input_name) == str(plan.add.outputs[0]):
                    _replace_operator_input_at(
                        model_ir=model_ir,
                        op=user,
                        input_index=int(input_index),
                        new_input_name=plan.bridge_tensor.name,
                        graph_index=graph_index,
                    )

    _replace_tensor_inputs(
        model_ir,
        plan.post_output_name,
        str(plan.expand.outputs[0]),
        graph_index=graph_index,
    )
    model_ir.outputs = [
        str(plan.expand.outputs[0])
        if str(output_name) == plan.post_output_name
        else str(output_name)
        for output_name in model_ir.outputs
    ]
    remove_indices = [
        graph_index.operator_index(operator) for operator in plan.removed_ops
    ]
    if any(index is None for index in remove_indices):
        return False
    graph_index.remove_operators(
        int(index) for index in remove_indices if index is not None
    )
    return True


def _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    required = {
        "TRANSPOSE": 3,
        "SQUEEZE": 1,
        "RESHAPE": 2,
        "MEAN": 2,
        "SLICE": 2,
        "LOGISTIC": 1,
        "MUL": 5,
        "ADD": 3,
        "SQRT": 1,
        "DIV": 1,
        "EXPAND_DIMS": 1,
        "CONV_2D": 1,
    }
    counts = Counter(str(operator.op_type) for operator in model_ir.operators)
    if any(counts[op_type] < count for op_type, count in required.items()):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {_STATS_KEY: 0}

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    prefix_pre_by_reshape2: Dict[int, OperatorIR] = {}
    for pre_index in active_index.operator_indices("TRANSPOSE"):
        prefix = _resolve_flattened_instance_norm_prefix(
            model_ir,
            active_index,
            int(pre_index),
        )
        if prefix is not None:
            prefix_pre_by_reshape2[id(prefix.reshape2)] = prefix.pre
    candidates = [
        model_ir.operators[index]
        for index in active_index.operator_indices("TRANSPOSE")
    ]
    rewritten = 0
    for post in candidates:
        post_index = active_index.operator_index(post)
        if post_index is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            int(post_index),
            prefix_pre_by_reshape2,
        )
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
