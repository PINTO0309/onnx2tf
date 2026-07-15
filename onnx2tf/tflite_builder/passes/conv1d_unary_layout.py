from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Any, Dict, FrozenSet, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_per_tensor_quantization,
    _is_same_per_tensor_quantization,
    _normalize_squeeze_axes_for_rank,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_STATS_KEY = (
    "optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_chains"
)
_RANK4_STATS_KEY = (
    "optimized_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains"
)
_FANOUT_STATS_KEY = (
    "optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains"
)
_PERM_NHCW_FROM_NCHW = (0, 2, 1, 3)
_PERM_HCW_TO_HWC = (0, 2, 1)
_UNARY_OPS = {
    "RELU",
    "RELU6",
    "RELU_0_TO_1",
    "LEAKY_RELU",
    "LOGISTIC",
    "TANH",
    "GELU",
    "ABS",
    "NEG",
    "SQRT",
    "EXP",
    "CAST",
    "FLOOR",
    "CEIL",
    "ROUND",
    "HARD_SWISH",
}


@dataclass(frozen=True)
class _TensorContract:
    tensor: TensorIR
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _UnaryPrefixPlan:
    public_inputs: FrozenSet[str]
    public_outputs: FrozenSet[str]
    pre: OperatorIR
    squeeze: OperatorIR
    unary: OperatorIR
    source_name: str
    pre_output_name: str
    squeeze_output_name: str
    unary_output_name: str
    source_contract: _TensorContract
    pre_contract: _TensorContract
    squeeze_contract: _TensorContract
    unary_contract: _TensorContract
    squeeze_axis: int
    unary_index: int


@dataclass(frozen=True)
class _RewritePlan:
    pre: OperatorIR
    squeeze: OperatorIR
    unary: OperatorIR
    expand: OperatorIR
    post: OperatorIR
    source_name: str
    output_name: str
    output_tensor: TensorIR
    output_dtype: str
    output_quantization: Any


@dataclass(frozen=True)
class _ConstantUpdate:
    operator: OperatorIR
    input_index: int
    tensor: TensorIR
    data: np.ndarray
    clone_name: Optional[str]
    clone: Optional[TensorIR]


@dataclass(frozen=True)
class _Rank4RewritePlan:
    pre: OperatorIR
    unary: OperatorIR
    unary_tensor: TensorIR
    mid: OperatorIR
    reshape: OperatorIR
    reshape_tensor: TensorIR
    reshape_options: Optional[Dict[str, Any]]
    reshape_shape_update: _ConstantUpdate
    expand: OperatorIR
    expand_axis_update: _ConstantUpdate
    expand_tensor: TensorIR
    expand_shape: Tuple[int, ...]
    expand_signature: Tuple[int, ...]
    expand_dtype: str
    expand_quantization: Any
    post: OperatorIR
    post_output_name: str
    source_name: str
    non_expand_users: Tuple[OperatorIR, ...]
    bridge_permutation: Optional[TensorIR]
    bridge_tensor: Optional[TensorIR]
    bridge: Optional[OperatorIR]


@dataclass(frozen=True)
class _FanoutRewritePlan:
    pre: OperatorIR
    squeeze: OperatorIR
    unary: OperatorIR
    expand: OperatorIR
    post: OperatorIR
    source_name: str
    unary_output_name: str
    post_output_name: str
    pre_output_tensor: TensorIR
    post_output_tensor: TensorIR
    output_dtype: str
    pre_output_quantization: Any
    post_output_quantization: Any


def _tensor_contract(
    model_ir: ModelIR,
    tensor_name: str,
    rank: int,
) -> Optional[_TensorContract]:
    tensor = model_ir.tensors.get(str(tensor_name))
    if tensor is None:
        return None
    try:
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
    except (TypeError, ValueError):
        return None
    if len(shape) != int(rank) or len(signature) != int(rank):
        return None
    if any(int(value) <= 0 for value in shape) or any(
        int(signature_value) == 0
        or int(signature_value) < -1
        or (int(signature_value) > 0 and int(signature_value) != int(shape_value))
        for shape_value, signature_value in zip(shape, signature)
    ):
        return None
    return _TensorContract(
        tensor=tensor,
        shape=shape,
        signature=signature,
    )


def _constant_vector(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    size: int,
    public_inputs: AbstractSet[str],
) -> Optional[Tuple[int, ...]]:
    name = str(tensor_name)
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or name in public_inputs
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or str(tensor.dtype) not in {"INT32", "INT64"}
        or list(tensor.shape) != [int(size)]
        or (
            tensor.shape_signature is not None
            and [int(value) for value in tensor.shape_signature] != [int(size)]
        )
    ):
        return None
    try:
        data = np.asarray(tensor.data)
        expected_dtype = np.int32 if str(tensor.dtype) == "INT32" else np.int64
        if data.dtype != np.dtype(expected_dtype) or data.size != int(size):
            return None
        return tuple(int(value) for value in data.reshape(-1).tolist())
    except Exception:
        return None


def _producer_is_valid(
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    expected_index: int,
) -> bool:
    name = str(tensor_name)
    return name not in graph_index.duplicate_producers and graph_index.producers.get(
        name
    ) == int(expected_index)


def _quantization_contract(contracts: Tuple[_TensorContract, ...]) -> bool:
    quantizations = [contract.tensor.quantization for contract in contracts]
    if all(quantization is None for quantization in quantizations):
        return True
    if not all(
        _is_per_tensor_quantization(quantization)
        for quantization in quantizations
    ):
        return False
    anchor = quantizations[0]
    return all(
        _is_same_per_tensor_quantization(anchor, quantization)
        for quantization in quantizations[1:]
    )


def _squeeze_axis(
    squeeze: OperatorIR,
    pre_contract: _TensorContract,
    squeeze_contract: _TensorContract,
) -> Optional[int]:
    options = dict(squeeze.options) if isinstance(squeeze.options, dict) else {}
    if "squeezeDims" in options:
        try:
            raw_axes = np.asarray(options.get("squeezeDims", []), dtype=np.int64)
            normalized_axes = _normalize_squeeze_axes_for_rank(
                [int(value) for value in raw_axes.reshape(-1).tolist()],
                4,
            )
        except Exception:
            return None
        if normalized_axes is None or len(normalized_axes) != 1:
            return None
        candidates = [int(normalized_axes[0])]
    else:
        candidates = [
            axis
            for axis in range(4)
            if int(pre_contract.shape[axis]) == 1
            and int(pre_contract.signature[axis]) == 1
        ]

    matches = []
    for axis in candidates:
        if (
            int(pre_contract.shape[axis]) != 1
            or int(pre_contract.signature[axis]) != 1
        ):
            continue
        expected_shape = tuple(
            value
            for index, value in enumerate(pre_contract.shape)
            if int(index) != int(axis)
        )
        expected_signature = tuple(
            value
            for index, value in enumerate(pre_contract.signature)
            if int(index) != int(axis)
        )
        if (
            squeeze_contract.shape == expected_shape
            and squeeze_contract.signature == expected_signature
        ):
            matches.append(int(axis))
    if len(matches) != 1:
        return None
    return int(matches[0])


def _resolve_unary_prefix_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
    *,
    allow_public_unary_output: bool,
) -> Optional[_UnaryPrefixPlan]:
    pre = model_ir.operators[int(pre_index)]
    if len(pre.inputs) != 2 or len(pre.outputs) != 1:
        return None
    public_inputs = frozenset(str(value) for value in model_ir.inputs)
    public_outputs = frozenset(str(value) for value in model_ir.outputs)
    if (
        _constant_vector(
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
    if source_contract is None or pre_contract is None:
        return None
    if (
        pre_contract.shape
        != tuple(source_contract.shape[index] for index in _PERM_NHWC_TO_NCHW)
        or pre_contract.signature
        != tuple(source_contract.signature[index] for index in _PERM_NHWC_TO_NCHW)
    ):
        return None

    source_producer = graph_index.producers.get(source_name)
    if source_name in graph_index.duplicate_producers:
        return None
    if source_producer is not None:
        if source_name in public_inputs or int(source_producer) >= int(pre_index):
            return None
    elif source_name not in public_inputs and source_contract.tensor.data is None:
        return None

    pre_users = graph_index.consumer_indices(pre_output_name)
    if len(pre_users) != 1:
        return None
    squeeze_index = int(pre_users[0])
    if squeeze_index <= int(pre_index):
        return None
    squeeze = model_ir.operators[squeeze_index]
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or len(squeeze.outputs) != 1
        or str(squeeze.inputs[0]) != pre_output_name
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
    if squeeze_contract is None:
        return None
    squeeze_axis = _squeeze_axis(squeeze, pre_contract, squeeze_contract)
    if squeeze_axis is None:
        return None

    squeeze_users = graph_index.consumer_indices(squeeze_output_name)
    if len(squeeze_users) != 1:
        return None
    unary_index = int(squeeze_users[0])
    if unary_index <= squeeze_index:
        return None
    unary = model_ir.operators[unary_index]
    if (
        str(unary.op_type) not in _UNARY_OPS
        or len(unary.inputs) != 1
        or len(unary.outputs) != 1
        or str(unary.inputs[0]) != squeeze_output_name
    ):
        return None
    unary_output_name = str(unary.outputs[0])
    if (
        not unary_output_name
        or unary_output_name in public_inputs
        or (
            not allow_public_unary_output
            and unary_output_name in public_outputs
        )
        or not _producer_is_valid(graph_index, unary_output_name, unary_index)
    ):
        return None
    unary_contract = _tensor_contract(model_ir, unary_output_name, 3)
    if (
        unary_contract is None
        or unary_contract.shape != squeeze_contract.shape
        or unary_contract.signature != squeeze_contract.signature
    ):
        return None

    return _UnaryPrefixPlan(
        public_inputs=public_inputs,
        public_outputs=public_outputs,
        pre=pre,
        squeeze=squeeze,
        unary=unary,
        source_name=source_name,
        pre_output_name=pre_output_name,
        squeeze_output_name=squeeze_output_name,
        unary_output_name=unary_output_name,
        source_contract=source_contract,
        pre_contract=pre_contract,
        squeeze_contract=squeeze_contract,
        unary_contract=unary_contract,
        squeeze_axis=int(squeeze_axis),
        unary_index=unary_index,
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_RewritePlan]:
    prefix = _resolve_unary_prefix_candidate(
        model_ir,
        graph_index,
        pre_index,
        allow_public_unary_output=False,
    )
    if prefix is None:
        return None
    public_inputs = prefix.public_inputs
    public_outputs = prefix.public_outputs
    pre = prefix.pre
    squeeze = prefix.squeeze
    unary = prefix.unary
    source_name = prefix.source_name
    pre_output_name = prefix.pre_output_name
    squeeze_output_name = prefix.squeeze_output_name
    unary_output_name = prefix.unary_output_name
    source_contract = prefix.source_contract
    pre_contract = prefix.pre_contract
    squeeze_contract = prefix.squeeze_contract
    unary_contract = prefix.unary_contract
    squeeze_axis = prefix.squeeze_axis
    unary_index = prefix.unary_index
    unary_users = graph_index.consumer_indices(unary_output_name)
    if len(unary_users) != 1:
        return None
    expand_index = int(unary_users[0])
    if expand_index <= unary_index:
        return None
    expand = model_ir.operators[expand_index]
    if (
        str(expand.op_type) != "EXPAND_DIMS"
        or len(expand.inputs) != 2
        or len(expand.outputs) != 1
        or str(expand.inputs[0]) != unary_output_name
    ):
        return None
    expand_axis = _constant_vector(
        model_ir,
        graph_index,
        str(expand.inputs[1]),
        1,
        public_inputs,
    )
    if expand_axis is None:
        return None
    normalized_expand_axis = int(expand_axis[0])
    if normalized_expand_axis < 0:
        normalized_expand_axis += 4
    if normalized_expand_axis != int(squeeze_axis):
        return None

    expand_output_name = str(expand.outputs[0])
    if (
        not expand_output_name
        or expand_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, expand_output_name, expand_index)
    ):
        return None
    expand_contract = _tensor_contract(model_ir, expand_output_name, 4)
    if (
        expand_contract is None
        or expand_contract.shape != pre_contract.shape
        or expand_contract.signature != pre_contract.signature
    ):
        return None

    expand_users = graph_index.consumer_indices(expand_output_name)
    if len(expand_users) != 1:
        return None
    post_index = int(expand_users[0])
    if post_index <= expand_index:
        return None
    post = model_ir.operators[post_index]
    if (
        str(post.op_type) != "TRANSPOSE"
        or len(post.inputs) != 2
        or len(post.outputs) != 1
        or str(post.inputs[0]) != expand_output_name
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
    output_name = str(post.outputs[0])
    if (
        not output_name
        or output_name in public_inputs
        or not _producer_is_valid(graph_index, output_name, post_index)
    ):
        return None
    output_contract = _tensor_contract(model_ir, output_name, 4)
    if (
        output_contract is None
        or output_contract.shape != source_contract.shape
        or output_contract.signature != source_contract.signature
    ):
        return None
    if any(
        int(consumer_index) <= int(post_index)
        for consumer_index in graph_index.consumer_indices(output_name)
    ):
        return None

    data_names = (
        source_name,
        pre_output_name,
        squeeze_output_name,
        unary_output_name,
        expand_output_name,
        output_name,
    )
    if len(set(data_names)) != len(data_names):
        return None

    input_group = (source_contract, pre_contract, squeeze_contract)
    output_group = (unary_contract, expand_contract)
    if (
        len({str(contract.tensor.dtype) for contract in input_group}) != 1
        or len({str(contract.tensor.dtype) for contract in output_group}) != 1
        or not _quantization_contract(input_group)
        or not _quantization_contract(output_group)
    ):
        return None
    if str(unary.op_type) != "CAST" and (
        str(source_contract.tensor.dtype) != str(unary_contract.tensor.dtype)
        or not _quantization_contract(input_group + output_group)
    ):
        return None

    try:
        output_quantization = _clone_quantization(
            unary_contract.tensor.quantization
        )
    except Exception:
        return None
    return _RewritePlan(
        pre=pre,
        squeeze=squeeze,
        unary=unary,
        expand=expand,
        post=post,
        source_name=source_name,
        output_name=output_name,
        output_tensor=output_contract.tensor,
        output_dtype=str(unary_contract.tensor.dtype),
        output_quantization=output_quantization,
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _RewritePlan,
) -> bool:
    indices = [
        graph_index.operator_index(operator)
        for operator in (
            plan.pre,
            plan.squeeze,
            plan.unary,
            plan.expand,
            plan.post,
        )
    ]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != 5:
        return False

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.unary,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.unary,
        new_outputs=[plan.output_name],
        graph_index=graph_index,
    )
    plan.output_tensor.dtype = str(plan.output_dtype)
    plan.output_tensor.quantization = plan.output_quantization
    graph_index.remove_operators(
        [resolved[index] for index in (0, 1, 3, 4)]
    )
    return True


def _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    required = {"TRANSPOSE": 2, "SQUEEZE": 1, "EXPAND_DIMS": 1}
    has_unary = False
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required and required[op_type] > 0:
            required[op_type] -= 1
        if op_type in _UNARY_OPS:
            has_unary = True
    if not has_unary or any(count > 0 for count in required.values()):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
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
        plan = _resolve_candidate(model_ir, active_index, pre_index)
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}


def _resolve_fanout_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_FanoutRewritePlan]:
    prefix = _resolve_unary_prefix_candidate(
        model_ir,
        graph_index,
        pre_index,
        allow_public_unary_output=True,
    )
    if prefix is None:
        return None
    public_inputs = prefix.public_inputs
    public_outputs = prefix.public_outputs
    pre = prefix.pre
    squeeze = prefix.squeeze
    unary = prefix.unary
    source_name = prefix.source_name
    pre_output_name = prefix.pre_output_name
    squeeze_output_name = prefix.squeeze_output_name
    unary_output_name = prefix.unary_output_name
    source_contract = prefix.source_contract
    pre_contract = prefix.pre_contract
    squeeze_contract = prefix.squeeze_contract
    unary_contract = prefix.unary_contract
    squeeze_axis = prefix.squeeze_axis
    unary_index = prefix.unary_index
    unary_users = list(dict.fromkeys(graph_index.consumer_indices(unary_output_name)))
    if not unary_users or any(index <= unary_index for index in unary_users):
        return None
    selected: Optional[
        Tuple[int, OperatorIR, _TensorContract, int, OperatorIR, _TensorContract]
    ] = None
    for expand_index in unary_users:
        expand = model_ir.operators[int(expand_index)]
        if (
            str(expand.op_type) != "EXPAND_DIMS"
            or len(expand.inputs) != 2
            or len(expand.outputs) != 1
            or str(expand.inputs[0]) != unary_output_name
        ):
            continue
        expand_axis = _constant_vector(
            model_ir,
            graph_index,
            str(expand.inputs[1]),
            1,
            public_inputs,
        )
        if expand_axis is None:
            continue
        normalized_expand_axis = int(expand_axis[0])
        if normalized_expand_axis < 0:
            normalized_expand_axis += 4
        if normalized_expand_axis != int(squeeze_axis):
            continue

        expand_output_name = str(expand.outputs[0])
        if (
            not expand_output_name
            or expand_output_name in public_inputs | public_outputs
            or not _producer_is_valid(
                graph_index,
                expand_output_name,
                int(expand_index),
            )
        ):
            continue
        expand_contract = _tensor_contract(model_ir, expand_output_name, 4)
        expected_expand_shape = tuple(
            list(unary_contract.shape[:squeeze_axis])
            + [1]
            + list(unary_contract.shape[squeeze_axis:])
        )
        expected_expand_signature = tuple(
            list(unary_contract.signature[:squeeze_axis])
            + [1]
            + list(unary_contract.signature[squeeze_axis:])
        )
        if (
            expand_contract is None
            or expand_contract.shape != expected_expand_shape
            or expand_contract.signature != expected_expand_signature
            or expand_contract.shape != pre_contract.shape
            or expand_contract.signature != pre_contract.signature
        ):
            continue

        expand_users = graph_index.consumer_indices(expand_output_name)
        if len(expand_users) != 1:
            continue
        post_index = int(expand_users[0])
        if post_index <= int(expand_index):
            continue
        post = model_ir.operators[post_index]
        if (
            str(post.op_type) != "TRANSPOSE"
            or len(post.inputs) != 2
            or len(post.outputs) != 1
            or str(post.inputs[0]) != expand_output_name
            or _constant_vector(
                model_ir,
                graph_index,
                str(post.inputs[1]),
                4,
                public_inputs,
            )
            != _PERM_NCHW_TO_NHWC
        ):
            continue
        post_output_name = str(post.outputs[0])
        if (
            not post_output_name
            or post_output_name in public_inputs
            or not _producer_is_valid(graph_index, post_output_name, post_index)
        ):
            continue
        post_contract = _tensor_contract(model_ir, post_output_name, 4)
        expected_post_shape = tuple(
            expand_contract.shape[index] for index in _PERM_NCHW_TO_NHWC
        )
        expected_post_signature = tuple(
            expand_contract.signature[index] for index in _PERM_NCHW_TO_NHWC
        )
        if (
            post_contract is None
            or post_contract.shape != expected_post_shape
            or post_contract.signature != expected_post_signature
            or post_contract.shape != source_contract.shape
            or post_contract.signature != source_contract.signature
            or any(
                int(consumer_index) <= post_index
                for consumer_index in graph_index.consumer_indices(post_output_name)
            )
        ):
            continue
        selected = (
            int(expand_index),
            expand,
            expand_contract,
            post_index,
            post,
            post_contract,
        )
        break
    if selected is None:
        return None
    expand_index, expand, expand_contract, _, post, post_contract = selected
    if unary_output_name not in public_outputs and not any(
        int(index) != int(expand_index) for index in unary_users
    ):
        return None

    data_names = (
        source_name,
        pre_output_name,
        squeeze_output_name,
        unary_output_name,
        str(expand.outputs[0]),
        str(post.outputs[0]),
    )
    if len(set(data_names)) != len(data_names):
        return None
    input_group = (source_contract, pre_contract, squeeze_contract)
    output_group = (unary_contract, expand_contract)
    if (
        len({str(contract.tensor.dtype) for contract in input_group}) != 1
        or len({str(contract.tensor.dtype) for contract in output_group}) != 1
        or not _quantization_contract(input_group)
        or not _quantization_contract(output_group)
    ):
        return None
    if str(unary.op_type) != "CAST" and (
        str(source_contract.tensor.dtype) != str(unary_contract.tensor.dtype)
        or not _quantization_contract(input_group + output_group)
    ):
        return None

    try:
        pre_output_quantization = _clone_quantization(
            unary_contract.tensor.quantization
        )
        post_output_quantization = _clone_quantization(
            unary_contract.tensor.quantization
        )
    except Exception:
        return None
    return _FanoutRewritePlan(
        pre=pre,
        squeeze=squeeze,
        unary=unary,
        expand=expand,
        post=post,
        source_name=source_name,
        unary_output_name=unary_output_name,
        post_output_name=str(post.outputs[0]),
        pre_output_tensor=pre_contract.tensor,
        post_output_tensor=post_contract.tensor,
        output_dtype=str(unary_contract.tensor.dtype),
        pre_output_quantization=pre_output_quantization,
        post_output_quantization=post_output_quantization,
    )


def _apply_fanout_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _FanoutRewritePlan,
) -> bool:
    ordered_ops = (
        plan.pre,
        plan.squeeze,
        plan.unary,
        plan.expand,
        plan.post,
    )
    indices = [graph_index.operator_index(operator) for operator in ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != len(ordered_ops):
        return False

    insertion_index = int(resolved[0])
    graph_index.remove_operators([resolved[index] for index in (2, 3, 4)])
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.unary,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.unary,
        new_outputs=[plan.post_output_name],
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.pre,
        input_index=0,
        new_input_name=plan.post_output_name,
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.squeeze,
        new_outputs=[plan.unary_output_name],
        graph_index=graph_index,
    )
    plan.pre_output_tensor.dtype = str(plan.output_dtype)
    plan.pre_output_tensor.quantization = plan.pre_output_quantization
    plan.post_output_tensor.dtype = str(plan.output_dtype)
    plan.post_output_tensor.quantization = plan.post_output_quantization
    graph_index.insert_operator(insertion_index, plan.unary)
    return True


def _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    required = {"TRANSPOSE": 2, "SQUEEZE": 1, "EXPAND_DIMS": 1}
    has_unary = False
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required and required[op_type] > 0:
            required[op_type] -= 1
        if op_type in _UNARY_OPS:
            has_unary = True
    if not has_unary or any(count > 0 for count in required.values()):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {_FANOUT_STATS_KEY: 0}

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
        plan = _resolve_fanout_candidate(model_ir, active_index, pre_index)
        if plan is not None and _apply_fanout_plan(model_ir, active_index, plan):
            rewritten += 1

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {_FANOUT_STATS_KEY: int(rewritten)}


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    name = str(base)
    serial = 1
    while name in model_ir.tensors:
        name = f"{base}_{serial}"
        serial += 1
    return name


def _plan_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    operator_index: int,
    input_index: int,
    values: Tuple[int, ...],
    suffix: str,
    public_outputs: set[str],
) -> Optional[_ConstantUpdate]:
    if int(input_index) >= len(operator.inputs):
        return None
    tensor_name = str(operator.inputs[int(input_index)])
    tensor = model_ir.tensors.get(tensor_name)
    if tensor is None or tensor_name in public_outputs:
        return None
    try:
        data = np.asarray(values, dtype=np.asarray(tensor.data).dtype)
    except Exception:
        return None
    clone_name: Optional[str] = None
    clone: Optional[TensorIR] = None
    if set(graph_index.consumer_indices(tensor_name)) != {int(operator_index)}:
        clone_name = _unique_tensor_name(model_ir, f"{tensor_name}_{suffix}")
        try:
            quantization = _clone_quantization(tensor.quantization)
        except Exception:
            return None
        clone = TensorIR(
            name=clone_name,
            dtype=str(tensor.dtype),
            shape=[len(values)],
            shape_signature=[len(values)],
            data=np.asarray(data),
            is_variable=False,
            quantization=quantization,
        )
    return _ConstantUpdate(
        operator=operator,
        input_index=int(input_index),
        tensor=tensor,
        data=data,
        clone_name=clone_name,
        clone=clone,
    )


def _apply_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    update: _ConstantUpdate,
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
    return _write_const_ints_to_tensor(
        update.tensor,
        [int(value) for value in np.asarray(update.data).reshape(-1).tolist()],
    )


def _resolve_rank4_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
) -> Optional[_Rank4RewritePlan]:
    pre = model_ir.operators[int(pre_index)]
    if len(pre.inputs) != 2 or len(pre.outputs) != 1:
        return None
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    if (
        _constant_vector(
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
    if source_contract is None or pre_contract is None:
        return None
    if (
        int(source_contract.shape[0]) != 1
        or int(source_contract.signature[0]) != 1
        or pre_contract.shape
        != tuple(source_contract.shape[index] for index in _PERM_NHWC_TO_NCHW)
        or pre_contract.signature
        != tuple(source_contract.signature[index] for index in _PERM_NHWC_TO_NCHW)
    ):
        return None
    source_producer = graph_index.producers.get(source_name)
    if source_name in graph_index.duplicate_producers:
        return None
    if source_producer is not None:
        if source_name in public_inputs or int(source_producer) >= int(pre_index):
            return None
    elif source_name not in public_inputs and source_contract.tensor.data is None:
        return None

    pre_users = graph_index.consumer_indices(pre_output_name)
    if len(pre_users) != 1:
        return None
    unary_index = int(pre_users[0])
    if unary_index <= int(pre_index):
        return None
    unary = model_ir.operators[unary_index]
    if (
        str(unary.op_type) not in _UNARY_OPS
        or len(unary.inputs) != 1
        or len(unary.outputs) != 1
        or str(unary.inputs[0]) != pre_output_name
    ):
        return None
    unary_output_name = str(unary.outputs[0])
    if (
        not unary_output_name
        or unary_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, unary_output_name, unary_index)
    ):
        return None
    unary_contract = _tensor_contract(model_ir, unary_output_name, 4)
    if (
        unary_contract is None
        or unary_contract.shape != pre_contract.shape
        or unary_contract.signature != pre_contract.signature
    ):
        return None

    unary_users = graph_index.consumer_indices(unary_output_name)
    if len(unary_users) != 1:
        return None
    mid_index = int(unary_users[0])
    if mid_index <= unary_index:
        return None
    mid = model_ir.operators[mid_index]
    if (
        str(mid.op_type) != "TRANSPOSE"
        or len(mid.inputs) != 2
        or len(mid.outputs) != 1
        or str(mid.inputs[0]) != unary_output_name
        or _constant_vector(
            model_ir,
            graph_index,
            str(mid.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NHCW_FROM_NCHW
    ):
        return None
    mid_output_name = str(mid.outputs[0])
    if (
        not mid_output_name
        or mid_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, mid_output_name, mid_index)
    ):
        return None
    mid_contract = _tensor_contract(model_ir, mid_output_name, 4)
    expected_mid_shape = tuple(
        unary_contract.shape[index] for index in _PERM_NHCW_FROM_NCHW
    )
    expected_mid_signature = tuple(
        unary_contract.signature[index] for index in _PERM_NHCW_FROM_NCHW
    )
    if (
        mid_contract is None
        or mid_contract.shape != expected_mid_shape
        or mid_contract.signature != expected_mid_signature
    ):
        return None

    mid_users = graph_index.consumer_indices(mid_output_name)
    if len(mid_users) != 1:
        return None
    reshape_index = int(mid_users[0])
    if reshape_index <= mid_index:
        return None
    reshape = model_ir.operators[reshape_index]
    if (
        str(reshape.op_type) != "RESHAPE"
        or len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != mid_output_name
    ):
        return None
    reshape_output_name = str(reshape.outputs[0])
    if (
        not reshape_output_name
        or reshape_output_name in public_inputs | public_outputs
        or not _producer_is_valid(graph_index, reshape_output_name, reshape_index)
    ):
        return None
    reshape_contract = _tensor_contract(model_ir, reshape_output_name, 3)
    expected_reshape_shape = tuple(mid_contract.shape[1:])
    expected_reshape_signature = tuple(mid_contract.signature[1:])
    if (
        reshape_contract is None
        or reshape_contract.shape != expected_reshape_shape
        or reshape_contract.signature != expected_reshape_signature
    ):
        return None
    shape_values = _constant_vector(
        model_ir,
        graph_index,
        str(reshape.inputs[1]),
        3,
        public_inputs,
    )
    if shape_values != expected_reshape_shape:
        return None

    reshape_users = list(dict.fromkeys(graph_index.consumer_indices(reshape_output_name)))
    if not reshape_users:
        return None
    selected: Optional[
        Tuple[int, OperatorIR, _TensorContract, int, OperatorIR, _TensorContract]
    ] = None
    for expand_index in reshape_users:
        if int(expand_index) <= reshape_index:
            continue
        expand = model_ir.operators[int(expand_index)]
        if (
            str(expand.op_type) != "EXPAND_DIMS"
            or len(expand.inputs) != 2
            or len(expand.outputs) != 1
            or str(expand.inputs[0]) != reshape_output_name
        ):
            continue
        axis_values = _constant_vector(
            model_ir,
            graph_index,
            str(expand.inputs[1]),
            1,
            public_inputs,
        )
        if axis_values is None:
            continue
        axis = int(axis_values[0])
        if axis < 0:
            axis += 4
        if axis != 2:
            continue
        expand_output_name = str(expand.outputs[0])
        if (
            not expand_output_name
            or expand_output_name in public_inputs | public_outputs
            or not _producer_is_valid(graph_index, expand_output_name, expand_index)
        ):
            continue
        expand_contract = _tensor_contract(model_ir, expand_output_name, 4)
        expected_expand_shape = (
            reshape_contract.shape[0],
            reshape_contract.shape[1],
            1,
            reshape_contract.shape[2],
        )
        expected_expand_signature = (
            reshape_contract.signature[0],
            reshape_contract.signature[1],
            1,
            reshape_contract.signature[2],
        )
        if (
            expand_contract is None
            or expand_contract.shape != expected_expand_shape
            or expand_contract.signature != expected_expand_signature
        ):
            continue
        expand_users = graph_index.consumer_indices(expand_output_name)
        if len(expand_users) != 1:
            continue
        post_index = int(expand_users[0])
        if post_index <= int(expand_index):
            continue
        post = model_ir.operators[post_index]
        if (
            str(post.op_type) != "TRANSPOSE"
            or len(post.inputs) != 2
            or len(post.outputs) != 1
            or str(post.inputs[0]) != expand_output_name
            or _constant_vector(
                model_ir,
                graph_index,
                str(post.inputs[1]),
                4,
                public_inputs,
            )
            != _PERM_NCHW_TO_NHWC
        ):
            continue
        post_output_name = str(post.outputs[0])
        if (
            not post_output_name
            or post_output_name in public_inputs
            or not _producer_is_valid(graph_index, post_output_name, post_index)
        ):
            continue
        post_contract = _tensor_contract(model_ir, post_output_name, 4)
        expected_post_shape = tuple(
            expand_contract.shape[index] for index in _PERM_NCHW_TO_NHWC
        )
        expected_post_signature = tuple(
            expand_contract.signature[index] for index in _PERM_NCHW_TO_NHWC
        )
        if (
            post_contract is None
            or post_contract.shape != expected_post_shape
            or post_contract.signature != expected_post_signature
            or any(
                int(consumer_index) <= post_index
                for consumer_index in graph_index.consumer_indices(post_output_name)
            )
        ):
            continue
        selected = (
            int(expand_index),
            expand,
            expand_contract,
            post_index,
            post,
            post_contract,
        )
        break
    if selected is None:
        return None
    (
        expand_index,
        expand,
        expand_contract,
        post_index,
        post,
        post_contract,
    ) = selected

    non_expand_indices = [
        int(index) for index in reshape_users if int(index) != int(expand_index)
    ]
    if any(index <= reshape_index for index in non_expand_indices):
        return None
    non_expand_users = tuple(
        model_ir.operators[index] for index in non_expand_indices
    )

    data_names = (
        source_name,
        pre_output_name,
        unary_output_name,
        mid_output_name,
        reshape_output_name,
        str(expand.outputs[0]),
        str(post.outputs[0]),
    )
    if len(set(data_names)) != len(data_names):
        return None
    input_group = (source_contract, pre_contract)
    output_group = (unary_contract, mid_contract, reshape_contract, post_contract)
    if (
        len({str(contract.tensor.dtype) for contract in input_group}) != 1
        or len({str(contract.tensor.dtype) for contract in output_group}) != 1
        or not _quantization_contract(input_group)
        or not _quantization_contract(output_group)
    ):
        return None
    if str(unary.op_type) != "CAST" and (
        str(source_contract.tensor.dtype) != str(unary_contract.tensor.dtype)
        or not _quantization_contract(input_group + output_group)
    ):
        return None

    new_reshape_shape = tuple(
        reshape_contract.shape[index] for index in _PERM_HCW_TO_HWC
    )
    new_reshape_signature = tuple(
        reshape_contract.signature[index] for index in _PERM_HCW_TO_HWC
    )
    if new_reshape_signature.count(-1) > 1 or new_reshape_shape == shape_values:
        return None
    reshape_options: Optional[Dict[str, Any]] = None
    if isinstance(reshape.options, dict):
        reshape_options = dict(reshape.options)
        for key in ("newShape", "onnxRawNewShape"):
            value = reshape_options.get(key)
            if isinstance(value, list):
                if tuple(int(item) for item in value) != expected_reshape_shape:
                    return None
                reshape_options[key] = list(new_reshape_signature)
        if new_reshape_signature != new_reshape_shape:
            reshape_options["newShape"] = list(new_reshape_signature)
            if "onnxRawNewShape" in reshape_options:
                reshape_options["onnxRawNewShape"] = list(new_reshape_signature)
            reshape_options["preserveDynamicShape"] = True
    elif new_reshape_signature != new_reshape_shape:
        return None

    reshape_shape_update = _plan_constant_update(
        model_ir,
        graph_index,
        reshape,
        reshape_index,
        1,
        new_reshape_signature,
        "nhwc_shape",
        public_outputs,
    )
    expand_axis_update = _plan_constant_update(
        model_ir,
        graph_index,
        expand,
        expand_index,
        1,
        (1,),
        "nhwc_axis",
        public_outputs,
    )
    if reshape_shape_update is None or expand_axis_update is None:
        return None
    try:
        expand_quantization = _clone_quantization(post_contract.tensor.quantization)
    except Exception:
        return None

    bridge_permutation: Optional[TensorIR] = None
    bridge_tensor: Optional[TensorIR] = None
    bridge: Optional[OperatorIR] = None
    if non_expand_users:
        permutation_name = _unique_tensor_name(
            model_ir,
            f"{reshape_output_name}_legacy_hcw_perm",
        )
        bridge_name = _unique_tensor_name(
            model_ir,
            f"{reshape_output_name}_legacy_hcw",
        )
        try:
            bridge_quantization = _clone_quantization(
                reshape_contract.tensor.quantization
            )
        except Exception:
            return None
        bridge_permutation = TensorIR(
            name=permutation_name,
            dtype="INT32",
            shape=[3],
            shape_signature=[3],
            data=np.asarray(_PERM_HCW_TO_HWC, dtype=np.int32),
            is_variable=False,
        )
        bridge_tensor = TensorIR(
            name=bridge_name,
            dtype=str(reshape_contract.tensor.dtype),
            shape=list(reshape_contract.shape),
            shape_signature=list(reshape_contract.signature),
            quantization=bridge_quantization,
        )
        bridge = OperatorIR(
            op_type="TRANSPOSE",
            inputs=[reshape_output_name, permutation_name],
            outputs=[bridge_name],
            options={},
        )

    return _Rank4RewritePlan(
        pre=pre,
        unary=unary,
        unary_tensor=unary_contract.tensor,
        mid=mid,
        reshape=reshape,
        reshape_tensor=reshape_contract.tensor,
        reshape_options=reshape_options,
        reshape_shape_update=reshape_shape_update,
        expand=expand,
        expand_axis_update=expand_axis_update,
        expand_tensor=expand_contract.tensor,
        expand_shape=post_contract.shape,
        expand_signature=post_contract.signature,
        expand_dtype=str(post_contract.tensor.dtype),
        expand_quantization=expand_quantization,
        post=post,
        post_output_name=str(post.outputs[0]),
        source_name=source_name,
        non_expand_users=non_expand_users,
        bridge_permutation=bridge_permutation,
        bridge_tensor=bridge_tensor,
        bridge=bridge,
    )


def _apply_rank4_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _Rank4RewritePlan,
) -> bool:
    ordered_ops = (
        plan.pre,
        plan.unary,
        plan.mid,
        plan.reshape,
        plan.expand,
        plan.post,
    )
    indices = [graph_index.operator_index(operator) for operator in ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != len(ordered_ops):
        return False
    if not _apply_constant_update(
        model_ir,
        graph_index,
        plan.reshape_shape_update,
    ):
        return False
    if not _apply_constant_update(
        model_ir,
        graph_index,
        plan.expand_axis_update,
    ):
        return False
    if plan.reshape_options is not None:
        plan.reshape.options = dict(plan.reshape_options)

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.unary,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    _permute_tensor_metadata_if_rank_matches(
        plan.unary_tensor,
        list(_PERM_NCHW_TO_NHWC),
    )
    reshape_inputs = [str(value) for value in plan.reshape.inputs]
    reshape_inputs[0] = str(plan.unary.outputs[0])
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.reshape,
        new_inputs=reshape_inputs,
        graph_index=graph_index,
    )
    _permute_tensor_metadata_if_rank_matches(
        plan.reshape_tensor,
        list(_PERM_HCW_TO_HWC),
    )

    if plan.bridge is not None:
        if (
            plan.bridge_permutation is None
            or plan.bridge_tensor is None
            or plan.bridge_permutation.name in model_ir.tensors
            or plan.bridge_tensor.name in model_ir.tensors
        ):
            return False
        user_indices = [
            graph_index.operator_index(operator)
            for operator in plan.non_expand_users
        ]
        if any(index is None for index in user_indices):
            return False
        insertion_index = min(int(index) for index in user_indices if index is not None)
        model_ir.tensors[plan.bridge_permutation.name] = plan.bridge_permutation
        model_ir.tensors[plan.bridge_tensor.name] = plan.bridge_tensor
        graph_index.insert_operator(insertion_index, plan.bridge)
        for operator in plan.non_expand_users:
            operator_index = graph_index.operator_index(operator)
            if operator_index is None:
                return False
            for input_index, input_name in enumerate(list(operator.inputs)):
                if str(input_name) == str(plan.reshape.outputs[0]):
                    _replace_operator_input_at(
                        model_ir=model_ir,
                        op=operator,
                        input_index=input_index,
                        new_input_name=plan.bridge_tensor.name,
                        graph_index=graph_index,
                    )

    plan.expand_tensor.shape = list(plan.expand_shape)
    plan.expand_tensor.shape_signature = list(plan.expand_signature)
    plan.expand_tensor.dtype = str(plan.expand_dtype)
    plan.expand_tensor.quantization = plan.expand_quantization
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
        graph_index.operator_index(operator)
        for operator in (plan.pre, plan.mid, plan.post)
    ]
    if any(index is None for index in remove_indices):
        return False
    graph_index.remove_operators(
        [int(index) for index in remove_indices if index is not None]
    )
    return True


def _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    required = {"TRANSPOSE": 3, "RESHAPE": 1, "EXPAND_DIMS": 1}
    has_unary = False
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required and required[op_type] > 0:
            required[op_type] -= 1
        if op_type in _UNARY_OPS:
            has_unary = True
    if not has_unary or any(count > 0 for count in required.values()):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {_RANK4_STATS_KEY: 0}

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
        plan = _resolve_rank4_candidate(model_ir, active_index, pre_index)
        if plan is not None and _apply_rank4_plan(model_ir, active_index, plan):
            rewritten += 1

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {_RANK4_STATS_KEY: int(rewritten)}
