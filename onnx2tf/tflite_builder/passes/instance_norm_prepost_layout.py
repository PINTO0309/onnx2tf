from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _normalize_squeeze_axes_for_rank,
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_inputs,
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


_STATS_KEY = (
    "optimized_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains"
)
_SIDE_STATS_KEY = (
    "optimized_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains"
)
_UNARY_RESHAPE_STATS_KEY = (
    "optimized_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains"
)
_RESIDUAL_RESHAPE_STATS_KEY = (
    "optimized_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains"
)
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_PERM_CHW_TO_HWC = (1, 2, 0)
_PERM_HWC_TO_CHW = (2, 0, 1)
_FLOAT_DTYPES = {
    "FLOAT16": np.dtype(np.float16),
    "FLOAT32": np.dtype(np.float32),
}
_SIDE_ADAPTER_PERM_NAME = "__instancenorm_tail_nhwc_to_nchw_perm_rank4__"
_RESIDUAL_ADAPTER_PERM_NAME = "__instancenorm_tail_hwc_to_chw_perm_rank3__"
_TAIL_UNARY_OPS = {
    "RELU",
    "RELU6",
    "LEAKY_RELU",
    "LOGISTIC",
    "TANH",
    "ABS",
    "NEG",
    "SQRT",
    "EXP",
    "CAST",
    "FLOOR",
    "CEIL",
    "ROUND",
}


@dataclass(frozen=True)
class _TensorMetadataUpdate:
    contract: _TensorContract
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _ConstantUse:
    operator: OperatorIR
    input_index: int


@dataclass(frozen=True)
class _ConstantUpdate:
    tensor: TensorIR
    data: np.ndarray
    uses: Tuple[_ConstantUse, ...]
    clone_name: Optional[str]
    clone: Optional[TensorIR]


@dataclass(frozen=True)
class _ResidualSourcePlan:
    bridge: OperatorIR
    add_input_index: int
    add_replacement_name: Optional[str]
    squeeze: Optional[OperatorIR]
    squeeze_source_name: Optional[str]


@dataclass(frozen=True)
class _FanoutAdapterPlan:
    uses: Tuple[_ConstantUse, ...]
    permutation: Optional[TensorIR]
    tensor: TensorIR


@dataclass(frozen=True)
class _InstanceNormPrepostPlan:
    ordered_ops: Tuple[OperatorIR, ...]
    pre: OperatorIR
    squeeze: OperatorIR
    reshape: OperatorIR
    reshape_options: Dict[str, object]
    source_name: str
    mean1: OperatorIR
    sub: OperatorIR
    add_bias: OperatorIR
    post: OperatorIR
    post_output_name: str
    tail_mode: str
    side_squeeze: Optional[OperatorIR]
    side_adapter_permutation: Optional[TensorIR]
    tail_squeeze: Optional[OperatorIR]
    tail_unary: Optional[OperatorIR]
    tail_reshape: Optional[OperatorIR]
    tail_reshape_options: Optional[Dict[str, object]]
    tail_add: Optional[OperatorIR]
    residual_source: Optional[_ResidualSourcePlan]
    fanout_adapter: Optional[_FanoutAdapterPlan]
    constant_updates: Tuple[_ConstantUpdate, ...]
    metadata_updates: Tuple[_TensorMetadataUpdate, ...]
    channel_last_names: Tuple[str, ...]


@dataclass(frozen=True)
class _ResidualTailMatch:
    post_index: int
    post: OperatorIR
    squeeze: OperatorIR
    squeeze_contract: _TensorContract
    add: OperatorIR
    add_contract: _TensorContract
    reshape: OperatorIR
    reshape_contract: _TensorContract
    reshape_shape_name: str
    residual_source: _ResidualSourcePlan
    residual_metadata_updates: Tuple[_TensorMetadataUpdate, ...]
    fanout_adapter: Optional[_FanoutAdapterPlan]


def _contract(
    model_ir: ModelIR,
    name: str,
    rank: int,
    shape: Sequence[int],
    signature: Sequence[int],
) -> Optional[_TensorContract]:
    contract = _tensor_contract(model_ir, str(name), int(rank))
    if (
        contract is None
        or contract.shape != tuple(int(value) for value in shape)
        or contract.signature != tuple(int(value) for value in signature)
    ):
        return None
    return contract


def _normalized_axes(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    rank: int,
    size: int,
    public_inputs: set[str],
) -> Optional[Tuple[int, ...]]:
    values = _constant_vector(
        model_ir,
        graph_index,
        str(name),
        int(size),
        public_inputs,
    )
    if values is None:
        return None
    normalized = _normalize_squeeze_axes_for_rank(
        [int(value) for value in values],
        int(rank),
    )
    if normalized is None or len(normalized) != int(size):
        return None
    return tuple(int(value) for value in normalized)


def _squeeze_axis_zero(
    squeeze: OperatorIR,
    source: _TensorContract,
    output: _TensorContract,
) -> bool:
    options = dict(squeeze.options) if isinstance(squeeze.options, dict) else {}
    if "squeezeDims" in options:
        try:
            values = [
                int(value)
                for value in np.asarray(options["squeezeDims"])
                .reshape(-1)
                .tolist()
            ]
        except Exception:
            return False
        axes = _normalize_squeeze_axes_for_rank(values, 4)
        if axes != [0]:
            return False
    elif sum(int(value) == 1 for value in source.shape) != 1:
        return False
    return bool(
        source.shape[0] == 1
        and source.signature[0] == 1
        and output.shape == source.shape[1:]
        and output.signature == source.signature[1:]
    )


def _float_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    dtype: str,
    *,
    shape: Optional[Tuple[int, ...]] = None,
    value: Optional[float] = None,
    nonnegative: bool = False,
) -> Optional[np.ndarray]:
    tensor = model_ir.tensors.get(str(name))
    expected_dtype = _FLOAT_DTYPES.get(str(dtype))
    if (
        tensor is None
        or tensor.data is None
        or expected_dtype is None
        or str(tensor.dtype) != str(dtype)
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or tensor.quantization is not None
    ):
        return None
    try:
        data = np.asarray(tensor.data)
        tensor_shape = tuple(int(item) for item in tensor.shape)
        signature = (
            tensor_shape
            if tensor.shape_signature is None
            else tuple(int(item) for item in tensor.shape_signature)
        )
    except Exception:
        return None
    if (
        data.dtype != expected_dtype
        or data.shape != tensor_shape
        or signature != tensor_shape
        or not np.all(np.isfinite(data))
        or (shape is not None and tensor_shape != tuple(shape))
    ):
        return None
    if value is not None and (
        data.size != 1 or float(data.reshape(-1)[0]) != float(value)
    ):
        return None
    if nonnegative and (
        data.size != 1 or float(data.reshape(-1)[0]) < 0.0
    ):
        return None
    return data


def _plan_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    tensor_name: str,
    data: np.ndarray,
    uses: Sequence[_ConstantUse],
    suffix: str,
    public_names: set[str],
) -> Optional[_ConstantUpdate]:
    name = str(tensor_name)
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or name in public_names
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or not uses
    ):
        return None
    resolved_indices = []
    for use in uses:
        operator_index = graph_index.operator_index(use.operator)
        if (
            operator_index is None
            or int(use.input_index) < 0
            or int(use.input_index) >= len(use.operator.inputs)
            or str(use.operator.inputs[int(use.input_index)]) != name
        ):
            return None
        resolved_indices.append(int(operator_index))
    try:
        replacement = np.asarray(data, dtype=np.asarray(tensor.data).dtype)
    except Exception:
        return None
    clone_name: Optional[str] = None
    clone: Optional[TensorIR] = None
    if Counter(graph_index.consumer_indices(name)) != Counter(resolved_indices):
        clone_name = _unique_tensor_name(model_ir, f"{name}_{suffix}")
        try:
            quantization = _clone_quantization(tensor.quantization)
        except Exception:
            return None
        clone = TensorIR(
            name=clone_name,
            dtype=str(tensor.dtype),
            shape=[int(value) for value in replacement.shape],
            shape_signature=[int(value) for value in replacement.shape],
            data=np.asarray(replacement),
            is_variable=False,
            quantization=quantization,
            logical_layout=str(tensor.logical_layout),
            physical_layout=str(tensor.physical_layout),
            onnx_tensor_name=tensor.onnx_tensor_name,
        )
    return _ConstantUpdate(
        tensor=tensor,
        data=np.asarray(replacement),
        uses=tuple(uses),
        clone_name=clone_name,
        clone=clone,
    )


def _binary_other_input(
    operator: OperatorIR,
    data_name: str,
) -> Optional[Tuple[str, int]]:
    if len(operator.inputs) != 2:
        return None
    matches = [
        index
        for index, name in enumerate(operator.inputs)
        if str(name) == str(data_name)
    ]
    if len(matches) != 1:
        return None
    data_index = int(matches[0])
    return str(operator.inputs[1 - data_index]), 1 - data_index


def _sole_consumer(
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Optional[Tuple[int, OperatorIR]]:
    users = graph_index.consumer_indices(str(name))
    if len(users) != 1:
        return None
    index = int(users[0])
    return index, graph_index.model_ir.operators[index]


def _metadata_update_chw_to_hwc(
    contract: _TensorContract,
) -> _TensorMetadataUpdate:
    return _TensorMetadataUpdate(
        contract=contract,
        shape=tuple(contract.shape[index] for index in _PERM_CHW_TO_HWC),
        signature=tuple(
            contract.signature[index] for index in _PERM_CHW_TO_HWC
        ),
    )


def _resolve_residual_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    add_index: int,
    add_input_index: int,
    residual_name: str,
    expected_shape: Tuple[int, ...],
    expected_signature: Tuple[int, ...],
    dtype: str,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[Tuple[_ResidualSourcePlan, Tuple[_TensorMetadataUpdate, ...]]]:
    public_names = public_inputs | public_outputs
    residual = _contract(
        model_ir,
        residual_name,
        3,
        expected_shape,
        expected_signature,
    )
    producer_index = graph_index.producers.get(str(residual_name))
    if (
        residual is None
        or residual_name in public_names
        or producer_index is None
        or residual_name in graph_index.duplicate_producers
        or graph_index.consumer_indices(residual_name) != [int(add_index)]
        or int(producer_index) >= int(add_index)
        or str(residual.tensor.dtype) != str(dtype)
        or residual.tensor.quantization is not None
    ):
        return None
    producer = model_ir.operators[int(producer_index)]
    if (
        str(producer.op_type) == "TRANSPOSE"
        and len(producer.inputs) == 2
        and len(producer.outputs) == 1
        and str(producer.outputs[0]) == residual_name
        and _constant_vector(
            model_ir,
            graph_index,
            str(producer.inputs[1]),
            3,
            public_inputs,
        )
        == _PERM_HWC_TO_CHW
        and str(producer.inputs[1]) not in public_outputs
    ):
        source_name = str(producer.inputs[0])
        source = _contract(
            model_ir,
            source_name,
            3,
            tuple(expected_shape[index] for index in _PERM_CHW_TO_HWC),
            tuple(expected_signature[index] for index in _PERM_CHW_TO_HWC),
        )
        if (
            source is None
            or source_name in public_outputs
            or str(source.tensor.dtype) != str(dtype)
            or source.tensor.quantization is not None
            or not _valid_source(
                graph_index,
                source,
                source_name,
                int(producer_index),
                public_inputs,
            )
        ):
            return None
        return (
            _ResidualSourcePlan(
                bridge=producer,
                add_input_index=int(add_input_index),
                add_replacement_name=source_name,
                squeeze=None,
                squeeze_source_name=None,
            ),
            (),
        )

    residual_unary: Optional[OperatorIR] = None
    residual_unary_contract: Optional[_TensorContract] = None
    squeeze = producer
    squeeze_index = int(producer_index)
    if str(producer.op_type) in _TAIL_UNARY_OPS:
        if (
            len(producer.inputs) != 1
            or len(producer.outputs) != 1
            or str(producer.outputs[0]) != residual_name
        ):
            return None
        residual_unary = producer
        residual_unary_contract = residual
        squeeze_output_name = str(producer.inputs[0])
        squeeze_index_value = graph_index.producers.get(squeeze_output_name)
        if (
            squeeze_index_value is None
            or squeeze_output_name in graph_index.duplicate_producers
            or graph_index.consumer_indices(squeeze_output_name)
            != [int(producer_index)]
        ):
            return None
        squeeze_index = int(squeeze_index_value)
        squeeze = model_ir.operators[squeeze_index]
    squeeze_output_name = (
        str(squeeze.outputs[0]) if len(squeeze.outputs) == 1 else ""
    )
    squeeze_contract = _contract(
        model_ir,
        squeeze_output_name,
        3,
        expected_shape,
        expected_signature,
    )
    squeeze_input_name = str(squeeze.inputs[0]) if len(squeeze.inputs) == 1 else ""
    pre_index_value = graph_index.producers.get(squeeze_input_name)
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or len(squeeze.outputs) != 1
        or squeeze_contract is None
        or squeeze_output_name in public_names
        or not _producer_is_valid(
            graph_index,
            squeeze_output_name,
            squeeze_index,
        )
        or pre_index_value is None
        or squeeze_input_name in graph_index.duplicate_producers
    ):
        return None
    pre_index = int(pre_index_value)
    pre = model_ir.operators[pre_index]
    pre_output = _contract(
        model_ir,
        squeeze_input_name,
        4,
        (1, *expected_shape),
        (1, *expected_signature),
    )
    source_name = str(pre.inputs[0]) if len(pre.inputs) == 2 else ""
    source = _contract(
        model_ir,
        source_name,
        4,
        tuple((1, *expected_shape)[index] for index in _PERM_NCHW_TO_NHWC),
        tuple(
            (1, *expected_signature)[index]
            for index in _PERM_NCHW_TO_NHWC
        ),
    )
    ordered_indices = [pre_index, squeeze_index]
    if residual_unary is not None:
        ordered_indices.append(int(producer_index))
    ordered_indices.append(int(add_index))
    input_dtype = (
        str(squeeze_contract.tensor.dtype)
        if residual_unary is not None
        else str(dtype)
    )
    input_contracts = (source, pre_output, squeeze_contract)
    if (
        str(pre.op_type) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or str(pre.outputs[0]) != squeeze_input_name
        or _constant_vector(
            model_ir,
            graph_index,
            str(pre.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NHWC_TO_NCHW
        or str(pre.inputs[1]) in public_outputs
        or pre_output is None
        or source is None
        or squeeze_input_name in public_names
        or graph_index.consumer_indices(squeeze_input_name) != [squeeze_index]
        or not _producer_is_valid(graph_index, squeeze_input_name, pre_index)
        or not _valid_source(
            graph_index,
            source,
            source_name,
            pre_index,
            public_inputs,
        )
        or source_name in public_outputs
        or not _squeeze_axis_zero(squeeze, pre_output, squeeze_contract)
        or ordered_indices != sorted(ordered_indices)
        or any(contract is None for contract in input_contracts)
        or any(
            str(contract.tensor.dtype) != input_dtype
            or contract.tensor.quantization is not None
            for contract in input_contracts
            if contract is not None
        )
        or (
            residual_unary is None
            and input_dtype != str(dtype)
        )
        or (
            residual_unary is not None
            and str(residual_unary.op_type) != "CAST"
            and input_dtype != str(dtype)
        )
        or (
            residual_unary is not None
            and str(residual_unary.op_type) == "CAST"
            and input_dtype not in _FLOAT_DTYPES
        )
    ):
        return None
    metadata_updates = [_metadata_update_chw_to_hwc(squeeze_contract)]
    if residual_unary_contract is not None:
        metadata_updates.append(
            _metadata_update_chw_to_hwc(residual_unary_contract)
        )
    return (
        _ResidualSourcePlan(
            bridge=pre,
            add_input_index=int(add_input_index),
            add_replacement_name=None,
            squeeze=squeeze,
            squeeze_source_name=source_name,
        ),
        tuple(metadata_updates),
    )


def _resolve_residual_tail(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    inst_output_name: str,
    inst_output: _TensorContract,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_ResidualTailMatch]:
    public_names = public_inputs | public_outputs
    squeeze_match = _sole_consumer(graph_index, inst_output_name)
    if squeeze_match is None:
        return None
    squeeze_index, squeeze = squeeze_match
    squeeze_name = str(squeeze.outputs[0]) if len(squeeze.outputs) == 1 else ""
    squeeze_contract = _contract(
        model_ir,
        squeeze_name,
        3,
        inst_output.shape[1:],
        inst_output.signature[1:],
    )
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or len(squeeze.outputs) != 1
        or str(squeeze.inputs[0]) != inst_output_name
        or squeeze_contract is None
        or squeeze_name in public_names
        or not _producer_is_valid(graph_index, squeeze_name, squeeze_index)
        or not _squeeze_axis_zero(squeeze, inst_output, squeeze_contract)
    ):
        return None
    add_match = _sole_consumer(graph_index, squeeze_name)
    if add_match is None:
        return None
    add_index, add = add_match
    residual_match = _binary_other_input(add, squeeze_name)
    add_name = str(add.outputs[0]) if len(add.outputs) == 1 else ""
    add_contract = _contract(
        model_ir,
        add_name,
        3,
        squeeze_contract.shape,
        squeeze_contract.signature,
    )
    if (
        str(add.op_type) != "ADD"
        or residual_match is None
        or len(add.outputs) != 1
        or add_contract is None
        or add_name in public_names
        or not _producer_is_valid(graph_index, add_name, add_index)
    ):
        return None
    residual_name, residual_input_index = residual_match
    residual_source_match = _resolve_residual_source(
        model_ir,
        graph_index,
        add_index=add_index,
        add_input_index=residual_input_index,
        residual_name=residual_name,
        expected_shape=squeeze_contract.shape,
        expected_signature=squeeze_contract.signature,
        dtype=str(inst_output.tensor.dtype),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if residual_source_match is None:
        return None
    residual_source, residual_metadata_updates = residual_source_match

    add_user_indices = sorted(set(graph_index.consumer_indices(add_name)))
    reshape_matches = [
        index
        for index in add_user_indices
        if str(model_ir.operators[index].op_type) == "RESHAPE"
        and len(model_ir.operators[index].inputs) == 2
        and len(model_ir.operators[index].outputs) == 1
        and str(model_ir.operators[index].inputs[0]) == add_name
    ]
    if (
        len(reshape_matches) != 1
        or any(int(index) <= int(add_index) for index in add_user_indices)
    ):
        return None
    reshape_index = int(reshape_matches[0])
    reshape = model_ir.operators[reshape_index]
    reshape_name = str(reshape.outputs[0])
    reshape_contract = _contract(
        model_ir,
        reshape_name,
        4,
        inst_output.shape,
        inst_output.signature,
    )
    reshape_shape_name = str(reshape.inputs[1])
    if (
        reshape_contract is None
        or reshape_name in public_names
        or not _producer_is_valid(graph_index, reshape_name, reshape_index)
        or _constant_vector(
            model_ir,
            graph_index,
            reshape_shape_name,
            4,
            public_inputs,
        )
        != reshape_contract.shape
        or reshape_shape_name in public_outputs
        or model_ir.tensors[reshape_shape_name].quantization is not None
    ):
        return None
    post_match = _sole_consumer(graph_index, reshape_name)
    if post_match is None:
        return None
    post_index, post = post_match
    post_name = str(post.outputs[0]) if len(post.outputs) == 1 else ""
    post_contract = _contract(
        model_ir,
        post_name,
        4,
        tuple(inst_output.shape[index] for index in _PERM_NCHW_TO_NHWC),
        tuple(
            inst_output.signature[index]
            for index in _PERM_NCHW_TO_NHWC
        ),
    )
    tail_contracts = (
        squeeze_contract,
        add_contract,
        reshape_contract,
        post_contract,
    )
    if (
        str(post.op_type) != "TRANSPOSE"
        or len(post.inputs) != 2
        or len(post.outputs) != 1
        or str(post.inputs[0]) != reshape_name
        or _constant_vector(
            model_ir,
            graph_index,
            str(post.inputs[1]),
            4,
            public_inputs,
        )
        != _PERM_NCHW_TO_NHWC
        or str(post.inputs[1]) in public_outputs
        or post_contract is None
        or post_name in public_names
        or not _producer_is_valid(graph_index, post_name, post_index)
        or any(
            int(index) <= int(post_index)
            for index in graph_index.consumer_indices(post_name)
        )
        or any(contract is None for contract in tail_contracts)
        or any(
            str(contract.tensor.dtype) != str(inst_output.tensor.dtype)
            or contract.tensor.quantization is not None
            for contract in tail_contracts
            if contract is not None
        )
        or [squeeze_index, add_index, reshape_index, post_index]
        != sorted([squeeze_index, add_index, reshape_index, post_index])
    ):
        return None

    fanout_uses = tuple(
        _ConstantUse(model_ir.operators[index], input_index)
        for index in add_user_indices
        if int(index) != reshape_index
        for input_index, input_name in enumerate(model_ir.operators[index].inputs)
        if str(input_name) == add_name
    )
    fanout_adapter: Optional[_FanoutAdapterPlan] = None
    if fanout_uses:
        existing_permutation = model_ir.tensors.get(_RESIDUAL_ADAPTER_PERM_NAME)
        permutation: Optional[TensorIR] = None
        if existing_permutation is None:
            permutation = TensorIR(
                name=_RESIDUAL_ADAPTER_PERM_NAME,
                dtype="INT32",
                shape=[3],
                shape_signature=[3],
                data=np.asarray(_PERM_HWC_TO_CHW, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
        elif (
            _constant_vector(
                model_ir,
                graph_index,
                _RESIDUAL_ADAPTER_PERM_NAME,
                3,
                public_inputs,
            )
            != _PERM_HWC_TO_CHW
            or _RESIDUAL_ADAPTER_PERM_NAME in public_outputs
            or existing_permutation.quantization is not None
        ):
            return None
        adapter_name = _unique_tensor_name(model_ir, f"{add_name}_chw_adapter")
        try:
            adapter_quantization = _clone_quantization(
                add_contract.tensor.quantization
            )
        except Exception:
            return None
        fanout_adapter = _FanoutAdapterPlan(
            uses=fanout_uses,
            permutation=permutation,
            tensor=TensorIR(
                name=adapter_name,
                dtype=str(add_contract.tensor.dtype),
                shape=list(add_contract.shape),
                shape_signature=list(add_contract.signature),
                data=None,
                is_variable=False,
                quantization=adapter_quantization,
                logical_layout=str(add_contract.tensor.logical_layout),
                physical_layout=str(add_contract.tensor.physical_layout),
                onnx_tensor_name=add_contract.tensor.onnx_tensor_name,
            ),
        )
    return _ResidualTailMatch(
        post_index=post_index,
        post=post,
        squeeze=squeeze,
        squeeze_contract=squeeze_contract,
        add=add,
        add_contract=add_contract,
        reshape=reshape,
        reshape_contract=reshape_contract,
        reshape_shape_name=reshape_shape_name,
        residual_source=residual_source,
        residual_metadata_updates=residual_metadata_updates,
        fanout_adapter=fanout_adapter,
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre_index: int,
    *,
    tail_mode: str,
) -> Optional[_InstanceNormPrepostPlan]:
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    public_names = public_inputs | public_outputs
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
        or str(pre.inputs[1]) in public_outputs
    ):
        return None
    source_name = str(pre.inputs[0])
    pre_output_name = str(pre.outputs[0])
    source = _tensor_contract(model_ir, source_name, 4)
    if source is None:
        return None
    pre_output = _contract(
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

    squeeze_match = _sole_consumer(graph_index, pre_output_name)
    if squeeze_match is None:
        return None
    squeeze_index, squeeze = squeeze_match
    squeeze_output_name = (
        str(squeeze.outputs[0]) if len(squeeze.outputs) == 1 else ""
    )
    squeeze_output = _tensor_contract(model_ir, squeeze_output_name, 3)
    if (
        str(squeeze.op_type) != "SQUEEZE"
        or len(squeeze.inputs) != 1
        or len(squeeze.outputs) != 1
        or str(squeeze.inputs[0]) != pre_output_name
        or squeeze_output is None
        or squeeze_output_name in public_names
        or not _producer_is_valid(
            graph_index,
            squeeze_output_name,
            squeeze_index,
        )
        or not _squeeze_axis_zero(squeeze, pre_output, squeeze_output)
    ):
        return None

    reshape_match = _sole_consumer(graph_index, squeeze_output_name)
    if reshape_match is None:
        return None
    reshape_index, reshape = reshape_match
    x_name = str(reshape.outputs[0]) if len(reshape.outputs) == 1 else ""
    x = _contract(
        model_ir,
        x_name,
        4,
        pre_output.shape,
        pre_output.signature,
    )
    if (
        str(reshape.op_type) != "RESHAPE"
        or len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != squeeze_output_name
        or x is None
        or x_name in public_names
        or not _producer_is_valid(graph_index, x_name, reshape_index)
    ):
        return None
    reshape_shape_name = str(reshape.inputs[1])
    if _constant_vector(
        model_ir,
        graph_index,
        reshape_shape_name,
        4,
        public_inputs,
    ) != x.shape or reshape_shape_name in public_outputs:
        return None

    x_users = sorted(set(graph_index.consumer_indices(x_name)))
    if len(x_users) != 2:
        return None
    mean_matches = [
        index
        for index in x_users
        if str(model_ir.operators[index].op_type) == "MEAN"
        and len(model_ir.operators[index].inputs) == 2
        and str(model_ir.operators[index].inputs[0]) == x_name
    ]
    sub_matches = [
        index
        for index in x_users
        if str(model_ir.operators[index].op_type) == "SUB"
        and len(model_ir.operators[index].inputs) == 2
    ]
    if len(mean_matches) != 1 or len(sub_matches) != 1:
        return None
    mean1_index = int(mean_matches[0])
    sub_index = int(sub_matches[0])
    mean1 = model_ir.operators[mean1_index]
    sub = model_ir.operators[sub_index]
    mean1_options = dict(mean1.options) if isinstance(mean1.options, dict) else {}
    mean1_name = str(mean1.outputs[0]) if len(mean1.outputs) == 1 else ""
    reduced_shape = (
        pre_output.shape[0],
        pre_output.shape[1],
        1,
        1,
    )
    reduced_signature = (
        pre_output.signature[0],
        pre_output.signature[1],
        1,
        1,
    )
    mean1_contract = _contract(
        model_ir,
        mean1_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    mean1_axes_name = str(mean1.inputs[1])
    if (
        len(mean1.outputs) != 1
        or not bool(mean1_options.get("keepDims", False))
        or _normalized_axes(
            model_ir,
            graph_index,
            mean1_axes_name,
            4,
            2,
            public_inputs,
        )
        not in {(2, 3), (3, 2)}
        or mean1_axes_name in public_outputs
        or mean1_contract is None
        or mean1_name in public_names
        or not _producer_is_valid(graph_index, mean1_name, mean1_index)
        or graph_index.consumer_indices(mean1_name) != [sub_index]
        or [str(value) for value in sub.inputs] != [x_name, mean1_name]
        or len(sub.outputs) != 1
    ):
        return None
    centered_name = str(sub.outputs[0])
    centered = _contract(
        model_ir,
        centered_name,
        4,
        pre_output.shape,
        pre_output.signature,
    )
    if (
        centered is None
        or centered_name in public_names
        or not _producer_is_valid(graph_index, centered_name, sub_index)
    ):
        return None

    centered_user_indices = sorted(set(graph_index.consumer_indices(centered_name)))
    square_matches = [
        index
        for index in centered_user_indices
        if str(model_ir.operators[index].op_type) == "MUL"
        and [str(value) for value in model_ir.operators[index].inputs]
        == [centered_name, centered_name]
        and len(model_ir.operators[index].outputs) == 1
    ]
    norm_matches = [
        index
        for index in centered_user_indices
        if str(model_ir.operators[index].op_type) == "MUL"
        and len(model_ir.operators[index].inputs) == 2
        and len(model_ir.operators[index].outputs) == 1
        and Counter(str(value) for value in model_ir.operators[index].inputs)[
            centered_name
        ]
        == 1
    ]
    if len(square_matches) != 1 or len(norm_matches) != 1:
        return None
    square_index = int(square_matches[0])
    norm_index = int(norm_matches[0])
    if Counter(graph_index.consumer_indices(centered_name)) != Counter(
        [square_index, square_index, norm_index]
    ):
        return None
    square = model_ir.operators[square_index]
    squared_name = str(square.outputs[0])
    squared = _contract(
        model_ir,
        squared_name,
        4,
        pre_output.shape,
        pre_output.signature,
    )
    if (
        squared is None
        or squared_name in public_names
        or not _producer_is_valid(graph_index, squared_name, square_index)
    ):
        return None

    mean2_match = _sole_consumer(graph_index, squared_name)
    if mean2_match is None:
        return None
    mean2_index, mean2 = mean2_match
    mean2_name = str(mean2.outputs[0]) if len(mean2.outputs) == 1 else ""
    mean2_contract = _contract(
        model_ir,
        mean2_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    mean2_options = dict(mean2.options) if isinstance(mean2.options, dict) else {}
    mean2_axes_name = str(mean2.inputs[1]) if len(mean2.inputs) == 2 else ""
    if (
        str(mean2.op_type) != "MEAN"
        or len(mean2.inputs) != 2
        or len(mean2.outputs) != 1
        or str(mean2.inputs[0]) != squared_name
        or not bool(mean2_options.get("keepDims", False))
        or _normalized_axes(
            model_ir,
            graph_index,
            mean2_axes_name,
            4,
            2,
            public_inputs,
        )
        not in {(2, 3), (3, 2)}
        or mean2_axes_name in public_outputs
        or mean2_contract is None
        or mean2_name in public_names
        or not _producer_is_valid(graph_index, mean2_name, mean2_index)
    ):
        return None

    add_epsilon_match = _sole_consumer(graph_index, mean2_name)
    if add_epsilon_match is None:
        return None
    add_epsilon_index, add_epsilon = add_epsilon_match
    epsilon_match = _binary_other_input(add_epsilon, mean2_name)
    add_epsilon_name = (
        str(add_epsilon.outputs[0]) if len(add_epsilon.outputs) == 1 else ""
    )
    add_epsilon_contract = _contract(
        model_ir,
        add_epsilon_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    if (
        str(add_epsilon.op_type) != "ADD"
        or epsilon_match is None
        or len(add_epsilon.outputs) != 1
        or add_epsilon_contract is None
        or add_epsilon_name in public_names
        or not _producer_is_valid(
            graph_index,
            add_epsilon_name,
            add_epsilon_index,
        )
    ):
        return None
    epsilon_name = epsilon_match[0]

    sqrt_match = _sole_consumer(graph_index, add_epsilon_name)
    if sqrt_match is None:
        return None
    sqrt_index, sqrt = sqrt_match
    sqrt_name = str(sqrt.outputs[0]) if len(sqrt.outputs) == 1 else ""
    sqrt_contract = _contract(
        model_ir,
        sqrt_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    if (
        str(sqrt.op_type) != "SQRT"
        or [str(value) for value in sqrt.inputs] != [add_epsilon_name]
        or len(sqrt.outputs) != 1
        or sqrt_contract is None
        or sqrt_name in public_names
        or not _producer_is_valid(graph_index, sqrt_name, sqrt_index)
    ):
        return None

    div_match = _sole_consumer(graph_index, sqrt_name)
    if div_match is None:
        return None
    div_index, div = div_match
    div_name = str(div.outputs[0]) if len(div.outputs) == 1 else ""
    div_contract = _contract(
        model_ir,
        div_name,
        4,
        reduced_shape,
        reduced_signature,
    )
    if (
        str(div.op_type) != "DIV"
        or len(div.inputs) != 2
        or len(div.outputs) != 1
        or str(div.inputs[1]) != sqrt_name
        or div_contract is None
        or div_name in public_names
        or not _producer_is_valid(graph_index, div_name, div_index)
    ):
        return None
    one_name = str(div.inputs[0])

    norm = model_ir.operators[norm_index]
    norm_other = _binary_other_input(norm, centered_name)
    normalized_name = str(norm.outputs[0])
    normalized = _contract(
        model_ir,
        normalized_name,
        4,
        pre_output.shape,
        pre_output.signature,
    )
    if (
        norm_other is None
        or norm_other[0] != div_name
        or graph_index.consumer_indices(div_name) != [norm_index]
        or normalized is None
        or normalized_name in public_names
        or not _producer_is_valid(graph_index, normalized_name, norm_index)
    ):
        return None

    scale_match = _sole_consumer(graph_index, normalized_name)
    if scale_match is None:
        return None
    scale_index, scale = scale_match
    scale_constant_match = _binary_other_input(scale, normalized_name)
    scaled_name = str(scale.outputs[0]) if len(scale.outputs) == 1 else ""
    scaled = _contract(
        model_ir,
        scaled_name,
        4,
        pre_output.shape,
        pre_output.signature,
    )
    if (
        str(scale.op_type) != "MUL"
        or scale_constant_match is None
        or len(scale.outputs) != 1
        or scaled is None
        or scaled_name in public_names
        or not _producer_is_valid(graph_index, scaled_name, scale_index)
    ):
        return None
    scale_name, scale_input_index = scale_constant_match

    bias_match = _sole_consumer(graph_index, scaled_name)
    if bias_match is None:
        return None
    bias_index, add_bias = bias_match
    bias_constant_match = _binary_other_input(add_bias, scaled_name)
    inst_output_name = (
        str(add_bias.outputs[0]) if len(add_bias.outputs) == 1 else ""
    )
    inst_output = _contract(
        model_ir,
        inst_output_name,
        4,
        pre_output.shape,
        pre_output.signature,
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
    if bias_name == scale_name:
        return None

    if tail_mode not in {
        "direct",
        "side_squeeze",
        "unary_reshape",
        "residual_reshape",
    }:
        return None
    inst_output_users = sorted(set(graph_index.consumer_indices(inst_output_name)))
    side_squeeze: Optional[OperatorIR] = None
    side_contract: Optional[_TensorContract] = None
    side_adapter_permutation: Optional[TensorIR] = None
    tail_squeeze: Optional[OperatorIR] = None
    tail_squeeze_contract: Optional[_TensorContract] = None
    tail_unary: Optional[OperatorIR] = None
    tail_unary_contract: Optional[_TensorContract] = None
    tail_reshape: Optional[OperatorIR] = None
    tail_reshape_contract: Optional[_TensorContract] = None
    tail_reshape_shape_name: Optional[str] = None
    tail_add: Optional[OperatorIR] = None
    tail_add_contract: Optional[_TensorContract] = None
    residual_source: Optional[_ResidualSourcePlan] = None
    residual_metadata_updates: Tuple[_TensorMetadataUpdate, ...] = ()
    fanout_adapter: Optional[_FanoutAdapterPlan] = None
    if tail_mode in {"direct", "side_squeeze"}:
        post_matches = []
        side_matches = []
        for user_index in inst_output_users:
            user = model_ir.operators[int(user_index)]
            if (
                str(user.op_type) == "TRANSPOSE"
                and len(user.inputs) == 2
                and len(user.outputs) == 1
                and str(user.inputs[0]) == inst_output_name
                and _constant_vector(
                    model_ir,
                    graph_index,
                    str(user.inputs[1]),
                    4,
                    public_inputs,
                )
                == _PERM_NCHW_TO_NHWC
                and str(user.inputs[1]) not in public_outputs
            ):
                post_matches.append((int(user_index), user))
            if (
                str(user.op_type) == "SQUEEZE"
                and len(user.inputs) == 1
                and len(user.outputs) == 1
                and str(user.inputs[0]) == inst_output_name
            ):
                side_matches.append((int(user_index), user))
        if len(post_matches) != 1:
            return None
        post_index, post = post_matches[0]
        if tail_mode == "direct":
            if inst_output_users != [post_index]:
                return None
        else:
            if len(side_matches) != 1:
                return None
            side_index, side_squeeze = side_matches[0]
            if inst_output_users != sorted([post_index, side_index]):
                return None
            side_output_name = str(side_squeeze.outputs[0])
            side_contract = _contract(
                model_ir,
                side_output_name,
                3,
                inst_output.shape[1:],
                inst_output.signature[1:],
            )
            if (
                side_contract is None
                or side_output_name in public_inputs
                or not _producer_is_valid(
                    graph_index,
                    side_output_name,
                    side_index,
                )
                or not _squeeze_axis_zero(
                    side_squeeze,
                    inst_output,
                    side_contract,
                )
                or any(
                    int(consumer_index) <= side_index
                    for consumer_index in graph_index.consumer_indices(
                        side_output_name
                    )
                )
            ):
                return None
            existing_adapter_perm = model_ir.tensors.get(_SIDE_ADAPTER_PERM_NAME)
            if existing_adapter_perm is None:
                side_adapter_permutation = TensorIR(
                    name=_SIDE_ADAPTER_PERM_NAME,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(_PERM_NHWC_TO_NCHW, dtype=np.int32),
                    is_variable=False,
                    quantization=None,
                )
            elif (
                _constant_vector(
                    model_ir,
                    graph_index,
                    _SIDE_ADAPTER_PERM_NAME,
                    4,
                    public_inputs,
                )
                != _PERM_NHWC_TO_NCHW
                or _SIDE_ADAPTER_PERM_NAME in public_outputs
                or existing_adapter_perm.quantization is not None
            ):
                return None
    elif tail_mode == "unary_reshape":
        tail_squeeze_match = _sole_consumer(graph_index, inst_output_name)
        if tail_squeeze_match is None:
            return None
        tail_squeeze_index, tail_squeeze = tail_squeeze_match
        tail_squeeze_name = (
            str(tail_squeeze.outputs[0])
            if len(tail_squeeze.outputs) == 1
            else ""
        )
        tail_squeeze_contract = _contract(
            model_ir,
            tail_squeeze_name,
            3,
            inst_output.shape[1:],
            inst_output.signature[1:],
        )
        if (
            str(tail_squeeze.op_type) != "SQUEEZE"
            or len(tail_squeeze.inputs) != 1
            or len(tail_squeeze.outputs) != 1
            or str(tail_squeeze.inputs[0]) != inst_output_name
            or tail_squeeze_contract is None
            or tail_squeeze_name in public_names
            or not _producer_is_valid(
                graph_index,
                tail_squeeze_name,
                tail_squeeze_index,
            )
            or not _squeeze_axis_zero(
                tail_squeeze,
                inst_output,
                tail_squeeze_contract,
            )
        ):
            return None
        tail_unary_match = _sole_consumer(graph_index, tail_squeeze_name)
        if tail_unary_match is None:
            return None
        tail_unary_index, tail_unary = tail_unary_match
        tail_unary_name = (
            str(tail_unary.outputs[0]) if len(tail_unary.outputs) == 1 else ""
        )
        tail_unary_contract = _contract(
            model_ir,
            tail_unary_name,
            3,
            inst_output.shape[1:],
            inst_output.signature[1:],
        )
        if (
            str(tail_unary.op_type) not in _TAIL_UNARY_OPS
            or len(tail_unary.inputs) != 1
            or len(tail_unary.outputs) != 1
            or str(tail_unary.inputs[0]) != tail_squeeze_name
            or tail_unary_contract is None
            or tail_unary_name in public_names
            or not _producer_is_valid(
                graph_index,
                tail_unary_name,
                tail_unary_index,
            )
        ):
            return None
        tail_reshape_match = _sole_consumer(graph_index, tail_unary_name)
        if tail_reshape_match is None:
            return None
        tail_reshape_index, tail_reshape = tail_reshape_match
        tail_reshape_name = (
            str(tail_reshape.outputs[0])
            if len(tail_reshape.outputs) == 1
            else ""
        )
        tail_reshape_contract = _contract(
            model_ir,
            tail_reshape_name,
            4,
            inst_output.shape,
            inst_output.signature,
        )
        if (
            str(tail_reshape.op_type) != "RESHAPE"
            or len(tail_reshape.inputs) != 2
            or len(tail_reshape.outputs) != 1
            or str(tail_reshape.inputs[0]) != tail_unary_name
            or tail_reshape_contract is None
            or tail_reshape_name in public_names
            or not _producer_is_valid(
                graph_index,
                tail_reshape_name,
                tail_reshape_index,
            )
        ):
            return None
        tail_reshape_shape_name = str(tail_reshape.inputs[1])
        if (
            _constant_vector(
                model_ir,
                graph_index,
                tail_reshape_shape_name,
                4,
                public_inputs,
            )
            != tail_reshape_contract.shape
            or tail_reshape_shape_name in public_outputs
            or model_ir.tensors[tail_reshape_shape_name].quantization is not None
        ):
            return None
        post_match = _sole_consumer(graph_index, tail_reshape_name)
        if post_match is None:
            return None
        post_index, post = post_match
        if (
            str(post.op_type) != "TRANSPOSE"
            or len(post.inputs) != 2
            or len(post.outputs) != 1
            or str(post.inputs[0]) != tail_reshape_name
            or _constant_vector(
                model_ir,
                graph_index,
                str(post.inputs[1]),
                4,
                public_inputs,
            )
            != _PERM_NCHW_TO_NHWC
            or str(post.inputs[1]) in public_outputs
        ):
            return None
    else:
        residual_match = _resolve_residual_tail(
            model_ir,
            graph_index,
            inst_output_name=inst_output_name,
            inst_output=inst_output,
            public_inputs=public_inputs,
            public_outputs=public_outputs,
        )
        if residual_match is None:
            return None
        post_index = residual_match.post_index
        post = residual_match.post
        tail_squeeze = residual_match.squeeze
        tail_squeeze_contract = residual_match.squeeze_contract
        tail_add = residual_match.add
        tail_add_contract = residual_match.add_contract
        tail_reshape = residual_match.reshape
        tail_reshape_contract = residual_match.reshape_contract
        tail_reshape_shape_name = residual_match.reshape_shape_name
        residual_source = residual_match.residual_source
        residual_metadata_updates = residual_match.residual_metadata_updates
        fanout_adapter = residual_match.fanout_adapter
    post_output_name = str(post.outputs[0]) if len(post.outputs) == 1 else ""
    post_output = _contract(
        model_ir,
        post_output_name,
        4,
        source.shape,
        source.signature,
    )
    if (
        post_output is None
        or post_output_name in public_names
        or not _producer_is_valid(graph_index, post_output_name, post_index)
        or any(
            int(consumer_index) <= post_index
            for consumer_index in graph_index.consumer_indices(post_output_name)
        )
    ):
        return None

    core_ops = (
        pre,
        squeeze,
        reshape,
        mean1,
        sub,
        square,
        mean2,
        add_epsilon,
        sqrt,
        div,
        norm,
        scale,
        add_bias,
    )
    tail_ops = [post]
    if side_squeeze is not None:
        tail_ops.append(side_squeeze)
    if tail_squeeze is not None and tail_reshape is not None:
        tail_ops.append(tail_squeeze)
        if tail_unary is not None:
            tail_ops.append(tail_unary)
        if tail_add is not None:
            tail_ops.append(tail_add)
        tail_ops.append(tail_reshape)
    ordered_ops = core_ops + tuple(
        sorted(
            tail_ops,
            key=lambda operator: int(
                graph_index.operator_index(operator)
                if graph_index.operator_index(operator) is not None
                else len(model_ir.operators)
            ),
        )
    )
    ordered_indices = [
        graph_index.operator_index(operator) for operator in ordered_ops
    ]
    if (
        any(index is None for index in ordered_indices)
        or [int(index) for index in ordered_indices if index is not None]
        != sorted(int(index) for index in ordered_indices if index is not None)
        or len({id(operator) for operator in ordered_ops}) != len(ordered_ops)
    ):
        return None

    core_contracts = (
        source,
        pre_output,
        squeeze_output,
        x,
        mean1_contract,
        centered,
        squared,
        mean2_contract,
        add_epsilon_contract,
        sqrt_contract,
        div_contract,
        normalized,
        scaled,
        inst_output,
    )
    if tail_mode == "direct":
        core_contracts += (post_output,)
    elif tail_mode == "side_squeeze":
        assert side_contract is not None
        core_contracts += (post_output, side_contract)
    elif tail_mode == "unary_reshape":
        assert tail_squeeze_contract is not None
        core_contracts += (tail_squeeze_contract,)
    else:
        assert tail_squeeze_contract is not None
        assert tail_reshape_contract is not None
        assert tail_add is not None
        assert tail_add_contract is not None
        core_contracts += (
            tail_squeeze_contract,
            tail_add_contract,
            tail_reshape_contract,
            post_output,
        )
    dtype = str(source.tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(
            str(contract.tensor.dtype) != dtype
            for contract in core_contracts
        )
        or any(
            contract.tensor.quantization is not None
            for contract in core_contracts
        )
    ):
        return None
    if tail_mode == "unary_reshape":
        assert tail_unary is not None
        assert tail_unary_contract is not None
        assert tail_reshape_contract is not None
        tail_dtype = str(tail_unary_contract.tensor.dtype)
        if (
            not tail_dtype
            or (
                str(tail_unary.op_type) != "CAST"
                and tail_dtype != dtype
            )
            or any(
                str(contract.tensor.dtype) != tail_dtype
                for contract in (
                    tail_unary_contract,
                    tail_reshape_contract,
                    post_output,
                )
            )
            or any(
                contract.tensor.quantization is not None
                for contract in (
                    tail_unary_contract,
                    tail_reshape_contract,
                    post_output,
                )
            )
        ):
            return None
    channel_count = int(pre_output.shape[1])
    old_coefficient_shape = (1, channel_count, 1, 1)
    scale_data = _float_constant(
        model_ir,
        graph_index,
        scale_name,
        dtype,
        shape=old_coefficient_shape,
    )
    bias_data = _float_constant(
        model_ir,
        graph_index,
        bias_name,
        dtype,
        shape=old_coefficient_shape,
    )
    if (
        _float_constant(
            model_ir,
            graph_index,
            epsilon_name,
            dtype,
            nonnegative=True,
        )
        is None
        or _float_constant(
            model_ir,
            graph_index,
            one_name,
            dtype,
            value=1.0,
        )
        is None
        or scale_data is None
        or bias_data is None
    ):
        return None

    constant_updates = []
    reshape_update = _plan_constant_update(
        model_ir,
        graph_index,
        reshape_shape_name,
        np.asarray(source.shape, dtype=np.asarray(
            model_ir.tensors[reshape_shape_name].data
        ).dtype),
        (_ConstantUse(reshape, 1),),
        "nhwc_shape",
        public_names,
    )
    if reshape_update is None:
        return None
    constant_updates.append(reshape_update)
    if mean1_axes_name == mean2_axes_name:
        axes_updates = (
            (
                mean1_axes_name,
                (_ConstantUse(mean1, 1), _ConstantUse(mean2, 1)),
            ),
        )
    else:
        axes_updates = (
            (mean1_axes_name, (_ConstantUse(mean1, 1),)),
            (mean2_axes_name, (_ConstantUse(mean2, 1),)),
        )
    for axes_name, uses in axes_updates:
        update = _plan_constant_update(
            model_ir,
            graph_index,
            axes_name,
            np.asarray(
                [1, 2],
                dtype=np.asarray(model_ir.tensors[axes_name].data).dtype,
            ),
            uses,
            "nhwc_axes",
            public_names,
        )
        if update is None:
            return None
        constant_updates.append(update)
    for name, data, use, suffix in (
        (
            scale_name,
            np.transpose(scale_data, _PERM_NCHW_TO_NHWC),
            _ConstantUse(scale, scale_input_index),
            "nhwc_scale",
        ),
        (
            bias_name,
            np.transpose(bias_data, _PERM_NCHW_TO_NHWC),
            _ConstantUse(add_bias, bias_input_index),
            "nhwc_bias",
        ),
    ):
        update = _plan_constant_update(
            model_ir,
            graph_index,
            name,
            data,
            (use,),
            suffix,
            public_names,
        )
        if update is None:
            return None
        constant_updates.append(update)
    tail_reshape_options: Optional[Dict[str, object]] = None
    if tail_mode in {"unary_reshape", "residual_reshape"}:
        assert tail_reshape is not None
        assert tail_reshape_shape_name is not None
        update = _plan_constant_update(
            model_ir,
            graph_index,
            tail_reshape_shape_name,
            np.asarray(
                source.shape,
                dtype=np.asarray(
                    model_ir.tensors[tail_reshape_shape_name].data
                ).dtype,
            ),
            (_ConstantUse(tail_reshape, 1),),
            "nhwc_tail_shape",
            public_names,
        )
        if update is None:
            return None
        constant_updates.append(update)
        tail_reshape_options = (
            dict(tail_reshape.options)
            if isinstance(tail_reshape.options, dict)
            else {}
        )
        tail_reshape_options["newShape"] = [
            int(value) for value in source.shape
        ]

    full_contracts = (
        x,
        centered,
        squared,
        normalized,
        scaled,
    ) + (
        (inst_output,)
        if tail_mode in {"direct", "unary_reshape", "residual_reshape"}
        else ()
    )
    if tail_reshape_contract is not None:
        full_contracts += (tail_reshape_contract,)
    reduced_contracts = (
        mean1_contract,
        mean2_contract,
        add_epsilon_contract,
        sqrt_contract,
        div_contract,
    )
    metadata_updates = [
        _TensorMetadataUpdate(
            squeeze_output,
            tuple(squeeze_output.shape[index] for index in _PERM_CHW_TO_HWC),
            tuple(
                squeeze_output.signature[index]
                for index in _PERM_CHW_TO_HWC
            ),
        )
    ]
    metadata_updates.extend(
        _TensorMetadataUpdate(
            contract,
            tuple(contract.shape[index] for index in _PERM_NCHW_TO_NHWC),
            tuple(
                contract.signature[index]
                for index in _PERM_NCHW_TO_NHWC
            ),
        )
        for contract in full_contracts + reduced_contracts
    )
    tail_rank3_contracts = tuple(
        contract
        for contract in (
            tail_squeeze_contract,
            tail_unary_contract,
            tail_add_contract,
        )
        if contract is not None
    )
    metadata_updates.extend(
        _TensorMetadataUpdate(
            contract,
            tuple(contract.shape[index] for index in _PERM_CHW_TO_HWC),
            tuple(
                contract.signature[index]
                for index in _PERM_CHW_TO_HWC
            ),
        )
        for contract in tail_rank3_contracts
    )
    metadata_updates.extend(residual_metadata_updates)
    channel_last_names = tuple(
        dict.fromkeys(
            [
                squeeze_output_name,
                *(contract.tensor.name for contract in full_contracts),
                *(contract.tensor.name for contract in reduced_contracts),
                *(contract.tensor.name for contract in tail_rank3_contracts),
                *(
                    update.contract.tensor.name
                    for update in residual_metadata_updates
                ),
                post_output_name,
            ]
        )
    )
    reshape_options = (
        dict(reshape.options) if isinstance(reshape.options, dict) else {}
    )
    reshape_options["newShape"] = [int(value) for value in source.shape]
    return _InstanceNormPrepostPlan(
        ordered_ops=ordered_ops,
        pre=pre,
        squeeze=squeeze,
        reshape=reshape,
        reshape_options=reshape_options,
        source_name=source_name,
        mean1=mean1,
        sub=sub,
        add_bias=add_bias,
        post=post,
        post_output_name=post_output_name,
        tail_mode=tail_mode,
        side_squeeze=side_squeeze,
        side_adapter_permutation=side_adapter_permutation,
        tail_squeeze=tail_squeeze,
        tail_unary=tail_unary,
        tail_reshape=tail_reshape,
        tail_reshape_options=tail_reshape_options,
        tail_add=tail_add,
        residual_source=residual_source,
        fanout_adapter=fanout_adapter,
        constant_updates=tuple(constant_updates),
        metadata_updates=tuple(metadata_updates),
        channel_last_names=channel_last_names,
    )


def _apply_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    update: _ConstantUpdate,
) -> bool:
    target = update.tensor
    if update.clone_name is not None:
        if update.clone is None or update.clone_name in model_ir.tensors:
            return False
        model_ir.tensors[update.clone_name] = update.clone
        target = update.clone
        for use in update.uses:
            _replace_operator_input_at(
                model_ir=model_ir,
                op=use.operator,
                input_index=use.input_index,
                new_input_name=update.clone_name,
                graph_index=graph_index,
            )
    target.data = np.asarray(update.data)
    target.shape = [int(value) for value in update.data.shape]
    target.shape_signature = [int(value) for value in update.data.shape]
    return True


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _InstanceNormPrepostPlan,
) -> bool:
    indices = [graph_index.operator_index(operator) for operator in plan.ordered_ops]
    if any(index is None for index in indices):
        return False
    resolved = [int(index) for index in indices if index is not None]
    if resolved != sorted(resolved) or len(set(resolved)) != len(resolved):
        return False
    clone_names = [
        update.clone_name
        for update in plan.constant_updates
        if update.clone_name is not None
    ]
    residual_bridge_index: Optional[int] = None
    if plan.residual_source is not None:
        residual_bridge_index = graph_index.operator_index(
            plan.residual_source.bridge
        )
        if (
            residual_bridge_index is None
            or plan.residual_source.bridge in plan.ordered_ops
        ):
            return False
    fanout_use_indices = []
    if plan.fanout_adapter is not None:
        if (
            plan.tail_add is None
            or plan.fanout_adapter.tensor.name in model_ir.tensors
            or (
                plan.fanout_adapter.permutation is not None
                and _RESIDUAL_ADAPTER_PERM_NAME in model_ir.tensors
            )
        ):
            return False
        tail_add_name = str(plan.tail_add.outputs[0])
        for use in plan.fanout_adapter.uses:
            use_index = graph_index.operator_index(use.operator)
            if (
                use_index is None
                or int(use.input_index) < 0
                or int(use.input_index) >= len(use.operator.inputs)
                or str(use.operator.inputs[int(use.input_index)])
                != tail_add_name
            ):
                return False
            fanout_use_indices.append(int(use_index))
        if not fanout_use_indices:
            return False
    if (
        len(clone_names) != len(set(clone_names))
        or any(name in model_ir.tensors for name in clone_names)
        or any(
            update.clone_name is not None and update.clone is None
            for update in plan.constant_updates
        )
        or (
            plan.tail_mode == "side_squeeze"
            and plan.side_squeeze is None
        )
        or (
            plan.tail_mode == "unary_reshape"
            and (
                plan.tail_squeeze is None
                or plan.tail_unary is None
                or plan.tail_reshape is None
                or plan.tail_reshape_options is None
            )
        )
        or (
            plan.tail_mode == "residual_reshape"
            and (
                plan.tail_squeeze is None
                or plan.tail_add is None
                or plan.tail_reshape is None
                or plan.tail_reshape_options is None
                or plan.residual_source is None
                or residual_bridge_index is None
            )
        )
        or (
            plan.side_adapter_permutation is not None
            and _SIDE_ADAPTER_PERM_NAME in model_ir.tensors
        )
    ):
        return False
    for update in plan.constant_updates:
        if not _apply_constant_update(model_ir, graph_index, update):
            return False

    if plan.residual_source is not None:
        assert plan.tail_add is not None
        if plan.residual_source.add_replacement_name is not None:
            _replace_operator_input_at(
                model_ir=model_ir,
                op=plan.tail_add,
                input_index=plan.residual_source.add_input_index,
                new_input_name=plan.residual_source.add_replacement_name,
                graph_index=graph_index,
            )
        else:
            assert plan.residual_source.squeeze is not None
            assert plan.residual_source.squeeze_source_name is not None
            _set_operator_inputs(
                model_ir=model_ir,
                op=plan.residual_source.squeeze,
                new_inputs=[plan.residual_source.squeeze_source_name],
                graph_index=graph_index,
            )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.mean1,
        input_index=0,
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.squeeze,
        new_inputs=[plan.source_name],
        graph_index=graph_index,
    )
    plan.reshape.options = dict(plan.reshape_options)
    if plan.tail_mode in {"unary_reshape", "residual_reshape"}:
        assert plan.tail_reshape is not None
        assert plan.tail_reshape_options is not None
        plan.tail_reshape.options = dict(plan.tail_reshape_options)
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.sub,
        input_index=0,
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )
    for update in plan.metadata_updates:
        update.contract.tensor.shape = list(update.shape)
        update.contract.tensor.shape_signature = list(update.signature)
    output_owner = (
        plan.tail_reshape
        if plan.tail_mode in {"unary_reshape", "residual_reshape"}
        else plan.add_bias
    )
    assert output_owner is not None
    _set_operator_outputs(
        model_ir=model_ir,
        op=output_owner,
        new_outputs=[plan.post_output_name],
        graph_index=graph_index,
    )
    remove_indices = [
        graph_index.operator_index(plan.pre),
        graph_index.operator_index(plan.post),
    ]
    if residual_bridge_index is not None:
        remove_indices.append(residual_bridge_index)
    if any(index is None for index in remove_indices):
        return False
    graph_index.remove_operators(
        [int(index) for index in remove_indices if index is not None]
    )
    if plan.tail_mode == "side_squeeze":
        assert plan.side_squeeze is not None
        if plan.side_adapter_permutation is not None:
            model_ir.tensors[_SIDE_ADAPTER_PERM_NAME] = (
                plan.side_adapter_permutation
            )
        side_index = graph_index.operator_index(plan.side_squeeze)
        if side_index is None:
            return False
        graph_index.insert_operator(
            int(side_index),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[plan.post_output_name, _SIDE_ADAPTER_PERM_NAME],
                outputs=[str(plan.side_squeeze.inputs[0])],
            ),
        )
    if plan.fanout_adapter is not None:
        assert plan.tail_add is not None
        if plan.fanout_adapter.permutation is not None:
            model_ir.tensors[_RESIDUAL_ADAPTER_PERM_NAME] = (
                plan.fanout_adapter.permutation
            )
        model_ir.tensors[plan.fanout_adapter.tensor.name] = (
            plan.fanout_adapter.tensor
        )
        current_use_indices = []
        for use in plan.fanout_adapter.uses:
            use_index = graph_index.operator_index(use.operator)
            assert use_index is not None
            new_inputs = [str(value) for value in use.operator.inputs]
            new_inputs[int(use.input_index)] = plan.fanout_adapter.tensor.name
            _set_operator_inputs(
                model_ir=model_ir,
                op=use.operator,
                new_inputs=new_inputs,
                graph_index=graph_index,
            )
            current_use_indices.append(int(use_index))
        graph_index.insert_operator(
            min(current_use_indices),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[
                    str(plan.tail_add.outputs[0]),
                    _RESIDUAL_ADAPTER_PERM_NAME,
                ],
                outputs=[plan.fanout_adapter.tensor.name],
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


def _run_indexed_instance_norm_prepost_tail(
    model_ir: ModelIR,
    *,
    tail_mode: str,
    stats_key: str,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    if candidate is None:
        counts = Counter(str(operator.op_type) for operator in model_ir.operators)
        has_extended_tail = tail_mode in {
            "side_squeeze",
            "unary_reshape",
            "residual_reshape",
        }
        required = {
            "TRANSPOSE": 3 if tail_mode == "residual_reshape" else 2,
            "SQUEEZE": 2 if has_extended_tail else 1,
            "RESHAPE": (
                2
                if tail_mode in {"unary_reshape", "residual_reshape"}
                else 1
            ),
            "MEAN": 2,
            "SUB": 1,
            "MUL": 3,
            "ADD": 3 if tail_mode == "residual_reshape" else 2,
            "SQRT": 1,
            "DIV": 1,
        }
        if any(
            counts[op_type] < minimum
            for op_type, minimum in required.items()
        ):
            return {stats_key: 0}
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
        plan = _resolve_candidate(
            model_ir,
            active_index,
            pre_index,
            tail_mode=tail_mode,
        )
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1
    if rewritten:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {stats_key: int(rewritten)}


def _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _run_indexed_instance_norm_prepost_tail(
        model_ir,
        tail_mode="direct",
        stats_key=_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _run_indexed_instance_norm_prepost_tail(
        model_ir,
        tail_mode="side_squeeze",
        stats_key=_SIDE_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _run_indexed_instance_norm_prepost_tail(
        model_ir,
        tail_mode="unary_reshape",
        stats_key=_UNARY_RESHAPE_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _run_indexed_instance_norm_prepost_tail(
        model_ir,
        tail_mode="residual_reshape",
        stats_key=_RESIDUAL_RESHAPE_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )
