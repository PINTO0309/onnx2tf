from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _replace_operator_input_at,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    _FLOAT_DTYPES,
    _NCHW_TO_NHWC,
    _NHWC_TO_NCHW,
    _constant_replacement as _broadcast_constant_replacement,
    _data_and_constant_inputs,
    _permute,
    _plain_binary,
    _tensor_contract,
    _typed_permutation,
)
from onnx2tf.tflite_builder.passes.sinet_concat_resize_layout import (
    _InputRewrite,
    _input_rewrites,
    _layout_allows,
)
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    _ConstantPlan,
    _MetadataUpdate,
    _apply_constant_plans,
    _apply_metadata_updates,
    _constant_plans_equal,
    _metadata_update,
    _plan_constants,
    _plain_prelu,
    _producer,
    _resolved_source,
)


_STATS_KEY = "optimized_sinet_softmax_mask_residual_nhwc_tail_chains"
_NCHW_TO_NWHC = (0, 3, 2, 1)


@dataclass(frozen=True)
class _AdapterMatch:
    pre: OperatorIR
    source_name: str
    nchw_name: str


@dataclass(frozen=True)
class _SoftmaxMaskPlan:
    root: OperatorIR
    post_adapters: Tuple[OperatorIR, ...]
    legacy_consumers: Tuple[OperatorIR, ...]
    side_adapter: _AdapterMatch
    main_adapter: _AdapterMatch
    side_prelu: OperatorIR
    main_mul: OperatorIR
    main_add: OperatorIR
    soft_pre: OperatorIR
    softmax: OperatorIR
    soft_post: OperatorIR
    reduce_max: OperatorIR
    sub: OperatorIR
    reshape: OperatorIR
    expand_mul: OperatorIR
    side_mask_mul: OperatorIR
    residual_add: OperatorIR
    main_mul_data_index: int
    main_name: str
    soft_back_name: str
    residual_output_name: str
    post_output_name: str
    reshape_options: dict[str, object]
    constant_plans: Tuple[_ConstantPlan, ...]
    metadata_updates: Tuple[_MetadataUpdate, ...]
    alias_rewrites: Tuple[_InputRewrite, ...]
    remove_operators: Tuple[OperatorIR, ...]
    insert_legacy_adapter: bool
    legacy_adapter_perm_name: Optional[str]


def _data_tensor(
    model_ir: ModelIR,
    name: str,
    *,
    rank: int,
    dtype: Optional[str] = None,
) -> Optional[Tuple[TensorIR, Tuple[int, ...], Tuple[int, ...]]]:
    contract = _tensor_contract(model_ir, str(name), int(rank))
    if contract is None:
        return None
    tensor = contract.tensor
    normalized_dtype = str(tensor.dtype)
    if (
        normalized_dtype not in _FLOAT_DTYPES
        or (dtype is not None and normalized_dtype != str(dtype))
        or tensor.is_variable
        or tensor.quantization is not None
    ):
        return None
    return tensor, contract.shape, contract.signature


def _exact_data_contract(
    model_ir: ModelIR,
    name: str,
    *,
    dtype: str,
    shape: Tuple[int, ...],
    signature: Tuple[int, ...],
    layout: Optional[str] = None,
) -> Optional[TensorIR]:
    contract = _data_tensor(
        model_ir,
        str(name),
        rank=len(shape),
        dtype=str(dtype),
    )
    if contract is None:
        return None
    tensor, actual_shape, actual_signature = contract
    if (
        actual_shape != tuple(shape)
        or actual_signature != tuple(signature)
        or (layout is not None and not _layout_allows(tensor, str(layout)))
    ):
        return None
    return tensor


def _int_constant_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    old_values: Tuple[int, ...],
    new_values: Tuple[int, ...],
    public_names: set[str],
) -> Optional[np.ndarray]:
    tensor = model_ir.tensors.get(str(name))
    if tensor is None or tensor.data is None:
        return None
    try:
        data = np.asarray(tensor.data)
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
    except (TypeError, ValueError):
        return None
    expected_dtype = {
        "INT32": np.dtype(np.int32),
        "INT64": np.dtype(np.int64),
    }.get(str(tensor.dtype))
    if (
        expected_dtype is None
        or str(name) in public_names
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or tensor.is_variable
        or tensor.quantization is not None
        or data.dtype != expected_dtype
        or data.shape != (len(old_values),)
        or shape != data.shape
        or signature != shape
        or tuple(int(value) for value in data.tolist()) != tuple(old_values)
    ):
        return None
    return np.asarray(new_values, dtype=expected_dtype)


def _singleton_float_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    dtype: str,
    public_names: set[str],
) -> bool:
    tensor = model_ir.tensors.get(str(name))
    expected_dtype = _FLOAT_DTYPES.get(str(dtype))
    if tensor is None or tensor.data is None or expected_dtype is None:
        return False
    try:
        data = np.asarray(tensor.data)
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
    except (TypeError, ValueError):
        return False
    return bool(
        str(name) not in public_names
        and str(name) not in graph_index.producers
        and str(name) not in graph_index.duplicate_producers
        and str(tensor.dtype) == str(dtype)
        and not tensor.is_variable
        and tensor.quantization is None
        and data.dtype == expected_dtype
        and int(data.size) == 1
        and shape in {(), (1,)}
        and signature == shape
        and np.all(np.isfinite(data))
    )


def _resolve_adapter(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    nchw_name: str,
    consumer_index: int,
    public_inputs: set[str],
    public_outputs: set[str],
) -> Optional[_AdapterMatch]:
    public_names = public_inputs | public_outputs
    match = _producer(model_ir, graph_index, str(nchw_name), "TRANSPOSE")
    if match is None:
        return None
    pre_index, pre = match
    if (
        int(pre_index) >= int(consumer_index)
        or str(nchw_name) in public_names
        or graph_index.consumer_indices(str(nchw_name))
        != [int(consumer_index)]
        or not _typed_permutation(
            model_ir,
            graph_index,
            pre,
            _NHWC_TO_NCHW,
            public_names,
        )
    ):
        return None
    source_name = str(pre.inputs[0])
    if not _resolved_source(
        graph_index,
        name=source_name,
        adapter_index=int(pre_index),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    ):
        return None
    return _AdapterMatch(pre, source_name, str(nchw_name))


def _sole_consumer(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    producer_index: int,
    op_type: str,
    public_names: set[str],
) -> Optional[Tuple[int, OperatorIR]]:
    consumers = graph_index.consumer_indices(str(name))
    if (
        str(name) in public_names
        or str(name) in graph_index.duplicate_producers
        or graph_index.producers.get(str(name)) != int(producer_index)
        or len(consumers) != 1
    ):
        return None
    consumer_index = int(consumers[0])
    if consumer_index <= int(producer_index):
        return None
    consumer = model_ir.operators[consumer_index]
    if (
        str(consumer.op_type) != str(op_type)
        or len(consumer.outputs) != 1
        or str(consumer.outputs[0]) in graph_index.duplicate_producers
        or graph_index.producers.get(str(consumer.outputs[0]))
        != int(consumer_index)
    ):
        return None
    return consumer_index, consumer


def _producer_role(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    op_type: str,
) -> Optional[Tuple[int, str, int, OperatorIR]]:
    matches = []
    for input_index, input_name in enumerate(operator.inputs):
        producer = _producer(
            model_ir,
            graph_index,
            str(input_name),
            str(op_type),
        )
        if producer is not None:
            matches.append(
                (int(input_index), str(input_name), int(producer[0]), producer[1])
            )
    return matches[0] if len(matches) == 1 else None


def _reshape_options(
    operator: OperatorIR,
    *,
    old_shape: Tuple[int, ...],
    new_shape: Tuple[int, ...],
) -> Optional[dict[str, object]]:
    if not isinstance(operator.options, dict):
        return None
    options = dict(operator.options)
    for key in ("newShape", "onnxRawNewShape"):
        value = options.get(key)
        if value is None:
            continue
        if not isinstance(value, (list, tuple)):
            return None
        try:
            normalized = tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
        if normalized != tuple(old_shape):
            return None
        options[key] = [int(item) for item in new_shape]
    return options


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    root: OperatorIR,
) -> Optional[_SoftmaxMaskPlan]:
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    root_index = graph_index.operator_index(root)
    if (
        root_index is None
        or str(root.op_type) != "TRANSPOSE"
        or len(root.outputs) != 1
        or str(root.outputs[0]) in public_names
        or graph_index.producers.get(str(root.outputs[0])) != int(root_index)
        or str(root.outputs[0]) in graph_index.duplicate_producers
        or not _typed_permutation(
            model_ir,
            graph_index,
            root,
            _NCHW_TO_NHWC,
            public_names,
        )
    ):
        return None

    residual_output_name = str(root.inputs[0])
    residual_match = _producer(
        model_ir,
        graph_index,
        residual_output_name,
        "ADD",
    )
    if residual_match is None or residual_output_name in public_names:
        return None
    residual_index, residual_add = residual_match
    if (
        not _plain_binary(residual_add, "ADD")
        or int(residual_index) >= int(root_index)
    ):
        return None

    post_adapters = []
    legacy_indices = []
    for consumer_index in sorted(
        set(graph_index.consumer_indices(residual_output_name))
    ):
        consumer = model_ir.operators[int(consumer_index)]
        if (
            int(consumer_index) > int(residual_index)
            and str(consumer.op_type) == "TRANSPOSE"
            and len(consumer.outputs) == 1
            and str(consumer.inputs[0]) == residual_output_name
            and str(consumer.outputs[0]) not in public_names
            and graph_index.producers.get(str(consumer.outputs[0]))
            == int(consumer_index)
            and str(consumer.outputs[0]) not in graph_index.duplicate_producers
            and _typed_permutation(
                model_ir,
                graph_index,
                consumer,
                _NCHW_TO_NHWC,
                public_names,
            )
        ):
            post_adapters.append((int(consumer_index), consumer))
        else:
            legacy_indices.append(int(consumer_index))
    if (
        not post_adapters
        or post_adapters[0][1] is not root
        or any(index <= int(residual_index) for index in legacy_indices)
    ):
        return None
    for adapter_index, adapter in post_adapters:
        if any(
            consumer_index <= int(adapter_index)
            for consumer_index in graph_index.consumer_indices(
                str(adapter.outputs[0])
            )
        ):
            return None

    main_role = _producer_role(
        model_ir, graph_index, residual_add, "ADD"
    )
    side_role = _producer_role(
        model_ir, graph_index, residual_add, "MUL"
    )
    if (
        main_role is None
        or side_role is None
        or main_role[0] == side_role[0]
        or main_role[2] >= int(residual_index)
        or side_role[2] >= int(residual_index)
    ):
        return None
    _, main_name, main_add_index, main_add = main_role
    _, side_mask_name, side_mask_index, side_mask_mul = side_role
    if (
        not _plain_binary(main_add, "ADD")
        or not _plain_binary(side_mask_mul, "MUL")
        or main_name in public_names
        or side_mask_name in public_names
        or graph_index.consumer_indices(side_mask_name) != [int(residual_index)]
    ):
        return None

    main_add_inputs = _data_and_constant_inputs(model_ir, main_add)
    if main_add_inputs is None:
        return None
    _, main_mul_output_name, main_add_constant_index, main_add_constant_name = (
        main_add_inputs
    )
    main_mul_match = _producer(
        model_ir,
        graph_index,
        main_mul_output_name,
        "MUL",
    )
    if main_mul_match is None:
        return None
    main_mul_index, main_mul = main_mul_match
    main_mul_inputs = _data_and_constant_inputs(model_ir, main_mul)
    if (
        main_mul_inputs is None
        or not _plain_binary(main_mul, "MUL")
        or int(main_mul_index) >= int(main_add_index)
        or graph_index.consumer_indices(main_mul_output_name)
        != [int(main_add_index)]
    ):
        return None
    (
        main_mul_data_index,
        main_pre_output_name,
        main_mul_constant_index,
        main_mul_constant_name,
    ) = main_mul_inputs
    main_adapter = _resolve_adapter(
        model_ir,
        graph_index,
        nchw_name=main_pre_output_name,
        consumer_index=int(main_mul_index),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if main_adapter is None:
        return None

    main_users = graph_index.consumer_indices(main_name)
    if len(main_users) != 2 or main_users.count(int(residual_index)) != 1:
        return None
    soft_candidates = sorted(
        set(index for index in main_users if index != int(residual_index))
    )
    if len(soft_candidates) != 1:
        return None
    soft_pre_index = int(soft_candidates[0])
    soft_pre = model_ir.operators[soft_pre_index]
    if (
        soft_pre_index <= int(main_add_index)
        or str(soft_pre.op_type) != "TRANSPOSE"
        or len(soft_pre.outputs) != 1
        or str(soft_pre.inputs[0]) != main_name
        or str(soft_pre.outputs[0]) in graph_index.duplicate_producers
        or graph_index.producers.get(str(soft_pre.outputs[0]))
        != int(soft_pre_index)
        or not _typed_permutation(
            model_ir,
            graph_index,
            soft_pre,
            _NCHW_TO_NWHC,
            public_names,
        )
    ):
        return None
    soft_pre_output_name = str(soft_pre.outputs[0])
    softmax_match = _sole_consumer(
        model_ir,
        graph_index,
        name=soft_pre_output_name,
        producer_index=soft_pre_index,
        op_type="SOFTMAX",
        public_names=public_names,
    )
    if softmax_match is None:
        return None
    softmax_index, softmax = softmax_match
    try:
        softmax_axis = int(softmax.options.get("axis", -1))
        softmax_beta = float(softmax.options.get("beta", 1.0))
    except (TypeError, ValueError):
        return None
    if (
        len(softmax.inputs) != 1
        or len(softmax.outputs) != 1
        or str(softmax.inputs[0]) != soft_pre_output_name
        or softmax_axis != 3
        or not np.isfinite(softmax_beta)
    ):
        return None
    softmax_output_name = str(softmax.outputs[0])
    soft_post_match = _sole_consumer(
        model_ir,
        graph_index,
        name=softmax_output_name,
        producer_index=softmax_index,
        op_type="TRANSPOSE",
        public_names=public_names,
    )
    if soft_post_match is None:
        return None
    soft_post_index, soft_post = soft_post_match
    if not _typed_permutation(
        model_ir,
        graph_index,
        soft_post,
        _NCHW_TO_NWHC,
        public_names,
    ):
        return None
    soft_back_name = str(soft_post.outputs[0])

    reduce_match = _sole_consumer(
        model_ir,
        graph_index,
        name=soft_back_name,
        producer_index=soft_post_index,
        op_type="REDUCE_MAX",
        public_names=public_names,
    )
    if reduce_match is None:
        return None
    reduce_index, reduce_max = reduce_match
    if (
        len(reduce_max.inputs) != 2
        or len(reduce_max.outputs) != 1
        or str(reduce_max.inputs[0]) != soft_back_name
        or bool(reduce_max.options.get("keepDims", False))
    ):
        return None
    reduce_axis_name = str(reduce_max.inputs[1])
    reduce_output_name = str(reduce_max.outputs[0])

    sub_match = _sole_consumer(
        model_ir,
        graph_index,
        name=reduce_output_name,
        producer_index=reduce_index,
        op_type="SUB",
        public_names=public_names,
    )
    if sub_match is None:
        return None
    sub_index, sub = sub_match
    if not _plain_binary(sub, "SUB"):
        return None
    sub_inputs = _data_and_constant_inputs(model_ir, sub)
    if sub_inputs is None or sub_inputs[1] != reduce_output_name:
        return None
    _, _, _, sub_constant_name = sub_inputs
    sub_output_name = str(sub.outputs[0])

    reshape_match = _sole_consumer(
        model_ir,
        graph_index,
        name=sub_output_name,
        producer_index=sub_index,
        op_type="RESHAPE",
        public_names=public_names,
    )
    if reshape_match is None:
        return None
    reshape_index, reshape = reshape_match
    if (
        len(reshape.inputs) != 2
        or len(reshape.outputs) != 1
        or str(reshape.inputs[0]) != sub_output_name
    ):
        return None
    reshape_shape_name = str(reshape.inputs[1])
    reshape_output_name = str(reshape.outputs[0])

    expand_match = _sole_consumer(
        model_ir,
        graph_index,
        name=reshape_output_name,
        producer_index=reshape_index,
        op_type="MUL",
        public_names=public_names,
    )
    if expand_match is None:
        return None
    expand_index, expand_mul = expand_match
    expand_inputs = _data_and_constant_inputs(model_ir, expand_mul)
    if (
        expand_inputs is None
        or not _plain_binary(expand_mul, "MUL")
        or expand_inputs[1] != reshape_output_name
    ):
        return None
    _, _, expand_constant_index, expand_constant_name = expand_inputs
    expand_output_name = str(expand_mul.outputs[0])

    if (
        graph_index.consumer_indices(expand_output_name)
        != [int(side_mask_index)]
        or not _plain_binary(side_mask_mul, "MUL")
    ):
        return None
    side_prelu_role = _producer_role(
        model_ir,
        graph_index,
        side_mask_mul,
        "PRELU",
    )
    if side_prelu_role is None:
        return None
    side_prelu_input_index, side_prelu_output_name, side_prelu_index, side_prelu = (
        side_prelu_role
    )
    if (
        not _plain_prelu(side_prelu)
        or side_prelu_input_index == list(side_mask_mul.inputs).index(
            expand_output_name
        )
        or int(side_prelu_index) >= int(side_mask_index)
        or graph_index.consumer_indices(side_prelu_output_name)
        != [int(side_mask_index)]
    ):
        return None
    side_adapter = _resolve_adapter(
        model_ir,
        graph_index,
        nchw_name=str(side_prelu.inputs[0]),
        consumer_index=int(side_prelu_index),
        public_inputs=public_inputs,
        public_outputs=public_outputs,
    )
    if side_adapter is None:
        return None

    post_output_name = str(root.outputs[0])
    canonical = _data_tensor(model_ir, post_output_name, rank=4)
    if canonical is None:
        return None
    canonical_tensor, nhwc_shape, nhwc_signature = canonical
    dtype = str(canonical_tensor.dtype)
    nchw_shape = _permute(nhwc_shape, _NHWC_TO_NCHW)
    nchw_signature = _permute(nhwc_signature, _NHWC_TO_NCHW)
    nwhc_shape = _permute(nchw_shape, _NCHW_TO_NWHC)
    nwhc_signature = _permute(nchw_signature, _NCHW_TO_NWHC)
    reduce_shape = (nchw_shape[0], nchw_shape[2], nchw_shape[3])
    reduce_signature = (
        nchw_signature[0],
        nchw_signature[2],
        nchw_signature[3],
    )
    old_reshape_shape = (
        nchw_shape[0],
        1,
        nchw_shape[2],
        nchw_shape[3],
    )
    old_reshape_signature = (
        nchw_signature[0],
        1,
        nchw_signature[2],
        nchw_signature[3],
    )
    new_reshape_shape = (
        nhwc_shape[0],
        nhwc_shape[1],
        nhwc_shape[2],
        1,
    )
    new_reshape_signature = (
        nhwc_signature[0],
        nhwc_signature[1],
        nhwc_signature[2],
        1,
    )
    if nhwc_shape[0] != 1:
        return None

    exact_contracts = (
        (main_adapter.source_name, nhwc_shape, nhwc_signature, "NHWC"),
        (side_adapter.source_name, nhwc_shape, nhwc_signature, "NHWC"),
        (main_adapter.nchw_name, nchw_shape, nchw_signature, "NCHW"),
        (main_mul_output_name, nchw_shape, nchw_signature, "NCHW"),
        (main_name, nchw_shape, nchw_signature, "NCHW"),
        (soft_pre_output_name, nwhc_shape, nwhc_signature, None),
        (softmax_output_name, nwhc_shape, nwhc_signature, None),
        (soft_back_name, nchw_shape, nchw_signature, "NCHW"),
        (reduce_output_name, reduce_shape, reduce_signature, None),
        (sub_output_name, reduce_shape, reduce_signature, None),
        (
            reshape_output_name,
            old_reshape_shape,
            old_reshape_signature,
            "NCHW",
        ),
        (expand_output_name, nchw_shape, nchw_signature, "NCHW"),
        (side_adapter.nchw_name, nchw_shape, nchw_signature, "NCHW"),
        (side_prelu_output_name, nchw_shape, nchw_signature, "NCHW"),
        (side_mask_name, nchw_shape, nchw_signature, "NCHW"),
        (residual_output_name, nchw_shape, nchw_signature, "NCHW"),
    )
    if not _layout_allows(canonical_tensor, "NHWC") or any(
        _exact_data_contract(
            model_ir,
            name,
            dtype=dtype,
            shape=shape,
            signature=signature,
            layout=layout,
        )
        is None
        for name, shape, signature, layout in exact_contracts
    ):
        return None
    for _, adapter in post_adapters:
        if (
            _exact_data_contract(
                model_ir,
                str(adapter.outputs[0]),
                dtype=dtype,
                shape=nhwc_shape,
                signature=nhwc_signature,
                layout="NHWC",
            )
            is None
        ):
            return None

    if not _singleton_float_constant(
        model_ir,
        graph_index,
        name=sub_constant_name,
        dtype=dtype,
        public_names=public_names,
    ):
        return None
    reduce_axis = _int_constant_replacement(
        model_ir,
        graph_index,
        name=reduce_axis_name,
        old_values=(1,),
        new_values=(3,),
        public_names=public_names,
    )
    reshape_shape = _int_constant_replacement(
        model_ir,
        graph_index,
        name=reshape_shape_name,
        old_values=old_reshape_shape,
        new_values=new_reshape_shape,
        public_names=public_names,
    )
    if reduce_axis is None or reshape_shape is None:
        return None
    planned_reshape_options = _reshape_options(
        reshape,
        old_shape=old_reshape_shape,
        new_shape=new_reshape_shape,
    )
    if planned_reshape_options is None:
        return None

    main_mul_constant = _broadcast_constant_replacement(
        model_ir,
        graph_index,
        name=main_mul_constant_name,
        dtype=dtype,
        old_nchw_shape=nchw_shape,
        target_nhwc_shape=nhwc_shape,
        public_names=public_names,
    )
    main_add_constant = _broadcast_constant_replacement(
        model_ir,
        graph_index,
        name=main_add_constant_name,
        dtype=dtype,
        old_nchw_shape=nchw_shape,
        target_nhwc_shape=nhwc_shape,
        public_names=public_names,
    )
    side_prelu_constant_name = str(side_prelu.inputs[1])
    side_prelu_constant = _broadcast_constant_replacement(
        model_ir,
        graph_index,
        name=side_prelu_constant_name,
        dtype=dtype,
        old_nchw_shape=nchw_shape,
        target_nhwc_shape=nhwc_shape,
        public_names=public_names,
    )
    expand_constant = _broadcast_constant_replacement(
        model_ir,
        graph_index,
        name=expand_constant_name,
        dtype=dtype,
        old_nchw_shape=nchw_shape,
        target_nhwc_shape=nhwc_shape,
        public_names=public_names,
    )
    if any(
        value is None
        for value in (
            main_mul_constant,
            main_add_constant,
            side_prelu_constant,
            expand_constant,
        )
    ):
        return None
    constant_plans = _plan_constants(
        model_ir,
        graph_index,
        (
            (
                main_mul_constant_name,
                np.asarray(main_mul_constant),
                main_mul,
                int(main_mul_constant_index),
            ),
            (
                main_add_constant_name,
                np.asarray(main_add_constant),
                main_add,
                int(main_add_constant_index),
            ),
            (
                side_prelu_constant_name,
                np.asarray(side_prelu_constant),
                side_prelu,
                1,
            ),
            (
                expand_constant_name,
                np.asarray(expand_constant),
                expand_mul,
                int(expand_constant_index),
            ),
            (reduce_axis_name, reduce_axis, reduce_max, 1),
            (reshape_shape_name, reshape_shape, reshape, 1),
        ),
    )
    if constant_plans is None:
        return None

    alias_rewrites = []
    post_indices = {index for index, _ in post_adapters}
    for _, adapter in post_adapters[1:]:
        alias_rewrites.extend(
            _input_rewrites(
                model_ir,
                graph_index,
                old_name=str(adapter.outputs[0]),
                new_name=post_output_name,
                excluded=post_indices,
            )
        )
    legacy_consumers = tuple(
        model_ir.operators[index] for index in legacy_indices
    )
    metadata_updates = tuple(
        _metadata_update(name, canonical_tensor)
        for name in (
            main_mul_output_name,
            main_name,
            soft_back_name,
            side_prelu_output_name,
            expand_output_name,
            side_mask_name,
        )
    ) + (
        _MetadataUpdate(
            name=reshape_output_name,
            shape=new_reshape_shape,
            signature=new_reshape_signature,
            logical_layout=str(canonical_tensor.logical_layout),
            physical_layout=str(canonical_tensor.physical_layout),
        ),
    )
    return _SoftmaxMaskPlan(
        root=root,
        post_adapters=tuple(adapter for _, adapter in post_adapters),
        legacy_consumers=legacy_consumers,
        side_adapter=side_adapter,
        main_adapter=main_adapter,
        side_prelu=side_prelu,
        main_mul=main_mul,
        main_add=main_add,
        soft_pre=soft_pre,
        softmax=softmax,
        soft_post=soft_post,
        reduce_max=reduce_max,
        sub=sub,
        reshape=reshape,
        expand_mul=expand_mul,
        side_mask_mul=side_mask_mul,
        residual_add=residual_add,
        main_mul_data_index=int(main_mul_data_index),
        main_name=main_name,
        soft_back_name=soft_back_name,
        residual_output_name=residual_output_name,
        post_output_name=post_output_name,
        reshape_options=planned_reshape_options,
        constant_plans=constant_plans,
        metadata_updates=metadata_updates,
        alias_rewrites=tuple(alias_rewrites),
        remove_operators=(
            side_adapter.pre,
            main_adapter.pre,
            soft_pre,
            soft_post,
            *(adapter for _, adapter in post_adapters),
        ),
        insert_legacy_adapter=bool(legacy_consumers),
        legacy_adapter_perm_name=(
            str(main_adapter.pre.inputs[1]) if legacy_consumers else None
        ),
    )


def _plans_equal(
    expected: _SoftmaxMaskPlan,
    actual: _SoftmaxMaskPlan,
) -> bool:
    operator_fields = (
        "root",
        "side_prelu",
        "main_mul",
        "main_add",
        "soft_pre",
        "softmax",
        "soft_post",
        "reduce_max",
        "sub",
        "reshape",
        "expand_mul",
        "side_mask_mul",
        "residual_add",
    )
    return bool(
        all(
            getattr(expected, name) is getattr(actual, name)
            for name in operator_fields
        )
        and expected.side_adapter.pre is actual.side_adapter.pre
        and expected.side_adapter.source_name
        == actual.side_adapter.source_name
        and expected.side_adapter.nchw_name == actual.side_adapter.nchw_name
        and expected.main_adapter.pre is actual.main_adapter.pre
        and expected.main_adapter.source_name
        == actual.main_adapter.source_name
        and expected.main_adapter.nchw_name == actual.main_adapter.nchw_name
        and len(expected.post_adapters) == len(actual.post_adapters)
        and all(
            lhs is rhs
            for lhs, rhs in zip(
                expected.post_adapters,
                actual.post_adapters,
            )
        )
        and len(expected.legacy_consumers) == len(actual.legacy_consumers)
        and all(
            lhs is rhs
            for lhs, rhs in zip(
                expected.legacy_consumers,
                actual.legacy_consumers,
            )
        )
        and all(
            getattr(expected, name) == getattr(actual, name)
            for name in (
                "main_mul_data_index",
                "main_name",
                "soft_back_name",
                "residual_output_name",
                "post_output_name",
                "reshape_options",
                "metadata_updates",
                "insert_legacy_adapter",
                "legacy_adapter_perm_name",
            )
        )
        and len(expected.alias_rewrites) == len(actual.alias_rewrites)
        and all(
            lhs.operator is rhs.operator
            and lhs.input_index == rhs.input_index
            and lhs.old_name == rhs.old_name
            and lhs.new_name == rhs.new_name
            for lhs, rhs in zip(
                expected.alias_rewrites,
                actual.alias_rewrites,
            )
        )
        and len(expected.remove_operators) == len(actual.remove_operators)
        and all(
            lhs is rhs
            for lhs, rhs in zip(
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
    plan: _SoftmaxMaskPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.root)
    if current is None or not _plans_equal(plan, current):
        return False
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in plan.remove_operators
    ]
    mutation_operators = (
        plan.side_prelu,
        plan.main_mul,
        plan.softmax,
        plan.reshape,
        plan.residual_add,
        *(rewrite.operator for rewrite in plan.alias_rewrites),
    )
    mutation_indices = [
        graph_index.operator_index(operator)
        for operator in mutation_operators
    ]
    if (
        any(index is None for index in remove_indices)
        or len({int(index) for index in remove_indices if index is not None})
        != len(remove_indices)
        or any(index is None for index in mutation_indices)
        or any(
            rewrite.input_index < 0
            or rewrite.input_index >= len(rewrite.operator.inputs)
            or str(rewrite.operator.inputs[rewrite.input_index])
            != rewrite.old_name
            for rewrite in plan.alias_rewrites
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
        or (
            plan.insert_legacy_adapter
            and (
                not plan.legacy_consumers
                or plan.legacy_adapter_perm_name not in model_ir.tensors
                or any(
                    graph_index.operator_index(operator) is None
                    for operator in plan.legacy_consumers
                )
            )
        )
    ):
        return False

    _apply_constant_plans(model_ir, graph_index, plan.constant_plans)
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.side_prelu,
        input_index=0,
        new_input_name=plan.side_adapter.source_name,
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.main_mul,
        input_index=plan.main_mul_data_index,
        new_input_name=plan.main_adapter.source_name,
        graph_index=graph_index,
    )
    softmax_index = graph_index.operator_index(plan.softmax)
    reshape_index = graph_index.operator_index(plan.reshape)
    residual_index = graph_index.operator_index(plan.residual_add)
    assert softmax_index is not None
    assert reshape_index is not None
    assert residual_index is not None
    graph_index.replace_operator_inputs(softmax_index, [plan.main_name])
    graph_index.replace_operator_outputs(softmax_index, [plan.soft_back_name])
    plan.reshape.options = dict(plan.reshape_options)
    graph_index.replace_operator_outputs(
        residual_index,
        [plan.post_output_name],
    )
    for rewrite in plan.alias_rewrites:
        _replace_operator_input_at(
            model_ir=model_ir,
            op=rewrite.operator,
            input_index=rewrite.input_index,
            new_input_name=rewrite.new_name,
            graph_index=graph_index,
        )
    _apply_metadata_updates(model_ir, plan.metadata_updates)
    graph_index.remove_operators([int(index) for index in remove_indices])

    if plan.insert_legacy_adapter:
        live_indices = [
            graph_index.operator_index(operator)
            for operator in plan.legacy_consumers
        ]
        if any(index is None for index in live_indices):
            raise AssertionError("legacy consumer disappeared after preflight")
        insert_index = min(
            int(index) for index in live_indices if index is not None
        )
        graph_index.insert_operator(
            insert_index,
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[
                    plan.post_output_name,
                    str(plan.legacy_adapter_perm_name),
                ],
                outputs=[plan.residual_output_name],
                options={},
            ),
        )
    return True


def optimize_sinet_softmax_mask_residual_nhwc_tail_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift a strict SiNet Softmax-mask residual island to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {
        "TRANSPOSE": 5,
        "PRELU": 1,
        "MUL": 3,
        "ADD": 2,
        "SOFTMAX": 1,
        "REDUCE_MAX": 1,
        "SUB": 1,
        "RESHAPE": 1,
    }
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if all(value == 0 for value in required_counts.values()):
            break
    if rewrite_limit == 0 or any(
        value > 0 for value in required_counts.values()
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
