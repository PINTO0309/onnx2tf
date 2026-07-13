from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    preflight_required_op_types,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _is_singleton_constant_tensor,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
    normalize_onnx_shape,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_pad import (
    NhwcConcatPadPlan,
    apply_nhwc_concat_pad_plan,
    resolve_nhwc_concat_pad_plan,
)


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
_PERM_NHWC_TO_NHCW = [0, 1, 3, 2]
_PERM_NCHW_TO_NHCW = [0, 2, 1, 3]
_DIRECT_STATS_KEY = "optimized_transpose_pre_concat_nhwc_direct_chains"
_UNARY_STATS_KEY = "optimized_transpose_pre_concat_nhwc_unary_chains"
_PAD_STATS_KEY = "optimized_transpose_pre_concat_nhwc_pad_chains"
_DEQUANTIZE_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_dequantize_chains"
)
_PRELU_STATS_KEY = "optimized_transpose_pre_concat_nhwc_prelu_chains"
_SOFTMAX_STATS_KEY = "optimized_transpose_pre_concat_nhwc_softmax_chains"
_SWISH_STATS_KEY = "optimized_transpose_pre_concat_nhwc_swish_chains"
_SLICE_STATS_KEY = "optimized_transpose_pre_concat_nhwc_slice_chains"
_SPLIT_STATS_KEY = "optimized_transpose_pre_concat_nhwc_split_chains"
_ADD_STATS_KEY = "optimized_transpose_pre_concat_nhwc_add_chains"
_LEAKY_STATS_KEY = "optimized_transpose_pre_concat_nhwc_leaky_chains"
_UNARY_OPS = {"RELU", "RELU6", "LOGISTIC", "TANH", "GELU"}
_MAX_ADD_PLAN_DEPTH = 64


@dataclass(frozen=True)
class _NhwcConcatInputPlan:
    kind: str
    adapter_op: OperatorIR
    source_name: str
    output_name: str
    remove_adapter: bool
    unary_op: Optional[OperatorIR] = None
    pad_plan: Optional[NhwcConcatPadPlan] = None
    dequantize_op: Optional[OperatorIR] = None
    prelu_op: Optional[OperatorIR] = None
    softmax_op: Optional[OperatorIR] = None
    logistic_op: Optional[OperatorIR] = None
    mul_op: Optional[OperatorIR] = None
    slice_op: Optional[OperatorIR] = None
    split_op: Optional[OperatorIR] = None
    add_op: Optional[OperatorIR] = None
    add_input_names: Tuple[str, ...] = ()
    add_operand_plans: Tuple["_NhwcConcatInputPlan", ...] = ()
    extra_source_adapter_ops: Tuple[OperatorIR, ...] = ()
    output_post_adapter_ops: Tuple[OperatorIR, ...] = ()
    leaky_neg_op: Optional[OperatorIR] = None
    leaky_pos_relu_op: Optional[OperatorIR] = None
    leaky_tensor_names: Tuple[str, ...] = ()
    alpha_tensor_name: Optional[str] = None
    selected_alpha: Optional[np.ndarray] = None
    alpha_permutation: Optional[Tuple[int, ...]] = None
    rewrite_alpha: bool = False
    clone_alpha: bool = False
    adapter_output_name: Optional[str] = None
    logistic_output_name: Optional[str] = None
    mul_data_input_index: Optional[int] = None
    begin_tensor_name: Optional[str] = None
    size_tensor_name: Optional[str] = None
    begin_nhwc: Optional[Tuple[int, ...]] = None
    size_nhwc: Optional[Tuple[int, ...]] = None
    clone_begin: bool = False
    clone_size: bool = False
    split_axis_tensor_name: Optional[str] = None
    clone_split_axis: bool = False


@dataclass(frozen=True)
class _NhwcConcatCandidate:
    input_plans: Tuple[_NhwcConcatInputPlan, ...]
    concat_op: OperatorIR
    concat_output_name: str
    post_ops: Tuple[OperatorIR, ...]
    post_output_names: Tuple[str, ...]


def _clone_permuted_quantization(
    quantization: Any,
    permutation: List[int] | Tuple[int, ...],
) -> Any:
    cloned = _clone_quantization(quantization)
    if isinstance(cloned, QuantParamIR):
        old_dimension = int(cloned.quantized_dimension)
        if 0 <= old_dimension < len(permutation):
            cloned.quantized_dimension = int(
                list(permutation).index(old_dimension)
            )
    elif isinstance(cloned, dict) and "quantized_dimension" in cloned:
        old_dimension = int(cloned["quantized_dimension"])
        if 0 <= old_dimension < len(permutation):
            cloned["quantized_dimension"] = int(
                list(permutation).index(old_dimension)
            )
    return cloned


def _clone_nhwc_quantization(quantization: Any) -> Any:
    return _clone_permuted_quantization(
        quantization,
        _PERM_NCHW_TO_NHWC,
    )


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    name = str(base)
    suffix = 1
    while name in model_ir.tensors:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _find_or_create_perm_tensor(
    model_ir: ModelIR,
    *,
    base_name: str,
    permutation: List[int],
) -> str:
    expected = np.asarray(permutation, dtype=np.int32)
    for tensor_name, tensor in model_ir.tensors.items():
        if tensor.data is None:
            continue
        try:
            data = np.asarray(tensor.data)
        except Exception:
            continue
        if (
            data.dtype == np.int32
            and int(data.size) == len(permutation)
            and np.array_equal(data.reshape(-1), expected)
        ):
            return str(tensor_name)
    tensor_name = _unique_tensor_name(model_ir, base_name)
    model_ir.tensors[tensor_name] = TensorIR(
        name=tensor_name,
        dtype="INT32",
        shape=[len(permutation)],
        shape_signature=[len(permutation)],
        data=np.array(expected, copy=True),
        is_variable=False,
    )
    return tensor_name


def _resolve_direct_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    adapter_op = graph_index.producer(input_name)
    if (
        adapter_op is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != input_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
    ):
        return None
    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    consumers = set(graph_index.consumer_indices(input_name))
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or int(concat_index) not in consumers
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="direct",
        adapter_op=adapter_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=(
            consumers == {int(concat_index)}
            and input_name not in model_outputs
        ),
    )


def _resolve_unary_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    unary_op = graph_index.producer(input_name)
    unary_index = (
        None if unary_op is None else graph_index.operator_index(unary_op)
    )
    if (
        unary_op is None
        or unary_index is None
        or str(unary_op.op_type) not in _UNARY_OPS
        or len(unary_op.inputs) != 1
        or len(unary_op.outputs) != 1
        or str(unary_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    adapter_output_name = str(unary_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
        or set(graph_index.consumer_indices(adapter_output_name))
        != {int(unary_index)}
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(input_name)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="unary",
        adapter_op=adapter_op,
        unary_op=unary_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=True,
    )


def _resolve_pad_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    pad_plan = resolve_nhwc_concat_pad_plan(
        model_ir,
        graph_index,
        output_name=input_name,
        concat_index=concat_index,
        model_outputs=model_outputs,
        public_names=public_names,
    )
    if pad_plan is None:
        return None
    return _NhwcConcatInputPlan(
        kind="pad",
        adapter_op=pad_plan.adapter_op,
        pad_plan=pad_plan,
        source_name=pad_plan.source_name,
        output_name=input_name,
        remove_adapter=pad_plan.remove_adapter,
    )


def _resolve_dequantize_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    dequantize_op = graph_index.producer(input_name)
    dequantize_index = (
        None
        if dequantize_op is None
        else graph_index.operator_index(dequantize_op)
    )
    if (
        dequantize_op is None
        or dequantize_index is None
        or str(dequantize_op.op_type) != "DEQUANTIZE"
        or len(dequantize_op.inputs) != 1
        or len(dequantize_op.outputs) != 1
        or str(dequantize_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    adapter_output_name = str(dequantize_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(input_name)
    if (
        source_tensor is None
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="dequantize",
        adapter_op=adapter_op,
        dequantize_op=dequantize_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=(
            set(graph_index.consumer_indices(adapter_output_name))
            == {int(dequantize_index)}
        ),
    )


def _select_prelu_alpha_for_nhwc(
    *,
    alpha_data: np.ndarray,
    target_nhwc_shape: Optional[List[int]],
) -> Optional[Tuple[np.ndarray, Optional[Tuple[int, ...]]]]:
    candidates: List[Tuple[np.ndarray, Optional[Tuple[int, ...]]]] = []
    if int(alpha_data.ndim) == 4:
        candidates.append(
            (
                np.transpose(alpha_data, axes=_PERM_NCHW_TO_NHWC).astype(
                    alpha_data.dtype,
                    copy=False,
                ),
                tuple(_PERM_NCHW_TO_NHWC),
            )
        )
    candidates.append((np.asarray(alpha_data), None))
    if int(alpha_data.ndim) == 3:
        candidates.append(
            (
                np.transpose(alpha_data, axes=[1, 2, 0]).astype(
                    alpha_data.dtype,
                    copy=False,
                ),
                (1, 2, 0),
            )
        )

    for candidate, permutation in candidates:
        if (
            target_nhwc_shape is None
            or not _is_fully_known_positive_shape(target_nhwc_shape)
            or _broadcast_static_shapes(
                [int(value) for value in target_nhwc_shape],
                [int(value) for value in candidate.shape],
            )
            is not None
        ):
            return np.asarray(candidate), permutation
    return None


def _resolve_prelu_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    prelu_op = graph_index.producer(input_name)
    prelu_index = (
        None if prelu_op is None else graph_index.operator_index(prelu_op)
    )
    if (
        prelu_op is None
        or prelu_index is None
        or str(prelu_op.op_type) != "PRELU"
        or len(prelu_op.inputs) != 2
        or len(prelu_op.outputs) != 1
        or str(prelu_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    adapter_output_name = str(prelu_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
        or set(graph_index.consumer_indices(adapter_output_name))
        != {int(prelu_index)}
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(input_name)
    alpha_tensor_name = str(prelu_op.inputs[1])
    alpha_tensor = model_ir.tensors.get(alpha_tensor_name)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
        or alpha_tensor is None
        or alpha_tensor.data is None
    ):
        return None
    selected = _select_prelu_alpha_for_nhwc(
        alpha_data=np.asarray(alpha_tensor.data),
        target_nhwc_shape=[int(value) for value in source_tensor.shape],
    )
    if selected is None:
        return None
    selected_alpha, alpha_permutation = selected
    alpha_data = np.asarray(alpha_tensor.data)
    rewrite_alpha = (
        selected_alpha.shape != alpha_data.shape
        or not np.array_equal(selected_alpha, alpha_data)
    )
    return _NhwcConcatInputPlan(
        kind="prelu",
        adapter_op=adapter_op,
        prelu_op=prelu_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=True,
        alpha_tensor_name=alpha_tensor_name,
        selected_alpha=np.asarray(selected_alpha),
        alpha_permutation=alpha_permutation,
        rewrite_alpha=bool(rewrite_alpha),
        clone_alpha=(
            bool(rewrite_alpha)
            and (
                alpha_tensor_name in public_names
                or set(graph_index.consumer_indices(alpha_tensor_name))
                != {int(prelu_index)}
            )
        ),
    )


def _resolve_softmax_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    softmax_op = graph_index.producer(input_name)
    softmax_index = (
        None if softmax_op is None else graph_index.operator_index(softmax_op)
    )
    if (
        softmax_op is None
        or softmax_index is None
        or str(softmax_op.op_type) != "SOFTMAX"
        or len(softmax_op.inputs) != 1
        or len(softmax_op.outputs) != 1
        or str(softmax_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    adapter_output_name = str(softmax_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
        or set(graph_index.consumer_indices(adapter_output_name))
        != {int(softmax_index)}
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    adapter_output_tensor = model_ir.tensors.get(adapter_output_name)
    output_tensor = model_ir.tensors.get(input_name)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or adapter_output_tensor is None
        or len(list(adapter_output_tensor.shape)) != 4
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="softmax",
        adapter_op=adapter_op,
        softmax_op=softmax_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=True,
        adapter_output_name=adapter_output_name,
    )


def _resolve_swish_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    mul_op = graph_index.producer(input_name)
    mul_index = None if mul_op is None else graph_index.operator_index(mul_op)
    if (
        mul_op is None
        or mul_index is None
        or str(mul_op.op_type) != "MUL"
        or len(mul_op.inputs) != 2
        or len(mul_op.outputs) != 1
        or str(mul_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    logistic_op: Optional[OperatorIR] = None
    logistic_index: Optional[int] = None
    logistic_output_name: Optional[str] = None
    adapter_output_name: Optional[str] = None
    mul_data_input_index: Optional[int] = None
    for logistic_input_index, logistic_candidate_name in enumerate(mul_op.inputs):
        candidate = graph_index.producer(str(logistic_candidate_name))
        candidate_index = (
            None if candidate is None else graph_index.operator_index(candidate)
        )
        if (
            candidate is not None
            and candidate_index is not None
            and str(candidate.op_type) == "LOGISTIC"
            and len(candidate.inputs) == 1
            and len(candidate.outputs) == 1
            and str(candidate.outputs[0]) == str(logistic_candidate_name)
        ):
            logistic_op = candidate
            logistic_index = int(candidate_index)
            logistic_output_name = str(logistic_candidate_name)
            adapter_output_name = str(candidate.inputs[0])
            mul_data_input_index = 1 - int(logistic_input_index)
            break
    if (
        logistic_op is None
        or logistic_index is None
        or logistic_output_name is None
        or adapter_output_name is None
        or mul_data_input_index is None
        or str(mul_op.inputs[mul_data_input_index]) != adapter_output_name
    ):
        return None

    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
        or logistic_output_name in model_outputs
        or set(graph_index.consumer_indices(adapter_output_name))
        != {int(logistic_index), int(mul_index)}
        or set(graph_index.consumer_indices(logistic_output_name))
        != {int(mul_index)}
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    logistic_output_tensor = model_ir.tensors.get(logistic_output_name)
    mul_output_tensor = model_ir.tensors.get(input_name)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or logistic_output_tensor is None
        or len(list(logistic_output_tensor.shape)) != 4
        or mul_output_tensor is None
        or len(list(mul_output_tensor.shape)) != 4
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="swish",
        adapter_op=adapter_op,
        logistic_op=logistic_op,
        mul_op=mul_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=True,
        adapter_output_name=adapter_output_name,
        logistic_output_name=logistic_output_name,
        mul_data_input_index=int(mul_data_input_index),
    )


def _resolve_slice_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    slice_op = graph_index.producer(input_name)
    slice_index = (
        None if slice_op is None else graph_index.operator_index(slice_op)
    )
    if (
        slice_op is None
        or slice_index is None
        or str(slice_op.op_type) != "SLICE"
        or len(slice_op.inputs) < 3
        or len(slice_op.outputs) != 1
        or str(slice_op.outputs[0]) != input_name
        or input_name in model_outputs
        or int(concat_index)
        not in set(graph_index.consumer_indices(input_name))
    ):
        return None

    output_post_adapter_ops: List[OperatorIR] = []
    for consumer_index in graph_index.consumer_indices(input_name):
        if int(consumer_index) == int(concat_index):
            continue
        post_op = model_ir.operators[int(consumer_index)]
        if (
            str(post_op.op_type) != "TRANSPOSE"
            or len(post_op.inputs) < 2
            or len(post_op.outputs) != 1
            or str(post_op.inputs[0]) != input_name
            or _read_transpose_perm(model_ir, post_op)
            != _PERM_NCHW_TO_NHWC
            or str(post_op.outputs[0]) in model_outputs
        ):
            return None
        post_tensor = model_ir.tensors.get(str(post_op.outputs[0]))
        if post_tensor is None or len(list(post_tensor.shape)) != 4:
            return None
        output_post_adapter_ops.append(post_op)

    adapter_output_name = str(slice_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    adapter_consumers = set(
        graph_index.consumer_indices(adapter_output_name)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or int(slice_index) not in adapter_consumers
    ):
        return None

    begin_tensor_name = str(slice_op.inputs[1])
    size_tensor_name = str(slice_op.inputs[2])
    begin_values = _read_const_ints_from_tensor(
        model_ir.tensors.get(begin_tensor_name)
    )
    size_values = _read_const_ints_from_tensor(
        model_ir.tensors.get(size_tensor_name)
    )
    if (
        begin_values is None
        or len(begin_values) != 4
        or size_values is None
        or len(size_values) != 4
        or int(size_values[1]) <= 0
        or int(begin_values[2]) != 0
        or int(begin_values[3]) != 0
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    adapter_output_tensor = model_ir.tensors.get(adapter_output_name)
    output_tensor = model_ir.tensors.get(input_name)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or adapter_output_tensor is None
        or len(list(adapter_output_tensor.shape)) != 4
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
    ):
        return None

    return _NhwcConcatInputPlan(
        kind="slice",
        adapter_op=adapter_op,
        slice_op=slice_op,
        source_name=source_name,
        output_name=input_name,
        output_post_adapter_ops=tuple(output_post_adapter_ops),
        remove_adapter=(
            adapter_consumers == {int(slice_index)}
            and adapter_output_name not in model_outputs
        ),
        adapter_output_name=adapter_output_name,
        begin_tensor_name=begin_tensor_name,
        size_tensor_name=size_tensor_name,
        begin_nhwc=(
            int(begin_values[0]),
            int(begin_values[2]),
            int(begin_values[3]),
            int(begin_values[1]),
        ),
        size_nhwc=(
            int(size_values[0]),
            int(size_values[2]),
            int(size_values[3]),
            int(size_values[1]),
        ),
        clone_begin=(
            begin_tensor_name in public_names
            or set(graph_index.consumer_indices(begin_tensor_name))
            != {int(slice_index)}
        ),
        clone_size=(
            size_tensor_name in public_names
            or set(graph_index.consumer_indices(size_tensor_name))
            != {int(slice_index)}
        ),
    )


def _resolve_split_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
    allowed_consumer_indices: Optional[frozenset[int]] = None,
) -> Optional[_NhwcConcatInputPlan]:
    split_op = graph_index.producer(input_name)
    split_index = (
        None if split_op is None else graph_index.operator_index(split_op)
    )
    if (
        split_op is None
        or split_index is None
        or str(split_op.op_type) != "SPLIT"
        or len(split_op.inputs) < 2
        or len(split_op.outputs) < 2
        or input_name not in {str(name) for name in split_op.outputs}
        or input_name in model_outputs
        or int(concat_index)
        not in set(graph_index.consumer_indices(input_name))
    ):
        return None

    split_axis_tensor_name = str(split_op.inputs[0])
    split_axis_values = _read_const_ints_from_tensor(
        model_ir.tensors.get(split_axis_tensor_name)
    )
    adapter_output_name = str(split_op.inputs[1])
    adapter_output_tensor = model_ir.tensors.get(adapter_output_name)
    if (
        split_axis_values is None
        or len(split_axis_values) != 1
        or adapter_output_tensor is None
        or len(list(adapter_output_tensor.shape)) != 4
    ):
        return None
    split_axis = int(split_axis_values[0])
    if split_axis < 0:
        split_axis += 4
    if split_axis != 1:
        return None

    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    adapter_consumers = set(
        graph_index.consumer_indices(adapter_output_name)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or int(split_index) not in adapter_consumers
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    if source_tensor is None or len(list(source_tensor.shape)) != 4:
        return None
    allowed_consumers = (
        {int(concat_index)}
        if allowed_consumer_indices is None
        else {int(index) for index in allowed_consumer_indices}
    )
    output_post_adapter_ops: List[OperatorIR] = []
    for split_output_name in [str(name) for name in split_op.outputs]:
        split_output_tensor = model_ir.tensors.get(split_output_name)
        if (
            split_output_name in model_outputs
            or split_output_tensor is None
            or len(list(split_output_tensor.shape)) != 4
        ):
            return None
        for consumer_index in graph_index.consumer_indices(split_output_name):
            if int(consumer_index) in allowed_consumers:
                continue
            post_op = model_ir.operators[int(consumer_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != split_output_name
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
                or str(post_op.outputs[0]) in model_outputs
            ):
                return None
            post_tensor = model_ir.tensors.get(str(post_op.outputs[0]))
            if post_tensor is None or len(list(post_tensor.shape)) != 4:
                return None
            output_post_adapter_ops.append(post_op)

    return _NhwcConcatInputPlan(
        kind="split",
        adapter_op=adapter_op,
        split_op=split_op,
        output_post_adapter_ops=tuple(output_post_adapter_ops),
        source_name=source_name,
        output_name=input_name,
        remove_adapter=(
            adapter_consumers == {int(split_index)}
            and adapter_output_name not in model_outputs
        ),
        adapter_output_name=adapter_output_name,
        split_axis_tensor_name=split_axis_tensor_name,
        clone_split_axis=(
            split_axis_tensor_name in public_names
            or set(graph_index.consumer_indices(split_axis_tensor_name))
            != {int(split_index)}
        ),
    )


def _collect_upstream_add_indices(
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    visited_add_outputs: frozenset[str] = frozenset(),
) -> Optional[frozenset[int]]:
    add_op = graph_index.producer(input_name)
    if add_op is None or str(add_op.op_type) != "ADD":
        return frozenset()
    if (
        input_name in visited_add_outputs
        or len(visited_add_outputs) >= _MAX_ADD_PLAN_DEPTH
    ):
        return None
    add_index = graph_index.operator_index(add_op)
    if (
        add_index is None
        or len(add_op.inputs) != 2
        or len(add_op.outputs) != 1
        or str(add_op.outputs[0]) != input_name
    ):
        return None

    next_visited_outputs = frozenset(
        {*visited_add_outputs, input_name}
    )
    add_indices = {int(add_index)}
    for add_input_name in [str(name) for name in add_op.inputs]:
        nested_indices = _collect_upstream_add_indices(
            graph_index,
            input_name=add_input_name,
            visited_add_outputs=next_visited_outputs,
        )
        if nested_indices is None:
            return None
        add_indices.update(int(index) for index in nested_indices)
    return frozenset(add_indices)


def _resolve_add_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
    visited_add_outputs: Optional[frozenset[str]] = None,
    allowed_candidate_consumer_indices: Optional[frozenset[int]] = None,
) -> Optional[_NhwcConcatInputPlan]:
    visited_add_outputs = visited_add_outputs or frozenset()
    if (
        input_name in visited_add_outputs
        or len(visited_add_outputs) >= _MAX_ADD_PLAN_DEPTH
    ):
        return None
    if allowed_candidate_consumer_indices is None:
        add_indices = _collect_upstream_add_indices(
            graph_index,
            input_name=input_name,
        )
        if add_indices is None:
            return None
        allowed_candidate_consumer_indices = frozenset(
            {int(concat_index), *[int(index) for index in add_indices]}
        )
    next_visited_add_outputs = frozenset(
        {*visited_add_outputs, input_name}
    )
    add_op = graph_index.producer(input_name)
    add_index = None if add_op is None else graph_index.operator_index(add_op)
    if (
        add_op is None
        or add_index is None
        or str(add_op.op_type) != "ADD"
        or len(add_op.inputs) != 2
        or len(add_op.outputs) != 1
        or str(add_op.outputs[0]) != input_name
        or input_name in model_outputs
        or int(concat_index)
        not in set(graph_index.consumer_indices(input_name))
    ):
        return None

    allowed_consumers = {
        int(index) for index in allowed_candidate_consumer_indices
    }
    output_post_adapter_ops: List[OperatorIR] = []
    for consumer_index in graph_index.consumer_indices(input_name):
        if int(consumer_index) in allowed_consumers:
            continue
        post_op = model_ir.operators[int(consumer_index)]
        if (
            str(post_op.op_type) != "TRANSPOSE"
            or len(post_op.inputs) < 2
            or len(post_op.outputs) != 1
            or str(post_op.inputs[0]) != input_name
            or _read_transpose_perm(model_ir, post_op)
            != _PERM_NCHW_TO_NHWC
            or str(post_op.outputs[0]) in model_outputs
        ):
            return None
        post_tensor = model_ir.tensors.get(str(post_op.outputs[0]))
        if post_tensor is None or len(list(post_tensor.shape)) != 4:
            return None
        output_post_adapter_ops.append(post_op)

    output_tensor = model_ir.tensors.get(input_name)
    if output_tensor is None or len(list(output_tensor.shape)) != 4:
        return None

    operand_plans: List[_NhwcConcatInputPlan] = []
    for add_input_name in [str(name) for name in add_op.inputs]:
        operand_plan = _resolve_direct_input_plan(
            model_ir,
            graph_index,
            input_name=add_input_name,
            concat_index=int(add_index),
            model_outputs=model_outputs,
        )
        if operand_plan is None:
            operand_plan = _resolve_unary_input_plan(
                model_ir,
                graph_index,
                input_name=add_input_name,
                concat_index=int(add_index),
                model_outputs=model_outputs,
            )
        if operand_plan is None:
            operand_plan = _resolve_swish_input_plan(
                model_ir,
                graph_index,
                input_name=add_input_name,
                concat_index=int(add_index),
                model_outputs=model_outputs,
            )
        if operand_plan is None:
            operand_plan = _resolve_dequantize_input_plan(
                model_ir,
                graph_index,
                input_name=add_input_name,
                concat_index=int(add_index),
                model_outputs=model_outputs,
            )
        if operand_plan is None:
            operand_plan = _resolve_prelu_input_plan(
                model_ir,
                graph_index,
                input_name=add_input_name,
                concat_index=int(add_index),
                model_outputs=model_outputs,
                public_names={
                    *[str(name) for name in model_ir.inputs],
                    *model_outputs,
                },
            )
        if operand_plan is None:
            operand_plan = _resolve_pad_input_plan(
                model_ir,
                graph_index,
                input_name=add_input_name,
                concat_index=int(add_index),
                model_outputs=model_outputs,
                public_names={
                    *[str(name) for name in model_ir.inputs],
                    *model_outputs,
                },
            )
        if operand_plan is None:
            operand_plan = _resolve_slice_input_plan(
                model_ir,
                graph_index,
                input_name=add_input_name,
                concat_index=int(add_index),
                model_outputs=model_outputs,
                public_names={
                    *[str(name) for name in model_ir.inputs],
                    *model_outputs,
                },
            )
        if operand_plan is None:
            operand_plan = _resolve_split_input_plan(
                model_ir,
                graph_index,
                input_name=add_input_name,
                concat_index=int(add_index),
                model_outputs=model_outputs,
                public_names={
                    *[str(name) for name in model_ir.inputs],
                    *model_outputs,
                },
                allowed_consumer_indices=(
                    allowed_candidate_consumer_indices
                ),
            )
        if operand_plan is None:
            operand_plan = _resolve_add_input_plan(
                model_ir,
                graph_index,
                input_name=add_input_name,
                concat_index=int(add_index),
                model_outputs=model_outputs,
                visited_add_outputs=next_visited_add_outputs,
                allowed_candidate_consumer_indices=(
                    allowed_candidate_consumer_indices
                ),
            )
        if operand_plan is None:
            return None
        adapter_output_tensor = model_ir.tensors.get(add_input_name)
        if adapter_output_tensor is None or len(list(adapter_output_tensor.shape)) != 4:
            return None
        operand_plans.append(operand_plan)

    unique_adapter_ops: List[OperatorIR] = []
    unique_remove_flags: List[bool] = []
    seen_adapter_ids: set[int] = set()
    for operand_plan in operand_plans:
        for adapter_offset, adapter_op in enumerate(
            (
                operand_plan.adapter_op,
                *operand_plan.extra_source_adapter_ops,
            )
        ):
            if id(adapter_op) in seen_adapter_ids:
                continue
            seen_adapter_ids.add(id(adapter_op))
            unique_adapter_ops.append(adapter_op)
            unique_remove_flags.append(
                bool(operand_plan.remove_adapter)
                if adapter_offset == 0
                else False
            )
    add_input_names = tuple(
        operand_plan.source_name
        if operand_plan.kind == "direct"
        else operand_plan.output_name
        for operand_plan in operand_plans
    )
    return _NhwcConcatInputPlan(
        kind="add",
        adapter_op=unique_adapter_ops[0],
        add_op=add_op,
        output_post_adapter_ops=tuple(output_post_adapter_ops),
        add_input_names=add_input_names,
        add_operand_plans=tuple(operand_plans),
        extra_source_adapter_ops=tuple(unique_adapter_ops[1:]),
        source_name=operand_plans[0].source_name,
        output_name=input_name,
        remove_adapter=unique_remove_flags[0],
    )


def _resolve_leaky_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    sub_op = graph_index.producer(input_name)
    sub_index = None if sub_op is None else graph_index.operator_index(sub_op)
    if (
        sub_op is None
        or sub_index is None
        or str(sub_op.op_type) != "SUB"
        or len(sub_op.inputs) != 2
        or len(sub_op.outputs) != 1
        or str(sub_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    pos_relu_output_name = str(sub_op.inputs[0])
    mul_output_name = str(sub_op.inputs[1])
    pos_relu_op = graph_index.producer(pos_relu_output_name)
    mul_op = graph_index.producer(mul_output_name)
    pos_relu_index = (
        None
        if pos_relu_op is None
        else graph_index.operator_index(pos_relu_op)
    )
    mul_index = None if mul_op is None else graph_index.operator_index(mul_op)
    if (
        pos_relu_op is None
        or pos_relu_index is None
        or str(pos_relu_op.op_type) != "RELU"
        or len(pos_relu_op.inputs) != 1
        or len(pos_relu_op.outputs) != 1
        or str(pos_relu_op.outputs[0]) != pos_relu_output_name
        or mul_op is None
        or mul_index is None
        or str(mul_op.op_type) != "MUL"
        or len(mul_op.inputs) != 2
        or len(mul_op.outputs) != 1
        or str(mul_op.outputs[0]) != mul_output_name
    ):
        return None

    neg_relu_op: Optional[OperatorIR] = None
    neg_relu_index: Optional[int] = None
    neg_relu_output_name: Optional[str] = None
    for alpha_index in (0, 1):
        alpha_name = str(mul_op.inputs[alpha_index])
        candidate_name = str(mul_op.inputs[1 - alpha_index])
        if not _is_singleton_constant_tensor(model_ir, alpha_name):
            continue
        candidate_op = graph_index.producer(candidate_name)
        candidate_index = (
            None
            if candidate_op is None
            else graph_index.operator_index(candidate_op)
        )
        if (
            candidate_op is not None
            and candidate_index is not None
            and str(candidate_op.op_type) == "RELU"
            and len(candidate_op.inputs) == 1
            and len(candidate_op.outputs) == 1
            and str(candidate_op.outputs[0]) == candidate_name
        ):
            neg_relu_op = candidate_op
            neg_relu_index = int(candidate_index)
            neg_relu_output_name = candidate_name
            break
    if (
        neg_relu_op is None
        or neg_relu_index is None
        or neg_relu_output_name is None
    ):
        return None

    neg_output_name = str(neg_relu_op.inputs[0])
    neg_op = graph_index.producer(neg_output_name)
    neg_index = None if neg_op is None else graph_index.operator_index(neg_op)
    if (
        neg_op is None
        or neg_index is None
        or str(neg_op.op_type) != "NEG"
        or len(neg_op.inputs) != 1
        or len(neg_op.outputs) != 1
        or str(neg_op.outputs[0]) != neg_output_name
    ):
        return None

    adapter_output_name = str(neg_op.inputs[0])
    if str(pos_relu_op.inputs[0]) != adapter_output_name:
        return None
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
        or set(graph_index.consumer_indices(adapter_output_name))
        != {int(neg_index), int(pos_relu_index)}
        or set(graph_index.consumer_indices(neg_output_name))
        != {int(neg_relu_index)}
        or set(graph_index.consumer_indices(neg_relu_output_name))
        != {int(mul_index)}
        or set(graph_index.consumer_indices(pos_relu_output_name))
        != {int(sub_index)}
        or set(graph_index.consumer_indices(mul_output_name))
        != {int(sub_index)}
    ):
        return None

    leaky_tensor_names = (
        neg_output_name,
        neg_relu_output_name,
        pos_relu_output_name,
        mul_output_name,
        input_name,
    )
    if any(name in model_outputs for name in leaky_tensor_names):
        return None
    source_name = str(adapter_op.inputs[0])
    rank_four_names = (
        source_name,
        adapter_output_name,
        *leaky_tensor_names,
    )
    if any(
        model_ir.tensors.get(name) is None
        or len(list(model_ir.tensors[name].shape)) != 4
        for name in rank_four_names
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="leaky",
        adapter_op=adapter_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=True,
        leaky_neg_op=neg_op,
        leaky_pos_relu_op=pos_relu_op,
        leaky_tensor_names=leaky_tensor_names,
        adapter_output_name=adapter_output_name,
    )


def _resolve_family_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    family: str,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
    allowed_candidate_consumer_indices: Optional[frozenset[int]] = None,
) -> Optional[_NhwcConcatInputPlan]:
    direct_plan = _resolve_direct_input_plan(
        model_ir,
        graph_index,
        input_name=input_name,
        concat_index=concat_index,
        model_outputs=model_outputs,
    )
    if direct_plan is not None:
        return direct_plan
    if family == "unary":
        return _resolve_unary_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
    if family == "pad":
        return _resolve_pad_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        )
    if family == "dequantize":
        return _resolve_dequantize_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
    if family == "prelu":
        return _resolve_prelu_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        )
    if family == "softmax":
        return _resolve_softmax_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
    if family == "swish":
        unary_plan = _resolve_unary_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
        if unary_plan is not None:
            return unary_plan
        return _resolve_swish_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
    if family == "slice":
        return _resolve_slice_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        )
    if family == "split":
        return _resolve_split_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        )
    if family == "add":
        add_plan = _resolve_add_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            allowed_candidate_consumer_indices=(
                allowed_candidate_consumer_indices
            ),
        )
        if add_plan is not None:
            return add_plan
        dequantize_plan = _resolve_dequantize_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
        if dequantize_plan is not None:
            return dequantize_plan
        prelu_plan = _resolve_prelu_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        )
        if prelu_plan is not None:
            return prelu_plan
        pad_plan = _resolve_pad_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        )
        if pad_plan is not None:
            return pad_plan
        slice_plan = _resolve_slice_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        )
        if slice_plan is not None:
            return slice_plan
        return _resolve_split_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
            allowed_consumer_indices=allowed_candidate_consumer_indices,
        )
    if family == "leaky":
        unary_plan = _resolve_unary_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
        if unary_plan is not None:
            return unary_plan
        return _resolve_leaky_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
    return None


def _resolve_nhwc_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    family: str,
) -> _NhwcConcatCandidate | None:
    """Find one strict float-path direct or unary Concat island."""

    model_outputs = {str(name) for name in model_ir.outputs}
    public_names = {
        *[str(name) for name in model_ir.inputs],
        *model_outputs,
    }
    for concat_op in model_ir.operators:
        concat_index = graph_index.operator_index(concat_op)
        if (
            concat_index is None
            or str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.inputs) == 0
            or len(concat_op.outputs) != 1
        ):
            continue

        concat_output_name = str(concat_op.outputs[0])
        concat_output_tensor = model_ir.tensors.get(concat_output_name)
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 4
        if (
            concat_axis != 1
            or concat_output_name in model_outputs
            or concat_output_tensor is None
            or len(list(concat_output_tensor.shape)) != 4
        ):
            continue

        allowed_candidate_consumer_indices: Optional[frozenset[int]] = None
        if family == "add":
            selected_consumer_indices = {int(concat_index)}
            add_tree_valid = True
            for input_name in [str(name) for name in concat_op.inputs]:
                add_indices = _collect_upstream_add_indices(
                    graph_index,
                    input_name=input_name,
                )
                if add_indices is None:
                    add_tree_valid = False
                    break
                selected_consumer_indices.update(
                    int(index) for index in add_indices
                )
            if not add_tree_valid:
                continue
            allowed_candidate_consumer_indices = frozenset(
                selected_consumer_indices
            )

        input_plans: List[_NhwcConcatInputPlan] = []
        inputs_valid = True
        for input_name in [str(name) for name in concat_op.inputs]:
            input_plan = _resolve_family_input_plan(
                model_ir,
                graph_index,
                family=family,
                input_name=input_name,
                concat_index=int(concat_index),
                model_outputs=model_outputs,
                public_names=public_names,
                allowed_candidate_consumer_indices=(
                    allowed_candidate_consumer_indices
                ),
            )
            if input_plan is None:
                inputs_valid = False
                break
            input_plans.append(input_plan)
        if not inputs_valid or not input_plans:
            continue
        unary_count = sum(plan.kind == "unary" for plan in input_plans)
        if family == "direct" and unary_count != 0:
            continue
        if family == "unary" and unary_count < 1:
            continue
        pad_count = sum(plan.kind == "pad" for plan in input_plans)
        if family == "pad" and (
            pad_count < 1 or len(input_plans) <= pad_count
        ):
            continue
        dequantize_count = sum(
            plan.kind == "dequantize" for plan in input_plans
        )
        if family == "dequantize" and dequantize_count < 1:
            continue
        prelu_count = sum(plan.kind == "prelu" for plan in input_plans)
        if family == "prelu" and prelu_count < 1:
            continue
        softmax_count = sum(plan.kind == "softmax" for plan in input_plans)
        if family == "softmax" and (
            softmax_count != 1
            or len(input_plans) - softmax_count < 1
        ):
            continue
        swish_count = sum(plan.kind == "swish" for plan in input_plans)
        if family == "swish" and swish_count < 1:
            continue
        slice_plans = [
            plan for plan in input_plans if plan.kind == "slice"
        ]
        if family == "slice" and (
            len(slice_plans) < 1
            or len(
                {
                    id(plan.slice_op)
                    for plan in slice_plans
                    if plan.slice_op is not None
                }
            )
            != len(slice_plans)
        ):
            continue
        split_count = sum(plan.kind == "split" for plan in input_plans)
        if family == "split" and split_count < 1:
            continue
        add_count = sum(plan.kind == "add" for plan in input_plans)
        if family == "add" and add_count < 1:
            continue
        leaky_count = sum(plan.kind == "leaky" for plan in input_plans)
        if family == "leaky" and leaky_count < 1:
            continue

        if family in {
            "unary",
            "pad",
            "dequantize",
            "prelu",
            "softmax",
            "swish",
            "slice",
            "split",
            "add",
            "leaky",
        }:
            reference_shape: Optional[List[int]] = None
            shapes_compatible = True
            for input_plan in input_plans:
                tensor = model_ir.tensors.get(
                    input_plan.source_name
                    if input_plan.kind == "direct"
                    else input_plan.output_name
                )
                if tensor is None or len(list(tensor.shape)) != 4:
                    shapes_compatible = False
                    break
                shape = [int(value) for value in tensor.shape]
                if input_plan.kind in {
                    "unary",
                    "pad",
                    "dequantize",
                    "prelu",
                    "softmax",
                    "swish",
                    "slice",
                    "split",
                    "add",
                    "leaky",
                }:
                    shape = [shape[index] for index in _PERM_NCHW_TO_NHWC]
                if reference_shape is None:
                    reference_shape = shape
                elif any(
                    int(shape[index]) != int(reference_shape[index])
                    for index in (0, 1, 2)
                ):
                    shapes_compatible = False
                    break
            if not shapes_compatible:
                continue

        concat_user_indices = graph_index.consumer_indices(concat_output_name)
        if not concat_user_indices:
            continue
        post_ops: List[OperatorIR] = []
        post_output_names: List[str] = []
        posts_valid = True
        for user_index in concat_user_indices:
            post_op = model_ir.operators[int(user_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != concat_output_name
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
                or str(post_op.outputs[0]) in model_outputs
            ):
                posts_valid = False
                break
            post_ops.append(post_op)
            post_output_names.append(str(post_op.outputs[0]))
        if not posts_valid or not post_ops:
            continue

        return _NhwcConcatCandidate(
            input_plans=tuple(input_plans),
            concat_op=concat_op,
            concat_output_name=concat_output_name,
            post_ops=tuple(post_ops),
            post_output_names=tuple(post_output_names),
        )
    return None


def _has_nhwc_direct_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="direct",
        )
        is not None
    )


def _has_nhwc_unary_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="unary",
        )
        is not None
    )


def _has_nhwc_pad_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="pad",
        )
        is not None
    )


def _has_nhwc_dequantize_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="dequantize",
        )
        is not None
    )


def _has_nhwc_prelu_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="prelu",
        )
        is not None
    )


def _has_nhwc_softmax_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="softmax",
        )
        is not None
    )


def _has_nhwc_swish_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="swish",
        )
        is not None
    )


def _has_nhwc_slice_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="slice",
        )
        is not None
    )


def _has_nhwc_split_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="split",
        )
        is not None
    )


def _has_nhwc_add_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="add",
        )
        is not None
    )


def _has_nhwc_leaky_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="leaky",
        )
        is not None
    )


def _optimize_transpose_pre_concat_nhwc_direct_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="direct",
        stats_key=_DIRECT_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_unary_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="unary",
        stats_key=_UNARY_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_pad_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="pad",
        stats_key=_PAD_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_dequantize_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="dequantize",
        stats_key=_DEQUANTIZE_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_prelu_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="prelu",
        stats_key=_PRELU_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_softmax_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="softmax",
        stats_key=_SOFTMAX_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_swish_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="swish",
        stats_key=_SWISH_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_slice_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="slice",
        stats_key=_SLICE_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_split_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="split",
        stats_key=_SPLIT_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_add_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="add",
        stats_key=_ADD_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_leaky_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="leaky",
        stats_key=_LEAKY_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _apply_dequantize_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
) -> None:
    assert input_plan.dequantize_op is not None
    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.dequantize_op,
        new_inputs=[input_plan.source_name],
        graph_index=graph_index,
    )
    output_tensor = model_ir.tensors.get(input_plan.output_name)
    _permute_tensor_metadata_if_rank_matches(
        output_tensor,
        _PERM_NCHW_TO_NHWC,
    )
    if output_tensor is not None:
        output_tensor.quantization = _clone_nhwc_quantization(
            output_tensor.quantization
        )


def _apply_prelu_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
    *,
    materialized_alphas: Optional[
        Dict[Tuple[str, Optional[Tuple[int, ...]], Tuple[int, ...]], str]
    ] = None,
) -> None:
    assert input_plan.prelu_op is not None
    assert input_plan.alpha_tensor_name is not None
    assert input_plan.selected_alpha is not None
    alpha_tensor = model_ir.tensors[input_plan.alpha_tensor_name]
    selected_alpha_name = input_plan.alpha_tensor_name
    if materialized_alphas is None:
        materialized_alphas = {}

    if input_plan.rewrite_alpha:
        selected_alpha = np.asarray(input_plan.selected_alpha)
        materialization_key = (
            input_plan.alpha_tensor_name,
            input_plan.alpha_permutation,
            tuple(int(value) for value in selected_alpha.shape),
        )
        materialized_alpha_name = materialized_alphas.get(
            materialization_key
        )
        if materialized_alpha_name is not None:
            selected_alpha_name = materialized_alpha_name
        else:
            shape, signature = normalize_onnx_shape(
                list(selected_alpha.shape)
            )
            selected_quantization = (
                _clone_permuted_quantization(
                    alpha_tensor.quantization,
                    input_plan.alpha_permutation,
                )
                if input_plan.alpha_permutation is not None
                else _clone_quantization(alpha_tensor.quantization)
            )
            if input_plan.clone_alpha:
                selected_alpha_name = _unique_tensor_name(
                    model_ir,
                    f"{input_plan.alpha_tensor_name}_nhwc",
                )
                model_ir.tensors[selected_alpha_name] = TensorIR(
                    name=selected_alpha_name,
                    dtype=str(alpha_tensor.dtype),
                    shape=[int(value) for value in shape],
                    shape_signature=[int(value) for value in signature],
                    data=np.array(selected_alpha, copy=True),
                    is_variable=bool(alpha_tensor.is_variable),
                    quantization=selected_quantization,
                    logical_layout=str(alpha_tensor.logical_layout),
                    physical_layout=str(alpha_tensor.physical_layout),
                    onnx_tensor_name=alpha_tensor.onnx_tensor_name,
                )
            else:
                alpha_tensor.data = np.array(selected_alpha, copy=True)
                alpha_tensor.shape = [int(value) for value in shape]
                alpha_tensor.shape_signature = [
                    int(value) for value in signature
                ]
                alpha_tensor.quantization = selected_quantization
            materialized_alphas[materialization_key] = selected_alpha_name

    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.prelu_op,
        new_inputs=[input_plan.source_name, selected_alpha_name],
        graph_index=graph_index,
    )
    prelu_output_tensor = model_ir.tensors.get(input_plan.output_name)
    _permute_tensor_metadata_if_rank_matches(
        prelu_output_tensor,
        _PERM_NCHW_TO_NHWC,
    )
    if prelu_output_tensor is not None:
        prelu_output_tensor.quantization = _clone_nhwc_quantization(
            prelu_output_tensor.quantization
        )


def _apply_softmax_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
) -> None:
    assert input_plan.softmax_op is not None
    assert input_plan.adapter_output_name is not None
    source_tensor = model_ir.tensors[input_plan.source_name]
    softmax_output_tensor = model_ir.tensors[input_plan.output_name]
    softmax_index = graph_index.operator_index(input_plan.softmax_op)
    if softmax_index is None:
        raise RuntimeError("indexed Softmax candidate disappeared before apply")

    source_shape = [int(value) for value in source_tensor.shape]
    source_signature = (
        [int(value) for value in source_tensor.shape_signature]
        if source_tensor.shape_signature is not None
        else list(source_shape)
    )
    axis_last_shape = _permute_shape(
        source_shape,
        _PERM_NHWC_TO_NHCW,
    )
    axis_last_signature = _permute_shape(
        source_signature,
        _PERM_NHWC_TO_NHCW,
    )
    if axis_last_shape is None or axis_last_signature is None:
        raise RuntimeError("invalid rank-four Softmax layout projection")

    axis_last_input_name = _unique_tensor_name(
        model_ir,
        f"{input_plan.adapter_output_name}_axis_last",
    )
    axis_last_output_name = _unique_tensor_name(
        model_ir,
        f"{input_plan.output_name}_axis_last",
    )
    perm_tensor_name = _find_or_create_perm_tensor(
        model_ir,
        base_name=f"{input_plan.output_name}_nhwc_to_nhcw_perm",
        permutation=_PERM_NHWC_TO_NHCW,
    )
    original_output_quantization = _clone_quantization(
        softmax_output_tensor.quantization
    )
    model_ir.tensors[axis_last_input_name] = TensorIR(
        name=axis_last_input_name,
        dtype=str(source_tensor.dtype),
        shape=[int(value) for value in axis_last_shape],
        shape_signature=[int(value) for value in axis_last_signature],
        data=None,
        is_variable=False,
        quantization=_clone_permuted_quantization(
            source_tensor.quantization,
            _PERM_NHWC_TO_NHCW,
        ),
        onnx_tensor_name=source_tensor.onnx_tensor_name,
    )
    model_ir.tensors[axis_last_output_name] = TensorIR(
        name=axis_last_output_name,
        dtype=str(softmax_output_tensor.dtype),
        shape=[int(value) for value in axis_last_shape],
        shape_signature=[int(value) for value in axis_last_signature],
        data=None,
        is_variable=False,
        quantization=_clone_permuted_quantization(
            original_output_quantization,
            _PERM_NCHW_TO_NHCW,
        ),
        onnx_tensor_name=softmax_output_tensor.onnx_tensor_name,
    )

    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.softmax_op,
        new_inputs=[axis_last_input_name],
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=input_plan.softmax_op,
        new_outputs=[axis_last_output_name],
        graph_index=graph_index,
    )
    graph_index.insert_operator(
        int(softmax_index),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[input_plan.source_name, perm_tensor_name],
            outputs=[axis_last_input_name],
        ),
    )
    updated_softmax_index = graph_index.operator_index(input_plan.softmax_op)
    if updated_softmax_index is None:
        raise RuntimeError("indexed Softmax disappeared after adapter insert")
    graph_index.insert_operator(
        int(updated_softmax_index) + 1,
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[axis_last_output_name, perm_tensor_name],
            outputs=[input_plan.output_name],
        ),
    )

    _permute_tensor_metadata_if_rank_matches(
        softmax_output_tensor,
        _PERM_NCHW_TO_NHWC,
    )
    softmax_output_tensor.quantization = _clone_permuted_quantization(
        original_output_quantization,
        _PERM_NCHW_TO_NHWC,
    )


def _apply_unary_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
) -> None:
    assert input_plan.unary_op is not None
    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.unary_op,
        new_inputs=[input_plan.source_name],
        graph_index=graph_index,
    )
    output_tensor = model_ir.tensors.get(input_plan.output_name)
    _permute_tensor_metadata_if_rank_matches(
        output_tensor,
        _PERM_NCHW_TO_NHWC,
    )
    if output_tensor is not None:
        output_tensor.quantization = _clone_nhwc_quantization(
            output_tensor.quantization
        )


def _apply_swish_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
) -> None:
    assert input_plan.logistic_op is not None
    assert input_plan.mul_op is not None
    assert input_plan.logistic_output_name is not None
    assert input_plan.mul_data_input_index is not None
    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.logistic_op,
        new_inputs=[input_plan.source_name],
        graph_index=graph_index,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=input_plan.mul_op,
        input_index=input_plan.mul_data_input_index,
        new_input_name=input_plan.source_name,
        graph_index=graph_index,
    )
    for tensor_name in (
        input_plan.logistic_output_name,
        input_plan.output_name,
    ):
        tensor = model_ir.tensors.get(tensor_name)
        _permute_tensor_metadata_if_rank_matches(
            tensor,
            _PERM_NCHW_TO_NHWC,
        )
        if tensor is not None:
            tensor.quantization = _clone_nhwc_quantization(
                tensor.quantization
            )


def _materialize_int_parameter(
    model_ir: ModelIR,
    *,
    tensor_name: str,
    values: Tuple[int, ...],
    clone: bool,
    materialized: Dict[Tuple[str, Tuple[int, ...]], str],
) -> str:
    key = (str(tensor_name), tuple(int(value) for value in values))
    existing_name = materialized.get(key)
    if existing_name is not None:
        return existing_name

    tensor = model_ir.tensors[str(tensor_name)]
    rewritten_name = str(tensor_name)
    if clone:
        rewritten_name = _unique_tensor_name(
            model_ir,
            f"{tensor_name}_nhwc",
        )
        np_dtype = np.int32
        if tensor.data is not None:
            try:
                np_dtype = np.asarray(tensor.data).dtype
            except Exception:
                np_dtype = np.int32
        model_ir.tensors[rewritten_name] = TensorIR(
            name=rewritten_name,
            dtype=str(tensor.dtype),
            shape=[len(values)],
            shape_signature=[len(values)],
            data=np.asarray(values, dtype=np_dtype),
            is_variable=bool(tensor.is_variable),
            quantization=_clone_quantization(tensor.quantization),
            logical_layout=str(tensor.logical_layout),
            physical_layout=str(tensor.physical_layout),
            onnx_tensor_name=tensor.onnx_tensor_name,
        )
    else:
        _write_const_ints_to_tensor(
            tensor,
            [int(value) for value in values],
        )
    materialized[key] = rewritten_name
    return rewritten_name


def _apply_slice_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
    *,
    materialized: Dict[Tuple[str, Tuple[int, ...]], str],
) -> None:
    assert input_plan.slice_op is not None
    assert input_plan.begin_tensor_name is not None
    assert input_plan.size_tensor_name is not None
    assert input_plan.begin_nhwc is not None
    assert input_plan.size_nhwc is not None
    begin_name = _materialize_int_parameter(
        model_ir,
        tensor_name=input_plan.begin_tensor_name,
        values=input_plan.begin_nhwc,
        clone=input_plan.clone_begin,
        materialized=materialized,
    )
    size_name = _materialize_int_parameter(
        model_ir,
        tensor_name=input_plan.size_tensor_name,
        values=input_plan.size_nhwc,
        clone=input_plan.clone_size,
        materialized=materialized,
    )
    new_inputs = [str(name) for name in input_plan.slice_op.inputs]
    new_inputs[0] = input_plan.source_name
    new_inputs[1] = begin_name
    new_inputs[2] = size_name
    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.slice_op,
        new_inputs=new_inputs,
        graph_index=graph_index,
    )
    output_tensor = model_ir.tensors.get(input_plan.output_name)
    _permute_tensor_metadata_if_rank_matches(
        output_tensor,
        _PERM_NCHW_TO_NHWC,
    )
    if output_tensor is not None:
        output_tensor.quantization = _clone_nhwc_quantization(
            output_tensor.quantization
        )
    for post_op in input_plan.output_post_adapter_ops:
        _replace_tensor_inputs(
            model_ir=model_ir,
            src_name=str(post_op.outputs[0]),
            dst_name=input_plan.output_name,
            graph_index=graph_index,
        )


def _apply_split_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
    *,
    materialized: Dict[Tuple[str, Tuple[int, ...]], str],
) -> None:
    assert input_plan.split_op is not None
    assert input_plan.split_axis_tensor_name is not None
    axis_name = _materialize_int_parameter(
        model_ir,
        tensor_name=input_plan.split_axis_tensor_name,
        values=(3,),
        clone=input_plan.clone_split_axis,
        materialized=materialized,
    )
    new_inputs = [str(name) for name in input_plan.split_op.inputs]
    new_inputs[0] = axis_name
    new_inputs[1] = input_plan.source_name
    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.split_op,
        new_inputs=new_inputs,
        graph_index=graph_index,
    )
    for output_name in [str(name) for name in input_plan.split_op.outputs]:
        output_tensor = model_ir.tensors.get(output_name)
        _permute_tensor_metadata_if_rank_matches(
            output_tensor,
            _PERM_NCHW_TO_NHWC,
        )
        if output_tensor is not None:
            output_tensor.quantization = _clone_nhwc_quantization(
                output_tensor.quantization
            )
    for post_op in input_plan.output_post_adapter_ops:
        _replace_tensor_inputs(
            model_ir=model_ir,
            src_name=str(post_op.outputs[0]),
            dst_name=str(post_op.inputs[0]),
            graph_index=graph_index,
        )


def _apply_add_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
    *,
    materialized_int_parameters: Optional[
        Dict[Tuple[str, Tuple[int, ...]], str]
    ] = None,
    materialized_pads: Optional[Dict[str, str]] = None,
    materialized_prelu_alphas: Optional[
        Dict[Tuple[str, Optional[Tuple[int, ...]], Tuple[int, ...]], str]
    ] = None,
    applied_split_operators: Optional[set[int]] = None,
    applied_add_operators: Optional[set[int]] = None,
) -> None:
    assert input_plan.add_op is not None
    assert len(input_plan.add_input_names) == 2
    add_operator_id = id(input_plan.add_op)
    if applied_add_operators is None:
        applied_add_operators = set()
    if add_operator_id in applied_add_operators:
        return
    if materialized_int_parameters is None:
        materialized_int_parameters = {}
    if materialized_pads is None:
        materialized_pads = {}
    if materialized_prelu_alphas is None:
        materialized_prelu_alphas = {}
    if applied_split_operators is None:
        applied_split_operators = set()
    for operand_plan in input_plan.add_operand_plans:
        if operand_plan.unary_op is not None:
            _apply_unary_input_plan(
                model_ir,
                graph_index,
                operand_plan,
            )
        elif operand_plan.logistic_op is not None:
            _apply_swish_input_plan(
                model_ir,
                graph_index,
                operand_plan,
            )
        elif operand_plan.dequantize_op is not None:
            _apply_dequantize_input_plan(
                model_ir,
                graph_index,
                operand_plan,
            )
        elif operand_plan.prelu_op is not None:
            _apply_prelu_input_plan(
                model_ir,
                graph_index,
                operand_plan,
                materialized_alphas=materialized_prelu_alphas,
            )
        elif operand_plan.pad_plan is not None:
            apply_nhwc_concat_pad_plan(
                model_ir,
                graph_index,
                operand_plan.pad_plan,
                materialized_pads=materialized_pads,
            )
        elif operand_plan.slice_op is not None:
            _apply_slice_input_plan(
                model_ir,
                graph_index,
                operand_plan,
                materialized=materialized_int_parameters,
            )
        elif operand_plan.split_op is not None:
            split_operator_id = id(operand_plan.split_op)
            if split_operator_id not in applied_split_operators:
                _apply_split_input_plan(
                    model_ir,
                    graph_index,
                    operand_plan,
                    materialized=materialized_int_parameters,
                )
                applied_split_operators.add(split_operator_id)
        elif operand_plan.add_op is not None:
            _apply_add_input_plan(
                model_ir,
                graph_index,
                operand_plan,
                materialized_int_parameters=materialized_int_parameters,
                materialized_pads=materialized_pads,
                materialized_prelu_alphas=materialized_prelu_alphas,
                applied_split_operators=applied_split_operators,
                applied_add_operators=applied_add_operators,
            )
    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.add_op,
        new_inputs=[str(name) for name in input_plan.add_input_names],
        graph_index=graph_index,
    )
    output_tensor = model_ir.tensors.get(input_plan.output_name)
    _permute_tensor_metadata_if_rank_matches(
        output_tensor,
        _PERM_NCHW_TO_NHWC,
    )
    if output_tensor is not None:
        output_tensor.quantization = _clone_nhwc_quantization(
            output_tensor.quantization
        )
    for post_op in input_plan.output_post_adapter_ops:
        _replace_tensor_inputs(
            model_ir=model_ir,
            src_name=str(post_op.outputs[0]),
            dst_name=input_plan.output_name,
            graph_index=graph_index,
        )
    applied_add_operators.add(add_operator_id)


def _apply_leaky_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    input_plan: _NhwcConcatInputPlan,
) -> None:
    assert input_plan.leaky_neg_op is not None
    assert input_plan.leaky_pos_relu_op is not None
    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.leaky_neg_op,
        new_inputs=[input_plan.source_name],
        graph_index=graph_index,
    )
    _set_operator_inputs(
        model_ir=model_ir,
        op=input_plan.leaky_pos_relu_op,
        new_inputs=[input_plan.source_name],
        graph_index=graph_index,
    )
    for tensor_name in input_plan.leaky_tensor_names:
        tensor = model_ir.tensors.get(tensor_name)
        _permute_tensor_metadata_if_rank_matches(
            tensor,
            _PERM_NCHW_TO_NHWC,
        )
        if tensor is not None:
            tensor.quantization = _clone_nhwc_quantization(
                tensor.quantization
            )


def _optimize_transpose_pre_concat_nhwc_family(
    model_ir: ModelIR,
    *,
    family: str,
    stats_key: str,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Lift one strict direct/unary NCHW Concat family into NHWC."""

    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        candidate = _resolve_nhwc_concat_candidate(
            model_ir,
            graph_index,
            family=family,
        )
        if candidate is None:
            break

        new_concat_inputs: List[str] = []
        materialized_pads: Dict[str, str] = {}
        materialized_prelu_alphas: Dict[
            Tuple[str, Optional[Tuple[int, ...]], Tuple[int, ...]],
            str,
        ] = {}
        materialized_int_parameters: Dict[
            Tuple[str, Tuple[int, ...]],
            str,
        ] = {}
        applied_split_operators: set[int] = set()
        applied_add_operators: set[int] = set()
        applied_leaky_operators: set[int] = set()
        for input_plan in candidate.input_plans:
            if input_plan.unary_op is not None:
                _apply_unary_input_plan(
                    model_ir,
                    graph_index,
                    input_plan,
                )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.logistic_op is not None:
                _apply_swish_input_plan(
                    model_ir,
                    graph_index,
                    input_plan,
                )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.slice_op is not None:
                _apply_slice_input_plan(
                    model_ir,
                    graph_index,
                    input_plan,
                    materialized=materialized_int_parameters,
                )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.split_op is not None:
                split_operator_id = id(input_plan.split_op)
                if split_operator_id not in applied_split_operators:
                    _apply_split_input_plan(
                        model_ir,
                        graph_index,
                        input_plan,
                        materialized=materialized_int_parameters,
                    )
                    applied_split_operators.add(split_operator_id)
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.add_op is not None:
                add_operator_id = id(input_plan.add_op)
                if add_operator_id not in applied_add_operators:
                    _apply_add_input_plan(
                        model_ir,
                        graph_index,
                        input_plan,
                        materialized_int_parameters=(
                            materialized_int_parameters
                        ),
                        materialized_pads=materialized_pads,
                        materialized_prelu_alphas=(
                            materialized_prelu_alphas
                        ),
                        applied_split_operators=applied_split_operators,
                        applied_add_operators=applied_add_operators,
                    )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.leaky_neg_op is not None:
                leaky_operator_id = id(input_plan.leaky_neg_op)
                if leaky_operator_id not in applied_leaky_operators:
                    _apply_leaky_input_plan(
                        model_ir,
                        graph_index,
                        input_plan,
                    )
                    applied_leaky_operators.add(leaky_operator_id)
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.softmax_op is not None:
                _apply_softmax_input_plan(
                    model_ir,
                    graph_index,
                    input_plan,
                )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.prelu_op is not None:
                _apply_prelu_input_plan(
                    model_ir,
                    graph_index,
                    input_plan,
                    materialized_alphas=materialized_prelu_alphas,
                )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.dequantize_op is not None:
                _apply_dequantize_input_plan(
                    model_ir,
                    graph_index,
                    input_plan,
                )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.pad_plan is not None:
                apply_nhwc_concat_pad_plan(
                    model_ir,
                    graph_index,
                    input_plan.pad_plan,
                    materialized_pads=materialized_pads,
                )
                new_concat_inputs.append(input_plan.output_name)
            else:
                new_concat_inputs.append(input_plan.source_name)
        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=new_concat_inputs,
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 3

        canonical_output_name = candidate.post_output_names[0]
        _set_operator_outputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_outputs=[canonical_output_name],
            graph_index=graph_index,
        )
        for alias_output_name in candidate.post_output_names[1:]:
            _replace_tensor_inputs(
                model_ir,
                alias_output_name,
                canonical_output_name,
                graph_index=graph_index,
            )

        old_concat_tensor = model_ir.tensors.get(candidate.concat_output_name)
        canonical_output_tensor = model_ir.tensors.get(canonical_output_name)
        if old_concat_tensor is not None and canonical_output_tensor is not None:
            canonical_output_tensor.dtype = str(old_concat_tensor.dtype)
            canonical_output_tensor.shape = [
                int(value) for value in old_concat_tensor.shape
            ]
            canonical_output_tensor.shape_signature = (
                [int(value) for value in old_concat_tensor.shape_signature]
                if old_concat_tensor.shape_signature is not None
                else [int(value) for value in old_concat_tensor.shape]
            )
            _permute_tensor_metadata_if_rank_matches(
                canonical_output_tensor,
                _PERM_NCHW_TO_NHWC,
            )
            canonical_output_tensor.quantization = _clone_nhwc_quantization(
                old_concat_tensor.quantization
            )

        public_names = {
            *[str(name) for name in model_ir.inputs],
            *[str(name) for name in model_ir.outputs],
        }

        def _adapter_is_now_removable(adapter_op: OperatorIR) -> bool:
            output_names = [str(name) for name in adapter_op.outputs]
            return bool(output_names) and all(
                output_name not in public_names
                and not graph_index.consumer_indices(output_name)
                for output_name in output_names
            )

        def _walk_input_plans(
            input_plans: Tuple[_NhwcConcatInputPlan, ...],
        ) -> List[_NhwcConcatInputPlan]:
            plans: List[_NhwcConcatInputPlan] = []
            for input_plan in input_plans:
                plans.append(input_plan)
                plans.extend(_walk_input_plans(input_plan.add_operand_plans))
            return plans

        all_input_plans = _walk_input_plans(candidate.input_plans)

        remove_ops = [
            *[
                plan.adapter_op
                for plan in all_input_plans
                if plan.remove_adapter
                or _adapter_is_now_removable(plan.adapter_op)
            ],
            *[
                adapter_op
                for plan in all_input_plans
                for adapter_op in plan.extra_source_adapter_ops
                if _adapter_is_now_removable(adapter_op)
            ],
            *[
                adapter_op
                for plan in all_input_plans
                for adapter_op in plan.output_post_adapter_ops
            ],
            *candidate.post_ops,
        ]
        remove_indices = sorted(
            {
                int(operator_index)
                for operator in remove_ops
                if (
                    operator_index := graph_index.operator_index(operator)
                )
                is not None
            },
            reverse=True,
        )
        for remove_index in remove_indices:
            graph_index.remove_operator(remove_index)
        optimized += 1

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {stats_key: int(optimized)}


def run_nhwc_concat_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run the transactional rank-four Concat layout family group."""

    def _run_direct(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_direct_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_DIRECT_STATS_KEY, 0))}

    def _run_unary(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_unary_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_UNARY_STATS_KEY, 0))}

    def _run_pad(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_pad_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_PAD_STATS_KEY, 0))}

    def _run_dequantize(
        pass_state: ModelIRPassState,
    ) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_dequantize_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_DEQUANTIZE_STATS_KEY, 0)),
        }

    def _run_prelu(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_prelu_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_PRELU_STATS_KEY, 0))}

    def _run_softmax(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_softmax_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_SOFTMAX_STATS_KEY, 0))}

    def _run_swish(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_swish_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_SWISH_STATS_KEY, 0))}

    def _run_slice(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_slice_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_SLICE_STATS_KEY, 0))}

    def _run_split(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_split_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_SPLIT_STATS_KEY, 0))}

    def _run_add(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_add_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_ADD_STATS_KEY, 0))}

    def _run_leaky(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_leaky_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_LEAKY_STATS_KEY, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.nhwc_pre_concat_direct",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_direct,
                precondition=_has_nhwc_direct_concat_candidate,
                priority=10,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_unary",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_unary,
                precondition=_has_nhwc_unary_concat_candidate,
                priority=20,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_pad",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_pad,
                precondition=_has_nhwc_pad_concat_candidate,
                priority=30,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_dequantize",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_dequantize,
                precondition=_has_nhwc_dequantize_concat_candidate,
                priority=40,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_prelu",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_prelu,
                precondition=_has_nhwc_prelu_concat_candidate,
                priority=50,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_softmax",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_softmax,
                precondition=_has_nhwc_softmax_concat_candidate,
                priority=60,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_swish",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_swish,
                precondition=_has_nhwc_swish_concat_candidate,
                priority=70,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_slice",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_slice,
                precondition=_has_nhwc_slice_concat_candidate,
                priority=80,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_split",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_split,
                precondition=_has_nhwc_split_concat_candidate,
                priority=90,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_add",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_add,
                precondition=_has_nhwc_add_concat_candidate,
                priority=100,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_leaky",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_leaky,
                precondition=_has_nhwc_leaky_concat_candidate,
                priority=110,
                transactional=True,
            ),
        ],
        layout_state=layout_state,
        default_details={
            _DIRECT_STATS_KEY: 0,
            _UNARY_STATS_KEY: 0,
            _PAD_STATS_KEY: 0,
            _DEQUANTIZE_STATS_KEY: 0,
            _PRELU_STATS_KEY: 0,
            _SOFTMAX_STATS_KEY: 0,
            _SWISH_STATS_KEY: 0,
            _SLICE_STATS_KEY: 0,
            _SPLIT_STATS_KEY: 0,
            _ADD_STATS_KEY: 0,
            _LEAKY_STATS_KEY: 0,
        },
        diagnostics=diagnostics,
        preflight=lambda candidate_model: preflight_required_op_types(
            candidate_model,
            {"TRANSPOSE", "CONCATENATION"},
        ),
    )
    return {str(key): int(value) for key, value in details.items()}
