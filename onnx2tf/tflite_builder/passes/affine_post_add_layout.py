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
    _broadcasts_to,
    _constant_replacement,
    _data_and_constant_inputs,
    _has_exact_producer,
    _permute,
    _plain_binary,
    _tensor_contract,
    _typed_permutation,
    _unique_planned_name,
)


_STATS_KEY = "optimized_transpose_mul_posttranspose_add_nhwc_chains"


@dataclass(frozen=True)
class _ConstantUpdatePlan:
    name: str
    tensor: TensorIR
    data: np.ndarray
    input_index: int
    clone_name: Optional[str]


@dataclass(frozen=True)
class _AddTail:
    operator: OperatorIR
    input_index: int


@dataclass(frozen=True)
class _AffinePostAddPlan:
    pre: OperatorIR
    mul: OperatorIR
    post: OperatorIR
    add_tails: Tuple[_AddTail, ...]
    source_name: str
    pre_output_name: str
    mul_output_name: str
    post_output_name: str
    remove_pre: bool
    output_shape: Tuple[int, ...]
    output_signature: Tuple[int, ...]
    output_logical_layout: str
    output_physical_layout: str
    constant_update: _ConstantUpdatePlan


def _constant_contract(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    dtype: str,
    target_shape: Tuple[int, ...],
    public_names: set[str],
) -> bool:
    tensor = model_ir.tensors.get(str(name))
    expected_dtype = _FLOAT_DTYPES.get(str(dtype))
    if tensor is None:
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
    if (
        expected_dtype is None
        or str(name) in public_names
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or str(tensor.dtype) != str(dtype)
        or tensor.data is None
        or tensor.is_variable
        or tensor.quantization is not None
        or data.dtype != expected_dtype
        or tuple(int(value) for value in data.shape) != shape
        or signature != shape
        or not np.all(np.isfinite(data))
        or not _broadcasts_to(shape, target_shape)
    ):
        return False
    if int(data.size) == 1:
        return True
    return bool(
        len(shape) == 4
        and shape[:3] == (1, 1, 1)
        and shape[3] == target_shape[3]
    )


def _plan_constant_update(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    mul: OperatorIR,
    input_index: int,
    name: str,
    replacement: np.ndarray,
) -> Optional[_ConstantUpdatePlan]:
    tensor = model_ir.tensors.get(str(name))
    mul_index = graph_index.operator_index(mul)
    if tensor is None or mul_index is None:
        return None
    original = np.asarray(tensor.data)
    unchanged = bool(
        original.dtype == replacement.dtype
        and original.shape == replacement.shape
        and np.array_equal(original, replacement)
    )
    clone_name = None
    if not unchanged and graph_index.consumer_indices(str(name)) != [int(mul_index)]:
        clone_name = _unique_planned_name(
            f"{name}_nhwc",
            set(str(tensor_name) for tensor_name in model_ir.tensors),
        )
    return _ConstantUpdatePlan(
        name=str(name),
        tensor=tensor,
        data=np.asarray(replacement),
        input_index=int(input_index),
        clone_name=clone_name,
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    mul: OperatorIR,
) -> Optional[_AffinePostAddPlan]:
    mul_index = graph_index.operator_index(mul)
    if mul_index is None or not _plain_binary(mul, "MUL"):
        return None
    parsed_mul = _data_and_constant_inputs(model_ir, mul)
    if parsed_mul is None:
        return None
    _, pre_output_name, constant_input_index, constant_name = parsed_mul

    pre_index = graph_index.producers.get(str(pre_output_name))
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    if (
        pre_index is None
        or int(pre_index) >= int(mul_index)
        or not _has_exact_producer(
            graph_index,
            str(pre_output_name),
            int(pre_index),
        )
    ):
        return None
    pre = model_ir.operators[int(pre_index)]
    if (
        str(pre.op_type) != "TRANSPOSE"
        or len(pre.inputs) != 2
        or len(pre.outputs) != 1
        or str(pre.outputs[0]) != str(pre_output_name)
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
    if (
        source_name in public_outputs
        or str(pre_output_name) in public_names
        or source_name in graph_index.duplicate_producers
    ):
        return None
    source_producer_index = graph_index.producers.get(source_name)
    if source_producer_index is None and source_name not in public_inputs:
        return None
    if source_producer_index is not None and int(source_producer_index) >= int(
        pre_index
    ):
        return None

    mul_output_name = str(mul.outputs[0])
    mul_users = graph_index.consumer_indices(mul_output_name)
    if (
        mul_output_name in public_names
        or not _has_exact_producer(
            graph_index,
            mul_output_name,
            int(mul_index),
        )
        or len(mul_users) != 1
    ):
        return None
    post_index = int(mul_users[0])
    if int(mul_index) >= int(post_index):
        return None
    post = model_ir.operators[int(post_index)]
    if (
        str(post.op_type) != "TRANSPOSE"
        or len(post.inputs) != 2
        or len(post.outputs) != 1
        or str(post.inputs[0]) != mul_output_name
        or not _typed_permutation(
            model_ir,
            graph_index,
            post,
            _NCHW_TO_NHWC,
            public_names,
        )
    ):
        return None
    post_output_name = str(post.outputs[0])
    add_indices = graph_index.consumer_indices(post_output_name)
    if (
        post_output_name in public_names
        or not _has_exact_producer(
            graph_index,
            post_output_name,
            int(post_index),
        )
        or not add_indices
        or any(int(index) <= int(post_index) for index in add_indices)
    ):
        return None

    source = _tensor_contract(model_ir, source_name, 4)
    pre_output = _tensor_contract(model_ir, pre_output_name, 4)
    mul_output = _tensor_contract(model_ir, mul_output_name, 4)
    post_output = _tensor_contract(model_ir, post_output_name, 4)
    if any(
        contract is None
        for contract in (source, pre_output, mul_output, post_output)
    ):
        return None
    assert source is not None
    assert pre_output is not None
    assert mul_output is not None
    assert post_output is not None
    dtype = str(source.tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(
            str(contract.tensor.dtype) != dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in (source, pre_output, mul_output, post_output)
        )
        or pre_output.shape != _permute(source.shape, _NHWC_TO_NCHW)
        or pre_output.signature != _permute(
            source.signature,
            _NHWC_TO_NCHW,
        )
        or mul_output.shape != pre_output.shape
        or mul_output.signature != pre_output.signature
        or post_output.shape != source.shape
        or post_output.signature != source.signature
    ):
        return None

    add_tails = []
    for add_index in add_indices:
        add = model_ir.operators[int(add_index)]
        if not _plain_binary(add, "ADD"):
            return None
        matches = [
            input_index
            for input_index, name in enumerate(add.inputs)
            if str(name) == post_output_name
        ]
        if len(matches) != 1:
            return None
        input_index = int(matches[0])
        side_name = str(add.inputs[1 - input_index])
        output_name = str(add.outputs[0])
        add_output = _tensor_contract(model_ir, output_name, 4)
        if (
            output_name in graph_index.duplicate_producers
            or not _has_exact_producer(
                graph_index,
                output_name,
                int(add_index),
            )
            or add_output is None
            or str(add_output.tensor.dtype) != dtype
            or add_output.tensor.data is not None
            or add_output.tensor.quantization is not None
            or add_output.shape != post_output.shape
            or add_output.signature != post_output.signature
            or any(
                int(consumer_index) <= int(add_index)
                for consumer_index in graph_index.consumer_indices(output_name)
            )
            or not _constant_contract(
                model_ir,
                graph_index,
                name=side_name,
                dtype=dtype,
                target_shape=post_output.shape,
                public_names=public_names,
            )
        ):
            return None
        add_tails.append(_AddTail(add, input_index))

    replacement = _constant_replacement(
        model_ir,
        graph_index,
        name=str(constant_name),
        dtype=dtype,
        old_nchw_shape=pre_output.shape,
        target_nhwc_shape=post_output.shape,
        public_names=public_names,
        allow_direct_non_rank4=True,
    )
    if replacement is None or not _broadcasts_to(
        replacement.shape,
        post_output.shape,
    ):
        return None
    constant_update = _plan_constant_update(
        model_ir,
        graph_index,
        mul=mul,
        input_index=int(constant_input_index),
        name=str(constant_name),
        replacement=replacement,
    )
    if constant_update is None:
        return None

    pre_users = graph_index.consumer_indices(str(pre_output_name))
    if any(int(index) <= int(pre_index) for index in pre_users):
        return None
    return _AffinePostAddPlan(
        pre=pre,
        mul=mul,
        post=post,
        add_tails=tuple(add_tails),
        source_name=source_name,
        pre_output_name=str(pre_output_name),
        mul_output_name=mul_output_name,
        post_output_name=post_output_name,
        remove_pre=pre_users == [int(mul_index)],
        output_shape=post_output.shape,
        output_signature=post_output.signature,
        output_logical_layout=str(post_output.tensor.logical_layout),
        output_physical_layout=str(post_output.tensor.physical_layout),
        constant_update=constant_update,
    )


def _plans_equal(expected: _AffinePostAddPlan, actual: _AffinePostAddPlan) -> bool:
    expected_update = expected.constant_update
    actual_update = actual.constant_update
    return bool(
        expected.pre is actual.pre
        and expected.mul is actual.mul
        and expected.post is actual.post
        and expected.add_tails == actual.add_tails
        and expected.source_name == actual.source_name
        and expected.pre_output_name == actual.pre_output_name
        and expected.mul_output_name == actual.mul_output_name
        and expected.post_output_name == actual.post_output_name
        and expected.remove_pre == actual.remove_pre
        and expected.output_shape == actual.output_shape
        and expected.output_signature == actual.output_signature
        and expected.output_logical_layout == actual.output_logical_layout
        and expected.output_physical_layout == actual.output_physical_layout
        and expected_update.name == actual_update.name
        and expected_update.tensor is actual_update.tensor
        and expected_update.input_index == actual_update.input_index
        and expected_update.clone_name == actual_update.clone_name
        and expected_update.data.dtype == actual_update.data.dtype
        and expected_update.data.shape == actual_update.data.shape
        and np.array_equal(expected_update.data, actual_update.data)
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _AffinePostAddPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.mul)
    if current is None or not _plans_equal(plan, current):
        return False

    update = plan.constant_update
    mul_inputs = [str(name) for name in plan.mul.inputs]
    pre_slots = [
        index
        for index, name in enumerate(mul_inputs)
        if str(name) == plan.pre_output_name
    ]
    remove_ops = [plan.post]
    if plan.remove_pre:
        remove_ops.append(plan.pre)
    remove_indices = [
        graph_index.operator_index(operator) for operator in remove_ops
    ]
    if (
        len(pre_slots) != 1
        or any(index is None for index in remove_indices)
        or (
            update.clone_name is not None
            and update.clone_name in model_ir.tensors
        )
    ):
        return False

    target = update.tensor
    if update.clone_name is not None:
        target = TensorIR(
            name=str(update.clone_name),
            dtype=str(update.tensor.dtype),
            shape=[int(value) for value in update.data.shape],
            shape_signature=[int(value) for value in update.data.shape],
            data=np.asarray(update.data),
            is_variable=False,
            quantization=None,
            logical_layout=str(update.tensor.logical_layout),
            physical_layout=str(update.tensor.physical_layout),
            onnx_tensor_name=update.tensor.onnx_tensor_name,
        )
        model_ir.tensors[str(update.clone_name)] = target
        _replace_operator_input_at(
            model_ir=model_ir,
            op=plan.mul,
            input_index=int(update.input_index),
            new_input_name=str(update.clone_name),
            graph_index=graph_index,
        )
    target.data = np.asarray(update.data)
    target.shape = [int(value) for value in update.data.shape]
    target.shape_signature = [int(value) for value in update.data.shape]

    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.mul,
        input_index=int(pre_slots[0]),
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )
    for add_tail in plan.add_tails:
        _replace_operator_input_at(
            model_ir=model_ir,
            op=add_tail.operator,
            input_index=int(add_tail.input_index),
            new_input_name=plan.mul_output_name,
            graph_index=graph_index,
        )

    mul_output = model_ir.tensors[plan.mul_output_name]
    mul_output.shape = [int(value) for value in plan.output_shape]
    mul_output.shape_signature = [int(value) for value in plan.output_signature]
    mul_output.logical_layout = str(plan.output_logical_layout)
    mul_output.physical_layout = str(plan.output_physical_layout)

    graph_index.remove_operators([int(index) for index in remove_indices])
    return True


def optimize_transpose_mul_posttranspose_add_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift strict Transpose-MUL-Transpose-ADD fan-out islands to NHWC."""

    rewrite_limit = max(0, int(max_rewrites))
    required_counts = {"TRANSPOSE": 2, "MUL": 1, "ADD": 1}
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
            for index in active_index.operator_indices("MUL")
        ]
    )
    rewritten = 0
    for mul in candidates:
        if rewritten >= rewrite_limit or mul is None:
            break
        if active_index.operator_index(mul) is None:
            continue
        plan = _resolve_candidate(model_ir, active_index, mul)
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
