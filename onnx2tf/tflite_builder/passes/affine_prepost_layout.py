from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_STATS_KEY = "optimized_transpose_mul_add_const_prepost_nhwc_chains"
_NCHW_TO_NHWC = (0, 2, 3, 1)
_NHWC_TO_NCHW = (0, 3, 1, 2)
_FLOAT_DTYPES = {
    "FLOAT16": np.dtype(np.float16),
    "FLOAT32": np.dtype(np.float32),
    "FLOAT64": np.dtype(np.float64),
}


@dataclass(frozen=True)
class _TensorContract:
    tensor: TensorIR
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _ConstantUse:
    operator: OperatorIR
    input_index: int


@dataclass(frozen=True)
class _ConstantUpdatePlan:
    name: str
    tensor: TensorIR
    data: np.ndarray
    uses: Tuple[_ConstantUse, ...]
    clone_name: Optional[str]


@dataclass(frozen=True)
class _AffinePrepostPlan:
    pre: OperatorIR
    mul: OperatorIR
    add: OperatorIR
    posts: Tuple[OperatorIR, ...]
    source_name: str
    pre_output_name: str
    mul_output_name: str
    add_output_name: str
    canonical_output_name: str
    alias_output_names: Tuple[str, ...]
    remove_pre: bool
    mul_output_shape: Tuple[int, ...]
    mul_output_signature: Tuple[int, ...]
    add_output_shape: Tuple[int, ...]
    add_output_signature: Tuple[int, ...]
    output_logical_layout: str
    output_physical_layout: str
    constant_updates: Tuple[_ConstantUpdatePlan, ...]


def _tensor_contract(
    model_ir: ModelIR,
    name: str,
    rank: int,
) -> Optional[_TensorContract]:
    tensor = model_ir.tensors.get(str(name))
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
    if (
        len(shape) != int(rank)
        or len(signature) != int(rank)
        or any(int(value) <= 0 for value in shape)
        or any(int(value) == 0 for value in signature)
    ):
        return None
    return _TensorContract(
        tensor=tensor,
        shape=shape,
        signature=signature,
    )


def _permute(values: Sequence[int], perm: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(values[int(index)]) for index in perm)


def _broadcasts_to(
    shape: Sequence[int],
    target: Sequence[int],
) -> bool:
    try:
        broadcast = tuple(
            int(value)
            for value in np.broadcast_shapes(tuple(shape), tuple(target))
        )
    except (TypeError, ValueError):
        return False
    return broadcast == tuple(int(value) for value in target)


def _plain_binary(operator: OperatorIR, op_type: str) -> bool:
    return bool(
        str(operator.op_type) == str(op_type)
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
        and str(
            operator.options.get("fusedActivationFunction", "NONE")
        ).upper()
        == "NONE"
    )


def _data_and_constant_inputs(
    model_ir: ModelIR,
    operator: OperatorIR,
) -> Optional[Tuple[int, str, int, str]]:
    if len(operator.inputs) != 2:
        return None
    names = [str(value) for value in operator.inputs]
    tensors = [model_ir.tensors.get(name) for name in names]
    if any(tensor is None for tensor in tensors):
        return None
    constant_flags = [bool(tensor and tensor.data is not None) for tensor in tensors]
    if constant_flags.count(True) != 1:
        return None
    constant_index = int(constant_flags.index(True))
    data_index = 1 - constant_index
    return (
        data_index,
        names[data_index],
        constant_index,
        names[constant_index],
    )


def _has_exact_producer(
    graph_index: ModelIRGraphIndex,
    name: str,
    operator_index: int,
) -> bool:
    return bool(
        str(name) not in graph_index.duplicate_producers
        and graph_index.producers.get(str(name)) == int(operator_index)
    )


def _typed_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    expected: Tuple[int, ...],
    public_names: set[str],
) -> bool:
    if len(operator.inputs) != 2:
        return False
    name = str(operator.inputs[1])
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or name in public_names
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or str(tensor.dtype) not in {"INT32", "INT64"}
        or tensor.is_variable
        or tensor.quantization is not None
    ):
        return False
    try:
        data = np.asarray(tensor.data)
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
        values = tuple(int(value) for value in data.reshape(-1).tolist())
    except (TypeError, ValueError):
        return False
    expected_dtype = np.dtype(
        np.int32 if str(tensor.dtype) == "INT32" else np.int64
    )
    return bool(
        data.dtype == expected_dtype
        and data.shape == (len(expected),)
        and shape == (len(expected),)
        and signature == shape
        and values == tuple(expected)
    )


def _constant_replacement(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    dtype: str,
    old_nchw_shape: Tuple[int, ...],
    target_nhwc_shape: Tuple[int, ...],
    public_names: set[str],
    allow_direct_non_rank4: bool = False,
) -> Optional[np.ndarray]:
    tensor = model_ir.tensors.get(str(name))
    expected_dtype = _FLOAT_DTYPES.get(str(dtype))
    if tensor is None:
        return None
    try:
        shape = tuple(int(value) for value in tensor.shape)
        signature = (
            shape
            if tensor.shape_signature is None
            else tuple(int(value) for value in tensor.shape_signature)
        )
        data = np.asarray(tensor.data)
    except (TypeError, ValueError):
        return None
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
    ):
        return None
    if int(data.size) == 1:
        return np.asarray(data)
    if data.ndim != 4:
        return (
            np.asarray(data)
            if allow_direct_non_rank4
            and _broadcasts_to(shape, target_nhwc_shape)
            else None
        )

    direct_ok = _broadcasts_to(shape, target_nhwc_shape)
    old_ok = _broadcasts_to(shape, old_nchw_shape)
    rotated = np.transpose(data, _NCHW_TO_NHWC).astype(
        expected_dtype,
        copy=False,
    )
    rotated_shape = tuple(int(value) for value in rotated.shape)
    rotated_ok = _broadcasts_to(rotated_shape, target_nhwc_shape)
    logical_layout = str(tensor.logical_layout).upper()
    physical_layout = str(tensor.physical_layout).upper()

    if "NHWC" in {logical_layout, physical_layout} and direct_ok:
        return np.asarray(data)
    if "NCHW" in {logical_layout, physical_layout} and rotated_ok:
        return np.asarray(rotated)
    if shape[0] == 1 and shape[1] > 1 and shape[2:] == (1, 1):
        return np.asarray(rotated) if rotated_ok else None
    if shape[:3] == (1, 1, 1) and shape[3] > 1:
        return np.asarray(data) if direct_ok else None
    if old_ok and rotated_ok and not direct_ok:
        return np.asarray(rotated)
    if direct_ok and not old_ok:
        return np.asarray(data)
    if direct_ok and rotated_ok:
        if data.shape == rotated.shape and np.array_equal(data, rotated):
            return np.asarray(data)
        return None
    if rotated_ok:
        return np.asarray(rotated)
    if direct_ok:
        return np.asarray(data)
    return None


def _unique_planned_name(base: str, reserved_names: set[str]) -> str:
    candidate = str(base)
    serial = 1
    while candidate in reserved_names:
        candidate = f"{base}_{serial}"
        serial += 1
    reserved_names.add(candidate)
    return candidate


def _plan_constant_updates(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    roles: Sequence[Tuple[str, np.ndarray, OperatorIR, int]],
) -> Optional[Tuple[_ConstantUpdatePlan, ...]]:
    grouped: dict[str, list[Tuple[np.ndarray, _ConstantUse]]] = {}
    for name, data, operator, input_index in roles:
        grouped.setdefault(str(name), []).append(
            (
                np.asarray(data),
                _ConstantUse(operator, int(input_index)),
            )
        )
    reserved_names = set(str(name) for name in model_ir.tensors)
    updates = []
    for name, entries in grouped.items():
        replacement = np.asarray(entries[0][0])
        if any(
            candidate.dtype != replacement.dtype
            or candidate.shape != replacement.shape
            or not np.array_equal(candidate, replacement)
            for candidate, _ in entries[1:]
        ):
            return None
        original = np.asarray(model_ir.tensors[name].data)
        if (
            original.dtype == replacement.dtype
            and original.shape == replacement.shape
            and np.array_equal(original, replacement)
        ):
            continue
        uses = tuple(use for _, use in entries)
        use_indices = []
        for use in uses:
            operator_index = graph_index.operator_index(use.operator)
            if (
                operator_index is None
                or use.input_index < 0
                or use.input_index >= len(use.operator.inputs)
                or str(use.operator.inputs[use.input_index]) != str(name)
            ):
                return None
            use_indices.append(int(operator_index))
        update_in_place = Counter(
            graph_index.consumer_indices(str(name))
        ) == Counter(use_indices)
        clone_name = (
            None
            if update_in_place
            else _unique_planned_name(f"{name}_nhwc", reserved_names)
        )
        updates.append(
            _ConstantUpdatePlan(
                name=str(name),
                tensor=model_ir.tensors[str(name)],
                data=np.asarray(replacement),
                uses=uses,
                clone_name=clone_name,
            )
        )
    return tuple(updates)


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    mul: OperatorIR,
) -> Optional[_AffinePrepostPlan]:
    mul_index = graph_index.operator_index(mul)
    if mul_index is None or not _plain_binary(mul, "MUL"):
        return None
    parsed_mul = _data_and_constant_inputs(model_ir, mul)
    if parsed_mul is None:
        return None
    _, pre_output_name, mul_constant_index, mul_constant_name = parsed_mul

    pre_index = graph_index.producers.get(str(pre_output_name))
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
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
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
    if (
        mul_output_name in public_names
        or not _has_exact_producer(
            graph_index,
            mul_output_name,
            int(mul_index),
        )
        or len(graph_index.consumer_indices(mul_output_name)) != 1
    ):
        return None
    add_index = int(graph_index.consumer_indices(mul_output_name)[0])
    if int(mul_index) >= int(add_index):
        return None
    add = model_ir.operators[int(add_index)]
    if not _plain_binary(add, "ADD"):
        return None
    parsed_add = _data_and_constant_inputs(model_ir, add)
    if parsed_add is None:
        return None
    _, add_data_name, add_constant_index, add_constant_name = parsed_add
    if str(add_data_name) != mul_output_name:
        return None

    add_output_name = str(add.outputs[0])
    if (
        add_output_name in public_names
        or not _has_exact_producer(
            graph_index,
            add_output_name,
            int(add_index),
        )
    ):
        return None
    post_indices = graph_index.consumer_indices(add_output_name)
    if not post_indices or any(int(index) <= int(add_index) for index in post_indices):
        return None
    posts = []
    post_output_names = []
    for post_index in post_indices:
        post = model_ir.operators[int(post_index)]
        if (
            str(post.op_type) != "TRANSPOSE"
            or len(post.inputs) != 2
            or len(post.outputs) != 1
            or str(post.inputs[0]) != add_output_name
            or not _typed_permutation(
                model_ir,
                graph_index,
                post,
                _NCHW_TO_NHWC,
                public_names,
            )
        ):
            return None
        output_name = str(post.outputs[0])
        if (
            output_name in public_names
            or output_name in graph_index.duplicate_producers
            or not _has_exact_producer(
                graph_index,
                output_name,
                int(post_index),
            )
            or any(
                int(consumer_index) <= int(post_index)
                for consumer_index in graph_index.consumer_indices(output_name)
            )
        ):
            return None
        posts.append(post)
        post_output_names.append(output_name)
    if len(set(post_output_names)) != len(post_output_names):
        return None

    source = _tensor_contract(model_ir, source_name, 4)
    pre_output = _tensor_contract(model_ir, pre_output_name, 4)
    mul_output = _tensor_contract(model_ir, mul_output_name, 4)
    add_output = _tensor_contract(model_ir, add_output_name, 4)
    post_outputs = [
        _tensor_contract(model_ir, output_name, 4)
        for output_name in post_output_names
    ]
    if any(
        contract is None
        for contract in (source, pre_output, mul_output, add_output, *post_outputs)
    ):
        return None
    assert source is not None
    assert pre_output is not None
    assert mul_output is not None
    assert add_output is not None
    resolved_post_outputs = tuple(
        contract for contract in post_outputs if contract is not None
    )
    dtype = str(source.tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(
            str(contract.tensor.dtype) != dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in (
                source,
                pre_output,
                mul_output,
                add_output,
                *resolved_post_outputs,
            )
        )
        or pre_output.shape != _permute(source.shape, _NHWC_TO_NCHW)
        or pre_output.signature != _permute(
            source.signature,
            _NHWC_TO_NCHW,
        )
        or mul_output.shape != pre_output.shape
        or mul_output.signature != pre_output.signature
        or add_output.shape != pre_output.shape
        or add_output.signature != pre_output.signature
    ):
        return None
    target_shape = _permute(add_output.shape, _NCHW_TO_NHWC)
    target_signature = _permute(add_output.signature, _NCHW_TO_NHWC)
    canonical_post_output = resolved_post_outputs[0]
    if any(
        contract.shape != target_shape
        or contract.signature != target_signature
        or str(contract.tensor.logical_layout)
        != str(canonical_post_output.tensor.logical_layout)
        or str(contract.tensor.physical_layout)
        != str(canonical_post_output.tensor.physical_layout)
        for contract in resolved_post_outputs
    ):
        return None

    mul_replacement = _constant_replacement(
        model_ir,
        graph_index,
        name=str(mul_constant_name),
        dtype=dtype,
        old_nchw_shape=pre_output.shape,
        target_nhwc_shape=target_shape,
        public_names=public_names,
    )
    add_replacement = _constant_replacement(
        model_ir,
        graph_index,
        name=str(add_constant_name),
        dtype=dtype,
        old_nchw_shape=pre_output.shape,
        target_nhwc_shape=target_shape,
        public_names=public_names,
    )
    if (
        mul_replacement is None
        or add_replacement is None
        or not _broadcasts_to(mul_replacement.shape, target_shape)
        or not _broadcasts_to(add_replacement.shape, target_shape)
    ):
        return None
    constant_updates = _plan_constant_updates(
        model_ir,
        graph_index,
        (
            (
                str(mul_constant_name),
                mul_replacement,
                mul,
                int(mul_constant_index),
            ),
            (
                str(add_constant_name),
                add_replacement,
                add,
                int(add_constant_index),
            ),
        ),
    )
    if constant_updates is None:
        return None

    pre_users = graph_index.consumer_indices(pre_output_name)
    if any(int(index) <= int(pre_index) for index in pre_users):
        return None
    canonical_output_name = str(post_output_names[0])
    return _AffinePrepostPlan(
        pre=pre,
        mul=mul,
        add=add,
        posts=tuple(posts),
        source_name=source_name,
        pre_output_name=pre_output_name,
        mul_output_name=mul_output_name,
        add_output_name=add_output_name,
        canonical_output_name=canonical_output_name,
        alias_output_names=tuple(post_output_names[1:]),
        remove_pre=Counter(pre_users) == Counter([int(mul_index)]),
        mul_output_shape=target_shape,
        mul_output_signature=target_signature,
        add_output_shape=target_shape,
        add_output_signature=target_signature,
        output_logical_layout=str(canonical_post_output.tensor.logical_layout),
        output_physical_layout=str(
            canonical_post_output.tensor.physical_layout
        ),
        constant_updates=constant_updates,
    )


def _plans_equal(expected: _AffinePrepostPlan, actual: _AffinePrepostPlan) -> bool:
    if (
        expected.pre is not actual.pre
        or expected.mul is not actual.mul
        or expected.add is not actual.add
        or expected.posts != actual.posts
        or expected.source_name != actual.source_name
        or expected.pre_output_name != actual.pre_output_name
        or expected.mul_output_name != actual.mul_output_name
        or expected.add_output_name != actual.add_output_name
        or expected.canonical_output_name != actual.canonical_output_name
        or expected.alias_output_names != actual.alias_output_names
        or expected.remove_pre != actual.remove_pre
        or expected.mul_output_shape != actual.mul_output_shape
        or expected.mul_output_signature != actual.mul_output_signature
        or expected.add_output_shape != actual.add_output_shape
        or expected.add_output_signature != actual.add_output_signature
        or expected.output_logical_layout != actual.output_logical_layout
        or expected.output_physical_layout != actual.output_physical_layout
        or len(expected.constant_updates) != len(actual.constant_updates)
    ):
        return False
    for expected_update, actual_update in zip(
        expected.constant_updates,
        actual.constant_updates,
    ):
        if (
            expected_update.name != actual_update.name
            or expected_update.tensor is not actual_update.tensor
            or expected_update.uses != actual_update.uses
            or expected_update.clone_name != actual_update.clone_name
            or expected_update.data.dtype != actual_update.data.dtype
            or expected_update.data.shape != actual_update.data.shape
            or not np.array_equal(
                expected_update.data,
                actual_update.data,
            )
        ):
            return False
    return True


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _AffinePrepostPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.mul)
    if current is None or not _plans_equal(plan, current):
        return False

    for update in plan.constant_updates:
        target = update.tensor
        if update.clone_name is not None:
            if update.clone_name in model_ir.tensors:
                return False
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
            for use in update.uses:
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=use.operator,
                    input_index=int(use.input_index),
                    new_input_name=str(update.clone_name),
                    graph_index=graph_index,
                )
        target.data = np.asarray(update.data)
        target.shape = [int(value) for value in update.data.shape]
        target.shape_signature = [int(value) for value in update.data.shape]

    mul_inputs = [str(name) for name in plan.mul.inputs]
    pre_slots = [
        index
        for index, name in enumerate(mul_inputs)
        if str(name) == plan.pre_output_name
    ]
    if len(pre_slots) != 1:
        return False
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.mul,
        input_index=int(pre_slots[0]),
        new_input_name=plan.source_name,
        graph_index=graph_index,
    )

    mul_output = model_ir.tensors[plan.mul_output_name]
    mul_output.shape = [int(value) for value in plan.mul_output_shape]
    mul_output.shape_signature = [
        int(value) for value in plan.mul_output_signature
    ]
    mul_output.logical_layout = str(plan.output_logical_layout)
    mul_output.physical_layout = str(plan.output_physical_layout)
    add_output = model_ir.tensors[plan.add_output_name]
    add_output.shape = [int(value) for value in plan.add_output_shape]
    add_output.shape_signature = [
        int(value) for value in plan.add_output_signature
    ]
    add_output.logical_layout = str(plan.output_logical_layout)
    add_output.physical_layout = str(plan.output_physical_layout)

    for alias_name in plan.alias_output_names:
        _replace_tensor_inputs(
            model_ir=model_ir,
            src_name=str(alias_name),
            dst_name=plan.canonical_output_name,
            graph_index=graph_index,
        )

    remove_ops = list(plan.posts)
    if plan.remove_pre:
        remove_ops.append(plan.pre)
    remove_indices = []
    for operator in remove_ops:
        operator_index = graph_index.operator_index(operator)
        if operator_index is None:
            return False
        remove_indices.append(int(operator_index))
    graph_index.remove_operators(remove_indices)
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.add,
        new_outputs=[plan.canonical_output_name],
        graph_index=graph_index,
    )
    return True


def optimize_transpose_mul_add_const_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Lift strict Transpose-MUL-ADD-Transpose affine islands to NHWC."""

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
        _prune_unused_tensors(model_ir, layout_state=layout_state)
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

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
