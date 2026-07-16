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
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_STATS_KEY = "optimized_fold_mul_add_mul_affine_chains"
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
class _AffineFoldPlan:
    mul1: OperatorIR
    add: OperatorIR
    mul2: OperatorIR
    source_name: str
    mul1_output_name: str
    add_output_name: str
    final_output_name: str
    mul2_constant_name: str
    mul1_output_shape: Tuple[int, ...]
    mul1_output_signature: Tuple[int, ...]
    constant_updates: Tuple[_ConstantUpdatePlan, ...]


def _tensor_contract(
    model_ir: ModelIR,
    name: str,
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
        len(shape) != len(signature)
        or any(int(value) < 0 for value in shape)
        or any(int(value) == 0 for value in signature)
    ):
        return None
    return _TensorContract(
        tensor=tensor,
        shape=shape,
        signature=signature,
    )


def _broadcast_shape(
    lhs: Sequence[int],
    rhs: Sequence[int],
) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(
            int(value)
            for value in np.broadcast_shapes(tuple(lhs), tuple(rhs))
        )
    except (TypeError, ValueError):
        return None


def _broadcast_signature(
    lhs: Sequence[int],
    rhs: Sequence[int],
) -> Optional[Tuple[int, ...]]:
    left = [int(value) for value in lhs]
    right = [int(value) for value in rhs]
    rank = max(len(left), len(right))
    left = [1] * (rank - len(left)) + left
    right = [1] * (rank - len(right)) + right
    output = []
    for left_value, right_value in zip(left, right):
        if left_value == right_value:
            output.append(left_value)
        elif left_value == 1:
            output.append(-1 if right_value < 0 else right_value)
        elif right_value == 1:
            output.append(-1 if left_value < 0 else left_value)
        elif left_value < 0 and right_value < 0:
            output.append(-1)
        elif left_value < 0 and right_value > 1:
            output.append(right_value)
        elif right_value < 0 and left_value > 1:
            output.append(left_value)
        else:
            return None
    return tuple(output)


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
    constant_flags = [tensor.data is not None for tensor in tensors if tensor]
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


def _constant_array(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    dtype: str,
    public_names: set[str],
) -> Optional[Tuple[_TensorContract, np.ndarray]]:
    contract = _tensor_contract(model_ir, str(name))
    expected_dtype = _FLOAT_DTYPES.get(str(dtype))
    if (
        contract is None
        or expected_dtype is None
        or str(name) in public_names
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or str(contract.tensor.dtype) != str(dtype)
        or contract.tensor.data is None
        or contract.tensor.is_variable
        or contract.tensor.quantization is not None
        or contract.signature != contract.shape
    ):
        return None
    try:
        data = np.asarray(contract.tensor.data)
    except (TypeError, ValueError):
        return None
    if (
        data.dtype != expected_dtype
        or tuple(int(value) for value in data.shape) != contract.shape
        or not np.all(np.isfinite(data))
    ):
        return None
    return contract, data


def _has_exact_producer(
    graph_index: ModelIRGraphIndex,
    name: str,
    operator_index: int,
) -> bool:
    return bool(
        str(name) not in graph_index.duplicate_producers
        and graph_index.producers.get(str(name)) == int(operator_index)
    )


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
    *,
    roles: Sequence[Tuple[str, np.ndarray, OperatorIR, int]],
    mul2: OperatorIR,
    mul2_constant_name: str,
) -> Optional[Tuple[_ConstantUpdatePlan, ...]]:
    grouped: dict[str, list[Tuple[np.ndarray, _ConstantUse]]] = {}
    for name, data, operator, input_index in roles:
        grouped.setdefault(str(name), []).append(
            (
                np.asarray(data),
                _ConstantUse(operator, int(input_index)),
            )
        )

    mul2_index = graph_index.operator_index(mul2)
    if mul2_index is None:
        return None
    reserved_names = set(str(name) for name in model_ir.tensors)
    updates = []
    for name, entries in grouped.items():
        replacement = np.asarray(entries[0][0])
        if any(
            candidate.shape != replacement.shape
            or candidate.dtype != replacement.dtype
            or not np.array_equal(candidate, replacement)
            for candidate, _ in entries[1:]
        ):
            return None
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
        allowed_consumers = list(use_indices)
        if str(name) == str(mul2_constant_name):
            allowed_consumers.append(int(mul2_index))
        update_in_place = Counter(
            graph_index.consumer_indices(str(name))
        ) == Counter(allowed_consumers)
        clone_name = (
            None
            if update_in_place
            else _unique_planned_name(f"{name}_folded", reserved_names)
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
    mul2: OperatorIR,
) -> Optional[_AffineFoldPlan]:
    mul2_index = graph_index.operator_index(mul2)
    if mul2_index is None or not _plain_binary(mul2, "MUL"):
        return None
    parsed_mul2 = _data_and_constant_inputs(model_ir, mul2)
    if parsed_mul2 is None:
        return None
    _, add_output_name, _, mul2_constant_name = parsed_mul2

    add_index = graph_index.producers.get(str(add_output_name))
    if (
        add_index is None
        or int(add_index) >= int(mul2_index)
        or not _has_exact_producer(
            graph_index,
            str(add_output_name),
            int(add_index),
        )
        or graph_index.consumer_indices(str(add_output_name))
        != [int(mul2_index)]
    ):
        return None
    add = model_ir.operators[int(add_index)]
    if not _plain_binary(add, "ADD") or str(add.outputs[0]) != str(
        add_output_name
    ):
        return None
    parsed_add = _data_and_constant_inputs(model_ir, add)
    if parsed_add is None:
        return None
    _, mul1_output_name, add_constant_index, add_constant_name = parsed_add

    mul1_index = graph_index.producers.get(str(mul1_output_name))
    if (
        mul1_index is None
        or int(mul1_index) >= int(add_index)
        or not _has_exact_producer(
            graph_index,
            str(mul1_output_name),
            int(mul1_index),
        )
        or graph_index.consumer_indices(str(mul1_output_name))
        != [int(add_index)]
    ):
        return None
    mul1 = model_ir.operators[int(mul1_index)]
    if not _plain_binary(mul1, "MUL") or str(mul1.outputs[0]) != str(
        mul1_output_name
    ):
        return None
    parsed_mul1 = _data_and_constant_inputs(model_ir, mul1)
    if parsed_mul1 is None:
        return None
    _, source_name, mul1_constant_index, mul1_constant_name = parsed_mul1

    final_output_name = str(mul2.outputs[0])
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_names = public_inputs | public_outputs
    if (
        len(
            {
                str(source_name),
                str(mul1_output_name),
                str(add_output_name),
                str(final_output_name),
            }
        )
        != 4
        or str(mul1_output_name) in public_names
        or str(add_output_name) in public_names
        or str(final_output_name) in public_inputs
        or not _has_exact_producer(
            graph_index,
            str(final_output_name),
            int(mul2_index),
        )
        or any(
            int(consumer_index) <= int(mul2_index)
            for consumer_index in graph_index.consumer_indices(
                str(final_output_name)
            )
        )
    ):
        return None
    source_producer_index = graph_index.producers.get(str(source_name))
    if str(source_name) in graph_index.duplicate_producers or (
        source_producer_index is None
        and str(source_name) not in public_inputs
    ):
        return None
    if source_producer_index is not None and int(source_producer_index) >= int(
        mul1_index
    ):
        return None

    source = _tensor_contract(model_ir, str(source_name))
    mul1_output = _tensor_contract(model_ir, str(mul1_output_name))
    add_output = _tensor_contract(model_ir, str(add_output_name))
    final_output = _tensor_contract(model_ir, str(final_output_name))
    if any(
        contract is None
        for contract in (source, mul1_output, add_output, final_output)
    ):
        return None
    assert source is not None
    assert mul1_output is not None
    assert add_output is not None
    assert final_output is not None
    dtype = str(source.tensor.dtype)
    if (
        dtype not in _FLOAT_DTYPES
        or any(
            str(contract.tensor.dtype) != dtype
            or contract.tensor.data is not None
            or contract.tensor.quantization is not None
            for contract in (source, mul1_output, add_output, final_output)
        )
    ):
        return None

    mul1_constant = _constant_array(
        model_ir,
        graph_index,
        str(mul1_constant_name),
        dtype,
        public_names,
    )
    add_constant = _constant_array(
        model_ir,
        graph_index,
        str(add_constant_name),
        dtype,
        public_names,
    )
    mul2_constant = _constant_array(
        model_ir,
        graph_index,
        str(mul2_constant_name),
        dtype,
        public_names,
    )
    if any(
        constant is None
        for constant in (mul1_constant, add_constant, mul2_constant)
    ):
        return None
    assert mul1_constant is not None
    assert add_constant is not None
    assert mul2_constant is not None
    mul1_constant_contract, mul1_values = mul1_constant
    add_constant_contract, add_values = add_constant
    mul2_constant_contract, mul2_values = mul2_constant

    original_shapes = (
        _broadcast_shape(source.shape, mul1_constant_contract.shape),
        _broadcast_shape(mul1_output.shape, add_constant_contract.shape),
        _broadcast_shape(add_output.shape, mul2_constant_contract.shape),
    )
    original_signatures = (
        _broadcast_signature(
            source.signature,
            mul1_constant_contract.signature,
        ),
        _broadcast_signature(
            mul1_output.signature,
            add_constant_contract.signature,
        ),
        _broadcast_signature(
            add_output.signature,
            mul2_constant_contract.signature,
        ),
    )
    if original_shapes != (
        mul1_output.shape,
        add_output.shape,
        final_output.shape,
    ) or original_signatures != (
        mul1_output.signature,
        add_output.signature,
        final_output.signature,
    ):
        return None

    expected_dtype = _FLOAT_DTYPES[dtype]
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            mul1_replacement = np.asarray(
                np.multiply(mul1_values, mul2_values),
                dtype=expected_dtype,
            )
            add_replacement = np.asarray(
                np.multiply(add_values, mul2_values),
                dtype=expected_dtype,
            )
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(mul1_replacement)) or not np.all(
        np.isfinite(add_replacement)
    ):
        return None

    folded_mul_shape = _broadcast_shape(
        source.shape,
        mul1_replacement.shape,
    )
    folded_mul_signature = _broadcast_signature(
        source.signature,
        mul1_replacement.shape,
    )
    folded_add_shape = (
        None
        if folded_mul_shape is None
        else _broadcast_shape(folded_mul_shape, add_replacement.shape)
    )
    folded_add_signature = (
        None
        if folded_mul_signature is None
        else _broadcast_signature(
            folded_mul_signature,
            add_replacement.shape,
        )
    )
    if (
        folded_mul_shape is None
        or folded_mul_signature is None
        or folded_add_shape != final_output.shape
        or folded_add_signature != final_output.signature
    ):
        return None

    constant_updates = _plan_constant_updates(
        model_ir,
        graph_index,
        roles=(
            (
                str(mul1_constant_name),
                mul1_replacement,
                mul1,
                int(mul1_constant_index),
            ),
            (
                str(add_constant_name),
                add_replacement,
                add,
                int(add_constant_index),
            ),
        ),
        mul2=mul2,
        mul2_constant_name=str(mul2_constant_name),
    )
    if constant_updates is None:
        return None
    return _AffineFoldPlan(
        mul1=mul1,
        add=add,
        mul2=mul2,
        source_name=str(source_name),
        mul1_output_name=str(mul1_output_name),
        add_output_name=str(add_output_name),
        final_output_name=str(final_output_name),
        mul2_constant_name=str(mul2_constant_name),
        mul1_output_shape=tuple(folded_mul_shape),
        mul1_output_signature=tuple(folded_mul_signature),
        constant_updates=constant_updates,
    )


def _plans_equal(expected: _AffineFoldPlan, actual: _AffineFoldPlan) -> bool:
    if (
        expected.mul1 is not actual.mul1
        or expected.add is not actual.add
        or expected.mul2 is not actual.mul2
        or expected.source_name != actual.source_name
        or expected.mul1_output_name != actual.mul1_output_name
        or expected.add_output_name != actual.add_output_name
        or expected.final_output_name != actual.final_output_name
        or expected.mul2_constant_name != actual.mul2_constant_name
        or expected.mul1_output_shape != actual.mul1_output_shape
        or expected.mul1_output_signature != actual.mul1_output_signature
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
            or expected_update.clone_name != actual_update.clone_name
            or expected_update.uses != actual_update.uses
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
    plan: _AffineFoldPlan,
) -> bool:
    current = _resolve_candidate(model_ir, graph_index, plan.mul2)
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

    mul1_output = model_ir.tensors[plan.mul1_output_name]
    mul1_output.shape = [int(value) for value in plan.mul1_output_shape]
    mul1_output.shape_signature = [
        int(value) for value in plan.mul1_output_signature
    ]

    mul2_index = graph_index.operator_index(plan.mul2)
    if mul2_index is None:
        return False
    graph_index.remove_operator(int(mul2_index))
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.add,
        new_outputs=[str(plan.final_output_name)],
        graph_index=graph_index,
    )
    return True


def optimize_fold_mul_add_mul_affine_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Fold strict floating-point MUL/ADD/MUL affine tails transactionally."""

    rewrite_limit = max(0, int(max_rewrites))
    if rewrite_limit == 0:
        return {_STATS_KEY: 0}

    required_counts = {"MUL": 2, "ADD": 1}
    for operator in model_ir.operators:
        op_type = str(operator.op_type)
        if op_type in required_counts and required_counts[op_type] > 0:
            required_counts[op_type] -= 1
        if all(value == 0 for value in required_counts.values()):
            break
    if any(value > 0 for value in required_counts.values()):
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
    for mul2 in candidates:
        if rewritten >= rewrite_limit or mul2 is None:
            break
        if active_index.operator_index(mul2) is None:
            continue
        plan = _resolve_candidate(model_ir, active_index, mul2)
        if plan is not None and _apply_plan(model_ir, active_index, plan):
            rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}
