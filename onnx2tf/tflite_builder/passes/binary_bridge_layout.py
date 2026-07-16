from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _broadcast_shape_signatures,
    _broadcast_static_shapes,
    _clone_quantization,
    _invert_perm,
    _is_fully_known_positive_shape,
    _permute_shape,
    _prune_unused_tensors,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _shapes_match_if_known,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
)


_BINARY_OPS = frozenset({"ADD", "SUB", "MUL", "DIV"})
_SYMMETRIC_STATS_KEY = "removed_transpose_binary_bridges"
_ASYMMETRIC_STATS_KEY = "removed_transpose_binary_asymmetric_bridges"


@dataclass(frozen=True)
class _SymmetricPlan:
    binary: OperatorIR
    pre0: OperatorIR
    pre1: OperatorIR
    post: Optional[OperatorIR]
    mode: str
    original_inputs: Tuple[str, str]
    raw_inputs: Tuple[str, str]
    original_output: str
    final_output: str
    permutation: Tuple[int, ...]
    pre_permutation_name: str
    raw_output_name: Optional[str]
    raw_output_shape: Optional[Tuple[int, ...]]
    raw_output_signature: Optional[Tuple[int, ...]]
    raw_logical_layout: str
    raw_physical_layout: str
    first_consumer: Optional[OperatorIR]
    contracts: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class _AsymmetricPlan:
    binary: OperatorIR
    pre: OperatorIR
    post: OperatorIR
    transpose_on_lhs: bool
    transposed_input_name: str
    plain_input_name: str
    raw_input_name: str
    original_output: str
    final_output: str
    post_permutation_name: str
    expected_shape: Tuple[int, ...]
    expected_signature: Tuple[int, ...]
    expected_dtype: str
    expected_quantization: Any
    expected_logical_layout: str
    expected_physical_layout: str
    contracts: Tuple[Tuple[Any, ...], ...]


def _normalized_op_type(operator: OperatorIR) -> str:
    return str(operator.op_type).upper()


def _plain_binary(operator: OperatorIR) -> bool:
    return bool(
        _normalized_op_type(operator) in _BINARY_OPS
        and len(operator.inputs) == 2
        and len(operator.outputs) == 1
        and str(
            operator.options.get("fusedActivationFunction", "NONE")
        ).upper()
        in {"", "NONE"}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (
            str(value.dtype),
            tuple(int(v) for v in value.shape),
            tuple(value.reshape(-1).tolist()),
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _freeze_value(item)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if hasattr(value, "scale") and hasattr(value, "zero_point"):
        return (
            tuple(float(item) for item in value.scale),
            tuple(int(item) for item in value.zero_point),
            int(value.quantized_dimension),
            _freeze_value(value.min),
            _freeze_value(value.max),
        )
    return value


def _tensor_contract(name: str, tensor: TensorIR) -> Tuple[Any, ...]:
    return (
        str(name),
        id(tensor),
        str(tensor.dtype),
        tuple(int(value) for value in tensor.shape),
        tuple(
            int(value)
            for value in (
                tensor.shape_signature
                if tensor.shape_signature is not None
                else tensor.shape
            )
        ),
        bool(tensor.is_variable),
        _freeze_value(tensor.quantization),
        str(tensor.logical_layout),
        str(tensor.physical_layout),
    )


def _tensor_contracts(
    model_ir: ModelIR,
    names: Sequence[str],
) -> Optional[Tuple[Tuple[Any, ...], ...]]:
    contracts = []
    for name in names:
        tensor = model_ir.tensors.get(str(name))
        if tensor is None:
            return None
        contracts.append(_tensor_contract(str(name), tensor))
    return tuple(contracts)


def _operator_index(
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
) -> Optional[int]:
    return graph_index.operator_index(operator)


def _unique_producer(
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Optional[int]:
    normalized_name = str(name)
    if normalized_name in graph_index.duplicate_producers:
        return None
    producer = graph_index.producers.get(normalized_name)
    return None if producer is None else int(producer)


def _resolved_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    before_index: int,
) -> bool:
    normalized_name = str(name)
    tensor = model_ir.tensors.get(normalized_name)
    if tensor is None or normalized_name in graph_index.duplicate_producers:
        return False
    producer = graph_index.producers.get(normalized_name)
    if producer is not None:
        return int(producer) < int(before_index)
    return bool(
        normalized_name in {str(value) for value in model_ir.inputs}
        or tensor.data is not None
    )


def _typed_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    *,
    public_names: set[str],
) -> Optional[Tuple[int, ...]]:
    if _normalized_op_type(operator) != "TRANSPOSE" or len(operator.inputs) != 2:
        return None
    permutation_name = str(operator.inputs[1])
    permutation_tensor = model_ir.tensors.get(permutation_name)
    if (
        permutation_tensor is None
        or permutation_tensor.data is None
        or permutation_name in public_names
        or permutation_name in graph_index.producers
        or permutation_name in graph_index.duplicate_producers
        or bool(permutation_tensor.is_variable)
        or permutation_tensor.quantization is not None
        or str(permutation_tensor.dtype).upper() not in {"INT32", "INT64"}
    ):
        return None
    try:
        array = np.asarray(permutation_tensor.data)
    except Exception:
        return None
    if array.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        return None
    values = tuple(int(value) for value in array.reshape(-1).tolist())
    if len(values) == 0 or sorted(values) != list(range(len(values))):
        return None
    return values


def _inverse_permutations(
    first: Sequence[int],
    second: Sequence[int],
) -> bool:
    inverse = _invert_perm([int(value) for value in first])
    return inverse is not None and tuple(inverse) == tuple(int(v) for v in second)


def _signature_compatible(
    expected: Optional[Sequence[int]],
    actual: Optional[Sequence[int]],
) -> bool:
    if expected is None or actual is None or len(expected) != len(actual):
        return False
    return all(
        int(left) == int(right) or int(left) < 0 or int(right) < 0
        for left, right in zip(expected, actual)
    )


def _quantizations_equal(tensors: Sequence[TensorIR]) -> bool:
    values = {_freeze_value(tensor.quantization) for tensor in tensors}
    return len(values) <= 1


def _same_dtype(tensors: Sequence[TensorIR]) -> bool:
    return len({str(tensor.dtype) for tensor in tensors}) <= 1


def _unique_name(model_ir: ModelIR, base: str) -> str:
    candidate = str(base)
    serial = 0
    while candidate in model_ir.tensors:
        serial += 1
        candidate = f"{base}_{serial}"
    return candidate


def _layout_if_equal(tensors: Sequence[TensorIR], attribute: str) -> str:
    values = {str(getattr(tensor, attribute)) for tensor in tensors}
    return values.pop() if len(values) == 1 else LOGICAL_LAYOUT_UNKNOWN


def _resolve_symmetric(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    binary: OperatorIR,
) -> Optional[_SymmetricPlan]:
    binary_index = _operator_index(graph_index, binary)
    if binary_index is None or not _plain_binary(binary):
        return None
    original_inputs = tuple(str(value) for value in binary.inputs)
    original_output = str(binary.outputs[0])
    public_names = {
        str(value) for value in tuple(model_ir.inputs) + tuple(model_ir.outputs)
    }
    if (
        any(name in public_names for name in (*original_inputs, original_output))
        or graph_index.producers.get(original_output) != binary_index
    ):
        return None
    if any(name in graph_index.duplicate_producers for name in (*original_inputs, original_output)):
        return None
    pre_indices = tuple(
        _unique_producer(graph_index, name) for name in original_inputs
    )
    if any(index is None for index in pre_indices):
        return None
    pre0_index, pre1_index = (int(pre_indices[0]), int(pre_indices[1]))
    if pre0_index == pre1_index or not (
        pre0_index < binary_index and pre1_index < binary_index
    ):
        return None
    pre0 = model_ir.operators[pre0_index]
    pre1 = model_ir.operators[pre1_index]
    if (
        _normalized_op_type(pre0) != "TRANSPOSE"
        or _normalized_op_type(pre1) != "TRANSPOSE"
        or len(pre0.inputs) != 2
        or len(pre1.inputs) != 2
        or len(pre0.outputs) != 1
        or len(pre1.outputs) != 1
        or str(pre0.outputs[0]) != original_inputs[0]
        or str(pre1.outputs[0]) != original_inputs[1]
        or graph_index.consumer_indices(original_inputs[0]) != [binary_index]
        or graph_index.consumer_indices(original_inputs[1]) != [binary_index]
    ):
        return None
    permutation0 = _typed_permutation(
        model_ir,
        graph_index,
        pre0,
        public_names=public_names,
    )
    permutation1 = _typed_permutation(
        model_ir,
        graph_index,
        pre1,
        public_names=public_names,
    )
    if permutation0 is None or permutation0 != permutation1:
        return None
    raw_inputs = (str(pre0.inputs[0]), str(pre1.inputs[0]))
    if not _resolved_source(
        model_ir,
        graph_index,
        name=raw_inputs[0],
        before_index=pre0_index,
    ) or not _resolved_source(
        model_ir,
        graph_index,
        name=raw_inputs[1],
        before_index=pre1_index,
    ):
        return None

    out_users = sorted(set(graph_index.consumer_indices(original_output)))
    if len(out_users) == 0 or any(index <= binary_index for index in out_users):
        return None
    post_candidates = []
    unexpected_transpose = False
    for user_index in out_users:
        user = model_ir.operators[user_index]
        if _normalized_op_type(user) != "TRANSPOSE":
            continue
        user_permutation = _typed_permutation(
            model_ir,
            graph_index,
            user,
            public_names=public_names,
        )
        if (
            len(user.outputs) == 1
            and str(user.inputs[0]) == original_output
            and user_permutation is not None
            and _inverse_permutations(permutation0, user_permutation)
        ):
            post_candidates.append(user_index)
        else:
            unexpected_transpose = True
    if unexpected_transpose or len(post_candidates) > 1:
        return None

    post: Optional[OperatorIR] = None
    mode = "legacy_only"
    final_output = original_output
    first_consumer: Optional[OperatorIR] = model_ir.operators[out_users[0]]
    if len(post_candidates) == 1:
        post_index = int(post_candidates[0])
        post = model_ir.operators[post_index]
        if post_index <= binary_index:
            return None
        final_output = str(post.outputs[0])
        if (
            final_output in {str(value) for value in model_ir.inputs}
            or final_output in graph_index.duplicate_producers
        ):
            return None
        if graph_index.producers.get(final_output) != post_index:
            return None
        # The rewrite moves final_output's producer from the post-transpose to
        # the binary operator. Consumers may therefore precede the old post as
        # long as they follow the binary, which also repairs a historically
        # supported non-topological intermediate ordering.
        if any(
            index <= binary_index
            for index in set(graph_index.consumer_indices(final_output))
        ):
            return None
        extra_users = [index for index in out_users if index != post_index]
        if extra_users:
            if any(index <= post_index for index in extra_users):
                return None
            mode = "single_post_fanout"
        else:
            if graph_index.consumer_indices(original_output) != [post_index]:
                return None
            mode = "single_post"
        first_consumer = None

    contract_names = (
        raw_inputs[0],
        raw_inputs[1],
        original_inputs[0],
        original_inputs[1],
        original_output,
        final_output,
        str(pre0.inputs[1]),
        str(pre1.inputs[1]),
    )
    if post is not None:
        contract_names += (str(post.inputs[1]),)
    contracts = _tensor_contracts(model_ir, contract_names)
    if contracts is None:
        return None
    raw0 = model_ir.tensors[raw_inputs[0]]
    raw1 = model_ir.tensors[raw_inputs[1]]
    transposed0 = model_ir.tensors[original_inputs[0]]
    transposed1 = model_ir.tensors[original_inputs[1]]
    output = model_ir.tensors[original_output]
    final = model_ir.tensors[final_output]
    data_tensors = [raw0, raw1, transposed0, transposed1, output, final]
    if not _same_dtype(data_tensors) or not _all_per_tensor_quantized(data_tensors):
        return None
    if not _quantizations_equal(data_tensors):
        return None
    rank = len(permutation0)
    if any(len(tensor.shape) != rank for tensor in data_tensors):
        return None
    expected_transposed0 = _permute_shape(list(raw0.shape), list(permutation0))
    expected_transposed1 = _permute_shape(list(raw1.shape), list(permutation0))
    if (
        expected_transposed0 is None
        or expected_transposed1 is None
        or not _shapes_match_if_known(expected_transposed0, list(transposed0.shape))
        or not _shapes_match_if_known(expected_transposed1, list(transposed1.shape))
    ):
        return None
    raw_broadcast = _broadcast_static_shapes(list(raw0.shape), list(raw1.shape))
    if (
        raw_broadcast is None
        and _is_fully_known_positive_shape(list(raw0.shape))
        and _is_fully_known_positive_shape(list(raw1.shape))
    ):
        return None
    raw_signatures = (
        list(raw0.shape_signature or raw0.shape),
        list(raw1.shape_signature or raw1.shape),
    )
    broadcast_signature = _broadcast_shape_signatures(*raw_signatures)
    if broadcast_signature is None:
        return None
    expected_output_shape = (
        _permute_shape(raw_broadcast, list(permutation0))
        if raw_broadcast is not None
        else None
    )
    expected_output_signature = _permute_shape(
        broadcast_signature,
        list(permutation0),
    )
    if (
        expected_output_signature is None
        or (
            expected_output_shape is not None
            and not _shapes_match_if_known(expected_output_shape, list(output.shape))
        )
        or not _signature_compatible(
            expected_output_signature,
            output.shape_signature or output.shape,
        )
    ):
        return None
    expected_raw_shape = (
        list(raw_broadcast)
        if raw_broadcast is not None
        else _permute_shape(list(output.shape), list(_invert_perm(list(permutation0)) or []))
    )
    expected_raw_signature = broadcast_signature
    if expected_raw_shape is None:
        return None
    if mode != "legacy_only" and (
        not _shapes_match_if_known(expected_raw_shape, list(final.shape))
        or not _signature_compatible(
            expected_raw_signature,
            final.shape_signature or final.shape,
        )
    ):
        return None

    raw_output_name = None
    if mode == "legacy_only":
        raw_output_name = _unique_name(model_ir, f"{original_output}__raw")
    return _SymmetricPlan(
        binary=binary,
        pre0=pre0,
        pre1=pre1,
        post=post,
        mode=mode,
        original_inputs=original_inputs,
        raw_inputs=raw_inputs,
        original_output=original_output,
        final_output=final_output,
        permutation=tuple(permutation0),
        pre_permutation_name=str(pre0.inputs[1]),
        raw_output_name=raw_output_name,
        raw_output_shape=(
            tuple(int(value) for value in expected_raw_shape)
            if raw_output_name is not None
            else None
        ),
        raw_output_signature=(
            tuple(int(value) for value in expected_raw_signature)
            if raw_output_name is not None
            else None
        ),
        raw_logical_layout=_layout_if_equal((raw0, raw1), "logical_layout"),
        raw_physical_layout=_layout_if_equal((raw0, raw1), "physical_layout"),
        first_consumer=first_consumer,
        contracts=contracts,
    )


def _resolve_asymmetric(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    binary: OperatorIR,
) -> Optional[_AsymmetricPlan]:
    binary_index = _operator_index(graph_index, binary)
    if binary_index is None or not _plain_binary(binary):
        return None
    input_names = tuple(str(value) for value in binary.inputs)
    original_output = str(binary.outputs[0])
    public_names = {
        str(value) for value in tuple(model_ir.inputs) + tuple(model_ir.outputs)
    }
    if (
        original_output in public_names
        or original_output in graph_index.duplicate_producers
        or graph_index.producers.get(original_output) != binary_index
    ):
        return None
    input_producers = tuple(
        _unique_producer(graph_index, name) for name in input_names
    )
    transpose_matches = []
    for slot, producer_index in enumerate(input_producers):
        if producer_index is None or int(producer_index) >= binary_index:
            continue
        producer = model_ir.operators[int(producer_index)]
        if (
            _normalized_op_type(producer) == "TRANSPOSE"
            and len(producer.inputs) == 2
            and len(producer.outputs) == 1
            and str(producer.outputs[0]) == input_names[slot]
        ):
            transpose_matches.append((slot, int(producer_index), producer))
    if len(transpose_matches) != 1:
        return None
    transposed_slot, pre_index, pre = transpose_matches[0]
    transposed_input_name = input_names[transposed_slot]
    plain_input_name = input_names[1 - transposed_slot]
    if (
        transposed_input_name in public_names
        or graph_index.consumer_indices(transposed_input_name) != [binary_index]
    ):
        return None
    out_users = graph_index.consumer_indices(original_output)
    if len(out_users) != 1 or int(out_users[0]) <= binary_index:
        return None
    post_index = int(out_users[0])
    post = model_ir.operators[post_index]
    if (
        _normalized_op_type(post) != "TRANSPOSE"
        or len(post.inputs) != 2
        or len(post.outputs) != 1
        or str(post.inputs[0]) != original_output
    ):
        return None
    permutation_pre = _typed_permutation(
        model_ir,
        graph_index,
        pre,
        public_names=public_names,
    )
    permutation_post = _typed_permutation(
        model_ir,
        graph_index,
        post,
        public_names=public_names,
    )
    if (
        permutation_pre is None
        or permutation_post is None
        or not _inverse_permutations(permutation_pre, permutation_post)
    ):
        return None
    raw_input_name = str(pre.inputs[0])
    final_output = str(post.outputs[0])
    if (
        final_output in {str(value) for value in model_ir.inputs}
        or final_output in graph_index.duplicate_producers
        or graph_index.producers.get(final_output) != post_index
        or any(
            index <= post_index
            for index in set(graph_index.consumer_indices(final_output))
        )
        or not _resolved_source(
            model_ir,
            graph_index,
            name=raw_input_name,
            before_index=pre_index,
        )
        or not _resolved_source(
            model_ir,
            graph_index,
            name=plain_input_name,
            before_index=pre_index,
        )
    ):
        return None
    contract_names = (
        transposed_input_name,
        plain_input_name,
        raw_input_name,
        original_output,
        final_output,
        str(pre.inputs[1]),
        str(post.inputs[1]),
    )
    contracts = _tensor_contracts(model_ir, contract_names)
    if contracts is None:
        return None
    transposed = model_ir.tensors[transposed_input_name]
    plain = model_ir.tensors[plain_input_name]
    raw = model_ir.tensors[raw_input_name]
    output = model_ir.tensors[original_output]
    final = model_ir.tensors[final_output]
    data_tensors = [transposed, plain, raw, output, final]
    if (
        not _same_dtype(data_tensors)
        or not _all_per_tensor_quantized(data_tensors)
        or not _quantizations_equal(data_tensors)
        or len(plain.shape) != len(permutation_post)
        or not _shapes_match_if_known(list(transposed.shape), list(plain.shape))
    ):
        return None
    expected_shape = _permute_shape(list(plain.shape), list(permutation_post))
    expected_signature = _permute_shape(
        list(plain.shape_signature or plain.shape),
        list(permutation_post),
    )
    if (
        expected_shape is None
        or expected_signature is None
        or not _shapes_match_if_known(expected_shape, list(raw.shape))
        or not _shapes_match_if_known(expected_shape, list(final.shape))
        or not _signature_compatible(
            expected_signature,
            raw.shape_signature or raw.shape,
        )
        or not _signature_compatible(
            expected_signature,
            final.shape_signature or final.shape,
        )
    ):
        return None
    return _AsymmetricPlan(
        binary=binary,
        pre=pre,
        post=post,
        transpose_on_lhs=transposed_slot == 0,
        transposed_input_name=transposed_input_name,
        plain_input_name=plain_input_name,
        raw_input_name=raw_input_name,
        original_output=original_output,
        final_output=final_output,
        post_permutation_name=str(post.inputs[1]),
        expected_shape=tuple(int(value) for value in expected_shape),
        expected_signature=tuple(int(value) for value in expected_signature),
        expected_dtype=str(plain.dtype),
        expected_quantization=_clone_quantization(plain.quantization),
        expected_logical_layout=str(raw.logical_layout),
        expected_physical_layout=str(raw.physical_layout),
        contracts=contracts,
    )


def _symmetric_plans_equal(
    expected: _SymmetricPlan,
    actual: _SymmetricPlan,
) -> bool:
    return bool(
        expected.binary is actual.binary
        and expected.pre0 is actual.pre0
        and expected.pre1 is actual.pre1
        and expected.post is actual.post
        and expected.first_consumer is actual.first_consumer
        and expected.mode == actual.mode
        and expected.original_inputs == actual.original_inputs
        and expected.raw_inputs == actual.raw_inputs
        and expected.original_output == actual.original_output
        and expected.final_output == actual.final_output
        and expected.permutation == actual.permutation
        and expected.pre_permutation_name == actual.pre_permutation_name
        and expected.raw_output_name == actual.raw_output_name
        and expected.raw_output_shape == actual.raw_output_shape
        and expected.raw_output_signature == actual.raw_output_signature
        and expected.raw_logical_layout == actual.raw_logical_layout
        and expected.raw_physical_layout == actual.raw_physical_layout
        and expected.contracts == actual.contracts
    )


def _asymmetric_plans_equal(
    expected: _AsymmetricPlan,
    actual: _AsymmetricPlan,
) -> bool:
    return bool(
        expected.binary is actual.binary
        and expected.pre is actual.pre
        and expected.post is actual.post
        and expected.transpose_on_lhs == actual.transpose_on_lhs
        and expected.transposed_input_name == actual.transposed_input_name
        and expected.plain_input_name == actual.plain_input_name
        and expected.raw_input_name == actual.raw_input_name
        and expected.original_output == actual.original_output
        and expected.final_output == actual.final_output
        and expected.post_permutation_name == actual.post_permutation_name
        and expected.expected_shape == actual.expected_shape
        and expected.expected_signature == actual.expected_signature
        and expected.expected_dtype == actual.expected_dtype
        and _freeze_value(expected.expected_quantization)
        == _freeze_value(actual.expected_quantization)
        and expected.expected_logical_layout == actual.expected_logical_layout
        and expected.expected_physical_layout == actual.expected_physical_layout
        and expected.contracts == actual.contracts
    )


def _apply_symmetric(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _SymmetricPlan,
) -> bool:
    current = _resolve_symmetric(model_ir, graph_index, plan.binary)
    if current is None or not _symmetric_plans_equal(plan, current):
        return False
    remove_operators = [plan.pre0, plan.pre1]
    if plan.mode == "single_post":
        if plan.post is None:
            return False
        remove_operators.append(plan.post)
    remove_indices = [_operator_index(graph_index, op) for op in remove_operators]
    if any(index is None for index in remove_indices):
        return False
    insertion_index = None
    if plan.mode == "single_post_fanout" and plan.post is None:
        return False
    if plan.mode == "legacy_only":
        if (
            plan.raw_output_name is None
            or plan.raw_output_shape is None
            or plan.raw_output_signature is None
            or plan.first_consumer is None
            or plan.raw_output_name in model_ir.tensors
        ):
            return False
        insertion_index = _operator_index(graph_index, plan.first_consumer)
        binary_index = _operator_index(graph_index, plan.binary)
        if (
            insertion_index is None
            or binary_index is None
            or insertion_index <= binary_index
        ):
            return False

    if plan.mode == "legacy_only":
        old_output = model_ir.tensors[plan.original_output]
        model_ir.tensors[str(plan.raw_output_name)] = TensorIR(
            name=str(plan.raw_output_name),
            dtype=str(old_output.dtype),
            shape=[int(value) for value in plan.raw_output_shape or ()],
            shape_signature=[
                int(value) for value in plan.raw_output_signature or ()
            ],
            data=None,
            is_variable=False,
            quantization=_clone_quantization(old_output.quantization),
            logical_layout=str(plan.raw_logical_layout),
            physical_layout=str(plan.raw_physical_layout),
        )
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.binary,
        new_inputs=list(plan.raw_inputs),
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.binary,
        new_outputs=[
            str(plan.raw_output_name)
            if plan.mode == "legacy_only"
            else plan.final_output
        ],
        graph_index=graph_index,
    )
    if plan.mode == "single_post_fanout":
        assert plan.post is not None
        _set_operator_inputs(
            model_ir=model_ir,
            op=plan.post,
            new_inputs=[plan.final_output, plan.pre_permutation_name],
            graph_index=graph_index,
        )
        _set_operator_outputs(
            model_ir=model_ir,
            op=plan.post,
            new_outputs=[plan.original_output],
            graph_index=graph_index,
        )
    graph_index.remove_operators(int(index) for index in remove_indices if index is not None)
    if plan.mode == "legacy_only":
        current_consumer_index = _operator_index(graph_index, plan.first_consumer)
        current_binary_index = _operator_index(graph_index, plan.binary)
        if current_consumer_index is None or current_binary_index is None:
            raise RuntimeError("validated binary bridge operators disappeared during apply")
        graph_index.insert_operator(
            current_consumer_index,
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[str(plan.raw_output_name), plan.pre_permutation_name],
                outputs=[plan.original_output],
            ),
        )
    return True


def _apply_asymmetric(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _AsymmetricPlan,
) -> bool:
    current = _resolve_asymmetric(model_ir, graph_index, plan.binary)
    if current is None or not _asymmetric_plans_equal(plan, current):
        return False
    pre_index = _operator_index(graph_index, plan.pre)
    binary_index = _operator_index(graph_index, plan.binary)
    post_index = _operator_index(graph_index, plan.post)
    if (
        pre_index is None
        or binary_index is None
        or post_index is None
        or not (pre_index < binary_index < post_index)
    ):
        return False
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.pre,
        new_inputs=[plan.plain_input_name, plan.post_permutation_name],
        graph_index=graph_index,
    )
    rewritten_inputs = (
        [plan.raw_input_name, plan.transposed_input_name]
        if plan.transpose_on_lhs
        else [plan.transposed_input_name, plan.raw_input_name]
    )
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.binary,
        new_inputs=rewritten_inputs,
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.binary,
        new_outputs=[plan.final_output],
        graph_index=graph_index,
    )
    transposed = model_ir.tensors[plan.transposed_input_name]
    transposed.shape = [int(value) for value in plan.expected_shape]
    transposed.shape_signature = [
        int(value) for value in plan.expected_signature
    ]
    transposed.dtype = str(plan.expected_dtype)
    transposed.quantization = _clone_quantization(plan.expected_quantization)
    transposed.logical_layout = str(plan.expected_logical_layout)
    transposed.physical_layout = str(plan.expected_physical_layout)
    graph_index.remove_operator(post_index)
    return True


def optimize_transpose_binary_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Fold strict transpose/binary islands with bounded indexed rewrites."""

    rewrite_limit = max(0, int(max_rewrites))
    if rewrite_limit == 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {
            _SYMMETRIC_STATS_KEY: 0,
            _ASYMMETRIC_STATS_KEY: 0,
        }
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
            for index in active_index.operator_indices_for_normalized_types(
                _BINARY_OPS
            )
        ]
    )
    symmetric_rewrites = 0
    asymmetric_rewrites = 0
    for binary in candidates:
        if symmetric_rewrites + asymmetric_rewrites >= rewrite_limit:
            break
        if binary is None or active_index.operator_index(binary) is None:
            continue
        plan = _resolve_symmetric(model_ir, active_index, binary)
        if plan is not None and _apply_symmetric(model_ir, active_index, plan):
            symmetric_rewrites += 1
    for binary in candidates:
        if symmetric_rewrites + asymmetric_rewrites >= rewrite_limit:
            break
        if binary is None or active_index.operator_index(binary) is None:
            continue
        plan = _resolve_asymmetric(model_ir, active_index, binary)
        if plan is not None and _apply_asymmetric(model_ir, active_index, plan):
            asymmetric_rewrites += 1
    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if symmetric_rewrites + asymmetric_rewrites > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        _SYMMETRIC_STATS_KEY: int(symmetric_rewrites),
        _ASYMMETRIC_STATS_KEY: int(asymmetric_rewrites),
    }


_SAFE_LEGACY_PHASE = "symmetric_legacy_only"
_SAFE_SINGLE_POST_PHASE = "single_post"
_SAFE_MIXED_FANOUT_PHASE = "mixed_fanout"
_SAFE_ASYMMETRIC_FANOUT_PHASE = "asymmetric_fanout"
_SAFE_FULL_POST_FANOUT_PHASE = "full_post_fanout"
_SAFE_PHASES = (
    _SAFE_LEGACY_PHASE,
    _SAFE_SINGLE_POST_PHASE,
    _SAFE_MIXED_FANOUT_PHASE,
    _SAFE_ASYMMETRIC_FANOUT_PHASE,
    _SAFE_FULL_POST_FANOUT_PHASE,
)
_SAFE_STATS_KEYS = {
    _SAFE_LEGACY_PHASE:
        "rewritten_transpose_binary_symmetric_legacy_only_bridges_safe",
    _SAFE_SINGLE_POST_PHASE:
        "rewritten_transpose_binary_single_post_bridges_safe",
    _SAFE_MIXED_FANOUT_PHASE:
        "rewritten_transpose_binary_mixed_fanout_bridges_safe",
    _SAFE_ASYMMETRIC_FANOUT_PHASE:
        "rewritten_transpose_binary_asymmetric_fanout_bridges",
    _SAFE_FULL_POST_FANOUT_PHASE:
        "rewritten_transpose_binary_full_post_fanout_bridges",
}
_PRESERVE_LAYOUT_BOUNDARY_MARKER = "__preserve_layout_boundary__"


@dataclass(frozen=True)
class _MultiPostPlan:
    binary: OperatorIR
    pre0: OperatorIR
    pre1: OperatorIR
    posts: Tuple[OperatorIR, ...]
    legacy_users: Tuple[OperatorIR, ...]
    mode: str
    original_inputs: Tuple[str, str]
    raw_inputs: Tuple[str, str]
    original_output: str
    canonical_output: str
    post_output_names: Tuple[str, ...]
    pre_permutation_name: str
    expected_shape: Tuple[int, ...]
    expected_signature: Tuple[int, ...]
    expected_dtype: str
    expected_quantization: Any
    expected_logical_layout: str
    expected_physical_layout: str
    contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]


@dataclass(frozen=True)
class _AsymmetricFanoutPlan:
    binary: OperatorIR
    pre: OperatorIR
    post: OperatorIR
    plain_transpose: OperatorIR
    transpose_on_lhs: bool
    transposed_input_name: str
    plain_input_name: str
    plain_transpose_output: str
    raw_input_name: str
    original_output: str
    final_output: str
    pre_permutation_name: str
    adapter_needed: bool
    extra_users: Tuple[OperatorIR, ...]
    contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]


def _operator_contract(operator: OperatorIR) -> Tuple[Any, ...]:
    return (
        id(operator),
        str(operator.op_type),
        tuple(str(value) for value in operator.inputs),
        tuple(str(value) for value in operator.outputs),
        _freeze_value(operator.options),
        _freeze_value(operator.axis_semantics),
        int(operator.version),
        operator.onnx_node_name,
        operator.onnx_op_type,
    )


def _current_binary_candidates(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    candidate: Optional[OperatorIR],
) -> list[Optional[OperatorIR]]:
    if candidate is not None:
        return [candidate]
    return [
        model_ir.operators[index]
        for index in graph_index.operator_indices_for_normalized_types(
            _BINARY_OPS
        )
    ]


def _resolve_multi_post(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    binary: OperatorIR,
    *,
    mode: str,
) -> Optional[_MultiPostPlan]:
    if mode not in {_SAFE_MIXED_FANOUT_PHASE, _SAFE_FULL_POST_FANOUT_PHASE}:
        return None
    binary_index = graph_index.operator_index(binary)
    if binary_index is None or not _plain_binary(binary):
        return None
    original_inputs = tuple(str(value) for value in binary.inputs)
    original_output = str(binary.outputs[0])
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    public_names = public_inputs | public_outputs
    if (
        any(name in public_names for name in original_inputs)
        or original_output in public_names
        or original_output in graph_index.duplicate_producers
        or graph_index.producers.get(original_output) != binary_index
    ):
        return None
    pre_indices = tuple(
        _unique_producer(graph_index, name) for name in original_inputs
    )
    if any(index is None for index in pre_indices):
        return None
    pre0_index, pre1_index = int(pre_indices[0]), int(pre_indices[1])
    if pre0_index == pre1_index or not (
        pre0_index < binary_index and pre1_index < binary_index
    ):
        return None
    pre0 = model_ir.operators[pre0_index]
    pre1 = model_ir.operators[pre1_index]
    if (
        _normalized_op_type(pre0) != "TRANSPOSE"
        or _normalized_op_type(pre1) != "TRANSPOSE"
        or len(pre0.inputs) != 2
        or len(pre1.inputs) != 2
        or len(pre0.outputs) != 1
        or len(pre1.outputs) != 1
        or str(pre0.outputs[0]) != original_inputs[0]
        or str(pre1.outputs[0]) != original_inputs[1]
        or graph_index.consumer_indices(original_inputs[0]) != [binary_index]
        or graph_index.consumer_indices(original_inputs[1]) != [binary_index]
    ):
        return None
    permutation0 = _typed_permutation(
        model_ir,
        graph_index,
        pre0,
        public_names=public_names,
    )
    permutation1 = _typed_permutation(
        model_ir,
        graph_index,
        pre1,
        public_names=public_names,
    )
    if permutation0 is None or permutation0 != permutation1:
        return None
    raw_inputs = (str(pre0.inputs[0]), str(pre1.inputs[0]))
    if not _resolved_source(
        model_ir,
        graph_index,
        name=raw_inputs[0],
        before_index=pre0_index,
    ) or not _resolved_source(
        model_ir,
        graph_index,
        name=raw_inputs[1],
        before_index=pre1_index,
    ):
        return None

    output_consumer_indices = graph_index.consumer_indices(original_output)
    if (
        len(output_consumer_indices) < 2
        or any(index <= binary_index for index in output_consumer_indices)
    ):
        return None
    unique_output_consumers = sorted(set(output_consumer_indices))
    posts: list[OperatorIR] = []
    legacy_users: list[OperatorIR] = []
    for consumer_index in unique_output_consumers:
        consumer = model_ir.operators[consumer_index]
        if _normalized_op_type(consumer) != "TRANSPOSE":
            legacy_users.append(consumer)
            continue
        permutation = _typed_permutation(
            model_ir,
            graph_index,
            consumer,
            public_names=public_names,
        )
        if (
            len(consumer.outputs) != 1
            or str(consumer.inputs[0]) != original_output
            or permutation is None
            or not _inverse_permutations(permutation0, permutation)
        ):
            return None
        posts.append(consumer)
    if mode == _SAFE_MIXED_FANOUT_PHASE:
        if len(posts) == 0 or len(legacy_users) == 0:
            return None
    elif len(posts) < 2 or len(legacy_users) != 0:
        return None

    post_output_names = tuple(str(post.outputs[0]) for post in posts)
    if len(set(post_output_names)) != len(post_output_names):
        return None
    post_indices = []
    for post, output_name in zip(posts, post_output_names):
        post_index = graph_index.operator_index(post)
        if (
            post_index is None
            or post_index <= binary_index
            or output_name in public_names
            or output_name in graph_index.duplicate_producers
            or graph_index.producers.get(output_name) != post_index
            or any(
                consumer_index <= post_index
                for consumer_index in set(
                    graph_index.consumer_indices(output_name)
                )
            )
        ):
            return None
        post_indices.append(int(post_index))
    if mode == _SAFE_MIXED_FANOUT_PHASE:
        keep_post_index = post_indices[0]
        if any(
            int(graph_index.operator_index(user) or -1) <= keep_post_index
            for user in legacy_users
        ):
            return None

    contract_names = (
        raw_inputs[0],
        raw_inputs[1],
        original_inputs[0],
        original_inputs[1],
        original_output,
        str(pre0.inputs[1]),
        str(pre1.inputs[1]),
        *tuple(str(post.inputs[1]) for post in posts),
        *post_output_names,
    )
    contracts = _tensor_contracts(model_ir, contract_names)
    if contracts is None:
        return None
    raw0 = model_ir.tensors[raw_inputs[0]]
    raw1 = model_ir.tensors[raw_inputs[1]]
    transposed0 = model_ir.tensors[original_inputs[0]]
    transposed1 = model_ir.tensors[original_inputs[1]]
    output = model_ir.tensors[original_output]
    post_tensors = [model_ir.tensors[name] for name in post_output_names]
    data_tensors = [raw0, raw1, transposed0, transposed1, output, *post_tensors]
    if (
        not _same_dtype(data_tensors)
        or not _all_per_tensor_quantized(data_tensors)
        or not _quantizations_equal(data_tensors)
    ):
        return None
    rank = len(permutation0)
    if any(len(tensor.shape) != rank for tensor in data_tensors):
        return None
    expected_transposed0 = _permute_shape(list(raw0.shape), list(permutation0))
    expected_transposed1 = _permute_shape(list(raw1.shape), list(permutation0))
    if (
        expected_transposed0 is None
        or expected_transposed1 is None
        or not _shapes_match_if_known(
            expected_transposed0,
            list(transposed0.shape),
        )
        or not _shapes_match_if_known(
            expected_transposed1,
            list(transposed1.shape),
        )
    ):
        return None
    if mode == _SAFE_MIXED_FANOUT_PHASE and not _shapes_match_if_known(
        list(raw0.shape),
        list(raw1.shape),
    ):
        return None
    raw_broadcast = _broadcast_static_shapes(list(raw0.shape), list(raw1.shape))
    if raw_broadcast is None:
        return None
    expected_transposed = _permute_shape(raw_broadcast, list(permutation0))
    if (
        expected_transposed is None
        or not _shapes_match_if_known(
            expected_transposed,
            list(output.shape),
        )
    ):
        return None
    raw_signature = _broadcast_shape_signatures(
        list(raw0.shape_signature or raw0.shape),
        list(raw1.shape_signature or raw1.shape),
    )
    if raw_signature is None:
        return None
    expected_transposed_signature = _permute_shape(
        raw_signature,
        list(permutation0),
    )
    if (
        expected_transposed_signature is None
        or not _signature_compatible(
            expected_transposed_signature,
            output.shape_signature or output.shape,
        )
        or any(
            not _shapes_match_if_known(raw_broadcast, list(tensor.shape))
            or not _signature_compatible(
                raw_signature,
                tensor.shape_signature or tensor.shape,
            )
            for tensor in post_tensors
        )
    ):
        return None
    return _MultiPostPlan(
        binary=binary,
        pre0=pre0,
        pre1=pre1,
        posts=tuple(posts),
        legacy_users=tuple(legacy_users),
        mode=mode,
        original_inputs=original_inputs,
        raw_inputs=raw_inputs,
        original_output=original_output,
        canonical_output=post_output_names[0],
        post_output_names=post_output_names,
        pre_permutation_name=str(pre0.inputs[1]),
        expected_shape=tuple(int(value) for value in raw_broadcast),
        expected_signature=tuple(int(value) for value in raw_signature),
        expected_dtype=str(output.dtype),
        expected_quantization=_clone_quantization(output.quantization),
        expected_logical_layout=_layout_if_equal((raw0, raw1), "logical_layout"),
        expected_physical_layout=_layout_if_equal((raw0, raw1), "physical_layout"),
        contracts=contracts,
        operator_contracts=tuple(
            _operator_contract(operator)
            for operator in (binary, pre0, pre1, *posts, *legacy_users)
        ),
    )


def _multi_post_plans_equal(
    expected: _MultiPostPlan,
    actual: _MultiPostPlan,
) -> bool:
    return bool(
        expected.binary is actual.binary
        and expected.pre0 is actual.pre0
        and expected.pre1 is actual.pre1
        and len(expected.posts) == len(actual.posts)
        and all(
            left is right
            for left, right in zip(expected.posts, actual.posts)
        )
        and len(expected.legacy_users) == len(actual.legacy_users)
        and all(
            left is right
            for left, right in zip(
                expected.legacy_users,
                actual.legacy_users,
            )
        )
        and expected.mode == actual.mode
        and expected.original_inputs == actual.original_inputs
        and expected.raw_inputs == actual.raw_inputs
        and expected.original_output == actual.original_output
        and expected.canonical_output == actual.canonical_output
        and expected.post_output_names == actual.post_output_names
        and expected.pre_permutation_name == actual.pre_permutation_name
        and expected.expected_shape == actual.expected_shape
        and expected.expected_signature == actual.expected_signature
        and expected.expected_dtype == actual.expected_dtype
        and _freeze_value(expected.expected_quantization)
        == _freeze_value(actual.expected_quantization)
        and expected.expected_logical_layout == actual.expected_logical_layout
        and expected.expected_physical_layout == actual.expected_physical_layout
        and expected.contracts == actual.contracts
        and expected.operator_contracts == actual.operator_contracts
    )


def _apply_multi_post(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _MultiPostPlan,
) -> bool:
    current = _resolve_multi_post(
        model_ir,
        graph_index,
        plan.binary,
        mode=plan.mode,
    )
    if current is None or not _multi_post_plans_equal(plan, current):
        return False
    remove_operators = [plan.pre0, plan.pre1]
    if plan.mode == _SAFE_FULL_POST_FANOUT_PHASE:
        remove_operators.extend(plan.posts)
    else:
        remove_operators.extend(plan.posts[1:])
    remove_indices = [
        graph_index.operator_index(operator)
        for operator in remove_operators
    ]
    if any(index is None for index in remove_indices):
        return False

    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.binary,
        new_inputs=list(plan.raw_inputs),
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.binary,
        new_outputs=[plan.canonical_output],
        graph_index=graph_index,
    )
    canonical = model_ir.tensors[plan.canonical_output]
    canonical.shape = [int(value) for value in plan.expected_shape]
    canonical.shape_signature = [
        int(value) for value in plan.expected_signature
    ]
    canonical.dtype = str(plan.expected_dtype)
    canonical.quantization = _clone_quantization(plan.expected_quantization)
    canonical.logical_layout = str(plan.expected_logical_layout)
    canonical.physical_layout = str(plan.expected_physical_layout)
    for alias_name in plan.post_output_names[1:]:
        _replace_tensor_inputs(
            model_ir=model_ir,
            src_name=alias_name,
            dst_name=plan.canonical_output,
            graph_index=graph_index,
        )
    if plan.mode == _SAFE_MIXED_FANOUT_PHASE:
        keep_post = plan.posts[0]
        keep_options = dict(keep_post.options)
        keep_options[_PRESERVE_LAYOUT_BOUNDARY_MARKER] = True
        keep_post.options = keep_options
        _set_operator_inputs(
            model_ir=model_ir,
            op=keep_post,
            new_inputs=[plan.canonical_output, plan.pre_permutation_name],
            graph_index=graph_index,
        )
        _set_operator_outputs(
            model_ir=model_ir,
            op=keep_post,
            new_outputs=[plan.original_output],
            graph_index=graph_index,
        )
    graph_index.remove_operators(
        int(index) for index in remove_indices if index is not None
    )
    return True


def _resolve_asymmetric_fanout(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    binary: OperatorIR,
) -> Optional[_AsymmetricFanoutPlan]:
    binary_index = graph_index.operator_index(binary)
    if binary_index is None or not _plain_binary(binary):
        return None
    input_names = tuple(str(value) for value in binary.inputs)
    original_output = str(binary.outputs[0])
    public_inputs = {str(value) for value in model_ir.inputs}
    public_outputs = {str(value) for value in model_ir.outputs}
    public_names = public_inputs | public_outputs
    if (
        original_output in public_inputs
        or original_output in graph_index.duplicate_producers
        or graph_index.producers.get(original_output) != binary_index
    ):
        return None
    transpose_matches = []
    for slot, input_name in enumerate(input_names):
        producer_index = _unique_producer(graph_index, input_name)
        if producer_index is None or producer_index >= binary_index:
            continue
        producer = model_ir.operators[producer_index]
        if (
            _normalized_op_type(producer) == "TRANSPOSE"
            and len(producer.inputs) == 2
            and len(producer.outputs) == 1
            and str(producer.outputs[0]) == input_name
        ):
            transpose_matches.append((slot, producer_index, producer))
    if len(transpose_matches) != 1:
        return None
    transposed_slot, pre_index, pre = transpose_matches[0]
    transposed_input_name = input_names[transposed_slot]
    plain_input_name = input_names[1 - transposed_slot]
    if (
        transposed_input_name in public_names
        or graph_index.consumer_indices(transposed_input_name) != [binary_index]
    ):
        return None
    permutation_pre = _typed_permutation(
        model_ir,
        graph_index,
        pre,
        public_names=public_names,
    )
    inverse_permutation = (
        tuple(_invert_perm(list(permutation_pre)) or ())
        if permutation_pre is not None
        else ()
    )
    if permutation_pre is None or len(inverse_permutation) == 0:
        return None
    raw_input_name = str(pre.inputs[0])
    if not _resolved_source(
        model_ir,
        graph_index,
        name=raw_input_name,
        before_index=pre_index,
    ):
        return None

    output_consumers = sorted(set(graph_index.consumer_indices(original_output)))
    if len(output_consumers) == 0 or any(
        index <= binary_index for index in output_consumers
    ):
        return None
    post: Optional[OperatorIR] = None
    for consumer_index in output_consumers:
        consumer = model_ir.operators[consumer_index]
        permutation = _typed_permutation(
            model_ir,
            graph_index,
            consumer,
            public_names=public_names,
        )
        if (
            _normalized_op_type(consumer) == "TRANSPOSE"
            and len(consumer.outputs) == 1
            and str(consumer.inputs[0]) == original_output
            and permutation == inverse_permutation
        ):
            post = consumer
            break
    if post is None:
        return None
    post_index = graph_index.operator_index(post)
    if post_index is None or post_index <= binary_index:
        return None
    final_output = str(post.outputs[0])
    if (
        final_output in public_inputs
        or final_output in graph_index.duplicate_producers
        or graph_index.producers.get(final_output) != post_index
        or (
            final_output in public_outputs
            and original_output in public_outputs
        )
        or any(
            index <= post_index
            for index in set(graph_index.consumer_indices(final_output))
        )
    ):
        return None

    plain_transpose: Optional[OperatorIR] = None
    for consumer_index in sorted(
        set(graph_index.consumer_indices(plain_input_name))
    ):
        if consumer_index >= binary_index:
            continue
        consumer = model_ir.operators[consumer_index]
        permutation = _typed_permutation(
            model_ir,
            graph_index,
            consumer,
            public_names=public_names,
        )
        if (
            _normalized_op_type(consumer) == "TRANSPOSE"
            and len(consumer.outputs) == 1
            and str(consumer.inputs[0]) == plain_input_name
            and permutation == inverse_permutation
        ):
            plain_transpose = consumer
            break
    if plain_transpose is None:
        return None
    plain_transpose_index = graph_index.operator_index(plain_transpose)
    plain_transpose_output = str(plain_transpose.outputs[0])
    if (
        plain_transpose_index is None
        or plain_transpose_index >= binary_index
        or plain_transpose_output in graph_index.duplicate_producers
        or graph_index.producers.get(plain_transpose_output)
        != plain_transpose_index
        or not _resolved_source(
            model_ir,
            graph_index,
            name=plain_input_name,
            before_index=plain_transpose_index,
        )
    ):
        return None
    extra_indices = [
        index for index in output_consumers if index != post_index
    ]
    adapter_needed = bool(
        len(extra_indices) > 0 or original_output in public_outputs
    )
    extra_users = tuple(model_ir.operators[index] for index in extra_indices)

    contract_names = (
        transposed_input_name,
        plain_input_name,
        plain_transpose_output,
        raw_input_name,
        original_output,
        final_output,
        str(pre.inputs[1]),
        str(post.inputs[1]),
        str(plain_transpose.inputs[1]),
    )
    contracts = _tensor_contracts(model_ir, contract_names)
    if contracts is None:
        return None
    transposed = model_ir.tensors[transposed_input_name]
    plain = model_ir.tensors[plain_input_name]
    plain_transposed = model_ir.tensors[plain_transpose_output]
    raw = model_ir.tensors[raw_input_name]
    output = model_ir.tensors[original_output]
    final = model_ir.tensors[final_output]
    data_tensors = [transposed, plain, plain_transposed, raw, output, final]
    if (
        not _same_dtype(data_tensors)
        or not _all_per_tensor_quantized(data_tensors)
        or not _quantizations_equal(data_tensors)
        or len(plain.shape) != len(inverse_permutation)
        or not _shapes_match_if_known(
            list(transposed.shape),
            list(plain.shape),
        )
        or not _shapes_match_if_known(
            list(output.shape),
            list(transposed.shape),
        )
    ):
        return None
    expected_raw_shape = _permute_shape(
        list(plain.shape),
        list(inverse_permutation),
    )
    expected_raw_signature = _permute_shape(
        list(plain.shape_signature or plain.shape),
        list(inverse_permutation),
    )
    if (
        expected_raw_shape is None
        or expected_raw_signature is None
        or any(
            not _shapes_match_if_known(
                expected_raw_shape,
                list(tensor.shape),
            )
            or not _signature_compatible(
                expected_raw_signature,
                tensor.shape_signature or tensor.shape,
            )
            for tensor in (raw, plain_transposed, final)
        )
    ):
        return None
    return _AsymmetricFanoutPlan(
        binary=binary,
        pre=pre,
        post=post,
        plain_transpose=plain_transpose,
        transpose_on_lhs=transposed_slot == 0,
        transposed_input_name=transposed_input_name,
        plain_input_name=plain_input_name,
        plain_transpose_output=plain_transpose_output,
        raw_input_name=raw_input_name,
        original_output=original_output,
        final_output=final_output,
        pre_permutation_name=str(pre.inputs[1]),
        adapter_needed=adapter_needed,
        extra_users=extra_users,
        contracts=contracts,
        operator_contracts=tuple(
            _operator_contract(operator)
            for operator in (
                binary,
                pre,
                post,
                plain_transpose,
                *extra_users,
            )
        ),
    )


def _asymmetric_fanout_plans_equal(
    expected: _AsymmetricFanoutPlan,
    actual: _AsymmetricFanoutPlan,
) -> bool:
    return bool(
        expected.binary is actual.binary
        and expected.pre is actual.pre
        and expected.post is actual.post
        and expected.plain_transpose is actual.plain_transpose
        and expected.transpose_on_lhs == actual.transpose_on_lhs
        and expected.transposed_input_name == actual.transposed_input_name
        and expected.plain_input_name == actual.plain_input_name
        and expected.plain_transpose_output == actual.plain_transpose_output
        and expected.raw_input_name == actual.raw_input_name
        and expected.original_output == actual.original_output
        and expected.final_output == actual.final_output
        and expected.pre_permutation_name == actual.pre_permutation_name
        and expected.adapter_needed == actual.adapter_needed
        and len(expected.extra_users) == len(actual.extra_users)
        and all(
            left is right
            for left, right in zip(
                expected.extra_users,
                actual.extra_users,
            )
        )
        and expected.contracts == actual.contracts
        and expected.operator_contracts == actual.operator_contracts
    )


def _apply_asymmetric_fanout(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _AsymmetricFanoutPlan,
) -> bool:
    current = _resolve_asymmetric_fanout(
        model_ir,
        graph_index,
        plan.binary,
    )
    if current is None or not _asymmetric_fanout_plans_equal(plan, current):
        return False
    pre_index = graph_index.operator_index(plan.pre)
    post_index = graph_index.operator_index(plan.post)
    binary_index = graph_index.operator_index(plan.binary)
    if (
        pre_index is None
        or post_index is None
        or binary_index is None
        or not (pre_index < binary_index < post_index)
    ):
        return False
    rewritten_inputs = (
        [plan.raw_input_name, plan.plain_transpose_output]
        if plan.transpose_on_lhs
        else [plan.plain_transpose_output, plan.raw_input_name]
    )
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.binary,
        new_inputs=rewritten_inputs,
        graph_index=graph_index,
    )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.binary,
        new_outputs=[plan.final_output],
        graph_index=graph_index,
    )
    graph_index.remove_operators([pre_index, post_index])
    if plan.adapter_needed:
        current_binary_index = graph_index.operator_index(plan.binary)
        if current_binary_index is None:
            raise RuntimeError(
                "validated asymmetric binary bridge disappeared during apply"
            )
        graph_index.insert_operator(
            current_binary_index + 1,
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[plan.final_output, plan.pre_permutation_name],
                outputs=[plan.original_output],
            ),
        )
    return True


def _mark_preserved_layout_adapter(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    output_name: str,
) -> None:
    producer_index = graph_index.producers.get(str(output_name))
    if producer_index is None:
        return
    producer = model_ir.operators[int(producer_index)]
    if _normalized_op_type(producer) != "TRANSPOSE":
        return
    options = dict(producer.options)
    options[_PRESERVE_LAYOUT_BOUNDARY_MARKER] = True
    producer.options = options


def _run_safe_phase(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    phase: str,
    max_rewrites: int,
    candidate: Optional[OperatorIR],
) -> int:
    rewrite_limit = max(0, int(max_rewrites))
    if rewrite_limit == 0:
        return 0
    rewritten = 0
    for binary in _current_binary_candidates(model_ir, graph_index, candidate):
        if rewritten >= rewrite_limit:
            break
        if binary is None or graph_index.operator_index(binary) is None:
            continue
        if phase == _SAFE_LEGACY_PHASE:
            plan = _resolve_symmetric(model_ir, graph_index, binary)
            if (
                plan is not None
                and plan.mode == "legacy_only"
                and _apply_symmetric(model_ir, graph_index, plan)
            ):
                _mark_preserved_layout_adapter(
                    model_ir,
                    graph_index,
                    plan.original_output,
                )
                rewritten += 1
        elif phase == _SAFE_SINGLE_POST_PHASE:
            plan = _resolve_symmetric(model_ir, graph_index, binary)
            if (
                plan is not None
                and plan.mode in {"single_post", "single_post_fanout"}
                and _apply_symmetric(model_ir, graph_index, plan)
            ):
                if plan.mode == "single_post_fanout":
                    _mark_preserved_layout_adapter(
                        model_ir,
                        graph_index,
                        plan.original_output,
                    )
                rewritten += 1
        elif phase in {
            _SAFE_MIXED_FANOUT_PHASE,
            _SAFE_FULL_POST_FANOUT_PHASE,
        }:
            multi_plan = _resolve_multi_post(
                model_ir,
                graph_index,
                binary,
                mode=phase,
            )
            if multi_plan is not None and _apply_multi_post(
                model_ir,
                graph_index,
                multi_plan,
            ):
                rewritten += 1
        elif phase == _SAFE_ASYMMETRIC_FANOUT_PHASE:
            asymmetric_plan = _resolve_asymmetric_fanout(
                model_ir,
                graph_index,
                binary,
            )
            if asymmetric_plan is not None and _apply_asymmetric_fanout(
                model_ir,
                graph_index,
                asymmetric_plan,
            ):
                rewritten += 1
        else:
            raise ValueError(f"unknown safe binary bridge phase: {phase}")
    return int(rewritten)


def _run_safe_phases(
    model_ir: ModelIR,
    *,
    phases: Sequence[str],
    graph_index: Optional[ModelIRGraphIndex],
    layout_state: Optional[LayoutState],
    max_rewrites: int,
    candidate: Optional[OperatorIR],
) -> dict[str, int]:
    normalized_phases = tuple(str(phase) for phase in phases)
    if any(phase not in _SAFE_PHASES for phase in normalized_phases):
        raise ValueError(f"unknown safe binary bridge phases: {normalized_phases}")
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    stats = {
        _SAFE_STATS_KEYS[phase]: _run_safe_phase(
            model_ir,
            active_index,
            phase=phase,
            max_rewrites=max_rewrites,
            candidate=candidate,
        )
        for phase in normalized_phases
    }
    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if any(value > 0 for value in stats.values()) and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return stats


def run_safe_binary_bridge_recovery(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    """Run the five historical safe bridge modes on one maintained index."""

    return _run_safe_phases(
        model_ir,
        phases=_SAFE_PHASES,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def optimize_transpose_binary_symmetric_legacy_only_bridges_safe(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    return _run_safe_phases(
        model_ir,
        phases=(_SAFE_LEGACY_PHASE,),
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def optimize_transpose_binary_single_post_bridges_safe(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    return _run_safe_phases(
        model_ir,
        phases=(_SAFE_SINGLE_POST_PHASE,),
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def optimize_transpose_binary_mixed_fanout_bridges_safe(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    return _run_safe_phases(
        model_ir,
        phases=(_SAFE_MIXED_FANOUT_PHASE,),
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def optimize_transpose_binary_asymmetric_fanout_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    return _run_safe_phases(
        model_ir,
        phases=(_SAFE_ASYMMETRIC_FANOUT_PHASE,),
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def optimize_transpose_binary_full_post_fanout_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> dict[str, int]:
    return _run_safe_phases(
        model_ir,
        phases=(_SAFE_FULL_POST_FANOUT_PHASE,),
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )
