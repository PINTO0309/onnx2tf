from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_shape_signatures,
    _broadcast_static_shapes,
    _clone_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NHWC,
    ModelIR,
    OperatorIR,
    TensorIR,
)


_STATS_KEY = "rewritten_transposeconv_output_nhwc_passthrough_chains"
_PERM_NHWC_TO_NCHW = (0, 3, 1, 2)
_PERM_NCHW_TO_NHWC = (0, 2, 3, 1)
_PRODUCER_TYPES = frozenset(
    {"CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV"}
)
_UNARY_TYPES = frozenset(
    {
        "QUANTIZE",
        "DEQUANTIZE",
        "CAST",
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
    }
)
_BINARY_TYPES = frozenset(
    {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM"}
)


@dataclass(frozen=True)
class _View:
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _InputRewrite:
    operator: OperatorIR
    original_inputs: Tuple[str, ...]
    new_inputs: Tuple[str, ...]


@dataclass(frozen=True)
class _OutputRewrite:
    operator: OperatorIR
    original_outputs: Tuple[str, ...]
    new_outputs: Tuple[str, ...]


@dataclass(frozen=True)
class _MetadataUpdate:
    name: str
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]
    dtype: str
    quantization: Any
    logical_layout: str = LOGICAL_LAYOUT_NHWC
    physical_layout: str = LOGICAL_LAYOUT_NHWC


@dataclass(frozen=True)
class _ConstantUpdate:
    source_name: str
    target_name: str
    in_place: bool
    data: np.ndarray
    shape: Tuple[int, ...]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class _Plan:
    pre: OperatorIR
    source: OperatorIR
    chain: Tuple[OperatorIR, ...]
    post: OperatorIR
    input_rewrites: Tuple[_InputRewrite, ...]
    output_rewrite: _OutputRewrite
    metadata_updates: Tuple[_MetadataUpdate, ...]
    constant_updates: Tuple[_ConstantUpdate, ...]
    removals: Tuple[OperatorIR, ...]
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]


def _op_type(operator: OperatorIR) -> str:
    return str(operator.op_type).upper()


def _operator_index(
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
) -> Optional[int]:
    value = graph_index.operator_index(operator)
    return None if value is None else int(value)


def _signature(tensor: TensorIR) -> Tuple[int, ...]:
    values = tensor.shape if tensor.shape_signature is None else tensor.shape_signature
    return tuple(int(value) for value in values)


def _view(tensor: TensorIR) -> _View:
    return _View(
        shape=tuple(int(value) for value in tensor.shape),
        signature=_signature(tensor),
        dtype=str(tensor.dtype),
    )


def _rank4(view: _View) -> bool:
    return bool(len(view.shape) == 4 and len(view.signature) == 4)


def _positive(shape: Sequence[int]) -> bool:
    return bool(len(shape) > 0 and all(int(value) > 0 for value in shape))


def _compatible(expected: Sequence[int], actual: Sequence[int]) -> bool:
    return bool(
        len(expected) == len(actual)
        and all(
            int(left) == int(right) or int(left) < 0 or int(right) < 0
            for left, right in zip(expected, actual)
        )
    )


def _per_tensor_quantization(quantization: Any) -> bool:
    if quantization is None:
        return True
    scale = getattr(quantization, "scale", None)
    if scale is None and isinstance(quantization, dict):
        scale = quantization.get("scale")
    if scale is None:
        return True
    try:
        return int(np.asarray(scale).size) <= 1
    except Exception:
        return False


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (
            str(array.dtype),
            tuple(int(item) for item in array.shape),
            sha256(array.tobytes()).digest(),
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, "scale") and hasattr(value, "zero_point"):
        return (
            tuple(float(item) for item in value.scale),
            tuple(int(item) for item in value.zero_point),
            int(value.quantized_dimension),
            _freeze(value.min),
            _freeze(value.max),
        )
    return value


def _tensor_contract(name: str, tensor: TensorIR) -> Tuple[Any, ...]:
    return (
        str(name),
        id(tensor),
        str(tensor.name),
        str(tensor.dtype),
        tuple(int(value) for value in tensor.shape),
        _signature(tensor),
        _freeze(tensor.data),
        bool(tensor.is_variable),
        _freeze(tensor.quantization),
        str(tensor.logical_layout),
        str(tensor.physical_layout),
        tensor.onnx_tensor_name,
    )


def _operator_contract(operator: OperatorIR) -> Tuple[Any, ...]:
    return (
        id(operator),
        str(operator.op_type),
        tuple(str(value) for value in operator.inputs),
        tuple(str(value) for value in operator.outputs),
        _freeze(operator.options),
        _freeze(operator.axis_semantics),
        int(operator.version),
        operator.onnx_node_name,
        operator.onnx_op_type,
    )


def _typed_permutation(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    expected: Sequence[int],
) -> bool:
    if _op_type(operator) != "TRANSPOSE" or len(operator.inputs) != 2:
        return False
    if len(operator.outputs) != 1:
        return False
    name = str(operator.inputs[1])
    tensor = model_ir.tensors.get(name)
    if (
        tensor is None
        or tensor.data is None
        or str(tensor.dtype) not in {"INT32", "INT64"}
        or bool(tensor.is_variable)
        or tuple(int(value) for value in tensor.shape) != (4,)
        or _signature(tensor) != (4,)
        or not _per_tensor_quantization(tensor.quantization)
        or name in graph_index.producers
        or name in graph_index.duplicate_producers
        or name in {str(value) for value in model_ir.inputs}
        or name in {str(value) for value in model_ir.outputs}
    ):
        return False
    data = np.asarray(tensor.data)
    if data.dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
        return False
    if tuple(int(value) for value in data.shape) != (4,):
        return False
    return tuple(int(value) for value in data.reshape(-1)) == tuple(
        int(value) for value in expected
    )


def _consumer_slots(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
) -> Tuple[Tuple[OperatorIR, int], ...]:
    slots = []
    for operator_index in graph_index.consumer_indices(str(name)):
        operator = model_ir.operators[int(operator_index)]
        slots.extend(
            (operator, int(input_index))
            for input_index, input_name in enumerate(operator.inputs)
            if str(input_name) == str(name)
        )
    return tuple(slots)


def _unique_name(base: str, occupied: set[str]) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in occupied:
        candidate = f"{base}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _metadata_update(
    name: str,
    tensor: TensorIR,
    *,
    shape: Sequence[int],
    signature: Sequence[int],
    dtype: Optional[str] = None,
    quantization: Any = None,
    copy_quantization_from_argument: bool = False,
) -> _MetadataUpdate:
    return _MetadataUpdate(
        name=str(name),
        shape=tuple(int(value) for value in shape),
        signature=tuple(int(value) for value in signature),
        dtype=str(tensor.dtype if dtype is None else dtype),
        quantization=_clone_quantization(
            quantization if copy_quantization_from_argument else tensor.quantization
        ),
    )


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.pre),
        id(plan.source),
        tuple(id(operator) for operator in plan.chain),
        id(plan.post),
        tuple(
            (
                id(rewrite.operator),
                rewrite.original_inputs,
                rewrite.new_inputs,
            )
            for rewrite in plan.input_rewrites
        ),
        (
            id(plan.output_rewrite.operator),
            plan.output_rewrite.original_outputs,
            plan.output_rewrite.new_outputs,
        ),
        tuple(
            (
                update.name,
                update.shape,
                update.signature,
                update.dtype,
                _freeze(update.quantization),
                update.logical_layout,
                update.physical_layout,
            )
            for update in plan.metadata_updates
        ),
        tuple(
            (
                update.source_name,
                update.target_name,
                update.in_place,
                _freeze(update.data),
                update.shape,
                update.signature,
            )
            for update in plan.constant_updates
        ),
        tuple(id(operator) for operator in plan.removals),
        plan.tensor_contracts,
        plan.operator_contracts,
    )


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    pre: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    del layout_state
    pre_index = _operator_index(graph_index, pre)
    if pre_index is None or not _typed_permutation(
        model_ir,
        graph_index,
        pre,
        _PERM_NHWC_TO_NCHW,
    ):
        return None

    pre_input_name = str(pre.inputs[0])
    pre_output_name = str(pre.outputs[0])
    public_names = {str(value) for value in model_ir.inputs} | {
        str(value) for value in model_ir.outputs
    }
    if pre_input_name in public_names or pre_output_name in public_names:
        return None
    if (
        pre_input_name in graph_index.duplicate_producers
        or pre_output_name in graph_index.duplicate_producers
    ):
        return None
    source_index = graph_index.producers.get(pre_input_name)
    if source_index is None or int(source_index) >= int(pre_index):
        return None
    source = model_ir.operators[int(source_index)]
    if _op_type(source) not in _PRODUCER_TYPES:
        return None
    if pre_input_name not in {str(value) for value in source.outputs}:
        return None

    pre_input_tensor = model_ir.tensors.get(pre_input_name)
    pre_output_tensor = model_ir.tensors.get(pre_output_name)
    if pre_input_tensor is None or pre_output_tensor is None:
        return None
    pre_input_view = _view(pre_input_tensor)
    pre_output_view = _view(pre_output_tensor)
    if (
        not _rank4(pre_input_view)
        or not _rank4(pre_output_view)
        or not _positive(pre_input_view.shape)
        or not _positive(pre_output_view.shape)
        or pre_input_view.dtype != pre_output_view.dtype
        or not _per_tensor_quantization(pre_input_tensor.quantization)
        or not _per_tensor_quantization(pre_output_tensor.quantization)
        or tuple(_permute_shape(list(pre_input_view.shape), list(_PERM_NHWC_TO_NCHW)) or ())
        != pre_output_view.shape
        or tuple(
            _permute_shape(
                list(pre_input_view.signature),
                list(_PERM_NHWC_TO_NCHW),
            )
            or ()
        )
        != pre_output_view.signature
    ):
        return None

    chain = []
    binary_uses: Dict[str, list[Tuple[OperatorIR, int]]] = {}
    converted_views: Dict[str, _View] = {}
    current_name = pre_output_name
    current_old_view = pre_output_view
    current_new_view = pre_input_view
    previous_index = int(pre_index)

    for _ in range(len(model_ir.operators)):
        user_indices = graph_index.consumer_indices(current_name)
        if len(user_indices) != 1:
            break
        operator_index = int(user_indices[0])
        if operator_index <= previous_index:
            return None
        operator = model_ir.operators[operator_index]
        operator_type = _op_type(operator)
        if operator_type not in _UNARY_TYPES | _BINARY_TYPES:
            break
        if len(operator.outputs) != 1:
            return None
        output_name = str(operator.outputs[0])
        if output_name in public_names or output_name in graph_index.duplicate_producers:
            return None
        output_tensor = model_ir.tensors.get(output_name)
        if output_tensor is None or not _per_tensor_quantization(
            output_tensor.quantization
        ):
            return None
        output_old_view = _view(output_tensor)
        if not _rank4(output_old_view) or not _positive(output_old_view.shape):
            return None

        if operator_type in _UNARY_TYPES:
            if len(operator.inputs) != 1 or str(operator.inputs[0]) != current_name:
                return None
            if (
                output_old_view.shape != current_old_view.shape
                or output_old_view.signature != current_old_view.signature
            ):
                return None
            output_new_view = _View(
                shape=current_new_view.shape,
                signature=current_new_view.signature,
                dtype=output_old_view.dtype,
            )
        else:
            if len(operator.inputs) != 2:
                return None
            input_names = tuple(str(value) for value in operator.inputs)
            matching_slots = [
                index for index, name in enumerate(input_names) if name == current_name
            ]
            if len(matching_slots) != 1:
                return None
            main_input_index = int(matching_slots[0])
            side_input_index = 1 - main_input_index
            side_name = input_names[side_input_index]
            side_tensor = model_ir.tensors.get(side_name)
            if (
                side_tensor is None
                or side_tensor.data is None
                or bool(side_tensor.is_variable)
                or side_name in public_names
                or side_name in graph_index.producers
                or side_name in graph_index.duplicate_producers
                or not _per_tensor_quantization(side_tensor.quantization)
            ):
                return None
            side_data = np.asarray(side_tensor.data)
            side_view = _view(side_tensor)
            if (
                not _positive(side_view.shape)
                or tuple(int(value) for value in side_data.shape) != side_view.shape
                or not _compatible(side_view.shape, side_view.signature)
                or side_view.dtype != current_old_view.dtype
                or output_old_view.dtype != current_old_view.dtype
            ):
                return None
            old_shape = _broadcast_static_shapes(
                list(current_old_view.shape),
                list(side_view.shape),
            )
            old_signature = _broadcast_shape_signatures(
                list(current_old_view.signature),
                list(side_view.signature),
            )
            if (
                old_shape is None
                or old_signature is None
                or tuple(old_shape) != output_old_view.shape
                or tuple(old_signature) != output_old_view.signature
            ):
                return None

            if int(side_data.size) == 1:
                side_new_shape = side_view.shape
                side_new_signature = side_view.signature
            else:
                if not _rank4(side_view):
                    return None
                side_new_shape = tuple(
                    _permute_shape(list(side_view.shape), list(_PERM_NCHW_TO_NHWC))
                    or ()
                )
                side_new_signature = tuple(
                    _permute_shape(
                        list(side_view.signature),
                        list(_PERM_NCHW_TO_NHWC),
                    )
                    or ()
                )
                if not _rank4(
                    _View(side_new_shape, side_new_signature, side_view.dtype)
                ):
                    return None
                binary_uses.setdefault(side_name, []).append(
                    (operator, side_input_index)
                )
            new_shape = _broadcast_static_shapes(
                list(current_new_view.shape),
                list(side_new_shape),
            )
            new_signature = _broadcast_shape_signatures(
                list(current_new_view.signature),
                list(side_new_signature),
            )
            expected_new_shape = tuple(
                _permute_shape(
                    list(output_old_view.shape),
                    list(_PERM_NCHW_TO_NHWC),
                )
                or ()
            )
            expected_new_signature = tuple(
                _permute_shape(
                    list(output_old_view.signature),
                    list(_PERM_NCHW_TO_NHWC),
                )
                or ()
            )
            if (
                new_shape is None
                or new_signature is None
                or tuple(new_shape) != expected_new_shape
                or tuple(new_signature) != expected_new_signature
            ):
                return None
            output_new_view = _View(
                expected_new_shape,
                expected_new_signature,
                output_old_view.dtype,
            )

        chain.append(operator)
        converted_views[output_name] = output_new_view
        current_name = output_name
        current_old_view = output_old_view
        current_new_view = output_new_view
        previous_index = operator_index

    if len(chain) == 0:
        return None
    post_user_indices = graph_index.consumer_indices(current_name)
    if len(post_user_indices) != 1:
        return None
    post_index = int(post_user_indices[0])
    if post_index <= previous_index:
        return None
    post = model_ir.operators[post_index]
    if not _typed_permutation(
        model_ir,
        graph_index,
        post,
        _PERM_NCHW_TO_NHWC,
    ) or str(post.inputs[0]) != current_name:
        return None
    post_output_name = str(post.outputs[0])
    if post_output_name in public_names or post_output_name in graph_index.duplicate_producers:
        return None
    post_output_tensor = model_ir.tensors.get(post_output_name)
    if post_output_tensor is None:
        return None
    post_output_view = _view(post_output_tensor)
    if (
        not _rank4(post_output_view)
        or not _positive(post_output_view.shape)
        or post_output_view.shape != current_new_view.shape
        or post_output_view.signature != current_new_view.signature
    ):
        return None
    for consumer_index in graph_index.consumer_indices(post_output_name):
        if int(consumer_index) <= post_index:
            return None

    occupied = {str(name) for name in model_ir.tensors}
    constant_updates = []
    constant_targets: Dict[str, str] = {}
    for side_name in sorted(binary_uses):
        side_tensor = model_ir.tensors[side_name]
        side_data = np.asarray(side_tensor.data)
        side_view = _view(side_tensor)
        planned_slots = {
            (id(operator), int(input_index))
            for operator, input_index in binary_uses[side_name]
        }
        actual_slots = {
            (id(operator), int(input_index))
            for operator, input_index in _consumer_slots(
                model_ir,
                graph_index,
                side_name,
            )
        }
        if not planned_slots <= actual_slots:
            return None
        in_place = actual_slots == planned_slots
        target_name = side_name
        if not in_place:
            target_name = _unique_name(f"{side_name}_nhwc", occupied)
        constant_targets[side_name] = target_name
        constant_updates.append(
            _ConstantUpdate(
                source_name=side_name,
                target_name=target_name,
                in_place=in_place,
                data=np.transpose(side_data, _PERM_NCHW_TO_NHWC).astype(
                    side_data.dtype,
                    copy=False,
                ),
                shape=tuple(
                    _permute_shape(list(side_view.shape), list(_PERM_NCHW_TO_NHWC))
                    or ()
                ),
                signature=tuple(
                    _permute_shape(
                        list(side_view.signature),
                        list(_PERM_NCHW_TO_NHWC),
                    )
                    or ()
                ),
            )
        )

    rewritten_inputs: Dict[int, list[str]] = {
        id(operator): [str(value) for value in operator.inputs]
        for operator in chain
    }
    first = chain[0]
    first_inputs = rewritten_inputs[id(first)]
    first_main_slots = [
        index for index, name in enumerate(first_inputs) if name == pre_output_name
    ]
    if len(first_main_slots) != 1:
        return None
    first_inputs[int(first_main_slots[0])] = pre_input_name
    for side_name, uses in binary_uses.items():
        target_name = constant_targets[side_name]
        for operator, input_index in uses:
            rewritten_inputs[id(operator)][int(input_index)] = target_name
    input_rewrites = tuple(
        _InputRewrite(
            operator=operator,
            original_inputs=tuple(str(value) for value in operator.inputs),
            new_inputs=tuple(rewritten_inputs[id(operator)]),
        )
        for operator in chain
        if tuple(str(value) for value in operator.inputs)
        != tuple(rewritten_inputs[id(operator)])
    )

    last = chain[-1]
    old_last_name = str(last.outputs[0])
    old_last_tensor = model_ir.tensors.get(old_last_name)
    if old_last_tensor is None:
        return None
    output_rewrite = _OutputRewrite(
        operator=last,
        original_outputs=tuple(str(value) for value in last.outputs),
        new_outputs=(post_output_name,),
    )
    metadata_updates = []
    for operator in chain[:-1]:
        name = str(operator.outputs[0])
        tensor = model_ir.tensors[name]
        converted = converted_views[name]
        metadata_updates.append(
            _metadata_update(
                name,
                tensor,
                shape=converted.shape,
                signature=converted.signature,
            )
        )
    metadata_updates.append(
        _metadata_update(
            post_output_name,
            post_output_tensor,
            shape=current_new_view.shape,
            signature=current_new_view.signature,
            dtype=str(old_last_tensor.dtype),
            quantization=old_last_tensor.quantization,
            copy_quantization_from_argument=True,
        )
    )

    contract_names = {
        pre_input_name,
        pre_output_name,
        str(pre.inputs[1]),
        current_name,
        str(post.inputs[1]),
        post_output_name,
    }
    contract_names.update(
        str(value)
        for operator in chain
        for value in (*operator.inputs, *operator.outputs)
    )
    tensor_contracts = []
    for name in sorted(contract_names):
        tensor = model_ir.tensors.get(name)
        if tensor is None:
            return None
        tensor_contracts.append(_tensor_contract(name, tensor))
    ordered_operators = (source, pre, *chain, post)
    return _Plan(
        pre=pre,
        source=source,
        chain=tuple(chain),
        post=post,
        input_rewrites=input_rewrites,
        output_rewrite=output_rewrite,
        metadata_updates=tuple(metadata_updates),
        constant_updates=tuple(constant_updates),
        removals=(pre, post),
        tensor_contracts=tuple(tensor_contracts),
        operator_contracts=tuple(
            _operator_contract(operator) for operator in ordered_operators
        ),
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _Plan,
    *,
    layout_state: Optional[LayoutState],
) -> bool:
    current = _resolve_candidate(
        model_ir,
        graph_index,
        plan.pre,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if any(
        _operator_index(graph_index, operator) is None for operator in plan.removals
    ):
        return False
    for rewrite in plan.input_rewrites:
        if tuple(str(value) for value in rewrite.operator.inputs) != rewrite.original_inputs:
            return False
    if (
        tuple(str(value) for value in plan.output_rewrite.operator.outputs)
        != plan.output_rewrite.original_outputs
    ):
        return False
    for update in plan.metadata_updates:
        if update.name not in model_ir.tensors:
            return False
    for update in plan.constant_updates:
        if update.source_name not in model_ir.tensors:
            return False
        if not update.in_place and update.target_name in model_ir.tensors:
            return False

    for update in plan.constant_updates:
        source = model_ir.tensors[update.source_name]
        if update.in_place:
            target = source
        else:
            target = TensorIR(
                name=update.target_name,
                dtype=str(source.dtype),
                shape=[int(value) for value in update.shape],
                shape_signature=[int(value) for value in update.signature],
                data=np.asarray(update.data).copy(),
                is_variable=False,
                quantization=_clone_quantization(source.quantization),
                logical_layout=LOGICAL_LAYOUT_NHWC,
                physical_layout=LOGICAL_LAYOUT_NHWC,
                onnx_tensor_name=source.onnx_tensor_name,
            )
            model_ir.tensors[update.target_name] = target
        target.data = np.asarray(update.data).copy()
        target.shape = [int(value) for value in update.shape]
        target.shape_signature = [int(value) for value in update.signature]
        target.logical_layout = LOGICAL_LAYOUT_NHWC
        target.physical_layout = LOGICAL_LAYOUT_NHWC
        if layout_state is not None:
            layout_state.set(
                update.target_name,
                logical=LOGICAL_LAYOUT_NHWC,
                physical=LOGICAL_LAYOUT_NHWC,
            )

    for rewrite in plan.input_rewrites:
        _set_operator_inputs(
            model_ir=model_ir,
            op=rewrite.operator,
            new_inputs=list(rewrite.new_inputs),
            graph_index=graph_index,
        )
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.output_rewrite.operator,
        new_outputs=list(plan.output_rewrite.new_outputs),
        graph_index=graph_index,
    )
    for update in plan.metadata_updates:
        tensor = model_ir.tensors[update.name]
        tensor.shape = [int(value) for value in update.shape]
        tensor.shape_signature = [int(value) for value in update.signature]
        tensor.dtype = str(update.dtype)
        tensor.quantization = _clone_quantization(update.quantization)
        tensor.logical_layout = str(update.logical_layout)
        tensor.physical_layout = str(update.physical_layout)
        if layout_state is not None:
            layout_state.set(
                update.name,
                logical=update.logical_layout,
                physical=update.physical_layout,
            )
    removal_indices = [
        int(_operator_index(graph_index, operator)) for operator in plan.removals
    ]
    graph_index.remove_operators(removal_indices)
    return True


def optimize_transposeconv_output_nhwc_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Propagate a fully owned Conv-output chain through NHWC transactionally."""

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
                {"TRANSPOSE"}
            )
        ]
    )
    rewrite_limit = len(candidates) if max_rewrites is None else max(0, int(max_rewrites))
    if rewrite_limit == 0:
        return {_STATS_KEY: 0}
    rewritten = 0
    for pre in candidates:
        if rewritten >= rewrite_limit:
            break
        if pre is None or _operator_index(active_index, pre) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            pre,
            layout_state=layout_state,
        )
        if plan is None:
            continue
        if _apply_plan(
            model_ir,
            active_index,
            plan,
            layout_state=layout_state,
        ):
            rewritten += 1
    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {_STATS_KEY: int(rewritten)}
