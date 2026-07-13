from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
    channel_first_logical_layout,
    channel_last_logical_layout,
    is_channel_last_logical_layout,
    logical_layout_permutation,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _assign_tensor_logical_layout,
    _clone_tensor,
    _has_channel_last_factorized_rank3_sequence_consumer,
    _infer_concat_peer_layout,
    _is_channel_last_factorized_reshape,
    _is_channel_last_factorized_rank3_sequence_reshape,
    _perm_cf_to_cl,
    _perm_cl_to_cf,
    _permute_shape,
    _read_transpose_perm,
    _shared_tensor_layout,
)


_FEATURE_LAST_LAYOUT_PASSTHROUGH_OP_TYPES = {
    "ABS",
    "ADD",
    "AVERAGE_POOL_2D",
    "ATAN",
    "BATCH_MATMUL",
    "BROADCAST_TO",
    "CAST",
    "CEIL",
    "CONCATENATION",
    "COS",
    "DEPTH_TO_SPACE",
    "DIV",
    "ELU",
    "ERF",
    "EXP",
    "EXPAND_DIMS",
    "GATHER",
    "GATHER_ND",
    "GELU",
    "IDENTITY",
    "LEAKY_RELU",
    "LOG",
    "LOGISTIC",
    "MATMUL",
    "MAXIMUM",
    "MEAN",
    "MINIMUM",
    "MUL",
    "MAX_POOL_2D",
    "NEG",
    "PACK",
    "POW",
    "RELU",
    "RELU6",
    "RESHAPE",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",
    "SIGMOID",
    "SIGN",
    "SIN",
    "SLICE",
    "SOFTMAX",
    "SPACE_TO_DEPTH",
    "SPLIT",
    "SQRT",
    "SQUARE",
    "SQUEEZE",
    "STRIDED_SLICE",
    "SUB",
    "SUM",
    "TANH",
    "TILE",
    "TRANSPOSE",
    "UNPACK",
}

_CHANNEL_LAST_LAYOUT_FORWARD_OP_TYPES = {
    "ABS",
    "ADD",
    "ATAN",
    "AVERAGE_POOL_2D",
    "BATCH_MATMUL",
    "CAST",
    "CONCATENATION",
    "DEPTH_TO_SPACE",
    "DIV",
    "ELU",
    "ERF",
    "EXP",
    "EXPAND_DIMS",
    "GELU",
    "IDENTITY",
    "LOGISTIC",
    "MAXIMUM",
    "MAX_POOL_2D",
    "MEAN",
    "MINIMUM",
    "MUL",
    "NEG",
    "PACK",
    "RELU",
    "RELU6",
    "RESHAPE",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",
    "SIGMOID",
    "SIGN",
    "SIN",
    "SLICE",
    "SOFTMAX",
    "SPACE_TO_DEPTH",
    "SPLIT",
    "SQRT",
    "SQUARE",
    "SQUEEZE",
    "STRIDED_SLICE",
    "SUB",
    "SUM",
    "TANH",
    "TILE",
    "UNPACK",
}

_PYTORCH_FRIENDLY_LAYOUT_UNARY_OP_TYPES = {
    "ABS",
    "ATAN",
    "CEIL",
    "COS",
    "ELU",
    "EXP",
    "FLOOR",
    "HARD_SWISH",
    "IDENTITY",
    "LEAKY_RELU",
    "LOG",
    "LOGICAL_NOT",
    "LOGISTIC",
    "NEG",
    "RELU",
    "RELU6",
    "ROUND",
    "RSQRT",
    "SIGMOID",
    "SIGN",
    "SIN",
    "SQRT",
    "SQUARE",
    "TAN",
    "TANH",
}

_PYTORCH_FRIENDLY_LAYOUT_BINARY_OP_TYPES = {
    "ADD",
    "DIV",
    "MAXIMUM",
    "MINIMUM",
    "MUL",
    "POW",
    "SUB",
}

_PYTORCH_FRIENDLY_LAYOUT_RESIZE_POOL_OP_TYPES = {
    "AVERAGE_POOL_2D",
    "MAX_POOL_2D",
    "RESIZE_BILINEAR",
    "RESIZE_NEAREST_NEIGHBOR",
}

_PYTORCH_FRIENDLY_LAYOUT_OP_TYPES = (
    _PYTORCH_FRIENDLY_LAYOUT_UNARY_OP_TYPES
    | _PYTORCH_FRIENDLY_LAYOUT_BINARY_OP_TYPES
    | _PYTORCH_FRIENDLY_LAYOUT_RESIZE_POOL_OP_TYPES
    | {"CONCATENATION", "PACK", "SPLIT", "UNPACK"}
)


def _propagate_pytorch_friendly_layouts(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> None:
    """Propagate compatible layouts through only affected graph edges."""

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    pending = deque(
        graph_index.operator_indices_for_types(
            _PYTORCH_FRIENDLY_LAYOUT_OP_TYPES
        )
    )
    queued = set(pending)

    def _enqueue_adjacent_ops(tensor_names: Sequence[str]) -> None:
        adjacent_indices: Set[int] = set()
        for tensor_name in tensor_names:
            normalized_name = str(tensor_name)
            producer_indices = graph_index.duplicate_producers.get(
                normalized_name,
                (
                    [int(graph_index.producers[normalized_name])]
                    if normalized_name in graph_index.producers
                    else []
                ),
            )
            adjacent_indices.update(int(value) for value in producer_indices)
            adjacent_indices.update(
                int(value)
                for value in graph_index.consumer_indices(normalized_name)
            )
        for op_index in sorted(adjacent_indices):
            if op_index in queued:
                continue
            queued.add(op_index)
            pending.append(op_index)

    while pending:
        op_index = int(pending.popleft())
        queued.discard(op_index)
        op = model_ir.operators[op_index]
        op_type = str(op.op_type)
        changed_tensor_names: List[str] = []
        if op_type in _PYTORCH_FRIENDLY_LAYOUT_UNARY_OP_TYPES and len(op.inputs) >= 1:
            propagated_layout = _shared_tensor_layout(
                [model_ir.tensors.get(str(op.inputs[0]), None)]
            )
        elif op_type in _PYTORCH_FRIENDLY_LAYOUT_BINARY_OP_TYPES and len(op.inputs) >= 2:
            propagated_layout = _shared_tensor_layout(
                [
                    model_ir.tensors.get(str(op.inputs[0]), None),
                    model_ir.tensors.get(str(op.inputs[1]), None),
                ]
            )
        elif op_type == "CONCATENATION":
            concat_input_tensors = [
                model_ir.tensors.get(str(input_name), None)
                for input_name in op.inputs
            ]
            propagated_layout = _shared_tensor_layout(concat_input_tensors)
            if propagated_layout == LOGICAL_LAYOUT_UNKNOWN:
                propagated_layout = _infer_concat_peer_layout(
                    op,
                    concat_input_tensors,
                )
                if propagated_layout != LOGICAL_LAYOUT_UNKNOWN:
                    for input_name, input_tensor in zip(
                        op.inputs,
                        concat_input_tensors,
                    ):
                        if _assign_tensor_logical_layout(
                            input_tensor,
                            propagated_layout,
                        ):
                            changed_tensor_names.append(str(input_name))
        elif op_type in {"PACK", "UNPACK"}:
            propagated_layout = _shared_tensor_layout(
                [
                    model_ir.tensors.get(str(input_name), None)
                    for input_name in op.inputs
                ]
            )
        elif op_type == "SPLIT":
            propagated_layout = _shared_tensor_layout(
                [model_ir.tensors.get(str(op.inputs[-1]), None)]
            )
        elif op_type in _PYTORCH_FRIENDLY_LAYOUT_RESIZE_POOL_OP_TYPES:
            propagated_layout = _shared_tensor_layout(
                [model_ir.tensors.get(str(op.inputs[0]), None)]
            )
        else:
            continue
        if propagated_layout != LOGICAL_LAYOUT_UNKNOWN:
            for output_name in op.outputs:
                if _assign_tensor_logical_layout(
                    model_ir.tensors.get(str(output_name), None),
                    propagated_layout,
                ):
                    changed_tensor_names.append(str(output_name))
        if changed_tensor_names:
            _enqueue_adjacent_ops(changed_tensor_names)


def _propagate_channel_last_layouts(
    model_ir: ModelIR,
    *,
    consumers: Dict[str, List[int]],
) -> bool:
    worklist = sorted(
        str(name)
        for name, tensor in model_ir.tensors.items()
        if is_channel_last_logical_layout(
            normalize_logical_layout(tensor.logical_layout)
        )
    )
    queued = set(worklist)
    changed = False
    worklist_index = 0
    while worklist_index < len(worklist):
        tensor_name = str(worklist[worklist_index])
        worklist_index += 1
        for op_index in consumers.get(tensor_name, []):
            op = model_ir.operators[int(op_index)]
            if (
                str(op.op_type) not in _CHANNEL_LAST_LAYOUT_FORWARD_OP_TYPES
                or len(op.outputs) == 0
            ):
                continue
            if not any(
                input_tensor is not None
                and is_channel_last_logical_layout(
                    normalize_logical_layout(input_tensor.logical_layout)
                )
                for input_tensor in (
                    model_ir.tensors.get(str(input_name), None)
                    for input_name in op.inputs
                )
            ):
                continue
            for output_name in op.outputs:
                normalized_output_name = str(output_name)
                output_tensor = model_ir.tensors.get(normalized_output_name, None)
                if output_tensor is None:
                    continue
                rank = len(list(output_tensor.shape))
                if rank not in {3, 4, 5}:
                    continue
                target_layout = channel_last_logical_layout(rank)
                if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                    output_tensor.logical_layout = target_layout
                    changed = True
                if normalized_output_name not in queued:
                    queued.add(normalized_output_name)
                    worklist.append(normalized_output_name)
    return changed


def _propagate_feature_last_tensor_names(
    model_ir: ModelIR,
    root_names: Set[str],
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Set[str]:
    preserve_names = {str(name) for name in root_names}
    if len(preserve_names) == 0:
        return preserve_names
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    worklist = sorted(preserve_names)
    worklist_index = 0

    def _enqueue(names: list[str]) -> None:
        for name in names:
            normalized_name = str(name)
            if normalized_name in preserve_names:
                continue
            preserve_names.add(normalized_name)
            worklist.append(normalized_name)

    while worklist_index < len(worklist):
        tensor_name = str(worklist[worklist_index])
        worklist_index += 1
        related_indices = set(graph_index.consumer_indices(tensor_name))
        producer_index = graph_index.producers.get(tensor_name, None)
        if producer_index is not None:
            related_indices.add(int(producer_index))
        for op_index in sorted(related_indices):
            op = model_ir.operators[int(op_index)]
            op_type = str(op.op_type)
            if op_type not in _FEATURE_LAST_LAYOUT_PASSTHROUGH_OP_TYPES:
                continue
            input_names = [str(value) for value in op.inputs]
            output_names = [str(value) for value in op.outputs]
            if len(output_names) == 0:
                continue
            has_preserved_input = any(name in preserve_names for name in input_names)
            has_preserved_output = any(name in preserve_names for name in output_names)
            if not has_preserved_input and not has_preserved_output:
                continue
            if has_preserved_input:
                if op_type != "TRANSPOSE" or len(op.outputs) != 1:
                    _enqueue(output_names)
                else:
                    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
                    rank = len(list(output_tensor.shape)) if output_tensor is not None else -1
                    perm = _read_transpose_perm(model_ir, op)
                    if not (
                        rank in {3, 4, 5}
                        and (perm == _perm_cl_to_cf(rank) or perm == _perm_cf_to_cl(rank))
                    ):
                        _enqueue(output_names)
            if has_preserved_output:
                if (
                    op_type == "RESHAPE"
                    and len(op.inputs) >= 1
                    and len(op.outputs) == 1
                    and _is_channel_last_factorized_rank3_sequence_reshape(
                        model_ir.tensors.get(str(op.inputs[0]), None),
                        model_ir.tensors.get(str(op.outputs[0]), None),
                    )
                ):
                    continue
                if op_type != "TRANSPOSE" or len(op.inputs) < 1:
                    _enqueue(input_names)
                else:
                    input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                    rank = len(list(input_tensor.shape)) if input_tensor is not None else -1
                    perm = _read_transpose_perm(model_ir, op)
                    if not (
                        rank in {3, 4, 5}
                        and (perm == _perm_cl_to_cf(rank) or perm == _perm_cf_to_cl(rank))
                    ):
                        _enqueue(input_names)
    return preserve_names


def _is_attention_like_softmax_op(
    model_ir: ModelIR,
    op: OperatorIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> bool:
    if str(op.op_type) != "SOFTMAX":
        return False
    reference_tensor: Optional[TensorIR] = None
    if len(op.inputs) > 0:
        reference_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    if reference_tensor is None and len(op.outputs) > 0:
        reference_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    if reference_tensor is None:
        return False
    shape = [int(v) for v in list(reference_tensor.shape)]
    rank = len(shape)
    if rank < 3:
        return False
    axis = op.options.get("axis", None)
    resolved_axis = int(axis) if axis is not None else rank - 1
    if resolved_axis < 0:
        resolved_axis += rank
    if resolved_axis != rank - 1:
        return False
    if int(shape[-1]) <= 1:
        return False
    output_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
    if output_name != "":
        graph_index = graph_index or ModelIRGraphIndex(model_ir)
        if any(
            str(consumer.op_type) == "BATCH_MATMUL"
            for consumer in graph_index.consumers_of(output_name)
        ):
            return True
    if rank == 3 and int(shape[-2]) == int(shape[-1]):
        return True
    if rank >= 4 and int(shape[-2]) == int(shape[-1]) and 0 < int(shape[-3]) <= 64:
        return True
    return False


def _is_transpose_sandwiched_last_axis_softmax_op(
    model_ir: ModelIR,
    op: OperatorIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> bool:
    if str(op.op_type) != "SOFTMAX" or len(op.inputs) < 1 or len(op.outputs) != 1:
        return False
    input_name = str(op.inputs[0])
    output_name = str(op.outputs[0])
    input_tensor = model_ir.tensors.get(input_name, None)
    output_tensor = model_ir.tensors.get(output_name, None)
    if input_tensor is None or output_tensor is None:
        return False
    rank = len(list(input_tensor.shape))
    if rank not in {3, 4, 5} or len(list(output_tensor.shape)) != rank:
        return False
    axis = int(op.options.get("axis", rank - 1))
    if axis < 0:
        axis += rank
    if axis != rank - 1:
        return False

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    if input_name in graph_index.duplicate_producers:
        return False
    producer_op = graph_index.producer(input_name)
    if (
        producer_op is None
        or str(producer_op.op_type) != "TRANSPOSE"
        or len(producer_op.inputs) < 1
    ):
        return False
    producer_perm = _read_transpose_perm(model_ir, producer_op)
    if (
        producer_perm is None
        or len(producer_perm) != rank
        or sorted(int(v) for v in producer_perm) != list(range(rank))
        or [int(v) for v in producer_perm] == list(range(rank))
    ):
        return False

    consumer_ops = graph_index.consumers_of(output_name)
    if len(consumer_ops) != 1:
        return False
    consumer_op = consumer_ops[0]
    if str(consumer_op.op_type) != "TRANSPOSE" or len(consumer_op.outputs) != 1:
        return False
    consumer_perm = _read_transpose_perm(model_ir, consumer_op)
    if consumer_perm is None or len(consumer_perm) != rank:
        return False
    inverse_perm = [0] * rank
    for new_axis, old_axis in enumerate(producer_perm):
        inverse_perm[int(old_axis)] = int(new_axis)
    if [int(v) for v in consumer_perm] != inverse_perm:
        return False

    source_tensor = model_ir.tensors.get(str(producer_op.inputs[0]), None)
    restored_tensor = model_ir.tensors.get(str(consumer_op.outputs[0]), None)
    if source_tensor is None or restored_tensor is None:
        return False
    source_layout = normalize_logical_layout(source_tensor.logical_layout)
    restored_layout = normalize_logical_layout(restored_tensor.logical_layout)
    if (
        source_layout == LOGICAL_LAYOUT_UNKNOWN
        or restored_layout == LOGICAL_LAYOUT_UNKNOWN
        or source_layout != restored_layout
    ):
        return False
    return True


def _apply_feature_last_sequence_layouts(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
    consumers: Optional[Dict[str, List[int]]] = None,
) -> bool:
    if len(preserve_channel_last_tensor_names) == 0:
        return False
    any_changed = False
    if consumers is None:
        consumers = ModelIRGraphIndex(model_ir).consumers
    for tensor_name in preserve_channel_last_tensor_names:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        if is_channel_last_logical_layout(
            normalize_logical_layout(tensor.logical_layout)
        ):
            continue
        tensor.logical_layout = LOGICAL_LAYOUT_UNKNOWN
        any_changed = True

    for op in model_ir.operators:
        output_name = str(op.outputs[0]) if len(op.outputs) == 1 else None
        if output_name is None or output_name not in preserve_channel_last_tensor_names:
            continue
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None:
            continue
        rank = len(list(output_tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        op_type = str(op.op_type)
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None
        if (
            rank in {4, 5}
            and _has_channel_last_factorized_rank3_sequence_consumer(
                model_ir=model_ir,
                consumers=consumers,
                tensor_name=output_name,
            )
        ):
            target_layout = channel_last_logical_layout(rank)
            if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                output_tensor.logical_layout = target_layout
                any_changed = True
            continue
        if op_type == "TRANSPOSE":
            perm = _read_transpose_perm(model_ir, op)
            if output_name in preserve_channel_last_tensor_names and rank == 3 and perm == [1, 0, 2]:
                target_layout = channel_last_logical_layout(rank)
                if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                    output_tensor.logical_layout = target_layout
                    any_changed = True
                continue
            if perm == _perm_cf_to_cl(rank):
                input_layout = normalize_logical_layout(
                    input_tensor.logical_layout if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
                )
                if (
                    rank == 3
                    and output_name in set(str(v) for v in model_ir.outputs)
                    and output_name not in preserve_channel_last_tensor_names
                    and is_channel_last_logical_layout(input_layout)
                ):
                    target_layout = channel_first_logical_layout(rank)
                else:
                    target_layout = channel_last_logical_layout(rank)
                if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                    output_tensor.logical_layout = target_layout
                    any_changed = True
            elif rank in {4, 5}:
                target_layout = channel_last_logical_layout(rank)
                if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                    output_tensor.logical_layout = target_layout
                    any_changed = True
            elif (
                input_tensor is not None
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                and isinstance(perm, list)
                and len(perm) == rank
                and sorted(int(v) for v in perm) == list(range(rank))
                and int(perm[0]) == 0
                and int(perm[-1]) == rank - 1
            ):
                target_layout = channel_last_logical_layout(rank)
                if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                    output_tensor.logical_layout = target_layout
                    any_changed = True
            continue
        if op_type == "RESHAPE":
            should_mark_channel_last = False
            if output_name in preserve_channel_last_tensor_names and rank == 3:
                should_mark_channel_last = True
            if (
                not should_mark_channel_last
                and output_name in set(str(v) for v in model_ir.outputs)
                and input_tensor is not None
                and len(list(input_tensor.shape)) >= rank
                and len(list(output_tensor.shape)) >= 1
                and len(list(input_tensor.shape)) >= 1
                and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
            ):
                should_mark_channel_last = True
            if (
                not should_mark_channel_last
                and input_tensor is not None
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
            ):
                should_mark_channel_last = True
            raw_shape = op.options.get("onnxRawNewShape", None)
            if not should_mark_channel_last and isinstance(raw_shape, list):
                raw_shape_values = [int(v) for v in list(raw_shape)]
                if len(raw_shape_values) == rank:
                    current_shape = [int(v) for v in list(output_tensor.shape)]
                    if raw_shape_values != current_shape and raw_shape_values[-1] == current_shape[-1]:
                        should_mark_channel_last = True
            if not should_mark_channel_last and _is_channel_last_factorized_reshape(input_tensor, output_tensor):
                should_mark_channel_last = True
            if (
                not should_mark_channel_last
                and _is_channel_last_factorized_rank3_sequence_reshape(input_tensor, output_tensor)
            ):
                should_mark_channel_last = True
            if should_mark_channel_last:
                target_layout = channel_last_logical_layout(rank)
                if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                    output_tensor.logical_layout = target_layout
                    any_changed = True
            continue

    if _propagate_channel_last_layouts(
        model_ir,
        consumers=consumers,
    ):
        any_changed = True
    for op in model_ir.operators:
        if str(op.op_type) != "RESHAPE" or len(op.outputs) != 1:
            continue
        output_name = str(op.outputs[0])
        if output_name not in preserve_channel_last_tensor_names:
            continue
        if consumers.get(output_name):
            if any(
                str(model_ir.operators[int(consumer_idx)].op_type) == "TRANSPOSE"
                and _read_transpose_perm(model_ir, model_ir.operators[int(consumer_idx)])
                == _perm_cf_to_cl(len(list(model_ir.tensors[output_name].shape)))
                for consumer_idx in consumers.get(output_name, [])
            ):
                continue
        raw_shape = op.options.get("onnxRawNewShape", None)
        if not isinstance(raw_shape, list):
            continue
        raw_shape_values = [int(v) for v in list(raw_shape)]
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None or len(raw_shape_values) != len(list(output_tensor.shape)):
            continue
        output_tensor.shape = list(raw_shape_values)
        output_tensor.shape_signature = list(raw_shape_values)
        op.options["newShape"] = list(raw_shape_values)
        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if shape_tensor is not None and isinstance(shape_tensor.data, np.ndarray):
                dtype = np.asarray(shape_tensor.data).dtype
                shape_tensor.data = np.asarray(raw_shape_values, dtype=dtype)
                shape_tensor.shape = [int(len(raw_shape_values))]
                shape_tensor.shape_signature = [int(len(raw_shape_values))]
        any_changed = True
    return any_changed


def _collect_feature_last_sequence_tensor_names(model_ir: ModelIR) -> Set[str]:
    graph_index = ModelIRGraphIndex(model_ir)
    consumers = graph_index.consumers
    producers = graph_index.producers

    def _is_time_major_recurrent_bridge(output_name: str) -> bool:
        for consumer_idx in consumers.get(str(output_name), []):
            consumer = model_ir.operators[int(consumer_idx)]
            if str(consumer.op_type) != "TRANSPOSE" or len(consumer.outputs) != 1:
                continue
            perm = _read_transpose_perm(model_ir, consumer)
            if perm != [1, 0, 2]:
                continue
            transpose_output_name = str(consumer.outputs[0])
            for next_idx in consumers.get(transpose_output_name, []):
                next_op_type = str(model_ir.operators[int(next_idx)].op_type)
                if next_op_type in {
                    "BIDIRECTIONAL_SEQUENCE_LSTM",
                    "UNIDIRECTIONAL_SEQUENCE_LSTM",
                    "UNIDIRECTIONAL_SEQUENCE_RNN",
                }:
                    return True
        return False

    def _trace_feature_last_rhs_seed(tensor_name: str) -> Optional[str]:
        visited: Set[str] = set()
        worklist: List[str] = [str(tensor_name)]
        passthrough_ops = {
            "CAST",
            "EXPAND_DIMS",
            "GATHER",
            "GATHER_ND",
            "IDENTITY",
            "RESHAPE",
            "SLICE",
            "SQUEEZE",
            "STRIDED_SLICE",
            "TRANSPOSE",
        }
        while len(worklist) > 0:
            current_name = str(worklist.pop())
            if current_name in visited:
                continue
            visited.add(current_name)
            current_tensor = model_ir.tensors.get(current_name, None)
            if current_tensor is not None:
                current_rank = len(list(current_tensor.shape))
                current_layout = normalize_logical_layout(current_tensor.logical_layout)
                if current_rank in {3, 4, 5} and is_channel_last_logical_layout(current_layout):
                    return current_name
            producer_idx = producers.get(current_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) not in passthrough_ops or len(producer.inputs) == 0:
                continue
            worklist.append(str(producer.inputs[0]))
        return None

    def _trace_feature_last_passthrough_inputs(tensor_name: str) -> Set[str]:
        traced_names: Set[str] = set()
        visited: Set[str] = set()
        worklist: List[str] = [str(tensor_name)]
        passthrough_ops = {
            "AVERAGE_POOL_2D",
            "CAST",
            "EXPAND_DIMS",
            "IDENTITY",
            "LEAKY_RELU",
            "LOGISTIC",
            "MAX_POOL_2D",
            "PAD",
            "PADV2",
            "RELU",
            "RELU6",
            "RESHAPE",
            "SQUEEZE",
            "STRIDED_SLICE",
            "TRANSPOSE",
        }
        while len(worklist) > 0:
            current_name = str(worklist.pop())
            if current_name in visited:
                continue
            visited.add(current_name)
            traced_names.add(current_name)
            producer_idx = producers.get(current_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) not in passthrough_ops or len(producer.inputs) == 0:
                continue
            upstream_name = str(producer.inputs[0])
            traced_names.add(upstream_name)
            worklist.append(upstream_name)
        return traced_names

    roots: Set[str] = set()
    if _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(model_ir):
        for tensor_name, tensor in model_ir.tensors.items():
            rank = len(list(tensor.shape))
            layout = normalize_logical_layout(tensor.logical_layout)
            if rank in {4, 5} and is_channel_last_logical_layout(layout):
                roots.add(str(tensor_name))
    for tensor_name, tensor in model_ir.tensors.items():
        normalized_name = str(tensor_name)
        rank = len(list(tensor.shape))
        layout = normalize_logical_layout(tensor.logical_layout)
        lowered_name = normalized_name.lower()
        if (
            rank in {3, 4, 5}
            and is_channel_last_logical_layout(layout)
            and any(token in lowered_name for token in ("_nwc", "_nhwc", "_ndhwc"))
        ):
            roots.add(normalized_name)
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type == "BATCH_MATMUL" and len(op.inputs) >= 2:
            rhs_seed = _trace_feature_last_rhs_seed(str(op.inputs[1]))
            if rhs_seed is not None:
                roots.add(rhs_seed)
        if op_type == "TRANSPOSE" and len(op.inputs) >= 1 and len(op.outputs) == 1:
            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            output_tensor = model_ir.tensors.get(output_name, None)
            input_tensor = model_ir.tensors.get(input_name, None)
            if output_tensor is None:
                continue
            rank = len(list(output_tensor.shape))
            if rank != 3:
                continue
            perm = _read_transpose_perm(model_ir, op)
            if (
                perm == [1, 0, 2]
                and input_tensor is not None
                and (
                    is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                    or is_channel_last_logical_layout(normalize_logical_layout(output_tensor.logical_layout))
                )
            ):
                roots.add(input_name)
                roots.add(output_name)
                continue
            if perm != _perm_cf_to_cl(rank):
                continue
            producer_idx = producers.get(input_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) != "RESHAPE" or len(producer.outputs) != 1:
                continue
            roots.add(output_name)
            continue
        if op_type != "RESHAPE" or len(op.outputs) != 1:
            continue
        output_name = str(op.outputs[0])
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None:
            continue
        rank = len(list(output_tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None
        if (
            output_name in set(str(v) for v in model_ir.outputs)
            and input_tensor is not None
            and len(list(input_tensor.shape)) >= rank
            and len(list(output_tensor.shape)) >= 1
            and len(list(input_tensor.shape)) >= 1
            and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
        ):
            roots.add(output_name)
            continue
        if (
            input_tensor is not None
            and len(list(input_tensor.shape)) >= rank
            and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
            and len(list(output_tensor.shape)) >= 1
            and len(list(input_tensor.shape)) >= 1
            and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
        ):
            roots.add(output_name)
            continue
        if _is_channel_last_factorized_reshape(input_tensor, output_tensor):
            roots.add(output_name)
            continue
        if _is_channel_last_factorized_rank3_sequence_reshape(input_tensor, output_tensor):
            roots.add(output_name)
            continue
        if input_tensor is not None and rank == 3 and len(list(input_tensor.shape)) == 3:
            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if (
                int(np.prod(input_shape, dtype=np.int64)) == int(np.prod(output_shape, dtype=np.int64))
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
            ):
                for consumer_idx in consumers.get(output_name, []):
                    consumer = model_ir.operators[int(consumer_idx)]
                    if (
                        str(consumer.op_type) != "BATCH_MATMUL"
                        or len(consumer.inputs) < 2
                        or str(consumer.inputs[0]) != output_name
                        or not bool(consumer.options.get("adjX", False))
                    ):
                        continue
                    rhs_tensor = model_ir.tensors.get(str(consumer.inputs[1]), None)
                    if rhs_tensor is None or len(list(rhs_tensor.shape)) < 2:
                        continue
                    rhs_contract = int(list(rhs_tensor.shape)[-2])
                    if rhs_contract != int(input_shape[-1]):
                        continue
                    roots.add(output_name)
                    break
                if output_name in roots:
                    continue
        if input_tensor is not None and rank == 3 and len(list(input_tensor.shape)) in {4, 5}:
            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if (
                int(np.prod(input_shape, dtype=np.int64)) == int(np.prod(output_shape, dtype=np.int64))
                and _is_time_major_recurrent_bridge(output_name)
            ):
                roots.add(output_name)
                input_name = str(op.inputs[0])
                roots.add(input_name)
                producer: Optional[OperatorIR] = None
                producer_output_name = ""
                producer_rank = -1
                producer_idx = producers.get(input_name, None)
                if producer_idx is not None:
                    producer = model_ir.operators[int(producer_idx)]
                    producer_output_name = str(producer.outputs[0]) if len(producer.outputs) == 1 else ""
                    producer_output_tensor = (
                        model_ir.tensors.get(producer_output_name, None)
                        if producer_output_name != ""
                        else None
                    )
                    producer_rank = (
                        len(list(producer_output_tensor.shape))
                        if producer_output_tensor is not None
                        else -1
                    )
                if (
                    producer is not None
                    and str(producer.op_type) == "TRANSPOSE"
                    and producer_rank in {4, 5}
                ):
                    if len(producer.inputs) >= 1:
                        producer_input_name = str(producer.inputs[0])
                        roots.update(_trace_feature_last_passthrough_inputs(producer_input_name))
                    roots.add(producer_output_name)
                continue
            if (
                int(np.prod(input_shape[1:], dtype=np.int64))
                == int(np.prod(output_shape[1:], dtype=np.int64))
                and (
                    is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                    or int(input_shape[-1]) == 1
                )
            ):
                for consumer_idx in consumers.get(output_name, []):
                    consumer = model_ir.operators[int(consumer_idx)]
                    if (
                        str(consumer.op_type) != "BATCH_MATMUL"
                        or len(consumer.inputs) < 2
                        or str(consumer.inputs[1]) != output_name
                    ):
                        continue
                    lhs_tensor = model_ir.tensors.get(str(consumer.inputs[0]), None)
                    if lhs_tensor is None or len(list(lhs_tensor.shape)) < 2:
                        continue
                    lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
                    if int(lhs_shape[-1]) != int(output_shape[-2]):
                        continue
                    roots.add(output_name)
                    break
                if output_name in roots:
                    continue
        raw_shape = op.options.get("onnxRawNewShape", None)
        new_shape = op.options.get("newShape", None)
        if not isinstance(raw_shape, list) or not isinstance(new_shape, list):
            continue
        raw_shape_values = [int(v) for v in list(raw_shape)]
        new_shape_values = [int(v) for v in list(new_shape)]
        if raw_shape_values == new_shape_values:
            continue
        if len(raw_shape_values) != rank or len(new_shape_values) != rank:
            continue
        if consumers.get(output_name):
            if any(
                str(model_ir.operators[int(consumer_idx)].op_type) == "TRANSPOSE"
                and _read_transpose_perm(model_ir, model_ir.operators[int(consumer_idx)]) == _perm_cf_to_cl(rank)
                for consumer_idx in consumers.get(output_name, [])
            ):
                continue
        roots.add(output_name)

    return _propagate_feature_last_tensor_names(
        model_ir,
        roots,
        graph_index=graph_index,
    )


def _is_rank4_channel_last_dynamic_tensor(tensor: Optional[TensorIR]) -> bool:
    if tensor is None or isinstance(tensor.data, np.ndarray):
        return False
    return (
        len(list(tensor.shape)) == 4
        and is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout))
    )


def _is_pytorch_channel_first_safe_rank4_island_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    passthrough_op_types = {
        "ADD",
        "AVERAGE_POOL_2D",
        "CONCATENATION",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "DIV",
        "LEAKY_RELU",
        "LOGISTIC",
        "MAX_POOL_2D",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "RELU",
        "RELU6",
        "SUB",
        "TANH",
    }
    if op_type in passthrough_op_types:
        relevant_dynamic_tensors = [
            tensor
            for tensor_name in list(op.inputs) + list(op.outputs)
            for tensor in [model_ir.tensors.get(str(tensor_name), None)]
            if _is_rank4_channel_last_dynamic_tensor(tensor)
        ]
        return len(relevant_dynamic_tensors) > 0
    return False


def _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(
    model_ir: ModelIR,
) -> bool:
    public_boundary_names = [str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)]
    if len(public_boundary_names) == 0:
        return False
    for tensor_name in public_boundary_names:
        tensor = model_ir.tensors.get(tensor_name, None)
        if tensor is None:
            return False
        rank = len(list(tensor.shape))
        if rank not in {4, 5}:
            return False
        if not is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout)):
            return False
    if any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators):
        return False
    for op in model_ir.operators:
        if not _is_pytorch_channel_first_safe_rank4_island_op(model_ir, op):
            return False
    return True


def _shrink_preserved_channel_last_regions_for_pytorch(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
    producer_index: Optional[Dict[str, int]] = None,
    consumers: Optional[Dict[str, List[int]]] = None,
) -> Set[str]:
    if len(preserve_channel_last_tensor_names) == 0:
        return set()
    if _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(model_ir):
        return {str(name) for name in preserve_channel_last_tensor_names}
    if producer_index is None or consumers is None:
        graph_index = ModelIRGraphIndex(model_ir)
        producer_index = graph_index.producers
        consumers = graph_index.consumers

    public_boundary_names = {str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)}
    shrunken_preserve_names: Set[str] = {
        str(name) for name in preserve_channel_last_tensor_names
    }
    for tensor_name in sorted(str(name) for name in preserve_channel_last_tensor_names):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if not _is_rank4_channel_last_dynamic_tensor(tensor):
            continue
        if str(tensor_name) in public_boundary_names:
            continue
        producer_idx = producer_index.get(str(tensor_name), None)
        if producer_idx is None:
            continue
        producer_op = model_ir.operators[int(producer_idx)]
        if not _is_pytorch_channel_first_safe_rank4_island_op(model_ir, producer_op):
            continue
        consumer_indices = consumers.get(str(tensor_name), [])
        if len(consumer_indices) == 0:
            continue
        if any(
            str(model_ir.operators[int(consumer_idx)].op_type) == "DEPTHWISE_CONV_2D"
            for consumer_idx in consumer_indices
        ):
            continue
        if any(
            not _is_pytorch_channel_first_safe_rank4_island_op(
                model_ir,
                model_ir.operators[int(consumer_idx)],
            )
            for consumer_idx in consumer_indices
        ):
            continue
        shrunken_preserve_names.discard(str(tensor_name))
    return shrunken_preserve_names


def _restore_non_preserved_channel_first_layouts(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    public_layout_bridge_tensor_names = {
        str(name)
        for name in list(model_ir.metadata.get("public_layout_bridge_tensor_names", []))
    }
    for tensor_name, tensor in model_ir.tensors.items():
        if str(tensor_name) in preserve_channel_last_tensor_names:
            continue
        if str(tensor_name) in public_layout_bridge_tensor_names:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        layout = normalize_logical_layout(tensor.logical_layout)
        if is_channel_last_logical_layout(layout):
            tensor.logical_layout = channel_first_logical_layout(rank)


def _ensure_public_boundary_layout_bridges(
    *,
    model_ir: ModelIR,
    desired_public_shape_map: Dict[str, List[int]],
    desired_public_layout_map: Dict[str, str],
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> None:
    used_tensor_names: Set[str] = set(model_ir.tensors.keys())
    bridge_tensor_names = model_ir.metadata.get("public_layout_bridge_tensor_names", [])
    if not isinstance(bridge_tensor_names, list):
        bridge_tensor_names = []
    model_ir.metadata["public_layout_bridge_tensor_names"] = bridge_tensor_names

    def _make_unique_identifier(base_name: str) -> str:
        candidate = str(base_name)
        suffix = 1
        while candidate in used_tensor_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_tensor_names.add(candidate)
        return candidate

    def _shared_graph_index() -> ModelIRGraphIndex:
        nonlocal graph_index
        if graph_index is None:
            graph_index = ModelIRGraphIndex(model_ir)
        return graph_index

    def _insert_public_boundary_layout_bridge(
        *,
        tensor_name: str,
        current_tensor: TensorIR,
        desired_shape: Sequence[int],
        desired_layout: str,
        is_input: bool,
    ) -> None:
        current_shape = [
            int(value)
            for value in list(current_tensor.shape_signature or current_tensor.shape)
        ]
        target_shape = [int(value) for value in list(desired_shape)]
        current_layout = normalize_logical_layout(current_tensor.logical_layout)
        normalized_target_layout = normalize_logical_layout(desired_layout)
        if (
            len(current_shape) not in {3, 4, 5}
            or len(current_shape) != len(target_shape)
            or current_layout == LOGICAL_LAYOUT_UNKNOWN
            or normalized_target_layout == LOGICAL_LAYOUT_UNKNOWN
            or current_layout == normalized_target_layout
        ):
            return
        perm = logical_layout_permutation(
            source_layout=normalized_target_layout if is_input else current_layout,
            target_layout=current_layout if is_input else normalized_target_layout,
        )
        expected_shape = current_shape if is_input else target_shape
        seed_shape = target_shape if is_input else current_shape
        if perm is None or _permute_shape(seed_shape, perm) != expected_shape:
            return

        bridge_tensor_name = _make_unique_identifier(
            f"{tensor_name}_public_layout_bridge"
        )
        bridge_tensor = _clone_tensor(current_tensor)
        bridge_tensor.name = str(bridge_tensor_name)
        model_ir.tensors[str(bridge_tensor_name)] = bridge_tensor
        if str(bridge_tensor_name) not in bridge_tensor_names:
            bridge_tensor_names.append(str(bridge_tensor_name))
        perm_name = _make_unique_identifier(f"{bridge_tensor_name}_perm")
        perm_arr = np.asarray([int(value) for value in list(perm)], dtype=np.int32)
        model_ir.tensors[str(perm_name)] = TensorIR(
            name=str(perm_name),
            dtype="INT32",
            shape=[int(perm_arr.size)],
            shape_signature=[int(perm_arr.size)],
            data=perm_arr,
        )

        active_index = _shared_graph_index()
        if is_input:
            for consumer in active_index.consumers_of(tensor_name):
                consumer_index = active_index.operator_index(consumer)
                if consumer_index is None:
                    continue
                active_index.replace_operator_inputs(
                    int(consumer_index),
                    [
                        str(bridge_tensor_name)
                        if str(name) == str(tensor_name)
                        else str(name)
                        for name in consumer.inputs
                    ],
                )
            active_index.insert_operator(
                0,
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[str(tensor_name), str(perm_name)],
                    outputs=[str(bridge_tensor_name)],
                    options={"perm": [int(value) for value in list(perm)]},
                ),
            )
            return

        producer_indices = active_index.duplicate_producers.get(
            str(tensor_name),
            (
                [int(active_index.producers[str(tensor_name)])]
                if str(tensor_name) in active_index.producers
                else []
            ),
        )
        producer_ops = [
            model_ir.operators[int(producer_index)]
            for producer_index in producer_indices
        ]
        consumer_ops = active_index.consumers_of(tensor_name)
        for producer in producer_ops:
            producer_index = active_index.operator_index(producer)
            if producer_index is None:
                continue
            active_index.replace_operator_outputs(
                int(producer_index),
                [
                    str(bridge_tensor_name)
                    if str(name) == str(tensor_name)
                    else str(name)
                    for name in producer.outputs
                ],
            )
        for consumer in consumer_ops:
            consumer_index = active_index.operator_index(consumer)
            if consumer_index is None:
                continue
            active_index.replace_operator_inputs(
                int(consumer_index),
                [
                    str(bridge_tensor_name)
                    if str(name) == str(tensor_name)
                    else str(name)
                    for name in consumer.inputs
                ],
            )
        active_index.append_operator(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[str(bridge_tensor_name), str(perm_name)],
                outputs=[str(tensor_name)],
                options={"perm": [int(value) for value in list(perm)]},
            )
        )

    for tensor_name in list(model_ir.inputs):
        current_tensor = model_ir.tensors.get(str(tensor_name), None)
        desired_shape = desired_public_shape_map.get(str(tensor_name), None)
        desired_layout = desired_public_layout_map.get(
            str(tensor_name),
            LOGICAL_LAYOUT_UNKNOWN,
        )
        if current_tensor is None or desired_shape is None:
            continue
        _insert_public_boundary_layout_bridge(
            tensor_name=str(tensor_name),
            current_tensor=current_tensor,
            desired_shape=desired_shape,
            desired_layout=desired_layout,
            is_input=True,
        )

    for tensor_name in list(model_ir.outputs):
        current_tensor = model_ir.tensors.get(str(tensor_name), None)
        desired_shape = desired_public_shape_map.get(str(tensor_name), None)
        desired_layout = desired_public_layout_map.get(
            str(tensor_name),
            LOGICAL_LAYOUT_UNKNOWN,
        )
        if current_tensor is None or desired_shape is None:
            continue
        _insert_public_boundary_layout_bridge(
            tensor_name=str(tensor_name),
            current_tensor=current_tensor,
            desired_shape=desired_shape,
            desired_layout=desired_layout,
            is_input=False,
        )
