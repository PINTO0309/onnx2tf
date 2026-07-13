from __future__ import annotations

from typing import Dict, List, Optional, Set

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
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _has_channel_last_factorized_rank3_sequence_consumer,
    _is_channel_last_factorized_reshape,
    _is_channel_last_factorized_rank3_sequence_reshape,
    _perm_cf_to_cl,
    _perm_cl_to_cf,
    _read_transpose_perm,
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
