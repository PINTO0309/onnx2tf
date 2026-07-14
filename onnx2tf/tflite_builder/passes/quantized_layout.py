from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _prune_unused_tensors
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


_NCHW_TO_NHWC = [0, 2, 3, 1]

_LAYOUT_PRESERVING_OPS = {
    "ABS",
    "ADD",
    "CAST",
    "DEQUANTIZE",
    "DIV",
    "ELU",
    "ERF",
    "EXP",
    "FLOOR",
    "GELU",
    "HARD_SWISH",
    "LEAKY_RELU",
    "LOG",
    "LOGICAL_NOT",
    "LOGISTIC",
    "MAXIMUM",
    "MINIMUM",
    "MUL",
    "NEG",
    "POW",
    "QUANTIZE",
    "RELU",
    "RELU6",
    "ROUND",
    "RSQRT",
    "SIGN",
    "SQRT",
    "SQUARE",
    "SUB",
    "TANH",
}


def _transpose_perm(model_ir: ModelIR, op: OperatorIR) -> Optional[List[int]]:
    if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 2:
        return None
    tensor = model_ir.tensors.get(str(op.inputs[1]))
    if tensor is None or tensor.data is None:
        return None
    values = np.asarray(tensor.data).reshape(-1)
    return [int(value) for value in values.tolist()]


def _rank4_runtime_inputs(model_ir: ModelIR, op: OperatorIR) -> List[str]:
    result: List[str] = []
    for input_name in op.inputs:
        tensor = model_ir.tensors.get(str(input_name))
        if tensor is None or tensor.data is not None:
            continue
        if len(list(tensor.shape)) == 4:
            result.append(str(input_name))
    return result


def _propagate_channel_last_layout_hints(model_ir: ModelIR) -> int:
    hints = {
        str(name)
        for name in model_ir.metadata.get(
            "assume_channel_last_layout_tensor_names",
            [],
        )
        if str(name) != ""
    }
    initial_count = len(hints)

    while True:
        changed = False
        for op in model_ir.operators:
            if len(op.outputs) != 1:
                continue
            output_name = str(op.outputs[0])
            if output_name in hints:
                continue
            output_tensor = model_ir.tensors.get(output_name)
            if output_tensor is None or len(list(output_tensor.shape)) not in {3, 4, 5}:
                continue

            runtime_inputs = [
                str(input_name)
                for input_name in op.inputs
                if (
                    model_ir.tensors.get(str(input_name)) is not None
                    and model_ir.tensors[str(input_name)].data is None
                )
            ]
            same_rank_runtime_inputs = [
                input_name
                for input_name in runtime_inputs
                if len(list(model_ir.tensors[input_name].shape))
                == len(list(output_tensor.shape))
            ]
            if not same_rank_runtime_inputs:
                continue

            op_type = str(op.op_type)
            if op_type == "MEAN":
                if (
                    not bool(op.options.get("keepDims", False))
                    or str(op.inputs[0]) not in hints
                ):
                    continue
            elif op_type in _LAYOUT_PRESERVING_OPS:
                if not all(name in hints for name in same_rank_runtime_inputs):
                    continue
            else:
                continue

            hints.add(output_name)
            output_tensor.logical_layout = "NHWC"
            output_tensor.physical_layout = "NHWC"
            changed = True

        if not changed:
            break

    model_ir.metadata["assume_channel_last_layout_tensor_names"] = sorted(hints)
    return int(len(hints) - initial_count)


def _trace_convinteger_data_root(
    *,
    model_ir: ModelIR,
    start_name: str,
    producer_map: Dict[str, int],
    consumer_map: Dict[str, List[int]],
    terminal_transpose_index: int,
) -> Optional[tuple[str, List[str]]]:
    current_name = str(start_name)
    chain_names: List[str] = [str(current_name)]
    current_consumer = int(terminal_transpose_index)

    while True:
        producer_index = producer_map.get(str(current_name))
        if producer_index is None:
            break
        producer = model_ir.operators[int(producer_index)]
        if str(producer.onnx_op_type or "") != "ConvInteger":
            break
        if str(producer.op_type) not in {"CAST", "SUB"}:
            break
        if set(consumer_map.get(str(current_name), [])) != {int(current_consumer)}:
            return None

        if str(producer.op_type) == "CAST":
            if len(producer.inputs) != 1:
                return None
            next_name = str(producer.inputs[0])
        else:
            candidates = _rank4_runtime_inputs(model_ir, producer)
            output_tensor = model_ir.tensors.get(str(current_name))
            if output_tensor is None:
                return None
            output_shape = [int(value) for value in list(output_tensor.shape)]
            candidates = [
                name
                for name in candidates
                if [int(value) for value in list(model_ir.tensors[name].shape)]
                == output_shape
            ]
            if len(candidates) != 1:
                return None
            next_name = str(candidates[0])

        current_consumer = int(producer_index)
        current_name = str(next_name)
        chain_names.append(str(current_name))

    return str(current_name), chain_names


def repair_channel_last_convinteger_input_transposes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Remove a stale ConvInteger NCHW->NHWC input bridge after layout promotion.

    ConvInteger is initially lowered while its ONNX input is NCHW. A later
    layout pass may promote the complete upstream elementwise region to NHWC.
    If the old input bridge survives, it transposes the already-NHWC tensor a
    second time. The channel-last provenance set is the guard that makes this
    repair independent of ambiguous shapes such as [1, 128, 128, 128].
    """
    propagated = _propagate_channel_last_layout_hints(model_ir)
    repaired = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    channel_last_names = {
        str(name)
        for name in model_ir.metadata.get(
            "assume_channel_last_layout_tensor_names",
            [],
        )
        if str(name) != ""
    }
    if not channel_last_names:
        return {
            "propagated_channel_last_layout_hints": int(propagated),
            "repaired_channel_last_convinteger_input_transposes": 0,
        }

    while True:
        producer_map = graph_index.producers
        consumer_map = graph_index.consumers
        changed = False

        for transpose_index in graph_index.operator_indices("TRANSPOSE"):
            transpose_op = model_ir.operators[int(transpose_index)]
            if str(transpose_op.onnx_op_type or "") != "ConvInteger":
                continue
            if _transpose_perm(model_ir, transpose_op) != _NCHW_TO_NHWC:
                continue
            if len(transpose_op.inputs) < 2 or len(transpose_op.outputs) != 1:
                continue

            transpose_output = str(transpose_op.outputs[0])
            output_users = consumer_map.get(transpose_output, [])
            if len(output_users) != 1:
                continue
            conv_index = int(output_users[0])
            conv_op = model_ir.operators[int(conv_index)]
            if (
                str(conv_op.onnx_op_type or "") != "ConvInteger"
                or str(conv_op.op_type) not in {"CONV_2D", "DEPTHWISE_CONV_2D"}
                or len(conv_op.inputs) < 1
                or str(conv_op.inputs[0]) != transpose_output
            ):
                continue

            traced = _trace_convinteger_data_root(
                model_ir=model_ir,
                start_name=str(transpose_op.inputs[0]),
                producer_map=producer_map,
                consumer_map=consumer_map,
                terminal_transpose_index=int(transpose_index),
            )
            if traced is None:
                continue
            root_name, chain_names = traced
            if str(root_name) not in channel_last_names:
                continue

            root_tensor = model_ir.tensors.get(str(root_name))
            transpose_output_tensor = model_ir.tensors.get(transpose_output)
            if root_tensor is None or transpose_output_tensor is None:
                continue
            root_shape = [int(value) for value in list(root_tensor.shape)]
            output_shape = [int(value) for value in list(transpose_output_tensor.shape)]
            if len(root_shape) != 4 or root_shape != output_shape:
                continue

            root_signature = (
                [int(value) for value in list(root_tensor.shape_signature)]
                if root_tensor.shape_signature is not None
                else list(root_shape)
            )
            for chain_name in chain_names:
                chain_tensor = model_ir.tensors.get(str(chain_name))
                if chain_tensor is None:
                    continue
                chain_tensor.shape = list(root_shape)
                chain_tensor.shape_signature = list(root_signature)
                chain_tensor.logical_layout = "NHWC"
                chain_tensor.physical_layout = "NHWC"
                channel_last_names.add(str(chain_name))

            new_conv_inputs = [str(name) for name in conv_op.inputs]
            new_conv_inputs[0] = str(transpose_op.inputs[0])
            graph_index.replace_operator_inputs(conv_index, new_conv_inputs)
            graph_index.remove_operator(int(transpose_index))
            repaired += 1
            changed = True
            break

        if not changed:
            break

    if repaired > 0:
        model_ir.metadata["assume_channel_last_layout_tensor_names"] = sorted(
            channel_last_names
        )
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)

    return {
        "propagated_channel_last_layout_hints": int(propagated),
        "repaired_channel_last_convinteger_input_transposes": int(repaired),
    }
