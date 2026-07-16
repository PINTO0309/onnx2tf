from __future__ import annotations

import re
from typing import Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR


def repair_orphan_recurrent_step_tensors(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> int:
    """Reconnect orphan recurrent step aliases to matching Reshape outputs."""

    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    candidates: list[tuple[str, str]] = []
    for tensor_name in model_ir.tensors:
        normalized_name = str(tensor_name)
        if normalized_name in public_inputs:
            continue
        match = re.match(r"^(.+_(?:h|c)_step_)(\d+)$", normalized_name)
        if match is None:
            continue
        candidates.append(
            (
                normalized_name,
                f"{match.group(1)}shape_{match.group(2)}",
            )
        )
    if len(candidates) == 0:
        return 0

    graph_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    repaired = 0
    for tensor_name, shape_tensor_name in candidates:
        if graph_index.producer(tensor_name) is not None:
            continue
        replacement_name: Optional[str] = None
        for op in graph_index.consumers_of(shape_tensor_name):
            if (
                str(op.op_type) != "RESHAPE"
                or len(op.inputs) < 2
                or len(op.outputs) != 1
                or str(op.inputs[1]) != shape_tensor_name
            ):
                continue
            candidate_name = str(op.outputs[0])
            if candidate_name == tensor_name:
                replacement_name = None
                break
            replacement_name = candidate_name
            break
        if replacement_name is None:
            continue
        for consumer in graph_index.consumers_of(tensor_name):
            consumer_index = graph_index.operator_index(consumer)
            if consumer_index is None:
                continue
            graph_index.replace_operator_inputs(
                int(consumer_index),
                [
                    replacement_name
                    if str(input_name) == tensor_name
                    else str(input_name)
                    for input_name in consumer.inputs
                ],
            )
        if tensor_name not in public_outputs:
            model_ir.tensors.pop(tensor_name, None)
        repaired += 1
    return int(repaired)
