from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


def sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Remove a channel-invalid Transpose in front of NHWC Conv consumers.

    The adapter is removable only when every consumer is a ``CONV_2D`` data
    input whose filter expects the source tensor's last dimension and rejects
    the adapter output's last dimension. Public adapter outputs and mixed
    fan-out remain untouched.
    """

    stat_name = "sanitized_wrong_way_nchw_to_nhwc_transpose_before_conv"
    if not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators):
        # Keep the former standalone owner's tensor-pruning side effect without
        # allocating an index for a graph that cannot contain a candidate.
        _prune_unused_tensors(model_ir)
        return {stat_name: 0}

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    removed = 0
    expected_permutation = [0, 2, 3, 1]
    model_outputs = {str(name) for name in model_ir.outputs}

    while True:
        changed = False
        for operator_index in active_index.operator_indices("TRANSPOSE"):
            operator = model_ir.operators[int(operator_index)]
            if len(operator.inputs) < 2 or len(operator.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, operator) != expected_permutation:
                continue

            input_name = str(operator.inputs[0])
            output_name = str(operator.outputs[0])
            if output_name in model_outputs:
                continue

            input_tensor = model_ir.tensors.get(input_name)
            output_tensor = model_ir.tensors.get(output_name)
            if (
                input_tensor is None
                or output_tensor is None
                or len(input_tensor.shape) != 4
                or len(output_tensor.shape) != 4
            ):
                continue

            consumer_indices = active_index.consumer_indices(output_name)
            if len(consumer_indices) == 0:
                continue

            input_channels = int(input_tensor.shape[3])
            output_channels = int(output_tensor.shape[3])
            removable = True
            for consumer_index in consumer_indices:
                consumer = model_ir.operators[int(consumer_index)]
                if (
                    str(consumer.op_type) != "CONV_2D"
                    or len(consumer.inputs) < 2
                    or str(consumer.inputs[0]) != output_name
                ):
                    removable = False
                    break
                filter_tensor = model_ir.tensors.get(str(consumer.inputs[1]))
                if filter_tensor is None or len(filter_tensor.shape) != 4:
                    removable = False
                    break
                expected_channels = int(filter_tensor.shape[3])
                if not (
                    input_channels == expected_channels
                    and output_channels != expected_channels
                ):
                    removable = False
                    break

            if not removable:
                continue

            _replace_tensor_inputs(
                model_ir,
                output_name,
                input_name,
                graph_index=active_index,
            )
            active_index.remove_operator(int(operator_index))
            removed += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {stat_name: int(removed)}
