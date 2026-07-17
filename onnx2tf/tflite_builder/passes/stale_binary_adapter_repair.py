from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


def _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Bypass stale layout adapters before channel-last binary constants.

    Late NHWC propagation can leave the data tensor of a decomposed
    BatchNormalization already channel-last while a later binary-layout pass
    inserts another NCHW->NHWC transpose. The channel-last constant provides
    an exact guard: its final dimension must match the source final dimension
    and must not match the adapter final dimension.
    """

    repaired = 0
    binary_ops = {"ADD", "MUL", "SUB", "DIV", "MAXIMUM", "MINIMUM"}
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    graph_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    while True:
        changed = False
        for binary_idx in graph_index.operator_indices_for_types(binary_ops):
            binary_op = model_ir.operators[int(binary_idx)]
            if len(binary_op.inputs) != 2 or len(binary_op.outputs) != 1:
                continue
            for data_input_idx, const_input_idx in ((0, 1), (1, 0)):
                adapter_output_name = str(binary_op.inputs[data_input_idx])
                adapter_op = graph_index.producer(adapter_output_name)
                if adapter_op is None:
                    continue
                adapter_idx = graph_index.operator_index(adapter_op)
                if adapter_idx is None:
                    continue
                if (
                    str(adapter_op.op_type) != "TRANSPOSE"
                    or len(adapter_op.inputs) < 2
                    or len(adapter_op.outputs) != 1
                    or _read_transpose_perm(model_ir, adapter_op)
                    != perm_nchw_to_nhwc
                    or adapter_output_name in model_ir.outputs
                    or graph_index.consumer_indices(adapter_output_name)
                    != [int(binary_idx)]
                ):
                    continue

                source_name = str(adapter_op.inputs[0])
                const_name = str(binary_op.inputs[const_input_idx])
                source_tensor = model_ir.tensors.get(source_name, None)
                adapter_tensor = model_ir.tensors.get(adapter_output_name, None)
                const_tensor = model_ir.tensors.get(const_name, None)
                output_tensor = model_ir.tensors.get(
                    str(binary_op.outputs[0]),
                    None,
                )
                if any(
                    tensor is None
                    for tensor in (
                        source_tensor,
                        adapter_tensor,
                        const_tensor,
                        output_tensor,
                    )
                ):
                    continue
                source_shape = [int(v) for v in list(source_tensor.shape)]
                adapter_shape = [int(v) for v in list(adapter_tensor.shape)]
                const_shape = [int(v) for v in list(const_tensor.shape)]
                if len(source_shape) != 4 or len(adapter_shape) != 4:
                    continue
                channelwise_const_matches = (
                    len(const_shape) == 4
                    and const_shape[:3] == [1, 1, 1]
                    and int(const_shape[3]) > 1
                    and int(source_shape[3]) == int(const_shape[3])
                    and int(adapter_shape[3]) != int(const_shape[3])
                )
                peer_producer_op = graph_index.producer(const_name)
                peer_producer_type = (
                    str(peer_producer_op.op_type)
                    if peer_producer_op is not None
                    else ""
                )
                nhwc_peer_matches = (
                    len(const_shape) == 4
                    and source_shape == const_shape
                    and adapter_shape != const_shape
                    and peer_producer_type
                    in {"CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV"}
                )
                if (
                    not (
                        channelwise_const_matches or nhwc_peer_matches
                    )
                ):
                    continue

                source_signature = (
                    [int(v) for v in list(source_tensor.shape_signature)]
                    if source_tensor.shape_signature is not None
                    else list(source_shape)
                )
                if len(source_signature) != 4:
                    continue

                updated_inputs = [str(v) for v in list(binary_op.inputs)]
                updated_inputs[data_input_idx] = source_name
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=binary_op,
                    new_inputs=updated_inputs,
                    graph_index=graph_index,
                )
                output_tensor.shape = list(source_shape)
                output_tensor.shape_signature = list(source_signature)
                graph_index.remove_operator(int(adapter_idx))
                repaired += 1
                changed = True
                break
            if changed:
                break
        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": int(
            repaired
        ),
    }
