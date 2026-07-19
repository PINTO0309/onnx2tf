from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


def _repair_singleton_nhwc_conv_input_reshapes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Remove stale singleton NCHW adapters in front of NHWC Conv inputs.

    Layout propagation can move a rank-4 tensor to NHWC after Conv lowering has
    already emitted its NCHW->NHWC adapter. When both spatial dimensions are
    singleton, that adapter is represented as RESHAPE and can survive generic
    transpose cleanup. The filter input-channel dimension provides an exact,
    model-independent guard for recognizing the already-NHWC source.
    """

    repaired = 0
    graph_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    while True:
        changed = False
        for conv_idx in graph_index.operator_indices("CONV_2D"):
            conv_op = model_ir.operators[int(conv_idx)]
            if len(conv_op.inputs) < 2 or len(conv_op.outputs) != 1:
                continue
            adapter_output_name = str(conv_op.inputs[0])
            adapter_op = graph_index.producer(adapter_output_name)
            if adapter_op is None:
                continue
            adapter_idx = graph_index.operator_index(adapter_op)
            if adapter_idx is None:
                continue
            if str(adapter_op.op_type) != "RESHAPE" or len(adapter_op.inputs) < 1 or len(adapter_op.outputs) != 1:
                continue
            if adapter_output_name in model_ir.outputs:
                continue
            adapter_users = graph_index.consumer_indices(adapter_output_name)
            if adapter_users != [int(conv_idx)]:
                continue

            source_name = str(adapter_op.inputs[0])
            filter_name = str(conv_op.inputs[1])
            conv_output_name = str(conv_op.outputs[0])
            source_tensor = model_ir.tensors.get(source_name, None)
            adapter_tensor = model_ir.tensors.get(adapter_output_name, None)
            filter_tensor = model_ir.tensors.get(filter_name, None)
            output_tensor = model_ir.tensors.get(conv_output_name, None)
            if any(t is None for t in [source_tensor, adapter_tensor, filter_tensor, output_tensor]):
                continue
            source_shape = [int(v) for v in list(source_tensor.shape)]
            adapter_shape = [int(v) for v in list(adapter_tensor.shape)]
            filter_shape = [int(v) for v in list(filter_tensor.shape)]
            if len(source_shape) != 4 or len(adapter_shape) != 4 or len(filter_shape) != 4:
                continue
            filter_input_channels = int(filter_shape[3])
            if (
                int(source_shape[1]) != 1
                or int(source_shape[2]) != 1
                or int(source_shape[3]) != filter_input_channels
                or int(adapter_shape[3]) == filter_input_channels
                or int(np.prod(source_shape, dtype=np.int64))
                != int(np.prod(adapter_shape, dtype=np.int64))
            ):
                continue

            source_signature = (
                [int(v) for v in list(source_tensor.shape_signature)]
                if source_tensor.shape_signature is not None
                else list(source_shape)
            )
            if len(source_signature) != 4:
                continue

            updated_inputs = [str(v) for v in list(conv_op.inputs)]
            updated_inputs[0] = source_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=conv_op,
                new_inputs=updated_inputs,
                graph_index=graph_index,
            )

            output_tensor.shape = [
                int(source_shape[0]),
                int(source_shape[1]),
                int(source_shape[2]),
                int(filter_shape[0]),
            ]
            output_tensor.shape_signature = [
                int(source_signature[0]),
                int(source_signature[1]),
                int(source_signature[2]),
                int(filter_shape[0]),
            ]
            graph_index.remove_operator(int(adapter_idx))
            repaired += 1
            changed = True
            break
        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"repaired_singleton_nhwc_conv_input_reshapes": int(repaired)}


def _repair_stale_nchw_to_nhwc_conv_input_transposes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Bypass stale Conv adapters when their source is already NHWC.

    Late layout propagation can move a channel Split to NHWC after Conv
    lowering has emitted the usual NCHW->NHWC transpose for one of the split
    branches. The transpose then moves a spatial dimension into the channel
    position and produces a model that LiteRT cannot prepare. The filter input
    channel dimension is an exact invariant: bypass the adapter only when the
    source channel count matches it and the adapter output does not.
    """

    repaired = 0
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    graph_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    while True:
        changed = False
        for conv_idx in graph_index.operator_indices("CONV_2D"):
            conv_op = model_ir.operators[int(conv_idx)]
            if len(conv_op.inputs) < 2 or len(conv_op.outputs) != 1:
                continue
            adapter_output_name = str(conv_op.inputs[0])
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
            ):
                continue
            if adapter_output_name in model_ir.outputs:
                continue
            if graph_index.consumer_indices(adapter_output_name) != [int(conv_idx)]:
                continue

            source_name = str(adapter_op.inputs[0])
            filter_name = str(conv_op.inputs[1])
            conv_output_name = str(conv_op.outputs[0])
            source_tensor = model_ir.tensors.get(source_name, None)
            adapter_tensor = model_ir.tensors.get(adapter_output_name, None)
            filter_tensor = model_ir.tensors.get(filter_name, None)
            output_tensor = model_ir.tensors.get(conv_output_name, None)
            if any(
                tensor is None
                for tensor in (
                    source_tensor,
                    adapter_tensor,
                    filter_tensor,
                    output_tensor,
                )
            ):
                continue
            source_shape = [int(v) for v in list(source_tensor.shape)]
            adapter_shape = [int(v) for v in list(adapter_tensor.shape)]
            filter_shape = [int(v) for v in list(filter_tensor.shape)]
            if (
                len(source_shape) != 4
                or len(adapter_shape) != 4
                or len(filter_shape) != 4
            ):
                continue
            filter_input_channels = int(filter_shape[3])
            if (
                filter_input_channels <= 0
                or int(source_shape[3]) != filter_input_channels
                or int(adapter_shape[3]) == filter_input_channels
            ):
                continue

            source_signature = (
                [int(v) for v in list(source_tensor.shape_signature)]
                if source_tensor.shape_signature is not None
                else list(source_shape)
            )
            if len(source_signature) != 4:
                continue

            updated_inputs = [str(v) for v in list(conv_op.inputs)]
            updated_inputs[0] = source_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=conv_op,
                new_inputs=updated_inputs,
                graph_index=graph_index,
            )

            output_tensor.shape = [
                int(source_shape[0]),
                int(source_shape[1]),
                int(source_shape[2]),
                int(filter_shape[0]),
            ]
            output_tensor.shape_signature = [
                int(source_signature[0]),
                int(source_signature[1]),
                int(source_signature[2]),
                int(filter_shape[0]),
            ]
            graph_index.remove_operator(int(adapter_idx))
            repaired += 1
            changed = True
            break
        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "repaired_stale_nchw_to_nhwc_conv_input_transposes": int(repaired),
    }


def _run_indexed_conv_input_adapter_repairs(model_ir: ModelIR) -> Dict[str, int]:
    """Run singleton-Reshape and stale-Transpose Conv repairs with one index."""

    graph_index = ModelIRGraphIndex(model_ir)
    reshape_stats = _repair_singleton_nhwc_conv_input_reshapes(
        model_ir,
        graph_index=graph_index,
    )
    transpose_stats = _repair_stale_nchw_to_nhwc_conv_input_transposes(
        model_ir,
        graph_index=graph_index,
    )
    return {
        "repaired_singleton_nhwc_conv_input_reshapes": int(
            reshape_stats.get("repaired_singleton_nhwc_conv_input_reshapes", 0)
        ),
        "repaired_stale_nchw_to_nhwc_conv_input_transposes": int(
            transpose_stats.get(
                "repaired_stale_nchw_to_nhwc_conv_input_transposes",
                0,
            )
        ),
    }


def run_indexed_conv_input_adapter_repairs_summary(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Run indexed Conv-input repairs and include prune-only evidence."""

    initial_tensor_count = len(model_ir.tensors)
    result = _run_indexed_conv_input_adapter_repairs(model_ir)
    return {
        **result,
        "pruned_unused_tensors": max(
            0,
            int(initial_tensor_count - len(model_ir.tensors)),
        ),
    }
