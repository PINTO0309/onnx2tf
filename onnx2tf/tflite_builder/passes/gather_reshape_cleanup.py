from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _shape_and_signature(
    tensor: Optional[TensorIR],
) -> Optional[Tuple[list[int], list[int]]]:
    """Return a usable shape/signature pair for a runtime tensor."""

    if tensor is None:
        return None
    try:
        if not _is_fully_known_positive_shape(tensor.shape):
            return None
        shape = [int(value) for value in tensor.shape]
        signature = (
            [int(value) for value in tensor.shape_signature]
            if tensor.shape_signature is not None
            else list(shape)
        )
    except (TypeError, ValueError):
        return None
    if len(signature) != len(shape):
        return None
    if any(value == 0 or value < -1 for value in signature):
        return None
    return shape, signature


def _is_singleton_zero_gather_index(tensor: Optional[TensorIR]) -> bool:
    """Accept one signed-integer zero regardless of scalar legalization shape."""

    if tensor is None or tensor.data is None:
        return False
    if str(tensor.dtype).upper() not in {"INT32", "INT64"}:
        return False
    try:
        values = np.asarray(tensor.data)
    except Exception:
        return False
    if values.size != 1 or not np.issubdtype(values.dtype, np.signedinteger):
        return False
    return int(values.reshape(-1)[0]) == 0


def _optimize_gather_axis0_singleton_to_reshape_input_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Bypass a value-preserving leading-singleton Gather before Reshape.

    Accepted graph::

        x[1, ...] -- GATHER(axis=0, indices=[0]) --> gathered
                  -- RESHAPE(shape) --> output

    The Gather may use a physically rank-one singleton index because direct
    TFLite lowering legalizes scalar indices that way. Its buffer still holds
    exactly one zero, so selecting the only leading slice preserves both the
    value order and element count of ``x``. The gathered tensor must describe
    the rank-reduced tail of ``x`` and be private to the Reshape data input.
    """

    stats_key = "optimized_gather_axis0_singleton_to_reshape_input_chains"
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        required_types = {"GATHER", "RESHAPE"}
        for operator in model_ir.operators:
            required_types.discard(str(operator.op_type))
            if not required_types:
                break
        if required_types:
            _prune_unused_tensors(model_ir, layout_state=layout_state)
            return {stats_key: 0}
        active_index = ModelIRGraphIndex(model_ir)
    elif any(
        not active_index.operator_indices(operator_type)
        for operator_type in ("GATHER", "RESHAPE")
    ):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {stats_key: 0}

    public_boundaries = {str(name) for name in [*model_ir.inputs, *model_ir.outputs]}
    rewritten = 0

    while True:
        changed = False
        for gather_index in active_index.operator_indices("GATHER"):
            gather_op = model_ir.operators[int(gather_index)]
            if len(gather_op.inputs) != 2 or len(gather_op.outputs) != 1:
                continue

            gather_input_name = str(gather_op.inputs[0])
            gather_indices_name = str(gather_op.inputs[1])
            gather_output_name = str(gather_op.outputs[0])
            if gather_output_name in public_boundaries:
                continue
            if gather_output_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(gather_output_name) != int(gather_index):
                continue

            options = gather_op.options
            if not isinstance(options, dict):
                continue
            try:
                input_rank = len(model_ir.tensors[gather_input_name].shape)
                axis = int(options.get("axis", 0))
                if axis < 0:
                    axis += int(input_rank)
                batch_dims = int(options.get("batchDims", 0))
                if "batch_dims" in options:
                    snake_batch_dims = int(options["batch_dims"])
                    if "batchDims" in options and snake_batch_dims != batch_dims:
                        continue
                    batch_dims = snake_batch_dims
            except (KeyError, TypeError, ValueError):
                continue
            if input_rank <= 0 or axis != 0 or batch_dims != 0:
                continue

            input_tensor = model_ir.tensors.get(gather_input_name)
            gathered_tensor = model_ir.tensors.get(gather_output_name)
            input_metadata = _shape_and_signature(input_tensor)
            gathered_metadata = _shape_and_signature(gathered_tensor)
            if input_metadata is None or gathered_metadata is None:
                continue
            input_shape, input_signature = input_metadata
            gathered_shape, gathered_signature = gathered_metadata
            if input_shape[0] != 1 or input_signature[0] != 1:
                continue
            if gathered_shape != input_shape[1:]:
                continue
            if gathered_signature != input_signature[1:]:
                continue
            if str(input_tensor.dtype).upper() != str(gathered_tensor.dtype).upper():
                continue
            try:
                quantization_matches = bool(
                    input_tensor.quantization == gathered_tensor.quantization
                )
            except (TypeError, ValueError):
                quantization_matches = False
            if not quantization_matches:
                continue
            if not _is_singleton_zero_gather_index(
                model_ir.tensors.get(gather_indices_name)
            ):
                continue

            gather_consumers = active_index.consumer_indices(gather_output_name)
            if len(gather_consumers) != 1:
                continue
            reshape_index = int(gather_consumers[0])
            if int(gather_index) >= reshape_index:
                continue
            reshape_op = model_ir.operators[reshape_index]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != gather_output_name
            ):
                continue

            # Commit only after the complete value, metadata, topology, and
            # public-boundary contract has been validated. Restarting against
            # the maintained index preserves the former fixed point when an
            # inner removal exposes another leading-singleton Gather.
            _set_operator_inputs(
                model_ir=model_ir,
                op=reshape_op,
                new_inputs=[
                    gather_input_name,
                    *[str(name) for name in reshape_op.inputs[1:]],
                ],
                graph_index=active_index,
            )
            active_index.remove_operator(gather_index)
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {stats_key: int(rewritten)}
