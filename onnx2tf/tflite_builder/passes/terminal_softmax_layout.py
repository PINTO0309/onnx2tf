from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR

_SOFTMAX_NHWC_PROPAGATED_MARKER = "__softmax_nhwc_propagated__"


def _rank4_metadata(
    tensor: Optional[TensorIR],
) -> Optional[Tuple[list[int], list[int]]]:
    """Plan rank-four shape metadata without mutating the public output."""

    if tensor is None:
        return None
    try:
        shape = [int(value) for value in tensor.shape]
        signature = (
            [int(value) for value in tensor.shape_signature]
            if tensor.shape_signature is not None
            else list(shape)
        )
    except (TypeError, ValueError):
        return None
    if len(shape) != 4 or len(signature) != 4:
        return None
    return shape, signature


def _optimize_terminal_softmax_transpose_after_nhwc_propagation(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Remove a marked Softmax's private terminal NHWC-to-NCHW adapter."""

    stats_key = "removed_terminal_softmax_transpose_after_nhwc_propagation"
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        required_types = {"SOFTMAX", "TRANSPOSE"}
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
        for operator_type in ("SOFTMAX", "TRANSPOSE")
    ):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {stats_key: 0}

    terminal_perm = [0, 3, 1, 2]
    public_inputs = {str(name) for name in model_ir.inputs}
    public_boundaries = public_inputs | {str(name) for name in model_ir.outputs}
    rewritten = 0

    while True:
        changed = False
        for raw_output_name in list(model_ir.outputs):
            output_name = str(raw_output_name)
            if output_name in public_inputs:
                continue
            if active_index.consumer_indices(output_name):
                continue
            if output_name in active_index.duplicate_producers:
                continue

            post_index = active_index.producers.get(output_name)
            if post_index is None:
                continue
            post_op = model_ir.operators[int(post_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.outputs[0]) != output_name
                or _read_transpose_perm(model_ir, post_op) != terminal_perm
            ):
                continue

            softmax_output_name = str(post_op.inputs[0])
            if softmax_output_name in public_boundaries:
                continue
            if softmax_output_name in active_index.duplicate_producers:
                continue
            softmax_index = active_index.producers.get(softmax_output_name)
            if softmax_index is None or int(softmax_index) >= int(post_index):
                continue
            softmax_op = model_ir.operators[int(softmax_index)]
            if (
                str(softmax_op.op_type) != "SOFTMAX"
                or len(softmax_op.inputs) != 1
                or len(softmax_op.outputs) != 1
                or str(softmax_op.outputs[0]) != softmax_output_name
                or active_index.consumer_indices(softmax_output_name)
                != [int(post_index)]
            ):
                continue

            options = (
                dict(softmax_op.options) if isinstance(softmax_op.options, dict) else {}
            )
            if not bool(options.get(_SOFTMAX_NHWC_PROPAGATED_MARKER, False)):
                continue

            source_tensor = model_ir.tensors.get(softmax_output_name)
            destination_tensor = model_ir.tensors.get(output_name)
            source_metadata = _rank4_metadata(source_tensor)
            if source_metadata is None or destination_tensor is None:
                continue
            source_shape, source_signature = source_metadata
            try:
                destination_quantization = _clone_quantization(
                    source_tensor.quantization
                )
            except Exception:
                continue
            options.pop(_SOFTMAX_NHWC_PROPAGATED_MARKER, None)

            # All topology, boundary, marker, and metadata guards are complete.
            # Preserve the public tensor object/provenance while moving its
            # producer identity and NHWC metadata to the retained Softmax.
            softmax_op.options = options
            _set_operator_outputs(
                model_ir=model_ir,
                op=softmax_op,
                new_outputs=[output_name],
                graph_index=active_index,
            )
            destination_tensor.dtype = str(source_tensor.dtype)
            destination_tensor.quantization = destination_quantization
            destination_tensor.shape = list(source_shape)
            destination_tensor.shape_signature = list(source_signature)
            active_index.remove_operator(int(post_index))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {stats_key: int(rewritten)}
