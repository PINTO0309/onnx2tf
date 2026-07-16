from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _shape_and_signature(
    tensor: Optional[TensorIR],
) -> Optional[Tuple[list[int], list[int]]]:
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
    if len(shape) != len(signature):
        return None
    return shape, signature


def _singleton_signed_axis(
    tensor: Optional[TensorIR],
) -> Optional[Tuple[int, np.dtype]]:
    if tensor is None or tensor.data is None:
        return None
    if str(tensor.dtype).upper() not in {"INT32", "INT64"}:
        return None
    try:
        values = np.asarray(tensor.data)
    except Exception:
        return None
    if values.size != 1 or not np.issubdtype(values.dtype, np.signedinteger):
        return None
    return int(values.reshape(-1)[0]), values.dtype


def _optimize_transpose_pre_argmax_nhwc_terminal_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Bypass a proven NHWC-to-NCHW adapter before channel ArgMax."""

    stats_key = "optimized_transpose_pre_argmax_nhwc_terminal_chains"
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        required_types = {"TRANSPOSE", "ARG_MAX"}
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
        for operator_type in ("TRANSPOSE", "ARG_MAX")
    ):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {stats_key: 0}

    permutation = [0, 3, 1, 2]
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_boundaries = public_inputs | public_outputs
    rewritten = 0

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    while True:
        changed = False
        for transpose_index in active_index.operator_indices("TRANSPOSE"):
            transpose = model_ir.operators[int(transpose_index)]
            if (
                len(transpose.inputs) != 2
                or len(transpose.outputs) != 1
                or _read_transpose_perm(model_ir, transpose) != permutation
            ):
                continue

            source_name = str(transpose.inputs[0])
            transposed_name = str(transpose.outputs[0])
            if source_name in public_outputs:
                continue
            if transposed_name in public_boundaries:
                continue
            if transposed_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(transposed_name) != int(transpose_index):
                continue

            transposed_consumers = active_index.consumer_indices(transposed_name)
            if len(transposed_consumers) != 1:
                continue
            argmax_index = int(transposed_consumers[0])
            if int(transpose_index) >= argmax_index:
                continue
            argmax = model_ir.operators[argmax_index]
            if (
                str(argmax.op_type) != "ARG_MAX"
                or len(argmax.inputs) != 2
                or len(argmax.outputs) != 1
                or str(argmax.inputs[0]) != transposed_name
            ):
                continue

            axis_name = str(argmax.inputs[1])
            axis_plan = _singleton_signed_axis(model_ir.tensors.get(axis_name))
            if axis_plan is None:
                continue
            old_axis, axis_dtype = axis_plan
            if old_axis < 0:
                old_axis += 4
            if old_axis != 1:
                continue
            new_axis = int(permutation[old_axis])

            source_metadata = _shape_and_signature(model_ir.tensors.get(source_name))
            transposed_metadata = _shape_and_signature(
                model_ir.tensors.get(transposed_name)
            )
            output_name = str(argmax.outputs[0])
            output_metadata = _shape_and_signature(model_ir.tensors.get(output_name))
            if (
                source_metadata is None
                or transposed_metadata is None
                or output_metadata is None
            ):
                continue
            source_shape, source_signature = source_metadata
            transposed_shape, transposed_signature = transposed_metadata
            output_shape, output_signature = output_metadata
            if len(source_shape) != 4 or len(transposed_shape) != 4:
                continue
            if _permute_shape(source_shape, permutation) != transposed_shape:
                continue
            if _permute_shape(source_signature, permutation) != transposed_signature:
                continue
            source_tensor = model_ir.tensors[source_name]
            transposed_tensor = model_ir.tensors[transposed_name]
            if str(source_tensor.dtype).upper() != str(transposed_tensor.dtype).upper():
                continue
            if output_shape != source_shape[:3]:
                continue
            if output_signature != source_signature[:3]:
                continue

            axis_consumers = active_index.consumer_indices(axis_name)
            clone_axis = (
                axis_consumers != [argmax_index] or axis_name in public_boundaries
            )
            axis_input_name = axis_name
            planned_axis_tensor: Optional[TensorIR] = None
            if clone_axis:
                axis_tensor = model_ir.tensors[axis_name]
                axis_input_name = _unique_tensor_name(f"{axis_name}_nhwc")
                try:
                    axis_quantization = _clone_quantization(axis_tensor.quantization)
                except Exception:
                    continue
                planned_axis_tensor = TensorIR(
                    name=axis_input_name,
                    dtype=str(axis_tensor.dtype),
                    shape=[1],
                    shape_signature=[1],
                    data=np.asarray([new_axis], dtype=axis_dtype),
                    is_variable=False,
                    quantization=axis_quantization,
                )

            # Every topology, boundary, axis, shape, and clone guard is now
            # complete. Commit the constant, ArgMax edge, and adapter removal.
            if planned_axis_tensor is not None:
                model_ir.tensors[axis_input_name] = planned_axis_tensor
            elif not _write_const_ints_to_tensor(
                model_ir.tensors.get(axis_name),
                [new_axis],
            ):
                continue
            _set_operator_inputs(
                model_ir=model_ir,
                op=argmax,
                new_inputs=[source_name, axis_input_name],
                graph_index=active_index,
            )
            active_index.remove_operator(int(transpose_index))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {stats_key: int(rewritten)}
