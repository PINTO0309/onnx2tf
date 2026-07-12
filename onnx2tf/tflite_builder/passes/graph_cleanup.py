from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_singleton_constant_tensor,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_singleton_constant_float,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


def _optimize_duplicate_transpose_fanout(model_ir: ModelIR) -> Dict[str, int]:
    """
    Deduplicate fan-out TRANSPOSE nodes with identical input and permutation.

    Target pattern:
      X --TRANSPOSE(P)--> Y0
      X --TRANSPOSE(P)--> Y1
      ...

    Rewritten:
      X --TRANSPOSE(P)--> Y0
      (all uses of Y1, ... are rewired to Y0; duplicate TRANSPOSE nodes removed)
    """
    removed_duplicates = 0
    graph_index = ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        canonical_by_key: Dict[Tuple[str, Tuple[int, ...]], int] = {}

        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue

            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            perm = _read_transpose_perm(model_ir, op)
            if perm is None:
                continue

            key = (input_name, tuple(int(value) for value in perm))
            canonical_idx = canonical_by_key.get(key, None)
            if canonical_idx is None:
                canonical_by_key[key] = int(op_idx)
                continue

            if output_name in model_ir.outputs:
                continue

            canonical_op = model_ir.operators[int(canonical_idx)]
            if len(canonical_op.outputs) != 1:
                continue
            canonical_output = str(canonical_op.outputs[0])
            if canonical_output == output_name:
                continue

            canonical_tensor = model_ir.tensors.get(canonical_output, None)
            duplicate_tensor = model_ir.tensors.get(output_name, None)
            if canonical_tensor is not None and duplicate_tensor is not None:
                if canonical_tensor.shape == [1] and duplicate_tensor.shape != [1]:
                    canonical_tensor.shape = [
                        int(value) for value in duplicate_tensor.shape
                    ]
                    canonical_tensor.shape_signature = (
                        [
                            int(value)
                            for value in duplicate_tensor.shape_signature
                        ]
                        if duplicate_tensor.shape_signature is not None
                        else [int(value) for value in duplicate_tensor.shape]
                    )
                if (
                    canonical_tensor.quantization is None
                    and duplicate_tensor.quantization is not None
                ):
                    canonical_tensor.quantization = _clone_quantization(
                        duplicate_tensor.quantization
                    )
                if (
                    str(canonical_tensor.dtype) == "FLOAT32"
                    and str(duplicate_tensor.dtype) != "FLOAT32"
                ):
                    canonical_tensor.dtype = str(duplicate_tensor.dtype)

            _replace_tensor_inputs(
                model_ir,
                output_name,
                canonical_output,
                graph_index=graph_index,
            )
            graph_index.remove_operator(op_idx)
            removed_duplicates += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"removed_duplicate_transpose_fanout": int(removed_duplicates)}


def _optimize_duplicate_reshape_fanout(model_ir: ModelIR) -> Dict[str, int]:
    """
    Deduplicate fan-out RESHAPE nodes with identical input and target shape.

    Target pattern:
      X --RESHAPE(S)--> Y0
      X --RESHAPE(S)--> Y1
      ...

    Rewritten:
      X --RESHAPE(S)--> Y0
      (all uses of Y1, ... are rewired to Y0; duplicate RESHAPE nodes removed)
    """
    removed_duplicates = 0
    graph_index = ModelIRGraphIndex(model_ir)

    def _read_reshape_target_shape(op: OperatorIR) -> Optional[List[int]]:
        if str(op.op_type) != "RESHAPE":
            return None
        if isinstance(op.options, dict):
            new_shape = op.options.get("newShape", None)
            if isinstance(new_shape, list) and len(new_shape) > 0:
                try:
                    return [int(v) for v in list(new_shape)]
                except Exception:
                    pass
        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            shape_values = _read_const_ints_from_tensor(shape_tensor)
            if shape_values is not None and len(shape_values) > 0:
                return [int(v) for v in list(shape_values)]
        return None

    while True:
        changed = False
        canonical_by_key: Dict[Tuple[str, Tuple[int, ...]], int] = {}

        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "RESHAPE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue

            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            target_shape = _read_reshape_target_shape(op)
            if target_shape is None:
                continue

            key = (input_name, tuple(int(v) for v in list(target_shape)))
            canonical_idx = canonical_by_key.get(key, None)
            if canonical_idx is None:
                canonical_by_key[key] = int(op_idx)
                continue

            if output_name in model_ir.outputs:
                # Preserve user-visible graph output names.
                continue

            canonical_op = model_ir.operators[int(canonical_idx)]
            if len(canonical_op.outputs) != 1:
                continue
            canonical_output = str(canonical_op.outputs[0])
            if canonical_output == output_name:
                continue

            canonical_tensor = model_ir.tensors.get(canonical_output, None)
            duplicate_tensor = model_ir.tensors.get(output_name, None)
            if canonical_tensor is not None and duplicate_tensor is not None:
                if canonical_tensor.shape == [1] and duplicate_tensor.shape != [1]:
                    canonical_tensor.shape = [int(v) for v in list(duplicate_tensor.shape)]
                    canonical_tensor.shape_signature = (
                        [int(v) for v in list(duplicate_tensor.shape_signature)]
                        if duplicate_tensor.shape_signature is not None
                        else [int(v) for v in list(duplicate_tensor.shape)]
                    )
                if canonical_tensor.quantization is None and duplicate_tensor.quantization is not None:
                    canonical_tensor.quantization = _clone_quantization(duplicate_tensor.quantization)
                if str(canonical_tensor.dtype) == "FLOAT32" and str(duplicate_tensor.dtype) != "FLOAT32":
                    canonical_tensor.dtype = str(duplicate_tensor.dtype)

            _replace_tensor_inputs(
                model_ir,
                output_name,
                canonical_output,
                graph_index=graph_index,
            )
            graph_index.remove_operator(op_idx)
            removed_duplicates += 1
            changed = True
            break

        if not changed:
            break

    if removed_duplicates > 0:
        _prune_unused_tensors(model_ir)
    return {
        "removed_duplicate_reshape_fanout": int(removed_duplicates),
    }


def _optimize_maximum_minimum_relu0to1_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Replace clamp chains MAXIMUM(0.0) -> MINIMUM(1.0) with RELU_0_TO_1.

    Target:
      X --MAXIMUM(0.0)--> M --MINIMUM(1.0)--> Y

    Rewrite:
      X --RELU_0_TO_1--> Y

    Safety:
    - MAXIMUM and MINIMUM side inputs must be singleton constants.
    - MAXIMUM output must be consumed only by the matched MINIMUM.
    """
    rewritten = 0
    atol = 1e-6
    graph_index = ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for min_idx, min_op in enumerate(model_ir.operators):
            if str(min_op.op_type) != "MINIMUM" or len(min_op.inputs) != 2 or len(min_op.outputs) != 1:
                continue

            min_input0 = str(min_op.inputs[0])
            min_input1 = str(min_op.inputs[1])
            if _is_singleton_constant_tensor(model_ir, min_input0):
                min_data_name = str(min_input1)
                min_const_name = str(min_input0)
            elif _is_singleton_constant_tensor(model_ir, min_input1):
                min_data_name = str(min_input0)
                min_const_name = str(min_input1)
            else:
                continue
            min_const_value = _read_singleton_constant_float(model_ir, min_const_name)
            if min_const_value is None or not np.isclose(float(min_const_value), 1.0, atol=atol):
                continue
            if min_data_name in model_outputs:
                continue

            max_idx = producers.get(min_data_name, None)
            if max_idx is None:
                continue
            max_op = model_ir.operators[int(max_idx)]
            if str(max_op.op_type) != "MAXIMUM" or len(max_op.inputs) != 2 or len(max_op.outputs) != 1:
                continue
            if str(max_op.outputs[0]) != str(min_data_name):
                continue

            max_input0 = str(max_op.inputs[0])
            max_input1 = str(max_op.inputs[1])
            if _is_singleton_constant_tensor(model_ir, max_input0):
                max_data_name = str(max_input1)
                max_const_name = str(max_input0)
            elif _is_singleton_constant_tensor(model_ir, max_input1):
                max_data_name = str(max_input0)
                max_const_name = str(max_input1)
            else:
                continue
            max_const_value = _read_singleton_constant_float(model_ir, max_const_name)
            if max_const_value is None or not np.isclose(float(max_const_value), 0.0, atol=atol):
                continue

            max_users = [int(v) for v in consumers.get(str(min_data_name), [])]
            if len(max_users) != 1 or int(max_users[0]) != int(min_idx):
                continue

            min_op.op_type = "RELU_0_TO_1"
            min_op.version = 1
            _set_operator_inputs(
                model_ir=model_ir,
                op=min_op,
                new_inputs=[str(max_data_name)],
                graph_index=graph_index,
            )
            min_op.options = {}

            graph_index.remove_operator(max_idx)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"rewritten_maximum_minimum_relu0to1_chains": int(rewritten)}
