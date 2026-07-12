from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


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
