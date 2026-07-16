from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _is_fully_known_positive_shape,
)
from onnx2tf.tflite_builder.ir import ModelIR


def sanitize_hardswish_tensor_shapes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """
    Keep HARD_SWISH output metadata identical to its input metadata.

    Late transpose/layout rewrites can leave stale HARD_SWISH output shapes.
    The pass is metadata-only and uses the shared graph index when available.
    """
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    candidate_operators = (
        (
            model_ir.operators[int(operator_index)]
            for operator_index in active_index.operator_indices("HARD_SWISH")
        )
        if active_index is not None
        else iter(model_ir.operators)
    )

    fixed = 0
    for op in candidate_operators:
        if str(op.op_type) != "HARD_SWISH":
            continue
        if len(op.inputs) < 1 or len(op.outputs) != 1:
            continue

        input_name = str(op.inputs[0])
        output_name = str(op.outputs[0])
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        if input_tensor is None or output_tensor is None:
            continue
        if input_tensor.shape is None or len(list(input_tensor.shape)) == 0:
            continue

        input_shape = [int(v) for v in list(input_tensor.shape)]
        input_signature_current = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else []
        )
        if _is_fully_known_positive_shape(input_shape):
            input_signature = [int(v) for v in list(input_shape)]
        else:
            input_signature = (
                [int(v) for v in list(input_signature_current)]
                if len(input_signature_current) > 0
                else [int(v) for v in list(input_shape)]
            )
        output_shape = (
            [int(v) for v in list(output_tensor.shape)]
            if output_tensor.shape is not None
            else []
        )
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_shape)]
        )

        if (
            input_signature_current == input_signature
            and output_shape == input_shape
            and output_signature == input_signature
        ):
            continue

        input_tensor.shape_signature = [int(v) for v in list(input_signature)]
        output_tensor.shape = [int(v) for v in list(input_shape)]
        output_tensor.shape_signature = [int(v) for v in list(input_signature)]
        fixed += 1

    return {"sanitized_hardswish_tensor_shapes": int(fixed)}
