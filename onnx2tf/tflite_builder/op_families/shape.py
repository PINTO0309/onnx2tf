from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.core.op_contracts import NodeValidationError


def _validate_range(node: Any, ctx: Any) -> None:
    """Accept the scalar and one-element forms allowed by ONNX Range."""

    for index, input_obj in enumerate(node.inputs[:3]):
        shape = [int(value) for value in ctx.get_tensor_shape(input_obj.name)]
        if shape == [] or shape == [1]:
            continue
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Range requires scalar-like inputs represented as shape [] or [1] "
                "in flatbuffer_direct. "
                f"input_index={index} input_shape={shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
