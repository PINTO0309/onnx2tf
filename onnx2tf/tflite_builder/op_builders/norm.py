from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.ir import OperatorIR


def build_l2_normalization_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    ctx.add_operator(
        OperatorIR(
            op_type="L2_NORMALIZATION",
            inputs=[input_name],
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
