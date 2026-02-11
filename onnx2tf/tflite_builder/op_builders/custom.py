from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.ir import OperatorIR


def build_custom_passthrough_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs if i.name != ""]
    output_names = [o.name for o in node.outputs if o.name != ""]
    for name in input_names:
        ctx.ensure_tensor(name)
    for name in output_names:
        ctx.ensure_tensor(name)

    custom_code = f"ONNX_{str(node.op).upper()}"
    ctx.add_operator(
        OperatorIR(
            op_type="CUSTOM",
            inputs=input_names,
            outputs=output_names,
            options={
                "customCode": custom_code,
                "customOptionsFormat": "FLEXBUFFERS",
                "customOptions": b"",
                "onnxOp": str(node.op),
                "onnxNodeName": str(node.name),
            },
        )
    )
