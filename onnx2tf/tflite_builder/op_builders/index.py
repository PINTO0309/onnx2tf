from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.ir import OperatorIR


def build_gather_op(node: Any, ctx: Any) -> None:
    params_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(params_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(output_name)

    input_rank = len(ctx.get_tensor_shape(params_name))
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims != 0:
        raise NotImplementedError(
            f"Gather batch_dims != 0 is not supported in flatbuffer_direct. "
            f"op={node.name} batch_dims={batch_dims}"
        )

    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[params_name, indices_name],
            outputs=[output_name],
            options={
                "axis": int(axis),
                "batchDims": int(batch_dims),
            },
        )
    )
