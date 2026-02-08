from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def build_reshape_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    shape_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    shape_values = ctx.get_constant_array(shape_name)
    if shape_values is None:
        raise NotImplementedError(
            f"Reshape shape tensor must be constant for flatbuffer_direct. op={node.name}"
        )
    new_shape = [int(v) for v in np.asarray(shape_values).reshape(-1).tolist()]
    shape_const = ctx.add_const_tensor(
        f"{output_name}_reshape_shape",
        np.asarray(new_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options={"newShape": new_shape},
        )
    )


def build_transpose_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    perm_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    perm = ctx.get_constant_array(perm_name)
    if perm is None:
        raise NotImplementedError(
            f"Transpose permutation tensor must be constant for flatbuffer_direct. op={node.name}"
        )
    perm_const = ctx.add_const_tensor(
        f"{output_name}_transpose_perm",
        np.asarray(perm, dtype=np.int32).reshape(-1),
    )

    ctx.add_operator(
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[input_name, perm_const],
            outputs=[output_name],
        )
    )


def build_concat_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)

    output_shape = ctx.get_tensor_shape(output_name)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += len(output_shape)

    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=input_names,
            outputs=[output_name],
            options={
                "axis": int(axis),
                "fusedActivationFunction": "NONE",
            },
        )
    )


def build_identity_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    output_shape = ctx.get_tensor_shape(output_name)
    shape_const = ctx.add_const_tensor(
        f"{output_name}_identity_shape",
        np.asarray(output_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape]},
        )
    )
