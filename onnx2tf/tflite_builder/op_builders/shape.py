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
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    perm = None
    if len(node.inputs) >= 2:
        perm_name = node.inputs[1].name
        perm = ctx.get_constant_array(perm_name)
    if perm is None and "perm" in node.attrs:
        attr_perm = node.attrs.get("perm")
        if isinstance(attr_perm, (list, tuple)):
            perm = np.asarray([int(v) for v in attr_perm], dtype=np.int32)
        elif attr_perm is not None:
            perm = np.asarray([int(attr_perm)], dtype=np.int32)
    if perm is None:
        input_rank = len(ctx.get_tensor_shape(input_name))
        perm = np.asarray(list(reversed(range(input_rank))), dtype=np.int32)

    if perm is None:
        raise NotImplementedError(
            f"Transpose permutation must be resolvable for flatbuffer_direct. op={node.name}"
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


def _resolve_axes_from_attr_or_input(node: Any, ctx: Any) -> list[int]:
    axes = None
    if len(node.inputs) >= 2:
        axes_arr = ctx.get_constant_array(node.inputs[1].name)
        if axes_arr is None:
            raise NotImplementedError(
                f"{node.op} axes must be constant for flatbuffer_direct. op={node.name}"
            )
        axes = [int(v) for v in np.asarray(axes_arr).reshape(-1).tolist()]
    elif "axes" in node.attrs:
        attr_axes = node.attrs["axes"]
        if isinstance(attr_axes, (list, tuple)):
            axes = [int(v) for v in attr_axes]
        else:
            axes = [int(attr_axes)]
    return [] if axes is None else axes


def build_squeeze_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
    axes = _resolve_axes_from_attr_or_input(node, ctx)
    if len(axes) == 0:
        axes = [idx for idx, dim in enumerate(input_shape) if int(dim) == 1]

    normalized_axes: list[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += rank
        if a < 0 or a >= rank:
            raise NotImplementedError(
                f"Squeeze axis is out of range. op={node.name} axis={axis} rank={rank}"
            )
        if a not in normalized_axes:
            normalized_axes.append(a)

    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[input_name],
            outputs=[output_name],
            options={"squeezeDims": [int(v) for v in normalized_axes]},
        )
    )


def build_unsqueeze_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    _ = _resolve_axes_from_attr_or_input(node, ctx)
    output_shape = ctx.get_tensor_shape(output_name)
    shape_const = ctx.add_const_tensor(
        f"{output_name}_unsqueeze_shape",
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


def build_space_to_depth_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    block_size = int(node.attrs.get("blocksize", 0))
    if block_size <= 1:
        raise NotImplementedError(
            f"SpaceToDepth blocksize must be > 1. op={node.name} blocksize={block_size}"
        )
    ctx.add_operator(
        OperatorIR(
            op_type="SPACE_TO_DEPTH",
            inputs=[input_name],
            outputs=[output_name],
            options={"blockSize": int(block_size)},
        )
    )
