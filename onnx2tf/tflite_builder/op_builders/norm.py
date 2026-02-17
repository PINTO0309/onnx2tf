from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def build_l2_normalization_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    if ctx.model_ir.tensors[output_name].shape == [1] and ctx.model_ir.tensors[input_name].shape != [1]:
        ctx.model_ir.tensors[output_name].shape = list(ctx.model_ir.tensors[input_name].shape)
        ctx.model_ir.tensors[output_name].shape_signature = (
            list(ctx.model_ir.tensors[input_name].shape_signature)
            if ctx.model_ir.tensors[input_name].shape_signature is not None
            else list(ctx.model_ir.tensors[input_name].shape)
        )
    ctx.add_operator(
        OperatorIR(
            op_type="L2_NORMALIZATION",
            inputs=[input_name],
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )


def build_lrn_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    if len(input_shape) != 4:
        raise NotImplementedError(
            f"LRN supports rank-4 tensor only in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    if len(output_shape) != 4:
        output_shape = list(input_shape)
        output_tensor = ctx.model_ir.tensors[output_name]
        output_tensor.shape = list(output_shape)
        output_tensor.shape_signature = (
            list(ctx.model_ir.tensors[input_name].shape_signature)
            if ctx.model_ir.tensors[input_name].shape_signature is not None
            else list(input_shape)
        )

    size = int(node.attrs.get("size", 1))
    bias = float(node.attrs.get("bias", 1.0))
    alpha = float(node.attrs.get("alpha", 1e-4)) / float(size)
    beta = float(node.attrs.get("beta", 0.75))
    radius = int((size - 1) // 2)

    nhwc_input_shape = [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
    nhwc_output_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[3]), int(output_shape[1])]

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=nhwc_input_shape,
    )
    x_nhwc = make_transpose(
        ctx,
        input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )

    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOCAL_RESPONSE_NORMALIZATION",
            inputs=[x_nhwc],
            outputs=[y_nhwc],
            options={
                "radius": int(radius),
                "bias": float(bias),
                "alpha": float(alpha),
                "beta": float(beta),
            },
        )
    )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )


def build_batch_normalization_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    scale_name = node.inputs[1].name
    bias_name = node.inputs[2].name
    mean_name = node.inputs[3].name
    var_name = node.inputs[4].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    scale = ctx.get_constant_array(scale_name)
    bias = ctx.get_constant_array(bias_name)
    mean = ctx.get_constant_array(mean_name)
    var = ctx.get_constant_array(var_name)
    if scale is None or bias is None or mean is None or var is None:
        raise NotImplementedError(
            "BatchNormalization requires constant scale/bias/mean/var in flatbuffer_direct. "
            f"op={node.name}"
        )

    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    bias = np.asarray(bias, dtype=np.float32).reshape(-1)
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    var = np.asarray(var, dtype=np.float32).reshape(-1)
    eps = float(node.attrs.get("epsilon", 1e-5))

    bn_mul = scale / np.sqrt(var + eps)
    bn_add = bias - (mean * bn_mul)

    input_shape = ctx.get_tensor_shape(input_name)
    input_rank = len(input_shape)
    if input_rank == 4:
        bn_mul = bn_mul.reshape(1, -1, 1, 1)
        bn_add = bn_add.reshape(1, -1, 1, 1)
    elif input_rank == 2:
        bn_mul = bn_mul.reshape(1, -1)
        bn_add = bn_add.reshape(1, -1)

    mul_const = ctx.add_const_tensor(
        f"{node.name}_bn_mul",
        np.asarray(bn_mul, dtype=np.float32),
    )
    add_const = ctx.add_const_tensor(
        f"{node.name}_bn_add",
        np.asarray(bn_add, dtype=np.float32),
    )

    mul_out = ctx.add_intermediate_tensor(
        f"{node.name}_mul_out",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[input_name, mul_const],
            outputs=[mul_out],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[mul_out, add_const],
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
