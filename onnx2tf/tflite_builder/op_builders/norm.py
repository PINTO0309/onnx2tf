from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


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
