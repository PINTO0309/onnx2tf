from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def _float_numpy_dtype(dtype: str) -> np.dtype:
    return np.float16 if str(dtype).upper() == "FLOAT16" else np.float32


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
    if input_rank >= 3:
        # ONNX BatchNormalization uses channel axis=1 for rank>=3.
        # TFLite binary broadcast aligns trailing axes, so channel coefficients
        # must be materialized as [1, C, 1, ...] to keep channel-wise semantics.
        coeff_shape = [1, -1] + [1] * int(input_rank - 2)
        bn_mul = bn_mul.reshape(coeff_shape)
        bn_add = bn_add.reshape(coeff_shape)
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


def build_instance_normalization_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    scale_name = node.inputs[1].name
    bias_name = node.inputs[2].name
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

    scale = ctx.get_constant_array(scale_name)
    bias = ctx.get_constant_array(bias_name)
    if scale is None or bias is None:
        raise NotImplementedError(
            "InstanceNormalization requires constant scale/bias in flatbuffer_direct. "
            f"op={node.name}"
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    rank = len(input_shape)
    if rank < 3:
        raise NotImplementedError(
            f"InstanceNormalization requires input rank >= 3 in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NotImplementedError(
            "InstanceNormalization supports FLOAT16/FLOAT32 only in flatbuffer_direct. "
            f"op={node.name} input_dtype={input_dtype} output_dtype={output_dtype}"
        )

    compute_dtype = input_dtype
    np_compute_dtype = _float_numpy_dtype(compute_dtype)
    epsilon = float(node.attrs.get("epsilon", 1e-5))

    x_name = input_name
    if input_dtype != compute_dtype:
        cast_in_name = ctx.add_intermediate_tensor(
            f"{node.name}_instnorm_input_cast",
            dtype=compute_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[cast_in_name],
                options={"inDataType": input_dtype, "outDataType": compute_dtype},
            )
        )
        x_name = cast_in_name

    axes_name = ctx.add_const_tensor(
        f"{node.name}_instnorm_axes",
        np.asarray([int(v) for v in range(2, rank)], dtype=np.int32),
    )
    reduced_shape = [int(input_shape[0]), int(input_shape[1])] + [1 for _ in range(rank - 2)]

    mean_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_mean",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MEAN",
            inputs=[x_name, axes_name],
            outputs=[mean_name],
            options={"keepDims": True},
        )
    )

    centered_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_centered",
        dtype=compute_dtype,
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[x_name, mean_name],
            outputs=[centered_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    squared_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_squared",
        dtype=compute_dtype,
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[centered_name, centered_name],
            outputs=[squared_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    var_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_var",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MEAN",
            inputs=[squared_name, axes_name],
            outputs=[var_name],
            options={"keepDims": True},
        )
    )

    eps_name = ctx.add_const_tensor(
        f"{node.name}_instnorm_eps",
        np.asarray(epsilon, dtype=np_compute_dtype),
    )
    var_eps_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_var_eps",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[var_name, eps_name],
            outputs=[var_eps_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    std_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_std",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[var_eps_name],
            outputs=[std_name],
        )
    )

    one_name = ctx.add_const_tensor(
        f"{node.name}_instnorm_one",
        np.asarray(1.0, dtype=np_compute_dtype),
    )
    inv_std_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_inv_std",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[one_name, std_name],
            outputs=[inv_std_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    normalized_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_normalized",
        dtype=compute_dtype,
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[centered_name, inv_std_name],
            outputs=[normalized_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    scale_size = int(np.asarray(scale).size)
    broadcast_shape = [1, scale_size] + [1 for _ in range(rank - 2)]
    scale_name_const = ctx.add_const_tensor(
        f"{node.name}_instnorm_scale",
        np.asarray(scale, dtype=np_compute_dtype).reshape(broadcast_shape),
    )
    bias_name_const = ctx.add_const_tensor(
        f"{node.name}_instnorm_bias",
        np.asarray(bias, dtype=np_compute_dtype).reshape(broadcast_shape),
    )

    scaled_name = ctx.add_intermediate_tensor(
        f"{node.name}_instnorm_scaled",
        dtype=compute_dtype,
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[normalized_name, scale_name_const],
            outputs=[scaled_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    pre_output_name = output_name
    if output_dtype != compute_dtype:
        pre_output_name = ctx.add_intermediate_tensor(
            f"{node.name}_instnorm_pre_output",
            dtype=compute_dtype,
            shape=input_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[scaled_name, bias_name_const],
            outputs=[pre_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    if pre_output_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[pre_output_name],
                outputs=[output_name],
                options={"inDataType": compute_dtype, "outDataType": output_dtype},
            )
        )
