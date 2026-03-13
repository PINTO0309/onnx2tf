from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import (
    OperatorIR,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.op_builders.shared import (
    make_transpose,
    materialize_broadcast_operand_for_gpu_delegate,
)


def _float_numpy_dtype(dtype: str) -> np.dtype:
    return np.dtype(np.float16) if str(dtype).upper() == "FLOAT16" else np.dtype(np.float32)


def _normalize_mvn_axes(
    *,
    axes_attr: Any,
    input_rank: int,
) -> list[int]:
    if input_rank < 1:
        raise NotImplementedError(
            "MeanVarianceNormalization requires input rank >= 1 in flatbuffer_direct."
        )

    if axes_attr is None:
        if input_rank < 3:
            source_axes = [0]
        elif input_rank == 3:
            source_axes = [0, 2]
        else:
            source_axes = [0, 2, 3]
    elif isinstance(axes_attr, np.ndarray):
        source_axes = [int(v) for v in np.asarray(axes_attr).reshape(-1).tolist()]
    elif isinstance(axes_attr, (list, tuple)):
        source_axes = [int(v) for v in axes_attr]
    else:
        source_axes = [int(axes_attr)]

    normalized_axes: list[int] = []
    for axis in source_axes:
        if axis < -input_rank or axis >= input_rank:
            raise NotImplementedError(
                "MeanVarianceNormalization axes must be within input rank in flatbuffer_direct. "
                f"rank={input_rank} axis={axis}"
            )
        normalized_axis = int(axis if axis >= 0 else axis + input_rank)
        if normalized_axis not in normalized_axes:
            normalized_axes.append(normalized_axis)
    return normalized_axes


def _reduced_shape_for_axes(
    *,
    input_shape: list[int],
    axes: list[int],
) -> list[int]:
    reduced_shape = [int(v) for v in input_shape]
    for axis in axes:
        reduced_shape[int(axis)] = 1
    return reduced_shape


def _infer_channel_axis_from_tensor_layout(
    *,
    input_shape: list[int],
    scale_size: int,
    logical_layout: str,
) -> int:
    rank = len(input_shape)
    normalized_layout = normalize_logical_layout(logical_layout)
    if is_channel_first_logical_layout(normalized_layout) and rank >= 2:
        return 1
    if is_channel_last_logical_layout(normalized_layout) and rank >= 2:
        return rank - 1

    candidate_axes: list[int] = []
    for axis in range(1, rank):
        dim = int(input_shape[axis])
        if dim > 0 and dim == int(scale_size):
            candidate_axes.append(int(axis))
    if len(candidate_axes) == 1:
        return int(candidate_axes[0])
    if len(candidate_axes) > 1:
        if rank - 1 in candidate_axes:
            return rank - 1
        if 1 in candidate_axes:
            return 1
        return int(candidate_axes[0])
    return 1 if rank >= 2 else 0


def _infer_channel_axis_from_transpose_producer(
    *,
    ctx: Any,
    tensor_name: str,
    rank: int,
) -> int | None:
    if rank not in {3, 4, 5}:
        return None
    producer_op = None
    for op in reversed(ctx.model_ir.operators):
        if str(tensor_name) in {str(v) for v in list(op.outputs)}:
            producer_op = op
            break
    if producer_op is None or str(producer_op.op_type) != "TRANSPOSE":
        return None

    perm: list[int] | None = None
    if len(list(producer_op.inputs)) >= 2:
        perm_tensor = ctx.model_ir.tensors.get(str(producer_op.inputs[1]), None)
        if perm_tensor is not None and perm_tensor.data is not None:
            try:
                perm = [int(v) for v in np.asarray(perm_tensor.data).reshape(-1).tolist()]
            except Exception:
                perm = None
    if perm is None and isinstance(getattr(producer_op, "options", None), dict):
        raw_perm = producer_op.options.get("perm", None)
        if raw_perm is not None:
            try:
                perm = [int(v) for v in list(raw_perm)]
            except Exception:
                perm = None
    if perm is None or len(perm) != rank:
        return None

    if rank == 3:
        if perm == [0, 2, 1]:
            return 1
        return None
    if rank == 4:
        if perm == [0, 3, 1, 2]:
            return 1
        if perm == [0, 2, 3, 1]:
            return 3
        return None
    if perm == [0, 4, 1, 2, 3]:
        return 1
    if perm == [0, 2, 3, 4, 1]:
        return 4
    return None


def _reshape_with_preserve_dynamic_shape(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    new_shape: list[int],
) -> None:
    shape_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape",
        np.asarray([int(v) for v in list(new_shape)], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            options={
                "newShape": [int(v) for v in list(new_shape)],
                "preserveDynamicShape": True,
            },
        )
    )


def build_group_normalization_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    scale_name = node.inputs[1].name
    bias_name = node.inputs[2].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor is not None and input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )

    groups = int(node.attrs.get("num_groups", node.attrs.get("groups", 1)))
    epsilon = float(node.attrs.get("epsilon", 1e-5))
    stash_type = int(node.attrs.get("stash_type", 1))
    activation = int(node.attrs.get("activation", 0))
    scale_size = int(np.asarray(ctx.get_constant_array(scale_name)).reshape(-1).size)
    preferred_channel_axes = [1, len(input_shape) - 1] if str(node.op) == "GroupNormalization" else [len(input_shape) - 1, 1]
    channel_axis = None
    for axis in preferred_channel_axes:
        if 0 <= int(axis) < len(input_shape) and int(input_shape[int(axis)]) == scale_size:
            channel_axis = int(axis)
            break
    if channel_axis is None:
        channel_axis = 1
    channels = (
        int(input_signature[channel_axis])
        if len(input_signature) > int(channel_axis) and int(input_signature[channel_axis]) > 0
        else int(input_shape[channel_axis])
    )
    group_size = int(channels // groups)

    compute_dtype = "FLOAT32" if int(stash_type) == 1 else str(output_dtype).upper()
    compute_input_name = input_name
    if input_dtype != compute_dtype:
        compute_input_name = ctx.add_intermediate_tensor(
            f"{node.name}_group_norm_input_cast",
            dtype=compute_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[compute_input_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    if int(channel_axis) == 1:
        grouped_signature = [int(input_signature[0]), int(groups), int(group_size)] + [int(v) for v in input_signature[2:]]
        grouped_shape = [int(input_shape[0]), int(groups), int(group_size)] + [int(v) for v in input_shape[2:]]
        reduce_axes = list(range(2, len(grouped_signature)))
        reduced_signature = [int(grouped_signature[0]), int(groups)] + [1] * int(len(grouped_signature) - 2)
        reduced_shape = [int(grouped_shape[0]), int(groups)] + [1] * int(len(grouped_shape) - 2)
        affine_shape = [1, int(channels)] + [1] * int(len(input_shape) - 2)
    else:
        grouped_signature = [int(input_signature[0])] + [int(v) for v in input_signature[1:-1]] + [int(groups), int(group_size)]
        grouped_shape = [int(input_shape[0])] + [int(v) for v in input_shape[1:-1]] + [int(groups), int(group_size)]
        reduce_axes = list(range(1, len(grouped_signature) - 2)) + [int(len(grouped_signature) - 1)]
        reduced_signature = [int(grouped_signature[0])] + [1] * int(len(grouped_signature) - 2) + [int(groups), 1]
        reduced_shape = [int(grouped_shape[0])] + [1] * int(len(grouped_shape) - 2) + [int(groups), 1]
        affine_shape = [1] + [1] * int(len(input_shape) - 2) + [int(channels)]
    grouped_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_grouped",
        dtype=compute_dtype,
        shape=grouped_shape,
    )
    grouped_tensor = ctx.model_ir.tensors.get(grouped_name, None)
    if grouped_tensor is not None:
        grouped_tensor.shape_signature = [int(v) for v in grouped_signature]
    _reshape_with_preserve_dynamic_shape(
        ctx=ctx,
        input_name=compute_input_name,
        output_name=grouped_name,
        new_shape=grouped_signature,
    )

    reduce_axes_name = ctx.add_const_tensor(
        f"{node.name}_group_norm_reduce_axes",
        np.asarray(reduce_axes, dtype=np.int32),
    )

    mean_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_mean",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    mean_tensor = ctx.model_ir.tensors.get(mean_name, None)
    if mean_tensor is not None:
        mean_tensor.shape_signature = [int(v) for v in reduced_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="MEAN",
            inputs=[grouped_name, reduce_axes_name],
            outputs=[mean_name],
            options={"keepDims": True},
        )
    )

    centered_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_centered",
        dtype=compute_dtype,
        shape=grouped_shape,
    )
    centered_tensor = ctx.model_ir.tensors.get(centered_name, None)
    if centered_tensor is not None:
        centered_tensor.shape_signature = [int(v) for v in grouped_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[grouped_name, mean_name],
            outputs=[centered_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    squared_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_squared",
        dtype=compute_dtype,
        shape=grouped_shape,
    )
    squared_tensor = ctx.model_ir.tensors.get(squared_name, None)
    if squared_tensor is not None:
        squared_tensor.shape_signature = [int(v) for v in grouped_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[centered_name, centered_name],
            outputs=[squared_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    variance_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_variance",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    variance_tensor = ctx.model_ir.tensors.get(variance_name, None)
    if variance_tensor is not None:
        variance_tensor.shape_signature = [int(v) for v in reduced_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="MEAN",
            inputs=[squared_name, reduce_axes_name],
            outputs=[variance_name],
            options={"keepDims": True},
        )
    )

    epsilon_name = ctx.add_const_tensor(
        f"{node.name}_group_norm_epsilon",
        np.asarray(epsilon, dtype=_float_numpy_dtype(compute_dtype)),
    )
    variance_eps_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_variance_eps",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    variance_eps_tensor = ctx.model_ir.tensors.get(variance_eps_name, None)
    if variance_eps_tensor is not None:
        variance_eps_tensor.shape_signature = [int(v) for v in reduced_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[variance_name, epsilon_name],
            outputs=[variance_eps_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    std_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_std",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    std_tensor = ctx.model_ir.tensors.get(std_name, None)
    if std_tensor is not None:
        std_tensor.shape_signature = [int(v) for v in reduced_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[variance_eps_name],
            outputs=[std_name],
        )
    )

    normalized_grouped_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_normalized_grouped",
        dtype=compute_dtype,
        shape=grouped_shape,
    )
    normalized_grouped_tensor = ctx.model_ir.tensors.get(normalized_grouped_name, None)
    if normalized_grouped_tensor is not None:
        normalized_grouped_tensor.shape_signature = [int(v) for v in grouped_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[centered_name, std_name],
            outputs=[normalized_grouped_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    normalized_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_normalized",
        dtype=compute_dtype,
        shape=input_shape,
    )
    normalized_tensor = ctx.model_ir.tensors.get(normalized_name, None)
    if normalized_tensor is not None:
        normalized_tensor.shape_signature = [int(v) for v in input_signature]
    _reshape_with_preserve_dynamic_shape(
        ctx=ctx,
        input_name=normalized_grouped_name,
        output_name=normalized_name,
        new_shape=input_signature,
    )

    scale = np.asarray(ctx.get_constant_array(scale_name), dtype=_float_numpy_dtype(output_dtype)).reshape(affine_shape)
    bias = np.asarray(ctx.get_constant_array(bias_name), dtype=_float_numpy_dtype(output_dtype)).reshape(affine_shape)
    scale_const_name = ctx.add_const_tensor(
        f"{node.name}_group_norm_scale",
        scale,
    )
    bias_const_name = ctx.add_const_tensor(
        f"{node.name}_group_norm_bias",
        bias,
    )

    affine_input_name = normalized_name
    if compute_dtype != output_dtype:
        affine_input_name = ctx.add_intermediate_tensor(
            f"{node.name}_group_norm_affine_input_cast",
            dtype=output_dtype,
            shape=input_shape,
        )
        affine_tensor = ctx.model_ir.tensors.get(affine_input_name, None)
        if affine_tensor is not None:
            affine_tensor.shape_signature = [int(v) for v in input_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[normalized_name],
                outputs=[affine_input_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )

    scaled_name = ctx.add_intermediate_tensor(
        f"{node.name}_group_norm_scaled",
        dtype=output_dtype,
        shape=input_shape,
    )
    scaled_tensor = ctx.model_ir.tensors.get(scaled_name, None)
    if scaled_tensor is not None:
        scaled_tensor.shape_signature = [int(v) for v in input_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[affine_input_name, scale_const_name],
            outputs=[scaled_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    affine_output_name = output_name
    if int(activation) == 1:
        affine_output_name = ctx.add_intermediate_tensor(
            f"{node.name}_group_norm_affine",
            dtype=output_dtype,
            shape=input_shape,
        )
        affine_output_tensor = ctx.model_ir.tensors.get(affine_output_name, None)
        if affine_output_tensor is not None:
            affine_output_tensor.shape_signature = [int(v) for v in input_signature]

    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[scaled_name, bias_const_name],
            outputs=[affine_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    if int(activation) == 1:
        sigmoid_name = ctx.add_intermediate_tensor(
            f"{node.name}_group_norm_sigmoid",
            dtype=output_dtype,
            shape=input_shape,
        )
        sigmoid_tensor = ctx.model_ir.tensors.get(sigmoid_name, None)
        if sigmoid_tensor is not None:
            sigmoid_tensor.shape_signature = [int(v) for v in input_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="LOGISTIC",
                inputs=[affine_output_name],
                outputs=[sigmoid_name],
                options={},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[affine_output_name, sigmoid_name],
                outputs=[output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.shape = [int(v) for v in input_shape]
        output_tensor.shape_signature = [int(v) for v in input_signature]


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
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    if input_rank >= 3:
        channel_axis = _infer_channel_axis_from_tensor_layout(
            input_shape=[int(v) for v in list(input_shape)],
            scale_size=int(np.asarray(scale).size),
            logical_layout=input_tensor.logical_layout if input_tensor is not None else "UNKNOWN",
        )
        coeff_shape = [1] * int(input_rank)
        coeff_shape[int(channel_axis)] = -1
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
    if bool(getattr(ctx, "optimization_for_gpu_delegate", False)):
        input_tensor = ctx.model_ir.tensors.get(input_name, None)
        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor is not None and input_tensor.shape_signature is not None
            else [int(v) for v in list(input_shape)]
        )
        mul_const = materialize_broadcast_operand_for_gpu_delegate(
            ctx=ctx,
            input_name=mul_const,
            target_shape=[int(v) for v in list(input_shape)],
            target_signature=input_signature,
            base_name=f"{node.name}_bn_mul",
        )
        add_const = materialize_broadcast_operand_for_gpu_delegate(
            ctx=ctx,
            input_name=add_const,
            target_shape=[int(v) for v in list(input_shape)],
            target_signature=input_signature,
            base_name=f"{node.name}_bn_add",
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


def build_mean_variance_normalization_op(node: Any, ctx: Any) -> None:
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

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_rank = len(input_shape)
    if input_rank < 1:
        raise NotImplementedError(
            f"MeanVarianceNormalization requires input rank >= 1 in flatbuffer_direct. op={node.name}"
        )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NotImplementedError(
            "MeanVarianceNormalization supports FLOAT16/FLOAT32 only in flatbuffer_direct. "
            f"op={node.name} input_dtype={input_dtype} output_dtype={output_dtype}"
        )

    compute_dtype = input_dtype
    np_compute_dtype = _float_numpy_dtype(compute_dtype)
    moments_axes = _normalize_mvn_axes(
        axes_attr=node.attrs.get("axes", None),
        input_rank=input_rank,
    )
    reduced_shape = _reduced_shape_for_axes(
        input_shape=input_shape,
        axes=moments_axes,
    )
    epsilon = float(getattr(ctx, "mvn_epsilon", 1e-10))

    x_name = input_name
    axes_name = ctx.add_const_tensor(
        f"{node.name}_mvn_axes",
        np.asarray(moments_axes, dtype=np.int32),
    )
    mean_name = ctx.add_intermediate_tensor(
        f"{node.name}_mvn_mean",
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
        f"{node.name}_mvn_centered",
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
        f"{node.name}_mvn_squared",
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

    variance_name = ctx.add_intermediate_tensor(
        f"{node.name}_mvn_variance",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MEAN",
            inputs=[squared_name, axes_name],
            outputs=[variance_name],
            options={"keepDims": True},
        )
    )

    epsilon_name = ctx.add_const_tensor(
        f"{node.name}_mvn_epsilon",
        np.asarray(epsilon, dtype=np_compute_dtype),
    )
    variance_eps_name = ctx.add_intermediate_tensor(
        f"{node.name}_mvn_variance_eps",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[variance_name, epsilon_name],
            outputs=[variance_eps_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    std_name = ctx.add_intermediate_tensor(
        f"{node.name}_mvn_std",
        dtype=compute_dtype,
        shape=reduced_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[variance_eps_name],
            outputs=[std_name],
        )
    )

    pre_output_name = output_name
    if output_dtype != compute_dtype:
        pre_output_name = ctx.add_intermediate_tensor(
            f"{node.name}_mvn_pre_output",
            dtype=compute_dtype,
            shape=input_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[centered_name, std_name],
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
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    channel_axis = _infer_channel_axis_from_tensor_layout(
        input_shape=input_shape,
        scale_size=int(np.asarray(scale).size),
        logical_layout=input_tensor.logical_layout if input_tensor is not None else "UNKNOWN",
    )
    if input_tensor is not None and normalize_logical_layout(input_tensor.logical_layout) == "UNKNOWN":
        ambiguous_candidate_axes = [
            int(axis)
            for axis in range(1, rank)
            if int(input_shape[axis]) > 0 and int(input_shape[axis]) == int(np.asarray(scale).size)
        ]
        if len(ambiguous_candidate_axes) > 1:
            producer_channel_axis = _infer_channel_axis_from_transpose_producer(
                ctx=ctx,
                tensor_name=input_name,
                rank=rank,
            )
            if producer_channel_axis is not None:
                channel_axis = int(producer_channel_axis)
    reduction_axes = [int(axis) for axis in range(1, rank) if int(axis) != int(channel_axis)]

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
        np.asarray(reduction_axes, dtype=np.int32),
    )
    reduced_shape = _reduced_shape_for_axes(
        input_shape=input_shape,
        axes=reduction_axes,
    )

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
    broadcast_shape = [1 for _ in range(rank)]
    broadcast_shape[int(channel_axis)] = int(scale_size)
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


def build_layer_normalization_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    scale_name = node.inputs[1].name
    bias_name = node.inputs[2].name if len(node.inputs) >= 3 and str(node.inputs[2].name) != "" else ""
    output_name = node.outputs[0].name

    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(scale_name)
    if bias_name != "":
        ctx.ensure_tensor(bias_name)
    ctx.ensure_tensor(output_name)
    if ctx.model_ir.tensors[output_name].shape == [1] and ctx.model_ir.tensors[input_name].shape != [1]:
        ctx.model_ir.tensors[output_name].shape = list(ctx.model_ir.tensors[input_name].shape)
        ctx.model_ir.tensors[output_name].shape_signature = (
            list(ctx.model_ir.tensors[input_name].shape_signature)
            if ctx.model_ir.tensors[input_name].shape_signature is not None
            else list(ctx.model_ir.tensors[input_name].shape)
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    rank = len(input_shape)
    axis_raw = int(node.attrs.get("axis", -1))
    axis = axis_raw + rank if axis_raw < 0 else axis_raw
    normalized_axes = [int(v) for v in range(axis, rank)]
    reduced_shape = list(input_shape)
    for axis_idx in normalized_axes:
        reduced_shape[axis_idx] = 1

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()

    stash_type = int(node.attrs.get("stash_type", 1))
    if stash_type == 1:
        compute_dtype = "FLOAT32"
    else:
        compute_dtype = input_dtype
    np_compute_dtype = _float_numpy_dtype(compute_dtype)
    epsilon = float(node.attrs.get("epsilon", 1e-5))

    x_name = input_name
    if input_dtype != compute_dtype:
        cast_in_name = ctx.add_intermediate_tensor(
            f"{node.name}_layernorm_input_cast",
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
        f"{node.name}_layernorm_axes",
        np.asarray(normalized_axes, dtype=np.int32),
    )
    mean_output_name = node.outputs[1].name if len(node.outputs) >= 2 else ""
    inv_std_output_name = node.outputs[2].name if len(node.outputs) >= 3 else ""
    mean_output_dtype = compute_dtype
    inv_std_output_dtype = compute_dtype
    if mean_output_name != "":
        ctx.ensure_tensor(mean_output_name, dtype=compute_dtype, shape=reduced_shape)
        mean_output_dtype = str(ctx.get_tensor_dtype(mean_output_name)).upper()
    if inv_std_output_name != "":
        ctx.ensure_tensor(inv_std_output_name, dtype=compute_dtype, shape=reduced_shape)
        inv_std_output_dtype = str(ctx.get_tensor_dtype(inv_std_output_name)).upper()

    if mean_output_name != "" and mean_output_dtype == compute_dtype:
        mean_name = mean_output_name
    else:
        mean_name = ctx.add_intermediate_tensor(
            f"{node.name}_layernorm_mean",
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
        f"{node.name}_layernorm_centered",
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
        f"{node.name}_layernorm_squared",
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
        f"{node.name}_layernorm_var",
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
        f"{node.name}_layernorm_eps",
        np.asarray(epsilon, dtype=np_compute_dtype),
    )
    var_eps_name = ctx.add_intermediate_tensor(
        f"{node.name}_layernorm_var_eps",
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
        f"{node.name}_layernorm_std",
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
        f"{node.name}_layernorm_one",
        np.asarray(1.0, dtype=np_compute_dtype),
    )
    if inv_std_output_name != "" and inv_std_output_dtype == compute_dtype:
        inv_std_name = inv_std_output_name
    else:
        inv_std_name = ctx.add_intermediate_tensor(
            f"{node.name}_layernorm_inv_std",
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
        f"{node.name}_layernorm_normalized",
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

    if mean_output_name != "":
        if mean_output_dtype != compute_dtype:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[mean_name],
                    outputs=[mean_output_name],
                    options={"inDataType": compute_dtype, "outDataType": mean_output_dtype},
                )
            )

    if inv_std_output_name != "":
        if inv_std_output_dtype != compute_dtype:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[inv_std_name],
                    outputs=[inv_std_output_name],
                    options={"inDataType": compute_dtype, "outDataType": inv_std_output_dtype},
                )
            )

    affine_dtype = input_dtype
    affine_input_name = normalized_name
    if compute_dtype != affine_dtype:
        affine_input_name = ctx.add_intermediate_tensor(
            f"{node.name}_layernorm_affine_input_cast",
            dtype=affine_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[normalized_name],
                outputs=[affine_input_name],
                options={"inDataType": compute_dtype, "outDataType": affine_dtype},
            )
        )

    scale_dtype = str(ctx.get_tensor_dtype(scale_name)).upper()
    scale_input_name = scale_name
    if scale_dtype != affine_dtype:
        scale_shape = [int(v) for v in ctx.get_tensor_shape(scale_name)]
        scale_input_name = ctx.add_intermediate_tensor(
            f"{node.name}_layernorm_scale_cast",
            dtype=affine_dtype,
            shape=scale_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[scale_name],
                outputs=[scale_input_name],
                options={"inDataType": scale_dtype, "outDataType": affine_dtype},
            )
        )

    scaled_name = ctx.add_intermediate_tensor(
        f"{node.name}_layernorm_scaled",
        dtype=affine_dtype,
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[affine_input_name, scale_input_name],
            outputs=[scaled_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    if bias_name == "":
        bias_input_name = ctx.add_const_tensor(
            f"{node.name}_layernorm_bias_default",
            np.asarray(0.0, dtype=_float_numpy_dtype(affine_dtype)),
        )
    else:
        bias_dtype = str(ctx.get_tensor_dtype(bias_name)).upper()
        bias_input_name = bias_name
        if bias_dtype != affine_dtype:
            bias_shape = [int(v) for v in ctx.get_tensor_shape(bias_name)]
            bias_input_name = ctx.add_intermediate_tensor(
                f"{node.name}_layernorm_bias_cast",
                dtype=affine_dtype,
                shape=bias_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[bias_name],
                    outputs=[bias_input_name],
                    options={"inDataType": bias_dtype, "outDataType": affine_dtype},
                )
            )

    pre_output_name = output_name
    if output_dtype != affine_dtype:
        pre_output_name = ctx.add_intermediate_tensor(
            f"{node.name}_layernorm_pre_output",
            dtype=affine_dtype,
            shape=input_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[scaled_name, bias_input_name],
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
                options={"inDataType": affine_dtype, "outDataType": output_dtype},
            )
        )
