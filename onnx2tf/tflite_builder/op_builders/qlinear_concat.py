from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.quantized_common import (
    _normalize_axis,
    _promote_internal_uint8_tensor_to_int8,
    _require_const,
    _set_tensor_dtype_from_array,
    _set_tensor_quantization,
)


def build_qlinear_concat_op(node: Any, ctx: Any) -> None:
    y_scale_name = node.inputs[0].name
    y_zero_name = node.inputs[1].name
    output_name = node.outputs[0].name

    if (len(node.inputs) - 2) % 3 != 0 or len(node.inputs) < 5:
        raise NotImplementedError(
            f"QLinearConcat inputs must be [y_scale, y_zero_point, (x, x_scale, x_zero_point)+]. "
            f"op={node.name} input_count={len(node.inputs)}"
        )

    input_groups = (len(node.inputs) - 2) // 3
    input_names: list[str] = []
    input_scale_names: list[str] = []
    input_zero_names: list[str] = []
    for i in range(input_groups):
        base = 2 + i * 3
        input_names.append(node.inputs[base].name)
        input_scale_names.append(node.inputs[base + 1].name)
        input_zero_names.append(node.inputs[base + 2].name)

    for input_name in input_names:
        ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    y_scale = _require_const(ctx, y_scale_name, "QLinearConcat output scale")
    y_zero = _require_const(ctx, y_zero_name, "QLinearConcat output zero_point")
    _set_tensor_dtype_from_array(ctx, output_name, y_zero)
    _promote_internal_uint8_tensor_to_int8(ctx, output_name)

    first_shape = [int(v) for v in ctx.get_tensor_shape(input_names[0])]
    rank = len(first_shape)
    axis = int(node.attrs.get("axis", 1))
    axis = _normalize_axis(axis, rank)

    input_signatures: list[list[int]] = []
    for idx, input_name in enumerate(input_names):
        input_scale = _require_const(ctx, input_scale_names[idx], f"QLinearConcat input[{idx}] scale")
        input_zero = _require_const(ctx, input_zero_names[idx], f"QLinearConcat input[{idx}] zero_point")
        _set_tensor_dtype_from_array(ctx, input_name, input_zero)
        _promote_internal_uint8_tensor_to_int8(ctx, input_name)
        _set_tensor_quantization(
            ctx=ctx,
            tensor_name=input_name,
            scale=input_scale,
            zero_point=input_zero,
            quantized_dimension=0 if np.asarray(input_scale).size <= 1 else _normalize_axis(1, rank),
        )
        input_tensor = ctx.model_ir.tensors[input_name]
        input_signature = (
            list(input_tensor.shape_signature)
            if input_tensor.shape_signature is not None
            else list(input_tensor.shape)
        )
        input_signatures.append(input_signature)

    _set_tensor_quantization(
        ctx=ctx,
        tensor_name=output_name,
        scale=y_scale,
        zero_point=y_zero,
        quantized_dimension=0 if np.asarray(y_scale).size <= 1 else _normalize_axis(1, rank),
    )

    output_shape = [int(v) for v in first_shape]
    output_signature = list(input_signatures[0]) if len(input_signatures) > 0 else list(output_shape)
    concat_dim = 0
    concat_sig_dim = 0
    for idx, input_name in enumerate(input_names):
        shape_i = [int(v) for v in ctx.get_tensor_shape(input_name)]
        sig_i = input_signatures[idx]
        if len(shape_i) != rank:
            raise NotImplementedError(
                f"QLinearConcat input ranks must match. op={node.name} input={input_name} shape={shape_i}"
            )
        concat_dim += int(shape_i[axis])
        concat_sig_dim += int(sig_i[axis]) if int(sig_i[axis]) >= 0 else 0
    output_shape[axis] = int(concat_dim)
    output_signature[axis] = int(concat_sig_dim) if concat_sig_dim > 0 else -1
    ctx.model_ir.tensors[output_name].shape = list(output_shape)
    ctx.model_ir.tensors[output_name].shape_signature = list(output_signature)

    dq_inputs: list[str] = []
    for input_name in input_names:
        dq_name = ctx.add_intermediate_tensor(
            f"{node.name}_{input_name}_dq",
            dtype="FLOAT32",
            shape=ctx.get_tensor_shape(input_name),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[input_name],
                outputs=[dq_name],
            )
        )
        dq_inputs.append(dq_name)

    concat_out = ctx.add_intermediate_tensor(
        f"{node.name}_concat_out",
        dtype="FLOAT32",
        shape=list(output_shape),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=dq_inputs,
            outputs=[concat_out],
            options={
                "axis": int(axis),
                "fusedActivationFunction": "NONE",
            },
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="QUANTIZE",
            inputs=[concat_out],
            outputs=[output_name],
        )
    )
