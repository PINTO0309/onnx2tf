from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.quantized import _propagate_shape


def build_dynamic_quantize_linear_op(node: Any, ctx: Any) -> None:
    x_name = node.inputs[0].name
    y_name = node.outputs[0].name
    y_scale_name = node.outputs[1].name
    y_zero_name = node.outputs[2].name
    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(y_name)
    ctx.ensure_tensor(y_scale_name)
    ctx.ensure_tensor(y_zero_name)

    x_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    x_rank = len(x_shape)
    if x_rank <= 0:
        raise NotImplementedError(
            f"DynamicQuantizeLinear requires input rank >= 1 in flatbuffer_direct. op={node.name}"
        )

    # Propagate q-output shape from input when output metadata is placeholder.
    _propagate_shape(ctx, x_name, y_name)

    x_dtype = str(ctx.get_tensor_dtype(x_name)).upper()
    compute_dtype = "FLOAT32"
    compute_np_dtype = np.float32
    x_compute_name = x_name
    if x_dtype != compute_dtype:
        x_compute_name = ctx.add_intermediate_tensor(
            f"{y_name}_dql_input_f32",
            dtype=compute_dtype,
            shape=x_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[x_name],
                outputs=[x_compute_name],
                options={
                    "inDataType": x_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    axes_name = ctx.add_const_tensor(
        f"{y_name}_dql_reduce_axes",
        np.asarray([int(v) for v in range(x_rank)], dtype=np.int32),
    )
    zero_f_name = ctx.add_const_tensor(
        f"{y_name}_dql_zero_f",
        np.asarray(0.0, dtype=compute_np_dtype),
    )
    qmax_f_name = ctx.add_const_tensor(
        f"{y_name}_dql_qmax_f",
        np.asarray(255.0, dtype=compute_np_dtype),
    )
    eps_f_name = ctx.add_const_tensor(
        f"{y_name}_dql_eps_f",
        np.asarray(1e-8, dtype=compute_np_dtype),
    )

    scalar_shape = [1]

    # min(x) = -max(-x)
    x_neg_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_x_neg",
        dtype=compute_dtype,
        shape=x_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="NEG",
            inputs=[x_compute_name],
            outputs=[x_neg_name],
        )
    )
    max_neg_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_max_neg",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="REDUCE_MAX",
            inputs=[x_neg_name, axes_name],
            outputs=[max_neg_name],
            options={"keepDims": False},
        )
    )
    min_x_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_min_x",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="NEG",
            inputs=[max_neg_name],
            outputs=[min_x_name],
        )
    )

    max_x_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_max_x",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="REDUCE_MAX",
            inputs=[x_compute_name, axes_name],
            outputs=[max_x_name],
            options={"keepDims": False},
        )
    )

    min_with_zero_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_min_with_zero",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MINIMUM",
            inputs=[min_x_name, zero_f_name],
            outputs=[min_with_zero_name],
            options={},
        )
    )
    max_with_zero_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_max_with_zero",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MAXIMUM",
            inputs=[max_x_name, zero_f_name],
            outputs=[max_with_zero_name],
            options={},
        )
    )

    range_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_range",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[max_with_zero_name, min_with_zero_name],
            outputs=[range_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    y_scale_raw_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_scale_raw",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[range_name, qmax_f_name],
            outputs=[y_scale_raw_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    # Avoid divide-by-zero in degenerate all-constant ranges.
    y_scale_compute_name = y_scale_name
    if str(ctx.get_tensor_dtype(y_scale_name)).upper() != compute_dtype:
        y_scale_compute_name = ctx.add_intermediate_tensor(
            f"{y_name}_dql_scale_f32",
            dtype=compute_dtype,
            shape=scalar_shape,
        )
    ctx.add_operator(
        OperatorIR(
            op_type="MAXIMUM",
            inputs=[y_scale_raw_name, eps_f_name],
            outputs=[y_scale_compute_name],
            options={},
        )
    )

    if y_scale_compute_name != y_scale_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[y_scale_compute_name],
                outputs=[y_scale_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": str(ctx.get_tensor_dtype(y_scale_name)).upper(),
                },
            )
        )

    min_over_scale_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_min_over_scale",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[min_with_zero_name, y_scale_compute_name],
            outputs=[min_over_scale_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    zp_float_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_zp_float",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="NEG",
            inputs=[min_over_scale_name],
            outputs=[zp_float_name],
        )
    )
    zp_clip_lo_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_zp_clip_lo",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MAXIMUM",
            inputs=[zp_float_name, zero_f_name],
            outputs=[zp_clip_lo_name],
            options={},
        )
    )
    zp_clip_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_zp_clip",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MINIMUM",
            inputs=[zp_clip_lo_name, qmax_f_name],
            outputs=[zp_clip_name],
            options={},
        )
    )
    zp_rounded_f_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_zp_rounded_f32",
        dtype=compute_dtype,
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ROUND",
            inputs=[zp_clip_name],
            outputs=[zp_rounded_f_name],
        )
    )
    zp_i32_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_zp_i32",
        dtype="INT32",
        shape=scalar_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[zp_rounded_f_name],
            outputs=[zp_i32_name],
            options={
                "inDataType": compute_dtype,
                "outDataType": "INT32",
            },
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[zp_i32_name],
            outputs=[y_zero_name],
            options={
                "inDataType": "INT32",
                "outDataType": str(ctx.get_tensor_dtype(y_zero_name)).upper(),
            },
        )
    )

    x_div_scale_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_x_div_scale",
        dtype=compute_dtype,
        shape=x_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[x_compute_name, y_scale_compute_name],
            outputs=[x_div_scale_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    x_rounded_f_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_x_rounded_f32",
        dtype=compute_dtype,
        shape=x_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ROUND",
            inputs=[x_div_scale_name],
            outputs=[x_rounded_f_name],
        )
    )
    x_plus_zp_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_x_plus_zp",
        dtype=compute_dtype,
        shape=x_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[x_rounded_f_name, zp_rounded_f_name],
            outputs=[x_plus_zp_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    x_clip_lo_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_x_clip_lo",
        dtype=compute_dtype,
        shape=x_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MAXIMUM",
            inputs=[x_plus_zp_name, zero_f_name],
            outputs=[x_clip_lo_name],
            options={},
        )
    )
    x_clip_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_x_clip",
        dtype=compute_dtype,
        shape=x_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MINIMUM",
            inputs=[x_clip_lo_name, qmax_f_name],
            outputs=[x_clip_name],
            options={},
        )
    )
    x_i32_name = ctx.add_intermediate_tensor(
        f"{y_name}_dql_x_i32",
        dtype="INT32",
        shape=x_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[x_clip_name],
            outputs=[x_i32_name],
            options={
                "inDataType": compute_dtype,
                "outDataType": "INT32",
            },
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[x_i32_name],
            outputs=[y_name],
            options={
                "inDataType": "INT32",
                "outDataType": str(ctx.get_tensor_dtype(y_name)).upper(),
            },
        )
    )
