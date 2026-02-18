from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def build_fully_connected_from_gemm_or_matmul(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(weight_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    input_rank = len(input_shape)
    if node.op == "Gemm":
        if input_rank != 2:
            raise NotImplementedError(
                "Gemm conversion supports only rank-2 input. "
                f"op={node.name} input_shape={input_shape}"
            )
    else:
        if input_rank < 2:
            raise NotImplementedError(
                "FullyConnected conversion supports rank >= 2 input for MatMul/Einsum. "
                f"op={node.name} input_shape={input_shape}"
            )

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(f"FC weight must be constant. op={node.name}")
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise NotImplementedError(
            f"FC weight rank must be 2. op={node.name} weight_shape={weights.shape}"
        )

    if node.op == "Gemm":
        alpha = float(node.attrs.get("alpha", 1.0))
        beta = float(node.attrs.get("beta", 1.0))
        trans_a = int(node.attrs.get("transA", 0))
        trans_b = int(node.attrs.get("transB", 0))
        if trans_a != 0:
            raise NotImplementedError(f"Gemm transA=1 is not supported. op={node.name}")
        if trans_b == 0:
            fc_weights = weights.T
        else:
            fc_weights = weights
        if alpha != 1.0:
            fc_weights = fc_weights * alpha

        bias_values = None
        if len(node.inputs) >= 3:
            bias_values = ctx.get_constant_array(node.inputs[2].name)
        if bias_values is None:
            bias_values = np.zeros((fc_weights.shape[0],), dtype=np.float32)
        else:
            bias_values = np.asarray(bias_values, dtype=np.float32).reshape(-1)
        if beta != 1.0:
            bias_values = bias_values * beta
    else:
        fc_weights = weights.T
        bias_values = np.zeros((fc_weights.shape[0],), dtype=np.float32)

    w_name = ctx.add_const_tensor(
        f"{node.name}_fc_weights",
        np.asarray(fc_weights, dtype=np.float32),
    )
    b_name = ctx.add_const_tensor(
        f"{node.name}_fc_bias",
        np.asarray(bias_values, dtype=np.float32),
    )

    ctx.add_operator(
        OperatorIR(
            op_type="FULLY_CONNECTED",
            inputs=[input_name, w_name, b_name],
            outputs=[output_name],
            options={
                "fusedActivationFunction": "NONE",
                "weightsFormat": "DEFAULT",
                "keepNumDims": bool(node.op != "Gemm" and input_rank > 2),
                "asymmetricQuantizeInputs": False,
            },
        )
        )


def build_einsum_op(node: Any, ctx: Any) -> None:
    # _validate_einsum limits builtin lowering to rank-2 matmul-style equations.
    # Prefer FULLY_CONNECTED when RHS is constant, otherwise use BATCH_MATMUL.
    rhs_name = node.inputs[1].name
    if ctx.get_constant_array(rhs_name) is not None:
        build_fully_connected_from_gemm_or_matmul(node, ctx)
    else:
        build_matmul_op(node, ctx)


def build_matmul_op(node: Any, ctx: Any) -> None:
    a_name = node.inputs[0].name
    b_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(a_name)
    ctx.ensure_tensor(b_name)
    ctx.ensure_tensor(output_name)

    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()

    a_dtype = str(ctx.get_tensor_dtype(a_name)).upper()
    b_dtype = str(ctx.get_tensor_dtype(b_name)).upper()
    a_compute = a_name
    b_compute = b_name
    compute_dtype = "FLOAT32"

    if a_dtype != compute_dtype:
        a_compute = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_a_f32",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(a_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[a_name],
                outputs=[a_compute],
                options={"inDataType": a_dtype, "outDataType": compute_dtype},
            )
        )
    if b_dtype != compute_dtype:
        b_compute = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_b_f32",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(b_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[b_name],
                outputs=[b_compute],
                options={"inDataType": b_dtype, "outDataType": compute_dtype},
            )
        )

    matmul_out = output_name
    if output_dtype != compute_dtype:
        matmul_out = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_f32",
            dtype=compute_dtype,
            shape=output_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[a_compute, b_compute],
            outputs=[matmul_out],
            options={
                "adjX": False,
                "adjY": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    if matmul_out != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[matmul_out],
                outputs=[output_name],
                options={"inDataType": compute_dtype, "outDataType": output_dtype},
            )
        )


def build_fused_matmul_op(node: Any, ctx: Any) -> None:
    a_name = node.inputs[0].name
    b_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(a_name)
    ctx.ensure_tensor(b_name)
    ctx.ensure_tensor(output_name)

    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    alpha = float(node.attrs.get("alpha", 1.0))

    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT32"

    a_dtype = str(ctx.get_tensor_dtype(a_name)).upper()
    b_dtype = str(ctx.get_tensor_dtype(b_name)).upper()
    a_compute = a_name
    b_compute = b_name
    if a_dtype != compute_dtype:
        a_compute = ctx.add_intermediate_tensor(
            f"{output_name}_fusedmatmul_a_f32",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(a_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[a_name],
                outputs=[a_compute],
                options={"inDataType": a_dtype, "outDataType": compute_dtype},
            )
        )
    if b_dtype != compute_dtype:
        b_compute = ctx.add_intermediate_tensor(
            f"{output_name}_fusedmatmul_b_f32",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(b_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[b_name],
                outputs=[b_compute],
                options={"inDataType": b_dtype, "outDataType": compute_dtype},
            )
        )

    matmul_output_name = output_name
    if abs(alpha - 1.0) > 1e-12 or output_dtype != compute_dtype:
        matmul_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_fusedmatmul_matmul",
            dtype=compute_dtype,
            shape=output_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[a_compute, b_compute],
            outputs=[matmul_output_name],
            options={
                "adjX": bool(trans_a),
                "adjY": bool(trans_b),
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    scaled_output_name = matmul_output_name
    if abs(alpha - 1.0) > 1e-12:
        alpha_np_dtype = np.float32
        alpha_name = ctx.add_const_tensor(
            f"{output_name}_fusedmatmul_alpha",
            np.asarray(alpha, dtype=alpha_np_dtype),
        )
        scaled_output_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
            f"{output_name}_fusedmatmul_scaled",
            dtype=compute_dtype,
            shape=output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[matmul_output_name, alpha_name],
                outputs=[scaled_output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[scaled_output_name],
                outputs=[output_name],
                options={"inDataType": compute_dtype, "outDataType": output_dtype},
            )
        )


def build_matmul_integer_op(node: Any, ctx: Any) -> None:
    a_name = node.inputs[0].name
    b_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(a_name)
    ctx.ensure_tensor(b_name)
    ctx.ensure_tensor(output_name)

    a_shape = [int(v) for v in ctx.get_tensor_shape(a_name)]
    b_shape = [int(v) for v in ctx.get_tensor_shape(b_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]

    a_dtype = str(ctx.get_tensor_dtype(a_name)).upper()
    b_dtype = str(ctx.get_tensor_dtype(b_name)).upper()
    compute_dtype = (
        "INT16"
        if a_dtype in {"INT8", "UINT8", "INT16"} and b_dtype in {"INT8", "UINT8", "INT16"}
        else "FLOAT32"
    )
    matmul_output_dtype = "INT32" if compute_dtype == "INT16" else "FLOAT32"

    def _cast_to_dtype(src_name: str, hint: str, shape: list[int], target_dtype: str) -> str:
        src_dtype = str(ctx.get_tensor_dtype(src_name)).upper()
        if src_dtype == target_dtype:
            return src_name
        cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_{hint}_{str(target_dtype).lower()}",
            dtype=target_dtype,
            shape=shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[src_name],
                outputs=[cast_name],
                options={
                    "inDataType": src_dtype,
                    "outDataType": target_dtype,
                },
            )
        )
        return cast_name

    a_compute = _cast_to_dtype(a_name, "matmulinteger_a", a_shape, compute_dtype)
    b_compute = _cast_to_dtype(b_name, "matmulinteger_b", b_shape, compute_dtype)

    if len(node.inputs) >= 3:
        a_zp_name = node.inputs[2].name
        ctx.ensure_tensor(a_zp_name)
        a_zp_shape = [int(v) for v in ctx.get_tensor_shape(a_zp_name)]
        a_zp_compute = _cast_to_dtype(a_zp_name, "matmulinteger_a_zero_point", a_zp_shape, compute_dtype)
        if len(a_zp_shape) == 1 and int(a_zp_shape[0]) > 1:
            if len(a_shape) != 2:
                raise NotImplementedError(
                    "MatMulInteger with vector a_zero_point currently supports rank-2 A only "
                    "in flatbuffer_direct. "
                    f"op={node.name} a_shape={a_shape} a_zero_shape={a_zp_shape}"
                )
            a_zp_reshaped = ctx.add_intermediate_tensor(
                f"{output_name}_matmulinteger_a_zero_point_reshaped",
                dtype=compute_dtype,
                shape=[int(a_shape[0]), 1],
            )
            a_zp_shape_const = ctx.add_const_tensor(
                f"{output_name}_matmulinteger_a_zero_point_shape",
                np.asarray([int(a_shape[0]), 1], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[a_zp_i32, a_zp_shape_const],
                    outputs=[a_zp_reshaped],
                    options={"newShape": [int(a_shape[0]), 1]},
                )
            )
            a_zp_compute = a_zp_reshaped
        a_sub_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmulinteger_a_sub",
            dtype=compute_dtype,
            shape=a_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[a_compute, a_zp_compute],
                outputs=[a_sub_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        a_compute = a_sub_name

    if len(node.inputs) >= 4:
        b_zp_name = node.inputs[3].name
        ctx.ensure_tensor(b_zp_name)
        b_zp_shape = [int(v) for v in ctx.get_tensor_shape(b_zp_name)]
        b_zp_compute = _cast_to_dtype(b_zp_name, "matmulinteger_b_zero_point", b_zp_shape, compute_dtype)
        b_sub_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmulinteger_b_sub",
            dtype=compute_dtype,
            shape=b_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[b_compute, b_zp_compute],
                outputs=[b_sub_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        b_compute = b_sub_name

    matmul_output_name = output_name
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if output_dtype != matmul_output_dtype:
        matmul_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmulinteger_compute",
            dtype=matmul_output_dtype,
            shape=output_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[a_compute, b_compute],
            outputs=[matmul_output_name],
            options={
                "adjX": False,
                "adjY": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    if matmul_output_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[matmul_output_name],
                outputs=[output_name],
                options={
                    "inDataType": matmul_output_dtype,
                    "outDataType": output_dtype,
                },
            )
        )
