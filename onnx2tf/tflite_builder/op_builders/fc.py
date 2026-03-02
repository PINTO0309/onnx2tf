from __future__ import annotations

import os
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

    def _add_cast_if_needed(*, tensor_name: str, suffix: str, target_dtype: str) -> str:
        src_dtype = str(ctx.get_tensor_dtype(tensor_name)).upper()
        if src_dtype == str(target_dtype).upper():
            return tensor_name
        cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_{suffix}_{str(target_dtype).lower()}",
            dtype=str(target_dtype).upper(),
            shape=[int(v) for v in ctx.get_tensor_shape(tensor_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[tensor_name],
                outputs=[cast_name],
                options={"inDataType": src_dtype, "outDataType": str(target_dtype).upper()},
            )
        )
        return cast_name

    if node.op == "Gemm":
        alpha = float(node.attrs.get("alpha", 1.0))
        beta = float(node.attrs.get("beta", 1.0))
        trans_a = int(node.attrs.get("transA", 0))
        trans_b = int(node.attrs.get("transB", 0))
    else:
        alpha = 1.0
        beta = 1.0
        trans_a = 0
        trans_b = 1

    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT32"
    has_c_input = bool(node.op == "Gemm" and len(node.inputs) >= 3 and str(node.inputs[2].name) != "")
    c_name = node.inputs[2].name if has_c_input else ""

    weights_const = ctx.get_constant_array(weight_name)
    use_fully_connected = bool(weights_const is not None)
    if node.op == "Gemm":
        # FULLY_CONNECTED path requires non-transposed A and constant/omitted bias semantics.
        use_fully_connected = bool(use_fully_connected and trans_a == 0)
        if has_c_input and beta != 0.0:
            c_const = ctx.get_constant_array(c_name)
            if c_const is None:
                use_fully_connected = False
            else:
                c_units = int(np.asarray(c_const).size)
                expected_units = int(np.asarray(weights_const).shape[0] if trans_b == 1 else np.asarray(weights_const).shape[1])
                if c_units not in {1, expected_units}:
                    use_fully_connected = False

    if use_fully_connected:
        weights = np.asarray(weights_const, dtype=np.float32)
        if weights.ndim != 2:
            raise NotImplementedError(
                f"FC weight rank must be 2. op={node.name} weight_shape={weights.shape}"
            )

        if node.op == "Gemm":
            if trans_b == 0:
                fc_weights = weights.T
            else:
                fc_weights = weights
            if alpha != 1.0:
                fc_weights = fc_weights * alpha

            bias_values = None
            if has_c_input and beta != 0.0:
                bias_values = ctx.get_constant_array(c_name)
            if bias_values is None:
                bias_values = np.zeros((fc_weights.shape[0],), dtype=np.float32)
            else:
                bias_values = np.asarray(bias_values, dtype=np.float32).reshape(-1)
                if bias_values.size == 1 and int(fc_weights.shape[0]) > 1:
                    bias_values = np.full((int(fc_weights.shape[0]),), float(bias_values[0]), dtype=np.float32)
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
        return

    # Dynamic GEMM fallback: BATCH_MATMUL (+ alpha/beta/C) so Gemm can remain builtin.
    if node.op != "Gemm":
        raise NotImplementedError(f"FC weight must be constant. op={node.name}")

    a_compute = _add_cast_if_needed(tensor_name=input_name, suffix="gemm_a", target_dtype=compute_dtype)
    b_compute = _add_cast_if_needed(tensor_name=weight_name, suffix="gemm_b", target_dtype=compute_dtype)

    matmul_out = output_name
    requires_post_ops = bool(abs(alpha - 1.0) > 1e-12 or (has_c_input and abs(beta) > 1e-12) or output_dtype != compute_dtype)
    if requires_post_ops:
        matmul_out = ctx.add_intermediate_tensor(
            f"{output_name}_gemm_matmul",
            dtype=compute_dtype,
            shape=output_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[a_compute, b_compute],
            outputs=[matmul_out],
            options={
                "adjX": bool(trans_a),
                "adjY": bool(trans_b),
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    current_name = matmul_out
    if abs(alpha - 1.0) > 1e-12:
        alpha_name = ctx.add_const_tensor(
            f"{output_name}_gemm_alpha",
            np.asarray(alpha, dtype=np.float32),
        )
        scaled_name = output_name if (not has_c_input or abs(beta) <= 1e-12) and output_dtype == compute_dtype else ctx.add_intermediate_tensor(
            f"{output_name}_gemm_scaled",
            dtype=compute_dtype,
            shape=output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[current_name, alpha_name],
                outputs=[scaled_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        current_name = scaled_name

    if has_c_input and abs(beta) > 1e-12:
        c_compute = _add_cast_if_needed(tensor_name=c_name, suffix="gemm_c", target_dtype=compute_dtype)
        c_term_name = c_compute
        if abs(beta - 1.0) > 1e-12:
            beta_name = ctx.add_const_tensor(
                f"{output_name}_gemm_beta",
                np.asarray(beta, dtype=np.float32),
            )
            c_scaled = ctx.add_intermediate_tensor(
                f"{output_name}_gemm_c_scaled",
                dtype=compute_dtype,
                shape=[int(v) for v in ctx.get_tensor_shape(c_name)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MUL",
                    inputs=[c_compute, beta_name],
                    outputs=[c_scaled],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            c_term_name = c_scaled

        add_out = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
            f"{output_name}_gemm_bias_added",
            dtype=compute_dtype,
            shape=output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[current_name, c_term_name],
                outputs=[add_out],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        current_name = add_out

    if current_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[current_name],
                outputs=[output_name],
                options={"inDataType": compute_dtype, "outDataType": output_dtype},
            )
        )


def build_einsum_op(node: Any, ctx: Any) -> None:
    equation = str(node.attrs.get("equation", "")).replace(" ", "")
    # Specialized rank-4 contraction:
    #   abgd,gf->abdf
    if equation != "":
        try:
            lhs, rhs_out = equation.split(",", 1)
            rhs, out = rhs_out.split("->", 1)
        except Exception:
            lhs, rhs, out = "", "", ""
        if (
            len(lhs) == 4
            and len(rhs) == 2
            and len(out) == 4
            and lhs[2] == rhs[0]
            and out[0] == lhs[0]
            and out[1] == lhs[1]
            and out[2] == lhs[3]
            and out[3] == rhs[1]
        ):
            a_name = node.inputs[0].name
            b_name = node.inputs[1].name
            output_name = node.outputs[0].name
            ctx.ensure_tensor(a_name)
            ctx.ensure_tensor(b_name)
            ctx.ensure_tensor(output_name)

            a_shape = [int(v) for v in ctx.get_tensor_shape(a_name)]
            b_shape = [int(v) for v in ctx.get_tensor_shape(b_name)]
            output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
            output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
            compute_dtype = "FLOAT32"

            def _cast_to_compute(src_name: str, suffix: str) -> str:
                src_dtype = str(ctx.get_tensor_dtype(src_name)).upper()
                if src_dtype == compute_dtype:
                    return src_name
                cast_name = ctx.add_intermediate_tensor(
                    f"{output_name}_{suffix}_f32",
                    dtype=compute_dtype,
                    shape=[int(v) for v in ctx.get_tensor_shape(src_name)],
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[src_name],
                        outputs=[cast_name],
                        options={"inDataType": src_dtype, "outDataType": compute_dtype},
                    )
                )
                return cast_name

            a_compute = _cast_to_compute(a_name, "einsum_a")
            b_compute = _cast_to_compute(b_name, "einsum_b")

            perm_name = ctx.add_const_tensor(
                f"{output_name}_einsum_perm",
                np.asarray([0, 1, 3, 2], dtype=np.int32),
            )
            a_transposed_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_a_transposed",
                dtype=compute_dtype,
                shape=[int(a_shape[0]), int(a_shape[1]), int(a_shape[3]), int(a_shape[2])],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[a_compute, perm_name],
                    outputs=[a_transposed_name],
                    options={},
                )
            )

            flattened_rows = int(a_shape[0]) * int(a_shape[1]) * int(a_shape[3])
            lhs_k = int(a_shape[2])
            rhs_n = int(b_shape[1])

            a2_shape_name = ctx.add_const_tensor(
                f"{output_name}_einsum_a2_shape",
                np.asarray([flattened_rows, lhs_k], dtype=np.int32),
            )
            a2_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_a2",
                dtype=compute_dtype,
                shape=[flattened_rows, lhs_k],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[a_transposed_name, a2_shape_name],
                    outputs=[a2_name],
                    options={"newShape": [flattened_rows, lhs_k]},
                )
            )

            matmul_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_matmul",
                dtype=compute_dtype,
                shape=[flattened_rows, rhs_n],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a2_name, b_compute],
                    outputs=[matmul_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )

            y4_shape_name = ctx.add_const_tensor(
                f"{output_name}_einsum_out_shape",
                np.asarray(output_shape, dtype=np.int32),
            )
            y4_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
                f"{output_name}_einsum_out_f32",
                dtype=compute_dtype,
                shape=output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[matmul_name, y4_shape_name],
                    outputs=[y4_name],
                    options={"newShape": [int(v) for v in output_shape]},
                )
            )

            if y4_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[y4_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

    # _validate_einsum limits remaining builtin lowering to rank-2 matmul-style equations.
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
    # NOTE:
    # Keep MatMulInteger lowering on FLOAT32 by default. INT16 lowering builds
    # SUB(B, zero_point) style integer pre-processing chains that can trigger
    # LiteRT prepare-time aborts on some models (e.g. bertsquad-12-int8) when
    # quantized SUB scale constraints are not satisfiable.
    # For controlled experiments, the legacy INT16 path can be re-enabled via:
    #   ONNX2TF_MATMULINTEGER_COMPUTE_DTYPE=INT16
    requested_compute_dtype = str(
        os.environ.get("ONNX2TF_MATMULINTEGER_COMPUTE_DTYPE", "FLOAT32")
    ).strip().upper()
    use_int16_compute = (
        requested_compute_dtype == "INT16"
        and a_dtype in {"INT8", "UINT8", "INT16"}
        and b_dtype in {"INT8", "UINT8", "INT16"}
    )
    compute_dtype = "INT16" if use_int16_compute else "FLOAT32"
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
                    inputs=[a_zp_compute, a_zp_shape_const],
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
