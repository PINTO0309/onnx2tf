from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def _einsum_has_duplicate_labels(term: str) -> bool:
    return len(term) != len(set(term))


def _prod(values: List[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _try_build_generic_two_input_einsum(
    node: Any,
    ctx: Any,
    lhs: str,
    rhs: str,
    out: str,
) -> bool:
    if out == "":
        return False
    if _einsum_has_duplicate_labels(lhs) or _einsum_has_duplicate_labels(rhs) or _einsum_has_duplicate_labels(out):
        return False

    # Keep the existing rank-2 matmul/fully-connected path for performance.
    if len(lhs) == 2 and len(rhs) == 2 and len(out) == 2:
        return False

    out_set = set(out)
    lhs_set = set(lhs)
    rhs_set = set(rhs)
    if not out_set.issubset(lhs_set.union(rhs_set)):
        return False

    a_name = node.inputs[0].name
    b_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(a_name)
    ctx.ensure_tensor(b_name)
    ctx.ensure_tensor(output_name)

    a_shape = [int(v) for v in ctx.get_tensor_shape(a_name)]
    b_shape = [int(v) for v in ctx.get_tensor_shape(b_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    if len(a_shape) != len(lhs) or len(b_shape) != len(rhs) or len(output_shape) != len(out):
        return False

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

    def _reduce_labels(
        src_name: str,
        src_labels: List[str],
        reduce_labels: List[str],
        suffix: str,
    ) -> tuple[str, List[str], List[int]]:
        src_shape = [int(v) for v in ctx.get_tensor_shape(src_name)]
        if len(reduce_labels) == 0:
            return src_name, list(src_labels), src_shape
        axes = [int(idx) for idx, label in enumerate(src_labels) if label in set(reduce_labels)]
        if len(axes) == 0:
            return src_name, list(src_labels), src_shape
        dst_labels = [label for idx, label in enumerate(src_labels) if int(idx) not in set(axes)]
        dst_shape = [int(dim) for idx, dim in enumerate(src_shape) if int(idx) not in set(axes)]
        if len(dst_shape) == 0:
            dst_shape = [1]
        axes_name = ctx.add_const_tensor(
            f"{output_name}_{suffix}_axes",
            np.asarray(axes, dtype=np.int32),
        )
        dst_name = ctx.add_intermediate_tensor(
            f"{output_name}_{suffix}",
            dtype=compute_dtype,
            shape=dst_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUM",
                inputs=[src_name, axes_name],
                outputs=[dst_name],
                options={"keepDims": False},
            )
        )
        return dst_name, dst_labels, dst_shape

    def _transpose_to_labels(
        src_name: str,
        src_labels: List[str],
        dst_labels: List[str],
        suffix: str,
    ) -> tuple[str, List[str], List[int]]:
        src_shape = [int(v) for v in ctx.get_tensor_shape(src_name)]
        if src_labels == dst_labels:
            return src_name, list(src_labels), src_shape
        if len(src_labels) == 0 and len(dst_labels) == 0:
            return src_name, [], src_shape
        if set(src_labels) != set(dst_labels):
            return src_name, list(src_labels), src_shape
        perm = [int(src_labels.index(label)) for label in dst_labels]
        if len(perm) <= 1:
            return src_name, list(src_labels), src_shape
        perm_name = ctx.add_const_tensor(
            f"{output_name}_{suffix}_perm",
            np.asarray(perm, dtype=np.int32),
        )
        dst_shape = [int(src_shape[idx]) for idx in perm]
        dst_name = ctx.add_intermediate_tensor(
            f"{output_name}_{suffix}",
            dtype=compute_dtype,
            shape=dst_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[src_name, perm_name],
                outputs=[dst_name],
                options={},
            )
        )
        return dst_name, list(dst_labels), dst_shape

    def _reshape_tensor(src_name: str, dst_shape: List[int], suffix: str) -> str:
        shape = [int(v) for v in dst_shape] if len(dst_shape) > 0 else [1]
        shape_name = ctx.add_const_tensor(
            f"{output_name}_{suffix}_shape",
            np.asarray(shape, dtype=np.int32),
        )
        dst_name = ctx.add_intermediate_tensor(
            f"{output_name}_{suffix}",
            dtype=compute_dtype,
            shape=shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[src_name, shape_name],
                outputs=[dst_name],
                options={"newShape": [int(v) for v in shape]},
            )
        )
        return dst_name

    a_cur = _cast_to_compute(a_name, "einsum_a")
    b_cur = _cast_to_compute(b_name, "einsum_b")
    a_labels = [str(v) for v in lhs]
    b_labels = [str(v) for v in rhs]

    # First reduce labels that exist only in one operand and are absent from output.
    a_unique_reduce = [label for label in a_labels if label not in set(b_labels) and label not in out_set]
    b_unique_reduce = [label for label in b_labels if label not in set(a_labels) and label not in out_set]
    a_cur, a_labels, _ = _reduce_labels(a_cur, a_labels, a_unique_reduce, "einsum_a_reduce")
    b_cur, b_labels, _ = _reduce_labels(b_cur, b_labels, b_unique_reduce, "einsum_b_reduce")

    shared = set(a_labels).intersection(set(b_labels))
    contract_labels = [label for label in a_labels if label in shared and label not in out_set]
    batch_labels = [label for label in out if label in shared]
    lhs_free_labels = [label for label in a_labels if label not in shared]
    rhs_free_labels = [label for label in b_labels if label not in shared]

    if any(label not in out_set for label in lhs_free_labels):
        return False
    if any(label not in out_set for label in rhs_free_labels):
        return False

    expected_out_labels = set(batch_labels + lhs_free_labels + rhs_free_labels)
    if expected_out_labels != out_set:
        return False

    target_a_labels = batch_labels + lhs_free_labels + contract_labels
    target_b_labels = batch_labels + contract_labels + rhs_free_labels
    a_cur, a_labels, _ = _transpose_to_labels(a_cur, a_labels, target_a_labels, "einsum_a_perm")
    b_cur, b_labels, _ = _transpose_to_labels(b_cur, b_labels, target_b_labels, "einsum_b_perm")
    if a_labels != target_a_labels or b_labels != target_b_labels:
        return False

    a_shape = [int(v) for v in ctx.get_tensor_shape(a_cur)]
    b_shape = [int(v) for v in ctx.get_tensor_shape(b_cur)]
    if not (len(a_labels) == len(a_shape) or (len(a_labels) == 0 and a_shape == [1])):
        return False
    if not (len(b_labels) == len(b_shape) or (len(b_labels) == 0 and b_shape == [1])):
        return False

    a_dims_by_label: Dict[str, int] = {}
    for idx, label in enumerate(a_labels):
        a_dims_by_label[label] = int(a_shape[idx])
    b_dims_by_label: Dict[str, int] = {}
    for idx, label in enumerate(b_labels):
        b_dims_by_label[label] = int(b_shape[idx])

    for label in shared:
        if int(a_dims_by_label.get(label, 1)) != int(b_dims_by_label.get(label, 1)):
            return False

    batch_dims = [int(a_dims_by_label[label]) for label in batch_labels]
    lhs_free_dims = [int(a_dims_by_label[label]) for label in lhs_free_labels]
    contract_dims = [int(a_dims_by_label[label]) for label in contract_labels]
    rhs_free_dims = [int(b_dims_by_label[label]) for label in rhs_free_labels]

    m_dim = int(_prod(lhs_free_dims) if len(lhs_free_dims) > 0 else 1)
    k_dim = int(_prod(contract_dims) if len(contract_dims) > 0 else 1)
    n_dim = int(_prod(rhs_free_dims) if len(rhs_free_dims) > 0 else 1)

    a_mat_shape = [int(v) for v in batch_dims] + [m_dim, k_dim]
    b_mat_shape = [int(v) for v in batch_dims] + [k_dim, n_dim]
    a_mat = _reshape_tensor(a_cur, a_mat_shape, "einsum_a_mat")
    b_mat = _reshape_tensor(b_cur, b_mat_shape, "einsum_b_mat")

    bmm_shape = [int(v) for v in batch_dims] + [m_dim, n_dim]
    bmm_name = ctx.add_intermediate_tensor(
        f"{output_name}_einsum_bmm",
        dtype=compute_dtype,
        shape=bmm_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[a_mat, b_mat],
            outputs=[bmm_name],
            options={
                "adjX": False,
                "adjY": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    pre_out_labels = batch_labels + lhs_free_labels + rhs_free_labels
    pre_out_shape = [int(v) for v in batch_dims + lhs_free_dims + rhs_free_dims]
    if len(pre_out_shape) == 0:
        pre_out_shape = [1]
    pre_out_name = _reshape_tensor(bmm_name, pre_out_shape, "einsum_pre_out")

    if pre_out_labels != [str(v) for v in out]:
        if set(pre_out_labels) != out_set:
            return False
        perm = [int(pre_out_labels.index(label)) for label in out]
        perm_name = ctx.add_const_tensor(
            f"{output_name}_einsum_out_perm",
            np.asarray(perm, dtype=np.int32),
        )
        y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
            f"{output_name}_einsum_out_f32",
            dtype=compute_dtype,
            shape=output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[pre_out_name, perm_name],
                outputs=[y_name],
                options={},
            )
        )
    else:
        y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
            f"{output_name}_einsum_out_f32",
            dtype=compute_dtype,
            shape=output_shape,
        )
        out_shape_name = ctx.add_const_tensor(
            f"{output_name}_einsum_out_shape",
            np.asarray(output_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[pre_out_name, out_shape_name],
                outputs=[y_name],
                options={"newShape": [int(v) for v in output_shape]},
            )
        )

    if y_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[y_name],
                outputs=[output_name],
                options={"inDataType": compute_dtype, "outDataType": output_dtype},
            )
        )
    return True


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
    input_terms: list[str] = []
    out = ""
    if equation != "":
        try:
            input_expr, out = equation.split("->", 1)
            input_terms = [str(v) for v in input_expr.split(",") if str(v) != ""]
        except Exception:
            input_terms = []
            out = ""

    # Specialized rank-4 attention weighted sum:
    #   nlhd,nhdv,nlh->nlhv
    # implemented as:
    #   transpose(nlhd -> nhld) + batch_matmul(nhld, nhdv) + transpose + mul(expand(nlh))
    if len(input_terms) == 3 and len(out) == 4:
        lhs = input_terms[0]
        rhs = input_terms[1]
        scale = input_terms[2]
        if (
            len(lhs) == 4
            and len(rhs) == 4
            and len(scale) == 3
            and lhs[0] == rhs[0] == scale[0] == out[0]
            and lhs[1] == scale[1] == out[1]
            and lhs[2] == rhs[1] == scale[2] == out[2]
            and lhs[3] == rhs[2]
            and rhs[3] == out[3]
            and lhs[3] not in out
        ):
            a_name = node.inputs[0].name
            b_name = node.inputs[1].name
            c_name = node.inputs[2].name
            output_name = node.outputs[0].name
            ctx.ensure_tensor(a_name)
            ctx.ensure_tensor(b_name)
            ctx.ensure_tensor(c_name)
            ctx.ensure_tensor(output_name)

            a_shape = [int(v) for v in ctx.get_tensor_shape(a_name)]
            b_shape = [int(v) for v in ctx.get_tensor_shape(b_name)]
            c_shape = [int(v) for v in ctx.get_tensor_shape(c_name)]
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
            c_compute = _cast_to_compute(c_name, "einsum_c")

            # nlhd -> nhld
            a_perm_name = ctx.add_const_tensor(
                f"{output_name}_einsum_a_perm",
                np.asarray([0, 2, 1, 3], dtype=np.int32),
            )
            a_nhld_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_a_nhld",
                dtype=compute_dtype,
                shape=[int(a_shape[0]), int(a_shape[2]), int(a_shape[1]), int(a_shape[3])],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[a_compute, a_perm_name],
                    outputs=[a_nhld_name],
                    options={},
                )
            )

            # [n,h,l,d] @ [n,h,d,v] -> [n,h,l,v]
            bmm_out_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_nhlv",
                dtype=compute_dtype,
                shape=[int(a_shape[0]), int(a_shape[2]), int(a_shape[1]), int(b_shape[3])],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_nhld_name, b_compute],
                    outputs=[bmm_out_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )

            # nhlv -> nlhv
            out_perm_name = ctx.add_const_tensor(
                f"{output_name}_einsum_out_perm",
                np.asarray([0, 2, 1, 3], dtype=np.int32),
            )
            bmm_transposed_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_nlhv",
                dtype=compute_dtype,
                shape=[int(a_shape[0]), int(a_shape[1]), int(a_shape[2]), int(b_shape[3])],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[bmm_out_name, out_perm_name],
                    outputs=[bmm_transposed_name],
                    options={},
                )
            )

            # nlh -> nlh1
            axis_name = ctx.add_const_tensor(
                f"{output_name}_einsum_scale_axis",
                np.asarray([3], dtype=np.int32),
            )
            c_expanded_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_scale_expanded",
                dtype=compute_dtype,
                shape=[int(c_shape[0]), int(c_shape[1]), int(c_shape[2]), 1],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="EXPAND_DIMS",
                    inputs=[c_compute, axis_name],
                    outputs=[c_expanded_name],
                )
            )

            y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
                f"{output_name}_einsum_out_f32",
                dtype=compute_dtype,
                shape=output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MUL",
                    inputs=[bmm_transposed_name, c_expanded_name],
                    outputs=[y_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )

            if y_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[y_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

    # Specialized rank-4 contraction:
    #   abgd,gf->abdf
    if len(input_terms) == 2:
        try:
            lhs = input_terms[0]
            rhs = input_terms[1]
        except Exception:
            lhs, rhs, out = "", "", ""
        if (
            len(lhs) == 4
            and len(rhs) == 4
            and len(out) == 4
            and lhs[0] == rhs[0] == out[0]
            and lhs[1] == rhs[1] == out[1]
            and lhs[2] == out[2]
            and rhs[2] == out[3]
            and lhs[3] == rhs[3]
            and lhs[3] not in out
        ):
            a_name = node.inputs[0].name
            b_name = node.inputs[1].name
            output_name = node.outputs[0].name
            ctx.ensure_tensor(a_name)
            ctx.ensure_tensor(b_name)
            ctx.ensure_tensor(output_name)

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
            y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
                f"{output_name}_einsum_out_f32",
                dtype=compute_dtype,
                shape=output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_compute, b_compute],
                    outputs=[y_name],
                    options={
                        "adjX": False,
                        "adjY": True,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )
            if y_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[y_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

        if (
            len(lhs) == 4
            and len(rhs) == 4
            and len(out) == 4
            and lhs[0] == rhs[0] == out[0]
            and lhs[1] == rhs[1] == out[1]
            and lhs[2] == out[2]
            and lhs[3] == rhs[2]
            and rhs[3] == out[3]
            and lhs[3] not in out
        ):
            a_name = node.inputs[0].name
            b_name = node.inputs[1].name
            output_name = node.outputs[0].name
            ctx.ensure_tensor(a_name)
            ctx.ensure_tensor(b_name)
            ctx.ensure_tensor(output_name)

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
            y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
                f"{output_name}_einsum_out_f32",
                dtype=compute_dtype,
                shape=output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_compute, b_compute],
                    outputs=[y_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )
            if y_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[y_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

        if (
            len(lhs) == 4
            and len(rhs) == 4
            and len(out) == 4
            and lhs[0] == rhs[0] == out[0]
            and lhs[1] == rhs[1] == out[1]
            and lhs[2] == rhs[2]
            and lhs[3] == out[2]
            and rhs[3] == out[3]
            and lhs[2] not in out
        ):
            a_name = node.inputs[0].name
            b_name = node.inputs[1].name
            output_name = node.outputs[0].name
            ctx.ensure_tensor(a_name)
            ctx.ensure_tensor(b_name)
            ctx.ensure_tensor(output_name)

            a_shape = [int(v) for v in ctx.get_tensor_shape(a_name)]
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

            y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
                f"{output_name}_einsum_out_f32",
                dtype=compute_dtype,
                shape=output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_transposed_name, b_compute],
                    outputs=[y_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )
            if y_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[y_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

        if (
            len(lhs) == 3
            and len(rhs) == 3
            and len(out) == 3
            and lhs[0] == rhs[0] == out[0]
            and lhs[1] == out[1]
            and rhs[1] == out[2]
            and lhs[2] == rhs[2]
            and lhs[2] not in out
        ):
            a_name = node.inputs[0].name
            b_name = node.inputs[1].name
            output_name = node.outputs[0].name
            ctx.ensure_tensor(a_name)
            ctx.ensure_tensor(b_name)
            ctx.ensure_tensor(output_name)

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
            y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
                f"{output_name}_einsum_out_f32",
                dtype=compute_dtype,
                shape=output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_compute, b_compute],
                    outputs=[y_name],
                    options={
                        "adjX": False,
                        "adjY": True,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )
            if y_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[y_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

        if (
            len(lhs) == 4
            and len(rhs) == 3
            and len(out) == 4
            and lhs[0] == rhs[0] == out[0]
            and lhs[1] == rhs[2]
            and rhs[1] == out[1]
            and lhs[2] == out[2]
            and lhs[3] == out[3]
            and lhs[1] not in out
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

            # bchw -> bhwc
            a_perm_name = ctx.add_const_tensor(
                f"{output_name}_einsum_a_perm",
                np.asarray([0, 2, 3, 1], dtype=np.int32),
            )
            a_bhwc_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_a_bhwc",
                dtype=compute_dtype,
                shape=[int(a_shape[0]), int(a_shape[2]), int(a_shape[3]), int(a_shape[1])],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[a_compute, a_perm_name],
                    outputs=[a_bhwc_name],
                    options={},
                )
            )

            # [b,h,w,c] -> [b,h*w,c]
            a_bhwc_3d_shape = [
                int(a_shape[0]),
                int(a_shape[2]) * int(a_shape[3]),
                int(a_shape[1]),
            ]
            a_reshape_shape_name = ctx.add_const_tensor(
                f"{output_name}_einsum_a_reshape_shape",
                np.asarray(a_bhwc_3d_shape, dtype=np.int32),
            )
            a_bhwc_3d_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_a_bhwc_3d",
                dtype=compute_dtype,
                shape=a_bhwc_3d_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[a_bhwc_name, a_reshape_shape_name],
                    outputs=[a_bhwc_3d_name],
                    options={"newShape": [int(v) for v in a_bhwc_3d_shape]},
                )
            )

            # bnc -> bcn
            b_perm_name = ctx.add_const_tensor(
                f"{output_name}_einsum_b_perm",
                np.asarray([0, 2, 1], dtype=np.int32),
            )
            b_bcn_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_b_bcn",
                dtype=compute_dtype,
                shape=[int(b_shape[0]), int(b_shape[2]), int(b_shape[1])],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[b_compute, b_perm_name],
                    outputs=[b_bcn_name],
                    options={},
                )
            )

            # [b,h*w,c] @ [b,c,n] -> [b,h*w,n]
            bmm_out_shape = [int(a_shape[0]), int(a_shape[2]) * int(a_shape[3]), int(b_shape[1])]
            bmm_out_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_bmm",
                dtype=compute_dtype,
                shape=bmm_out_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_bhwc_3d_name, b_bcn_name],
                    outputs=[bmm_out_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )

            # [b,h*w,n] -> [b,h,w,n]
            bhwn_shape = [
                int(output_shape[0]),
                int(output_shape[2]),
                int(output_shape[3]),
                int(output_shape[1]),
            ]
            bhwn_shape_name = ctx.add_const_tensor(
                f"{output_name}_einsum_bhwn_shape",
                np.asarray(bhwn_shape, dtype=np.int32),
            )
            bhwn_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_bhwn",
                dtype=compute_dtype,
                shape=bhwn_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[bmm_out_name, bhwn_shape_name],
                    outputs=[bhwn_name],
                    options={"newShape": [int(v) for v in bhwn_shape]},
                )
            )

            # bhwn -> bnhw
            out_perm_name = ctx.add_const_tensor(
                f"{output_name}_einsum_out_perm",
                np.asarray([0, 3, 1, 2], dtype=np.int32),
            )
            y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
                f"{output_name}_einsum_out_f32",
                dtype=compute_dtype,
                shape=output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[bhwn_name, out_perm_name],
                    outputs=[y_name],
                    options={},
                )
            )

            if y_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[y_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

        if (
            len(lhs) == 4
            and len(rhs) == 4
            and len(out) == 4
            and lhs[0] == rhs[0]
            and lhs[1] == rhs[1]
            and lhs[2] == rhs[2]
            and out[0] == lhs[0]
            and out[1] == lhs[2]
            and out[2] == lhs[3]
            and out[3] == rhs[3]
            and lhs[1] not in out
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

            a_perm_name = ctx.add_const_tensor(
                f"{output_name}_einsum_a_perm",
                np.asarray([0, 2, 3, 1], dtype=np.int32),
            )
            a_transposed_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_a_transposed",
                dtype=compute_dtype,
                shape=[int(a_shape[0]), int(a_shape[2]), int(a_shape[3]), int(a_shape[1])],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[a_compute, a_perm_name],
                    outputs=[a_transposed_name],
                    options={},
                )
            )

            b_perm_name = ctx.add_const_tensor(
                f"{output_name}_einsum_b_perm",
                np.asarray([0, 2, 1, 3], dtype=np.int32),
            )
            b_transposed_name = ctx.add_intermediate_tensor(
                f"{output_name}_einsum_b_transposed",
                dtype=compute_dtype,
                shape=[int(b_shape[0]), int(b_shape[2]), int(b_shape[1]), int(b_shape[3])],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[b_compute, b_perm_name],
                    outputs=[b_transposed_name],
                    options={},
                )
            )

            y_name = output_name if output_dtype == compute_dtype else ctx.add_intermediate_tensor(
                f"{output_name}_einsum_out_f32",
                dtype=compute_dtype,
                shape=output_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_transposed_name, b_transposed_name],
                    outputs=[y_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )

            if y_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[y_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

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

        if _try_build_generic_two_input_einsum(
            node=node,
            ctx=ctx,
            lhs=str(lhs),
            rhs=str(rhs),
            out=str(out),
        ):
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

    a_shape = [int(v) for v in ctx.get_tensor_shape(a_name)]
    b_shape = [int(v) for v in ctx.get_tensor_shape(b_name)]
    a_tensor = ctx.model_ir.tensors.get(a_name, None)
    b_tensor = ctx.model_ir.tensors.get(b_name, None)

    a_signature = (
        [int(v) for v in list(a_tensor.shape_signature)]
        if a_tensor is not None and a_tensor.shape_signature is not None
        else [int(v) for v in list(a_shape)]
    )
    b_signature = (
        [int(v) for v in list(b_tensor.shape_signature)]
        if b_tensor is not None and b_tensor.shape_signature is not None
        else [int(v) for v in list(b_shape)]
    )

    def _materialize_shape_from_signature(shape_signature: List[int]) -> List[int]:
        return [int(v) if int(v) > 0 else 1 for v in list(shape_signature)]

    def _update_output_tensor_shape(shape_signature: List[int]) -> List[int]:
        runtime_shape = _materialize_shape_from_signature(shape_signature)
        output_tensor = ctx.model_ir.tensors.get(output_name, None)
        if output_tensor is not None:
            output_tensor.shape = [int(v) for v in list(runtime_shape)]
            output_tensor.shape_signature = [int(v) for v in list(shape_signature)]
        return runtime_shape

    def _is_unresolved_rank1_placeholder(shape: List[int], signature: List[int]) -> bool:
        return (
            len(shape) == 1
            and len(signature) == 1
            and int(shape[0]) == 1
            and int(signature[0]) < 0
        )
    if len(a_shape) == 0 or len(b_shape) == 0:
        a_const = ctx.get_constant_array(a_name)
        b_const = ctx.get_constant_array(b_name)
        a_is_const_scalar = bool(a_const is not None and np.asarray(a_const).ndim == 0)
        b_is_const_scalar = bool(b_const is not None and np.asarray(b_const).ndim == 0)
        # Metadata can incorrectly collapse dynamic MatMul inputs to scalar.
        # Use MUL shortcut only for true constant scalar-scalar cases.
        if a_is_const_scalar and b_is_const_scalar:
            mul_out = output_name
            if output_dtype != compute_dtype:
                mul_out = ctx.add_intermediate_tensor(
                    f"{output_name}_matmul_scalar_mul",
                    dtype=compute_dtype,
                    shape=output_shape,
                )
            ctx.add_operator(
                OperatorIR(
                    op_type="MUL",
                    inputs=[a_compute, b_compute],
                    outputs=[mul_out],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            if mul_out != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[mul_out],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

    if (
        len(a_shape) == 1
        and len(b_shape) == 1
        and not _is_unresolved_rank1_placeholder(a_shape, a_signature)
        and not _is_unresolved_rank1_placeholder(b_shape, b_signature)
    ):
        output_shape_signature = [1]
        output_shape = _update_output_tensor_shape(output_shape_signature)
        # Keep low-rank dot-product reshapes resilient to downstream shape rewrites.
        a_k_dim = int(a_shape[0]) if len(a_shape) > 0 and int(a_shape[0]) > 0 else -1
        b_k_dim = int(b_shape[0]) if len(b_shape) > 0 and int(b_shape[0]) > 0 else -1
        a_matrix_new_shape = [1, int(a_k_dim) if int(a_k_dim) > 0 else -1]
        b_matrix_new_shape = [int(b_k_dim) if int(b_k_dim) > 0 else -1, 1]
        a_matrix_shape_name = ctx.add_const_tensor(
            f"{output_name}_matmul_a_dot_shape",
            np.asarray(a_matrix_new_shape, dtype=np.int32),
        )
        b_matrix_shape_name = ctx.add_const_tensor(
            f"{output_name}_matmul_b_dot_shape",
            np.asarray(b_matrix_new_shape, dtype=np.int32),
        )
        a_matrix_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_a_dot_matrix",
            dtype=compute_dtype,
            shape=[int(v) if int(v) > 0 else 1 for v in a_matrix_new_shape],
        )
        b_matrix_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_b_dot_matrix",
            dtype=compute_dtype,
            shape=[int(v) if int(v) > 0 else 1 for v in b_matrix_new_shape],
        )
        a_matrix_tensor = ctx.model_ir.tensors.get(a_matrix_name, None)
        b_matrix_tensor = ctx.model_ir.tensors.get(b_matrix_name, None)
        if a_matrix_tensor is not None:
            a_matrix_tensor.shape_signature = [int(v) for v in a_matrix_new_shape]
        if b_matrix_tensor is not None:
            b_matrix_tensor.shape_signature = [int(v) for v in b_matrix_new_shape]
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[a_compute, a_matrix_shape_name],
                outputs=[a_matrix_name],
                options={
                    "newShape": [int(v) for v in a_matrix_new_shape],
                    "preserveDynamicShape": True,
                },
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[b_compute, b_matrix_shape_name],
                outputs=[b_matrix_name],
                options={
                    "newShape": [int(v) for v in b_matrix_new_shape],
                    "preserveDynamicShape": True,
                },
            )
        )

        dot_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_dot_out",
            dtype=compute_dtype,
            shape=[1, 1],
        )
        dot_tensor = ctx.model_ir.tensors.get(dot_name, None)
        if dot_tensor is not None:
            dot_tensor.shape_signature = [1, 1]
        ctx.add_operator(
            OperatorIR(
                op_type="BATCH_MATMUL",
                inputs=[a_matrix_name, b_matrix_name],
                outputs=[dot_name],
                options={
                    "adjX": False,
                    "adjY": False,
                    "asymmetricQuantizeInputs": False,
                },
            )
        )

        squeeze_out = output_name
        if output_dtype != compute_dtype:
            squeeze_out = ctx.add_intermediate_tensor(
                f"{output_name}_matmul_dot_squeezed",
                dtype=compute_dtype,
                shape=output_shape,
            )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[dot_name],
                outputs=[squeeze_out],
                options={"squeezeDims": [0, 1]},
            )
        )
        if squeeze_out != output_name:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[squeeze_out],
                    outputs=[output_name],
                    options={"inDataType": compute_dtype, "outDataType": output_dtype},
                )
            )
        return

    if len(a_shape) == 1 and len(b_shape) >= 2:
        if _is_unresolved_rank1_placeholder(a_shape, a_signature) and len(b_shape) == 2:
            k_dim = int(b_shape[0]) if int(b_shape[0]) > 0 else (
                int(b_signature[0]) if len(b_signature) > 0 and int(b_signature[0]) > 0 else -1
            )
            n_dim = int(b_shape[1]) if int(b_shape[1]) > 0 else 1
            n_sig = int(b_signature[1]) if len(b_signature) > 1 and int(b_signature[1]) > 0 else int(n_dim)

            a_matrix_new_shape = [1, -1, int(k_dim) if int(k_dim) > 0 else -1]
            a_matrix_shape_name = ctx.add_const_tensor(
                f"{output_name}_matmul_a_unknown_shape",
                np.asarray(a_matrix_new_shape, dtype=np.int32),
            )
            a_matrix_name = ctx.add_intermediate_tensor(
                f"{output_name}_matmul_a_unknown_matrix",
                dtype=compute_dtype,
                shape=[int(v) if int(v) > 0 else 1 for v in a_matrix_new_shape],
            )
            a_matrix_tensor = ctx.model_ir.tensors.get(a_matrix_name, None)
            if a_matrix_tensor is not None:
                a_matrix_tensor.shape_signature = [int(v) for v in a_matrix_new_shape]
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[a_compute, a_matrix_shape_name],
                    outputs=[a_matrix_name],
                    options={
                        "newShape": [int(v) for v in a_matrix_new_shape],
                        "preserveDynamicShape": True,
                    },
                )
            )

            output_shape_signature = [1, -1, int(n_sig)]
            output_shape = _update_output_tensor_shape(output_shape_signature)
            matmul_out_name = output_name
            if output_dtype != compute_dtype:
                matmul_out_name = ctx.add_intermediate_tensor(
                    f"{output_name}_matmul_unknown_f32",
                    dtype=compute_dtype,
                    shape=[int(v) for v in list(output_shape)],
                )
                matmul_out_tensor = ctx.model_ir.tensors.get(matmul_out_name, None)
                if matmul_out_tensor is not None:
                    matmul_out_tensor.shape_signature = [int(v) for v in list(output_shape_signature)]

            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_matrix_name, b_compute],
                    outputs=[matmul_out_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )

            if matmul_out_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[matmul_out_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

        # Some ONNX exports carry broken scalar metadata for MatMul despite a
        # 2D constant RHS (KxN). In this ambiguous case, treat LHS as a
        # flattened batch of K-vectors to preserve accumulation dimension K
        # even when runtime batch becomes zero.
        if (
            len(b_shape) == 2
            and len(output_shape) == 1
            and int(output_shape[0]) != int(b_shape[-1])
            and int(b_shape[0]) > 0
        ):
            k_dim = int(b_shape[0])
            n_dim = int(b_shape[1]) if int(b_shape[1]) > 0 else 1
            n_sig = int(b_signature[1]) if int(b_signature[1]) > 0 else int(n_dim)

            a_matrix_new_shape = [-1, int(k_dim)]
            a_matrix_shape_name = ctx.add_const_tensor(
                f"{output_name}_matmul_a_flattened_shape",
                np.asarray(a_matrix_new_shape, dtype=np.int32),
            )
            a_matrix_name = ctx.add_intermediate_tensor(
                f"{output_name}_matmul_a_flattened",
                dtype=compute_dtype,
                shape=[1, int(k_dim)],
            )
            a_matrix_tensor = ctx.model_ir.tensors.get(a_matrix_name, None)
            if a_matrix_tensor is not None:
                a_matrix_tensor.shape_signature = [int(v) for v in a_matrix_new_shape]
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[a_compute, a_matrix_shape_name],
                    outputs=[a_matrix_name],
                    options={
                        "newShape": [int(v) for v in a_matrix_new_shape],
                        "preserveDynamicShape": True,
                    },
                )
            )

            output_shape_signature = [-1, int(n_sig)]
            output_shape = _update_output_tensor_shape(output_shape_signature)
            matmul_out_name = output_name
            if output_dtype != compute_dtype:
                matmul_out_name = ctx.add_intermediate_tensor(
                    f"{output_name}_matmul_flattened_f32",
                    dtype=compute_dtype,
                    shape=[int(v) for v in list(output_shape)],
                )
                matmul_out_tensor = ctx.model_ir.tensors.get(matmul_out_name, None)
                if matmul_out_tensor is not None:
                    matmul_out_tensor.shape_signature = [int(v) for v in list(output_shape_signature)]

            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[a_matrix_name, b_compute],
                    outputs=[matmul_out_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )

            if matmul_out_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[matmul_out_name],
                        outputs=[output_name],
                        options={"inDataType": compute_dtype, "outDataType": output_dtype},
                    )
                )
            return

        batch_shape = [int(v) for v in list(b_shape[:-2])]
        batch_signature = [int(v) for v in list(b_signature[:-2])]
        n_dim = int(b_shape[-1]) if int(b_shape[-1]) > 0 else 1
        n_sig = int(b_signature[-1]) if int(b_signature[-1]) > 0 else -1
        output_shape_signature = [int(v) for v in list(batch_signature)] + [int(n_sig)]
        output_shape = _update_output_tensor_shape(output_shape_signature)

        # Keep vector-lhs reshape robust even when optimizer mutates upstream static shapes.
        a_k_dim = int(a_shape[0]) if len(a_shape) > 0 and int(a_shape[0]) > 0 else -1
        a_matrix_new_shape = [1 for _ in range(len(b_shape) - 1)] + [
            int(a_k_dim) if int(a_k_dim) > 0 else -1
        ]
        a_matrix_shape_name = ctx.add_const_tensor(
            f"{output_name}_matmul_a_vector_shape",
            np.asarray(a_matrix_new_shape, dtype=np.int32),
        )
        a_matrix_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_a_vector_matrix",
            dtype=compute_dtype,
            shape=[int(v) if int(v) > 0 else 1 for v in a_matrix_new_shape],
        )
        a_matrix_tensor = ctx.model_ir.tensors.get(a_matrix_name, None)
        if a_matrix_tensor is not None:
            a_matrix_tensor.shape_signature = [int(v) for v in a_matrix_new_shape]
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[a_compute, a_matrix_shape_name],
                outputs=[a_matrix_name],
                options={
                    "newShape": [int(v) for v in a_matrix_new_shape],
                    "preserveDynamicShape": True,
                },
            )
        )

        squeeze_axis = int(len(batch_shape))
        matmul_vec_signature = [int(v) for v in list(batch_signature)] + [1, int(n_sig)]
        matmul_vec_shape = _materialize_shape_from_signature(matmul_vec_signature)
        matmul_vec_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_vec_lhs_out",
            dtype=compute_dtype,
            shape=[int(v) for v in list(matmul_vec_shape)],
        )
        matmul_vec_tensor = ctx.model_ir.tensors.get(matmul_vec_name, None)
        if matmul_vec_tensor is not None:
            matmul_vec_tensor.shape_signature = [int(v) for v in matmul_vec_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="BATCH_MATMUL",
                inputs=[a_matrix_name, b_compute],
                outputs=[matmul_vec_name],
                options={
                    "adjX": False,
                    "adjY": False,
                    "asymmetricQuantizeInputs": False,
                },
            )
        )

        squeeze_out = output_name
        if output_dtype != compute_dtype:
            squeeze_out = ctx.add_intermediate_tensor(
                f"{output_name}_matmul_vec_lhs_squeezed",
                dtype=compute_dtype,
                shape=output_shape,
            )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[matmul_vec_name],
                outputs=[squeeze_out],
                options={"squeezeDims": [int(squeeze_axis)]},
            )
        )
        if squeeze_out != output_name:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[squeeze_out],
                    outputs=[output_name],
                    options={"inDataType": compute_dtype, "outDataType": output_dtype},
                )
            )
        return

    if len(a_shape) >= 2 and len(b_shape) == 1:
        output_shape_signature = [int(v) for v in list(a_signature[:-1])]
        output_shape = _update_output_tensor_shape(output_shape_signature)
        k_dim = int(b_shape[0]) if len(b_shape) > 0 and int(b_shape[0]) > 0 else (
            int(b_signature[0]) if len(b_signature) > 0 else -1
        )
        b_matrix_new_shape = [int(k_dim) if int(k_dim) > 0 else -1, 1]
        b_matrix_shape_name = ctx.add_const_tensor(
            f"{output_name}_matmul_b_vector_shape",
            np.asarray(b_matrix_new_shape, dtype=np.int32),
        )
        b_matrix_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_b_vector_matrix",
            dtype=compute_dtype,
            shape=[int(v) if int(v) > 0 else 1 for v in b_matrix_new_shape],
        )
        b_matrix_tensor = ctx.model_ir.tensors.get(b_matrix_name, None)
        if b_matrix_tensor is not None:
            b_matrix_tensor.shape_signature = [int(v) for v in b_matrix_new_shape]
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[b_compute, b_matrix_shape_name],
                outputs=[b_matrix_name],
                options={"newShape": [int(v) for v in b_matrix_new_shape]},
            )
        )

        matmul_vec_signature = [int(v) for v in list(output_shape_signature)] + [1]
        matmul_vec_shape = _materialize_shape_from_signature(matmul_vec_signature)
        matmul_vec_name = ctx.add_intermediate_tensor(
            f"{output_name}_matmul_vec_out",
            dtype=compute_dtype,
            shape=[int(v) for v in list(matmul_vec_shape)],
        )
        matmul_vec_tensor = ctx.model_ir.tensors.get(matmul_vec_name, None)
        if matmul_vec_tensor is not None:
            matmul_vec_tensor.shape_signature = [int(v) for v in matmul_vec_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="BATCH_MATMUL",
                inputs=[a_compute, b_matrix_name],
                outputs=[matmul_vec_name],
                options={
                    "adjX": False,
                    "adjY": False,
                    "asymmetricQuantizeInputs": False,
                },
            )
        )

        squeeze_out = output_name
        if output_dtype != compute_dtype:
            squeeze_out = ctx.add_intermediate_tensor(
                f"{output_name}_matmul_vec_squeezed",
                dtype=compute_dtype,
                shape=output_shape,
            )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[matmul_vec_name],
                outputs=[squeeze_out],
                options={"squeezeDims": [int(len(matmul_vec_shape) - 1)]},
            )
        )
        if squeeze_out != output_name:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[squeeze_out],
                    outputs=[output_name],
                    options={"inDataType": compute_dtype, "outDataType": output_dtype},
                )
            )
        return

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
