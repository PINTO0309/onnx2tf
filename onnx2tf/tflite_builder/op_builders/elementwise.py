from __future__ import annotations

from typing import Any
import math
import copy

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def _propagate_shape(ctx: Any, src_tensor_name: str, dst_tensor_name: str) -> None:
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src = ctx.model_ir.tensors[src_tensor_name]
    dst = ctx.model_ir.tensors[dst_tensor_name]
    src_signature = (
        list(src.shape_signature)
        if src.shape_signature is not None
        else list(src.shape)
    )
    if dst.shape == [1] and src.shape != [1]:
        dst.shape = list(src.shape)
        dst.shape_signature = list(src_signature)
    elif len(list(dst.shape)) == len(list(src.shape)) and list(dst.shape) == list(src.shape):
        dst.shape_signature = list(src_signature)


def _normalize_axis_for_rank(axis: int, rank: int) -> int:
    a = int(axis)
    if a < 0:
        a += int(rank)
    if a < 0 or a >= int(rank):
        raise NotImplementedError(f"axis is out of range. axis={axis} normalized={a} rank={rank}")
    return int(a)


def _axis_to_last_permutations(axis: int, rank: int) -> tuple[list[int], list[int]]:
    perm_to_last = [int(v) for v in range(rank) if int(v) != int(axis)] + [int(axis)]
    perm_from_last = [0] * int(rank)
    for out_axis, in_axis in enumerate(perm_to_last):
        perm_from_last[int(in_axis)] = int(out_axis)
    return perm_to_last, perm_from_last


_FLOAT_TENSOR_DTYPES = {"FLOAT16", "FLOAT32"}


def _compute_dtype_for_output(output_dtype: str) -> str:
    return "FLOAT16" if str(output_dtype).upper() == "FLOAT16" else "FLOAT32"


def _require_float_input_output(node: Any, ctx: Any) -> tuple[str, str, str]:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if input_dtype not in _FLOAT_TENSOR_DTYPES or output_dtype not in _FLOAT_TENSOR_DTYPES:
        raise NotImplementedError(
            "This op currently supports FLOAT16/FLOAT32 only in flatbuffer_direct. "
            f"op={node.name} input_dtype={input_dtype} output_dtype={output_dtype}"
        )
    _propagate_shape(ctx, input_name, output_name)
    return input_name, output_name, _compute_dtype_for_output(output_dtype)


def _add_scalar_const(ctx: Any, base_name: str, value: float, dtype: str) -> str:
    np_dtype = np.float16 if str(dtype).upper() == "FLOAT16" else np.float32
    return ctx.add_const_tensor(
        base_name,
        np.asarray(value, dtype=np_dtype),
    )


def _cast_tensor_if_needed(
    *,
    ctx: Any,
    src_name: str,
    dst_dtype: str,
    base_name: str,
) -> str:
    src_dtype = str(ctx.get_tensor_dtype(src_name)).upper()
    if src_dtype == str(dst_dtype).upper():
        return src_name
    src_shape = [int(v) for v in ctx.get_tensor_shape(src_name)]
    cast_name = ctx.add_intermediate_tensor(
        base_name,
        dtype=str(dst_dtype).upper(),
        shape=src_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[src_name],
            outputs=[cast_name],
            options={
                "inDataType": src_dtype,
                "outDataType": str(dst_dtype).upper(),
            },
        )
    )
    return cast_name


def _prepare_float_compute(
    node: Any,
    ctx: Any,
    *,
    tag: str,
) -> tuple[str, str, str, str, str, list[int]]:
    input_name, output_name, compute_dtype = _require_float_input_output(node, ctx)
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    compute_input_name = _cast_tensor_if_needed(
        ctx=ctx,
        src_name=input_name,
        dst_dtype=compute_dtype,
        base_name=f"{output_name}_{tag}_input_cast",
    )
    compute_output_name = output_name
    if output_dtype != compute_dtype:
        compute_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_{tag}_out",
            dtype=compute_dtype,
            shape=output_shape,
        )
    return (
        compute_input_name,
        compute_output_name,
        output_name,
        output_dtype,
        compute_dtype,
        output_shape,
    )


def _finalize_float_compute_output(
    *,
    ctx: Any,
    compute_output_name: str,
    output_name: str,
    compute_dtype: str,
    output_dtype: str,
) -> None:
    if compute_output_name == output_name:
        return
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[compute_output_name],
            outputs=[output_name],
            options={
                "inDataType": compute_dtype,
                "outDataType": output_dtype,
            },
        )
    )


def build_binary_op(node: Any, ctx: Any, op_type: str) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)
    if len(input_names) > 0:
        _propagate_shape(ctx, input_names[0], output_name)

    options = {"fusedActivationFunction": "NONE"}
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=input_names,
            outputs=[output_name],
            options=options,
        )
    )


def build_pow_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)
    if len(input_names) > 0:
        _propagate_shape(ctx, input_names[0], output_name)

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"

    lhs_name = input_names[0]
    lhs_dtype = str(ctx.get_tensor_dtype(lhs_name)).upper()
    if lhs_dtype != compute_dtype:
        lhs_shape = [int(v) for v in ctx.get_tensor_shape(lhs_name)]
        lhs_cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_pow_lhs_cast",
            dtype=compute_dtype,
            shape=lhs_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[lhs_name],
                outputs=[lhs_cast_name],
                options={
                    "inDataType": lhs_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )
        lhs_name = lhs_cast_name

    rhs_name = input_names[1]
    rhs_dtype = str(ctx.get_tensor_dtype(rhs_name)).upper()
    if rhs_dtype != compute_dtype:
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(rhs_name)]
        rhs_cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_pow_rhs_cast",
            dtype=compute_dtype,
            shape=rhs_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[rhs_name],
                outputs=[rhs_cast_name],
                options={
                    "inDataType": rhs_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )
        rhs_name = rhs_cast_name

    pow_output_name = output_name
    if output_dtype != compute_dtype:
        output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
        pow_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_pow_out",
            dtype=compute_dtype,
            shape=output_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="POW",
            inputs=[lhs_name, rhs_name],
            outputs=[pow_output_name],
        )
    )

    if pow_output_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[pow_output_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_div_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)
    if len(input_names) > 0:
        _propagate_shape(ctx, input_names[0], output_name)

    lhs_name = input_names[0]
    rhs_name = input_names[1]
    lhs_dtype = str(ctx.get_tensor_dtype(lhs_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    rhs_const = ctx.get_constant_array(rhs_name)
    if rhs_const is not None:
        calc_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"
        np_calc_dtype = np.float16 if calc_dtype == "FLOAT16" else np.float32
        reciprocal = np.asarray(
            np.reciprocal(np.asarray(rhs_const, dtype=np_calc_dtype)),
            dtype=np_calc_dtype,
        )
        reciprocal_name = ctx.add_const_tensor(
            f"{output_name}_div_reciprocal",
            reciprocal,
        )

        mul_lhs_name = lhs_name
        if lhs_dtype != calc_dtype:
            lhs_shape = [int(v) for v in ctx.get_tensor_shape(lhs_name)]
            mul_lhs_name = ctx.add_intermediate_tensor(
                f"{output_name}_div_lhs_cast",
                dtype=calc_dtype,
                shape=lhs_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[lhs_name],
                    outputs=[mul_lhs_name],
                    options={
                        "inDataType": lhs_dtype,
                        "outDataType": calc_dtype,
                    },
                )
            )

        mul_out_name = output_name
        if output_dtype != calc_dtype:
            output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
            mul_out_name = ctx.add_intermediate_tensor(
                f"{output_name}_div_mul_out",
                dtype=calc_dtype,
                shape=output_shape,
            )

        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[mul_lhs_name, reciprocal_name],
                outputs=[mul_out_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

        if mul_out_name != output_name:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[mul_out_name],
                    outputs=[output_name],
                    options={
                        "inDataType": calc_dtype,
                        "outDataType": output_dtype,
                    },
                )
            )
        return

    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=input_names,
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )


def build_reciprocal_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"
    np_compute_dtype = np.float16 if compute_dtype == "FLOAT16" else np.float32

    denom_name = input_name
    if input_dtype != compute_dtype:
        input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
        denom_cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_reciprocal_input_cast",
            dtype=compute_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[denom_cast_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )
        denom_name = denom_cast_name

    one_name = ctx.add_const_tensor(
        f"{node.name}_reciprocal_one",
        np.asarray(1.0, dtype=np_compute_dtype),
    )

    div_output_name = output_name
    if output_dtype != compute_dtype:
        output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
        div_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_reciprocal_out",
            dtype=compute_dtype,
            shape=output_shape,
        )

    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[one_name, denom_name],
            outputs=[div_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    if div_output_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[div_output_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_mod_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)
    if len(input_names) > 0:
        _propagate_shape(ctx, input_names[0], output_name)

    ctx.add_operator(
        OperatorIR(
            op_type="FLOOR_MOD",
            inputs=input_names,
            outputs=[output_name],
        )
    )


def build_logistic_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)
    ctx.add_operator(
        OperatorIR(
            op_type="LOGISTIC",
            inputs=[input_name],
            outputs=[output_name],
        )
    )


def build_hardsigmoid_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    alpha = float(node.attrs.get("alpha", 0.2))
    beta = float(node.attrs.get("beta", 0.5))

    input_shape = list(ctx.get_tensor_shape(input_name))
    input_signature = (
        list(ctx.model_ir.tensors[input_name].shape_signature)
        if ctx.model_ir.tensors[input_name].shape_signature is not None
        else list(input_shape)
    )
    tensor_dtype = str(ctx.get_tensor_dtype(input_name))
    output_dtype = str(ctx.get_tensor_dtype(output_name))
    if tensor_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NotImplementedError(
            "HardSigmoid currently supports FLOAT16/FLOAT32 input in flatbuffer_direct. "
            f"op={node.name} input_dtype={tensor_dtype}"
        )
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NotImplementedError(
            "HardSigmoid currently supports FLOAT16/FLOAT32 output in flatbuffer_direct. "
            f"op={node.name} output_dtype={output_dtype}"
        )

    const_dtype = np.float16 if output_dtype == "FLOAT16" else np.float32
    alpha_name = ctx.add_const_tensor(
        f"{node.name}_hardsigmoid_alpha",
        np.asarray(alpha, dtype=const_dtype),
    )
    beta_name = ctx.add_const_tensor(
        f"{node.name}_hardsigmoid_beta",
        np.asarray(beta, dtype=const_dtype),
    )
    zero_name = ctx.add_const_tensor(
        f"{node.name}_hardsigmoid_zero",
        np.asarray(0.0, dtype=const_dtype),
    )
    one_name = ctx.add_const_tensor(
        f"{node.name}_hardsigmoid_one",
        np.asarray(1.0, dtype=const_dtype),
    )

    mul_out = ctx.add_intermediate_tensor(
        f"{node.name}_hardsigmoid_mul_out",
        dtype=output_dtype,
        shape=input_shape,
    )
    add_out = ctx.add_intermediate_tensor(
        f"{node.name}_hardsigmoid_add_out",
        dtype=output_dtype,
        shape=input_shape,
    )
    max_out = ctx.add_intermediate_tensor(
        f"{node.name}_hardsigmoid_max_out",
        dtype=output_dtype,
        shape=input_shape,
    )
    ctx.model_ir.tensors[mul_out].shape_signature = [int(v) for v in list(input_signature)]
    ctx.model_ir.tensors[add_out].shape_signature = [int(v) for v in list(input_signature)]
    ctx.model_ir.tensors[max_out].shape_signature = [int(v) for v in list(input_signature)]

    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[input_name, alpha_name],
            outputs=[mul_out],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[mul_out, beta_name],
            outputs=[add_out],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MAXIMUM",
            inputs=[add_out, zero_name],
            outputs=[max_out],
            options={},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MINIMUM",
            inputs=[max_out, one_name],
            outputs=[output_name],
            options={},
        )
    )


def build_unary_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[input_name],
            outputs=[output_name],
        )
    )


def build_erf_op(node: Any, ctx: Any) -> None:
    (
        compute_input_name,
        compute_output_name,
        output_name,
        output_dtype,
        compute_dtype,
        out_shape,
    ) = _prepare_float_compute(node, ctx, tag="erf")

    one_name = _add_scalar_const(ctx, f"{output_name}_erf_one", 1.0, compute_dtype)
    minus_one_name = _add_scalar_const(ctx, f"{output_name}_erf_minus_one", -1.0, compute_dtype)
    p_name = _add_scalar_const(ctx, f"{output_name}_erf_p", 0.3275911, compute_dtype)
    a1_name = _add_scalar_const(ctx, f"{output_name}_erf_a1", 0.254829592, compute_dtype)
    a2_name = _add_scalar_const(ctx, f"{output_name}_erf_a2", -0.284496736, compute_dtype)
    a3_name = _add_scalar_const(ctx, f"{output_name}_erf_a3", 1.421413741, compute_dtype)
    a4_name = _add_scalar_const(ctx, f"{output_name}_erf_a4", -1.453152027, compute_dtype)
    a5_name = _add_scalar_const(ctx, f"{output_name}_erf_a5", 1.061405429, compute_dtype)

    abs_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_abs",
        dtype=compute_dtype,
        shape=out_shape,
    )
    sign_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_sign",
        dtype=compute_dtype,
        shape=out_shape,
    )
    px_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_px",
        dtype=compute_dtype,
        shape=out_shape,
    )
    one_plus_px_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_one_plus_px",
        dtype=compute_dtype,
        shape=out_shape,
    )
    t_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_t",
        dtype=compute_dtype,
        shape=out_shape,
    )
    abs_sq_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_abs_sq",
        dtype=compute_dtype,
        shape=out_shape,
    )
    neg_abs_sq_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_neg_abs_sq",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_exp",
        dtype=compute_dtype,
        shape=out_shape,
    )
    s1_mul_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_s1_mul",
        dtype=compute_dtype,
        shape=out_shape,
    )
    s1_add_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_s1_add",
        dtype=compute_dtype,
        shape=out_shape,
    )
    s2_mul_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_s2_mul",
        dtype=compute_dtype,
        shape=out_shape,
    )
    s2_add_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_s2_add",
        dtype=compute_dtype,
        shape=out_shape,
    )
    s3_mul_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_s3_mul",
        dtype=compute_dtype,
        shape=out_shape,
    )
    s3_add_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_s3_add",
        dtype=compute_dtype,
        shape=out_shape,
    )
    s4_mul_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_s4_mul",
        dtype=compute_dtype,
        shape=out_shape,
    )
    s4_add_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_s4_add",
        dtype=compute_dtype,
        shape=out_shape,
    )
    poly_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_poly",
        dtype=compute_dtype,
        shape=out_shape,
    )
    poly_exp_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_poly_exp",
        dtype=compute_dtype,
        shape=out_shape,
    )
    one_minus_name = ctx.add_intermediate_tensor(
        f"{output_name}_erf_one_minus",
        dtype=compute_dtype,
        shape=out_shape,
    )

    ctx.add_operator(
        OperatorIR(
            op_type="ABS",
            inputs=[compute_input_name],
            outputs=[abs_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SIGN",
            inputs=[compute_input_name],
            outputs=[sign_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[abs_name, p_name],
            outputs=[px_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[one_name, px_name],
            outputs=[one_plus_px_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[one_name, one_plus_px_name],
            outputs=[t_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[abs_name, abs_name],
            outputs=[abs_sq_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[abs_sq_name, minus_one_name],
            outputs=[neg_abs_sq_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="EXP",
            inputs=[neg_abs_sq_name],
            outputs=[exp_name],
        )
    )

    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[a5_name, t_name],
            outputs=[s1_mul_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[s1_mul_name, a4_name],
            outputs=[s1_add_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[s1_add_name, t_name],
            outputs=[s2_mul_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[s2_mul_name, a3_name],
            outputs=[s2_add_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[s2_add_name, t_name],
            outputs=[s3_mul_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[s3_mul_name, a2_name],
            outputs=[s3_add_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[s3_add_name, t_name],
            outputs=[s4_mul_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[s4_mul_name, a1_name],
            outputs=[s4_add_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[s4_add_name, t_name],
            outputs=[poly_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[poly_name, exp_name],
            outputs=[poly_exp_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[one_name, poly_exp_name],
            outputs=[one_minus_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[sign_name, one_minus_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_where_op(node: Any, ctx: Any) -> None:
    condition_name = node.inputs[0].name
    x_name = node.inputs[1].name
    y_name = node.inputs[2].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(condition_name)
    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(y_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, x_name, output_name)

    condition_dtype = str(ctx.get_tensor_dtype(condition_name)).upper()
    cond_for_select = condition_name
    if condition_dtype != "BOOL":
        cond_shape = [int(v) for v in ctx.get_tensor_shape(condition_name)]
        cond_for_select = ctx.add_intermediate_tensor(
            f"{output_name}_where_condition_bool",
            dtype="BOOL",
            shape=cond_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[condition_name],
                outputs=[cond_for_select],
                options={
                    "inDataType": condition_dtype,
                    "outDataType": "BOOL",
                },
            )
        )

    ctx.add_operator(
        OperatorIR(
            op_type="SELECT",
            inputs=[cond_for_select, x_name, y_name],
            outputs=[output_name],
        )
    )


def build_bitwise_not_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if input_dtype == "BOOL":
        ctx.add_operator(
            OperatorIR(
                op_type="LOGICAL_NOT",
                inputs=[input_name],
                outputs=[output_name],
            )
        )
        return

    dtype_to_np = {
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
    }
    if input_dtype not in dtype_to_np:
        raise NotImplementedError(
            "BitwiseNot currently supports integer/bool only in flatbuffer_direct. "
            f"op={node.name} input_dtype={input_dtype}"
        )

    compute_input_name = input_name
    if output_dtype != input_dtype:
        compute_input_name = _cast_tensor_if_needed(
            ctx=ctx,
            src_name=input_name,
            dst_dtype=input_dtype,
            base_name=f"{output_name}_bitwise_not_input_cast",
        )

    not_out_name = output_name
    if output_dtype != input_dtype:
        not_out_name = ctx.add_intermediate_tensor(
            f"{output_name}_bitwise_not_out",
            dtype=input_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
        )

    np_dtype = dtype_to_np[input_dtype]
    if np.issubdtype(np_dtype, np.signedinteger):
        zero_name = ctx.add_const_tensor(
            f"{output_name}_bitwise_not_zero",
            np.asarray(0, dtype=np_dtype),
        )
        one_name = ctx.add_const_tensor(
            f"{output_name}_bitwise_not_one",
            np.asarray(1, dtype=np_dtype),
        )
        neg_name = ctx.add_intermediate_tensor(
            f"{output_name}_bitwise_not_neg",
            dtype=input_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(compute_input_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[zero_name, compute_input_name],
                outputs=[neg_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[neg_name, one_name],
                outputs=[not_out_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
    else:
        max_name = ctx.add_const_tensor(
            f"{output_name}_bitwise_not_max",
            np.asarray(np.iinfo(np_dtype).max, dtype=np_dtype),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[max_name, compute_input_name],
                outputs=[not_out_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

    if not_out_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[not_out_name],
                outputs=[output_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_bitshift_op(node: Any, ctx: Any) -> None:
    lhs_name = node.inputs[0].name
    rhs_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(lhs_name)
    ctx.ensure_tensor(rhs_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, lhs_name, output_name)

    lhs_dtype = str(ctx.get_tensor_dtype(lhs_name)).upper()
    if lhs_dtype not in {
        "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64",
    }:
        raise NotImplementedError(
            "BitShift currently supports integer input only in flatbuffer_direct. "
            f"op={node.name} input_dtype={lhs_dtype}"
        )

    direction = str(node.attrs.get("direction", "RIGHT")).upper()
    if direction == "RIGHT":
        ctx.add_operator(
            OperatorIR(
                op_type="RIGHT_SHIFT",
                inputs=[lhs_name, rhs_name],
                outputs=[output_name],
            )
        )
        return

    if direction != "LEFT":
        raise NotImplementedError(
            f"BitShift direction must be LEFT or RIGHT in flatbuffer_direct. op={node.name} direction={direction}"
        )

    rhs_const = ctx.get_constant_array(rhs_name)
    if rhs_const is None:
        raise NotImplementedError(
            "BitShift LEFT currently requires constant shift tensor in flatbuffer_direct. "
            f"op={node.name}"
        )
    shift_arr = np.asarray(rhs_const).astype(np.int64)
    if np.any(shift_arr < 0):
        raise NotImplementedError(
            f"BitShift LEFT requires non-negative shifts in flatbuffer_direct. op={node.name}"
        )
    np_dtype_map = {
        "INT8": np.int8,
        "INT16": np.int16,
        "INT32": np.int32,
        "INT64": np.int64,
        "UINT8": np.uint8,
        "UINT16": np.uint16,
        "UINT32": np.uint32,
        "UINT64": np.uint64,
    }
    multiplier = np.left_shift(
        np.ones_like(shift_arr, dtype=np.int64),
        shift_arr,
    ).astype(np_dtype_map[lhs_dtype], copy=False)
    multiplier_name = ctx.add_const_tensor(
        f"{output_name}_bitshift_left_multiplier",
        multiplier,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[lhs_name, multiplier_name],
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )


def build_atan_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, _ = (
        _prepare_float_compute(node, ctx, tag="atan")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_atan_one", 1.0, compute_dtype)
    ctx.add_operator(
        OperatorIR(
            op_type="ATAN2",
            inputs=[compute_input_name, one_name],
            outputs=[compute_output_name],
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_asin_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="asin")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_asin_one", 1.0, compute_dtype)
    x_sq_name = ctx.add_intermediate_tensor(
        f"{output_name}_asin_x_sq",
        dtype=compute_dtype,
        shape=out_shape,
    )
    one_minus_name = ctx.add_intermediate_tensor(
        f"{output_name}_asin_one_minus_x_sq",
        dtype=compute_dtype,
        shape=out_shape,
    )
    denom_name = ctx.add_intermediate_tensor(
        f"{output_name}_asin_denom",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[compute_input_name, compute_input_name],
            outputs=[x_sq_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[one_name, x_sq_name],
            outputs=[one_minus_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[one_minus_name],
            outputs=[denom_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ATAN2",
            inputs=[compute_input_name, denom_name],
            outputs=[compute_output_name],
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_acos_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="acos")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_acos_one", 1.0, compute_dtype)
    x_sq_name = ctx.add_intermediate_tensor(
        f"{output_name}_acos_x_sq",
        dtype=compute_dtype,
        shape=out_shape,
    )
    one_minus_name = ctx.add_intermediate_tensor(
        f"{output_name}_acos_one_minus_x_sq",
        dtype=compute_dtype,
        shape=out_shape,
    )
    numer_name = ctx.add_intermediate_tensor(
        f"{output_name}_acos_numer",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[compute_input_name, compute_input_name],
            outputs=[x_sq_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[one_name, x_sq_name],
            outputs=[one_minus_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[one_minus_name],
            outputs=[numer_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ATAN2",
            inputs=[numer_name, compute_input_name],
            outputs=[compute_output_name],
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_asinh_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="asinh")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_asinh_one", 1.0, compute_dtype)
    x_sq_name = ctx.add_intermediate_tensor(
        f"{output_name}_asinh_x_sq",
        dtype=compute_dtype,
        shape=out_shape,
    )
    x_sq_plus_one_name = ctx.add_intermediate_tensor(
        f"{output_name}_asinh_x_sq_plus_one",
        dtype=compute_dtype,
        shape=out_shape,
    )
    sqrt_name = ctx.add_intermediate_tensor(
        f"{output_name}_asinh_sqrt",
        dtype=compute_dtype,
        shape=out_shape,
    )
    sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_asinh_sum",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[compute_input_name, compute_input_name],
            outputs=[x_sq_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[x_sq_name, one_name],
            outputs=[x_sq_plus_one_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[x_sq_plus_one_name],
            outputs=[sqrt_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[compute_input_name, sqrt_name],
            outputs=[sum_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOG",
            inputs=[sum_name],
            outputs=[compute_output_name],
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_acosh_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="acosh")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_acosh_one", 1.0, compute_dtype)
    x_minus_one_name = ctx.add_intermediate_tensor(
        f"{output_name}_acosh_x_minus_one",
        dtype=compute_dtype,
        shape=out_shape,
    )
    x_plus_one_name = ctx.add_intermediate_tensor(
        f"{output_name}_acosh_x_plus_one",
        dtype=compute_dtype,
        shape=out_shape,
    )
    sqrt_lhs_name = ctx.add_intermediate_tensor(
        f"{output_name}_acosh_sqrt_lhs",
        dtype=compute_dtype,
        shape=out_shape,
    )
    sqrt_rhs_name = ctx.add_intermediate_tensor(
        f"{output_name}_acosh_sqrt_rhs",
        dtype=compute_dtype,
        shape=out_shape,
    )
    prod_name = ctx.add_intermediate_tensor(
        f"{output_name}_acosh_prod",
        dtype=compute_dtype,
        shape=out_shape,
    )
    sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_acosh_sum",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[compute_input_name, one_name],
            outputs=[x_minus_one_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[compute_input_name, one_name],
            outputs=[x_plus_one_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[x_minus_one_name],
            outputs=[sqrt_lhs_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[x_plus_one_name],
            outputs=[sqrt_rhs_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[sqrt_lhs_name, sqrt_rhs_name],
            outputs=[prod_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[compute_input_name, prod_name],
            outputs=[sum_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOG",
            inputs=[sum_name],
            outputs=[compute_output_name],
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_atanh_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="atanh")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_atanh_one", 1.0, compute_dtype)
    half_name = _add_scalar_const(ctx, f"{output_name}_atanh_half", 0.5, compute_dtype)
    one_plus_name = ctx.add_intermediate_tensor(
        f"{output_name}_atanh_one_plus",
        dtype=compute_dtype,
        shape=out_shape,
    )
    one_minus_name = ctx.add_intermediate_tensor(
        f"{output_name}_atanh_one_minus",
        dtype=compute_dtype,
        shape=out_shape,
    )
    div_name = ctx.add_intermediate_tensor(
        f"{output_name}_atanh_div",
        dtype=compute_dtype,
        shape=out_shape,
    )
    log_name = ctx.add_intermediate_tensor(
        f"{output_name}_atanh_log",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[one_name, compute_input_name],
            outputs=[one_plus_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[one_name, compute_input_name],
            outputs=[one_minus_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[one_plus_name, one_minus_name],
            outputs=[div_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOG",
            inputs=[div_name],
            outputs=[log_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[log_name, half_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_cosh_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="cosh")
    )
    half_name = _add_scalar_const(ctx, f"{output_name}_cosh_half", 0.5, compute_dtype)
    neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_cosh_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_pos_name = ctx.add_intermediate_tensor(
        f"{output_name}_cosh_exp_pos",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_cosh_exp_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_cosh_sum",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[_add_scalar_const(ctx, f"{output_name}_cosh_zero", 0.0, compute_dtype), compute_input_name],
            outputs=[neg_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(OperatorIR(op_type="EXP", inputs=[compute_input_name], outputs=[exp_pos_name]))
    ctx.add_operator(OperatorIR(op_type="EXP", inputs=[neg_name], outputs=[exp_neg_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[exp_pos_name, exp_neg_name],
            outputs=[sum_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[sum_name, half_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_sinh_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="sinh")
    )
    half_name = _add_scalar_const(ctx, f"{output_name}_sinh_half", 0.5, compute_dtype)
    zero_name = _add_scalar_const(ctx, f"{output_name}_sinh_zero", 0.0, compute_dtype)
    neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_sinh_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_pos_name = ctx.add_intermediate_tensor(
        f"{output_name}_sinh_exp_pos",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_sinh_exp_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    diff_name = ctx.add_intermediate_tensor(
        f"{output_name}_sinh_diff",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[zero_name, compute_input_name],
            outputs=[neg_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(OperatorIR(op_type="EXP", inputs=[compute_input_name], outputs=[exp_pos_name]))
    ctx.add_operator(OperatorIR(op_type="EXP", inputs=[neg_name], outputs=[exp_neg_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[exp_pos_name, exp_neg_name],
            outputs=[diff_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[diff_name, half_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_tan_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="tan")
    )
    sin_name = ctx.add_intermediate_tensor(
        f"{output_name}_tan_sin",
        dtype=compute_dtype,
        shape=out_shape,
    )
    cos_name = ctx.add_intermediate_tensor(
        f"{output_name}_tan_cos",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(OperatorIR(op_type="SIN", inputs=[compute_input_name], outputs=[sin_name]))
    ctx.add_operator(OperatorIR(op_type="COS", inputs=[compute_input_name], outputs=[cos_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[sin_name, cos_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_softplus_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="softplus")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_softplus_one", 1.0, compute_dtype)
    exp_name = ctx.add_intermediate_tensor(
        f"{output_name}_softplus_exp",
        dtype=compute_dtype,
        shape=out_shape,
    )
    sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_softplus_sum",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(OperatorIR(op_type="EXP", inputs=[compute_input_name], outputs=[exp_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[exp_name, one_name],
            outputs=[sum_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOG",
            inputs=[sum_name],
            outputs=[compute_output_name],
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_softsign_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="softsign")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_softsign_one", 1.0, compute_dtype)
    abs_name = ctx.add_intermediate_tensor(
        f"{output_name}_softsign_abs",
        dtype=compute_dtype,
        shape=out_shape,
    )
    denom_name = ctx.add_intermediate_tensor(
        f"{output_name}_softsign_denom",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(OperatorIR(op_type="ABS", inputs=[compute_input_name], outputs=[abs_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[abs_name, one_name],
            outputs=[denom_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[compute_input_name, denom_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_celu_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="celu")
    )
    alpha = float(node.attrs.get("alpha", 1.0))
    if alpha <= 0.0:
        raise NotImplementedError(
            f"Celu alpha must be > 0 in flatbuffer_direct. op={node.name} alpha={alpha}"
        )
    alpha_name = _add_scalar_const(ctx, f"{output_name}_celu_alpha", alpha, compute_dtype)
    zero_name = _add_scalar_const(ctx, f"{output_name}_celu_zero", 0.0, compute_dtype)
    one_name = _add_scalar_const(ctx, f"{output_name}_celu_one", 1.0, compute_dtype)
    pos_name = ctx.add_intermediate_tensor(
        f"{output_name}_celu_pos",
        dtype=compute_dtype,
        shape=out_shape,
    )
    neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_celu_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    neg_div_alpha_name = ctx.add_intermediate_tensor(
        f"{output_name}_celu_neg_div_alpha",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_name = ctx.add_intermediate_tensor(
        f"{output_name}_celu_exp",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_minus_one_name = ctx.add_intermediate_tensor(
        f"{output_name}_celu_exp_minus_one",
        dtype=compute_dtype,
        shape=out_shape,
    )
    scaled_neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_celu_scaled_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(OperatorIR(op_type="MAXIMUM", inputs=[compute_input_name, zero_name], outputs=[pos_name]))
    ctx.add_operator(OperatorIR(op_type="MINIMUM", inputs=[compute_input_name, zero_name], outputs=[neg_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[neg_name, alpha_name],
            outputs=[neg_div_alpha_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(OperatorIR(op_type="EXP", inputs=[neg_div_alpha_name], outputs=[exp_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[exp_name, one_name],
            outputs=[exp_minus_one_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[alpha_name, exp_minus_one_name],
            outputs=[scaled_neg_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[pos_name, scaled_neg_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_selu_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="selu")
    )
    alpha = float(node.attrs.get("alpha", 1.6732631921768188))
    gamma = float(node.attrs.get("gamma", 1.0507009873554805))
    alpha_name = _add_scalar_const(ctx, f"{output_name}_selu_alpha", alpha, compute_dtype)
    gamma_name = _add_scalar_const(ctx, f"{output_name}_selu_gamma", gamma, compute_dtype)
    zero_name = _add_scalar_const(ctx, f"{output_name}_selu_zero", 0.0, compute_dtype)
    one_name = _add_scalar_const(ctx, f"{output_name}_selu_one", 1.0, compute_dtype)
    pos_name = ctx.add_intermediate_tensor(
        f"{output_name}_selu_pos",
        dtype=compute_dtype,
        shape=out_shape,
    )
    neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_selu_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_selu_exp_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    exp_neg_minus_one_name = ctx.add_intermediate_tensor(
        f"{output_name}_selu_exp_neg_minus_one",
        dtype=compute_dtype,
        shape=out_shape,
    )
    scaled_neg_name = ctx.add_intermediate_tensor(
        f"{output_name}_selu_scaled_neg",
        dtype=compute_dtype,
        shape=out_shape,
    )
    elu_alpha_name = ctx.add_intermediate_tensor(
        f"{output_name}_selu_elu_alpha",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(OperatorIR(op_type="MAXIMUM", inputs=[compute_input_name, zero_name], outputs=[pos_name]))
    ctx.add_operator(OperatorIR(op_type="MINIMUM", inputs=[compute_input_name, zero_name], outputs=[neg_name]))
    ctx.add_operator(OperatorIR(op_type="EXP", inputs=[neg_name], outputs=[exp_neg_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[exp_neg_name, one_name],
            outputs=[exp_neg_minus_one_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[alpha_name, exp_neg_minus_one_name],
            outputs=[scaled_neg_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[pos_name, scaled_neg_name],
            outputs=[elu_alpha_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[gamma_name, elu_alpha_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_mish_op(node: Any, ctx: Any) -> None:
    compute_input_name, compute_output_name, output_name, output_dtype, compute_dtype, out_shape = (
        _prepare_float_compute(node, ctx, tag="mish")
    )
    one_name = _add_scalar_const(ctx, f"{output_name}_mish_one", 1.0, compute_dtype)
    exp_name = ctx.add_intermediate_tensor(
        f"{output_name}_mish_exp",
        dtype=compute_dtype,
        shape=out_shape,
    )
    softplus_sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_mish_softplus_sum",
        dtype=compute_dtype,
        shape=out_shape,
    )
    softplus_name = ctx.add_intermediate_tensor(
        f"{output_name}_mish_softplus",
        dtype=compute_dtype,
        shape=out_shape,
    )
    tanh_name = ctx.add_intermediate_tensor(
        f"{output_name}_mish_tanh",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(OperatorIR(op_type="EXP", inputs=[compute_input_name], outputs=[exp_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[exp_name, one_name],
            outputs=[softplus_sum_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(OperatorIR(op_type="LOG", inputs=[softplus_sum_name], outputs=[softplus_name]))
    ctx.add_operator(OperatorIR(op_type="TANH", inputs=[softplus_name], outputs=[tanh_name]))
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[compute_input_name, tanh_name],
            outputs=[compute_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def _get_clip_bound_value(value: Any, default_value: float) -> float:
    if value is None:
        return float(default_value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        import numpy as np
        arr = np.asarray(value)
        if arr.size == 0:
            return float(default_value)
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default_value)


def build_clip_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    clip_min = _get_clip_bound_value(node.attrs.get("min", None), float("-inf"))
    clip_max = _get_clip_bound_value(node.attrs.get("max", None), float("inf"))
    min_arr = None
    max_arr = None
    if len(node.inputs) >= 2:
        min_const = ctx.get_constant_array(node.inputs[1].name)
        if min_const is not None:
            min_arr = np.asarray(min_const, dtype=np.float32)
            clip_min = _get_clip_bound_value(min_arr, clip_min)
    if len(node.inputs) >= 3:
        max_const = ctx.get_constant_array(node.inputs[2].name)
        if max_const is not None:
            max_arr = np.asarray(max_const, dtype=np.float32)
            clip_max = _get_clip_bound_value(max_arr, clip_max)

    if min_arr is None and np.isfinite(clip_min):
        min_arr = np.asarray(clip_min, dtype=np.float32)
    if max_arr is None and np.isfinite(clip_max):
        max_arr = np.asarray(clip_max, dtype=np.float32)

    if abs(clip_min - 0.0) <= 1e-6 and abs(clip_max - 6.0) <= 1e-6 and min_arr is not None and max_arr is not None:
        op_type = "RELU6"
    elif abs(clip_min - 0.0) <= 1e-6 and math.isinf(clip_max) and clip_max > 0.0 and min_arr is not None and max_arr is None:
        op_type = "RELU"
    else:
        output_dtype = ctx.get_tensor_dtype(output_name)
        output_shape = ctx.get_tensor_shape(output_name)
        current_name = input_name
        if min_arr is not None:
            min_name = ctx.add_const_tensor(
                f"{node.name}_clip_min",
                np.asarray(min_arr, dtype=np.float32),
            )
            min_output_name = output_name
            if max_arr is not None:
                min_output_name = ctx.add_intermediate_tensor(
                    f"{node.name}_clip_min_out",
                    dtype=output_dtype,
                    shape=output_shape,
                )
                output_signature = (
                    list(ctx.model_ir.tensors[output_name].shape_signature)
                    if ctx.model_ir.tensors[output_name].shape_signature is not None
                    else list(output_shape)
                )
                ctx.model_ir.tensors[min_output_name].shape_signature = [
                    int(v) for v in list(output_signature)
                ]
            ctx.add_operator(
                OperatorIR(
                    op_type="MAXIMUM",
                    inputs=[current_name, min_name],
                    outputs=[min_output_name],
                    options={},
                )
            )
            current_name = min_output_name
        if max_arr is not None:
            max_name = ctx.add_const_tensor(
                f"{node.name}_clip_max",
                np.asarray(max_arr, dtype=np.float32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MINIMUM",
                    inputs=[current_name, max_name],
                    outputs=[output_name],
                    options={},
                )
            )
        if min_arr is None and max_arr is None:
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[
                        input_name,
                        ctx.add_const_tensor(
                            f"{node.name}_identity_shape",
                            np.asarray(output_shape, dtype=np.int32),
                        ),
                    ],
                    outputs=[output_name],
                    options={"newShape": [int(v) for v in output_shape]},
                )
            )
        return

    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[input_name],
            outputs=[output_name],
        )
    )


def build_softmax_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
    if rank <= 0:
        raise NotImplementedError(f"Softmax requires rank >= 1. op={node.name} shape={input_shape}")
    axis = int(node.attrs.get("axis", 1))
    axis = _normalize_axis_for_rank(axis=axis, rank=rank)

    if axis == rank - 1:
        ctx.add_operator(
            OperatorIR(
                op_type="SOFTMAX",
                inputs=[input_name],
                outputs=[output_name],
                options={"beta": float(node.attrs.get("beta", 1.0))},
            )
        )
        return

    perm_to_last, perm_from_last = _axis_to_last_permutations(axis=axis, rank=rank)
    axis_last_shape = [int(input_shape[int(v)]) for v in perm_to_last]
    input_axis_last_name = ctx.add_intermediate_tensor(
        f"{node.name}_softmax_input_axis_last",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=axis_last_shape,
    )
    input_axis_last_name = make_transpose(
        ctx,
        input_name,
        input_axis_last_name,
        perm_to_last,
    )
    output_axis_last_name = ctx.add_intermediate_tensor(
        f"{node.name}_softmax_output_axis_last",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=list(axis_last_shape),
    )

    ctx.add_operator(
        OperatorIR(
            op_type="SOFTMAX",
            inputs=[input_axis_last_name],
            outputs=[output_axis_last_name],
            options={"beta": float(node.attrs.get("beta", 1.0))},
        )
    )
    make_transpose(
        ctx,
        output_axis_last_name,
        output_name,
        perm_from_last,
    )


def build_logsoftmax_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
    if rank <= 0:
        raise NotImplementedError(f"LogSoftmax requires rank >= 1. op={node.name} shape={input_shape}")
    axis = int(node.attrs.get("axis", 1))
    axis = _normalize_axis_for_rank(axis=axis, rank=rank)

    output_dtype = str(ctx.get_tensor_dtype(output_name))
    if axis != rank - 1:
        perm_to_last, perm_from_last = _axis_to_last_permutations(axis=axis, rank=rank)
        axis_last_shape = [int(input_shape[int(v)]) for v in perm_to_last]
        input_axis_last_name = ctx.add_intermediate_tensor(
            f"{node.name}_logsoftmax_input_axis_last",
            dtype=ctx.get_tensor_dtype(input_name),
            shape=axis_last_shape,
        )
        input_axis_last_name = make_transpose(
            ctx,
            input_name,
            input_axis_last_name,
            perm_to_last,
        )
        softmax_output_name = ctx.add_intermediate_tensor(
            f"{node.name}_softmax_axis_last",
            dtype=output_dtype,
            shape=list(axis_last_shape),
        )
        log_output_axis_last_name = ctx.add_intermediate_tensor(
            f"{node.name}_logsoftmax_output_axis_last",
            dtype=output_dtype,
            shape=list(axis_last_shape),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SOFTMAX",
                inputs=[input_axis_last_name],
                outputs=[softmax_output_name],
                options={"beta": 1.0},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="LOG",
                inputs=[softmax_output_name],
                outputs=[log_output_axis_last_name],
            )
        )
        make_transpose(
            ctx,
            log_output_axis_last_name,
            output_name,
            perm_from_last,
        )
        return

    softmax_output_name = ctx.add_intermediate_tensor(
        f"{node.name}_softmax",
        dtype=output_dtype,
        shape=list(ctx.get_tensor_shape(output_name)),
    )
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    softmax_tensor = ctx.model_ir.tensors.get(softmax_output_name, None)
    if output_tensor is not None and softmax_tensor is not None:
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_tensor.shape)]
        )
        softmax_tensor.shape_signature = [int(v) for v in output_signature]

    ctx.add_operator(
        OperatorIR(
            op_type="SOFTMAX",
            inputs=[input_name],
            outputs=[softmax_output_name],
            options={"beta": 1.0},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOG",
            inputs=[softmax_output_name],
            outputs=[output_name],
        )
    )


def _clone_quantization(quantization: Any) -> Any:
    if quantization is None:
        return None
    if isinstance(quantization, QuantParamIR):
        return QuantParamIR(
            scale=list(quantization.scale),
            zero_point=list(quantization.zero_point),
            quantized_dimension=int(quantization.quantized_dimension),
            min=list(quantization.min) if quantization.min is not None else None,
            max=list(quantization.max) if quantization.max is not None else None,
        )
    return copy.deepcopy(quantization)


def _reshape_prelu_slope_for_input(
    slope: np.ndarray,
    input_shape: list[int],
) -> np.ndarray:
    if slope.ndim == 0:
        return slope.reshape([1])
    if len(input_shape) == 4 and len(input_shape) >= 2:
        channels = int(input_shape[1])
        if slope.ndim == 1 and slope.size == channels:
            return slope.reshape([1, channels, 1, 1])
        if slope.ndim == 3 and slope.shape[0] == channels and slope.shape[1] == 1 and slope.shape[2] == 1:
            return slope.reshape([1, channels, 1, 1])
    if len(input_shape) == 2 and len(input_shape) >= 2:
        channels = int(input_shape[1])
        if slope.ndim == 1 and slope.size == channels:
            return slope.reshape([1, channels])
    return slope


def _quantize_prelu_slope(
    slope: np.ndarray,
    target_dtype: str,
) -> tuple[np.ndarray, QuantParamIR]:
    if target_dtype == "INT8":
        max_abs = float(np.max(np.abs(slope))) if slope.size > 0 else 0.0
        scale = max(max_abs / 127.0, 1e-8)
        q = np.clip(np.round(slope / scale), -128, 127).astype(np.int8)
        return q, QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            quantized_dimension=0,
        )
    if target_dtype == "UINT8":
        mn = float(np.min(slope)) if slope.size > 0 else 0.0
        mx = float(np.max(slope)) if slope.size > 0 else 0.0
        scale = max((mx - mn) / 255.0, 1e-8)
        zp = int(np.round(-mn / scale))
        zp = int(np.clip(zp, 0, 255))
        q = np.clip(np.round(slope / scale) + zp, 0, 255).astype(np.uint8)
        return q, QuantParamIR(
            scale=[float(scale)],
            zero_point=[int(zp)],
            quantized_dimension=0,
        )
    raise NotImplementedError(
        f"PRelu quantized slope requires INT8/UINT8 input. got={target_dtype}"
    )


def build_prelu_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    slope_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    slope = ctx.get_constant_array(slope_name)
    if slope is None:
        raise NotImplementedError(
            "PRelu slope must be constant for flatbuffer_direct. "
            f"op={node.name} slope_tensor={slope_name}"
        )
    slope_f = _reshape_prelu_slope_for_input(
        np.asarray(slope, dtype=np.float32),
        ctx.get_tensor_shape(input_name),
    )

    input_dtype = str(ctx.get_tensor_dtype(input_name))
    slope_tensor_name = ""
    if input_dtype in {"INT8", "UINT8"}:
        slope_q, slope_qparams = _quantize_prelu_slope(slope_f, input_dtype)
        slope_tensor_name = ctx.add_const_tensor(
            f"{node.name}_prelu_alpha_q",
            slope_q,
        )
        ctx.model_ir.tensors[slope_tensor_name].quantization = slope_qparams
        ctx.model_ir.tensors[output_name].dtype = input_dtype
        in_quant = ctx.model_ir.tensors[input_name].quantization
        if in_quant is not None and ctx.model_ir.tensors[output_name].quantization is None:
            ctx.model_ir.tensors[output_name].quantization = _clone_quantization(in_quant)
    else:
        slope_tensor_name = ctx.add_const_tensor(
            f"{node.name}_prelu_alpha",
            np.asarray(slope_f, dtype=np.float32),
        )

    ctx.add_operator(
        OperatorIR(
            op_type="PRELU",
            inputs=[input_name, slope_tensor_name],
            outputs=[output_name],
        )
    )
