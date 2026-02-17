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
