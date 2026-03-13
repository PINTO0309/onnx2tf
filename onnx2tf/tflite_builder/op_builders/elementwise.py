from __future__ import annotations

from typing import Any
import math
import copy

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.shared import (
    make_transpose,
    materialize_broadcast_operand_for_gpu_delegate,
)


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


def _sync_shape_signature_from_src(
    *,
    ctx: Any,
    src_tensor_name: str,
    dst_tensor_name: str,
) -> None:
    """
    Force shape/signature sync for shape-preserving unary ops.

    Some preprocess rewrites insert synthetic tensors whose stale shape_map can
    conflict with runtime topology. For unary ops that preserve shape exactly,
    source tensor metadata is the authoritative shape.
    """
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src = ctx.model_ir.tensors[src_tensor_name]
    dst = ctx.model_ir.tensors[dst_tensor_name]
    src_signature = (
        list(src.shape_signature)
        if src.shape_signature is not None
        else list(src.shape)
    )
    dst.shape = [int(v) for v in list(src.shape)]
    dst.shape_signature = [int(v) for v in list(src_signature)]


def _tensor_shape_and_signature(
    *,
    tensor_name: str,
    ctx: Any,
) -> tuple[list[int], list[int]]:
    shape = [int(v) for v in list(ctx.get_tensor_shape(tensor_name))]
    tensor = ctx.model_ir.tensors.get(str(tensor_name), None)
    if (
        tensor is not None
        and tensor.shape_signature is not None
        and len(list(tensor.shape_signature)) == len(shape)
    ):
        return shape, [int(v) for v in list(tensor.shape_signature)]
    return shape, [int(v) for v in list(shape)]


def _broadcast_shape_signatures(
    signature_a: list[int],
    signature_b: list[int],
) -> list[int] | None:
    a = [int(v) for v in list(signature_a)]
    b = [int(v) for v in list(signature_b)]
    rank = int(max(len(a), len(b)))
    a = [1] * (rank - len(a)) + a
    b = [1] * (rank - len(b)) + b
    out: list[int] = []
    for dim_a, dim_b in zip(a, b):
        if int(dim_a) == int(dim_b):
            out.append(int(dim_a))
            continue
        if int(dim_a) == 1:
            out.append(-1 if int(dim_b) < 0 else int(dim_b))
            continue
        if int(dim_b) == 1:
            out.append(-1 if int(dim_a) < 0 else int(dim_a))
            continue
        if int(dim_a) < 0 and int(dim_b) < 0:
            out.append(-1)
            continue
        if int(dim_a) < 0 and int(dim_b) > 1:
            out.append(int(dim_b))
            continue
        if int(dim_b) < 0 and int(dim_a) > 1:
            out.append(int(dim_a))
            continue
        return None
    return out


def _materialize_shape_from_signature(shape_signature: list[int]) -> list[int]:
    return [int(v) if int(v) > 0 else 1 for v in list(shape_signature)]


def _normalize_axis_for_rank(axis: int, rank: int) -> int:
    a = int(axis)
    if a < 0:
        a += int(rank)
    if a < 0 or a >= int(rank):
        raise NotImplementedError(f"axis is out of range. axis={axis} normalized={a} rank={rank}")
    return int(a)


def _get_main_onnx_opset(ctx: Any) -> int | None:
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None:
        return None
    for opset in getattr(onnx_model, "opset_import", []):
        domain = str(getattr(opset, "domain", ""))
        if domain in {"", "ai.onnx"}:
            try:
                return int(opset.version)
            except Exception:
                return None
    return None


def _resolve_softmax_axis(node: Any, ctx: Any, rank: int) -> int:
    if "axis" in node.attrs:
        raw_axis = int(node.attrs["axis"])
    else:
        opset = _get_main_onnx_opset(ctx)
        raw_axis = -1 if opset is not None and int(opset) >= 13 else 1
    return _normalize_axis_for_rank(axis=raw_axis, rank=rank)


def _axis_to_last_permutations(axis: int, rank: int) -> tuple[list[int], list[int]]:
    perm_to_last = [int(v) for v in range(rank) if int(v) != int(axis)] + [int(axis)]
    perm_from_last = [0] * int(rank)
    for out_axis, in_axis in enumerate(perm_to_last):
        perm_from_last[int(in_axis)] = int(out_axis)
    return perm_to_last, perm_from_last


_FLOAT_TENSOR_DTYPES = {"FLOAT16", "FLOAT32"}


def _numpy_dtype_for_tensor(tflite_dtype: str) -> np.dtype:
    dt = str(tflite_dtype).upper()
    if dt == "BOOL":
        return np.dtype(np.bool_)
    if dt == "INT8":
        return np.dtype(np.int8)
    if dt == "UINT8":
        return np.dtype(np.uint8)
    if dt == "INT16":
        return np.dtype(np.int16)
    if dt == "UINT16":
        return np.dtype(np.uint16)
    if dt == "INT32":
        return np.dtype(np.int32)
    if dt == "UINT32":
        return np.dtype(np.uint32)
    if dt == "INT64":
        return np.dtype(np.int64)
    if dt == "UINT64":
        return np.dtype(np.uint64)
    if dt == "FLOAT16":
        return np.dtype(np.float16)
    if dt == "FLOAT32":
        return np.dtype(np.float32)
    if dt == "FLOAT64":
        return np.dtype(np.float64)
    raise NotImplementedError(f"Unsupported dtype for clip lowering: dtype={tflite_dtype}")


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


def _add_slice_last_axis(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    input_shape: list[int],
    input_signature: list[int],
    begin_last: int,
    size_last: int,
) -> None:
    rank = int(len(input_shape))
    begin = [0] * rank
    size = [-1] * rank
    begin[-1] = int(begin_last)
    size[-1] = int(size_last)
    begin_name = ctx.add_const_tensor(
        f"{output_name}_slice_begin",
        np.asarray(begin, dtype=np.int32),
    )
    size_name = ctx.add_const_tensor(
        f"{output_name}_slice_size",
        np.asarray(size, dtype=np.int32),
    )
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.shape = [int(v) for v in list(input_shape[:-1])] + [int(size_last)]
        output_tensor.shape_signature = [int(v) for v in list(input_signature[:-1])] + [int(size_last)]
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[input_name, begin_name, size_name],
            outputs=[output_name],
        )
    )


def _add_binary_tensor_op(
    *,
    ctx: Any,
    op_type: str,
    lhs_name: str,
    rhs_name: str,
    output_name: str,
) -> None:
    options = {"fusedActivationFunction": "NONE"} if op_type in {"ADD", "SUB", "MUL", "DIV"} else {}
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[lhs_name, rhs_name],
            outputs=[output_name],
            options=options,
        )
    )


def _sanitize_where_arithmetic_operand_if_constant_nonfinite(
    *,
    ctx: Any,
    operand_name: str,
    output_dtype: str,
    base_name: str,
) -> str:
    """
    Arithmetic Where fallback uses MUL-based masking and can produce NaN for
    0 * (+/-inf). If operand is a constant float tensor with non-finite values,
    replace them with finite sentinels before MUL.
    """
    tensor = ctx.model_ir.tensors.get(str(operand_name), None)
    if tensor is None or tensor.data is None:
        return str(operand_name)

    data = np.asarray(tensor.data)
    if not np.issubdtype(data.dtype, np.floating):
        return str(operand_name)
    if not bool(np.any(~np.isfinite(data))):
        return str(operand_name)

    if str(output_dtype).upper() == "FLOAT16":
        finfo = np.finfo(np.float16)
        data_f16 = data.astype(np.float16, copy=False)
        sanitized = np.where(np.isnan(data_f16), np.float16(0.0), data_f16)
        sanitized = np.where(np.isposinf(sanitized), np.float16(finfo.max), sanitized)
        sanitized = np.where(np.isneginf(sanitized), np.float16(finfo.min), sanitized)
        return ctx.add_const_tensor(base_name, np.asarray(sanitized, dtype=np.float16))

    finfo = np.finfo(np.float32)
    data_f32 = data.astype(np.float32, copy=False)
    sanitized = np.where(np.isnan(data_f32), np.float32(0.0), data_f32)
    sanitized = np.where(np.isposinf(sanitized), np.float32(finfo.max), sanitized)
    sanitized = np.where(np.isneginf(sanitized), np.float32(finfo.min), sanitized)
    return ctx.add_const_tensor(base_name, np.asarray(sanitized, dtype=np.float32))


def _add_ones_like_tensor(
    ctx: Any,
    *,
    reference_name: str,
    base_name: str,
    dtype: str,
) -> str:
    ref_shape = [int(v) for v in ctx.get_tensor_shape(reference_name)]
    target_dtype = str(dtype).upper()

    # Prefer exact-shape ones for fully known static tensors so ATAN2 does not
    # rely on broadcast when it is unnecessary.
    if len(ref_shape) > 0 and all(int(dim) > 0 for dim in ref_shape):
        np_dtype = np.float16 if target_dtype == "FLOAT16" else np.float32
        return ctx.add_const_tensor(
            base_name,
            np.ones(ref_shape, dtype=np_dtype),
        )

    # Dynamic extents: keep rank-matched singleton shape to satisfy ATAN2 rank
    # checks while avoiding redundant full-size materialization.
    np_dtype = np.float16 if target_dtype == "FLOAT16" else np.float32
    singleton_shape = [1 for _ in ref_shape] if len(ref_shape) > 0 else [1]
    return ctx.add_const_tensor(
        base_name,
        np.ones(singleton_shape, dtype=np_dtype),
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


def _raw_onnx_attr_int(node: Any, name: str) -> int | None:
    for attr in getattr(node, "attribute", []):
        if str(getattr(attr, "name", "")) != str(name):
            continue
        try:
            return int(attr.i)
        except Exception:
            return None
    return None


def _div_const_has_integer_cast_descendant(
    *,
    ctx: Any,
    tensor_name: str,
    max_depth: int = 6,
) -> bool:
    """
    Keep explicit DIV when a constant-division result later crosses an integer
    CAST through a short arithmetic-only chain.

    Reciprocal-MUL fusion is numerically safe for most float paths, but it can
    change which side of an integer boundary a coordinate lands on after a few
    affine ops. That breaks descriptor/index sampling even when the float output
    difference is tiny.
    """
    arithmetic_ops = {"Add", "Sub", "Mul", "Div"}
    visited: set[tuple[str, int]] = set()
    stack: list[tuple[str, int]] = [(str(tensor_name), 0)]

    while len(stack) > 0:
        current_name, depth = stack.pop()
        state = (str(current_name), int(depth))
        if state in visited:
            continue
        visited.add(state)

        for consumer in ctx.onnx_tensor_consumers.get(str(current_name), []):
            op_type = str(getattr(consumer, "op_type", ""))
            if op_type == "Cast":
                cast_to = _raw_onnx_attr_int(consumer, "to")
                if cast_to in {3, 5, 6, 7, 12, 13}:
                    return True
                continue
            if int(depth) >= int(max_depth):
                continue
            if op_type not in arithmetic_ops:
                continue
            for output_name in getattr(consumer, "output", []):
                if str(output_name).strip() != "":
                    stack.append((str(output_name), int(depth) + 1))
    return False


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

    # Infer broadcast output metadata eagerly so downstream ops (e.g. MatMul)
    # do not branch on unresolved rank-1 placeholders.
    inferred_signature: list[int] | None = None
    for input_name in input_names:
        _, input_signature = _tensor_shape_and_signature(
            tensor_name=str(input_name),
            ctx=ctx,
        )
        if inferred_signature is None:
            inferred_signature = [int(v) for v in list(input_signature)]
            continue
        inferred_signature = _broadcast_shape_signatures(inferred_signature, input_signature)
        if inferred_signature is None:
            break
    if inferred_signature is not None:
        output_tensor = ctx.model_ir.tensors.get(str(output_name), None)
        if output_tensor is not None:
            inferred_shape = _materialize_shape_from_signature(inferred_signature)
            current_shape = [int(v) for v in list(output_tensor.shape)]
            current_signature = (
                [int(v) for v in list(output_tensor.shape_signature)]
                if output_tensor.shape_signature is not None
                else [int(v) for v in list(current_shape)]
            )
            should_update = (
                current_shape == [1]
                or len(current_shape) != len(inferred_shape)
                or len(current_signature) != len(inferred_signature)
                or any(
                    int(inferred_signature[idx]) > 0 and int(current_signature[idx]) <= 0
                    for idx in range(len(inferred_signature))
                )
            )
            if should_update:
                output_tensor.shape = [int(v) for v in list(inferred_shape)]
                output_tensor.shape_signature = [int(v) for v in list(inferred_signature)]

    output_tensor = ctx.model_ir.tensors.get(str(output_name), None)
    broadcast_target_shape = (
        [int(v) for v in list(output_tensor.shape)]
        if output_tensor is not None
        else [int(v) for v in ctx.get_tensor_shape(output_name)]
    )
    broadcast_target_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in list(broadcast_target_shape)]
    )
    if len(input_names) == 2:
        input_names = [
            materialize_broadcast_operand_for_gpu_delegate(
                ctx=ctx,
                input_name=str(input_name),
                target_shape=broadcast_target_shape,
                target_signature=broadcast_target_signature,
                base_name=f"{output_name}_{idx}",
            )
            for idx, input_name in enumerate(input_names)
        ]

    def _normalize_binary_dtype(dtype: str) -> str:
        dt = str(dtype).upper()
        if dt == "INT64":
            return "INT32"
        if dt == "UINT64":
            return "UINT32"
        return dt

    integer_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64"}
    unsigned_integer_dtypes = {"UINT8", "UINT16", "UINT32", "UINT64"}
    float_dtypes = {"FLOAT16", "FLOAT32", "FLOAT64"}
    logical_binary_ops = {"LOGICAL_AND", "LOGICAL_OR", "LOGICAL_XOR"}
    comparison_binary_ops = {
        "EQUAL",
        "NOT_EQUAL",
        "GREATER",
        "GREATER_EQUAL",
        "LESS",
        "LESS_EQUAL",
    }
    upper_op_type = str(op_type).upper()
    normalized_input_dtypes = [
        _normalize_binary_dtype(str(ctx.get_tensor_dtype(name)).upper())
        for name in input_names
    ]
    requested_output_dtype = _normalize_binary_dtype(str(ctx.get_tensor_dtype(output_name)).upper())
    input_target_dtype = requested_output_dtype
    output_target_dtype = requested_output_dtype

    def _promote_integer_target_dtype() -> str:
        raw_dtypes = [str(ctx.get_tensor_dtype(name)).upper() for name in input_names]
        if len(raw_dtypes) > 0 and all(dt in unsigned_integer_dtypes for dt in raw_dtypes):
            return "UINT32"
        return "INT32"

    if upper_op_type in logical_binary_ops:
        input_target_dtype = "BOOL"
        output_target_dtype = "BOOL"
    elif upper_op_type in comparison_binary_ops:
        output_target_dtype = "BOOL"

    if (
        len(normalized_input_dtypes) > 0
        and (
            upper_op_type in comparison_binary_ops
            or any(dt != input_target_dtype for dt in normalized_input_dtypes)
        )
    ):
        unique_input_dtypes = set(normalized_input_dtypes)
        if upper_op_type in logical_binary_ops:
            input_target_dtype = "BOOL"
        elif upper_op_type in comparison_binary_ops:
            if len(unique_input_dtypes) == 1:
                input_target_dtype = normalized_input_dtypes[0]
                # LiteRT comparison kernels do not accept FLOAT64. Normalize
                # homogeneous float64 comparisons to FLOAT32.
                if input_target_dtype == "FLOAT64":
                    input_target_dtype = "FLOAT32"
            elif all(str(ctx.get_tensor_dtype(name)).upper() in integer_dtypes for name in input_names):
                input_target_dtype = _promote_integer_target_dtype()
            elif all(str(ctx.get_tensor_dtype(name)).upper() in float_dtypes for name in input_names):
                input_target_dtype = "FLOAT32"
            else:
                raise NotImplementedError(
                    "Comparison op input dtypes must be compatible in flatbuffer_direct. "
                    f"op={node.name} op_type={op_type} input_dtypes={normalized_input_dtypes}"
                )
        elif len(unique_input_dtypes) == 1:
            input_target_dtype = normalized_input_dtypes[0]
        elif all(str(ctx.get_tensor_dtype(name)).upper() in integer_dtypes for name in input_names):
            input_target_dtype = _promote_integer_target_dtype()
        elif all(str(ctx.get_tensor_dtype(name)).upper() in float_dtypes for name in input_names):
            input_target_dtype = "FLOAT32"
        else:
            raise NotImplementedError(
                "Binary op input dtypes must be compatible in flatbuffer_direct. "
                f"op={node.name} op_type={op_type} input_dtypes={normalized_input_dtypes} "
                f"output_dtype={output_target_dtype}"
            )
    if upper_op_type not in comparison_binary_ops and upper_op_type not in logical_binary_ops:
        output_target_dtype = input_target_dtype

    casted_input_names: list[str] = []
    for idx, name in enumerate(input_names):
        input_dtype = str(ctx.get_tensor_dtype(name)).upper()
        normalized_input_dtype = _normalize_binary_dtype(input_dtype)
        if normalized_input_dtype == input_target_dtype and input_dtype == input_target_dtype:
            casted_input_names.append(name)
            continue
        cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_{op_type.lower()}_input{idx}_{input_target_dtype.lower()}",
            dtype=input_target_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(name)],
        )
        src_tensor = ctx.model_ir.tensors.get(name, None)
        cast_tensor = ctx.model_ir.tensors.get(cast_name, None)
        if src_tensor is not None and cast_tensor is not None:
            cast_tensor.shape_signature = (
                [int(v) for v in list(src_tensor.shape_signature)]
                if src_tensor.shape_signature is not None
                else [int(v) for v in list(src_tensor.shape)]
            )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[name],
                outputs=[cast_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": input_target_dtype,
                },
            )
        )
        casted_input_names.append(cast_name)

    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.dtype = output_target_dtype
        if len(casted_input_names) > 0:
            ref_tensor = ctx.model_ir.tensors.get(casted_input_names[0], None)
            if ref_tensor is not None:
                output_tensor.quantization = _clone_quantization(ref_tensor.quantization)
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(output_name)] = output_target_dtype

    options = {"fusedActivationFunction": "NONE"}
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=casted_input_names,
            outputs=[output_name],
            options=options,
        )
    )


def _build_nary_minmax_op(
    *,
    node: Any,
    ctx: Any,
    op_type: str,
) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    if len(input_names) < 2:
        raise NotImplementedError(
            f"{op_type.title()} requires at least 2 inputs in flatbuffer_direct. op={node.name}"
        )
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_names[0], output_name)

    def _normalize_minmax_dtype(dtype: str) -> str:
        dt = str(dtype).upper()
        if dt == "INT64":
            return "INT32"
        if dt == "UINT64":
            return "UINT32"
        return dt

    integer_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64"}
    unsigned_integer_dtypes = {"UINT8", "UINT16", "UINT32", "UINT64"}
    float_dtypes = {"FLOAT16", "FLOAT32", "FLOAT64"}

    normalized_input_dtypes = [
        _normalize_minmax_dtype(str(ctx.get_tensor_dtype(name)).upper())
        for name in input_names
    ]
    unique_input_dtypes = set(normalized_input_dtypes)
    if len(unique_input_dtypes) == 1:
        target_dtype = normalized_input_dtypes[0]
    elif all(str(ctx.get_tensor_dtype(name)).upper() in integer_dtypes for name in input_names):
        raw_dtypes = [str(ctx.get_tensor_dtype(name)).upper() for name in input_names]
        target_dtype = "UINT32" if all(dt in unsigned_integer_dtypes for dt in raw_dtypes) else "INT32"
    elif all(str(ctx.get_tensor_dtype(name)).upper() in float_dtypes for name in input_names):
        target_dtype = "FLOAT32"
    else:
        raise NotImplementedError(
            f"{op_type.title()} input dtypes must be compatible in flatbuffer_direct. "
            f"op={node.name} input_dtypes={normalized_input_dtypes}"
        )

    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in output_shape]
    )
    casted_input_names: list[str] = []
    for idx, name in enumerate(input_names):
        input_dtype = str(ctx.get_tensor_dtype(name)).upper()
        normalized_input_dtype = _normalize_minmax_dtype(input_dtype)
        if normalized_input_dtype == target_dtype and input_dtype == target_dtype:
            casted_input_names.append(str(name))
            continue
        cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_{str(op_type).lower()}_input{idx}_{str(target_dtype).lower()}",
            dtype=target_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(name)],
        )
        src_tensor = ctx.model_ir.tensors.get(str(name), None)
        cast_tensor = ctx.model_ir.tensors.get(str(cast_name), None)
        if src_tensor is not None and cast_tensor is not None:
            cast_tensor.shape_signature = (
                [int(v) for v in list(src_tensor.shape_signature)]
                if src_tensor.shape_signature is not None
                else [int(v) for v in list(src_tensor.shape)]
            )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[str(name)],
                outputs=[cast_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": target_dtype,
                },
            )
        )
        casted_input_names.append(cast_name)

    current_name = casted_input_names[0]
    for idx, rhs_name in enumerate(casted_input_names[1:], start=1):
        is_last = idx == int(len(input_names) - 1)
        reduced_output_name = output_name
        if not is_last:
            reduced_output_name = ctx.add_intermediate_tensor(
                f"{output_name}_{str(op_type).lower()}_{idx}",
                dtype=target_dtype,
                shape=output_shape,
            )
        ctx.add_operator(
            OperatorIR(
                op_type=op_type,
                inputs=[current_name, rhs_name],
                outputs=[reduced_output_name],
                options={},
            )
        )
        current_name = reduced_output_name

    if output_tensor is not None:
        output_tensor.dtype = target_dtype
        output_tensor.shape_signature = [int(v) for v in output_signature]
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(output_name)] = target_dtype


def build_min_op(node: Any, ctx: Any) -> None:
    _build_nary_minmax_op(
        node=node,
        ctx=ctx,
        op_type="MINIMUM",
    )


def build_max_op(node: Any, ctx: Any) -> None:
    _build_nary_minmax_op(
        node=node,
        ctx=ctx,
        op_type="MAXIMUM",
    )


def build_sum_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    if len(input_names) < 2:
        raise NotImplementedError(
            f"Sum requires at least 2 inputs in flatbuffer_direct. op={node.name}"
        )
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_names[0], output_name)

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in output_shape]
    )

    current_name = input_names[0]
    for idx, rhs_name in enumerate(input_names[1:], start=1):
        is_last = idx == int(len(input_names) - 1)
        add_output_name = output_name
        if not is_last:
            add_output_name = ctx.add_intermediate_tensor(
                f"{output_name}_sum_{idx}",
                dtype=output_dtype,
                shape=output_shape,
            )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[current_name, rhs_name],
                outputs=[add_output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        current_name = add_output_name

    if output_tensor is not None:
        output_tensor.shape_signature = [int(v) for v in output_signature]


def build_mean_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    if len(input_names) < 1:
        raise NotImplementedError(
            f"Mean requires at least 1 input in flatbuffer_direct. op={node.name}"
        )
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_names[0], output_name)

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in output_shape]
    )

    casted_inputs = [
        _cast_tensor_if_needed(
            ctx=ctx,
            src_name=name,
            dst_dtype=compute_dtype,
            base_name=f"{output_name}_mean_input_{idx}_cast",
        )
        for idx, name in enumerate(input_names)
    ]

    current_name = casted_inputs[0]
    for idx, rhs_name in enumerate(casted_inputs[1:], start=1):
        add_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_mean_add_{idx}",
            dtype=compute_dtype,
            shape=output_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[current_name, rhs_name],
                outputs=[add_output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        current_name = add_output_name

    div_output_name = output_name
    if output_dtype != compute_dtype:
        div_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_mean_div",
            dtype=compute_dtype,
            shape=output_shape,
        )
    divisor_name = _add_scalar_const(
        ctx,
        f"{output_name}_mean_divisor",
        float(len(casted_inputs)),
        compute_dtype,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[current_name, divisor_name],
            outputs=[div_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    if output_dtype != compute_dtype:
        _finalize_float_compute_output(
            ctx=ctx,
            compute_output_name=div_output_name,
            output_name=output_name,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )
    if output_tensor is not None:
        output_tensor.shape_signature = [int(v) for v in output_signature]


def _det_tensor_shape_with_signature(ctx: Any, tensor_name: str) -> list[int]:
    tensor = ctx.model_ir.tensors.get(tensor_name, None)
    if tensor is not None and tensor.shape_signature is not None:
        return [int(v) for v in list(tensor.shape_signature)]
    return [int(v) for v in list(ctx.get_tensor_shape(tensor_name))]


def _det_gather_scalar(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    axis: int,
    index: int,
    dtype: str,
) -> str:
    input_sig = _det_tensor_shape_with_signature(ctx, input_name)
    rank = len(input_sig)
    normalized_axis = int(axis if axis >= 0 else axis + rank)
    output_sig = [int(v) for i, v in enumerate(input_sig) if i != normalized_axis]
    output_shape = [int(v) if int(v) > 0 else 1 for v in output_sig]
    gathered_shape = (
        [int(v) if i != normalized_axis else 1 for i, v in enumerate(input_sig)]
        if len(input_sig) > 0
        else [1]
    )
    gathered_name = ctx.add_intermediate_tensor(
        f"{output_name}_gathered",
        dtype=dtype,
        shape=[int(v) if int(v) > 0 else 1 for v in gathered_shape],
    )
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is None:
        ctx.add_intermediate_tensor(
            output_name,
            dtype=dtype,
            shape=output_shape,
        )
        output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.shape_signature = [int(v) for v in output_sig]
    index_name = ctx.add_const_tensor(
        f"{output_name}_index",
        np.asarray(int(index), dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[input_name, index_name],
            outputs=[gathered_name],
            options={"axis": int(normalized_axis), "batchDims": 0},
        )
    )
    shape_name = ctx.add_const_tensor(
        f"{output_name}_shape",
        np.asarray([int(v) for v in output_shape], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[gathered_name, shape_name],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape]},
        )
    )
    return str(output_name)


def build_det_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    input_sig = _det_tensor_shape_with_signature(ctx, input_name)
    rank = len(input_sig)
    matrix_dim = int(input_sig[-1])

    row0_name = _det_gather_scalar(
        ctx=ctx,
        input_name=input_name,
        output_name=f"{node.name}_det_row0",
        axis=rank - 2,
        index=0,
        dtype=input_dtype,
    )
    row1_name = _det_gather_scalar(
        ctx=ctx,
        input_name=input_name,
        output_name=f"{node.name}_det_row1",
        axis=rank - 2,
        index=1,
        dtype=input_dtype,
    )

    def _mul(lhs: str, rhs: str, suffix: str) -> str:
        name = ctx.add_intermediate_tensor(
            f"{node.name}_{suffix}",
            dtype=input_dtype,
            shape=ctx.get_tensor_shape(output_name),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[lhs, rhs],
                outputs=[name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return str(name)

    def _sub(lhs: str, rhs: str, suffix: str) -> str:
        name = ctx.add_intermediate_tensor(
            f"{node.name}_{suffix}",
            dtype=input_dtype,
            shape=ctx.get_tensor_shape(output_name),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[lhs, rhs],
                outputs=[name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return str(name)

    a_name = _det_gather_scalar(ctx=ctx, input_name=row0_name, output_name=f"{node.name}_det_a", axis=rank - 2, index=0, dtype=input_dtype)
    b_name = _det_gather_scalar(ctx=ctx, input_name=row0_name, output_name=f"{node.name}_det_b", axis=rank - 2, index=1, dtype=input_dtype)
    c_name = _det_gather_scalar(ctx=ctx, input_name=row1_name, output_name=f"{node.name}_det_c", axis=rank - 2, index=0, dtype=input_dtype)
    d_name = _det_gather_scalar(ctx=ctx, input_name=row1_name, output_name=f"{node.name}_det_d", axis=rank - 2, index=1, dtype=input_dtype)

    if matrix_dim == 2:
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[_mul(a_name, d_name, "det_ad"), _mul(b_name, c_name, "det_bc")],
                outputs=[output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return

    row2_name = _det_gather_scalar(
        ctx=ctx,
        input_name=input_name,
        output_name=f"{node.name}_det_row2",
        axis=rank - 2,
        index=2,
        dtype=input_dtype,
    )
    c01_name = _det_gather_scalar(ctx=ctx, input_name=row0_name, output_name=f"{node.name}_det_c01", axis=rank - 2, index=2, dtype=input_dtype)
    e_name = _det_gather_scalar(ctx=ctx, input_name=row1_name, output_name=f"{node.name}_det_e", axis=rank - 2, index=1, dtype=input_dtype)
    f_name = _det_gather_scalar(ctx=ctx, input_name=row1_name, output_name=f"{node.name}_det_f", axis=rank - 2, index=2, dtype=input_dtype)
    g_name = _det_gather_scalar(ctx=ctx, input_name=row2_name, output_name=f"{node.name}_det_g", axis=rank - 2, index=0, dtype=input_dtype)
    h_name = _det_gather_scalar(ctx=ctx, input_name=row2_name, output_name=f"{node.name}_det_h", axis=rank - 2, index=1, dtype=input_dtype)
    i_name = _det_gather_scalar(ctx=ctx, input_name=row2_name, output_name=f"{node.name}_det_i", axis=rank - 2, index=2, dtype=input_dtype)

    term1_name = _mul(a_name, _sub(_mul(e_name, i_name, "det_ei"), _mul(f_name, h_name, "det_fh"), "det_m1"), "det_t1")
    term2_name = _mul(b_name, _sub(_mul(c_name, i_name, "det_ci"), _mul(f_name, g_name, "det_fg"), "det_m2"), "det_t2")
    term3_name = _mul(c01_name, _sub(_mul(c_name, h_name, "det_ch"), _mul(e_name, g_name, "det_eg"), "det_m3"), "det_t3")
    partial_name = _sub(term1_name, term2_name, "det_partial")
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[partial_name, term3_name],
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
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
    rhs_const = ctx.get_constant_array(input_names[1])
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

    # Prefer exact square to avoid POW numerical drift for x^2.
    if rhs_const is not None:
        rhs_arr = np.asarray(rhs_const)
        if rhs_arr.size > 0 and np.all(np.isfinite(rhs_arr)):
            rhs_f32 = rhs_arr.astype(np.float32, copy=False)
            if bool(np.all(np.isclose(rhs_f32, 2.0, rtol=0.0, atol=1e-6))):
                mul_output_name = output_name
                if output_dtype != compute_dtype:
                    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
                    mul_output_name = ctx.add_intermediate_tensor(
                        f"{output_name}_pow_mul_out",
                        dtype=compute_dtype,
                        shape=output_shape,
                    )
                ctx.add_operator(
                    OperatorIR(
                        op_type="MUL",
                        inputs=[lhs_name, lhs_name],
                        outputs=[mul_output_name],
                        options={"fusedActivationFunction": "NONE"},
                    )
                )
                if mul_output_name != output_name:
                    ctx.add_operator(
                        OperatorIR(
                            op_type="CAST",
                            inputs=[mul_output_name],
                            outputs=[output_name],
                            options={
                                "inDataType": compute_dtype,
                                "outDataType": output_dtype,
                            },
                        )
                    )
                return

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
    integer_output_dtypes = {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    }
    if rhs_const is not None:
        calc_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"
        np_calc_dtype = np.float16 if calc_dtype == "FLOAT16" else np.float32
        preserve_exact_div = output_dtype in integer_output_dtypes
        if (
            not preserve_exact_div
            and output_dtype in _FLOAT_TENSOR_DTYPES
            and _div_const_has_integer_cast_descendant(
                ctx=ctx,
                tensor_name=output_name,
            )
        ):
            preserve_exact_div = True
        if preserve_exact_div:
            div_lhs_name = lhs_name
            if lhs_dtype != calc_dtype:
                lhs_shape = [int(v) for v in ctx.get_tensor_shape(lhs_name)]
                div_lhs_name = ctx.add_intermediate_tensor(
                    f"{output_name}_div_lhs_cast",
                    dtype=calc_dtype,
                    shape=lhs_shape,
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[lhs_name],
                        outputs=[div_lhs_name],
                        options={
                            "inDataType": lhs_dtype,
                            "outDataType": calc_dtype,
                        },
                    )
                )

            rhs_cast_name = ctx.add_const_tensor(
                f"{output_name}_div_rhs_cast",
                np.asarray(rhs_const, dtype=np_calc_dtype),
            )

            div_out_name = output_name
            if output_dtype != calc_dtype:
                output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
                div_out_name = ctx.add_intermediate_tensor(
                    f"{output_name}_div_out",
                    dtype=calc_dtype,
                    shape=output_shape,
                )

            ctx.add_operator(
                OperatorIR(
                    op_type="DIV",
                    inputs=[div_lhs_name, rhs_cast_name],
                    outputs=[div_out_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )

            if div_out_name != output_name:
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[div_out_name],
                        outputs=[output_name],
                        options={
                            "inDataType": calc_dtype,
                            "outDataType": output_dtype,
                        },
                    )
                )
            return

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

    lhs_name = input_names[0]
    rhs_name = input_names[1]
    lhs_dtype = str(ctx.get_tensor_dtype(lhs_name)).upper()
    rhs_dtype = str(ctx.get_tensor_dtype(rhs_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()

    def _normalize_mod_dtype(dtype: str) -> str:
        dt = str(dtype).upper()
        if dt == "INT64":
            return "INT32"
        if dt == "UINT64":
            return "UINT32"
        return dt

    integer_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64"}
    unsigned_integer_dtypes = {"UINT8", "UINT16", "UINT32", "UINT64"}
    float_dtypes = {"FLOAT16", "FLOAT32", "FLOAT64"}
    lhs_norm = _normalize_mod_dtype(lhs_dtype)
    rhs_norm = _normalize_mod_dtype(rhs_dtype)
    out_norm = _normalize_mod_dtype(output_dtype)

    target_dtype = out_norm
    if lhs_norm == rhs_norm:
        target_dtype = lhs_norm
    elif lhs_dtype in integer_dtypes and rhs_dtype in integer_dtypes:
        if lhs_dtype in unsigned_integer_dtypes and rhs_dtype in unsigned_integer_dtypes:
            target_dtype = "UINT32"
        else:
            target_dtype = "INT32"
    elif lhs_dtype in float_dtypes and rhs_dtype in float_dtypes:
        target_dtype = "FLOAT32"
    else:
        raise NotImplementedError(
            "Mod input dtypes must be compatible in flatbuffer_direct. "
            f"op={node.name} lhs_dtype={lhs_dtype} rhs_dtype={rhs_dtype} output_dtype={output_dtype}"
        )

    lhs_name = _cast_tensor_if_needed(
        ctx=ctx,
        src_name=lhs_name,
        dst_dtype=target_dtype,
        base_name=f"{output_name}_mod_lhs_cast",
    )
    rhs_name = _cast_tensor_if_needed(
        ctx=ctx,
        src_name=rhs_name,
        dst_dtype=target_dtype,
        base_name=f"{output_name}_mod_rhs_cast",
    )

    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.dtype = str(target_dtype)
        src_tensor = ctx.model_ir.tensors.get(lhs_name, None)
        if src_tensor is not None:
            output_tensor.quantization = _clone_quantization(src_tensor.quantization)
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(output_name)] = str(target_dtype)

    ctx.add_operator(
        OperatorIR(
            op_type="FLOOR_MOD",
            inputs=[lhs_name, rhs_name],
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


def _is_integer_dtype(dtype: str) -> bool:
    return str(dtype).upper() in {
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
    }


def _normalize_unary_integer_work_dtype(dtype: str) -> str:
    dt = str(dtype).upper()
    if dt == "INT64":
        return "INT32"
    if dt == "UINT64":
        return "UINT32"
    return dt


def _prepare_passthrough_unary_runtime(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
) -> tuple[str, str]:
    """
    Ensure unary builtin runtime dtype compatibility.

    TFLite unary kernels require matching input/output dtypes. Prefer INT32/UINT32
    work dtypes for integer passthrough ops, then cast back to the ONNX contract
    only when needed.
    """
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    op_tag = str(getattr(node, "op_type", getattr(node, "op", "unary"))).lower()
    if input_dtype == output_dtype:
        return str(input_name), str(output_name)

    def _add_cast(
        *,
        source_name: str,
        source_dtype: str,
        target_dtype: str,
        target_name: str,
    ) -> None:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[source_name],
                outputs=[target_name],
                options={
                    "inDataType": source_dtype,
                    "outDataType": target_dtype,
                },
            )
        )

    if _is_integer_dtype(input_dtype) and _is_integer_dtype(output_dtype):
        work_dtype = _normalize_unary_integer_work_dtype(input_dtype)
        runtime_input_name = str(input_name)
        runtime_output_name = str(output_name)
        src_tensor = ctx.model_ir.tensors.get(str(input_name), None)
        if input_dtype != work_dtype:
            runtime_input_name = ctx.add_intermediate_tensor(
                f"{output_name}_{op_tag}_input_{work_dtype.lower()}",
                dtype=work_dtype,
                shape=[int(v) for v in ctx.get_tensor_shape(input_name)],
            )
            cast_tensor = ctx.model_ir.tensors.get(str(runtime_input_name), None)
            if src_tensor is not None and cast_tensor is not None:
                cast_tensor.shape_signature = (
                    [int(v) for v in list(src_tensor.shape_signature)]
                    if src_tensor.shape_signature is not None
                    else [int(v) for v in list(src_tensor.shape)]
                )
                cast_tensor.quantization = _clone_quantization(src_tensor.quantization)
            _add_cast(
                source_name=str(input_name),
                source_dtype=input_dtype,
                target_dtype=work_dtype,
                target_name=str(runtime_input_name),
            )
        if output_dtype != work_dtype:
            runtime_output_name = ctx.add_intermediate_tensor(
                f"{output_name}_{op_tag}_{work_dtype.lower()}",
                dtype=work_dtype,
                shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
            )
            runtime_output_tensor = ctx.model_ir.tensors.get(str(runtime_output_name), None)
            if runtime_output_tensor is not None:
                output_tensor = ctx.model_ir.tensors.get(str(output_name), None)
                runtime_output_tensor.shape_signature = (
                    [int(v) for v in list(output_tensor.shape_signature)]
                    if output_tensor is not None and output_tensor.shape_signature is not None
                    else [int(v) for v in ctx.get_tensor_shape(output_name)]
                )
                if src_tensor is not None:
                    runtime_output_tensor.quantization = _clone_quantization(src_tensor.quantization)
        return str(runtime_input_name), str(runtime_output_name)

    cast_name = ctx.add_intermediate_tensor(
        f"{output_name}_{op_tag}_input_cast",
        dtype=output_dtype,
        shape=[int(v) for v in ctx.get_tensor_shape(input_name)],
    )
    src_tensor = ctx.model_ir.tensors.get(str(input_name), None)
    cast_tensor = ctx.model_ir.tensors.get(str(cast_name), None)
    if src_tensor is not None and cast_tensor is not None:
        cast_tensor.shape_signature = (
            [int(v) for v in list(src_tensor.shape_signature)]
            if src_tensor.shape_signature is not None
            else [int(v) for v in list(src_tensor.shape)]
        )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[input_name],
            outputs=[cast_name],
            options={
                "inDataType": input_dtype,
                "outDataType": output_dtype,
            },
        )
    )
    return str(cast_name), str(output_name)


def _finalize_passthrough_unary_runtime(
    *,
    node: Any,
    ctx: Any,
    runtime_output_name: str,
    output_name: str,
) -> None:
    if str(runtime_output_name) == str(output_name):
        return
    runtime_output_dtype = str(ctx.get_tensor_dtype(runtime_output_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[str(runtime_output_name)],
            outputs=[str(output_name)],
            options={
                "inDataType": runtime_output_dtype,
                "outDataType": output_dtype,
            },
        )
    )


def build_unary_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)
    runtime_input_name, runtime_output_name = _prepare_passthrough_unary_runtime(
        node=node,
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
    )
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[runtime_input_name],
            outputs=[runtime_output_name],
        )
    )
    _finalize_passthrough_unary_runtime(
        node=node,
        ctx=ctx,
        runtime_output_name=runtime_output_name,
        output_name=output_name,
    )


def build_leaky_relu_op(node: Any, ctx: Any) -> None:
    (
        compute_input_name,
        compute_output_name,
        output_name,
        output_dtype,
        compute_dtype,
        _,
    ) = _prepare_float_compute(node, ctx, tag="leaky_relu")
    ctx.add_operator(
        OperatorIR(
            op_type="LEAKY_RELU",
            inputs=[compute_input_name],
            outputs=[compute_output_name],
            options={"alpha": float(node.attrs.get("alpha", 0.01))},
        )
    )
    _finalize_float_compute_output(
        ctx=ctx,
        compute_output_name=compute_output_name,
        output_name=output_name,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )


def build_thresholded_relu_op(node: Any, ctx: Any) -> None:
    (
        compute_input_name,
        compute_output_name,
        output_name,
        output_dtype,
        compute_dtype,
        out_shape,
    ) = _prepare_float_compute(node, ctx, tag="thresholded_relu")
    alpha_name = _add_scalar_const(
        ctx,
        f"{output_name}_thresholded_relu_alpha",
        float(node.attrs.get("alpha", 1.0)),
        compute_dtype,
    )
    greater_name = ctx.add_intermediate_tensor(
        f"{output_name}_thresholded_relu_greater",
        dtype="BOOL",
        shape=out_shape,
    )
    mask_name = ctx.add_intermediate_tensor(
        f"{output_name}_thresholded_relu_mask",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GREATER",
            inputs=[compute_input_name, alpha_name],
            outputs=[greater_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[greater_name],
            outputs=[mask_name],
            options={
                "inDataType": "BOOL",
                "outDataType": compute_dtype,
            },
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[compute_input_name, mask_name],
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


def build_is_nan_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)
    _sync_shape_signature_from_src(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="NOT_EQUAL",
            inputs=[input_name, input_name],
            outputs=[output_name],
        )
    )


def build_is_inf_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)
    _sync_shape_signature_from_src(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    out_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    np_dtype = np.float16 if input_dtype == "FLOAT16" else np.float32
    detect_negative = bool(int(node.attrs.get("detect_negative", 1)))
    detect_positive = bool(int(node.attrs.get("detect_positive", 1)))

    if detect_negative and detect_positive:
        abs_name = ctx.add_intermediate_tensor(
            f"{output_name}_is_inf_abs",
            dtype=input_dtype,
            shape=out_shape,
        )
        inf_name = ctx.add_const_tensor(
            f"{output_name}_is_inf_abs_inf",
            np.asarray(np.inf, dtype=np_dtype),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ABS",
                inputs=[input_name],
                outputs=[abs_name],
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EQUAL",
                inputs=[abs_name, inf_name],
                outputs=[output_name],
            )
        )
        return

    if detect_positive or detect_negative:
        inf_name = ctx.add_const_tensor(
            f"{output_name}_is_inf_target",
            np.asarray(np.inf if detect_positive else -np.inf, dtype=np_dtype),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EQUAL",
                inputs=[input_name, inf_name],
                outputs=[output_name],
            )
        )
        return

    zero_name = ctx.add_const_tensor(
        f"{output_name}_is_inf_zero",
        np.asarray(0.0, dtype=np_dtype),
    )
    less_name = ctx.add_intermediate_tensor(
        f"{output_name}_is_inf_less",
        dtype="BOOL",
        shape=out_shape,
    )
    greater_name = ctx.add_intermediate_tensor(
        f"{output_name}_is_inf_greater",
        dtype="BOOL",
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LESS",
            inputs=[input_name, zero_name],
            outputs=[less_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GREATER",
            inputs=[input_name, zero_name],
            outputs=[greater_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LOGICAL_AND",
            inputs=[less_name, greater_name],
            outputs=[output_name],
        )
    )


def build_inverse_op(node: Any, ctx: Any) -> None:
    (
        compute_input_name,
        compute_output_name,
        output_name,
        output_dtype,
        compute_dtype,
        output_shape,
    ) = _prepare_float_compute(node, ctx, tag="inverse")

    input_shape = [int(v) for v in ctx.get_tensor_shape(compute_input_name)]
    rank = len(input_shape)
    if rank < 2:
        raise NotImplementedError(
            f"Inverse requires input/output rank >= 2 in flatbuffer_direct. op={node.name}"
        )

    input_tensor = ctx.model_ir.tensors.get(compute_input_name, None)
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor is not None and input_tensor.shape_signature is not None
        else [int(v) for v in input_shape]
    )
    row_sig = int(input_signature[-2])
    col_sig = int(input_signature[-1])
    row_known = row_sig > 0
    col_known = col_sig > 0
    rows = int(row_sig if row_known else input_shape[-2])
    cols = int(col_sig if col_known else input_shape[-1])
    if row_known and not col_known:
        cols = int(rows)
    elif col_known and not row_known:
        rows = int(cols)
    if rows != cols or rows < 1:
        raise NotImplementedError(
            "Inverse currently supports only square matrix last dims [N,N] with N >= 1 in "
            f"flatbuffer_direct. op={node.name} input_shape={input_shape} input_signature={input_signature}"
        )
    if rows > 16:
        raise NotImplementedError(
            "Inverse currently supports matrix last dims up to [16,16] in "
            f"flatbuffer_direct. op={node.name} input_shape={input_shape} input_signature={input_signature}"
        )

    matrix_signature = [int(v) for v in input_signature[:-2]] + [rows, cols]
    matrix_shape = [int(v) for v in input_shape[:-2]] + [rows, cols]
    for target_name in [compute_output_name, output_name]:
        target_tensor = ctx.model_ir.tensors.get(target_name, None)
        if target_tensor is None:
            continue
        target_tensor.shape = [int(v) for v in matrix_shape]
        target_tensor.shape_signature = [int(v) for v in matrix_signature]

    prefix_shape = [int(v) for v in matrix_shape[:-2]]
    scalar_shape = prefix_shape + [1, 1]
    row_shape = prefix_shape + [1, cols]

    def _slice_matrix_entry(row: int, col: int) -> str:
        begin = [0 for _ in range(rank)]
        size = [-1 for _ in range(rank)]
        begin[-2] = int(row)
        begin[-1] = int(col)
        size[-2] = 1
        size[-1] = 1
        begin_name = ctx.add_const_tensor(
            f"{compute_output_name}_inverse_begin_r{row}_c{col}",
            np.asarray(begin, dtype=np.int32),
        )
        size_name = ctx.add_const_tensor(
            f"{compute_output_name}_inverse_size_r{row}_c{col}",
            np.asarray(size, dtype=np.int32),
        )
        out_name = ctx.add_intermediate_tensor(
            f"{compute_output_name}_inverse_r{row}_c{col}",
            dtype=compute_dtype,
            shape=scalar_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[compute_input_name, begin_name, size_name],
                outputs=[out_name],
            )
        )
        return out_name

    def _binary(op_type: str, lhs: str, rhs: str, suffix: str) -> str:
        out_name = ctx.add_intermediate_tensor(
            f"{compute_output_name}_inverse_{suffix}",
            dtype=compute_dtype,
            shape=scalar_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type=op_type,
                inputs=[lhs, rhs],
                outputs=[out_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return out_name

    def _mul_sub(a_name: str, b_name: str, c_name: str, d_name: str, suffix: str) -> str:
        lhs = _binary("MUL", a_name, b_name, f"{suffix}_mul_l")
        rhs = _binary("MUL", c_name, d_name, f"{suffix}_mul_r")
        return _binary("SUB", lhs, rhs, f"{suffix}_sub")

    def _concat(input_names: list[str], axis: int, shape: list[int], suffix: str) -> str:
        out_name = ctx.add_intermediate_tensor(
            f"{compute_output_name}_inverse_{suffix}",
            dtype=compute_dtype,
            shape=shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=list(input_names),
                outputs=[out_name],
                options={"axis": int(axis), "fusedActivationFunction": "NONE"},
            )
        )
        return out_name

    def _neg(src_name: str, suffix: str) -> str:
        out_name = ctx.add_intermediate_tensor(
            f"{compute_output_name}_inverse_{suffix}",
            dtype=compute_dtype,
            shape=scalar_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="NEG",
                inputs=[src_name],
                outputs=[out_name],
            )
        )
        return out_name

    if cols > 3:
        np_dtype = np.float16 if compute_dtype == "FLOAT16" else np.float32
        prefix_ones = [1 for _ in prefix_shape]
        matrix_rank_shape = prefix_ones + [rows, cols]
        row_selector_shape = prefix_ones + [rows, 1]

        identity = np.eye(rows, dtype=np_dtype).reshape(matrix_rank_shape)
        if len(prefix_shape) > 0:
            identity = np.tile(identity, prefix_shape + [1, 1])
        identity_name = ctx.add_const_tensor(
            f"{compute_output_name}_inverse_identity",
            identity.astype(np_dtype),
        )

        eps_name = _add_scalar_const(
            ctx,
            f"{compute_output_name}_inverse_eps",
            float(1e-6),
            compute_dtype,
        )

        def _slice_matrix_row(src_name: str, row: int, suffix: str) -> str:
            begin = [0 for _ in range(rank)]
            size = [-1 for _ in range(rank)]
            begin[-2] = int(row)
            size[-2] = 1
            begin_name = ctx.add_const_tensor(
                f"{compute_output_name}_inverse_{suffix}_begin",
                np.asarray(begin, dtype=np.int32),
            )
            size_name = ctx.add_const_tensor(
                f"{compute_output_name}_inverse_{suffix}_size",
                np.asarray(size, dtype=np.int32),
            )
            out_name = ctx.add_intermediate_tensor(
                f"{compute_output_name}_inverse_{suffix}",
                dtype=compute_dtype,
                shape=row_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[src_name, begin_name, size_name],
                    outputs=[out_name],
                )
            )
            return out_name

        def _slice_matrix_col(src_name: str, col: int, suffix: str) -> str:
            begin = [0 for _ in range(rank)]
            size = [-1 for _ in range(rank)]
            begin[-1] = int(col)
            size[-1] = 1
            begin_name = ctx.add_const_tensor(
                f"{compute_output_name}_inverse_{suffix}_begin",
                np.asarray(begin, dtype=np.int32),
            )
            size_name = ctx.add_const_tensor(
                f"{compute_output_name}_inverse_{suffix}_size",
                np.asarray(size, dtype=np.int32),
            )
            out_name = ctx.add_intermediate_tensor(
                f"{compute_output_name}_inverse_{suffix}",
                dtype=compute_dtype,
                shape=prefix_shape + [rows, 1],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[src_name, begin_name, size_name],
                    outputs=[out_name],
                )
            )
            return out_name

        def _slice_matrix_scalar(src_name: str, row: int, col: int, suffix: str) -> str:
            begin = [0 for _ in range(rank)]
            size = [-1 for _ in range(rank)]
            begin[-2] = int(row)
            begin[-1] = int(col)
            size[-2] = 1
            size[-1] = 1
            begin_name = ctx.add_const_tensor(
                f"{compute_output_name}_inverse_{suffix}_begin",
                np.asarray(begin, dtype=np.int32),
            )
            size_name = ctx.add_const_tensor(
                f"{compute_output_name}_inverse_{suffix}_size",
                np.asarray(size, dtype=np.int32),
            )
            out_name = ctx.add_intermediate_tensor(
                f"{compute_output_name}_inverse_{suffix}",
                dtype=compute_dtype,
                shape=scalar_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[src_name, begin_name, size_name],
                    outputs=[out_name],
                )
            )
            return out_name

        def _binary_tensor(
            op_type: str,
            lhs: str,
            rhs: str,
            shape: list[int],
            suffix: str,
        ) -> str:
            out_name = ctx.add_intermediate_tensor(
                f"{compute_output_name}_inverse_{suffix}",
                dtype=compute_dtype,
                shape=shape,
            )
            options = (
                {"fusedActivationFunction": "NONE"}
                if str(op_type) in {"ADD", "SUB", "MUL", "DIV"}
                else {}
            )
            ctx.add_operator(
                OperatorIR(
                    op_type=op_type,
                    inputs=[lhs, rhs],
                    outputs=[out_name],
                    options=options,
                )
            )
            return out_name

        def _batch_matmul(lhs: str, rhs: str, suffix: str) -> str:
            out_name = ctx.add_intermediate_tensor(
                f"{compute_output_name}_inverse_{suffix}",
                dtype=compute_dtype,
                shape=matrix_shape,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[lhs, rhs],
                    outputs=[out_name],
                    options={
                        "adjX": False,
                        "adjY": False,
                        "asymmetricQuantizeInputs": False,
                    },
                )
            )
            return out_name

        matrix_state = str(compute_input_name)
        inverse_state = str(identity_name)
        for row_idx in range(rows):
            row_mask_arr = np.zeros(row_selector_shape, dtype=np_dtype)
            row_mask_arr[..., int(row_idx), 0] = np_dtype(1.0)
            row_mask_name = ctx.add_const_tensor(
                f"{compute_output_name}_inverse_rowmask_{row_idx}",
                row_mask_arr,
            )

            elim_mask_arr = np.ones(row_selector_shape, dtype=np_dtype)
            elim_mask_arr[..., int(row_idx), 0] = np_dtype(0.0)
            elim_mask_name = ctx.add_const_tensor(
                f"{compute_output_name}_inverse_elimmask_{row_idx}",
                elim_mask_arr,
            )

            row_a = _slice_matrix_row(matrix_state, row_idx, f"iter{row_idx}_row_a")
            row_inv = _slice_matrix_row(inverse_state, row_idx, f"iter{row_idx}_row_inv")
            pivot = _slice_matrix_scalar(matrix_state, row_idx, row_idx, f"iter{row_idx}_pivot")
            pivot_safe = _binary_tensor(
                "ADD",
                pivot,
                eps_name,
                scalar_shape,
                f"iter{row_idx}_pivot_safe",
            )

            row_a_norm = _binary_tensor(
                "DIV",
                row_a,
                pivot_safe,
                row_shape,
                f"iter{row_idx}_row_a_norm",
            )
            row_inv_norm = _binary_tensor(
                "DIV",
                row_inv,
                pivot_safe,
                row_shape,
                f"iter{row_idx}_row_inv_norm",
            )

            delta_row_a = _binary_tensor(
                "SUB",
                row_a_norm,
                row_a,
                row_shape,
                f"iter{row_idx}_delta_row_a",
            )
            delta_row_inv = _binary_tensor(
                "SUB",
                row_inv_norm,
                row_inv,
                row_shape,
                f"iter{row_idx}_delta_row_inv",
            )

            row_update_a = _batch_matmul(
                row_mask_name,
                delta_row_a,
                f"iter{row_idx}_row_update_a",
            )
            row_update_inv = _batch_matmul(
                row_mask_name,
                delta_row_inv,
                f"iter{row_idx}_row_update_inv",
            )

            matrix_norm = _binary_tensor(
                "ADD",
                matrix_state,
                row_update_a,
                matrix_shape,
                f"iter{row_idx}_matrix_norm",
            )
            inverse_norm = _binary_tensor(
                "ADD",
                inverse_state,
                row_update_inv,
                matrix_shape,
                f"iter{row_idx}_inverse_norm",
            )

            factor_col = _slice_matrix_col(
                matrix_norm,
                row_idx,
                f"iter{row_idx}_factor_col",
            )
            factor_col_masked = _binary_tensor(
                "MUL",
                factor_col,
                elim_mask_name,
                prefix_shape + [rows, 1],
                f"iter{row_idx}_factor_col_masked",
            )

            elim_matrix = _batch_matmul(
                factor_col_masked,
                row_a_norm,
                f"iter{row_idx}_elim_matrix",
            )
            elim_inverse = _batch_matmul(
                factor_col_masked,
                row_inv_norm,
                f"iter{row_idx}_elim_inverse",
            )

            matrix_state = _binary_tensor(
                "SUB",
                matrix_norm,
                elim_matrix,
                matrix_shape,
                f"iter{row_idx}_matrix_state",
            )
            inverse_state = _binary_tensor(
                "SUB",
                inverse_norm,
                elim_inverse,
                matrix_shape,
                f"iter{row_idx}_inverse_state",
            )

        final_shape_name = ctx.add_const_tensor(
            f"{compute_output_name}_inverse_final_shape",
            np.asarray(matrix_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[inverse_state, final_shape_name],
                outputs=[compute_output_name],
                options={"newShape": [int(v) for v in matrix_shape]},
            )
        )
        _finalize_float_compute_output(
            ctx=ctx,
            compute_output_name=compute_output_name,
            output_name=output_name,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )
        return

    if cols == 2:
        a00 = _slice_matrix_entry(0, 0)
        a01 = _slice_matrix_entry(0, 1)
        a10 = _slice_matrix_entry(1, 0)
        a11 = _slice_matrix_entry(1, 1)

        det = _mul_sub(a00, a11, a01, a10, "det")
        adj00 = a11
        adj01 = _neg(a01, "adj01_neg")
        adj10 = _neg(a10, "adj10_neg")
        adj11 = a00

        row0 = _concat([adj00, adj01], rank - 1, row_shape, "adj_row0")
        row1 = _concat([adj10, adj11], rank - 1, row_shape, "adj_row1")
        adj = _concat([row0, row1], rank - 2, matrix_shape, "adj")
    else:
        a00 = _slice_matrix_entry(0, 0)
        a01 = _slice_matrix_entry(0, 1)
        a02 = _slice_matrix_entry(0, 2)
        a10 = _slice_matrix_entry(1, 0)
        a11 = _slice_matrix_entry(1, 1)
        a12 = _slice_matrix_entry(1, 2)
        a20 = _slice_matrix_entry(2, 0)
        a21 = _slice_matrix_entry(2, 1)
        a22 = _slice_matrix_entry(2, 2)

        c00 = _mul_sub(a11, a22, a12, a21, "c00")
        c01 = _mul_sub(a12, a20, a10, a22, "c01")
        c02 = _mul_sub(a10, a21, a11, a20, "c02")

        d0 = _binary("MUL", a00, c00, "det0")
        d1 = _binary("MUL", a01, c01, "det1")
        d2 = _binary("MUL", a02, c02, "det2")
        d01 = _binary("ADD", d0, d1, "det01")
        det = _binary("ADD", d01, d2, "det")

        adj00 = c00
        adj01 = _mul_sub(a02, a21, a01, a22, "adj01")
        adj02 = _mul_sub(a01, a12, a02, a11, "adj02")
        adj10 = c01
        adj11 = _mul_sub(a00, a22, a02, a20, "adj11")
        adj12 = _mul_sub(a02, a10, a00, a12, "adj12")
        adj20 = c02
        adj21 = _mul_sub(a01, a20, a00, a21, "adj21")
        adj22 = _mul_sub(a00, a11, a01, a10, "adj22")

        row0 = _concat([adj00, adj01, adj02], rank - 1, row_shape, "adj_row0")
        row1 = _concat([adj10, adj11, adj12], rank - 1, row_shape, "adj_row1")
        row2 = _concat([adj20, adj21, adj22], rank - 1, row_shape, "adj_row2")
        adj = _concat([row0, row1, row2], rank - 2, matrix_shape, "adj")

    ctx.add_operator(
        OperatorIR(
            op_type="DIV",
            inputs=[adj, det],
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


def build_abs_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    runtime_input_name, runtime_output_name = _prepare_passthrough_unary_runtime(
        node=node,
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ABS",
            inputs=[runtime_input_name],
            outputs=[runtime_output_name],
        )
    )
    _finalize_passthrough_unary_runtime(
        node=node,
        ctx=ctx,
        runtime_output_name=runtime_output_name,
        output_name=output_name,
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

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    x_dtype = str(ctx.get_tensor_dtype(x_name)).upper()
    y_dtype = str(ctx.get_tensor_dtype(y_name)).upper()

    def _normalize_where_dtype(dtype: str) -> str:
        dt = str(dtype).upper()
        if dt == "INT64":
            return "INT32"
        if dt == "UINT64":
            return "UINT32"
        return dt

    normalized_output_dtype = _normalize_where_dtype(output_dtype)
    normalized_x_dtype = _normalize_where_dtype(x_dtype)
    normalized_y_dtype = _normalize_where_dtype(y_dtype)
    integer_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32", "INT64", "UINT64"}
    unsigned_integer_dtypes = {"UINT8", "UINT16", "UINT32", "UINT64"}
    where_target_dtype = normalized_output_dtype
    if normalized_x_dtype != where_target_dtype or normalized_y_dtype != where_target_dtype:
        if normalized_x_dtype == normalized_y_dtype:
            where_target_dtype = normalized_x_dtype
        elif x_dtype in integer_dtypes and y_dtype in integer_dtypes:
            if x_dtype in unsigned_integer_dtypes and y_dtype in unsigned_integer_dtypes:
                where_target_dtype = "UINT32"
            else:
                where_target_dtype = "INT32"
        else:
            raise NotImplementedError(
                "Where input dtypes must be compatible in flatbuffer_direct. "
                f"op={node.name} x_dtype={x_dtype} y_dtype={y_dtype} output_dtype={output_dtype}"
            )

    x_name = _cast_tensor_if_needed(
        ctx=ctx,
        src_name=x_name,
        dst_dtype=where_target_dtype,
        base_name=f"{output_name}_where_x_cast",
    )
    y_name = _cast_tensor_if_needed(
        ctx=ctx,
        src_name=y_name,
        dst_dtype=where_target_dtype,
        base_name=f"{output_name}_where_y_cast",
    )
    output_dtype = where_target_dtype
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.dtype = output_dtype
        src_tensor = ctx.model_ir.tensors.get(x_name, None)
        if src_tensor is not None:
            output_tensor.quantization = _clone_quantization(src_tensor.quantization)
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(output_name)] = output_dtype

    condition_dtype = str(ctx.get_tensor_dtype(condition_name)).upper()
    cond_bool_name = condition_name
    if condition_dtype != "BOOL":
        cond_shape = [int(v) for v in ctx.get_tensor_shape(condition_name)]
        cond_bool_name = ctx.add_intermediate_tensor(
            f"{output_name}_where_condition_bool",
            dtype="BOOL",
            shape=cond_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[condition_name],
                outputs=[cond_bool_name],
                options={
                    "inDataType": condition_dtype,
                    "outDataType": "BOOL",
                },
            )
        )

    cond_shape = [int(v) for v in ctx.get_tensor_shape(cond_bool_name)]
    x_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    y_shape = [int(v) for v in ctx.get_tensor_shape(y_name)]
    select_broadcast_supported = (
        (cond_shape == x_shape and x_shape == y_shape)
        or (len(cond_shape) <= 1 and x_shape == y_shape)
    )

    # TFLite SELECT broadcasting is limited. For unsupported broadcast patterns,
    # rewrite Where to arithmetic equivalent when output is float.
    if not select_broadcast_supported and output_dtype in {"FLOAT16", "FLOAT32"}:
        output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
        x_cast_name = _cast_tensor_if_needed(
            ctx=ctx,
            src_name=x_name,
            dst_dtype=output_dtype,
            base_name=f"{output_name}_where_x_cast",
        )
        y_cast_name = _cast_tensor_if_needed(
            ctx=ctx,
            src_name=y_name,
            dst_dtype=output_dtype,
            base_name=f"{output_name}_where_y_cast",
        )
        x_cast_name = _sanitize_where_arithmetic_operand_if_constant_nonfinite(
            ctx=ctx,
            operand_name=x_cast_name,
            output_dtype=output_dtype,
            base_name=f"{output_name}_where_x_sanitized",
        )
        y_cast_name = _sanitize_where_arithmetic_operand_if_constant_nonfinite(
            ctx=ctx,
            operand_name=y_cast_name,
            output_dtype=output_dtype,
            base_name=f"{output_name}_where_y_sanitized",
        )
        cond_cast_name = _cast_tensor_if_needed(
            ctx=ctx,
            src_name=cond_bool_name,
            dst_dtype=output_dtype,
            base_name=f"{output_name}_where_condition_cast",
        )
        one_np_dtype = np.float16 if output_dtype == "FLOAT16" else np.float32
        one_const_name = ctx.add_const_tensor(
            f"{output_name}_where_one",
            np.asarray(1.0, dtype=one_np_dtype),
        )

        inv_cond_name = ctx.add_intermediate_tensor(
            f"{output_name}_where_condition_inv",
            dtype=output_dtype,
            shape=cond_shape,
        )
        x_term_name = ctx.add_intermediate_tensor(
            f"{output_name}_where_x_term",
            dtype=output_dtype,
            shape=output_shape,
        )
        y_term_name = ctx.add_intermediate_tensor(
            f"{output_name}_where_y_term",
            dtype=output_dtype,
            shape=output_shape,
        )

        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[one_const_name, cond_cast_name],
                outputs=[inv_cond_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[cond_cast_name, x_cast_name],
                outputs=[x_term_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[inv_cond_name, y_cast_name],
                outputs=[y_term_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ADD",
                inputs=[x_term_name, y_term_name],
                outputs=[output_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        return

    select_op_type = "SELECT" if len(cond_shape) <= 1 else "SELECT_V2"
    ctx.add_operator(
        OperatorIR(
            op_type=select_op_type,
            inputs=[cond_bool_name, x_name, y_name],
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
    one_name = _add_ones_like_tensor(
        ctx,
        reference_name=compute_input_name,
        base_name=f"{output_name}_atan_ones_like",
        dtype=compute_dtype,
    )
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
    compute_input_tensor = ctx.model_ir.tensors.get(compute_input_name, None)
    compute_signature = (
        [int(v) for v in list(compute_input_tensor.shape_signature)]
        if compute_input_tensor is not None and compute_input_tensor.shape_signature is not None
        else [int(v) for v in list(out_shape)]
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
    for tensor_name in (
        pos_name,
        neg_name,
        exp_neg_name,
        exp_neg_minus_one_name,
        scaled_neg_name,
        elu_alpha_name,
    ):
        tensor = ctx.model_ir.tensors.get(tensor_name, None)
        if tensor is not None:
            tensor.shape_signature = [int(v) for v in list(compute_signature)]
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

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    clip_np_dtype = _numpy_dtype_for_tensor(input_dtype)
    float_clip_input = input_dtype in {"FLOAT16", "FLOAT32", "FLOAT64"}

    clip_min = _get_clip_bound_value(node.attrs.get("min", None), float("-inf"))
    clip_max = _get_clip_bound_value(node.attrs.get("max", None), float("inf"))
    min_arr = None
    max_arr = None
    min_name_dynamic = None
    max_name_dynamic = None
    if len(node.inputs) >= 2:
        min_name = str(node.inputs[1].name)
        if min_name != "":
            ctx.ensure_tensor(min_name)
        min_const = ctx.get_constant_array(min_name)
        if min_const is not None:
            min_arr = np.asarray(min_const, dtype=clip_np_dtype)
            clip_min = _get_clip_bound_value(min_arr, clip_min)
        elif min_name != "":
            min_name_dynamic = min_name
    if len(node.inputs) >= 3:
        max_name = str(node.inputs[2].name)
        if max_name != "":
            ctx.ensure_tensor(max_name)
        max_const = ctx.get_constant_array(max_name)
        if max_const is not None:
            max_arr = np.asarray(max_const, dtype=clip_np_dtype)
            clip_max = _get_clip_bound_value(max_arr, clip_max)
        elif max_name != "":
            max_name_dynamic = max_name

    if min_arr is None and np.isfinite(clip_min):
        min_arr = np.asarray(clip_min, dtype=clip_np_dtype)
    if max_arr is None and np.isfinite(clip_max):
        max_arr = np.asarray(clip_max, dtype=clip_np_dtype)

    if (
        float_clip_input
        and abs(clip_min - 0.0) <= 1e-6
        and abs(clip_max - 6.0) <= 1e-6
        and min_arr is not None
        and max_arr is not None
    ):
        op_type = "RELU6"
    elif (
        float_clip_input
        and abs(clip_min + 1.0) <= 1e-6
        and abs(clip_max - 1.0) <= 1e-6
        and min_arr is not None
        and max_arr is not None
    ):
        op_type = "RELU_N1_TO_1"
    elif (
        float_clip_input
        and abs(clip_min - 0.0) <= 1e-6
        and math.isinf(clip_max)
        and clip_max > 0.0
        and min_arr is not None
        and max_arr is None
    ):
        op_type = "RELU"
    else:
        output_dtype = ctx.get_tensor_dtype(output_name)
        output_shape = ctx.get_tensor_shape(output_name)
        current_name = input_name
        has_max_bound = (max_arr is not None) or (max_name_dynamic is not None)
        if min_arr is not None or min_name_dynamic is not None:
            min_name = (
                ctx.add_const_tensor(
                    f"{node.name}_clip_min",
                    np.asarray(min_arr, dtype=clip_np_dtype),
                )
                if min_arr is not None
                else _cast_tensor_if_needed(
                    ctx=ctx,
                    src_name=str(min_name_dynamic),
                    dst_dtype=input_dtype,
                    base_name=f"{node.name}_clip_min_cast",
                )
            )
            min_output_name = output_name
            if has_max_bound:
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
        if max_arr is not None or max_name_dynamic is not None:
            max_name = (
                ctx.add_const_tensor(
                    f"{node.name}_clip_max",
                    np.asarray(max_arr, dtype=clip_np_dtype),
                )
                if max_arr is not None
                else _cast_tensor_if_needed(
                    ctx=ctx,
                    src_name=str(max_name_dynamic),
                    dst_dtype=input_dtype,
                    base_name=f"{node.name}_clip_max_cast",
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MINIMUM",
                    inputs=[current_name, max_name],
                    outputs=[output_name],
                    options={},
                )
            )
        if min_arr is None and max_arr is None and min_name_dynamic is None and max_name_dynamic is None:
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
    _sync_shape_signature_from_src(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
    if rank <= 0:
        raise NotImplementedError(f"Softmax requires rank >= 1. op={node.name} shape={input_shape}")
    axis = _resolve_softmax_axis(node=node, ctx=ctx, rank=rank)
    beta = float(node.attrs.get("beta", 1.0))

    if axis == rank - 1:
        ctx.add_operator(
            OperatorIR(
                op_type="SOFTMAX",
                inputs=[input_name],
                outputs=[output_name],
                options={"axis": axis, "beta": beta},
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
            options={"axis": rank - 1, "beta": beta},
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
    _sync_shape_signature_from_src(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
    if rank <= 0:
        raise NotImplementedError(f"LogSoftmax requires rank >= 1. op={node.name} shape={input_shape}")
    axis = _resolve_softmax_axis(node=node, ctx=ctx, rank=rank)

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


def build_shrink_op(node: Any, ctx: Any) -> None:
    (
        compute_input_name,
        compute_output_name,
        output_name,
        output_dtype,
        compute_dtype,
        out_shape,
    ) = _prepare_float_compute(node, ctx, tag="shrink")
    lambd = float(node.attrs.get("lambd", 0.5))
    bias = float(node.attrs.get("bias", 0.0))
    pos_lambd_name = _add_scalar_const(
        ctx,
        f"{output_name}_shrink_pos_lambd",
        lambd,
        compute_dtype,
    )
    neg_lambd_name = _add_scalar_const(
        ctx,
        f"{output_name}_shrink_neg_lambd",
        -lambd,
        compute_dtype,
    )
    bias_name = _add_scalar_const(
        ctx,
        f"{output_name}_shrink_bias",
        bias,
        compute_dtype,
    )
    zero_name = _add_scalar_const(
        ctx,
        f"{output_name}_shrink_zero",
        0.0,
        compute_dtype,
    )

    plus_name = ctx.add_intermediate_tensor(
        f"{output_name}_shrink_plus",
        dtype=compute_dtype,
        shape=out_shape,
    )
    minus_name = ctx.add_intermediate_tensor(
        f"{output_name}_shrink_minus",
        dtype=compute_dtype,
        shape=out_shape,
    )
    less_name = ctx.add_intermediate_tensor(
        f"{output_name}_shrink_less",
        dtype="BOOL",
        shape=out_shape,
    )
    greater_name = ctx.add_intermediate_tensor(
        f"{output_name}_shrink_greater",
        dtype="BOOL",
        shape=out_shape,
    )
    greater_select_name = ctx.add_intermediate_tensor(
        f"{output_name}_shrink_greater_select",
        dtype=compute_dtype,
        shape=out_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[compute_input_name, bias_name],
            outputs=[plus_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[compute_input_name, bias_name],
            outputs=[minus_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LESS",
            inputs=[compute_input_name, neg_lambd_name],
            outputs=[less_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GREATER",
            inputs=[compute_input_name, pos_lambd_name],
            outputs=[greater_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[greater_name, minus_name, zero_name],
            outputs=[greater_select_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[less_name, plus_name, greater_select_name],
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


def build_rotary_embedding_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    cos_name = node.inputs[1].name
    sin_name = node.inputs[2].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(cos_name)
    ctx.ensure_tensor(sin_name)
    ctx.ensure_tensor(output_name)

    input_shape, input_signature = _tensor_shape_and_signature(tensor_name=input_name, ctx=ctx)
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = str(input_dtype)

    transpose_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_transposed",
        dtype=compute_dtype,
        shape=[int(input_shape[0]), int(input_shape[2]), int(input_shape[1]), int(input_shape[3])],
    )
    transpose_name = make_transpose(ctx, input_name, transpose_name, [0, 2, 1, 3])
    transposed_shape, transposed_signature = _tensor_shape_and_signature(tensor_name=transpose_name, ctx=ctx)

    head_size = int(transposed_shape[-1])
    rotary_dim_attr = int(node.attrs.get("rotary_embedding_dim", 0))
    rotary_dim = int(head_size if rotary_dim_attr == 0 else rotary_dim_attr)
    half_rotary_dim = int(rotary_dim // 2)
    rotated_shape = [int(v) for v in transposed_shape[:-1]] + [int(rotary_dim)]
    rotated_signature = [int(v) for v in transposed_signature[:-1]] + [int(rotary_dim)]
    half_shape = [int(v) for v in transposed_shape[:-1]] + [int(half_rotary_dim)]
    half_signature = [int(v) for v in transposed_signature[:-1]] + [int(half_rotary_dim)]

    x_rotate_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_rotate",
        dtype=compute_dtype,
        shape=rotated_shape,
    )
    _add_slice_last_axis(
        ctx=ctx,
        input_name=transpose_name,
        output_name=x_rotate_name,
        input_shape=transposed_shape,
        input_signature=transposed_signature,
        begin_last=0,
        size_last=rotary_dim,
    )

    x_rest_name = ""
    if int(rotary_dim) < int(head_size):
        x_rest_name = ctx.add_intermediate_tensor(
            f"{output_name}_rotary_rest",
            dtype=compute_dtype,
            shape=[int(v) for v in transposed_shape[:-1]] + [int(head_size - rotary_dim)],
        )
        _add_slice_last_axis(
            ctx=ctx,
            input_name=transpose_name,
            output_name=x_rest_name,
            input_shape=transposed_shape,
            input_signature=transposed_signature,
            begin_last=rotary_dim,
            size_last=int(head_size - rotary_dim),
        )

    x1_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_x1",
        dtype=compute_dtype,
        shape=half_shape,
    )
    x2_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_x2",
        dtype=compute_dtype,
        shape=half_shape,
    )
    _add_slice_last_axis(
        ctx=ctx,
        input_name=x_rotate_name,
        output_name=x1_name,
        input_shape=rotated_shape,
        input_signature=rotated_signature,
        begin_last=0,
        size_last=half_rotary_dim,
    )
    _add_slice_last_axis(
        ctx=ctx,
        input_name=x_rotate_name,
        output_name=x2_name,
        input_shape=rotated_shape,
        input_signature=rotated_signature,
        begin_last=half_rotary_dim,
        size_last=half_rotary_dim,
    )

    cos_working_name = cos_name
    sin_working_name = sin_name
    cos_dtype = str(ctx.get_tensor_dtype(cos_name)).upper()
    sin_dtype = str(ctx.get_tensor_dtype(sin_name)).upper()
    cache_shape = [1, int(transposed_shape[1]), 1, int(half_rotary_dim)]
    if cos_dtype != compute_dtype:
        cos_cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_rotary_cos_cast",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(cos_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[cos_name],
                outputs=[cos_cast_name],
                options={"inDataType": cos_dtype, "outDataType": compute_dtype},
            )
        )
        cos_working_name = cos_cast_name
    if sin_dtype != compute_dtype:
        sin_cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_rotary_sin_cast",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(sin_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[sin_name],
                outputs=[sin_cast_name],
                options={"inDataType": sin_dtype, "outDataType": compute_dtype},
            )
        )
        sin_working_name = sin_cast_name

    cache_shape_name = ctx.add_const_tensor(
        f"{output_name}_rotary_cache_shape",
        np.asarray(cache_shape, dtype=np.int32),
    )
    cos_broadcast_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_cos_broadcast",
        dtype=compute_dtype,
        shape=cache_shape,
    )
    sin_broadcast_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_sin_broadcast",
        dtype=compute_dtype,
        shape=cache_shape,
    )
    for tensor_name in [cos_broadcast_name, sin_broadcast_name]:
        tensor = ctx.model_ir.tensors.get(tensor_name, None)
        if tensor is not None:
            tensor.shape_signature = [int(v) for v in cache_shape]
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[cos_working_name, cache_shape_name],
            outputs=[cos_broadcast_name],
            options={"newShape": [int(v) for v in cache_shape]},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[sin_working_name, cache_shape_name],
            outputs=[sin_broadcast_name],
            options={"newShape": [int(v) for v in cache_shape]},
        )
    )

    mul_output_names = {
        "cos_x1": ctx.add_intermediate_tensor(f"{output_name}_rotary_cos_x1", dtype=compute_dtype, shape=half_shape),
        "sin_x2": ctx.add_intermediate_tensor(f"{output_name}_rotary_sin_x2", dtype=compute_dtype, shape=half_shape),
        "sin_x1": ctx.add_intermediate_tensor(f"{output_name}_rotary_sin_x1", dtype=compute_dtype, shape=half_shape),
        "cos_x2": ctx.add_intermediate_tensor(f"{output_name}_rotary_cos_x2", dtype=compute_dtype, shape=half_shape),
    }
    for tensor_name in list(mul_output_names.values()):
        tensor = ctx.model_ir.tensors.get(tensor_name, None)
        if tensor is not None:
            tensor.shape_signature = [int(v) for v in half_signature]
    _add_binary_tensor_op(ctx=ctx, op_type="MUL", lhs_name=cos_broadcast_name, rhs_name=x1_name, output_name=mul_output_names["cos_x1"])
    _add_binary_tensor_op(ctx=ctx, op_type="MUL", lhs_name=sin_broadcast_name, rhs_name=x2_name, output_name=mul_output_names["sin_x2"])
    _add_binary_tensor_op(ctx=ctx, op_type="MUL", lhs_name=sin_broadcast_name, rhs_name=x1_name, output_name=mul_output_names["sin_x1"])
    _add_binary_tensor_op(ctx=ctx, op_type="MUL", lhs_name=cos_broadcast_name, rhs_name=x2_name, output_name=mul_output_names["cos_x2"])

    real_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_real",
        dtype=compute_dtype,
        shape=half_shape,
    )
    imag_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_imag",
        dtype=compute_dtype,
        shape=half_shape,
    )
    for tensor_name in [real_name, imag_name]:
        tensor = ctx.model_ir.tensors.get(tensor_name, None)
        if tensor is not None:
            tensor.shape_signature = [int(v) for v in half_signature]
    _add_binary_tensor_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=mul_output_names["cos_x1"],
        rhs_name=mul_output_names["sin_x2"],
        output_name=real_name,
    )
    _add_binary_tensor_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=mul_output_names["sin_x1"],
        rhs_name=mul_output_names["cos_x2"],
        output_name=imag_name,
    )

    rotated_out_name = ctx.add_intermediate_tensor(
        f"{output_name}_rotary_rotated",
        dtype=compute_dtype,
        shape=rotated_shape,
    )
    rotated_out_tensor = ctx.model_ir.tensors.get(rotated_out_name, None)
    if rotated_out_tensor is not None:
        rotated_out_tensor.shape_signature = [int(v) for v in rotated_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[real_name, imag_name],
            outputs=[rotated_out_name],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        )
    )

    transposed_output_name = rotated_out_name
    if str(x_rest_name) != "":
        transposed_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_rotary_concat",
            dtype=compute_dtype,
            shape=transposed_shape,
        )
        transposed_output_tensor = ctx.model_ir.tensors.get(transposed_output_name, None)
        if transposed_output_tensor is not None:
            transposed_output_tensor.shape_signature = [int(v) for v in transposed_signature]
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[rotated_out_name, x_rest_name],
                outputs=[transposed_output_name],
                options={"axis": 3, "fusedActivationFunction": "NONE"},
            )
        )

    final_compute_name = output_name if output_dtype == compute_dtype else f"{output_name}_rotary_transpose_back"
    final_compute_name = make_transpose(ctx, transposed_output_name, final_compute_name, [0, 2, 1, 3])
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.shape = [int(v) for v in input_shape]
        output_tensor.shape_signature = [int(v) for v in input_signature]
    if final_compute_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[final_compute_name],
                outputs=[output_name],
                options={"inDataType": compute_dtype, "outDataType": output_dtype},
            )
        )
