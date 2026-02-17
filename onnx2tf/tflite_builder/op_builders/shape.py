from __future__ import annotations

from typing import Any
import copy

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def _numpy_dtype_from_tflite_dtype(tflite_dtype: str) -> np.dtype:
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
    raise NotImplementedError(
        f"Unsupported TFLite dtype in shape builder: {tflite_dtype}"
    )


def _normalize_axis(axis: int, rank: int, *, op_name: str) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += int(rank)
    if normalized < 0 or normalized >= int(rank):
        raise NotImplementedError(
            f"Slice axis out of range in flatbuffer_direct. op={op_name} axis={axis} rank={rank}"
        )
    return normalized


def _propagate_passthrough_dtype_and_quantization(
    *,
    ctx: Any,
    src_tensor_name: str,
    dst_tensor_name: str,
) -> None:
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src_tensor = ctx.model_ir.tensors[src_tensor_name]
    dst_tensor = ctx.model_ir.tensors[dst_tensor_name]
    dst_tensor.dtype = str(src_tensor.dtype)
    if src_tensor.quantization is not None:
        dst_tensor.quantization = _clone_quantization(src_tensor.quantization)


def _propagate_passthrough_shape_signature(
    *,
    ctx: Any,
    src_tensor_name: str,
    dst_tensor_name: str,
) -> None:
    ctx.ensure_tensor(src_tensor_name)
    ctx.ensure_tensor(dst_tensor_name)
    src_tensor = ctx.model_ir.tensors[src_tensor_name]
    dst_tensor = ctx.model_ir.tensors[dst_tensor_name]
    dst_tensor.shape = [int(v) for v in list(src_tensor.shape)]
    if src_tensor.shape_signature is not None:
        dst_tensor.shape_signature = [int(v) for v in list(src_tensor.shape_signature)]
    else:
        dst_tensor.shape_signature = [int(v) for v in list(src_tensor.shape)]


def _parse_slice_axes_or_steps(
    *,
    node: Any,
    ctx: Any,
    input_index: int,
    attr_name: str,
    default_values: list[int],
    label: str,
) -> list[int]:
    values: list[int] | None = None
    if len(node.inputs) > input_index:
        arr = ctx.get_constant_array(node.inputs[input_index].name)
        if arr is None:
            raise NotImplementedError(
                f"Slice {label} must be constant for flatbuffer_direct. op={node.name}"
            )
        values = [int(v) for v in np.asarray(arr).reshape(-1).tolist()]
    elif attr_name in node.attrs:
        attr_val = node.attrs.get(attr_name)
        if isinstance(attr_val, (list, tuple, np.ndarray)):
            values = [int(v) for v in np.asarray(attr_val).reshape(-1).tolist()]
        elif attr_val is None:
            values = []
        else:
            values = [int(attr_val)]
    if values is None:
        values = [int(v) for v in default_values]
    return [int(v) for v in values]


def build_slice_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    starts_arr = ctx.get_constant_array(node.inputs[1].name)
    ends_arr = ctx.get_constant_array(node.inputs[2].name)
    if starts_arr is None or ends_arr is None:
        raise NotImplementedError(
            f"Slice starts/ends must be constant for flatbuffer_direct. op={node.name}"
        )
    starts = [int(v) for v in np.asarray(starts_arr).reshape(-1).tolist()]
    ends = [int(v) for v in np.asarray(ends_arr).reshape(-1).tolist()]
    if len(starts) != len(ends):
        raise NotImplementedError(
            f"Slice starts and ends length mismatch. op={node.name} "
            f"starts_len={len(starts)} ends_len={len(ends)}"
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor is not None and input_tensor.shape_signature is not None
        else list(input_shape)
    )
    rank = len(input_shape)
    axes = _parse_slice_axes_or_steps(
        node=node,
        ctx=ctx,
        input_index=3,
        attr_name="axes",
        default_values=[int(v) for v in range(len(starts))],
        label="axes",
    )
    steps = _parse_slice_axes_or_steps(
        node=node,
        ctx=ctx,
        input_index=4,
        attr_name="steps",
        default_values=[1 for _ in range(len(starts))],
        label="steps",
    )
    if len(axes) != len(starts) or len(steps) != len(starts):
        raise NotImplementedError(
            f"Slice starts/axes/steps length mismatch. op={node.name} "
            f"starts_len={len(starts)} axes_len={len(axes)} steps_len={len(steps)}"
        )

    known_dim_flags = [
        (
            axis < len(input_signature)
            and int(input_signature[axis]) >= 0
            and axis < len(input_shape)
            and int(input_shape[axis]) > 0
        )
        for axis in range(rank)
    ]
    begin = [0 for _ in range(rank)]
    end_for_strided = [
        int(input_shape[axis]) if known_dim_flags[axis] else int(np.iinfo(np.int32).max)
        for axis in range(rank)
    ]
    strides_for_strided = [1 for _ in range(rank)]
    size = [int(input_shape[axis]) if known_dim_flags[axis] else -1 for axis in range(rank)]
    large_int = int(np.iinfo(np.int64).max // 2)
    use_strided_slice = any(not flag for flag in known_dim_flags)

    for idx, axis_raw in enumerate(axes):
        axis = _normalize_axis(axis_raw, rank, op_name=node.name)
        step = int(steps[idx])
        if step == 0:
            raise NotImplementedError(
                f"Slice step must not be 0 for flatbuffer_direct. op={node.name} step={step}"
            )
        if step < 0:
            raise NotImplementedError(
                f"Slice negative step is not supported for flatbuffer_direct. op={node.name} step={step}"
            )
        if step != 1:
            use_strided_slice = True
        start = int(starts[idx])
        end = int(ends[idx])
        dim_is_known = (
            axis < len(input_signature)
            and int(input_signature[axis]) >= 0
        )
        dim = int(input_shape[axis]) if dim_is_known and axis < len(input_shape) else -1

        if dim > 0:
            if start < 0:
                start += dim
            if end < 0:
                end += dim
            start = max(0, min(start, dim))
            end = max(0, min(end, dim))
            begin[axis] = int(start)
            end_for_strided[axis] = int(end)
            strides_for_strided[axis] = int(step)
            if step == 1:
                size[axis] = int(max(end - start, 0))
            else:
                size[axis] = -1
        else:
            if start < 0:
                raise NotImplementedError(
                    f"Slice with negative start on unknown dimension is not supported. "
                    f"op={node.name} axis={axis} start={start}"
                )
            begin[axis] = int(start)
            if end >= large_int:
                end_for_strided[axis] = int(np.iinfo(np.int32).max)
                size[axis] = -1
            elif end < 0:
                raise NotImplementedError(
                    f"Slice with negative end on unknown dimension is not supported. "
                    f"op={node.name} axis={axis} end={end}"
                )
            else:
                end_for_strided[axis] = int(end)
                if step == 1:
                    size[axis] = int(max(end - start, 0))
                else:
                    size[axis] = -1
            strides_for_strided[axis] = int(step)

    if use_strided_slice:
        begin_name = ctx.add_const_tensor(
            f"{output_name}_stridedslice_begin",
            np.asarray(begin, dtype=np.int32),
        )
        end_name = ctx.add_const_tensor(
            f"{output_name}_stridedslice_end",
            np.asarray(end_for_strided, dtype=np.int32),
        )
        strides_name = ctx.add_const_tensor(
            f"{output_name}_stridedslice_strides",
            np.asarray(strides_for_strided, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="STRIDED_SLICE",
                inputs=[input_name, begin_name, end_name, strides_name],
                outputs=[output_name],
                options={
                    "beginMask": 0,
                    "endMask": 0,
                    "ellipsisMask": 0,
                    "newAxisMask": 0,
                    "shrinkAxisMask": 0,
                    "offset": False,
                },
            )
        )
    else:
        begin_name = ctx.add_const_tensor(
            f"{output_name}_slice_begin",
            np.asarray(begin, dtype=np.int32),
        )
        size_name = ctx.add_const_tensor(
            f"{output_name}_slice_size",
            np.asarray(size, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[input_name, begin_name, size_name],
                outputs=[output_name],
            )
        )


def _parse_split_sizes(
    *,
    node: Any,
    ctx: Any,
    input_axis_dim: int,
    output_count: int,
) -> list[int]:
    split_sizes: list[int] | None = None
    if len(node.inputs) >= 2:
        split_arr = ctx.get_constant_array(node.inputs[1].name)
        if split_arr is None:
            raise NotImplementedError(
                f"Split split tensor must be constant for flatbuffer_direct. op={node.name}"
            )
        split_sizes = [int(v) for v in np.asarray(split_arr).reshape(-1).tolist()]
    elif "split" in node.attrs:
        split_attr = node.attrs.get("split")
        if isinstance(split_attr, (list, tuple, np.ndarray)):
            split_sizes = [int(v) for v in np.asarray(split_attr).reshape(-1).tolist()]
        elif split_attr is not None:
            split_sizes = [int(split_attr)]

    if split_sizes is None or len(split_sizes) == 0:
        if input_axis_dim <= 0 or input_axis_dim % int(output_count) != 0:
            raise NotImplementedError(
                "Split requires explicit split sizes when axis dimension is unknown "
                "or not divisible by number of outputs in flatbuffer_direct. "
                f"op={node.name} axis_dim={input_axis_dim} outputs={output_count}"
            )
        each = int(input_axis_dim // int(output_count))
        split_sizes = [int(each) for _ in range(int(output_count))]

    if len(split_sizes) != int(output_count):
        raise NotImplementedError(
            f"Split split size count must match outputs. op={node.name} "
            f"split_len={len(split_sizes)} outputs={output_count}"
        )
    # If shape metadata is stale, axis_dim may not match explicit split sizes.
    # In that case, prefer explicit split sizes from ONNX and continue.
    if any(int(v) < 0 for v in split_sizes):
        raise NotImplementedError(
            f"Split split sizes must be non-negative. op={node.name} split={split_sizes}"
        )
    return [int(v) for v in split_sizes]


def build_split_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_names = [o.name for o in node.outputs]
    if len(output_names) == 0:
        return

    ctx.ensure_tensor(input_name)
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    rank = len(input_shape)
    axis = _normalize_axis(int(node.attrs.get("axis", 0)), rank, op_name=node.name)
    input_axis_dim = int(input_shape[axis]) if axis < len(input_shape) else -1
    split_sizes = _parse_split_sizes(
        node=node,
        ctx=ctx,
        input_axis_dim=input_axis_dim,
        output_count=len(output_names),
    )

    input_tensor = ctx.model_ir.tensors[input_name]
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_shape)
    )

    offset = 0
    for out_idx, output_name in enumerate(output_names):
        ctx.ensure_tensor(output_name)
        output_tensor = ctx.model_ir.tensors[output_name]
        output_tensor.dtype = input_tensor.dtype
        output_tensor.quantization = _clone_quantization(input_tensor.quantization)

        out_shape = list(input_shape)
        out_shape[axis] = int(split_sizes[out_idx])
        output_tensor.shape = [int(v) for v in out_shape]
        output_signature = list(input_signature)
        if axis < len(output_signature):
            output_signature[axis] = int(split_sizes[out_idx])
        output_tensor.shape_signature = [int(v) for v in output_signature]

        begin = [0 for _ in range(rank)]
        begin[axis] = int(offset)
        # Split should preserve all non-split axes even when static shape
        # metadata is stale. Use -1 to consume full extent on those axes.
        size = [-1 for _ in range(rank)]
        size[axis] = int(split_sizes[out_idx])
        offset += int(split_sizes[out_idx])

        begin_name = ctx.add_const_tensor(
            f"{output_name}_split_begin",
            np.asarray(begin, dtype=np.int32),
        )
        size_name = ctx.add_const_tensor(
            f"{output_name}_split_size",
            np.asarray(size, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[input_name, begin_name, size_name],
                outputs=[output_name],
            )
        )


def _resolve_reshape_shape_with_static_dims(
    *,
    new_shape: list[int],
    input_tensor: Any,
    output_tensor: Any,
    allowzero: bool,
) -> list[int]:
    input_signature = (
        list(input_tensor.shape_signature)
        if input_tensor.shape_signature is not None
        else list(input_tensor.shape)
    )
    output_signature = (
        list(output_tensor.shape_signature)
        if output_tensor.shape_signature is not None
        else list(output_tensor.shape)
    )
    if len(output_signature) == len(new_shape) and all(int(dim) > 0 for dim in output_signature):
        if len(input_signature) > 0 and all(int(dim) > 0 for dim in input_signature):
            output_product = int(np.prod(np.asarray(output_signature, dtype=np.int64)))
            input_product = int(np.prod(np.asarray(input_signature, dtype=np.int64)))
            if output_product == input_product:
                return [int(v) for v in output_tensor.shape]
        else:
            return [int(v) for v in output_tensor.shape]

    minus_one_indices = [idx for idx, dim in enumerate(new_shape) if int(dim) == -1]
    resolved_shape = [int(v) for v in new_shape]

    if not allowzero:
        for idx, dim in enumerate(resolved_shape):
            if int(dim) != 0:
                continue
            if idx >= len(input_signature):
                return [int(v) for v in new_shape]
            in_dim = int(input_signature[idx])
            if in_dim <= 0:
                return [int(v) for v in new_shape]
            resolved_shape[idx] = in_dim

    if len(minus_one_indices) != 1:
        return [int(v) for v in resolved_shape]
    if len(input_signature) == 0 or any(int(dim) <= 0 for dim in input_signature):
        return [int(v) for v in resolved_shape]

    known_product = 1
    for idx, raw_dim in enumerate(resolved_shape):
        dim = int(raw_dim)
        if dim == -1:
            continue
        if dim <= 0:
            return [int(v) for v in resolved_shape]
        known_product *= dim
    if known_product <= 0:
        return [int(v) for v in resolved_shape]

    input_product = int(np.prod(np.asarray(input_signature, dtype=np.int64)))
    if input_product <= 0 or input_product % known_product != 0:
        return [int(v) for v in resolved_shape]
    inferred = int(input_product // known_product)
    if inferred <= 0:
        return [int(v) for v in resolved_shape]
    resolved_shape[minus_one_indices[0]] = inferred
    return resolved_shape


def build_reshape_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    shape_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    shape_values = ctx.get_constant_array(shape_name)
    if shape_values is None:
        raise NotImplementedError(
            f"Reshape shape tensor must be constant for flatbuffer_direct. op={node.name}"
        )
    raw_new_shape = [int(v) for v in np.asarray(shape_values).reshape(-1).tolist()]
    new_shape = list(raw_new_shape)
    allowzero = bool(node.attrs.get("allowzero", 0))
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    new_shape = _resolve_reshape_shape_with_static_dims(
        new_shape=new_shape,
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        allowzero=allowzero,
    )
    if len(new_shape) > 0 and all(int(dim) >= 0 for dim in new_shape):
        output_tensor.shape = [int(dim) for dim in new_shape]
        output_tensor.shape_signature = [int(dim) for dim in new_shape]
    shape_const = ctx.add_const_tensor(
        f"{output_name}_reshape_shape",
        np.asarray(new_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options={
                "newShape": new_shape,
                "onnxRawNewShape": raw_new_shape,
                "allowZero": bool(allowzero),
            },
        )
    )


def build_transpose_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    perm = None
    if len(node.inputs) >= 2:
        perm_name = node.inputs[1].name
        perm = ctx.get_constant_array(perm_name)
    if perm is None and "perm" in node.attrs:
        attr_perm = node.attrs.get("perm")
        if isinstance(attr_perm, (list, tuple)):
            perm = np.asarray([int(v) for v in attr_perm], dtype=np.int32)
        elif attr_perm is not None:
            perm = np.asarray([int(attr_perm)], dtype=np.int32)
    if perm is None:
        input_rank = len(ctx.get_tensor_shape(input_name))
        perm = np.asarray(list(reversed(range(input_rank))), dtype=np.int32)

    if perm is None:
        raise NotImplementedError(
            f"Transpose permutation must be resolvable for flatbuffer_direct. op={node.name}"
        )
    perm_const = ctx.add_const_tensor(
        f"{output_name}_transpose_perm",
        np.asarray(perm, dtype=np.int32).reshape(-1),
    )

    ctx.add_operator(
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[input_name, perm_const],
            outputs=[output_name],
        )
    )


def build_concat_op(node: Any, ctx: Any) -> None:
    input_names = [i.name for i in node.inputs]
    output_name = node.outputs[0].name
    for name in input_names:
        ctx.ensure_tensor(name)
    ctx.ensure_tensor(output_name)

    output_shape = ctx.get_tensor_shape(output_name)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += len(output_shape)

    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=input_names,
            outputs=[output_name],
            options={
                "axis": int(axis),
                "fusedActivationFunction": "NONE",
            },
        )
    )


def build_identity_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    output_shape = ctx.get_tensor_shape(output_name)
    shape_const = ctx.add_const_tensor(
        f"{output_name}_identity_shape",
        np.asarray(output_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape]},
        )
    )


def build_eyelike_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    output_tensor = ctx.model_ir.tensors[output_name]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in output_shape]
    )
    if len(output_shape) != 2:
        raise NotImplementedError(
            f"EyeLike requires rank-2 output shape in flatbuffer_direct. op={node.name} output_shape={output_shape}"
        )
    if any(int(v) <= 0 for v in output_shape) or any(int(v) < 0 for v in output_signature):
        raise NotImplementedError(
            "EyeLike requires fully static positive shape in flatbuffer_direct. "
            f"op={node.name} output_shape={output_shape} output_signature={output_signature}"
        )

    rows = int(output_shape[0])
    cols = int(output_shape[1])
    k = int(node.attrs.get("k", 0))
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    output_np_dtype = _numpy_dtype_from_tflite_dtype(output_dtype)
    eye_const_name = ctx.add_const_tensor(
        f"{output_name}_eyelike_const",
        np.eye(rows, cols, k=k, dtype=output_np_dtype),
    )
    shape_const_name = ctx.add_const_tensor(
        f"{output_name}_eyelike_shape",
        np.asarray(output_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[eye_const_name, shape_const_name],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape]},
        )
    )
    output_tensor.shape = [int(v) for v in output_shape]
    output_tensor.shape_signature = [int(v) for v in output_signature]
    output_tensor.quantization = None


def build_trilu_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )
    _propagate_passthrough_shape_signature(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) < 2:
        raise NotImplementedError(
            f"Trilu requires rank >= 2 in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    m = int(input_shape[-2])
    n = int(input_shape[-1])
    if m <= 0 or n <= 0:
        raise NotImplementedError(
            "Trilu requires static positive matrix dimensions in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )
    k = 0
    if len(node.inputs) >= 2 and node.inputs[1].name != "":
        k_arr = ctx.get_constant_array(node.inputs[1].name)
        if k_arr is None:
            raise NotImplementedError(
                f"Trilu k input must be constant for flatbuffer_direct. op={node.name}"
            )
        k_flat = np.asarray(k_arr).reshape(-1)
        if int(k_flat.size) > 0:
            k = int(k_flat[0])

    upper = int(node.attrs.get("upper", 1)) != 0
    if upper:
        mask_2d = np.triu(np.ones((m, n), dtype=np.int32), k=int(k))
    else:
        mask_2d = np.tril(np.ones((m, n), dtype=np.int32), k=int(k))

    broadcast_shape = [1] * (len(input_shape) - 2) + [int(m), int(n)]
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    mask_name = ""
    op_type = "MUL"
    op_options: dict[str, Any] = {"fusedActivationFunction": "NONE"}
    if input_dtype == "BOOL":
        mask_name = ctx.add_const_tensor(
            f"{output_name}_trilu_mask",
            np.reshape(mask_2d.astype(np.bool_), broadcast_shape),
        )
        op_type = "LOGICAL_AND"
        op_options = {}
    else:
        mask_name = ctx.add_const_tensor(
            f"{output_name}_trilu_mask",
            np.reshape(
                mask_2d.astype(_numpy_dtype_from_tflite_dtype(input_dtype)),
                broadcast_shape,
            ),
        )
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[input_name, mask_name],
            outputs=[output_name],
            options=op_options,
        )
    )


def build_range_op(node: Any, ctx: Any) -> None:
    start_name = node.inputs[0].name
    limit_name = node.inputs[1].name
    delta_name = node.inputs[2].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(start_name)
    ctx.ensure_tensor(limit_name)
    ctx.ensure_tensor(delta_name)
    ctx.ensure_tensor(output_name)

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if output_dtype not in {
        "INT8", "INT16", "INT32", "INT64",
        "UINT8", "UINT16", "UINT32", "UINT64",
        "FLOAT16", "FLOAT32",
    }:
        raise NotImplementedError(
            f"Range output dtype is not supported in flatbuffer_direct. op={node.name} dtype={output_dtype}"
        )

    converted_inputs: list[str] = []
    for src_name, label in [
        (start_name, "start"),
        (limit_name, "limit"),
        (delta_name, "delta"),
    ]:
        src_dtype = str(ctx.get_tensor_dtype(src_name)).upper()
        if src_dtype == output_dtype:
            converted_inputs.append(src_name)
            continue
        cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_range_{label}_{output_dtype.lower()}",
            dtype=output_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(src_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[src_name],
                outputs=[cast_name],
                options={
                    "inDataType": src_dtype,
                    "outDataType": output_dtype,
                },
            )
        )
        converted_inputs.append(cast_name)

    scalar_inputs: list[str] = []
    for idx, src_name in enumerate(converted_inputs):
        src_shape = [int(v) for v in ctx.get_tensor_shape(src_name)]
        if len(src_shape) != 1 or int(src_shape[0]) != 1:
            raise NotImplementedError(
                "Range expects scalar-like inputs represented as shape [1] in flatbuffer_direct. "
                f"op={node.name} input_index={idx} input_shape={src_shape}"
            )
        scalar_name = ctx.add_intermediate_tensor(
            f"{output_name}_range_scalar_{idx}",
            dtype=output_dtype,
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[src_name],
                outputs=[scalar_name],
                options={"squeezeDims": [0]},
            )
        )
        scalar_inputs.append(scalar_name)

    ctx.add_operator(
        OperatorIR(
            op_type="RANGE",
            inputs=scalar_inputs,
            outputs=[output_name],
        )
    )


def _normalize_shape_slice_index(index: int, rank: int) -> int:
    normalized = int(index)
    if normalized < 0:
        normalized += int(rank)
    if normalized < 0:
        normalized = 0
    if normalized > int(rank):
        normalized = int(rank)
    return int(normalized)


def build_shape_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_rank = len(ctx.get_tensor_shape(input_name))
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    shape_output_dtype = output_dtype if output_dtype in {"INT32", "INT64"} else "INT32"

    start = int(node.attrs.get("start", 0))
    end = int(node.attrs.get("end", input_rank))
    start = _normalize_shape_slice_index(start, input_rank)
    end = _normalize_shape_slice_index(end, input_rank)
    if end < start:
        end = start

    output_tensor = ctx.model_ir.tensors[output_name]
    static_len = max(int(end - start), 0)
    output_tensor.shape = [int(static_len)]
    output_tensor.shape_signature = [int(static_len)]
    output_tensor.dtype = shape_output_dtype

    if start == 0 and end == int(input_rank):
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[input_name],
                outputs=[output_name],
                options={"outType": shape_output_dtype},
            )
        )
        return

    full_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_shape_full",
        dtype=shape_output_dtype,
        shape=[int(input_rank)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[full_shape_name],
            options={"outType": shape_output_dtype},
        )
    )
    begin_name = ctx.add_const_tensor(
        f"{output_name}_shape_slice_begin",
        np.asarray([int(start)], dtype=np.int32),
    )
    size_name = ctx.add_const_tensor(
        f"{output_name}_shape_slice_size",
        np.asarray([int(static_len)], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[full_shape_name, begin_name, size_name],
            outputs=[output_name],
        )
    )


def build_constant_of_shape_op(node: Any, ctx: Any) -> None:
    shape_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(shape_name)
    ctx.ensure_tensor(output_name)

    shape_dtype = str(ctx.get_tensor_dtype(shape_name)).upper()
    fill_dims_name = shape_name
    if shape_dtype not in {"INT32", "INT64"}:
        shape_cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_constofshape_dims_i32",
            dtype="INT32",
            shape=[int(v) for v in ctx.get_tensor_shape(shape_name)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[shape_name],
                outputs=[shape_cast_name],
                options={
                    "inDataType": shape_dtype,
                    "outDataType": "INT32",
                },
            )
        )
        fill_dims_name = shape_cast_name

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    output_np_dtype = _numpy_dtype_from_tflite_dtype(output_dtype)
    value_attr = node.attrs.get("value", None)
    value_array: np.ndarray
    if value_attr is None:
        value_array = np.asarray(0, dtype=output_np_dtype)
    elif hasattr(value_attr, "values"):
        value_array = np.asarray(getattr(value_attr, "values"), dtype=output_np_dtype)
    else:
        value_array = np.asarray(value_attr, dtype=output_np_dtype)
    if int(value_array.size) == 0:
        value_array = np.asarray(0, dtype=output_np_dtype)
    scalar_value = np.asarray(value_array).reshape(-1)[0]

    fill_value_name = ctx.add_const_tensor(
        f"{output_name}_constofshape_value",
        np.asarray(scalar_value, dtype=output_np_dtype),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="FILL",
            inputs=[fill_dims_name, fill_value_name],
            outputs=[output_name],
        )
    )


def build_cast_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_shape_signature(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[input_name],
            outputs=[output_name],
            options={
                "inDataType": input_dtype,
                "outDataType": output_dtype,
            },
        )
    )


def build_expand_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    shape_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    _ = shape_name
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    if len(output_shape) < len(input_shape):
        raise NotImplementedError(
            f"Expand output rank must be >= input rank in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )
    if any(int(v) <= 0 for v in output_shape):
        raise NotImplementedError(
            f"Expand requires static positive output shape for MUL-broadcast lowering in flatbuffer_direct. "
            f"op={node.name} output_shape={output_shape}"
        )

    rank_pad = int(len(output_shape) - len(input_shape))
    aligned_input_shape = [1] * rank_pad + [int(v) for v in input_shape]
    if any(int(v) <= 0 for v in aligned_input_shape):
        raise NotImplementedError(
            f"Expand requires static positive input shape for MUL-broadcast lowering in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )

    for in_dim, out_dim in zip(aligned_input_shape, output_shape):
        if int(in_dim) == int(out_dim):
            continue
        if int(in_dim) == 1 and int(out_dim) > 0:
            continue
        raise NotImplementedError(
            f"Expand shape is not broadcast-compatible for MUL-broadcast lowering in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape} output_shape={output_shape}"
        )

    mul_input_name = input_name
    if aligned_input_shape != input_shape:
        reshaped_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_expand_input_reshape",
            dtype=ctx.get_tensor_dtype(input_name),
            shape=[int(v) for v in aligned_input_shape],
        )
        reshape_shape = ctx.add_const_tensor(
            f"{output_name}_expand_input_shape",
            np.asarray(aligned_input_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, reshape_shape],
                outputs=[reshaped_input_name],
                options={"newShape": [int(v) for v in aligned_input_shape]},
            )
        )
        mul_input_name = reshaped_input_name

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    mul_input_dtype = str(ctx.get_tensor_dtype(mul_input_name)).upper()
    mul_lhs_name = mul_input_name
    mul_lhs_dtype = mul_input_dtype

    # tf backend Expand path also realizes broadcast via multiply-by-ones.
    # For BOOL, emulate tf behavior by int32 multiply then cast back.
    if output_dtype == "BOOL":
        if mul_lhs_dtype == "BOOL":
            mul_lhs_i32_name = ctx.add_intermediate_tensor(
                f"{output_name}_expand_input_i32",
                dtype="INT32",
                shape=[int(v) for v in aligned_input_shape],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[mul_lhs_name],
                    outputs=[mul_lhs_i32_name],
                    options={
                        "inDataType": "BOOL",
                        "outDataType": "INT32",
                    },
                )
            )
            mul_lhs_name = mul_lhs_i32_name
            mul_lhs_dtype = "INT32"
        else:
            raise NotImplementedError(
                f"Expand BOOL output expects BOOL input in flatbuffer_direct. "
                f"op={node.name} input_dtype={mul_lhs_dtype} output_dtype={output_dtype}"
            )

    ones_dtype = _numpy_dtype_from_tflite_dtype(mul_lhs_dtype)
    ones_name = ctx.add_const_tensor(
        f"{output_name}_expand_ones",
        np.ones(output_shape, dtype=ones_dtype),
    )

    mul_output_name = output_name
    if output_dtype == "BOOL":
        mul_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_expand_mul_out",
            dtype=mul_lhs_dtype,
            shape=[int(v) for v in output_shape],
        )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[mul_lhs_name, ones_name],
            outputs=[mul_output_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    if output_dtype == "BOOL":
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[mul_output_name],
                outputs=[output_name],
                options={
                    "inDataType": mul_lhs_dtype,
                    "outDataType": "BOOL",
                },
            )
        )


def build_pad_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    mode = str(node.attrs.get("mode", "constant")).lower()
    if mode != "constant":
        raise NotImplementedError(
            f"Pad mode is not supported in flatbuffer_direct. op={node.name} mode={mode}"
        )

    pads_arr = None
    if len(node.inputs) >= 2:
        pads_arr = ctx.get_constant_array(node.inputs[1].name)
    if pads_arr is None and "pads" in node.attrs:
        pads_arr = node.attrs.get("pads")
    if pads_arr is None:
        raise NotImplementedError(
            f"Pad pads must be constant for flatbuffer_direct. op={node.name}"
        )

    input_rank = len(ctx.get_tensor_shape(input_name))
    pads_flat = [int(v) for v in np.asarray(pads_arr).reshape(-1).tolist()]
    if len(pads_flat) != int(input_rank * 2):
        raise NotImplementedError(
            "Pad pads length must be 2 * input_rank for flatbuffer_direct. "
            f"op={node.name} rank={input_rank} pads_len={len(pads_flat)}"
        )
    pads_begin = pads_flat[:input_rank]
    pads_end = pads_flat[input_rank:]
    paddings = np.asarray(
        [[int(b), int(e)] for b, e in zip(pads_begin, pads_end)],
        dtype=np.int32,
    )
    pads_name = ctx.add_const_tensor(
        f"{output_name}_pads",
        paddings,
    )

    # Keep initial support minimal/safe: constant zero-padding only.
    if len(node.inputs) >= 3:
        constant_value_arr = ctx.get_constant_array(node.inputs[2].name)
        if constant_value_arr is None:
            raise NotImplementedError(
                f"Pad constant value input must be constant for flatbuffer_direct. op={node.name}"
            )
        constant_value = float(np.asarray(constant_value_arr).reshape(-1)[0])
        if abs(constant_value) > 1e-12:
            raise NotImplementedError(
                "Pad with non-zero constant value is not supported in flatbuffer_direct. "
                f"op={node.name} value={constant_value}"
            )

    ctx.add_operator(
        OperatorIR(
            op_type="PAD",
            inputs=[input_name, pads_name],
            outputs=[output_name],
        )
    )


def _resolve_axes_from_attr_or_input(node: Any, ctx: Any) -> list[int]:
    axes = None
    if len(node.inputs) >= 2:
        axes_arr = ctx.get_constant_array(node.inputs[1].name)
        if axes_arr is None:
            raise NotImplementedError(
                f"{node.op} axes must be constant for flatbuffer_direct. op={node.name}"
            )
        axes = [int(v) for v in np.asarray(axes_arr).reshape(-1).tolist()]
    elif "axes" in node.attrs:
        attr_axes = node.attrs["axes"]
        if isinstance(attr_axes, (list, tuple)):
            axes = [int(v) for v in attr_axes]
        else:
            axes = [int(attr_axes)]
    return [] if axes is None else axes


def build_squeeze_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    input_shape = ctx.get_tensor_shape(input_name)
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )
    rank = len(input_shape)
    axes = _resolve_axes_from_attr_or_input(node, ctx)
    explicit_axes = len(axes) > 0
    if len(axes) == 0:
        axes = []
        for idx in range(rank):
            if idx < len(input_signature):
                if int(input_signature[idx]) == 1:
                    axes.append(int(idx))
            elif int(input_shape[idx]) == 1:
                axes.append(int(idx))

    normalized_axes: list[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += rank
        if a < 0 or a >= rank:
            raise NotImplementedError(
                f"Squeeze axis is out of range. op={node.name} axis={axis} rank={rank}"
            )
        if a not in normalized_axes:
            normalized_axes.append(a)

    axes_to_remove = set(int(v) for v in normalized_axes)
    if not explicit_axes and len(normalized_axes) == 0:
        axes_to_remove = set()
    inferred_output_shape = [
        int(input_shape[idx])
        for idx in range(rank)
        if idx not in axes_to_remove
    ]
    inferred_output_signature = [
        int(input_signature[idx])
        for idx in range(rank)
        if idx not in axes_to_remove
    ]
    output_tensor.shape = [int(v) for v in list(inferred_output_shape)]
    output_tensor.shape_signature = [int(v) for v in list(inferred_output_signature)]

    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[input_name],
            outputs=[output_name],
            options={"squeezeDims": [int(v) for v in normalized_axes]},
        )
    )


def build_unsqueeze_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    axes = _resolve_axes_from_attr_or_input(node, ctx)
    input_rank = len(ctx.get_tensor_shape(input_name))
    normalized_axes: list[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += input_rank + 1
        if a < 0 or a > input_rank:
            raise NotImplementedError(
                f"Unsqueeze axis out of range in flatbuffer_direct. op={node.name} axis={axis} rank={input_rank}"
            )
        if a not in normalized_axes:
            normalized_axes.append(a)
    normalized_axes = sorted([int(v) for v in normalized_axes])
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor.shape_signature is not None
        else [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    )
    if (
        input_rank == 1
        and len(normalized_axes) == 1
        and any(int(v) < 0 for v in input_signature)
    ):
        axis = int(normalized_axes[0])
        if axis not in (0, 1):
            raise NotImplementedError(
                f"Unsqueeze axis out of range for dynamic rank-1 input in flatbuffer_direct. "
                f"op={node.name} axis={axis}"
            )
        reshape_shape = [1, -1] if axis == 0 else [-1, 1]
        inferred_signature = (
            [1, int(input_signature[0])] if axis == 0 else [int(input_signature[0]), 1]
        )
        output_tensor.shape_signature = [int(v) for v in inferred_signature]
        output_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in inferred_signature]
        reshape_shape_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_shape",
            np.asarray(reshape_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, reshape_shape_name],
                outputs=[output_name],
                options={"newShape": [int(v) for v in reshape_shape]},
            )
        )
        return

    output_shape = ctx.get_tensor_shape(output_name)
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_shape)]
    )
    has_dynamic_dim = any(int(v) < 0 for v in output_signature)

    if all(int(v) >= 0 for v in input_signature):
        inferred_shape: list[int] = []
        src_dim_idx = 0
        output_rank = int(input_rank + len(normalized_axes))
        axes_set = set(int(v) for v in normalized_axes)
        for out_axis in range(output_rank):
            if out_axis in axes_set:
                inferred_shape.append(1)
            else:
                inferred_shape.append(int(input_signature[src_dim_idx]))
                src_dim_idx += 1
        output_tensor.shape = [int(v) for v in inferred_shape]
        output_tensor.shape_signature = [int(v) for v in inferred_shape]
        shape_const = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_shape",
            np.asarray(inferred_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, shape_const],
                outputs=[output_name],
                options={"newShape": [int(v) for v in inferred_shape]},
            )
        )
        return

    if not has_dynamic_dim:
        shape_const = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_shape",
            np.asarray(output_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, shape_const],
                outputs=[output_name],
                options={"newShape": [int(v) for v in output_shape]},
            )
        )
        return

    input_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_unsqueeze_input_shape",
        dtype="INT32",
        shape=[input_rank],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[input_shape_name],
            options={"outType": "INT32"},
        )
    )

    current_shape_name = input_shape_name
    current_rank = int(input_rank)
    for axis in normalized_axes:
        prefix_name = ctx.add_intermediate_tensor(
            f"{output_name}_unsqueeze_prefix_{axis}",
            dtype="INT32",
            shape=[max(int(axis), 0)],
        )
        prefix_begin_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_prefix_begin_{axis}",
            np.asarray([0], dtype=np.int32),
        )
        prefix_size_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_prefix_size_{axis}",
            np.asarray([int(axis)], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[current_shape_name, prefix_begin_name, prefix_size_name],
                outputs=[prefix_name],
            )
        )

        suffix_len = int(max(current_rank - int(axis), 0))
        suffix_name = ctx.add_intermediate_tensor(
            f"{output_name}_unsqueeze_suffix_{axis}",
            dtype="INT32",
            shape=[suffix_len],
        )
        suffix_begin_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_suffix_begin_{axis}",
            np.asarray([int(axis)], dtype=np.int32),
        )
        suffix_size_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_suffix_size_{axis}",
            np.asarray([suffix_len], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[current_shape_name, suffix_begin_name, suffix_size_name],
                outputs=[suffix_name],
            )
        )

        one_dim_name = ctx.add_const_tensor(
            f"{output_name}_unsqueeze_one_{axis}",
            np.asarray([1], dtype=np.int32),
        )
        merged_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_unsqueeze_shape_merged_{axis}",
            dtype="INT32",
            shape=[current_rank + 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[prefix_name, one_dim_name, suffix_name],
                outputs=[merged_shape_name],
                options={
                    "axis": 0,
                    "fusedActivationFunction": "NONE",
                },
            )
        )
        current_shape_name = merged_shape_name
        current_rank += 1

    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, current_shape_name],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape]},
        )
    )


def build_space_to_depth_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    block_size = int(node.attrs.get("blocksize", 0))
    if block_size <= 1:
        raise NotImplementedError(
            f"SpaceToDepth blocksize must be > 1. op={node.name} blocksize={block_size}"
        )
    ctx.add_operator(
        OperatorIR(
            op_type="SPACE_TO_DEPTH",
            inputs=[input_name],
            outputs=[output_name],
            options={"blockSize": int(block_size)},
        )
    )


def build_flatten_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_passthrough_dtype_and_quantization(
        ctx=ctx,
        src_tensor_name=input_name,
        dst_tensor_name=output_name,
    )

    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    shape_const = ctx.add_const_tensor(
        f"{output_name}_flatten_shape",
        np.asarray(output_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_const],
            outputs=[output_name],
            options={"newShape": output_shape},
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


def _resolve_resize_target_hw(node: Any, ctx: Any, input_shape: list[int]) -> tuple[int, int]:
    def _resolve_from_sizes(arr: np.ndarray) -> tuple[int, int]:
        values = np.asarray(arr).reshape(-1).astype(np.int64)
        if values.size >= 4:
            return int(values[-2]), int(values[-1])
        if values.size == 2:
            return int(values[0]), int(values[1])
        raise NotImplementedError(
            f"Resize sizes must have at least 2 values. op={node.name} sizes_shape={list(values.shape)}"
        )

    def _resolve_from_scales(arr: np.ndarray) -> tuple[int, int]:
        values = np.asarray(arr).reshape(-1).astype(np.float32)
        in_h = int(input_shape[2])
        in_w = int(input_shape[3])
        if values.size >= 4:
            out_h = int(round(float(in_h) * float(values[-2])))
            out_w = int(round(float(in_w) * float(values[-1])))
            return max(out_h, 1), max(out_w, 1)
        if values.size == 2:
            out_h = int(round(float(in_h) * float(values[0])))
            out_w = int(round(float(in_w) * float(values[1])))
            return max(out_h, 1), max(out_w, 1)
        raise NotImplementedError(
            f"Resize scales must have at least 2 values. op={node.name} scales_shape={list(values.shape)}"
        )

    if len(node.inputs) >= 4:
        sizes_name = node.inputs[3].name
        if sizes_name != "":
            sizes = ctx.get_constant_array(sizes_name)
            if sizes is not None and int(np.asarray(sizes).size) >= 2:
                return _resolve_from_sizes(np.asarray(sizes))
    if len(node.inputs) >= 3:
        scales_name = node.inputs[2].name
        if scales_name != "":
            scales = ctx.get_constant_array(scales_name)
            if scales is not None and int(np.asarray(scales).size) >= 2:
                arr = np.asarray(scales)
                if np.issubdtype(arr.dtype, np.integer):
                    return _resolve_from_sizes(arr)
                return _resolve_from_scales(arr)
    if len(node.inputs) == 2:
        param_name = node.inputs[1].name
        if param_name != "":
            param = ctx.get_constant_array(param_name)
            if param is not None and int(np.asarray(param).size) >= 2:
                arr = np.asarray(param)
                if np.issubdtype(arr.dtype, np.integer):
                    return _resolve_from_sizes(arr)
                return _resolve_from_scales(arr)
    raise NotImplementedError(
        f"Resize target size must be resolvable from constant sizes/scales. op={node.name}"
    )


def _build_resize_dynamic_size_input(node: Any, ctx: Any) -> str | None:
    if len(node.inputs) < 4:
        return None
    sizes_name = node.inputs[3].name
    if sizes_name == "":
        return None
    sizes_const = ctx.get_constant_array(sizes_name)
    if sizes_const is not None and int(np.asarray(sizes_const).size) >= 2:
        return None

    ctx.ensure_tensor(sizes_name)
    sizes_shape = [int(v) for v in ctx.get_tensor_shape(sizes_name)]
    if sizes_shape != [1] and len(sizes_shape) != 1:
        raise NotImplementedError(
            f"Resize dynamic sizes input must be rank-1. op={node.name} sizes_shape={sizes_shape}"
        )
    sizes_tensor = ctx.model_ir.tensors[sizes_name]
    sizes_signature = (
        [int(v) for v in list(sizes_tensor.shape_signature)]
        if sizes_tensor.shape_signature is not None and len(list(sizes_tensor.shape_signature)) == 1
        else [int(v) for v in list(sizes_shape)]
    )
    sizes_len = int(sizes_shape[0]) if len(sizes_shape) == 1 else 1
    if sizes_len <= 0 and len(sizes_signature) == 1 and int(sizes_signature[0]) > 0:
        sizes_len = int(sizes_signature[0])
    if sizes_len == 1:
        # Placeholder rank-1 length from symbolic shape inference.
        # ONNX Resize `sizes` for rank-4 tensors is typically length-4 (N,C,H,W).
        sizes_len = 4

    sizes_i32 = sizes_name
    sizes_dtype = str(ctx.get_tensor_dtype(sizes_name)).upper()
    if sizes_dtype != "INT32":
        cast_sizes_name = ctx.add_intermediate_tensor(
            f"{node.name}_resize_sizes_i32",
            dtype="INT32",
            shape=[int(v) for v in list(sizes_shape)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[sizes_name],
                outputs=[cast_sizes_name],
                options={
                    "inDataType": sizes_dtype,
                    "outDataType": "INT32",
                },
            )
        )
        sizes_i32 = cast_sizes_name

    if sizes_len == 2:
        return sizes_i32
    if sizes_len == 4:
        size_hw = ctx.add_intermediate_tensor(
            f"{node.name}_resize_size_hw",
            dtype="INT32",
            shape=[2],
        )
        begin = ctx.add_const_tensor(
            f"{node.name}_resize_sizes_begin",
            np.asarray([2], dtype=np.int32),
        )
        size = ctx.add_const_tensor(
            f"{node.name}_resize_sizes_size",
            np.asarray([2], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[sizes_i32, begin, size],
                outputs=[size_hw],
            )
        )
        return size_hw

    raise NotImplementedError(
        f"Resize dynamic sizes input length must be 2 or 4. "
        f"op={node.name} sizes_shape={sizes_shape} sizes_signature={sizes_signature}"
    )


def _infer_resize_output_signature_nchw(
    *,
    input_signature_nchw: list[int],
    output_shape_nchw: list[int],
    onnx_sizes_hw: list[int] | None,
    onnx_scales_hw: list[float] | None,
    existing_output_signature_nchw: list[int] | None,
) -> list[int]:
    signature = [int(v) for v in list(output_shape_nchw)]
    if existing_output_signature_nchw is not None and len(existing_output_signature_nchw) == 4:
        for axis in range(4):
            if int(existing_output_signature_nchw[axis]) < 0:
                signature[axis] = -1
    if len(input_signature_nchw) == 4:
        if int(input_signature_nchw[0]) < 0:
            signature[0] = -1
        if int(input_signature_nchw[1]) < 0:
            signature[1] = -1
        if onnx_sizes_hw is not None and len(onnx_sizes_hw) >= 2:
            signature[2] = int(onnx_sizes_hw[0])
            signature[3] = int(onnx_sizes_hw[1])
        elif onnx_scales_hw is not None and len(onnx_scales_hw) >= 2:
            if int(input_signature_nchw[2]) < 0:
                signature[2] = -1
            else:
                signature[2] = max(int(round(float(input_signature_nchw[2]) * float(onnx_scales_hw[0]))), 1)
            if int(input_signature_nchw[3]) < 0:
                signature[3] = -1
            else:
                signature[3] = max(int(round(float(input_signature_nchw[3]) * float(onnx_scales_hw[1]))), 1)
    return [int(v) for v in signature]


def _resolve_integer_resize_scales_hw(onnx_scales_hw: list[float] | None) -> list[int] | None:
    if onnx_scales_hw is None or len(onnx_scales_hw) < 2:
        return None
    scales_int: list[int] = []
    for raw_scale in onnx_scales_hw[:2]:
        scale = float(raw_scale)
        rounded = int(round(scale))
        if rounded <= 0:
            return None
        if abs(scale - float(rounded)) > 1e-6:
            return None
        scales_int.append(int(rounded))
    return scales_int


def _extract_resize_onnx_hw_hints(node: Any, ctx: Any) -> tuple[list[int] | None, list[float] | None]:
    onnx_sizes_hw: list[int] | None = None
    onnx_scales_hw: list[float] | None = None

    if len(node.inputs) >= 4:
        sizes_name = node.inputs[3].name
        if sizes_name != "":
            sizes = ctx.get_constant_array(sizes_name)
            if sizes is not None and int(np.asarray(sizes).size) >= 2:
                values = np.asarray(sizes).reshape(-1).astype(np.int64)
                if values.size >= 4:
                    onnx_sizes_hw = [int(values[-2]), int(values[-1])]
                elif values.size == 2:
                    onnx_sizes_hw = [int(values[0]), int(values[1])]

    if len(node.inputs) >= 3:
        scales_name = node.inputs[2].name
        if scales_name != "":
            scales = ctx.get_constant_array(scales_name)
            if scales is not None and int(np.asarray(scales).size) >= 2:
                values = np.asarray(scales).reshape(-1).astype(np.float32)
                if values.size >= 4:
                    onnx_scales_hw = [float(values[-2]), float(values[-1])]
                elif values.size == 2:
                    onnx_scales_hw = [float(values[0]), float(values[1])]

    return onnx_sizes_hw, onnx_scales_hw


def _resolve_resize_flags(node: Any) -> tuple[str, bool, bool]:
    mode = str(node.attrs.get("mode", "nearest")).lower()
    ctm = str(node.attrs.get("coordinate_transformation_mode", "half_pixel")).lower()
    align_corners = bool(ctm == "align_corners")
    half_pixel_centers = bool(ctm in {"half_pixel", "pytorch_half_pixel"})
    if mode == "nearest" and ctm == "asymmetric":
        align_corners = False
        half_pixel_centers = False
    return mode, align_corners, half_pixel_centers


def build_resize_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor = ctx.model_ir.tensors[output_name]
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )
    existing_output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None and len(list(output_tensor.shape_signature)) == 4
        else None
    )
    if len(input_shape) != 4:
        raise NotImplementedError(
            f"Resize supports only rank-4 tensors in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    has_dynamic_sizes_input = False
    if len(node.inputs) >= 4 and node.inputs[3].name != "":
        sizes_const = ctx.get_constant_array(node.inputs[3].name)
        has_dynamic_sizes_input = (
            sizes_const is None
            or int(np.asarray(sizes_const).size) == 0
        )
    if len(output_shape) != 4 or output_shape == [1]:
        if has_dynamic_sizes_input:
            output_shape = [int(input_shape[0]), int(input_shape[1]), 1, 1]
        else:
            out_h, out_w = _resolve_resize_target_hw(node, ctx, input_shape)
            output_shape = [int(input_shape[0]), int(input_shape[1]), int(out_h), int(out_w)]
        ctx.model_ir.tensors[output_name].shape = list(output_shape)

    input_dtype = str(ctx.get_tensor_dtype(input_name))
    output_dtype = str(ctx.get_tensor_dtype(output_name))
    if output_dtype == "FLOAT32" and input_dtype != "FLOAT32":
        ctx.model_ir.tensors[output_name].dtype = input_dtype
    if ctx.model_ir.tensors[output_name].quantization is None:
        in_quant = ctx.model_ir.tensors[input_name].quantization
        if in_quant is not None:
            ctx.model_ir.tensors[output_name].quantization = _clone_quantization(in_quant)

    mode, align_corners, half_pixel_centers = _resolve_resize_flags(node)
    tflite_op = "RESIZE_NEAREST_NEIGHBOR" if mode == "nearest" else "RESIZE_BILINEAR"
    onnx_sizes_hw, onnx_scales_hw = _extract_resize_onnx_hw_hints(node, ctx)
    output_signature = _infer_resize_output_signature_nchw(
        input_signature_nchw=input_signature,
        output_shape_nchw=output_shape,
        onnx_sizes_hw=onnx_sizes_hw,
        onnx_scales_hw=onnx_scales_hw,
        existing_output_signature_nchw=existing_output_signature,
    )
    output_tensor.shape_signature = [int(v) for v in list(output_signature)]

    nhwc_input_shape = [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
    nhwc_output_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[3]), int(output_shape[1])]
    nhwc_output_signature = [int(v) for v in list(nhwc_output_shape)]
    if len(output_signature) == 4:
        nhwc_output_signature = [
            int(output_signature[0]),
            int(output_signature[2]),
            int(output_signature[3]),
            int(output_signature[1]),
        ]

    x_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_input_nhwc",
        dtype=ctx.get_tensor_dtype(input_name),
        shape=nhwc_input_shape,
    )
    x_quant = ctx.model_ir.tensors[input_name].quantization
    if x_quant is not None:
        ctx.model_ir.tensors[x_nhwc].quantization = _clone_quantization(x_quant)
    x_nhwc = make_transpose(
        ctx,
        input_name,
        x_nhwc,
        [0, 2, 3, 1],
        allow_elide_inverse_chain=True,
    )

    size_input_name = ""
    dynamic_size_input_name = _build_resize_dynamic_size_input(node, ctx)
    has_dynamic_spatial_input = (
        len(input_signature) == 4 and (int(input_signature[2]) < 0 or int(input_signature[3]) < 0)
    )
    resize_scales_hw_int = _resolve_integer_resize_scales_hw(onnx_scales_hw)
    if dynamic_size_input_name is not None:
        size_input_name = dynamic_size_input_name
    elif has_dynamic_spatial_input and onnx_sizes_hw is None and onnx_scales_hw is not None:
        if resize_scales_hw_int is None:
            raise NotImplementedError(
                f"Resize with dynamic spatial input supports integer scales only in flatbuffer_direct. "
                f"op={node.name} scales_hw={onnx_scales_hw}"
            )
        input_hw_shape = ctx.add_intermediate_tensor(
            f"{node.name}_input_hw_shape",
            dtype="INT32",
            shape=[2],
        )
        shape_vec = ctx.add_intermediate_tensor(
            f"{node.name}_input_shape_vec",
            dtype="INT32",
            shape=[4],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[x_nhwc],
                outputs=[shape_vec],
                options={"outType": "INT32"},
            )
        )
        shape_begin = ctx.add_const_tensor(
            f"{node.name}_shape_slice_begin",
            np.asarray([1], dtype=np.int32),
        )
        shape_size = ctx.add_const_tensor(
            f"{node.name}_shape_slice_size",
            np.asarray([2], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[shape_vec, shape_begin, shape_size],
                outputs=[input_hw_shape],
            )
        )
        scales_const = ctx.add_const_tensor(
            f"{node.name}_resize_scales_hw_int",
            np.asarray([int(resize_scales_hw_int[0]), int(resize_scales_hw_int[1])], dtype=np.int32),
        )
        dynamic_size = ctx.add_intermediate_tensor(
            f"{node.name}_resize_size_dynamic",
            dtype="INT32",
            shape=[2],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[input_hw_shape, scales_const],
                outputs=[dynamic_size],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        size_input_name = dynamic_size
    else:
        size_const = ctx.add_const_tensor(
            f"{node.name}_resize_size",
            np.asarray([int(output_shape[2]), int(output_shape[3])], dtype=np.int32),
        )
        size_input_name = size_const
    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    y_quant = ctx.model_ir.tensors[output_name].quantization
    if y_quant is not None:
        ctx.model_ir.tensors[y_nhwc].quantization = _clone_quantization(y_quant)
    ctx.model_ir.tensors[y_nhwc].shape_signature = [int(v) for v in list(nhwc_output_signature)]

    ctx.add_operator(
        OperatorIR(
            op_type=tflite_op,
            inputs=[x_nhwc, size_input_name],
            outputs=[y_nhwc],
            options={
                "alignCorners": bool(align_corners),
                "halfPixelCenters": bool(half_pixel_centers),
                "onnxSizesHW": list(onnx_sizes_hw) if onnx_sizes_hw is not None else None,
                "onnxScalesHW": list(onnx_scales_hw) if onnx_scales_hw is not None else None,
            },
        )
    )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )
