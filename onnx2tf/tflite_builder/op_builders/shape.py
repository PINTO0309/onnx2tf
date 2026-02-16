from __future__ import annotations

from typing import Any
import copy

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


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

    begin = [0 for _ in range(rank)]
    end_for_strided = [
        int(dim) if int(dim) > 0 else int(np.iinfo(np.int32).max)
        for dim in input_shape
    ]
    strides_for_strided = [1 for _ in range(rank)]
    size = [int(dim) if int(dim) > 0 else -1 for dim in input_shape]
    large_int = int(np.iinfo(np.int64).max // 2)
    use_strided_slice = False

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
        dim = int(input_shape[axis]) if axis < len(input_shape) else -1

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

    _ = _resolve_axes_from_attr_or_input(node, ctx)
    output_shape = ctx.get_tensor_shape(output_name)
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
    if len(output_shape) != 4 or output_shape == [1]:
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
    has_dynamic_spatial_input = (
        len(input_signature) == 4 and (int(input_signature[2]) < 0 or int(input_signature[3]) < 0)
    )
    resize_scales_hw_int = _resolve_integer_resize_scales_hw(onnx_scales_hw)
    if has_dynamic_spatial_input and onnx_sizes_hw is None and onnx_scales_hw is not None:
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
