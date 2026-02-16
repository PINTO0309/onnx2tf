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
    size = [int(dim) if int(dim) > 0 else -1 for dim in input_shape]
    large_int = int(np.iinfo(np.int64).max // 2)

    for idx, axis_raw in enumerate(axes):
        axis = _normalize_axis(axis_raw, rank, op_name=node.name)
        step = int(steps[idx])
        if step != 1:
            raise NotImplementedError(
                f"Slice step must be 1 for flatbuffer_direct. op={node.name} step={step}"
            )
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
            size[axis] = int(max(end - start, 0))
        else:
            if start < 0:
                raise NotImplementedError(
                    f"Slice with negative start on unknown dimension is not supported. "
                    f"op={node.name} axis={axis} start={start}"
                )
            begin[axis] = int(start)
            if end >= large_int:
                size[axis] = -1
            elif end < 0:
                raise NotImplementedError(
                    f"Slice with negative end on unknown dimension is not supported. "
                    f"op={node.name} axis={axis} end={end}"
                )
            else:
                size[axis] = int(max(end - start, 0))

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

    shape_values = ctx.get_constant_array(shape_name)
    if shape_values is None:
        raise NotImplementedError(
            f"Reshape shape tensor must be constant for flatbuffer_direct. op={node.name}"
        )
    new_shape = [int(v) for v in np.asarray(shape_values).reshape(-1).tolist()]
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
            options={"newShape": new_shape},
        )
    )


def build_transpose_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

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

    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
    axes = _resolve_axes_from_attr_or_input(node, ctx)
    if len(axes) == 0:
        axes = [idx for idx, dim in enumerate(input_shape) if int(dim) == 1]

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


def _resolve_resize_flags(node: Any) -> tuple[str, bool, bool]:
    mode = str(node.attrs.get("mode", "nearest")).lower()
    ctm = str(node.attrs.get("coordinate_transformation_mode", "half_pixel")).lower()
    align_corners = bool(ctm == "align_corners")
    half_pixel_centers = bool(ctm == "half_pixel")
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
    if len(input_shape) != 4:
        raise NotImplementedError(
            f"Resize supports only rank-4 tensors in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    if len(output_shape) != 4 or output_shape == [1]:
        out_h, out_w = _resolve_resize_target_hw(node, ctx, input_shape)
        output_shape = [int(input_shape[0]), int(input_shape[1]), int(out_h), int(out_w)]
        ctx.model_ir.tensors[output_name].shape = list(output_shape)
        ctx.model_ir.tensors[output_name].shape_signature = list(output_shape)

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

    nhwc_input_shape = [int(input_shape[0]), int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
    nhwc_output_shape = [int(output_shape[0]), int(output_shape[2]), int(output_shape[3]), int(output_shape[1])]

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

    size_const = ctx.add_const_tensor(
        f"{node.name}_resize_size",
        np.asarray([int(output_shape[2]), int(output_shape[3])], dtype=np.int32),
    )
    y_nhwc = ctx.add_intermediate_tensor(
        f"{node.name}_output_nhwc",
        dtype=ctx.get_tensor_dtype(output_name),
        shape=nhwc_output_shape,
    )
    y_quant = ctx.model_ir.tensors[output_name].quantization
    if y_quant is not None:
        ctx.model_ir.tensors[y_nhwc].quantization = _clone_quantization(y_quant)

    ctx.add_operator(
        OperatorIR(
            op_type=tflite_op,
            inputs=[x_nhwc, size_const],
            outputs=[y_nhwc],
            options={
                "alignCorners": bool(align_corners),
                "halfPixelCenters": bool(half_pixel_centers),
            },
        )
    )
    make_transpose(
        ctx,
        y_nhwc,
        output_name,
        [0, 3, 1, 2],
    )
