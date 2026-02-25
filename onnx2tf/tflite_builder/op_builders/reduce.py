from __future__ import annotations

from typing import Any, List

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def _normalize_axes(axes: List[int], rank: int, node_name: str) -> List[int]:
    normalized: List[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += rank
        if a < 0 or a >= rank:
            raise NotImplementedError(
                f"Reduce axis is out of range. op={node_name} axis={axis} rank={rank}"
            )
        if a not in normalized:
            normalized.append(a)
    return normalized


def _resolve_reduce_axes(node: Any, ctx: Any, input_rank: int) -> List[int]:
    axes: List[int]
    if len(node.inputs) >= 2:
        axes_arr = ctx.get_constant_array(node.inputs[1].name)
        if axes_arr is None:
            raise NotImplementedError(
                f"Reduce axes must be constant for flatbuffer_direct. op={node.name}"
            )
        axes = [int(v) for v in np.asarray(axes_arr).reshape(-1).tolist()]
    else:
        attr_axes = node.attrs.get("axes", None)
        if attr_axes is None:
            axes = [int(v) for v in range(input_rank)]
        elif isinstance(attr_axes, (list, tuple)):
            axes = [int(v) for v in attr_axes]
        else:
            axes = [int(attr_axes)]

    if len(axes) == 0:
        if int(node.attrs.get("noop_with_empty_axes", 0)) == 1:
            return []
        axes = [int(v) for v in range(input_rank)]
    return _normalize_axes(axes, input_rank, node.name)


def _resolve_cumsum_axis(node: Any, ctx: Any, input_rank: int) -> int:
    axis_raw: int
    if len(node.inputs) >= 2 and str(node.inputs[1].name) != "":
        axis_arr = ctx.get_constant_array(node.inputs[1].name)
        if axis_arr is None:
            raise NotImplementedError(
                f"CumSum axis must be constant for flatbuffer_direct. op={node.name}"
            )
        axis_values = np.asarray(axis_arr).reshape(-1)
        if int(axis_values.size) != 1:
            raise NotImplementedError(
                f"CumSum axis must be scalar. op={node.name} axis_shape={list(np.asarray(axis_arr).shape)}"
            )
        axis_raw = int(axis_values[0])
    else:
        axis_raw = int(node.attrs.get("axis", 0))

    axis = int(axis_raw)
    if axis < 0:
        axis += int(input_rank)
    if axis < 0 or axis >= int(input_rank):
        raise NotImplementedError(
            f"CumSum axis is out of range. op={node.name} axis={axis_raw} rank={input_rank}"
        )
    return int(axis)


def build_cumsum_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    axis = _resolve_cumsum_axis(node=node, ctx=ctx, input_rank=len(input_shape))
    axis_const = ctx.add_const_tensor(
        f"{output_name}_cumsum_axis",
        np.asarray([axis], dtype=np.int32),
    )
    exclusive = bool(int(node.attrs.get("exclusive", 0)))
    reverse = bool(int(node.attrs.get("reverse", 0)))

    ctx.add_operator(
        OperatorIR(
            op_type="CUMSUM",
            inputs=[input_name, axis_const],
            outputs=[output_name],
            options={
                "exclusive": exclusive,
                "reverse": reverse,
            },
        )
    )


def build_reduce_op(node: Any, ctx: Any, op_type: str) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    axes = _resolve_reduce_axes(node, ctx, len(input_shape))
    if len(axes) == 0 and int(node.attrs.get("noop_with_empty_axes", 0)) == 1:
        shape_const = ctx.add_const_tensor(
            f"{output_name}_reduce_noop_shape",
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

    axes_const = ctx.add_const_tensor(
        f"{output_name}_{op_type.lower()}_axes",
        np.asarray(axes, dtype=np.int32),
    )
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[input_name, axes_const],
            outputs=[output_name],
            options={"keepDims": keepdims},
        )
    )


def build_global_average_pool_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) < 3:
        raise NotImplementedError(
            f"GlobalAveragePool requires rank>=3. op={node.name} input_shape={input_shape}"
        )

    spatial_axes = [int(v) for v in range(2, len(input_shape))]
    axes_const = ctx.add_const_tensor(
        f"{output_name}_global_avg_axes",
        np.asarray(spatial_axes, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MEAN",
            inputs=[input_name, axes_const],
            outputs=[output_name],
            options={"keepDims": True},
        )
    )


def build_global_max_pool_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) < 3:
        raise NotImplementedError(
            f"GlobalMaxPool requires rank>=3. op={node.name} input_shape={input_shape}"
        )

    spatial_axes = [int(v) for v in range(2, len(input_shape))]
    axes_const = ctx.add_const_tensor(
        f"{output_name}_global_max_axes",
        np.asarray(spatial_axes, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="REDUCE_MAX",
            inputs=[input_name, axes_const],
            outputs=[output_name],
            options={"keepDims": True},
        )
    )


def _build_sum_reduce_from_input(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
) -> None:
    input_shape = ctx.get_tensor_shape(input_name)
    output_shape = ctx.get_tensor_shape(output_name)
    axes = _resolve_reduce_axes(node, ctx, len(input_shape))
    if len(axes) == 0 and int(node.attrs.get("noop_with_empty_axes", 0)) == 1:
        shape_const = ctx.add_const_tensor(
            f"{output_name}_reduce_noop_shape",
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
    axes_const = ctx.add_const_tensor(
        f"{output_name}_sum_axes",
        np.asarray(axes, dtype=np.int32),
    )
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    ctx.add_operator(
        OperatorIR(
            op_type="SUM",
            inputs=[input_name, axes_const],
            outputs=[output_name],
            options={"keepDims": keepdims},
        )
    )


def build_reduce_l1_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    abs_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_l1_abs",
        dtype=str(ctx.get_tensor_dtype(input_name)).upper(),
        shape=[int(v) for v in ctx.get_tensor_shape(input_name)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ABS",
            inputs=[input_name],
            outputs=[abs_name],
        )
    )
    _build_sum_reduce_from_input(
        node=node,
        ctx=ctx,
        input_name=abs_name,
        output_name=output_name,
    )


def build_reduce_l2_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"

    square_input_name = input_name
    if input_dtype != compute_dtype:
        cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_l2_input_cast",
            dtype=compute_dtype,
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[cast_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )
        square_input_name = cast_name

    squared_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_l2_squared",
        dtype=compute_dtype,
        shape=input_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[square_input_name, square_input_name],
            outputs=[squared_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    sum_name = ctx.add_intermediate_tensor(
        f"{output_name}_reduce_l2_sum",
        dtype=compute_dtype,
        shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
    )
    _build_sum_reduce_from_input(
        node=node,
        ctx=ctx,
        input_name=squared_name,
        output_name=sum_name,
    )

    sqrt_name = output_name
    if output_dtype != compute_dtype:
        sqrt_name = ctx.add_intermediate_tensor(
            f"{output_name}_reduce_l2_sqrt",
            dtype=compute_dtype,
            shape=[int(v) for v in ctx.get_tensor_shape(output_name)],
        )
    ctx.add_operator(
        OperatorIR(
            op_type="SQRT",
            inputs=[sum_name],
            outputs=[sqrt_name],
        )
    )

    if output_dtype != compute_dtype:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[sqrt_name],
                outputs=[output_name],
                options={
                    "inDataType": compute_dtype,
                    "outDataType": output_dtype,
                },
            )
        )
