from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


_DTYPE_TO_NP = {
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "BOOL": np.bool_,
}


def build_gather_op(node: Any, ctx: Any) -> None:
    params_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(params_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(output_name)

    input_rank = len(ctx.get_tensor_shape(params_name))
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims != 0:
        raise NotImplementedError(
            f"Gather batch_dims != 0 is not supported in flatbuffer_direct. "
            f"op={node.name} batch_dims={batch_dims}"
        )

    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[params_name, indices_name],
            outputs=[output_name],
            options={
                "axis": int(axis),
                "batchDims": int(batch_dims),
            },
        )
    )


def build_argmax_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_rank = len(input_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NotImplementedError(
            f"ArgMax axis out of range in flatbuffer_direct. op={node.name} axis={axis} rank={input_rank}"
        )

    select_last_index = int(node.attrs.get("select_last_index", 0))
    if select_last_index != 0:
        raise NotImplementedError(
            f"ArgMax select_last_index != 0 is not supported in flatbuffer_direct. "
            f"op={node.name} select_last_index={select_last_index}"
        )

    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if output_dtype not in {"INT32", "INT64"}:
        output_dtype = "INT64"

    argmax_output_name = output_name
    if keepdims:
        reduced_shape = [
            int(dim) for idx, dim in enumerate(input_shape) if idx != axis
        ]
        if len(reduced_shape) == 0:
            reduced_shape = [1]
        argmax_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_argmax",
            dtype=output_dtype,
            shape=reduced_shape,
        )

    axis_name = ctx.add_const_tensor(
        f"{output_name}_argmax_axis",
        np.asarray([axis], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ARG_MAX",
            inputs=[input_name, axis_name],
            outputs=[argmax_output_name],
            options={
                "outputType": output_dtype,
            },
        )
    )

    if keepdims:
        output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
        shape_name = ctx.add_const_tensor(
            f"{output_name}_argmax_keepdims_shape",
            np.asarray(output_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[argmax_output_name, shape_name],
                outputs=[output_name],
                options={
                    "newShape": output_shape,
                },
            )
        )


def build_one_hot_op(node: Any, ctx: Any) -> None:
    indices_name = node.inputs[0].name
    depth_name = node.inputs[1].name
    values_name = node.inputs[2].name
    output_name = node.outputs[0].name

    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(depth_name)
    ctx.ensure_tensor(values_name)
    ctx.ensure_tensor(output_name)

    depth_values = ctx.get_constant_array(depth_name)
    if depth_values is None:
        raise NotImplementedError(
            f"OneHot depth input must be constant for flatbuffer_direct. op={node.name}"
        )
    depth_arr = np.asarray(depth_values).reshape(-1)
    if depth_arr.size != 1:
        raise NotImplementedError(
            f"OneHot depth must be scalar. op={node.name} depth_shape={list(np.asarray(depth_values).shape)}"
        )
    depth_int = int(depth_arr[0])
    if depth_int <= 0:
        raise NotImplementedError(
            f"OneHot depth must be > 0. op={node.name} depth={depth_int}"
        )

    values = ctx.get_constant_array(values_name)
    if values is None:
        raise NotImplementedError(
            f"OneHot values input must be constant for flatbuffer_direct. op={node.name}"
        )
    values_arr = np.asarray(values).reshape(-1)
    if values_arr.size != 2:
        raise NotImplementedError(
            f"OneHot values must have exactly 2 elements [off, on]. op={node.name} size={int(values_arr.size)}"
        )

    indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
    indices_rank = len(indices_shape)
    axis = int(node.attrs.get("axis", -1))
    if axis < 0:
        axis += int(indices_rank + 1)
    if axis < 0 or axis > int(indices_rank):
        raise NotImplementedError(
            f"OneHot axis out of range in flatbuffer_direct. op={node.name} axis={axis} rank={indices_rank}"
        )

    output_tensor = ctx.model_ir.tensors[output_name]
    if output_tensor.shape == [1] and indices_rank >= 0:
        resolved_shape = list(indices_shape)
        resolved_shape.insert(int(axis), int(depth_int))
        output_tensor.shape = [int(v) for v in resolved_shape]
        output_tensor.shape_signature = [int(v) for v in resolved_shape]

    indices_cast_name = indices_name
    indices_dtype = str(ctx.get_tensor_dtype(indices_name)).upper()
    if indices_dtype != "INT32":
        indices_cast_name = ctx.add_intermediate_tensor(
            f"{output_name}_onehot_indices_i32",
            dtype="INT32",
            shape=indices_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[indices_name],
                outputs=[indices_cast_name],
                options={
                    "inDataType": indices_dtype,
                    "outDataType": "INT32",
                },
            )
        )

    depth_const_name = ctx.add_const_tensor(
        f"{node.name}_onehot_depth_i32",
        np.asarray(depth_int, dtype=np.int32),
    )

    # ONNX OneHot supports negative indices via wrap-around semantics.
    indices_add_depth_name = ctx.add_intermediate_tensor(
        f"{output_name}_onehot_indices_add_depth",
        dtype="INT32",
        shape=indices_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[indices_cast_name, depth_const_name],
            outputs=[indices_add_depth_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )

    indices_wrapped_name = ctx.add_intermediate_tensor(
        f"{output_name}_onehot_indices_wrapped",
        dtype="INT32",
        shape=indices_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="FLOOR_MOD",
            inputs=[indices_add_depth_name, depth_const_name],
            outputs=[indices_wrapped_name],
        )
    )

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    output_np_dtype = _DTYPE_TO_NP.get(output_dtype, None)
    if output_np_dtype is None:
        raise NotImplementedError(
            f"OneHot output dtype is not supported in flatbuffer_direct. op={node.name} dtype={output_dtype}"
        )

    off_name = ctx.add_const_tensor(
        f"{node.name}_onehot_off_value",
        np.asarray(values_arr[0], dtype=output_np_dtype),
    )
    on_name = ctx.add_const_tensor(
        f"{node.name}_onehot_on_value",
        np.asarray(values_arr[1], dtype=output_np_dtype),
    )

    ctx.add_operator(
        OperatorIR(
            op_type="ONE_HOT",
            inputs=[indices_wrapped_name, depth_const_name, on_name, off_name],
            outputs=[output_name],
            options={
                "axis": int(axis),
            },
        )
    )


def build_gather_elements_op(node: Any, ctx: Any) -> None:
    data_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(data_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(output_name)

    data_shape = [int(v) for v in ctx.get_tensor_shape(data_name)]
    indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    if len(data_shape) != len(indices_shape):
        raise NotImplementedError(
            "GatherElements requires data and indices with the same rank in flatbuffer_direct. "
            f"op={node.name} data_shape={data_shape} indices_shape={indices_shape}"
        )
    if indices_shape != output_shape:
        raise NotImplementedError(
            "GatherElements requires output shape equal to indices shape in flatbuffer_direct. "
            f"op={node.name} indices_shape={indices_shape} output_shape={output_shape}"
        )
    if any(int(v) <= 0 for v in output_shape):
        raise NotImplementedError(
            "GatherElements requires fully static positive output shape in flatbuffer_direct. "
            f"op={node.name} output_shape={output_shape}"
        )

    rank = len(data_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise NotImplementedError(
            f"GatherElements axis out of range in flatbuffer_direct. op={node.name} axis={axis} rank={rank}"
        )

    indices_i32_name = indices_name
    indices_dtype = str(ctx.get_tensor_dtype(indices_name)).upper()
    if indices_dtype != "INT32":
        indices_i32_name = ctx.add_intermediate_tensor(
            f"{output_name}_gather_elements_indices_i32",
            dtype="INT32",
            shape=indices_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[indices_name],
                outputs=[indices_i32_name],
                options={
                    "inDataType": indices_dtype,
                    "outDataType": "INT32",
                },
            )
        )

    axis_coord_shape = [int(v) for v in output_shape] + [1]
    axis_coord_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_elements_axis_coord",
        dtype="INT32",
        shape=axis_coord_shape,
    )
    axis_coord_shape_const = ctx.add_const_tensor(
        f"{output_name}_gather_elements_axis_coord_shape",
        np.asarray(axis_coord_shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[indices_i32_name, axis_coord_shape_const],
            outputs=[axis_coord_name],
            options={"newShape": [int(v) for v in axis_coord_shape]},
        )
    )

    grid = np.indices(output_shape, dtype=np.int32)
    coord_tensors: list[str] = []
    for dim in range(rank):
        if dim == axis:
            coord_tensors.append(axis_coord_name)
            continue
        coord_const = ctx.add_const_tensor(
            f"{output_name}_gather_elements_coord_{dim}",
            np.expand_dims(grid[dim], axis=-1),
        )
        coord_tensors.append(coord_const)

    coords_name = coord_tensors[0]
    if len(coord_tensors) > 1:
        coords_name = ctx.add_intermediate_tensor(
            f"{output_name}_gather_elements_coords",
            dtype="INT32",
            shape=[int(v) for v in output_shape] + [int(rank)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=coord_tensors,
                outputs=[coords_name],
                options={
                    "axis": int(rank),
                    "fusedActivationFunction": "NONE",
                },
            )
        )

    ctx.add_operator(
        OperatorIR(
            op_type="GATHER_ND",
            inputs=[data_name, coords_name],
            outputs=[output_name],
        )
    )
