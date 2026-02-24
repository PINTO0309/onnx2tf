from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import _clone_quantization, make_transpose


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


def _inverse_permutation(perm: list[int]) -> list[int]:
    inv = [0] * int(len(perm))
    for out_axis, in_axis in enumerate(perm):
        inv[int(in_axis)] = int(out_axis)
    return inv


def build_gather_op(node: Any, ctx: Any) -> None:
    params_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(params_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(params_name)]
    params_tensor = ctx.model_ir.tensors.get(params_name, None)
    input_signature = (
        [int(v) for v in list(params_tensor.shape_signature)]
        if params_tensor is not None and params_tensor.shape_signature is not None
        else [int(v) for v in input_shape]
    )
    input_rank = len(input_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims != 0:
        raise NotImplementedError(
            f"Gather batch_dims != 0 is not supported in flatbuffer_direct. "
            f"op={node.name} batch_dims={batch_dims}"
        )

    output_tensor = ctx.model_ir.tensors[output_name]
    existing_output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_tensor.shape)]
    )
    scalarized_indices_name = indices_name
    indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
    indices_tensor = ctx.model_ir.tensors.get(indices_name, None)
    indices_signature = (
        [int(v) for v in list(indices_tensor.shape_signature)]
        if indices_tensor is not None and indices_tensor.shape_signature is not None
        else [int(v) for v in indices_shape]
    )
    indices_const = ctx.get_constant_array(indices_name)
    if indices_const is not None:
        indices_const_arr = np.asarray(indices_const)
        if np.issubdtype(indices_const_arr.dtype, np.integer) and bool(np.any(indices_const_arr < 0)):
            axis_dim = int(input_shape[int(axis)]) if int(axis) < int(len(input_shape)) else -1
            if axis_dim <= 0:
                raise NotImplementedError(
                    f"Gather negative constant indices require known positive axis dimension. "
                    f"op={node.name} axis={axis} axis_dim={axis_dim}"
                )
            wrapped_indices_i64 = np.where(
                indices_const_arr.astype(np.int64, copy=False) < 0,
                indices_const_arr.astype(np.int64, copy=False) + int(axis_dim),
                indices_const_arr.astype(np.int64, copy=False),
            )
            if bool(np.any(wrapped_indices_i64 < 0)) or bool(np.any(wrapped_indices_i64 >= int(axis_dim))):
                raise NotImplementedError(
                    f"Gather constant indices are out of bounds after negative-index normalization. "
                    f"op={node.name} axis={axis} axis_dim={axis_dim}"
                )
            indices_const_arr = wrapped_indices_i64.astype(indices_const_arr.dtype, copy=False)
            scalarized_indices_name = ctx.add_const_tensor(
                f"{output_name}_gather_indices_wrapped",
                indices_const_arr,
            )
            indices_shape = [int(v) for v in list(indices_const_arr.shape)]
            indices_signature = [int(v) for v in list(indices_const_arr.shape)]
        scalarize_single_index = indices_const_arr.ndim == 0
        if not scalarize_single_index and int(indices_const_arr.size) == 1 and input_rank > 1:
            expected_rank = len(existing_output_signature)
            scalar_output_rank = int(input_rank) - 1
            scalarize_single_index = expected_rank == scalar_output_rank
        if scalarize_single_index:
            scalar_value = np.asarray(indices_const_arr.reshape(-1)[0], dtype=indices_const_arr.dtype)
            scalarized_indices_name = ctx.add_const_tensor(
                f"{output_name}_gather_indices_scalar",
                scalar_value,
            )
            scalar_tensor = ctx.model_ir.tensors.get(scalarized_indices_name, None)
            if scalar_tensor is not None:
                scalar_tensor.shape = []
                scalar_tensor.shape_signature = []
            indices_shape = []
            indices_signature = []

    inferred_output_shape = (
        [int(v) for v in input_shape[:int(axis)]]
        + [int(v) for v in indices_shape]
        + [int(v) for v in input_shape[int(axis) + 1:]]
    )
    inferred_output_signature = (
        [int(v) for v in input_signature[:int(axis)]]
        + [int(v) for v in indices_signature]
        + [int(v) for v in input_signature[int(axis) + 1:]]
    )
    if len(inferred_output_signature) == 0:
        inferred_output_signature = [1]
    existing_output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else None
    )
    final_output_signature = [int(v) for v in inferred_output_signature]
    if (
        existing_output_signature is not None
        and len(existing_output_signature) == len(inferred_output_signature)
    ):
        final_output_signature = [
            int(existing_dim) if int(existing_dim) < 0 else int(inferred_dim)
            for existing_dim, inferred_dim in zip(existing_output_signature, inferred_output_signature)
        ]
    output_tensor.shape_signature = [int(v) for v in final_output_signature]
    output_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in final_output_signature]

    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[params_name, scalarized_indices_name],
            outputs=[output_name],
            options={
                "axis": int(axis),
                "batchDims": int(batch_dims),
            },
        )
    )


def build_gather_nd_op(node: Any, ctx: Any) -> None:
    params_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(params_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(output_name)

    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims != 0:
        raise NotImplementedError(
            f"GatherND batch_dims != 0 is not supported in flatbuffer_direct. "
            f"op={node.name} batch_dims={batch_dims}"
        )

    indices_for_gather_nd = indices_name
    indices_dtype = str(ctx.get_tensor_dtype(indices_name)).upper()
    if indices_dtype != "INT32":
        indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
        indices_for_gather_nd = ctx.add_intermediate_tensor(
            f"{output_name}_gather_nd_indices_i32",
            dtype="INT32",
            shape=indices_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[indices_name],
                outputs=[indices_for_gather_nd],
                options={
                    "inDataType": indices_dtype,
                    "outDataType": "INT32",
                },
            )
        )

    ctx.add_operator(
        OperatorIR(
            op_type="GATHER_ND",
            inputs=[params_name, indices_for_gather_nd],
            outputs=[output_name],
        )
    )


def build_scatter_nd_op(node: Any, ctx: Any) -> None:
    data_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    updates_name = node.inputs[2].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(data_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(updates_name)
    ctx.ensure_tensor(output_name)

    data_dtype = str(ctx.get_tensor_dtype(data_name)).upper()
    output_tensor = ctx.model_ir.tensors[output_name]
    data_tensor = ctx.model_ir.tensors[data_name]
    output_tensor.dtype = data_dtype
    output_tensor.quantization = _clone_quantization(data_tensor.quantization)
    _propagate_shape(ctx, data_name, output_name)

    indices_for_scatter = indices_name
    indices_dtype = str(ctx.get_tensor_dtype(indices_name)).upper()
    if indices_dtype != "INT32":
        indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
        indices_for_scatter = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_nd_indices_i32",
            dtype="INT32",
            shape=indices_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[indices_name],
                outputs=[indices_for_scatter],
                options={
                    "inDataType": indices_dtype,
                    "outDataType": "INT32",
                },
            )
        )

    data_shape = [int(v) for v in ctx.get_tensor_shape(data_name)]
    rank = int(len(data_shape))
    shape_for_scatter = ""
    if rank > 0 and all(int(dim) > 0 for dim in data_shape):
        shape_for_scatter = ctx.add_const_tensor(
            f"{output_name}_scatter_nd_shape",
            np.asarray(data_shape, dtype=np.int32),
        )
    else:
        shape_for_scatter = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_nd_shape",
            dtype="INT32",
            shape=[rank] if rank > 0 else [1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[data_name],
                outputs=[shape_for_scatter],
                options={"outType": "INT32"},
            )
        )

    updates_shape = [int(v) for v in ctx.get_tensor_shape(updates_name)]
    ones_scalar = ctx.add_const_tensor(
        f"{output_name}_scatter_nd_one",
        np.asarray(1, dtype=_DTYPE_TO_NP[data_dtype]),
    )
    updates_ones = ""
    if len(updates_shape) > 0 and all(int(dim) > 0 for dim in updates_shape):
        updates_ones = ctx.add_const_tensor(
            f"{output_name}_scatter_nd_updates_ones",
            np.ones(updates_shape, dtype=_DTYPE_TO_NP[data_dtype]),
        )
    else:
        updates_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_nd_updates_shape",
            dtype="INT32",
            shape=[len(updates_shape)] if len(updates_shape) > 0 else [1],
        )
        updates_ones = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_nd_updates_ones",
            dtype=data_dtype,
            shape=updates_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[updates_name],
                outputs=[updates_shape_name],
                options={"outType": "INT32"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="FILL",
                inputs=[updates_shape_name, ones_scalar],
                outputs=[updates_ones],
            )
        )

    mask_scatter = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_nd_mask",
        dtype=data_dtype,
        shape=data_shape,
    )
    inverse_mask = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_nd_inverse_mask",
        dtype=data_dtype,
        shape=data_shape,
    )
    retained = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_nd_retained",
        dtype=data_dtype,
        shape=data_shape,
    )
    scattered_updates = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_nd_updates",
        dtype=data_dtype,
        shape=data_shape,
    )

    ctx.add_operator(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=[indices_for_scatter, updates_ones, shape_for_scatter],
            outputs=[mask_scatter],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SUB",
            inputs=[ones_scalar, mask_scatter],
            outputs=[inverse_mask],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[data_name, inverse_mask],
            outputs=[retained],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=[indices_for_scatter, updates_name, shape_for_scatter],
            outputs=[scattered_updates],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[retained, scattered_updates],
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
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


def build_argmin_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_rank = len(input_shape)
    axis = _normalize_axis_for_rank(int(node.attrs.get("axis", 0)), input_rank)
    select_last_index = int(node.attrs.get("select_last_index", 0))
    if select_last_index != 0:
        raise NotImplementedError(
            f"ArgMin select_last_index != 0 is not supported in flatbuffer_direct. "
            f"op={node.name} select_last_index={select_last_index}"
        )

    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if output_dtype not in {"INT32", "INT64"}:
        output_dtype = "INT64"

    argmin_output_name = output_name
    if keepdims:
        reduced_shape = [
            int(dim) for idx, dim in enumerate(input_shape) if idx != axis
        ]
        if len(reduced_shape) == 0:
            reduced_shape = [1]
        argmin_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_argmin",
            dtype=output_dtype,
            shape=reduced_shape,
        )

    axis_name = ctx.add_const_tensor(
        f"{output_name}_argmin_axis",
        np.asarray([axis], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ARG_MIN",
            inputs=[input_name, axis_name],
            outputs=[argmin_output_name],
            options={"outputType": output_dtype},
        )
    )

    if keepdims:
        output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
        shape_name = ctx.add_const_tensor(
            f"{output_name}_argmin_keepdims_shape",
            np.asarray(output_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[argmin_output_name, shape_name],
                outputs=[output_name],
                options={"newShape": output_shape},
            )
        )


def build_topk_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    k_name = node.inputs[1].name
    values_output_name = node.outputs[0].name
    indices_output_name = node.outputs[1].name if len(node.outputs) >= 2 else ""
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(k_name)
    ctx.ensure_tensor(values_output_name)
    if indices_output_name != "":
        ctx.ensure_tensor(indices_output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_rank = int(len(input_shape))
    axis = _normalize_axis_for_rank(int(node.attrs.get("axis", -1)), input_rank)
    largest = bool(int(node.attrs.get("largest", 1)))
    sorted_values = bool(int(node.attrs.get("sorted", 1)))
    if not sorted_values:
        raise NotImplementedError(
            f"TopK sorted=0 is not supported in flatbuffer_direct. op={node.name}"
        )

    values_output_shape = [int(v) for v in ctx.get_tensor_shape(values_output_name)]
    values_output_tensor = ctx.model_ir.tensors.get(values_output_name, None)
    values_output_signature = (
        [int(v) for v in list(values_output_tensor.shape_signature)]
        if values_output_tensor is not None and values_output_tensor.shape_signature is not None
        else [int(v) for v in values_output_shape]
    )
    indices_output_shape = (
        [int(v) for v in ctx.get_tensor_shape(indices_output_name)]
        if indices_output_name != ""
        else [int(v) for v in values_output_shape]
    )
    indices_output_tensor = ctx.model_ir.tensors.get(indices_output_name, None)
    indices_output_signature = (
        [int(v) for v in list(indices_output_tensor.shape_signature)]
        if indices_output_tensor is not None and indices_output_tensor.shape_signature is not None
        else [int(v) for v in indices_output_shape]
    )

    work_input_name = input_name
    perm_to_last: list[int] | None = None
    perm_from_last: list[int] | None = None
    if axis != input_rank - 1:
        perm_to_last = [int(v) for v in range(input_rank) if int(v) != int(axis)] + [int(axis)]
        perm_from_last = _inverse_permutation(perm_to_last)
        transposed_input_shape = [int(input_shape[int(v)]) for v in perm_to_last]
        transposed_input_name = ctx.add_intermediate_tensor(
            f"{values_output_name}_topk_transposed_input",
            dtype=str(ctx.get_tensor_dtype(input_name)).upper(),
            shape=transposed_input_shape,
        )
        make_transpose(
            ctx=ctx,
            input_name=input_name,
            output_name=transposed_input_name,
            perm_values=perm_to_last,
        )
        work_input_name = transposed_input_name

    topk_input_name = work_input_name
    if not largest:
        neg_input_shape = [int(v) for v in ctx.get_tensor_shape(work_input_name)]
        neg_input_name = ctx.add_intermediate_tensor(
            f"{values_output_name}_topk_neg_input",
            dtype=str(ctx.get_tensor_dtype(work_input_name)).upper(),
            shape=neg_input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="NEG",
                inputs=[work_input_name],
                outputs=[neg_input_name],
            )
        )
        topk_input_name = neg_input_name

    k_for_topk_name = k_name
    k_dtype = str(ctx.get_tensor_dtype(k_for_topk_name)).upper()
    if k_dtype != "INT32":
        k_shape = [int(v) for v in ctx.get_tensor_shape(k_for_topk_name)]
        k_i32_name = ctx.add_intermediate_tensor(
            f"{values_output_name}_topk_k_i32",
            dtype="INT32",
            shape=k_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[k_for_topk_name],
                outputs=[k_i32_name],
                options={
                    "inDataType": k_dtype,
                    "outDataType": "INT32",
                },
            )
        )
        k_for_topk_name = k_i32_name

    k_shape = [int(v) for v in ctx.get_tensor_shape(k_for_topk_name)]
    if len(k_shape) == 1:
        if int(k_shape[0]) > 1:
            raise NotImplementedError(
                "TopK k input must be scalar-like (shape [] or [1]) in flatbuffer_direct. "
                f"op={node.name} k_shape={k_shape}"
            )
        k_scalar_name = ctx.add_intermediate_tensor(
            f"{values_output_name}_topk_k_scalar",
            dtype="INT32",
            shape=[],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[k_for_topk_name],
                outputs=[k_scalar_name],
                options={"squeezeDims": [0]},
            )
        )
        k_for_topk_name = k_scalar_name
    elif len(k_shape) != 0:
        raise NotImplementedError(
            "TopK k input must be scalar-like (shape [] or [1]) in flatbuffer_direct. "
            f"op={node.name} k_shape={k_shape}"
        )

    topk_values_shape = (
        [int(values_output_shape[int(v)]) for v in perm_to_last]
        if perm_to_last is not None and len(values_output_shape) == len(perm_to_last)
        else [int(v) for v in values_output_shape]
    )
    topk_values_signature = (
        [int(values_output_signature[int(v)]) for v in perm_to_last]
        if perm_to_last is not None and len(values_output_signature) == len(perm_to_last)
        else [int(v) for v in values_output_signature]
    )
    topk_indices_shape = (
        [int(indices_output_shape[int(v)]) for v in perm_to_last]
        if perm_to_last is not None and len(indices_output_shape) == len(perm_to_last)
        else [int(v) for v in indices_output_shape]
    )
    topk_indices_signature = (
        [int(indices_output_signature[int(v)]) for v in perm_to_last]
        if perm_to_last is not None and len(indices_output_signature) == len(perm_to_last)
        else [int(v) for v in indices_output_signature]
    )

    values_topk_name = (
        values_output_name
        if largest and perm_from_last is None
        else ctx.add_intermediate_tensor(
            f"{values_output_name}_topk_values_raw",
            dtype=str(ctx.get_tensor_dtype(values_output_name)).upper(),
            shape=topk_values_shape,
        )
    )
    if indices_output_name != "":
        indices_topk_name = (
            indices_output_name
            if perm_from_last is None and str(ctx.get_tensor_dtype(indices_output_name)).upper() == "INT32"
            else ctx.add_intermediate_tensor(
                f"{indices_output_name}_topk_indices_raw",
                dtype="INT32",
                shape=topk_indices_shape,
            )
        )
    else:
        indices_topk_name = ctx.add_intermediate_tensor(
            f"{values_output_name}_topk_indices_raw_unused",
            dtype="INT32",
            shape=topk_indices_shape,
        )
    values_topk_tensor = ctx.model_ir.tensors.get(values_topk_name, None)
    if values_topk_tensor is not None:
        values_topk_tensor.shape_signature = [int(v) for v in topk_values_signature]
    indices_topk_tensor = ctx.model_ir.tensors.get(indices_topk_name, None)
    if indices_topk_tensor is not None:
        indices_topk_tensor.shape_signature = [int(v) for v in topk_indices_signature]

    ctx.add_operator(
        OperatorIR(
            op_type="TOPK_V2",
            inputs=[topk_input_name, k_for_topk_name],
            outputs=[values_topk_name, indices_topk_name],
        )
    )

    values_post_largest_name = values_topk_name
    if not largest:
        values_post_largest_name = (
            values_output_name
            if perm_from_last is None
            else ctx.add_intermediate_tensor(
                f"{values_output_name}_topk_values_largest",
                dtype=str(ctx.get_tensor_dtype(values_output_name)).upper(),
                shape=topk_values_shape,
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="NEG",
                inputs=[values_topk_name],
                outputs=[values_post_largest_name],
            )
        )
        values_post_largest_tensor = ctx.model_ir.tensors.get(values_post_largest_name, None)
        if values_post_largest_tensor is not None:
            values_post_largest_tensor.shape_signature = [int(v) for v in topk_values_signature]

    values_final_name = values_post_largest_name
    indices_final_i32_name = indices_topk_name
    if perm_from_last is not None:
        if values_final_name != values_output_name:
            make_transpose(
                ctx=ctx,
                input_name=values_final_name,
                output_name=values_output_name,
                perm_values=perm_from_last,
            )
            values_final_name = values_output_name

        indices_transposed_name = (
            indices_output_name
            if indices_output_name != ""
            and str(ctx.get_tensor_dtype(indices_output_name)).upper() == "INT32"
            else ctx.add_intermediate_tensor(
                f"{values_output_name}_topk_indices_axis_restored",
                dtype="INT32",
                shape=indices_output_shape,
            )
        )
        make_transpose(
            ctx=ctx,
            input_name=indices_final_i32_name,
            output_name=indices_transposed_name,
            perm_values=perm_from_last,
        )
        if indices_transposed_name != indices_output_name:
            indices_transposed_tensor = ctx.model_ir.tensors.get(indices_transposed_name, None)
            if indices_transposed_tensor is not None:
                indices_transposed_tensor.shape_signature = [int(v) for v in indices_output_signature]
        indices_final_i32_name = indices_transposed_name

    if indices_output_name != "":
        indices_dtype = str(ctx.get_tensor_dtype(indices_output_name)).upper()
        if indices_dtype != "INT32":
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[indices_final_i32_name],
                    outputs=[indices_output_name],
                    options={
                        "inDataType": "INT32",
                        "outDataType": indices_dtype,
                    },
                )
            )


def build_hardmax_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _propagate_shape(ctx, input_name, output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    rank = len(input_shape)
    if rank <= 0:
        raise NotImplementedError(f"Hardmax requires rank >= 1. op={node.name} shape={input_shape}")
    axis = int(node.attrs.get("axis", 1))
    axis = _normalize_axis_for_rank(axis=axis, rank=rank)

    work_input_name = input_name
    transposed_shape = list(input_shape)
    perm_to_last: list[int] | None = None
    perm_from_last: list[int] | None = None
    if axis != rank - 1:
        perm_to_last = [int(v) for v in range(rank) if int(v) != axis] + [int(axis)]
        perm_from_last = [0] * int(rank)
        for out_axis, in_axis in enumerate(perm_to_last):
            perm_from_last[int(in_axis)] = int(out_axis)
        transposed_shape = [int(input_shape[int(v)]) for v in perm_to_last]
        transposed_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_hardmax_input_axis_last",
            dtype=str(ctx.get_tensor_dtype(input_name)).upper(),
            shape=transposed_shape,
        )
        work_input_name = make_transpose(
            ctx=ctx,
            input_name=input_name,
            output_name=transposed_input_name,
            perm_values=perm_to_last,
        )

    depth = int(transposed_shape[-1])
    if depth <= 0:
        raise NotImplementedError(
            "Hardmax requires static positive depth on target axis in flatbuffer_direct. "
            f"op={node.name} axis={axis} shape={input_shape}"
        )

    indices_shape = [int(v) for v in transposed_shape[:-1]]
    if len(indices_shape) == 0:
        indices_shape = [1]
    indices_name = ctx.add_intermediate_tensor(
        f"{output_name}_hardmax_indices",
        dtype="INT32",
        shape=indices_shape,
    )
    argmax_axis_name = ctx.add_const_tensor(
        f"{output_name}_hardmax_axis",
        np.asarray([rank - 1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ARG_MAX",
            inputs=[work_input_name, argmax_axis_name],
            outputs=[indices_name],
            options={"outputType": "INT32"},
        )
    )

    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    output_np_dtype = _DTYPE_TO_NP.get(output_dtype, None)
    if output_np_dtype is None:
        raise NotImplementedError(
            f"Hardmax output dtype is not supported in flatbuffer_direct. op={node.name} dtype={output_dtype}"
        )
    off_name = ctx.add_const_tensor(
        f"{output_name}_hardmax_off",
        np.asarray(0, dtype=output_np_dtype),
    )
    on_name = ctx.add_const_tensor(
        f"{output_name}_hardmax_on",
        np.asarray(1, dtype=output_np_dtype),
    )
    depth_name = ctx.add_const_tensor(
        f"{output_name}_hardmax_depth",
        np.asarray(depth, dtype=np.int32),
    )

    onehot_output_name = output_name
    if perm_from_last is not None:
        onehot_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_hardmax_axis_last",
            dtype=output_dtype,
            shape=transposed_shape,
        )
    ctx.add_operator(
        OperatorIR(
            op_type="ONE_HOT",
            inputs=[indices_name, depth_name, on_name, off_name],
            outputs=[onehot_output_name],
            options={"axis": -1},
        )
    )
    if perm_from_last is not None:
        make_transpose(
            ctx=ctx,
            input_name=onehot_output_name,
            output_name=output_name,
            perm_values=perm_from_last,
        )


def build_nonzero_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    rank = len(input_shape)
    if rank <= 0:
        raise NotImplementedError(
            f"NonZero requires rank >= 1 for flatbuffer_direct. op={node.name} shape={input_shape}"
        )

    condition_name = input_name
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    if input_dtype != "BOOL":
        zero_name = ctx.add_const_tensor(
            f"{output_name}_nonzero_zero",
            np.asarray(0, dtype=_DTYPE_TO_NP.get(input_dtype, np.float32)),
        )
        condition_name = ctx.add_intermediate_tensor(
            f"{output_name}_nonzero_condition",
            dtype="BOOL",
            shape=input_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="NOT_EQUAL",
                inputs=[input_name, zero_name],
                outputs=[condition_name],
            )
        )

    where_out_name = ctx.add_intermediate_tensor(
        f"{output_name}_nonzero_where",
        dtype="INT64",
        shape=[-1, rank],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="WHERE",
            inputs=[condition_name],
            outputs=[where_out_name],
        )
    )

    transpose_out_name = output_name
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if output_dtype != "INT64":
        transpose_out_name = ctx.add_intermediate_tensor(
            f"{output_name}_nonzero_i64",
            dtype="INT64",
            shape=[rank, -1],
        )
    transposed_name = make_transpose(
        ctx=ctx,
        input_name=where_out_name,
        output_name=transpose_out_name,
        perm_values=[1, 0],
    )

    out_tensor = ctx.model_ir.tensors[output_name]
    out_tensor.shape = [int(rank), 1]
    out_tensor.shape_signature = [int(rank), -1]
    if transposed_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[transposed_name],
                outputs=[output_name],
                options={
                    "inDataType": "INT64",
                    "outDataType": output_dtype,
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


def build_non_max_suppression_op(node: Any, ctx: Any) -> None:
    boxes_name = node.inputs[0].name
    scores_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(boxes_name)
    ctx.ensure_tensor(scores_name)
    ctx.ensure_tensor(output_name)

    boxes_shape = [int(v) for v in ctx.get_tensor_shape(boxes_name)]
    scores_shape = [int(v) for v in ctx.get_tensor_shape(scores_name)]
    output_nms_with_argmax = bool(getattr(ctx, "output_nms_with_argmax", False))
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if output_dtype not in _DTYPE_TO_NP:
        output_dtype = "INT64"
    if len(boxes_shape) != 3 or len(scores_shape) != 3:
        raise NotImplementedError(
            "NonMaxSuppression currently supports rank-3 boxes/scores only in flatbuffer_direct. "
            f"op={node.name} boxes_shape={boxes_shape} scores_shape={scores_shape}"
        )
    if boxes_shape[0] != 1 or scores_shape[0] != 1:
        raise NotImplementedError(
            "NonMaxSuppression builtin lowering currently supports batch=1 only "
            "in flatbuffer_direct. "
            f"op={node.name} boxes_shape={boxes_shape} scores_shape={scores_shape}"
        )
    if boxes_shape[2] != 4:
        raise NotImplementedError(
            "NonMaxSuppression requires boxes last dimension = 4 in flatbuffer_direct builtin lowering. "
            f"op={node.name} boxes_shape={boxes_shape}"
        )

    num_boxes = int(boxes_shape[1])
    if num_boxes <= 0:
        raise NotImplementedError(
            "NonMaxSuppression requires static positive num_boxes in flatbuffer_direct builtin lowering. "
            f"op={node.name} boxes_shape={boxes_shape}"
        )

    boxes_2d_name = ctx.add_intermediate_tensor(
        f"{output_name}_nms_boxes_2d",
        dtype=str(ctx.get_tensor_dtype(boxes_name)).upper(),
        shape=[num_boxes, 4],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[boxes_name],
            outputs=[boxes_2d_name],
            options={
                "squeezeDims": [0],
            },
        )
    )

    if not output_nms_with_argmax and int(scores_shape[1]) <= 0:
        raise NotImplementedError(
            "NonMaxSuppression requires static positive class dimension when "
            "--output_nms_with_argmax is disabled in flatbuffer_direct builtin lowering. "
            f"op={node.name} scores_shape={scores_shape}"
        )

    scores_for_nms_name = scores_name
    selected_class_ids_name = None
    if output_nms_with_argmax:
        scores_argmax_2d_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_scores_argmax_2d",
            dtype="INT32",
            shape=[1, num_boxes],
        )
        scores_argmax_axis_name = ctx.add_const_tensor(
            f"{output_name}_nms_scores_argmax_axis",
            np.asarray([1], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="ARG_MAX",
                inputs=[scores_name, scores_argmax_axis_name],
                outputs=[scores_argmax_2d_name],
                options={"outputType": "INT32"},
            )
        )
        selected_class_ids_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_scores_argmax",
            dtype="INT32",
            shape=[num_boxes],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[scores_argmax_2d_name],
                outputs=[selected_class_ids_name],
                options={"squeezeDims": [0]},
            )
        )
        scores_reduced_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_scores_reduced",
            dtype=str(ctx.get_tensor_dtype(scores_name)).upper(),
            shape=[1, 1, num_boxes],
        )
        scores_reduce_axis_name = ctx.add_const_tensor(
            f"{output_name}_nms_scores_reduce_axis",
            np.asarray([1], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="REDUCE_MAX",
                inputs=[scores_name, scores_reduce_axis_name],
                outputs=[scores_reduced_name],
                options={"keepDims": True},
            )
        )
        scores_for_nms_name = scores_reduced_name

    scores_1d_name = ctx.add_intermediate_tensor(
        f"{output_name}_nms_scores_1d",
        dtype=str(ctx.get_tensor_dtype(scores_for_nms_name)).upper(),
        shape=[num_boxes],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[scores_for_nms_name],
            outputs=[scores_1d_name],
            options={
                "squeezeDims": [0, 1],
            },
        )
    )

    max_output = num_boxes
    if len(node.inputs) >= 3:
        max_output_arr = ctx.get_constant_array(node.inputs[2].name)
        if max_output_arr is not None:
            max_output_flat = np.asarray(max_output_arr).reshape(-1)
            if max_output_flat.size > 0:
                max_output = int(max_output_flat[0])
    if max_output <= 0:
        max_output = int(num_boxes)
    if max_output <= 0:
        raise NotImplementedError(
            "NonMaxSuppression max_output_boxes_per_class must resolve to > 0 "
            "in flatbuffer_direct builtin lowering. "
            f"op={node.name} max_output={max_output}"
        )
    # Prefer scalar const for NMS scalar inputs to avoid redundant SQUEEZE(const_vec).
    max_output_size_name = ctx.add_const_tensor(
        f"{output_name}_nms_max_output_size",
        np.asarray(max_output, dtype=np.int32),
    )
    max_output_tensor = ctx.model_ir.tensors.get(max_output_size_name, None)
    if max_output_tensor is not None:
        max_output_tensor.shape = []
        max_output_tensor.shape_signature = []

    iou_threshold_value = np.asarray([0.0], dtype=np.float32)
    if len(node.inputs) >= 4:
        iou_threshold_arr = ctx.get_constant_array(node.inputs[3].name)
        if iou_threshold_arr is not None:
            iou_threshold_flat = np.asarray(iou_threshold_arr, dtype=np.float32).reshape(-1)
            if iou_threshold_flat.size > 0:
                iou_threshold_value = np.asarray([float(iou_threshold_flat[0])], dtype=np.float32)
    iou_threshold_name = ctx.add_const_tensor(
        f"{output_name}_nms_iou_threshold",
        np.asarray(float(iou_threshold_value.reshape(-1)[0]), dtype=np.float32),
    )
    iou_threshold_tensor = ctx.model_ir.tensors.get(iou_threshold_name, None)
    if iou_threshold_tensor is not None:
        iou_threshold_tensor.shape = []
        iou_threshold_tensor.shape_signature = []

    score_threshold_value = np.asarray([-np.inf], dtype=np.float32)
    if len(node.inputs) >= 5:
        score_threshold_arr = ctx.get_constant_array(node.inputs[4].name)
        if score_threshold_arr is not None:
            score_threshold_flat = np.asarray(score_threshold_arr, dtype=np.float32).reshape(-1)
            if score_threshold_flat.size > 0:
                score_threshold_value = np.asarray([float(score_threshold_flat[0])], dtype=np.float32)
    score_threshold_name = ctx.add_const_tensor(
        f"{output_name}_nms_score_threshold",
        np.asarray(float(score_threshold_value.reshape(-1)[0]), dtype=np.float32),
    )
    score_threshold_tensor = ctx.model_ir.tensors.get(score_threshold_name, None)
    if score_threshold_tensor is not None:
        score_threshold_tensor.shape = []
        score_threshold_tensor.shape_signature = []

    def _build_single_class_nms_triplets(
        *,
        class_scores_1d_name: str,
        suffix: str,
        class_id_value: int | None = None,
        class_ids_vector_name: str | None = None,
    ) -> str:
        nms_selected_indices_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_selected_indices{suffix}",
            dtype="INT32",
            shape=[max_output],
        )
        nms_valid_count_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_valid_count{suffix}",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="NON_MAX_SUPPRESSION_V4",
                inputs=[
                    boxes_2d_name,
                    class_scores_1d_name,
                    max_output_size_name,
                    iou_threshold_name,
                    score_threshold_name,
                ],
                outputs=[
                    nms_selected_indices_name,
                    nms_valid_count_name,
                ],
            )
        )

        selected_indices_valid_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_selected_indices_valid{suffix}",
            dtype="INT32",
            shape=[-1],
        )
        valid_count_vec_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_valid_count_vec{suffix}",
            dtype="INT32",
            shape=[1],
        )
        valid_count_vec_shape_name = ctx.add_const_tensor(
            f"{output_name}_nms_valid_count_vec_shape{suffix}",
            np.asarray([1], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[nms_valid_count_name, valid_count_vec_shape_name],
                outputs=[valid_count_vec_name],
                options={"newShape": [1]},
            )
        )
        selected_indices_valid_begin_name = ctx.add_const_tensor(
            f"{output_name}_nms_selected_indices_valid_begin{suffix}",
            np.asarray([0], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[
                    nms_selected_indices_name,
                    selected_indices_valid_begin_name,
                    valid_count_vec_name,
                ],
                outputs=[selected_indices_valid_name],
            )
        )

        selected_indices_col_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_selected_indices_col{suffix}",
            dtype="INT32",
            shape=[-1, 1],
        )
        selected_indices_col_shape_name = ctx.add_const_tensor(
            f"{output_name}_nms_selected_indices_col_shape{suffix}",
            np.asarray([-1, 1], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[selected_indices_valid_name, selected_indices_col_shape_name],
                outputs=[selected_indices_col_name],
                options={"newShape": [-1, 1]},
            )
        )

        zero_col_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_zero_col{suffix}",
            dtype="INT32",
            shape=[-1, 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SUB",
                inputs=[selected_indices_col_name, selected_indices_col_name],
                outputs=[zero_col_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

        class_ids_col_name = zero_col_name
        if class_ids_vector_name is not None:
            selected_class_ids_valid_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_selected_class_ids{suffix}",
                dtype="INT32",
                shape=[-1],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="GATHER",
                    inputs=[class_ids_vector_name, selected_indices_valid_name],
                    outputs=[selected_class_ids_valid_name],
                    options={
                        "axis": 0,
                        "batchDims": 0,
                    },
                )
            )
            class_ids_col_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_selected_class_ids_col{suffix}",
                dtype="INT32",
                shape=[-1, 1],
            )
            class_ids_col_shape_name = ctx.add_const_tensor(
                f"{output_name}_nms_selected_class_ids_col_shape{suffix}",
                np.asarray([-1, 1], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[selected_class_ids_valid_name, class_ids_col_shape_name],
                    outputs=[class_ids_col_name],
                    options={"newShape": [-1, 1]},
                )
            )
        elif class_id_value is not None and int(class_id_value) != 0:
            class_id_scalar_name = ctx.add_const_tensor(
                f"{output_name}_nms_class_id{suffix}",
                np.asarray(int(class_id_value), dtype=np.int32),
            )
            class_id_scalar_tensor = ctx.model_ir.tensors.get(class_id_scalar_name, None)
            if class_id_scalar_tensor is not None:
                class_id_scalar_tensor.shape = []
                class_id_scalar_tensor.shape_signature = []
            class_ids_col_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_class_ids_col{suffix}",
                dtype="INT32",
                shape=[-1, 1],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="ADD",
                    inputs=[zero_col_name, class_id_scalar_name],
                    outputs=[class_ids_col_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )

        indices_triplets_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_indices_triplets_i32{suffix}",
            dtype="INT32",
            shape=[-1, 3],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[
                    zero_col_name,
                    class_ids_col_name,
                    selected_indices_col_name,
                ],
                outputs=[indices_triplets_name],
                options={
                    "axis": 1,
                    "fusedActivationFunction": "NONE",
                },
            )
        )
        return indices_triplets_name

    indices_triplets_i32_names: list[str] = []

    if output_nms_with_argmax:
        scores_1d_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_scores_1d",
            dtype=str(ctx.get_tensor_dtype(scores_for_nms_name)).upper(),
            shape=[num_boxes],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[scores_for_nms_name],
                outputs=[scores_1d_name],
                options={
                    "squeezeDims": [0, 1],
                },
            )
        )
        indices_triplets_i32_names.append(
            _build_single_class_nms_triplets(
                class_scores_1d_name=scores_1d_name,
                suffix="_argmax",
                class_ids_vector_name=selected_class_ids_name,
            )
        )
    else:
        num_classes = int(scores_shape[1])
        if num_classes == 1:
            scores_1d_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_scores_1d",
                dtype=str(ctx.get_tensor_dtype(scores_for_nms_name)).upper(),
                shape=[num_boxes],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SQUEEZE",
                    inputs=[scores_for_nms_name],
                    outputs=[scores_1d_name],
                    options={
                        "squeezeDims": [0, 1],
                    },
                )
            )
            indices_triplets_i32_names.append(
                _build_single_class_nms_triplets(
                    class_scores_1d_name=scores_1d_name,
                    suffix="_c0",
                    class_id_value=0,
                )
            )
        else:
            scores_2d_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_scores_2d",
                dtype=str(ctx.get_tensor_dtype(scores_for_nms_name)).upper(),
                shape=[num_classes, num_boxes],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SQUEEZE",
                    inputs=[scores_for_nms_name],
                    outputs=[scores_2d_name],
                    options={
                        "squeezeDims": [0],
                    },
                )
            )
            for class_idx in range(num_classes):
                class_suffix = f"_c{int(class_idx)}"
                class_scores_row_name = ctx.add_intermediate_tensor(
                    f"{output_name}_nms_scores_row{class_suffix}",
                    dtype=str(ctx.get_tensor_dtype(scores_for_nms_name)).upper(),
                    shape=[1, num_boxes],
                )
                class_begin_name = ctx.add_const_tensor(
                    f"{output_name}_nms_scores_begin{class_suffix}",
                    np.asarray([int(class_idx), 0], dtype=np.int32),
                )
                class_size_name = ctx.add_const_tensor(
                    f"{output_name}_nms_scores_size{class_suffix}",
                    np.asarray([1, int(num_boxes)], dtype=np.int32),
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="SLICE",
                        inputs=[scores_2d_name, class_begin_name, class_size_name],
                        outputs=[class_scores_row_name],
                    )
                )
                class_scores_1d_name = ctx.add_intermediate_tensor(
                    f"{output_name}_nms_scores_1d{class_suffix}",
                    dtype=str(ctx.get_tensor_dtype(scores_for_nms_name)).upper(),
                    shape=[num_boxes],
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="SQUEEZE",
                        inputs=[class_scores_row_name],
                        outputs=[class_scores_1d_name],
                        options={
                            "squeezeDims": [0],
                        },
                    )
                )
                indices_triplets_i32_names.append(
                    _build_single_class_nms_triplets(
                        class_scores_1d_name=class_scores_1d_name,
                        suffix=class_suffix,
                        class_id_value=int(class_idx),
                    )
                )

    if len(indices_triplets_i32_names) == 0:
        raise NotImplementedError(
            "NonMaxSuppression lowering failed to build any class-wise NMS outputs. "
            f"op={node.name} scores_shape={scores_shape}"
        )
    if len(indices_triplets_i32_names) == 1:
        indices_triplets_i32_name = indices_triplets_i32_names[0]
    else:
        indices_triplets_i32_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_indices_triplets_i32",
            dtype="INT32",
            shape=[-1, 3],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=indices_triplets_i32_names,
                outputs=[indices_triplets_i32_name],
                options={
                    "axis": 0,
                    "fusedActivationFunction": "NONE",
                },
            )
        )

    if output_dtype == "INT32":
        if indices_triplets_i32_name != output_name:
            output_shape_name = ctx.add_const_tensor(
                f"{output_name}_nms_output_shape",
                np.asarray([-1, 3], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[indices_triplets_i32_name, output_shape_name],
                    outputs=[output_name],
                    options={"newShape": [-1, 3]},
                )
            )
    else:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[indices_triplets_i32_name],
                outputs=[output_name],
                options={
                    "inDataType": "INT32",
                    "outDataType": output_dtype,
                },
            )
        )

    output_tensor = ctx.model_ir.tensors[output_name]
    output_tensor.shape = [1, 3]
    output_tensor.shape_signature = [-1, 3]
