from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import _clone_quantization, make_transpose
from onnx2tf.utils.logging import warn


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


def _prefer_int32_index_output_dtype(
    *,
    ctx: Any,
    tensor_name: str,
    requested_dtype: str,
) -> str:
    dtype = str(requested_dtype).upper()
    tensor = ctx.model_ir.tensors.get(tensor_name, None)
    if tensor is not None:
        tensor.dtype = dtype
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(tensor_name)] = dtype
    return dtype


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


def _maybe_constantize_topk_k(
    *,
    ctx: Any,
    node: Any,
    k_input_name: str,
    values_output_name: str,
) -> str | None:
    k_const = ctx.get_constant_array(k_input_name)
    if k_const is None:
        return None
    k_arr = np.asarray(k_const)
    if int(k_arr.size) != 1:
        raise NotImplementedError(
            "TopK k input must be scalar-like (shape [] or [1]) in flatbuffer_direct. "
            f"op={node.name} k_shape={list(k_arr.shape)}"
        )
    k_value = int(np.asarray(k_arr).reshape(-1)[0])
    if k_value < np.iinfo(np.int32).min or k_value > np.iinfo(np.int32).max:
        raise NotImplementedError(
            "TopK constant k is out of INT32 range required by TFLite TOPK_V2. "
            f"op={node.name} k={k_value}"
        )
    return ctx.add_const_tensor(
        f"{values_output_name}_topk_k_const_i32",
        np.asarray(k_value, dtype=np.int32),
    )


def _resolve_static_topk_k_value(ctx: Any, k_input_name: str) -> int | None:
    k_const = ctx.get_constant_array(k_input_name)
    if k_const is None:
        return None
    k_arr = np.asarray(k_const)
    if int(k_arr.size) != 1:
        raise NotImplementedError(
            "TopK k input must be scalar-like (shape [] or [1]) in flatbuffer_direct. "
            f"k_shape={list(k_arr.shape)}"
        )
    return int(k_arr.reshape(-1)[0])


def _infer_topk_output_shape_and_signature(
    *,
    input_shape: list[int],
    input_signature: list[int],
    axis: int,
    static_k_value: int | None,
) -> tuple[list[int], list[int]]:
    output_shape = [int(v) for v in list(input_shape)]
    output_signature = [int(v) for v in list(input_signature)]
    if int(axis) >= len(output_shape):
        return output_shape, output_signature
    if static_k_value is None:
        output_signature[int(axis)] = -1
        return output_shape, output_signature
    resolved_k = int(static_k_value)
    if int(output_shape[int(axis)]) > 0:
        resolved_k = min(int(output_shape[int(axis)]), resolved_k)
    output_shape[int(axis)] = int(resolved_k) if int(resolved_k) > 0 else 1
    output_signature[int(axis)] = int(resolved_k)
    return output_shape, output_signature


def _tensor_shape_with_signature(ctx: Any, tensor_name: str) -> list[int]:
    shape = [int(v) for v in ctx.get_tensor_shape(tensor_name)]
    tensor = ctx.model_ir.tensors.get(tensor_name, None)
    signature = (
        [int(v) for v in list(tensor.shape_signature)]
        if tensor is not None and tensor.shape_signature is not None
        else [int(v) for v in shape]
    )
    if len(signature) != len(shape):
        return [int(v) for v in shape]
    return [
        int(signature[idx]) if int(signature[idx]) < 0 else int(shape[idx])
        for idx in range(len(shape))
    ]


def _is_unknown_rank_placeholder_tensor(ctx: Any, tensor_name: str) -> bool:
    shape = [int(v) for v in list(ctx.get_tensor_shape(tensor_name))]
    if len(shape) == 0 or not all(int(v) == 1 for v in shape):
        return False
    tensor = ctx.model_ir.tensors.get(str(tensor_name), None)
    if tensor is not None:
        signature = (
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else [int(v) for v in shape]
        )
        if len(signature) != len(shape):
            return False
        if any(int(v) < 0 for v in signature):
            return True
        if len(shape) == 1 and int(signature[0]) == 1:
            raw_shape = None
            if hasattr(ctx, "shape_map"):
                raw_shape = ctx.shape_map.get(str(tensor_name), None)
            if raw_shape is None:
                return True
            if isinstance(raw_shape, (list, tuple)) and len(list(raw_shape)) == 0:
                return True
        return False
    raw_shape = None
    if hasattr(ctx, "shape_map"):
        raw_shape = ctx.shape_map.get(str(tensor_name), None)
    if raw_shape is None:
        return True
    if not isinstance(raw_shape, (list, tuple)):
        return True
    if len(list(raw_shape)) == 0:
        return True
    unresolved = False
    for dim in list(raw_shape):
        if isinstance(dim, (int, np.integer)):
            if int(dim) <= 0:
                unresolved = True
                break
        else:
            unresolved = True
            break
    return bool(unresolved)


def _add_reshape_operator(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    new_shape: list[int],
    preserve_dynamic_shape: bool = False,
) -> None:
    shape_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape",
        np.asarray([int(v) for v in list(new_shape)], dtype=np.int32),
    )
    options: dict[str, Any] = {
        "newShape": [int(v) for v in list(new_shape)],
    }
    if bool(preserve_dynamic_shape):
        options["preserveDynamicShape"] = True
        output_tensor = ctx.model_ir.tensors.get(output_name, None)
        if output_tensor is not None:
            target_signature = [int(v) for v in list(new_shape)]
            output_tensor.shape_signature = [int(v) for v in target_signature]
            output_tensor.shape = [
                int(v) if int(v) >= 0 else 1 for v in target_signature
            ]
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            options=options,
        )
    )


def _add_binary_op(
    *,
    ctx: Any,
    op_type: str,
    lhs_name: str,
    rhs_name: str,
    output_name: str,
) -> None:
    options: dict[str, Any] = {}
    if op_type in {"ADD", "SUB", "MUL", "DIV"}:
        options = {"fusedActivationFunction": "NONE"}
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[lhs_name, rhs_name],
            outputs=[output_name],
            options=options,
        )
    )


def _resolve_positive_axis_dim(
    *,
    input_shape: list[int],
    input_signature: list[int],
    axis: int,
) -> int:
    if 0 <= int(axis) < int(len(input_signature)):
        signature_dim = int(input_signature[int(axis)])
        if signature_dim < 0:
            return -1
        if signature_dim > 0:
            return int(signature_dim)
    if 0 <= int(axis) < int(len(input_shape)):
        static_dim = int(input_shape[int(axis)])
        if static_dim > 0:
            return int(static_dim)
    return -1


def build_gather_op(node: Any, ctx: Any) -> None:
    params_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(params_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(params_name)]
    params_dtype = str(ctx.get_tensor_dtype(params_name)).upper()
    if params_dtype in {"INT64", "UINT64"}:
        runtime_params_dtype = _prefer_int32_index_output_dtype(
            ctx=ctx,
            tensor_name=params_name,
            requested_dtype=params_dtype,
        )
        if runtime_params_dtype != params_dtype:
            casted_params_name = ctx.add_intermediate_tensor(
                f"{output_name}_gather_params_{runtime_params_dtype.lower()}",
                dtype=runtime_params_dtype,
                shape=input_shape,
            )
            casted_params_tensor = ctx.model_ir.tensors.get(casted_params_name, None)
            src_params_tensor = ctx.model_ir.tensors.get(params_name, None)
            if casted_params_tensor is not None and src_params_tensor is not None:
                casted_params_tensor.shape_signature = (
                    [int(v) for v in list(src_params_tensor.shape_signature)]
                    if src_params_tensor.shape_signature is not None
                    else [int(v) for v in input_shape]
                )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[params_name],
                    outputs=[casted_params_name],
                    options={
                        "inDataType": params_dtype,
                        "outDataType": runtime_params_dtype,
                    },
                )
            )
            params_name = casted_params_name
            params_dtype = runtime_params_dtype
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
    output_tensor.dtype = params_dtype
    output_tensor.quantization = _clone_quantization(
        params_tensor.quantization if params_tensor is not None else None
    )
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(output_name)] = params_dtype
    existing_output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_tensor.shape)]
    )
    expected_output_rank = int(len(existing_output_signature))
    scalar_indices_semantics = False
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
            axis_dim = _resolve_positive_axis_dim(
                input_shape=input_shape,
                input_signature=input_signature,
                axis=axis,
            )
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
        if scalarize_single_index and int(input_rank) == 1:
            # The IR represents rank-0 tensors as shape [1]. For rank-1 params,
            # scalarizing indices would produce runtime scalar Gather output and
            # can break downstream rank-1 CONCAT size assembly chains.
            scalarize_single_index = False
        if scalarize_single_index:
            # Keep scalar indices as rank-1 [1] for broader runtime compatibility.
            scalar_value = np.asarray(
                [indices_const_arr.reshape(-1)[0]],
                dtype=indices_const_arr.dtype,
            )
            scalarized_indices_name = ctx.add_const_tensor(
                f"{output_name}_gather_indices_scalar_1d",
                scalar_value,
            )
            indices_shape = [1]
            indices_signature = [1]
            scalar_indices_semantics = True
    elif input_rank > 1 and expected_output_rank == input_rank - 1:
        # Runtime scalar indices are normalized to shape [1] in IR metadata.
        # Detect scalar-output gather semantics from output rank contract.
        if len(indices_signature) == 1 and int(indices_signature[0]) == 1:
            scalar_indices_semantics = True

    infer_indices_shape = (
        []
        if bool(scalar_indices_semantics) and input_rank > 1
        else [int(v) for v in indices_shape]
    )
    infer_indices_signature = (
        []
        if bool(scalar_indices_semantics) and input_rank > 1
        else [int(v) for v in indices_signature]
    )

    inferred_output_shape = (
        [int(v) for v in input_shape[:int(axis)]]
        + [int(v) for v in infer_indices_shape]
        + [int(v) for v in input_shape[int(axis) + 1:]]
    )
    inferred_output_signature = (
        [int(v) for v in input_signature[:int(axis)]]
        + [int(v) for v in infer_indices_signature]
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

    gather_indices_name = scalarized_indices_name
    gather_indices_tensor = ctx.model_ir.tensors.get(gather_indices_name, None)
    gather_indices_shape = [int(v) for v in ctx.get_tensor_shape(gather_indices_name)]
    gather_indices_signature = (
        [int(v) for v in list(gather_indices_tensor.shape_signature)]
        if gather_indices_tensor is not None and gather_indices_tensor.shape_signature is not None
        else [int(v) for v in gather_indices_shape]
    )
    if bool(scalar_indices_semantics) and len(gather_indices_signature) == 0:
        scalar_indices_1d_name = ctx.add_intermediate_tensor(
            f"{output_name}_gather_indices_scalar_1d",
            dtype=str(ctx.get_tensor_dtype(gather_indices_name)).upper(),
            shape=[1],
        )
        scalar_indices_1d_tensor = ctx.model_ir.tensors.get(scalar_indices_1d_name, None)
        if scalar_indices_1d_tensor is not None:
            scalar_indices_1d_tensor.shape_signature = [1]
            scalar_indices_1d_tensor.shape = [1]
        scalar_indices_1d_shape_name = ctx.add_const_tensor(
            f"{output_name}_gather_indices_scalar_1d_shape",
            np.asarray([1], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[gather_indices_name, scalar_indices_1d_shape_name],
                outputs=[scalar_indices_1d_name],
                options={
                    "newShape": [1],
                    "preserveDynamicShape": True,
                },
            )
        )
        gather_indices_name = scalar_indices_1d_name
        gather_indices_shape = [1]
        gather_indices_signature = [1]
    gather_indices_dtype = str(ctx.get_tensor_dtype(gather_indices_name)).upper()
    if gather_indices_dtype != "INT32":
        gather_indices_const = ctx.get_constant_array(gather_indices_name)
        if gather_indices_const is not None:
            gather_indices_arr = np.asarray(gather_indices_const)
            if int(gather_indices_arr.size) > 0:
                gather_indices_i64 = gather_indices_arr.astype(np.int64, copy=False)
                i32_info = np.iinfo(np.int32)
                if bool(np.any(gather_indices_i64 < int(i32_info.min))) or bool(
                    np.any(gather_indices_i64 > int(i32_info.max))
                ):
                    raise NotImplementedError(
                        "Gather constant indices are out of INT32 range required by TFLite GATHER. "
                        f"op={node.name} dtype={gather_indices_dtype}"
                    )
            gather_indices_name = ctx.add_const_tensor(
                f"{output_name}_gather_indices_i32_const",
                gather_indices_arr.astype(np.int32, copy=False),
            )
            gather_indices_i32_const_tensor = ctx.model_ir.tensors.get(gather_indices_name, None)
            if gather_indices_i32_const_tensor is not None:
                gather_indices_i32_const_tensor.shape_signature = [int(v) for v in gather_indices_signature]
                gather_indices_i32_const_tensor.shape = [
                    int(v) if int(v) >= 0 else 1 for v in gather_indices_signature
                ]
        else:
            gather_indices_i32_name = ctx.add_intermediate_tensor(
                f"{output_name}_gather_indices_i32",
                dtype="INT32",
                shape=gather_indices_shape,
            )
            gather_indices_i32_tensor = ctx.model_ir.tensors.get(gather_indices_i32_name, None)
            if gather_indices_i32_tensor is not None:
                gather_indices_i32_tensor.shape_signature = [int(v) for v in gather_indices_signature]
                gather_indices_i32_tensor.shape = [
                    int(v) if int(v) >= 0 else 1 for v in gather_indices_signature
                ]
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[gather_indices_name],
                    outputs=[gather_indices_i32_name],
                    options={
                        "inDataType": gather_indices_dtype,
                        "outDataType": "INT32",
                    },
                )
            )
            gather_indices_name = gather_indices_i32_name

    gather_indices_const = ctx.get_constant_array(gather_indices_name)
    if gather_indices_const is None:
        axis_dim = _resolve_positive_axis_dim(
            input_shape=input_shape,
            input_signature=input_signature,
            axis=axis,
        )
        if axis_dim > 0:
            runtime_indices_shape = [int(v) for v in gather_indices_shape]
            runtime_indices_signature = [int(v) for v in gather_indices_signature]
            if len(runtime_indices_shape) == 0:
                runtime_indices_shape = [1]
            if len(runtime_indices_signature) == 0:
                runtime_indices_signature = [1]

            gather_axis_dim_name = ctx.add_const_tensor(
                f"{output_name}_gather_axis_dim_i32",
                np.asarray([int(axis_dim)], dtype=np.int32),
            )
            gather_indices_zero_name = ctx.add_const_tensor(
                f"{output_name}_gather_indices_zero_i32",
                np.asarray([0], dtype=np.int32),
            )
            gather_indices_is_negative_name = ctx.add_intermediate_tensor(
                f"{output_name}_gather_indices_is_negative",
                dtype="BOOL",
                shape=runtime_indices_shape,
            )
            gather_indices_is_negative_tensor = ctx.model_ir.tensors.get(
                gather_indices_is_negative_name, None
            )
            if gather_indices_is_negative_tensor is not None:
                gather_indices_is_negative_tensor.shape_signature = [
                    int(v) for v in runtime_indices_signature
                ]
                gather_indices_is_negative_tensor.shape = [
                    int(v) if int(v) >= 0 else 1 for v in runtime_indices_signature
                ]
            ctx.add_operator(
                OperatorIR(
                    op_type="LESS",
                    inputs=[gather_indices_name, gather_indices_zero_name],
                    outputs=[gather_indices_is_negative_name],
                    options={},
                )
            )

            gather_indices_normalized_name = ctx.add_intermediate_tensor(
                f"{output_name}_gather_indices_normalized",
                dtype="INT32",
                shape=runtime_indices_shape,
            )
            gather_indices_normalized_tensor = ctx.model_ir.tensors.get(
                gather_indices_normalized_name, None
            )
            if gather_indices_normalized_tensor is not None:
                gather_indices_normalized_tensor.shape_signature = [
                    int(v) for v in runtime_indices_signature
                ]
                gather_indices_normalized_tensor.shape = [
                    int(v) if int(v) >= 0 else 1 for v in runtime_indices_signature
                ]
            if bool(getattr(ctx, "optimization_for_gpu_delegate", False)):
                one_name = ctx.add_const_tensor(
                    f"{output_name}_gather_indices_one_i32",
                    np.asarray([1], dtype=np.int32),
                )
                zero_mask_name = ctx.add_const_tensor(
                    f"{output_name}_gather_indices_zero_mask_i32",
                    np.asarray([0], dtype=np.int32),
                )
                gather_indices_negative_mask_name = ctx.add_intermediate_tensor(
                    f"{output_name}_gather_indices_negative_mask",
                    dtype="INT32",
                    shape=runtime_indices_shape,
                )
                gather_indices_negative_mask_tensor = ctx.model_ir.tensors.get(
                    gather_indices_negative_mask_name, None
                )
                if gather_indices_negative_mask_tensor is not None:
                    gather_indices_negative_mask_tensor.shape_signature = [
                        int(v) for v in runtime_indices_signature
                    ]
                    gather_indices_negative_mask_tensor.shape = [
                        int(v) if int(v) >= 0 else 1 for v in runtime_indices_signature
                    ]
                ctx.add_operator(
                    OperatorIR(
                        op_type="SELECT",
                        inputs=[
                            gather_indices_is_negative_name,
                            one_name,
                            zero_mask_name,
                        ],
                        outputs=[gather_indices_negative_mask_name],
                        options={},
                    )
                )
                gather_indices_offset_name = ctx.add_intermediate_tensor(
                    f"{output_name}_gather_indices_offset",
                    dtype="INT32",
                    shape=runtime_indices_shape,
                )
                gather_indices_offset_tensor = ctx.model_ir.tensors.get(
                    gather_indices_offset_name, None
                )
                if gather_indices_offset_tensor is not None:
                    gather_indices_offset_tensor.shape_signature = [
                        int(v) for v in runtime_indices_signature
                    ]
                    gather_indices_offset_tensor.shape = [
                        int(v) if int(v) >= 0 else 1 for v in runtime_indices_signature
                    ]
                _add_binary_op(
                    ctx=ctx,
                    op_type="MUL",
                    lhs_name=gather_indices_negative_mask_name,
                    rhs_name=gather_axis_dim_name,
                    output_name=gather_indices_offset_name,
                )
                _add_binary_op(
                    ctx=ctx,
                    op_type="ADD",
                    lhs_name=gather_indices_name,
                    rhs_name=gather_indices_offset_name,
                    output_name=gather_indices_normalized_name,
                )
            else:
                gather_indices_wrapped_runtime_name = ctx.add_intermediate_tensor(
                    f"{output_name}_gather_indices_wrapped_runtime",
                    dtype="INT32",
                    shape=runtime_indices_shape,
                )
                gather_indices_wrapped_runtime_tensor = ctx.model_ir.tensors.get(
                    gather_indices_wrapped_runtime_name, None
                )
                if gather_indices_wrapped_runtime_tensor is not None:
                    gather_indices_wrapped_runtime_tensor.shape_signature = [
                        int(v) for v in runtime_indices_signature
                    ]
                    gather_indices_wrapped_runtime_tensor.shape = [
                        int(v) if int(v) >= 0 else 1 for v in runtime_indices_signature
                    ]
                _add_binary_op(
                    ctx=ctx,
                    op_type="ADD",
                    lhs_name=gather_indices_name,
                    rhs_name=gather_axis_dim_name,
                    output_name=gather_indices_wrapped_runtime_name,
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="SELECT",
                        inputs=[
                            gather_indices_is_negative_name,
                            gather_indices_wrapped_runtime_name,
                            gather_indices_name,
                        ],
                        outputs=[gather_indices_normalized_name],
                        options={},
                    )
                )
            gather_indices_name = gather_indices_normalized_name
            gather_indices_shape = [int(v) for v in runtime_indices_shape]
            gather_indices_signature = [int(v) for v in runtime_indices_signature]

    gather_output_name = output_name
    gather_output_signature: list[int] = []
    if bool(scalar_indices_semantics) and input_rank > 1:
        gather_output_signature = (
            [int(v) for v in input_signature[:int(axis)]]
            + [1]
            + [int(v) for v in input_signature[int(axis) + 1:]]
        )
        gather_output_shape = [int(v) if int(v) >= 0 else 1 for v in gather_output_signature]
        gather_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_gather_scalar_1d",
            dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
            shape=gather_output_shape,
        )
        gather_output_tensor = ctx.model_ir.tensors.get(gather_output_name, None)
        if gather_output_tensor is not None:
            gather_output_tensor.shape_signature = [int(v) for v in gather_output_signature]
            gather_output_tensor.shape = [int(v) for v in gather_output_shape]

    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[params_name, gather_indices_name],
            outputs=[gather_output_name],
            options={
                "axis": int(axis),
                "batchDims": int(batch_dims),
            },
        )
    )

    if gather_output_name != output_name:
        if len(gather_output_signature) == 0:
            gather_output_tensor = ctx.model_ir.tensors.get(gather_output_name, None)
            if gather_output_tensor is not None:
                gather_output_signature = (
                    [int(v) for v in list(gather_output_tensor.shape_signature)]
                    if gather_output_tensor.shape_signature is not None
                    else [int(v) for v in list(gather_output_tensor.shape)]
                )
            if len(gather_output_signature) == 0:
                gather_output_signature = [1]
        reshape_target_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_tensor.shape)]
        )
        if _is_unknown_rank_placeholder_tensor(ctx, output_name):
            reshape_target_signature = (
                [int(v) for v in gather_output_signature[:int(axis)]]
                + [int(v) for v in gather_output_signature[int(axis) + 1:]]
            )
            if len(reshape_target_signature) == 0:
                reshape_target_signature = [1]
            output_tensor.shape_signature = [int(v) for v in reshape_target_signature]
            output_tensor.shape = [
                int(v) if int(v) >= 0 else 1 for v in reshape_target_signature
            ]
        reshape_const_shape = [int(v) for v in reshape_target_signature]
        reshape_options_shape = (
            []
            if any(int(v) < 0 for v in reshape_const_shape)
            else [int(v) for v in reshape_const_shape]
        )
        gather_out_shape_const = ctx.add_const_tensor(
            f"{output_name}_gather_scalar_reshape_shape",
            np.asarray(reshape_const_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[gather_output_name, gather_out_shape_const],
                outputs=[output_name],
                options={
                    "newShape": reshape_options_shape,
                    "preserveDynamicShape": True,
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
    if batch_dims < 0:
        raise NotImplementedError(
            f"GatherND batch_dims must be >= 0 in flatbuffer_direct. "
            f"op={node.name} batch_dims={batch_dims}"
        )

    params_shape = [int(v) for v in ctx.get_tensor_shape(params_name)]
    params_dtype = str(ctx.get_tensor_dtype(params_name)).upper()
    if params_dtype in {"INT64", "UINT64"}:
        runtime_params_dtype = _prefer_int32_index_output_dtype(
            ctx=ctx,
            tensor_name=params_name,
            requested_dtype=params_dtype,
        )
        if runtime_params_dtype != params_dtype:
            casted_params_name = ctx.add_intermediate_tensor(
                f"{output_name}_gather_nd_params_{runtime_params_dtype.lower()}",
                dtype=runtime_params_dtype,
                shape=params_shape,
            )
            casted_params_tensor = ctx.model_ir.tensors.get(casted_params_name, None)
            src_params_tensor = ctx.model_ir.tensors.get(params_name, None)
            if casted_params_tensor is not None and src_params_tensor is not None:
                casted_params_tensor.shape_signature = (
                    [int(v) for v in list(src_params_tensor.shape_signature)]
                    if src_params_tensor.shape_signature is not None
                    else [int(v) for v in params_shape]
                )
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[params_name],
                    outputs=[casted_params_name],
                    options={
                        "inDataType": params_dtype,
                        "outDataType": runtime_params_dtype,
                    },
                )
            )
            params_name = casted_params_name
            params_dtype = runtime_params_dtype
            params_shape = [int(v) for v in ctx.get_tensor_shape(params_name)]

    params_tensor = ctx.model_ir.tensors.get(params_name, None)
    params_signature = (
        [int(v) for v in list(params_tensor.shape_signature)]
        if params_tensor is not None and params_tensor.shape_signature is not None
        else [int(v) for v in params_shape]
    )
    indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
    indices_tensor = ctx.model_ir.tensors.get(indices_name, None)
    indices_signature = (
        [int(v) for v in list(indices_tensor.shape_signature)]
        if indices_tensor is not None and indices_tensor.shape_signature is not None
        else [int(v) for v in indices_shape]
    )
    indices_runtime_signature = [int(v) for v in indices_signature]
    # Non-constant GatherND indices can remain data-dependent at runtime
    # even when current static metadata is concrete. Keep gather-prefix dims
    # dynamic in shape_signature to avoid hard-coding downstream reshape shapes.
    if indices_tensor is None or indices_tensor.data is None:
        for dim_idx in range(max(len(indices_runtime_signature) - 1, 0)):
            if int(indices_runtime_signature[dim_idx]) > 0:
                indices_runtime_signature[dim_idx] = -1

    indices_for_gather_nd = indices_name
    indices_dtype = str(ctx.get_tensor_dtype(indices_name)).upper()
    if indices_dtype != "INT32":
        indices_for_gather_nd = ctx.add_intermediate_tensor(
            f"{output_name}_gather_nd_indices_i32",
            dtype="INT32",
            shape=indices_shape,
        )
        cast_output_tensor = ctx.model_ir.tensors.get(indices_for_gather_nd, None)
        if cast_output_tensor is not None:
            cast_output_tensor.shape_signature = [int(v) for v in indices_runtime_signature]
            cast_output_tensor.shape = [
                int(v) if int(v) >= 0 else 1 for v in indices_runtime_signature
            ]
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

    output_tensor = ctx.model_ir.tensors[output_name]
    output_tensor.dtype = params_dtype
    output_tensor.quantization = _clone_quantization(
        params_tensor.quantization if params_tensor is not None else None
    )
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(output_name)] = params_dtype
    inferred_output_signature: Optional[list[int]] = None
    if len(indices_runtime_signature) >= 1:
        gather_dims = int(indices_shape[-1]) if len(indices_shape) > 0 else -1
        if len(indices_runtime_signature) > 0 and int(indices_runtime_signature[-1]) > 0:
            gather_dims = int(indices_runtime_signature[-1])
        max_gather_dims = int(len(params_signature) - int(batch_dims))
        if gather_dims > 0 and gather_dims <= max_gather_dims:
            inferred_output_signature = (
                [int(v) for v in indices_runtime_signature[:-1]]
                + [int(v) for v in params_signature[int(batch_dims) + gather_dims:]]
            )
    if inferred_output_signature is None:
        inferred_output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_tensor.shape)]
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
            for existing_dim, inferred_dim in zip(
                existing_output_signature,
                inferred_output_signature,
            )
        ]
    output_tensor.shape_signature = [int(v) for v in final_output_signature]
    output_tensor.shape = [
        int(v) if int(v) >= 0 else 1 for v in final_output_signature
    ]

    if int(batch_dims) == 0:
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER_ND",
                inputs=[params_name, indices_for_gather_nd],
                outputs=[output_name],
            )
        )
        return

    # Lower GatherND(batch_dims>0) into builtin GatherND(batch_dims=0):
    # 1) flatten batch prefix into one axis
    # 2) prepend generated batch-id to each index vector
    # 3) gather_nd on reshaped params/indices
    # 4) reshape gathered result back to ONNX output shape
    if len(params_shape) <= int(batch_dims) or len(indices_shape) <= int(batch_dims):
        raise NotImplementedError(
            f"GatherND batch_dims is out of range for input ranks in flatbuffer_direct. "
            f"op={node.name} batch_dims={batch_dims} params_shape={params_shape} indices_shape={indices_shape}"
        )

    params_batch_shape = [int(v) for v in params_shape[: int(batch_dims)]]
    indices_batch_shape = [int(v) for v in indices_shape[: int(batch_dims)]]
    if params_batch_shape != indices_batch_shape:
        raise NotImplementedError(
            "GatherND requires params/indices batch prefix match for batch_dims>0 in flatbuffer_direct. "
            f"op={node.name} batch_dims={batch_dims} "
            f"params_batch_shape={params_batch_shape} indices_batch_shape={indices_batch_shape}"
        )
    if any(int(v) <= 0 for v in params_batch_shape):
        raise NotImplementedError(
            "GatherND batch_dims>0 currently requires static positive batch prefix dimensions "
            "in flatbuffer_direct. "
            f"op={node.name} params_batch_shape={params_batch_shape}"
        )
    if any(int(v) < 0 for v in indices_shape[int(batch_dims) : -1]):
        raise NotImplementedError(
            "GatherND batch_dims>0 currently requires static non-negative non-batch index dimensions "
            "in flatbuffer_direct. "
            f"op={node.name} indices_shape={indices_shape}"
        )

    gather_dims = int(indices_shape[-1]) if len(indices_shape) > 0 else -1
    if gather_dims <= 0:
        raise NotImplementedError(
            "GatherND requires static positive indices last dimension in flatbuffer_direct. "
            f"op={node.name} indices_shape={indices_shape}"
        )
    if int(gather_dims) > int(len(params_shape) - int(batch_dims)):
        raise NotImplementedError(
            "GatherND indices last dimension exceeds params rank after batch_dims in flatbuffer_direct. "
            f"op={node.name} batch_dims={batch_dims} indices_last_dim={gather_dims} params_rank={len(params_shape)}"
        )

    batch_count = int(np.prod(np.asarray(params_batch_shape, dtype=np.int64), dtype=np.int64))
    if batch_count < 0 or batch_count > int(np.iinfo(np.int32).max):
        raise NotImplementedError(
            "GatherND flattened batch size is out of supported range in flatbuffer_direct. "
            f"op={node.name} flattened_batch={batch_count}"
        )

    params_tail_shape = [int(v) for v in params_shape[int(batch_dims) :]]
    indices_inner_shape = [int(v) for v in indices_shape[int(batch_dims) : -1]]

    params_flat_shape = [int(batch_count)] + [int(v) for v in params_tail_shape]
    params_flat_shape_name = ctx.add_const_tensor(
        f"{output_name}_gather_nd_params_flat_shape",
        np.asarray(params_flat_shape, dtype=np.int32),
    )
    params_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_nd_params_flat",
        dtype=params_dtype,
        shape=[int(v) if int(v) > 0 else 1 for v in params_flat_shape],
    )
    params_flat_tensor = ctx.model_ir.tensors.get(params_flat_name, None)
    if params_flat_tensor is not None:
        params_flat_tensor.shape_signature = [int(v) for v in params_flat_shape]
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[params_name, params_flat_shape_name],
            outputs=[params_flat_name],
            options={"newShape": [int(v) for v in params_flat_shape]},
        )
    )

    indices_flat_shape = [int(batch_count)] + [int(v) for v in indices_inner_shape] + [int(gather_dims)]
    indices_flat_shape_name = ctx.add_const_tensor(
        f"{output_name}_gather_nd_indices_flat_shape",
        np.asarray(indices_flat_shape, dtype=np.int32),
    )
    indices_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_nd_indices_flat",
        dtype="INT32",
        shape=[int(v) if int(v) > 0 else 1 for v in indices_flat_shape],
    )
    indices_flat_tensor = ctx.model_ir.tensors.get(indices_flat_name, None)
    if indices_flat_tensor is not None:
        indices_flat_tensor.shape_signature = (
            [int(v) for v in indices_runtime_signature]
            if len(indices_runtime_signature) == len(indices_flat_shape)
            else [int(v) for v in indices_flat_shape]
        )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[indices_for_gather_nd, indices_flat_shape_name],
            outputs=[indices_flat_name],
            options={"newShape": [int(v) for v in indices_flat_shape]},
        )
    )

    batch_range_start_name = ctx.add_const_tensor(
        f"{output_name}_gather_nd_batch_range_start",
        np.asarray(0, dtype=np.int32),
    )
    batch_range_limit_name = ctx.add_const_tensor(
        f"{output_name}_gather_nd_batch_range_limit",
        np.asarray(int(batch_count), dtype=np.int32),
    )
    batch_range_delta_name = ctx.add_const_tensor(
        f"{output_name}_gather_nd_batch_range_delta",
        np.asarray(1, dtype=np.int32),
    )
    for scalar_name in [
        batch_range_start_name,
        batch_range_limit_name,
        batch_range_delta_name,
    ]:
        scalar_tensor = ctx.model_ir.tensors.get(scalar_name, None)
        if scalar_tensor is not None:
            scalar_tensor.shape = []
            scalar_tensor.shape_signature = []

    batch_ids_1d_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_nd_batch_ids_1d",
        dtype="INT32",
        shape=[int(batch_count) if int(batch_count) > 0 else 1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RANGE",
            inputs=[
                batch_range_start_name,
                batch_range_limit_name,
                batch_range_delta_name,
            ],
            outputs=[batch_ids_1d_name],
        )
    )

    batch_ids_base_shape = [int(batch_count)] + [1 for _ in indices_inner_shape] + [1]
    batch_ids_base_shape_name = ctx.add_const_tensor(
        f"{output_name}_gather_nd_batch_ids_base_shape",
        np.asarray(batch_ids_base_shape, dtype=np.int32),
    )
    batch_ids_base_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_nd_batch_ids_base",
        dtype="INT32",
        shape=[int(v) if int(v) > 0 else 1 for v in batch_ids_base_shape],
    )
    batch_ids_base_tensor = ctx.model_ir.tensors.get(batch_ids_base_name, None)
    if batch_ids_base_tensor is not None:
        batch_ids_base_tensor.shape_signature = [int(v) for v in batch_ids_base_shape]
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[batch_ids_1d_name, batch_ids_base_shape_name],
            outputs=[batch_ids_base_name],
            options={"newShape": [int(v) for v in batch_ids_base_shape]},
        )
    )

    batch_ids_tile_multiples = [1] + [int(v) for v in indices_inner_shape] + [1]
    batch_ids_tile_multiples_name = ctx.add_const_tensor(
        f"{output_name}_gather_nd_batch_ids_tile_multiples",
        np.asarray(batch_ids_tile_multiples, dtype=np.int32),
    )
    batch_ids_tiled_shape = [int(batch_count)] + [int(v) for v in indices_inner_shape] + [1]
    batch_ids_tiled_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_nd_batch_ids_tiled",
        dtype="INT32",
        shape=[int(v) if int(v) > 0 else 1 for v in batch_ids_tiled_shape],
    )
    batch_ids_tiled_tensor = ctx.model_ir.tensors.get(batch_ids_tiled_name, None)
    if batch_ids_tiled_tensor is not None:
        batch_ids_tiled_tensor.shape_signature = [int(v) for v in batch_ids_tiled_shape]
    ctx.add_operator(
        OperatorIR(
            op_type="TILE",
            inputs=[batch_ids_base_name, batch_ids_tile_multiples_name],
            outputs=[batch_ids_tiled_name],
        )
    )

    indices_with_batch_shape = [int(batch_count)] + [int(v) for v in indices_inner_shape] + [int(gather_dims) + 1]
    indices_with_batch_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_nd_indices_with_batch",
        dtype="INT32",
        shape=[int(v) if int(v) > 0 else 1 for v in indices_with_batch_shape],
    )
    indices_with_batch_tensor = ctx.model_ir.tensors.get(indices_with_batch_name, None)
    if indices_with_batch_tensor is not None:
        indices_with_batch_tensor.shape_signature = [int(v) for v in indices_with_batch_shape]
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[batch_ids_tiled_name, indices_flat_name],
            outputs=[indices_with_batch_name],
            options={
                "axis": int(len(indices_flat_shape) - 1),
                "fusedActivationFunction": "NONE",
            },
        )
    )

    gather_flat_shape = (
        [int(batch_count)]
        + [int(v) for v in indices_inner_shape]
        + [int(v) for v in params_tail_shape[int(gather_dims) :]]
    )
    gather_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_nd_flat_out",
        dtype=params_dtype,
        shape=[int(v) if int(v) > 0 else 1 for v in gather_flat_shape],
    )
    gather_flat_tensor = ctx.model_ir.tensors.get(gather_flat_name, None)
    if gather_flat_tensor is not None:
        gather_flat_tensor.shape_signature = [int(v) for v in gather_flat_shape]
        gather_flat_tensor.quantization = _clone_quantization(output_tensor.quantization)
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER_ND",
            inputs=[params_flat_name, indices_with_batch_name],
            outputs=[gather_flat_name],
        )
    )

    output_shape_static = [int(v) for v in params_batch_shape + indices_inner_shape + params_tail_shape[int(gather_dims) :]]
    output_shape_name = ctx.add_const_tensor(
        f"{output_name}_gather_nd_output_shape",
        np.asarray(output_shape_static, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[gather_flat_name, output_shape_name],
            outputs=[output_name],
            options={"newShape": [int(v) for v in output_shape_static]},
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
    data_meta_shape = _tensor_shape_with_signature(ctx, data_name)
    output_tensor = ctx.model_ir.tensors[output_name]
    data_tensor = ctx.model_ir.tensors[data_name]
    output_tensor.dtype = data_dtype
    output_tensor.quantization = _clone_quantization(data_tensor.quantization)
    _propagate_shape(ctx, data_name, output_name)

    updates_meta_shape = _tensor_shape_with_signature(ctx, updates_name)
    updates_for_scatter = updates_name
    updates_dtype = str(ctx.get_tensor_dtype(updates_name)).upper()
    if updates_dtype != data_dtype:
        updates_const_arr = ctx.get_constant_array(updates_name)
        if updates_const_arr is not None:
            updates_for_scatter = ctx.add_const_tensor(
                f"{output_name}_scatter_nd_updates_cast",
                np.asarray(updates_const_arr, dtype=_DTYPE_TO_NP[data_dtype]),
            )
        else:
            updates_for_scatter = ctx.add_intermediate_tensor(
                f"{output_name}_scatter_nd_updates_cast",
                dtype=data_dtype,
                shape=[int(v) if int(v) >= 0 else 1 for v in updates_meta_shape],
            )
            updates_cast_tensor = ctx.model_ir.tensors.get(updates_for_scatter, None)
            if updates_cast_tensor is not None:
                updates_cast_tensor.shape_signature = [int(v) for v in updates_meta_shape]
                updates_cast_tensor.shape = [
                    int(v) if int(v) >= 0 else 1 for v in updates_meta_shape
                ]
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[updates_name],
                    outputs=[updates_for_scatter],
                    options={
                        "inDataType": updates_dtype,
                        "outDataType": data_dtype,
                    },
                )
            )

    indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
    indices_meta_shape = _tensor_shape_with_signature(ctx, indices_name)

    indices_for_scatter = indices_name
    indices_dtype = str(ctx.get_tensor_dtype(indices_name)).upper()
    if indices_dtype != "INT32":
        indices_const_arr = ctx.get_constant_array(indices_name)
        if indices_const_arr is not None:
            indices_for_scatter = ctx.add_const_tensor(
                f"{output_name}_scatter_nd_indices_i32",
                np.asarray(indices_const_arr, dtype=np.int32),
            )
        else:
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
    if rank > 0 and all(int(dim) > 0 for dim in data_meta_shape):
        shape_for_scatter = ctx.add_const_tensor(
            f"{output_name}_scatter_nd_shape",
            np.asarray([int(v) for v in data_meta_shape], dtype=np.int32),
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

    # ONNX ScatterND accepts negative indices (wrap-around by each indexed axis).
    # TFLite SCATTER_ND rejects them, so normalize before scattering.
    k_dim = int(indices_meta_shape[-1]) if len(indices_meta_shape) > 0 else -1
    normalize_negative_indices = bool(node.attrs.get("normalize_negative_indices", True))
    if (
        bool(normalize_negative_indices)
        and int(k_dim) > 0
        and int(rank) > 0
        and int(k_dim) <= int(rank)
    ):
        shape_prefix_name = ""
        static_shape_prefix: np.ndarray | None = None
        if all(int(v) > 0 for v in data_meta_shape[:k_dim]):
            static_shape_prefix = np.asarray(data_meta_shape[:k_dim], dtype=np.int32)

        indices_const = ctx.get_constant_array(indices_for_scatter)
        normalized_const_name = ""
        if (
            indices_const is not None
            and static_shape_prefix is not None
        ):
            indices_arr = np.asarray(indices_const)
            if (
                indices_arr.ndim >= 1
                and int(indices_arr.shape[-1]) == int(k_dim)
                and np.any(indices_arr < 0)
            ):
                indices_i64 = indices_arr.astype(np.int64, copy=False)
                reshape_dims = [1] * (indices_i64.ndim - 1) + [int(k_dim)]
                mod_base = static_shape_prefix.astype(np.int64, copy=False).reshape(reshape_dims)
                normalized_const = np.mod(indices_i64 + mod_base, mod_base).astype(np.int32, copy=False)
                normalized_const_name = ctx.add_const_tensor(
                    f"{output_name}_scatter_nd_indices_i32_normalized",
                    normalized_const,
                )
                indices_for_scatter = normalized_const_name

        if not normalized_const_name:
            if static_shape_prefix is not None:
                shape_prefix_name = ctx.add_const_tensor(
                    f"{output_name}_scatter_nd_shape_prefix",
                    static_shape_prefix,
                )
            elif int(k_dim) == int(rank):
                shape_prefix_name = shape_for_scatter
            else:
                shape_prefix_indices_name = ctx.add_const_tensor(
                    f"{output_name}_scatter_nd_shape_prefix_indices",
                    np.arange(int(k_dim), dtype=np.int32),
                )
                shape_prefix_name = ctx.add_intermediate_tensor(
                    f"{output_name}_scatter_nd_shape_prefix",
                    dtype="INT32",
                    shape=[int(k_dim)],
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="GATHER",
                        inputs=[shape_for_scatter, shape_prefix_indices_name],
                        outputs=[shape_prefix_name],
                        options={"axis": 0, "batchDims": 0},
                    )
                )

            needs_runtime_normalize = indices_const is None
            if needs_runtime_normalize:
                zero_i32_name = ctx.add_const_tensor(
                    f"{output_name}_scatter_nd_zero_i32",
                    np.asarray(0, dtype=np.int32),
                )
                negative_mask_name = ctx.add_intermediate_tensor(
                    f"{output_name}_scatter_nd_negative_mask",
                    dtype="BOOL",
                    shape=indices_meta_shape,
                )
                indices_plus_shape_name = ctx.add_intermediate_tensor(
                    f"{output_name}_scatter_nd_indices_plus_shape",
                    dtype="INT32",
                    shape=indices_meta_shape,
                )
                indices_wrapped_name = ctx.add_intermediate_tensor(
                    f"{output_name}_scatter_nd_indices_wrapped",
                    dtype="INT32",
                    shape=indices_meta_shape,
                )
                normalized_indices_name = ctx.add_intermediate_tensor(
                    f"{output_name}_scatter_nd_indices_normalized",
                    dtype="INT32",
                    shape=indices_meta_shape,
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="LESS",
                        inputs=[indices_for_scatter, zero_i32_name],
                        outputs=[negative_mask_name],
                    )
                )
                _add_binary_op(
                    ctx=ctx,
                    op_type="ADD",
                    lhs_name=indices_for_scatter,
                    rhs_name=shape_prefix_name,
                    output_name=indices_plus_shape_name,
                )
                _add_binary_op(
                    ctx=ctx,
                    op_type="FLOOR_MOD",
                    lhs_name=indices_plus_shape_name,
                    rhs_name=shape_prefix_name,
                    output_name=indices_wrapped_name,
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="SELECT",
                        inputs=[negative_mask_name, indices_wrapped_name, indices_for_scatter],
                        outputs=[normalized_indices_name],
                    )
                )
                indices_for_scatter = normalized_indices_name

    updates_shape = [int(v) if int(v) >= 0 else 1 for v in updates_meta_shape]
    ones_scalar = ctx.add_const_tensor(
        f"{output_name}_scatter_nd_one",
        np.asarray(1, dtype=_DTYPE_TO_NP[data_dtype]),
    )
    ones_scalar_tensor = ctx.model_ir.tensors.get(ones_scalar, None)
    if ones_scalar_tensor is not None:
        ones_scalar_tensor.shape = []
        ones_scalar_tensor.shape_signature = []
    updates_ones = ""
    if len(updates_meta_shape) > 0 and all(int(dim) > 0 for dim in updates_meta_shape):
        updates_ones = ctx.add_const_tensor(
            f"{output_name}_scatter_nd_updates_ones",
            np.ones([int(v) for v in updates_meta_shape], dtype=_DTYPE_TO_NP[data_dtype]),
        )
    else:
        updates_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_nd_updates_shape",
            dtype="INT32",
            shape=[len(updates_meta_shape)] if len(updates_meta_shape) > 0 else [1],
        )
        updates_ones = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_nd_updates_ones",
            dtype=data_dtype,
            shape=updates_shape,
        )
        updates_ones_tensor = ctx.model_ir.tensors.get(updates_ones, None)
        if updates_ones_tensor is not None:
            updates_ones_tensor.shape_signature = [int(v) for v in updates_meta_shape]
            updates_ones_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in updates_meta_shape]
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[updates_for_scatter],
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

    scatter_work_shape = [int(v) if int(v) > 0 else 1 for v in data_meta_shape]
    mask_scatter = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_nd_mask",
        dtype=data_dtype,
        shape=scatter_work_shape,
    )
    inverse_mask = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_nd_inverse_mask",
        dtype=data_dtype,
        shape=scatter_work_shape,
    )
    retained = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_nd_retained",
        dtype=data_dtype,
        shape=scatter_work_shape,
    )
    scattered_updates = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_nd_updates",
        dtype=data_dtype,
        shape=scatter_work_shape,
    )
    for tensor_name in [mask_scatter, inverse_mask, retained, scattered_updates]:
        tensor_ir = ctx.model_ir.tensors.get(tensor_name, None)
        if tensor_ir is not None:
            tensor_ir.shape_signature = [int(v) for v in data_meta_shape]

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
            inputs=[indices_for_scatter, updates_for_scatter, shape_for_scatter],
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


def build_tensor_scatter_op(node: Any, ctx: Any) -> None:
    data_name = node.inputs[0].name
    updates_name = node.inputs[1].name
    write_indices_name = node.inputs[2].name if len(node.inputs) > 2 else ""
    output_name = node.outputs[0].name
    ctx.ensure_tensor(data_name)
    ctx.ensure_tensor(updates_name)
    if str(write_indices_name) != "":
        ctx.ensure_tensor(write_indices_name)
    ctx.ensure_tensor(output_name)

    data_shape = _tensor_shape_with_signature(ctx, data_name)
    updates_shape = _tensor_shape_with_signature(ctx, updates_name)
    rank = int(len(data_shape))
    axis = _normalize_axis_for_rank(int(node.attrs.get("axis", -2)), rank)
    mode = str(node.attrs.get("mode", "linear")).lower()
    if mode not in {"linear", "circular"}:
        raise NotImplementedError(
            f"TensorScatter supports mode=linear or mode=circular only. op={node.name} mode={mode}"
        )

    data_dtype = str(ctx.get_tensor_dtype(data_name)).upper()
    output_tensor = ctx.model_ir.tensors[output_name]
    data_tensor = ctx.model_ir.tensors[data_name]
    output_tensor.dtype = data_dtype
    output_tensor.quantization = _clone_quantization(data_tensor.quantization)
    _propagate_shape(ctx, data_name, output_name)

    write_indices_i32_name = write_indices_name
    if str(write_indices_name) == "":
        write_indices_i32_name = ctx.add_const_tensor(
            f"{output_name}_tensor_scatter_write_indices_zero",
            np.zeros((int(updates_shape[0]),), dtype=np.int32),
        )
    else:
        write_indices_dtype = str(ctx.get_tensor_dtype(write_indices_name)).upper()
        if write_indices_dtype != "INT32":
            write_indices_const = ctx.get_constant_array(write_indices_name)
            if write_indices_const is not None:
                write_indices_i32_name = ctx.add_const_tensor(
                    f"{output_name}_tensor_scatter_write_indices_i32",
                    np.asarray(write_indices_const, dtype=np.int32),
                )
            else:
                write_indices_i32_name = ctx.add_intermediate_tensor(
                    f"{output_name}_tensor_scatter_write_indices_i32",
                    dtype="INT32",
                    shape=[int(v) for v in _tensor_shape_with_signature(ctx, write_indices_name)],
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[write_indices_name],
                        outputs=[write_indices_i32_name],
                        options={
                            "inDataType": write_indices_dtype,
                            "outDataType": "INT32",
                        },
                    )
                )

    updates_shape_static = [int(v) for v in updates_shape]
    num_updates = int(np.prod(np.asarray(updates_shape_static, dtype=np.int64)))
    batch_index_grid = np.broadcast_to(
        np.arange(int(updates_shape_static[0]), dtype=np.int32).reshape(
            [int(updates_shape_static[0])] + [1] * int(rank - 1)
        ),
        tuple(int(v) for v in updates_shape_static),
    )
    batch_index_flat_name = ctx.add_const_tensor(
        f"{output_name}_tensor_scatter_batch_index_flat",
        batch_index_grid.reshape(-1),
    )
    write_offsets_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_tensor_scatter_write_offsets_flat",
        dtype="INT32",
        shape=[int(num_updates)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[write_indices_i32_name, batch_index_flat_name],
            outputs=[write_offsets_flat_name],
            options={"axis": 0, "batchDims": 0},
        )
    )
    write_offsets_name = ctx.add_intermediate_tensor(
        f"{output_name}_tensor_scatter_write_offsets",
        dtype="INT32",
        shape=[int(v) for v in updates_shape_static],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=write_offsets_flat_name,
        output_name=write_offsets_name,
        new_shape=[int(v) for v in updates_shape_static],
    )

    axis_coord_grid = np.broadcast_to(
        np.arange(int(updates_shape_static[axis]), dtype=np.int32).reshape(
            [1] * int(axis) + [int(updates_shape_static[axis])] + [1] * int(rank - axis - 1)
        ),
        tuple(int(v) for v in updates_shape_static),
    )
    axis_coord_name = ctx.add_const_tensor(
        f"{output_name}_tensor_scatter_axis_coord",
        axis_coord_grid,
    )
    axis_indices_name = ctx.add_intermediate_tensor(
        f"{output_name}_tensor_scatter_axis_indices",
        dtype="INT32",
        shape=[int(v) for v in updates_shape_static],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=axis_coord_name,
        rhs_name=write_offsets_name,
        output_name=axis_indices_name,
    )
    if mode == "circular":
        axis_dim_name = ctx.add_const_tensor(
            f"{output_name}_tensor_scatter_axis_dim",
            np.asarray(int(data_shape[axis]), dtype=np.int32),
        )
        axis_wrapped_name = ctx.add_intermediate_tensor(
            f"{output_name}_tensor_scatter_axis_indices_wrapped",
            dtype="INT32",
            shape=[int(v) for v in updates_shape_static],
        )
        _add_binary_op(
            ctx=ctx,
            op_type="FLOOR_MOD",
            lhs_name=axis_indices_name,
            rhs_name=axis_dim_name,
            output_name=axis_wrapped_name,
        )
        axis_indices_name = axis_wrapped_name

    coord_expanded_names: list[str] = []
    coord_expanded_shape = [int(v) for v in updates_shape_static] + [1]
    for dim in range(rank):
        if int(dim) == int(axis):
            coord_name = ctx.add_intermediate_tensor(
                f"{output_name}_tensor_scatter_dim_{dim}_coord",
                dtype="INT32",
                shape=coord_expanded_shape,
            )
            _add_reshape_operator(
                ctx=ctx,
                input_name=axis_indices_name,
                output_name=coord_name,
                new_shape=coord_expanded_shape,
            )
        else:
            coord_grid = np.broadcast_to(
                np.arange(int(updates_shape_static[dim]), dtype=np.int32).reshape(
                    [1] * int(dim)
                    + [int(updates_shape_static[dim])]
                    + [1] * int(rank - dim - 1)
                ),
                tuple(int(v) for v in updates_shape_static),
            )
            coord_name = ctx.add_const_tensor(
                f"{output_name}_tensor_scatter_dim_{dim}_coord",
                np.expand_dims(coord_grid, axis=-1),
            )
        coord_expanded_names.append(coord_name)

    coordinates_name = coord_expanded_names[0]
    if len(coord_expanded_names) > 1:
        coordinates_name = ctx.add_intermediate_tensor(
            f"{output_name}_tensor_scatter_coordinates",
            dtype="INT32",
            shape=[int(v) for v in updates_shape_static] + [int(rank)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=coord_expanded_names,
                outputs=[coordinates_name],
                options={"axis": int(rank), "fusedActivationFunction": "NONE"},
            )
        )

    indices_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_tensor_scatter_indices_flat",
        dtype="INT32",
        shape=[int(num_updates), int(rank)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=coordinates_name,
        output_name=indices_flat_name,
        new_shape=[int(num_updates), int(rank)],
    )

    updates_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_tensor_scatter_updates_flat",
        dtype=str(ctx.get_tensor_dtype(updates_name)).upper(),
        shape=[int(num_updates)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=updates_name,
        output_name=updates_flat_name,
        new_shape=[int(num_updates)],
    )

    scatter_proxy_node = SimpleNamespace(
        name=f"{node.name}_tensor_scatter_proxy",
        op="ScatterND",
        attrs={"normalize_negative_indices": False},
        inputs=[
            SimpleNamespace(name=data_name),
            SimpleNamespace(name=indices_flat_name),
            SimpleNamespace(name=updates_flat_name),
        ],
        outputs=[SimpleNamespace(name=output_name)],
    )
    build_scatter_nd_op(scatter_proxy_node, ctx)


def build_unique_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    input_meta_shape = _tensor_shape_with_signature(ctx, input_name)
    output_tensor = ctx.model_ir.tensors[output_name]
    input_tensor = ctx.model_ir.tensors[input_name]
    output_tensor.quantization = _clone_quantization(input_tensor.quantization)

    def _mark_scalar(tensor_name: str) -> None:
        tensor_ir = ctx.model_ir.tensors.get(tensor_name, None)
        if tensor_ir is not None:
            tensor_ir.shape = []
            tensor_ir.shape_signature = []

    axis_attr = node.attrs.get("axis", None)
    axis_is_none = axis_attr is None
    if not axis_is_none:
        axis_norm = _normalize_axis_for_rank(
            axis=int(axis_attr),
            rank=int(len(input_meta_shape)),
        )
        if int(axis_norm) != 0:
            raise NotImplementedError(
                f"Unique lowering supports axis=0 or omitted axis only. op={node.name} axis={axis_attr}"
            )
        if int(len(input_meta_shape)) != 2:
            raise NotImplementedError(
                f"Unique axis=0 lowering requires rank-2 input. op={node.name} input_shape={input_meta_shape}"
            )
        if int(input_meta_shape[1]) <= 0:
            raise NotImplementedError(
                f"Unique axis=0 lowering requires static positive second dimension. op={node.name} input_shape={input_meta_shape}"
            )

        working_input = input_name
        working_dtype = str(input_dtype)
        if working_dtype != "INT32":
            working_dtype = "INT32"
            working_input = ctx.add_intermediate_tensor(
                f"{output_name}_unique_input_i32",
                dtype="INT32",
                shape=[int(v) if int(v) > 0 else 1 for v in input_meta_shape],
            )
            working_input_tensor = ctx.model_ir.tensors.get(working_input, None)
            if working_input_tensor is not None:
                working_input_tensor.shape_signature = [int(v) for v in input_meta_shape]
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[input_name],
                    outputs=[working_input],
                    options={
                        "inDataType": input_dtype,
                        "outDataType": "INT32",
                    },
                )
            )

        row_dim = int(input_meta_shape[0])
        row_vec_shape = [int(row_dim) if int(row_dim) > 0 else -1]
        col0_index_name = ctx.add_const_tensor(
            f"{output_name}_unique_axis0_col0_index",
            np.asarray([0], dtype=np.int32),
        )
        col1_index_name = ctx.add_const_tensor(
            f"{output_name}_unique_axis0_col1_index",
            np.asarray([1], dtype=np.int32),
        )
        col0_2d_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col0_2d",
            dtype=working_dtype,
            shape=[int(row_dim) if int(row_dim) > 0 else 1, 1],
        )
        col1_2d_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col1_2d",
            dtype=working_dtype,
            shape=[int(row_dim) if int(row_dim) > 0 else 1, 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[working_input, col0_index_name],
                outputs=[col0_2d_name],
                options={"axis": 1, "batchDims": 0},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[working_input, col1_index_name],
                outputs=[col1_2d_name],
                options={"axis": 1, "batchDims": 0},
            )
        )

        col0_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col0",
            dtype=working_dtype,
            shape=row_vec_shape,
        )
        col1_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col1",
            dtype=working_dtype,
            shape=row_vec_shape,
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=col0_2d_name,
            output_name=col0_name,
            new_shape=[-1],
            preserve_dynamic_shape=True,
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=col1_2d_name,
            output_name=col1_name,
            new_shape=[-1],
            preserve_dynamic_shape=True,
        )

        reduce_axes_name = ctx.add_const_tensor(
            f"{output_name}_unique_axis0_reduce_axes",
            np.asarray([0], dtype=np.int32),
        )
        min0_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_min0",
            dtype=working_dtype,
            shape=[],
        )
        min1_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_min1",
            dtype=working_dtype,
            shape=[],
        )
        _mark_scalar(min0_name)
        _mark_scalar(min1_name)
        ctx.add_operator(
            OperatorIR(
                op_type="REDUCE_MIN",
                inputs=[col0_name, reduce_axes_name],
                outputs=[min0_name],
                options={"keepDims": False},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="REDUCE_MIN",
                inputs=[col1_name, reduce_axes_name],
                outputs=[min1_name],
                options={"keepDims": False},
            )
        )

        col0_shift_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col0_shift",
            dtype=working_dtype,
            shape=row_vec_shape,
        )
        col1_shift_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col1_shift",
            dtype=working_dtype,
            shape=row_vec_shape,
        )
        _add_binary_op(
            ctx=ctx,
            op_type="SUB",
            lhs_name=col0_name,
            rhs_name=min0_name,
            output_name=col0_shift_name,
        )
        _add_binary_op(
            ctx=ctx,
            op_type="SUB",
            lhs_name=col1_name,
            rhs_name=min1_name,
            output_name=col1_shift_name,
        )

        max1_shift_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_max1_shift",
            dtype=working_dtype,
            shape=[],
        )
        _mark_scalar(max1_shift_name)
        ctx.add_operator(
            OperatorIR(
                op_type="REDUCE_MAX",
                inputs=[col1_shift_name, reduce_axes_name],
                outputs=[max1_shift_name],
                options={"keepDims": False},
            )
        )
        one_i32_name = ctx.add_const_tensor(
            f"{output_name}_unique_axis0_one",
            np.asarray(1, dtype=np.int32),
        )
        _mark_scalar(one_i32_name)
        base_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_base",
            dtype=working_dtype,
            shape=[],
        )
        _mark_scalar(base_name)
        _add_binary_op(
            ctx=ctx,
            op_type="ADD",
            lhs_name=max1_shift_name,
            rhs_name=one_i32_name,
            output_name=base_name,
        )

        key_mul_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_key_mul",
            dtype=working_dtype,
            shape=row_vec_shape,
        )
        key_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_key",
            dtype=working_dtype,
            shape=row_vec_shape,
        )
        _add_binary_op(
            ctx=ctx,
            op_type="MUL",
            lhs_name=col0_shift_name,
            rhs_name=base_name,
            output_name=key_mul_name,
        )
        _add_binary_op(
            ctx=ctx,
            op_type="ADD",
            lhs_name=key_mul_name,
            rhs_name=col1_shift_name,
            output_name=key_name,
        )

        unique_key_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_keys_unique",
            dtype=working_dtype,
            shape=[-1],
        )
        unique_idx_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_indices",
            dtype="INT32",
            shape=row_vec_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="UNIQUE",
                inputs=[key_name],
                outputs=[unique_key_name, unique_idx_name],
                options={"idxOutType": "INT32"},
            )
        )

        col0_shift_unique_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col0_shift_unique",
            dtype=working_dtype,
            shape=[-1],
        )
        col1_shift_unique_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col1_shift_unique",
            dtype=working_dtype,
            shape=[-1],
        )
        _add_binary_op(
            ctx=ctx,
            op_type="DIV",
            lhs_name=unique_key_name,
            rhs_name=base_name,
            output_name=col0_shift_unique_name,
        )
        _add_binary_op(
            ctx=ctx,
            op_type="FLOOR_MOD",
            lhs_name=unique_key_name,
            rhs_name=base_name,
            output_name=col1_shift_unique_name,
        )

        col0_unique_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col0_unique",
            dtype=working_dtype,
            shape=[-1],
        )
        col1_unique_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col1_unique",
            dtype=working_dtype,
            shape=[-1],
        )
        _add_binary_op(
            ctx=ctx,
            op_type="ADD",
            lhs_name=col0_shift_unique_name,
            rhs_name=min0_name,
            output_name=col0_unique_name,
        )
        _add_binary_op(
            ctx=ctx,
            op_type="ADD",
            lhs_name=col1_shift_unique_name,
            rhs_name=min1_name,
            output_name=col1_unique_name,
        )

        col0_unique_2d_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col0_unique_2d",
            dtype=working_dtype,
            shape=[-1, 1],
        )
        col1_unique_2d_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_col1_unique_2d",
            dtype=working_dtype,
            shape=[-1, 1],
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=col0_unique_name,
            output_name=col0_unique_2d_name,
            new_shape=[-1, 1],
            preserve_dynamic_shape=True,
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=col1_unique_name,
            output_name=col1_unique_2d_name,
            new_shape=[-1, 1],
            preserve_dynamic_shape=True,
        )

        rows_i32_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_axis0_rows_i32",
            dtype=working_dtype,
            shape=[-1, int(input_meta_shape[1])],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[col0_unique_2d_name, col1_unique_2d_name],
                outputs=[rows_i32_name],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            )
        )

        if output_dtype == "INT32":
            _add_reshape_operator(
                ctx=ctx,
                input_name=rows_i32_name,
                output_name=output_name,
                new_shape=[-1, int(input_meta_shape[1])],
                preserve_dynamic_shape=True,
            )
            return
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[rows_i32_name],
                outputs=[output_name],
                options={
                    "inDataType": "INT32",
                    "outDataType": output_dtype,
                },
            )
        )
        return

    unique_input_name = input_name
    if int(len(input_meta_shape)) != 1:
        unique_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_flatten_input",
            dtype=input_dtype,
            shape=[-1],
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=input_name,
            output_name=unique_input_name,
            new_shape=[-1],
            preserve_dynamic_shape=True,
        )

    unique_input_dtype = str(ctx.get_tensor_dtype(unique_input_name)).upper()
    flat_shape = _tensor_shape_with_signature(ctx, unique_input_name)
    unique_values_name = output_name
    if str(output_dtype) != str(unique_input_dtype):
        unique_values_name = ctx.add_intermediate_tensor(
            f"{output_name}_unique_values_raw",
            dtype=unique_input_dtype,
            shape=[-1],
        )
    unique_idx_name = ctx.add_intermediate_tensor(
        f"{output_name}_unique_indices",
        dtype="INT32",
        shape=flat_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="UNIQUE",
            inputs=[unique_input_name],
            outputs=[unique_values_name, unique_idx_name],
            options={"idxOutType": "INT32"},
        )
    )
    if str(unique_values_name) != str(output_name):
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[unique_values_name],
                outputs=[output_name],
                options={
                    "inDataType": unique_input_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_scatter_elements_op(node: Any, ctx: Any) -> None:
    data_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    updates_name = node.inputs[2].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(data_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(updates_name)
    ctx.ensure_tensor(output_name)

    data_shape = [int(v) for v in ctx.get_tensor_shape(data_name)]
    indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
    indices_meta_shape = _tensor_shape_with_signature(ctx, indices_name)
    updates_meta_shape = _tensor_shape_with_signature(ctx, updates_name)
    rank = int(len(data_shape))
    indices_rank = int(len(indices_meta_shape))
    updates_rank = int(len(updates_meta_shape))
    coord_rank = int(indices_rank) if int(indices_rank) >= int(updates_rank) else int(updates_rank)
    coord_shape_meta = (
        [int(v) for v in indices_meta_shape]
        if int(indices_rank) >= int(updates_rank)
        else [int(v) for v in updates_meta_shape]
    )
    leading_rank_pad = int(max(int(indices_rank) - int(updates_rank), 0))
    axis = _normalize_axis_for_rank(
        axis=int(node.attrs.get("axis", 0)),
        rank=rank,
    )

    data_dtype = str(ctx.get_tensor_dtype(data_name)).upper()
    output_tensor = ctx.model_ir.tensors[output_name]
    data_tensor = ctx.model_ir.tensors[data_name]
    output_tensor.dtype = data_dtype
    output_tensor.quantization = _clone_quantization(data_tensor.quantization)
    _propagate_shape(ctx, data_name, output_name)

    indices_i32_name = indices_name
    indices_dtype = str(ctx.get_tensor_dtype(indices_name)).upper()
    if indices_dtype != "INT32":
        indices_i32_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_indices_i32",
            dtype="INT32",
            shape=indices_meta_shape,
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

    axis_dim_name = ""
    if int(data_shape[axis]) > 0:
        axis_dim_name = ctx.add_const_tensor(
            f"{output_name}_scatter_elements_axis_dim",
            np.asarray(int(data_shape[axis]), dtype=np.int32),
        )
    else:
        data_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_data_shape",
            dtype="INT32",
            shape=[int(rank)] if int(rank) > 0 else [1],
        )
        axis_index_name = ctx.add_const_tensor(
            f"{output_name}_scatter_elements_axis_index",
            np.asarray([int(axis)], dtype=np.int32),
        )
        axis_dim_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_axis_dim",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[data_name],
                outputs=[data_shape_name],
                options={"outType": "INT32"},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[data_shape_name, axis_index_name],
                outputs=[axis_dim_name],
                options={"axis": 0, "batchDims": 0},
            )
        )

    zero_i32_name = ctx.add_const_tensor(
        f"{output_name}_scatter_elements_zero_i32",
        np.asarray(0, dtype=np.int32),
    )
    negative_mask_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_negative_mask",
        dtype="BOOL",
        shape=indices_meta_shape,
    )
    wrapped_indices_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_indices_wrapped",
        dtype="INT32",
        shape=indices_meta_shape,
    )
    normalized_indices_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_indices_normalized",
        dtype="INT32",
        shape=indices_meta_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LESS",
            inputs=[indices_i32_name, zero_i32_name],
            outputs=[negative_mask_name],
        )
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=indices_i32_name,
        rhs_name=axis_dim_name,
        output_name=wrapped_indices_name,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT",
            inputs=[negative_mask_name, wrapped_indices_name, indices_i32_name],
            outputs=[normalized_indices_name],
        )
    )

    indices_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_indices_shape",
        dtype="INT32",
        shape=[int(coord_rank)] if int(coord_rank) > 0 else [1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[normalized_indices_name],
            outputs=[indices_shape_name],
            options={"outType": "INT32"},
        )
    )
    indices_shape_plus_one_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_indices_shape_plus_one",
        dtype="INT32",
        shape=[int(coord_rank + 1)],
    )
    coord_last_dim_name = ctx.add_const_tensor(
        f"{output_name}_scatter_elements_coord_last_dim",
        np.asarray([1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[indices_shape_name, coord_last_dim_name],
            outputs=[indices_shape_plus_one_name],
            options={
                "axis": 0,
                "fusedActivationFunction": "NONE",
            },
        )
    )

    coord_expanded_names: list[str] = []
    updates_shape_name = ""
    if any(int(v) <= 0 for v in updates_meta_shape):
        updates_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_updates_shape",
            dtype="INT32",
            shape=[int(updates_rank)] if int(updates_rank) > 0 else [1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[updates_name],
                outputs=[updates_shape_name],
                options={"outType": "INT32"},
            )
        )

    range_start_name = ctx.add_const_tensor(
        f"{output_name}_scatter_elements_range_start",
        np.asarray([0], dtype=np.int32),
    )
    range_delta_name = ctx.add_const_tensor(
        f"{output_name}_scatter_elements_range_delta",
        np.asarray([1], dtype=np.int32),
    )
    range_start_scalar_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_range_start_scalar",
        dtype="INT32",
        shape=[],
    )
    range_delta_scalar_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_range_delta_scalar",
        dtype="INT32",
        shape=[],
    )
    range_start_scalar_tensor = ctx.model_ir.tensors.get(range_start_scalar_name, None)
    if range_start_scalar_tensor is not None:
        range_start_scalar_tensor.shape = []
        range_start_scalar_tensor.shape_signature = []
    range_delta_scalar_tensor = ctx.model_ir.tensors.get(range_delta_scalar_name, None)
    if range_delta_scalar_tensor is not None:
        range_delta_scalar_tensor.shape = []
        range_delta_scalar_tensor.shape_signature = []
    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[range_start_name],
            outputs=[range_start_scalar_name],
            options={"squeezeDims": [0]},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[range_delta_name],
            outputs=[range_delta_scalar_name],
            options={"squeezeDims": [0]},
        )
    )
    for dim in range(rank):
        target_dim = int(dim + leading_rank_pad)
        coord_base_name = normalized_indices_name
        if dim != axis:
            dim_size_scalar_name = ""
            if int(updates_meta_shape[dim]) > 0:
                dim_size_scalar_name = ctx.add_const_tensor(
                    f"{output_name}_scatter_elements_dim_{dim}_size_scalar",
                    np.asarray(int(updates_meta_shape[dim]), dtype=np.int32),
                )
                dim_size_scalar_tensor = ctx.model_ir.tensors.get(dim_size_scalar_name, None)
                if dim_size_scalar_tensor is not None:
                    dim_size_scalar_tensor.shape = []
                    dim_size_scalar_tensor.shape_signature = []
            else:
                dim_index_name = ctx.add_const_tensor(
                    f"{output_name}_scatter_elements_dim_{dim}_index",
                    np.asarray([int(dim)], dtype=np.int32),
                )
                dim_size_name = ctx.add_intermediate_tensor(
                    f"{output_name}_scatter_elements_dim_{dim}_size",
                    dtype="INT32",
                    shape=[1],
                )
                dim_size_scalar_name = ctx.add_intermediate_tensor(
                    f"{output_name}_scatter_elements_dim_{dim}_size_scalar",
                    dtype="INT32",
                    shape=[],
                )
                dim_size_scalar_tensor = ctx.model_ir.tensors.get(dim_size_scalar_name, None)
                if dim_size_scalar_tensor is not None:
                    dim_size_scalar_tensor.shape = []
                    dim_size_scalar_tensor.shape_signature = []
                ctx.add_operator(
                    OperatorIR(
                        op_type="GATHER",
                        inputs=[updates_shape_name, dim_index_name],
                        outputs=[dim_size_name],
                        options={"axis": 0, "batchDims": 0},
                    )
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="SQUEEZE",
                        inputs=[dim_size_name],
                        outputs=[dim_size_scalar_name],
                        options={"squeezeDims": [0]},
                    )
                )
            range_name = ctx.add_intermediate_tensor(
                f"{output_name}_scatter_elements_dim_{dim}_range",
                dtype="INT32",
                shape=[int(updates_meta_shape[dim]) if int(updates_meta_shape[dim]) > 0 else -1],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RANGE",
                    inputs=[range_start_scalar_name, dim_size_scalar_name, range_delta_scalar_name],
                    outputs=[range_name],
                )
            )

            reshape_pattern = [1 for _ in range(coord_rank)]
            reshape_pattern[target_dim] = -1
            range_reshaped_name = ctx.add_intermediate_tensor(
                f"{output_name}_scatter_elements_dim_{dim}_range_reshaped",
                dtype="INT32",
                shape=[
                    int(coord_shape_meta[idx]) if idx == int(target_dim) else 1
                    for idx in range(coord_rank)
                ],
            )
            _add_reshape_operator(
                ctx=ctx,
                input_name=range_name,
                output_name=range_reshaped_name,
                new_shape=[int(v) for v in reshape_pattern],
            )

            tile_mask = np.ones((coord_rank,), dtype=np.int32)
            tile_mask[target_dim] = 0
            tile_unit = np.zeros((coord_rank,), dtype=np.int32)
            tile_unit[target_dim] = 1
            tile_mask_name = ctx.add_const_tensor(
                f"{output_name}_scatter_elements_dim_{dim}_tile_mask",
                tile_mask,
            )
            tile_unit_name = ctx.add_const_tensor(
                f"{output_name}_scatter_elements_dim_{dim}_tile_unit",
                tile_unit,
            )
            tile_multiple_masked_name = ctx.add_intermediate_tensor(
                f"{output_name}_scatter_elements_dim_{dim}_tile_masked",
                dtype="INT32",
                shape=[int(coord_rank)],
            )
            tile_multiple_name = ctx.add_intermediate_tensor(
                f"{output_name}_scatter_elements_dim_{dim}_tile_multiple",
                dtype="INT32",
                shape=[int(coord_rank)],
            )
            _add_binary_op(
                ctx=ctx,
                op_type="MUL",
                lhs_name=indices_shape_name,
                rhs_name=tile_mask_name,
                output_name=tile_multiple_masked_name,
            )
            _add_binary_op(
                ctx=ctx,
                op_type="ADD",
                lhs_name=tile_multiple_masked_name,
                rhs_name=tile_unit_name,
                output_name=tile_multiple_name,
            )
            coord_base_name = ctx.add_intermediate_tensor(
                f"{output_name}_scatter_elements_dim_{dim}_coord",
                dtype="INT32",
                shape=coord_shape_meta,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TILE",
                    inputs=[range_reshaped_name, tile_multiple_name],
                    outputs=[coord_base_name],
                )
            )

        coord_expanded_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_dim_{dim}_coord_expanded",
            dtype="INT32",
            shape=[int(v) for v in coord_shape_meta] + [1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[coord_base_name, indices_shape_plus_one_name],
                outputs=[coord_expanded_name],
                options={
                    "newShape": [int(v) for v in list(coord_shape_meta)] + [1],
                },
            )
        )
        coord_expanded_names.append(coord_expanded_name)

    coordinates_name = coord_expanded_names[0]
    if len(coord_expanded_names) > 1:
        coordinates_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_coordinates",
            dtype="INT32",
            shape=[int(v) for v in coord_shape_meta] + [int(rank)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=coord_expanded_names,
                outputs=[coordinates_name],
                options={
                    "axis": int(coord_rank),
                    "fusedActivationFunction": "NONE",
                },
            )
        )

    indices_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_indices_flat",
        dtype="INT32",
        shape=[-1, int(rank)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=coordinates_name,
        output_name=indices_flat_name,
        new_shape=[-1, int(rank)],
    )

    updates_for_scatter_name = updates_name
    updates_dtype = str(ctx.get_tensor_dtype(updates_name)).upper()
    if updates_dtype != data_dtype:
        updates_for_scatter_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_updates_cast",
            dtype=data_dtype,
            shape=updates_meta_shape,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[updates_name],
                outputs=[updates_for_scatter_name],
                options={
                    "inDataType": updates_dtype,
                    "outDataType": data_dtype,
                },
            )
        )
    if int(indices_rank) > int(updates_rank):
        updates_padded_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_updates_padded",
            dtype=data_dtype,
            shape=([1] * int(leading_rank_pad)) + [int(v) for v in updates_meta_shape],
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=updates_for_scatter_name,
            output_name=updates_padded_name,
            new_shape=([1] * int(leading_rank_pad)) + [int(v) for v in updates_meta_shape],
        )
        updates_tile_multiples = np.asarray(
            [int(v) for v in coord_shape_meta[:leading_rank_pad]] + [1] * int(updates_rank),
            dtype=np.int32,
        )
        updates_tile_multiples_name = ctx.add_const_tensor(
            f"{output_name}_scatter_elements_updates_tile_multiples",
            updates_tile_multiples,
        )
        updates_broadcast_name = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_updates_broadcast",
            dtype=data_dtype,
            shape=[int(v) for v in coord_shape_meta],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="TILE",
                inputs=[updates_padded_name, updates_tile_multiples_name],
                outputs=[updates_broadcast_name],
            )
        )
        updates_for_scatter_name = updates_broadcast_name

    updates_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_updates_flat",
        dtype=data_dtype,
        shape=[-1],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=updates_for_scatter_name,
        output_name=updates_flat_name,
        new_shape=[-1],
    )

    shape_for_scatter = ""
    if rank > 0 and all(int(dim) > 0 for dim in data_shape):
        shape_for_scatter = ctx.add_const_tensor(
            f"{output_name}_scatter_elements_shape",
            np.asarray(data_shape, dtype=np.int32),
        )
    else:
        shape_for_scatter = ctx.add_intermediate_tensor(
            f"{output_name}_scatter_elements_shape",
            dtype="INT32",
            shape=[int(rank)] if int(rank) > 0 else [1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[data_name],
                outputs=[shape_for_scatter],
                options={"outType": "INT32"},
            )
        )

    one_name = ctx.add_const_tensor(
        f"{output_name}_scatter_elements_one",
        np.asarray(1, dtype=_DTYPE_TO_NP[data_dtype]),
    )
    one_tensor = ctx.model_ir.tensors.get(one_name, None)
    if one_tensor is not None:
        one_tensor.shape = []
        one_tensor.shape_signature = []
    updates_flat_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_updates_flat_shape",
        dtype="INT32",
        shape=[1],
    )
    updates_ones_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_updates_ones",
        dtype=data_dtype,
        shape=[-1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[updates_flat_name],
            outputs=[updates_flat_shape_name],
            options={"outType": "INT32"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="FILL",
            inputs=[updates_flat_shape_name, one_name],
            outputs=[updates_ones_name],
        )
    )

    data_meta_shape = _tensor_shape_with_signature(ctx, data_name)
    mask_scatter_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_mask",
        dtype=data_dtype,
        shape=data_meta_shape,
    )
    inverse_mask_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_inverse_mask",
        dtype=data_dtype,
        shape=data_meta_shape,
    )
    retained_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_retained",
        dtype=data_dtype,
        shape=data_meta_shape,
    )
    scattered_updates_name = ctx.add_intermediate_tensor(
        f"{output_name}_scatter_elements_updates_scattered",
        dtype=data_dtype,
        shape=data_meta_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=[indices_flat_name, updates_ones_name, shape_for_scatter],
            outputs=[mask_scatter_name],
        )
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=one_name,
        rhs_name=mask_scatter_name,
        output_name=inverse_mask_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=data_name,
        rhs_name=inverse_mask_name,
        output_name=retained_name,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SCATTER_ND",
            inputs=[indices_flat_name, updates_flat_name, shape_for_scatter],
            outputs=[scattered_updates_name],
        )
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=retained_name,
        rhs_name=scattered_updates_name,
        output_name=output_name,
    )


def build_roi_align_op(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    rois_name = node.inputs[1].name
    batch_indices_name = node.inputs[2].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(rois_name)
    ctx.ensure_tensor(batch_indices_name)
    ctx.ensure_tensor(output_name)

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    if len(input_shape) != 4:
        raise NotImplementedError(
            f"RoiAlign supports rank-4 input only in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    if len(output_shape) != 4:
        raise NotImplementedError(
            f"RoiAlign supports rank-4 output only in flatbuffer_direct. op={node.name} output_shape={output_shape}"
        )

    _, channels, in_h, in_w = [int(v) for v in input_shape]
    if int(channels) <= 0 or int(in_h) <= 0 or int(in_w) <= 0:
        raise NotImplementedError(
            "RoiAlign requires static positive C/H/W on input in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )

    mode = str(node.attrs.get("mode", "avg")).lower()
    if mode not in {"avg", "max"}:
        raise NotImplementedError(
            f"RoiAlign supports mode in {{avg,max}} only in flatbuffer_direct. op={node.name} mode={mode}"
        )
    output_height = int(node.attrs.get("output_height", output_shape[2]))
    output_width = int(node.attrs.get("output_width", output_shape[3]))
    if int(output_height) <= 0 or int(output_width) <= 0:
        raise NotImplementedError(
            "RoiAlign requires positive output_height/output_width in flatbuffer_direct. "
            f"op={node.name} output_height={output_height} output_width={output_width}"
        )
    sampling_ratio = int(node.attrs.get("sampling_ratio", 0))
    if int(sampling_ratio) <= 0:
        sampling_ratio = int((int(output_height) + int(output_width)) / 2)
    if int(sampling_ratio) <= 0:
        sampling_ratio = 1
    pooled_h = int(output_height) * int(sampling_ratio)
    pooled_w = int(output_width) * int(sampling_ratio)
    spatial_scale = float(node.attrs.get("spatial_scale", 1.0))

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    rois_dtype = str(ctx.get_tensor_dtype(rois_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    compute_dtype = (
        "FLOAT32"
        if input_dtype == "FLOAT32" or rois_dtype == "FLOAT32" or output_dtype == "FLOAT32"
        else "FLOAT16"
    )
    if compute_dtype not in {"FLOAT16", "FLOAT32"}:
        compute_dtype = "FLOAT32"
    compute_np_dtype = np.float16 if compute_dtype == "FLOAT16" else np.float32

    output_signature = _tensor_shape_with_signature(ctx, output_name)
    roi_count_meta = int(output_signature[0]) if int(output_signature[0]) < 0 else int(output_shape[0])
    if int(roi_count_meta) == 0:
        roi_count_meta = -1

    input_compute_name = input_name
    if input_dtype != compute_dtype:
        input_compute_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_input_{compute_dtype.lower()}",
            dtype=compute_dtype,
            shape=_tensor_shape_with_signature(ctx, input_name),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[input_name],
                outputs=[input_compute_name],
                options={
                    "inDataType": input_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    rois_compute_name = rois_name
    if rois_dtype != compute_dtype:
        rois_compute_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_rois_{compute_dtype.lower()}",
            dtype=compute_dtype,
            shape=_tensor_shape_with_signature(ctx, rois_name),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[rois_name],
                outputs=[rois_compute_name],
                options={
                    "inDataType": rois_dtype,
                    "outDataType": compute_dtype,
                },
            )
        )

    rois_scaled_name = rois_compute_name
    if not np.isclose(spatial_scale, 1.0):
        scale_name = ctx.add_const_tensor(
            f"{output_name}_roialign_spatial_scale",
            np.asarray(spatial_scale, dtype=compute_np_dtype),
        )
        rois_scaled_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_rois_scaled",
            dtype=compute_dtype,
            shape=_tensor_shape_with_signature(ctx, rois_name),
        )
        _add_binary_op(
            ctx=ctx,
            op_type="MUL",
            lhs_name=rois_compute_name,
            rhs_name=scale_name,
            output_name=rois_scaled_name,
        )

    def _gather_roi_coord(coord_idx: int, coord_tag: str) -> str:
        index_name = ctx.add_const_tensor(
            f"{output_name}_roialign_coord_{coord_tag}_index",
            np.asarray([int(coord_idx)], dtype=np.int32),
        )
        coord_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_coord_{coord_tag}",
            dtype=compute_dtype,
            shape=[int(roi_count_meta), 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[rois_scaled_name, index_name],
                outputs=[coord_name],
                options={
                    "axis": 1,
                    "batchDims": 0,
                },
            )
        )
        return coord_name

    x0_name = _gather_roi_coord(0, "x0")
    y0_name = _gather_roi_coord(1, "y0")
    x1_name = _gather_roi_coord(2, "x1")
    y1_name = _gather_roi_coord(3, "y1")

    pooled_w_name = ctx.add_const_tensor(
        f"{output_name}_roialign_pooled_w",
        np.asarray(float(pooled_w), dtype=compute_np_dtype),
    )
    pooled_h_name = ctx.add_const_tensor(
        f"{output_name}_roialign_pooled_h",
        np.asarray(float(pooled_h), dtype=compute_np_dtype),
    )
    half_name = ctx.add_const_tensor(
        f"{output_name}_roialign_half",
        np.asarray(0.5, dtype=compute_np_dtype),
    )

    roi_w_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_roi_w",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1],
    )
    roi_h_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_roi_h",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1],
    )
    spacing_w_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_spacing_w",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1],
    )
    spacing_h_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_spacing_h",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1],
    )
    half_spacing_w_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_half_spacing_w",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1],
    )
    half_spacing_h_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_half_spacing_h",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1],
    )
    x_start_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x_start",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1],
    )
    y_start_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y_start",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=x1_name,
        rhs_name=x0_name,
        output_name=roi_w_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=y1_name,
        rhs_name=y0_name,
        output_name=roi_h_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="DIV",
        lhs_name=roi_w_name,
        rhs_name=pooled_w_name,
        output_name=spacing_w_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="DIV",
        lhs_name=roi_h_name,
        rhs_name=pooled_h_name,
        output_name=spacing_h_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=spacing_w_name,
        rhs_name=half_name,
        output_name=half_spacing_w_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=spacing_h_name,
        rhs_name=half_name,
        output_name=half_spacing_h_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=x0_name,
        rhs_name=half_spacing_w_name,
        output_name=x_start_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=y0_name,
        rhs_name=half_spacing_h_name,
        output_name=y_start_name,
    )

    x_index_name = ctx.add_const_tensor(
        f"{output_name}_roialign_x_index",
        np.arange(pooled_w, dtype=compute_np_dtype).reshape(1, pooled_w),
    )
    y_index_name = ctx.add_const_tensor(
        f"{output_name}_roialign_y_index",
        np.arange(pooled_h, dtype=compute_np_dtype).reshape(1, pooled_h),
    )
    x_offset_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x_offset",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_w)],
    )
    y_offset_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y_offset",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h)],
    )
    x_coords_2d_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x_coords_2d",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_w)],
    )
    y_coords_2d_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y_coords_2d",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h)],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=spacing_w_name,
        rhs_name=x_index_name,
        output_name=x_offset_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=spacing_h_name,
        rhs_name=y_index_name,
        output_name=y_offset_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=x_start_name,
        rhs_name=x_offset_name,
        output_name=x_coords_2d_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=y_start_name,
        rhs_name=y_offset_name,
        output_name=y_coords_2d_name,
    )

    x_coords_3d_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x_coords_3d",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), 1, int(pooled_w)],
    )
    y_coords_3d_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y_coords_3d",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), 1],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=x_coords_2d_name,
        output_name=x_coords_3d_name,
        new_shape=[-1, 1, int(pooled_w)],
        preserve_dynamic_shape=True,
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=y_coords_2d_name,
        output_name=y_coords_3d_name,
        new_shape=[-1, int(pooled_h), 1],
        preserve_dynamic_shape=True,
    )

    tile_x_name = ctx.add_const_tensor(
        f"{output_name}_roialign_tile_x",
        np.asarray([1, int(pooled_h), 1], dtype=np.int32),
    )
    tile_y_name = ctx.add_const_tensor(
        f"{output_name}_roialign_tile_y",
        np.asarray([1, 1, int(pooled_w)], dtype=np.int32),
    )
    x_coords_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x_coords",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y_coords_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y_coords",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="TILE",
            inputs=[x_coords_3d_name, tile_x_name],
            outputs=[x_coords_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="TILE",
            inputs=[y_coords_3d_name, tile_y_name],
            outputs=[y_coords_name],
        )
    )

    batch_indices_i32_name = batch_indices_name
    batch_indices_dtype = str(ctx.get_tensor_dtype(batch_indices_name)).upper()
    if batch_indices_dtype != "INT32":
        batch_indices_i32_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_batch_indices_i32",
            dtype="INT32",
            shape=_tensor_shape_with_signature(ctx, batch_indices_name),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[batch_indices_name],
                outputs=[batch_indices_i32_name],
                options={
                    "inDataType": batch_indices_dtype,
                    "outDataType": "INT32",
                },
            )
        )

    gathered_input_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_input_gathered",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(in_h), int(in_w)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[input_compute_name, batch_indices_i32_name],
            outputs=[gathered_input_name],
            options={
                "axis": 0,
                "batchDims": 0,
            },
        )
    )

    padded_input_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_input_padded",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(in_h + 2), int(in_w + 2)],
    )
    paddings_name = ctx.add_const_tensor(
        f"{output_name}_roialign_paddings",
        np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="PAD",
            inputs=[gathered_input_name, paddings_name],
            outputs=[padded_input_name],
        )
    )

    flattened_input_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_input_flattened",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int((in_h + 2) * (in_w + 2))],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=padded_input_name,
        output_name=flattened_input_name,
        new_shape=[-1, int(channels), int((in_h + 2) * (in_w + 2))],
        preserve_dynamic_shape=True,
    )

    neg_one_name = ctx.add_const_tensor(
        f"{output_name}_roialign_neg_one",
        np.asarray(-1.0, dtype=compute_np_dtype),
    )
    one_name = ctx.add_const_tensor(
        f"{output_name}_roialign_one",
        np.asarray(1.0, dtype=compute_np_dtype),
    )
    in_w_name = ctx.add_const_tensor(
        f"{output_name}_roialign_in_w",
        np.asarray(float(in_w), dtype=compute_np_dtype),
    )
    in_h_name = ctx.add_const_tensor(
        f"{output_name}_roialign_in_h",
        np.asarray(float(in_h), dtype=compute_np_dtype),
    )
    in_w_plus_one_name = ctx.add_const_tensor(
        f"{output_name}_roialign_in_w_plus_one",
        np.asarray(float(in_w + 1), dtype=compute_np_dtype),
    )
    in_h_plus_one_name = ctx.add_const_tensor(
        f"{output_name}_roialign_in_h_plus_one",
        np.asarray(float(in_h + 1), dtype=compute_np_dtype),
    )
    width_pad_i32_name = ctx.add_const_tensor(
        f"{output_name}_roialign_width_pad_i32",
        np.asarray(int(in_w + 2), dtype=np.int32),
    )

    x_clip_low_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x_clip_low",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    x_clip_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x_clip",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y_clip_low_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y_clip_low",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y_clip_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y_clip",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    x_shift_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x_shift",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y_shift_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y_shift",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MAXIMUM",
        lhs_name=x_coords_name,
        rhs_name=neg_one_name,
        output_name=x_clip_low_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MINIMUM",
        lhs_name=x_clip_low_name,
        rhs_name=in_w_name,
        output_name=x_clip_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MAXIMUM",
        lhs_name=y_coords_name,
        rhs_name=neg_one_name,
        output_name=y_clip_low_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MINIMUM",
        lhs_name=y_clip_low_name,
        rhs_name=in_h_name,
        output_name=y_clip_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=x_clip_name,
        rhs_name=one_name,
        output_name=x_shift_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=y_clip_name,
        rhs_name=one_name,
        output_name=y_shift_name,
    )

    x0_floor_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x0_floor",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y0_floor_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y0_floor",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    x1_floor_pre_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x1_floor_pre",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y1_floor_pre_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y1_floor_pre",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    x1_floor_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x1_floor",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y1_floor_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y1_floor",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="FLOOR",
            inputs=[x_shift_name],
            outputs=[x0_floor_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="FLOOR",
            inputs=[y_shift_name],
            outputs=[y0_floor_name],
        )
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=x0_floor_name,
        rhs_name=one_name,
        output_name=x1_floor_pre_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=y0_floor_name,
        rhs_name=one_name,
        output_name=y1_floor_pre_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MINIMUM",
        lhs_name=x1_floor_pre_name,
        rhs_name=in_w_plus_one_name,
        output_name=x1_floor_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MINIMUM",
        lhs_name=y1_floor_pre_name,
        rhs_name=in_h_plus_one_name,
        output_name=y1_floor_name,
    )

    x0_i32_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x0_i32",
        dtype="INT32",
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y0_i32_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y0_i32",
        dtype="INT32",
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    x1_i32_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_x1_i32",
        dtype="INT32",
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    y1_i32_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_y1_i32",
        dtype="INT32",
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[x0_floor_name],
            outputs=[x0_i32_name],
            options={"inDataType": compute_dtype, "outDataType": "INT32"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[y0_floor_name],
            outputs=[y0_i32_name],
            options={"inDataType": compute_dtype, "outDataType": "INT32"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[x1_floor_name],
            outputs=[x1_i32_name],
            options={"inDataType": compute_dtype, "outDataType": "INT32"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[y1_floor_name],
            outputs=[y1_i32_name],
            options={"inDataType": compute_dtype, "outDataType": "INT32"},
        )
    )

    def _build_linear_index(y_idx_name: str, x_idx_name: str, tag: str) -> str:
        y_mul_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_{tag}_y_mul",
            dtype="INT32",
            shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
        )
        linear_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_{tag}_linear",
            dtype="INT32",
            shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
        )
        _add_binary_op(
            ctx=ctx,
            op_type="MUL",
            lhs_name=y_idx_name,
            rhs_name=width_pad_i32_name,
            output_name=y_mul_name,
        )
        _add_binary_op(
            ctx=ctx,
            op_type="ADD",
            lhs_name=y_mul_name,
            rhs_name=x_idx_name,
            output_name=linear_name,
        )
        return linear_name

    idx_00_name = _build_linear_index(y0_i32_name, x0_i32_name, "idx00")
    idx_01_name = _build_linear_index(y0_i32_name, x1_i32_name, "idx01")
    idx_10_name = _build_linear_index(y1_i32_name, x0_i32_name, "idx10")
    idx_11_name = _build_linear_index(y1_i32_name, x1_i32_name, "idx11")

    def _build_gather(linear_index_name: str, tag: str) -> str:
        gathered_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_{tag}_gather",
            dtype=compute_dtype,
            shape=[int(roi_count_meta), int(channels), int(pooled_h), int(pooled_w)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[flattened_input_name, linear_index_name],
                outputs=[gathered_name],
                options={
                    "axis": 2,
                    "batchDims": 1,
                },
            )
        )
        return gathered_name

    gathered_00_name = _build_gather(idx_00_name, "v00")
    gathered_01_name = _build_gather(idx_01_name, "v01")
    gathered_10_name = _build_gather(idx_10_name, "v10")
    gathered_11_name = _build_gather(idx_11_name, "v11")

    wx_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_wx",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    wy_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_wy",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    one_minus_wx_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_one_minus_wx",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    one_minus_wy_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_one_minus_wy",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=x_shift_name,
        rhs_name=x0_floor_name,
        output_name=wx_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=y_shift_name,
        rhs_name=y0_floor_name,
        output_name=wy_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=one_name,
        rhs_name=wx_name,
        output_name=one_minus_wx_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=one_name,
        rhs_name=wy_name,
        output_name=one_minus_wy_name,
    )

    w00_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_w00",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    w01_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_w01",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    w10_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_w10",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    w11_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_w11",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(pooled_h), int(pooled_w)],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=one_minus_wx_name,
        rhs_name=one_minus_wy_name,
        output_name=w00_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=wx_name,
        rhs_name=one_minus_wy_name,
        output_name=w01_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=one_minus_wx_name,
        rhs_name=wy_name,
        output_name=w10_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=wx_name,
        rhs_name=wy_name,
        output_name=w11_name,
    )

    def _expand_weight(weight_name: str, tag: str) -> str:
        expanded_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_{tag}_expanded",
            dtype=compute_dtype,
            shape=[int(roi_count_meta), 1, int(pooled_h), int(pooled_w)],
        )
        _add_reshape_operator(
            ctx=ctx,
            input_name=weight_name,
            output_name=expanded_name,
            new_shape=[-1, 1, int(pooled_h), int(pooled_w)],
            preserve_dynamic_shape=True,
        )
        return expanded_name

    w00_expanded_name = _expand_weight(w00_name, "w00")
    w01_expanded_name = _expand_weight(w01_name, "w01")
    w10_expanded_name = _expand_weight(w10_name, "w10")
    w11_expanded_name = _expand_weight(w11_name, "w11")

    weighted_00_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_weighted_00",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(pooled_h), int(pooled_w)],
    )
    weighted_01_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_weighted_01",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(pooled_h), int(pooled_w)],
    )
    weighted_10_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_weighted_10",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(pooled_h), int(pooled_w)],
    )
    weighted_11_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_weighted_11",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(pooled_h), int(pooled_w)],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=gathered_00_name,
        rhs_name=w00_expanded_name,
        output_name=weighted_00_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=gathered_01_name,
        rhs_name=w01_expanded_name,
        output_name=weighted_01_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=gathered_10_name,
        rhs_name=w10_expanded_name,
        output_name=weighted_10_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=gathered_11_name,
        rhs_name=w11_expanded_name,
        output_name=weighted_11_name,
    )

    weighted_top_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_weighted_top",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(pooled_h), int(pooled_w)],
    )
    weighted_bottom_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_weighted_bottom",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(pooled_h), int(pooled_w)],
    )
    sampled_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_sampled",
        dtype=compute_dtype,
        shape=[int(roi_count_meta), int(channels), int(pooled_h), int(pooled_w)],
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=weighted_00_name,
        rhs_name=weighted_01_name,
        output_name=weighted_top_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=weighted_10_name,
        rhs_name=weighted_11_name,
        output_name=weighted_bottom_name,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="ADD",
        lhs_name=weighted_top_name,
        rhs_name=weighted_bottom_name,
        output_name=sampled_name,
    )

    output_compute_name = sampled_name
    if int(sampling_ratio) > 1:
        sampled_nhwc_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_sampled_nhwc",
            dtype=compute_dtype,
            shape=[int(roi_count_meta), int(pooled_h), int(pooled_w), int(channels)],
        )
        make_transpose(
            ctx=ctx,
            input_name=sampled_name,
            output_name=sampled_nhwc_name,
            perm_values=[0, 2, 3, 1],
        )
        pooled_nhwc_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_pooled_nhwc",
            dtype=compute_dtype,
            shape=[int(roi_count_meta), int(output_height), int(output_width), int(channels)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="AVERAGE_POOL_2D" if mode == "avg" else "MAX_POOL_2D",
                inputs=[sampled_nhwc_name],
                outputs=[pooled_nhwc_name],
                options={
                    "padding": "VALID",
                    "strideH": int(sampling_ratio),
                    "strideW": int(sampling_ratio),
                    "filterHeight": int(sampling_ratio),
                    "filterWidth": int(sampling_ratio),
                    "fusedActivationFunction": "NONE",
                },
            )
        )
        output_compute_name = ctx.add_intermediate_tensor(
            f"{output_name}_roialign_output_nchw",
            dtype=compute_dtype,
            shape=[int(roi_count_meta), int(channels), int(output_height), int(output_width)],
        )
        make_transpose(
            ctx=ctx,
            input_name=pooled_nhwc_name,
            output_name=output_compute_name,
            perm_values=[0, 3, 1, 2],
        )

    if output_compute_name != output_name:
        if compute_dtype == output_dtype:
            make_transpose(
                ctx=ctx,
                input_name=output_compute_name,
                output_name=output_name,
                perm_values=[0, 1, 2, 3],
            )
        else:
            ctx.add_operator(
                OperatorIR(
                    op_type="CAST",
                    inputs=[output_compute_name],
                    outputs=[output_name],
                    options={
                        "inDataType": compute_dtype,
                        "outDataType": output_dtype,
                    },
                )
            )


def _set_tensor_shape_signature(
    *,
    ctx: Any,
    tensor_name: str,
    shape: list[int],
    signature: list[int] | None = None,
) -> None:
    tensor = ctx.model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return
    tensor.shape = [int(v) if int(v) >= 0 else 1 for v in list(shape)]
    tensor.shape_signature = (
        [int(v) for v in list(signature)]
        if signature is not None
        else [int(v) for v in list(shape)]
    )


def _add_cast_op(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    out_dtype: str,
    shape: list[int],
    signature: list[int] | None = None,
) -> str:
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=output_name,
        shape=shape,
        signature=signature,
    )
    ctx.model_ir.tensors[output_name].dtype = str(out_dtype).upper()
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[input_name],
            outputs=[output_name],
            options={
                "inDataType": str(ctx.get_tensor_dtype(input_name)).upper(),
                "outDataType": str(out_dtype).upper(),
            },
        )
    )
    return str(output_name)


def _add_reshape_op(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    new_shape: list[int],
    signature: list[int] | None = None,
) -> str:
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(output_name)
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=output_name,
        shape=[int(v) if int(v) >= 0 else 1 for v in list(new_shape)],
        signature=signature if signature is not None else new_shape,
    )
    shape_name = ctx.add_const_tensor(
        f"{output_name}_shape",
        np.asarray([int(v) for v in list(new_shape)], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            options={
                "newShape": [int(v) for v in list(new_shape)],
                "preserveDynamicShape": True,
            },
        )
    )
    return str(output_name)


def _build_reducemax_argmax_op(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
    axis: int,
    keepdims: bool,
    final_dtype: str,
) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor is not None and input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )
    axis_dim = _resolve_positive_axis_dim(
        input_shape=input_shape,
        input_signature=input_signature,
        axis=axis,
    )
    if axis_dim <= 0:
        raise NotImplementedError(
            "ArgMax ReduceMax replacement requires static positive axis dimension in flatbuffer_direct. "
            f"op={node.name} axis={axis} axis_dim={axis_dim}"
        )

    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    output_shape = (
        [int(v) for v in list(output_tensor.shape)]
        if output_tensor is not None
        else [1]
    )
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in list(output_shape)]
    )

    compute_input_name = input_name
    if str(ctx.get_tensor_dtype(input_name)).upper() != "FLOAT32":
        compute_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_argmax_compute_input",
            dtype="FLOAT32",
            shape=input_shape,
        )
        _add_cast_op(
            ctx=ctx,
            input_name=input_name,
            output_name=compute_input_name,
            out_dtype="FLOAT32",
            shape=input_shape,
            signature=input_signature,
        )

    reduce_keepdims_shape = [int(v) for v in list(input_shape)]
    reduce_keepdims_shape[int(axis)] = 1
    reduce_keepdims_signature = [int(v) for v in list(input_signature)]
    reduce_keepdims_signature[int(axis)] = 1
    axis_name = ctx.add_const_tensor(
        f"{output_name}_argmax_reduce_axis",
        np.asarray([int(axis)], dtype=np.int32),
    )
    axis_max_name = ctx.add_intermediate_tensor(
        f"{output_name}_argmax_axis_max",
        dtype="FLOAT32",
        shape=[int(v) if int(v) >= 0 else 1 for v in list(reduce_keepdims_shape)],
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=axis_max_name,
        shape=reduce_keepdims_shape,
        signature=reduce_keepdims_signature,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="REDUCE_MAX",
            inputs=[compute_input_name, axis_name],
            outputs=[axis_max_name],
            options={"keepDims": True},
        )
    )
    zero_if_max_name = ctx.add_intermediate_tensor(
        f"{output_name}_argmax_zero_if_max",
        dtype="FLOAT32",
        shape=input_shape,
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=zero_if_max_name,
        shape=input_shape,
        signature=input_signature,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=axis_max_name,
        rhs_name=compute_input_name,
        output_name=zero_if_max_name,
    )

    mask_zero_name = ctx.add_intermediate_tensor(
        f"{output_name}_argmax_zero_mask",
        dtype="FLOAT32",
        shape=input_shape,
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=mask_zero_name,
        shape=input_shape,
        signature=input_signature,
    )
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    if input_dtype in {"FLOAT16", "FLOAT32"}:
        eps_name = ctx.add_const_tensor(
            f"{output_name}_argmax_eps",
            np.asarray([1.0e-6], dtype=np.float32),
        )
        clipped_name = ctx.add_intermediate_tensor(
            f"{output_name}_argmax_clipped",
            dtype="FLOAT32",
            shape=input_shape,
        )
        _set_tensor_shape_signature(
            ctx=ctx,
            tensor_name=clipped_name,
            shape=input_shape,
            signature=input_signature,
        )
        _add_binary_op(
            ctx=ctx,
            op_type="MINIMUM",
            lhs_name=zero_if_max_name,
            rhs_name=eps_name,
            output_name=clipped_name,
        )
        inv_eps_name = ctx.add_const_tensor(
            f"{output_name}_argmax_inv_eps",
            np.asarray([1.0e6], dtype=np.float32),
        )
        _add_binary_op(
            ctx=ctx,
            op_type="MUL",
            lhs_name=clipped_name,
            rhs_name=inv_eps_name,
            output_name=mask_zero_name,
        )
    else:
        one_name = ctx.add_const_tensor(
            f"{output_name}_argmax_one_int",
            np.asarray([1.0], dtype=np.float32),
        )
        _add_binary_op(
            ctx=ctx,
            op_type="MINIMUM",
            lhs_name=zero_if_max_name,
            rhs_name=one_name,
            output_name=mask_zero_name,
        )

    one_name = ctx.add_const_tensor(
        f"{output_name}_argmax_one_final",
        np.asarray([1.0], dtype=np.float32),
    )
    one_if_max_name = ctx.add_intermediate_tensor(
        f"{output_name}_argmax_one_if_max",
        dtype="FLOAT32",
        shape=input_shape,
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=one_if_max_name,
        shape=input_shape,
        signature=input_signature,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=one_name,
        rhs_name=mask_zero_name,
        output_name=one_if_max_name,
    )

    rev_shape = [1] * int(len(input_shape))
    rev_shape[int(axis)] = int(axis_dim)
    rev_index_name = ctx.add_const_tensor(
        f"{output_name}_argmax_rev_index",
        np.arange(int(axis_dim), 0, -1, dtype=np.float32).reshape(rev_shape),
    )
    rev_if_max_name = ctx.add_intermediate_tensor(
        f"{output_name}_argmax_rev_if_max",
        dtype="FLOAT32",
        shape=input_shape,
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=rev_if_max_name,
        shape=input_shape,
        signature=input_signature,
    )
    _add_binary_op(
        ctx=ctx,
        op_type="MUL",
        lhs_name=one_if_max_name,
        rhs_name=rev_index_name,
        output_name=rev_if_max_name,
    )
    reverse_argmax_shape = (
        reduce_keepdims_shape
        if keepdims
        else [int(v) for idx, v in enumerate(input_shape) if int(idx) != int(axis)] or [1]
    )
    reverse_argmax_signature = (
        reduce_keepdims_signature
        if keepdims
        else [int(v) for idx, v in enumerate(input_signature) if int(idx) != int(axis)] or [1]
    )
    reverse_argmax_name = ctx.add_intermediate_tensor(
        f"{output_name}_argmax_reverse",
        dtype="FLOAT32",
        shape=[int(v) if int(v) >= 0 else 1 for v in list(reverse_argmax_shape)],
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=reverse_argmax_name,
        shape=reverse_argmax_shape,
        signature=reverse_argmax_signature,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="REDUCE_MAX",
            inputs=[rev_if_max_name, axis_name],
            outputs=[reverse_argmax_name],
            options={"keepDims": bool(keepdims)},
        )
    )
    reduction_size_name = ctx.add_const_tensor(
        f"{output_name}_argmax_axis_dim",
        np.asarray([float(axis_dim)], dtype=np.float32),
    )
    raw_output_name = output_name
    if str(final_dtype).upper() != "FLOAT32":
        raw_output_name = ctx.add_intermediate_tensor(
            f"{output_name}_argmax_raw",
            dtype="FLOAT32",
            shape=[int(v) if int(v) >= 0 else 1 for v in list(output_shape)],
        )
        _set_tensor_shape_signature(
            ctx=ctx,
            tensor_name=raw_output_name,
            shape=output_shape,
            signature=output_signature,
        )
    _add_binary_op(
        ctx=ctx,
        op_type="SUB",
        lhs_name=reduction_size_name,
        rhs_name=reverse_argmax_name,
        output_name=raw_output_name,
    )
    if str(final_dtype).upper() != "FLOAT32":
        _add_cast_op(
            ctx=ctx,
            input_name=raw_output_name,
            output_name=output_name,
            out_dtype=str(final_dtype).upper(),
            shape=output_shape,
            signature=output_signature,
        )
    if output_tensor is not None:
        output_tensor.dtype = str(final_dtype).upper()
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(output_name)] = str(final_dtype).upper()


def _build_fused_resize_argmax_op(
    *,
    node: Any,
    ctx: Any,
    input_name: str,
    output_name: str,
    keepdims: bool,
    final_dtype: str,
) -> bool:
    restore_info = getattr(ctx, "fused_argmax_restore_shapes", {}).get(str(input_name), None)
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) != 4:
        return False

    producer_name = str(getattr(node.inputs[0], "name", input_name))
    producer_node = getattr(ctx, "onnx_tensor_producers", {}).get(producer_name, None)
    if producer_node is None or str(getattr(producer_node, "op_type", "")) not in {"Resize", "Upsample"}:
        return False

    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    output_shape = (
        [int(v) for v in list(output_tensor.shape)]
        if output_tensor is not None
        else []
    )
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in list(output_shape)]
    )
    if keepdims:
        if len(output_shape) != 4 or int(output_shape[2]) <= 0 or int(output_shape[3]) <= 0:
            return False
        batch_dim = int(output_shape[0])
        original_h = int(output_shape[2])
        original_w = int(output_shape[3])
    else:
        if len(output_shape) != 3 or int(output_shape[1]) <= 0 or int(output_shape[2]) <= 0:
            return False
        batch_dim = int(output_shape[0])
        original_h = int(output_shape[1])
        original_w = int(output_shape[2])

    scale_ratio = float(getattr(ctx, "fused_argmax_scale_ratio", 0.5))
    if isinstance(restore_info, dict):
        restored_original_shape = [int(v) for v in list(restore_info.get("original_shape", []))]
        if len(restored_original_shape) == 4:
            batch_dim = int(restored_original_shape[0])
            channel_dim = int(restored_original_shape[1])
            original_h = int(restored_original_shape[2])
            original_w = int(restored_original_shape[3])
        restored_resized_shape = [int(v) for v in list(restore_info.get("resized_shape", []))]
        target_small_h = int(
            restored_resized_shape[2]
            if len(restored_resized_shape) == 4 and int(restored_resized_shape[2]) > 0
            else max(1, int(float(original_h) * scale_ratio))
        )
        target_small_w = int(
            restored_resized_shape[3]
            if len(restored_resized_shape) == 4 and int(restored_resized_shape[3]) > 0
            else max(1, int(float(original_w) * scale_ratio))
        )
    else:
        target_small_h = max(1, int(float(original_h) * scale_ratio))
        target_small_w = max(1, int(float(original_w) * scale_ratio))
        channel_dim = -1

    layout_nhwc = (
        int(input_shape[0]) == int(batch_dim)
        and int(input_shape[1]) in {int(original_h), int(target_small_h)}
        and int(input_shape[2]) in {int(original_w), int(target_small_w)}
        and (int(channel_dim) <= 0 or int(input_shape[3]) == int(channel_dim))
    )
    layout_nchw = (
        int(input_shape[0]) == int(batch_dim)
        and int(input_shape[2]) in {int(original_h), int(target_small_h)}
        and int(input_shape[3]) in {int(original_w), int(target_small_w)}
        and (int(channel_dim) <= 0 or int(input_shape[1]) == int(channel_dim))
    )
    if not layout_nhwc and not layout_nchw:
        return False

    if int(channel_dim) <= 0:
        channel_dim = int(input_shape[3] if layout_nhwc else input_shape[1])
    original_shape = [int(batch_dim), int(channel_dim), int(original_h), int(original_w)]

    current_nhwc_name = str(input_name)
    current_nhwc_shape = [int(v) for v in list(input_shape)]
    if layout_nhwc:
        current_nhwc_name = str(input_name)
        current_nhwc_shape = [int(v) for v in list(input_shape)]
    elif layout_nchw:
        transposed_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_fused_argmax_input_nhwc",
            dtype=str(ctx.get_tensor_dtype(input_name)).upper(),
            shape=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), int(channel_dim)],
        )
        _set_tensor_shape_signature(
            ctx=ctx,
            tensor_name=transposed_input_name,
            shape=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), int(channel_dim)],
            signature=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), int(channel_dim)],
        )
        current_nhwc_name = make_transpose(
            ctx,
            input_name,
            transposed_input_name,
            [0, 2, 3, 1],
        )
        current_nhwc_shape = [int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), int(channel_dim)]

    if int(current_nhwc_shape[1]) != int(target_small_h) or int(current_nhwc_shape[2]) != int(target_small_w):
        shrunk_input_name = ctx.add_intermediate_tensor(
            f"{output_name}_fused_argmax_shrunk_nhwc",
            dtype=str(ctx.get_tensor_dtype(current_nhwc_name)).upper(),
            shape=[int(current_nhwc_shape[0]), int(target_small_h), int(target_small_w), int(current_nhwc_shape[3])],
        )
        _set_tensor_shape_signature(
            ctx=ctx,
            tensor_name=shrunk_input_name,
            shape=[int(current_nhwc_shape[0]), int(target_small_h), int(target_small_w), int(current_nhwc_shape[3])],
            signature=[int(current_nhwc_shape[0]), int(target_small_h), int(target_small_w), int(current_nhwc_shape[3])],
        )
        shrink_size_name = ctx.add_const_tensor(
            f"{output_name}_fused_argmax_shrink_size",
            np.asarray([int(target_small_h), int(target_small_w)], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESIZE_NEAREST_NEIGHBOR",
                inputs=[current_nhwc_name, shrink_size_name],
                outputs=[shrunk_input_name],
                options={"alignCorners": True, "halfPixelCenters": False},
            )
        )
        current_nhwc_name = shrunk_input_name
        current_nhwc_shape = [int(current_nhwc_shape[0]), int(target_small_h), int(target_small_w), int(current_nhwc_shape[3])]
    axis_name = ctx.add_const_tensor(
        f"{output_name}_fused_argmax_axis",
        np.asarray([3], dtype=np.int32),
    )
    small_argmax_name = ctx.add_intermediate_tensor(
        f"{output_name}_fused_argmax_small",
        dtype="INT32",
        shape=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2])],
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=small_argmax_name,
        shape=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2])],
        signature=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2])],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ARG_MAX",
            inputs=[current_nhwc_name, axis_name],
            outputs=[small_argmax_name],
            options={"outputType": "INT32"},
        )
    )
    small_argmax_f32_name = ctx.add_intermediate_tensor(
        f"{output_name}_fused_argmax_small_f32",
        dtype="FLOAT32",
        shape=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2])],
    )
    _add_cast_op(
        ctx=ctx,
        input_name=small_argmax_name,
        output_name=small_argmax_f32_name,
        out_dtype="FLOAT32",
        shape=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2])],
        signature=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2])],
    )
    small_argmax_nhwc_name = ctx.add_intermediate_tensor(
        f"{output_name}_fused_argmax_small_nhwc",
        dtype="FLOAT32",
        shape=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2]), 1],
    )
    _add_reshape_op(
        ctx=ctx,
        input_name=small_argmax_f32_name,
        output_name=small_argmax_nhwc_name,
        new_shape=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2]), 1],
        signature=[int(current_nhwc_shape[0]), int(current_nhwc_shape[1]), int(current_nhwc_shape[2]), 1],
    )
    restored_nhwc_name = ctx.add_intermediate_tensor(
        f"{output_name}_fused_argmax_restored_nhwc",
        dtype="FLOAT32",
        shape=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), 1],
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=restored_nhwc_name,
        shape=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), 1],
        signature=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), 1],
    )
    restore_size_name = ctx.add_const_tensor(
        f"{output_name}_fused_argmax_restore_size",
        np.asarray([int(original_shape[2]), int(original_shape[3])], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESIZE_NEAREST_NEIGHBOR",
            inputs=[small_argmax_nhwc_name, restore_size_name],
            outputs=[restored_nhwc_name],
            options={"alignCorners": True, "halfPixelCenters": False},
        )
    )
    current_name = restored_nhwc_name
    current_shape = [int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), 1]
    current_signature = [int(original_shape[0]), int(original_shape[2]), int(original_shape[3]), 1]
    if keepdims:
        keepdims_name = ctx.add_intermediate_tensor(
            f"{output_name}_fused_argmax_keepdims",
            dtype="FLOAT32",
            shape=[int(original_shape[0]), 1, int(original_shape[2]), int(original_shape[3])],
        )
        _set_tensor_shape_signature(
            ctx=ctx,
            tensor_name=keepdims_name,
            shape=[int(original_shape[0]), 1, int(original_shape[2]), int(original_shape[3])],
            signature=[int(original_shape[0]), 1, int(original_shape[2]), int(original_shape[3])],
        )
        current_name = make_transpose(
            ctx,
            current_name,
            keepdims_name,
            [0, 3, 1, 2],
        )
        current_shape = [int(original_shape[0]), 1, int(original_shape[2]), int(original_shape[3])]
        current_signature = [int(original_shape[0]), 1, int(original_shape[2]), int(original_shape[3])]
    else:
        squeezed_name = ctx.add_intermediate_tensor(
            f"{output_name}_fused_argmax_squeezed",
            dtype="FLOAT32",
            shape=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3])],
        )
        _add_reshape_op(
            ctx=ctx,
            input_name=current_name,
            output_name=squeezed_name,
            new_shape=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3])],
            signature=[int(original_shape[0]), int(original_shape[2]), int(original_shape[3])],
        )
        current_name = squeezed_name
        current_shape = [int(original_shape[0]), int(original_shape[2]), int(original_shape[3])]
        current_signature = [int(original_shape[0]), int(original_shape[2]), int(original_shape[3])]

    if str(final_dtype).upper() != "FLOAT32":
        _add_cast_op(
            ctx=ctx,
            input_name=current_name,
            output_name=output_name,
            out_dtype=str(final_dtype).upper(),
            shape=output_shape,
            signature=output_signature,
        )
    else:
        _add_reshape_op(
            ctx=ctx,
            input_name=current_name,
            output_name=output_name,
            new_shape=current_shape,
            signature=current_signature,
        )
        ctx.model_ir.tensors[output_name].dtype = "FLOAT32"
        _set_tensor_shape_signature(
            ctx=ctx,
            tensor_name=output_name,
            shape=output_shape,
            signature=output_signature,
        )
    if output_tensor is not None:
        output_tensor.dtype = str(final_dtype).upper()
    if hasattr(ctx, "dtype_map") and isinstance(ctx.dtype_map, dict):
        ctx.dtype_map[str(output_name)] = str(final_dtype).upper()
    return True


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
    argmax_mode = str(getattr(ctx, "argmax_mode", "none"))
    output_dtype = _prefer_int32_index_output_dtype(
        ctx=ctx,
        tensor_name=output_name,
        requested_dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
    )

    if argmax_mode == "reducemax_int64":
        _build_reducemax_argmax_op(
            node=node,
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
            axis=axis,
            keepdims=keepdims,
            final_dtype="INT64",
        )
        return
    if argmax_mode == "reducemax_float32":
        _build_reducemax_argmax_op(
            node=node,
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
            axis=axis,
            keepdims=keepdims,
            final_dtype="FLOAT32",
        )
        return
    if argmax_mode == "fused_int64" and int(axis) == 1:
        if _build_fused_resize_argmax_op(
            node=node,
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
            keepdims=keepdims,
            final_dtype="INT64",
        ):
            return
    if argmax_mode == "fused_float32" and int(axis) == 1:
        if _build_fused_resize_argmax_op(
            node=node,
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
            keepdims=keepdims,
            final_dtype="FLOAT32",
        ):
            return

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
    output_dtype = _prefer_int32_index_output_dtype(
        ctx=ctx,
        tensor_name=output_name,
        requested_dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
    )

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
    indices_output_dtype = ""
    if indices_output_name != "":
        ctx.ensure_tensor(indices_output_name)
        indices_output_dtype = _prefer_int32_index_output_dtype(
            ctx=ctx,
            tensor_name=indices_output_name,
            requested_dtype=str(ctx.get_tensor_dtype(indices_output_name)).upper(),
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_rank = int(len(input_shape))
    axis = _normalize_axis_for_rank(int(node.attrs.get("axis", -1)), input_rank)
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    input_signature = (
        [int(v) for v in list(input_tensor.shape_signature)]
        if input_tensor is not None and input_tensor.shape_signature is not None
        else [int(v) for v in list(input_shape)]
    )
    largest = bool(int(node.attrs.get("largest", 1)))
    sorted_attr = int(node.attrs.get("sorted", 1))
    # ONNX TopK(sorted=0) means output order is unspecified.
    # TFLite TOPK_V2 always emits sorted output, which still satisfies
    # the relaxed ONNX(sorted=0) requirement.
    if sorted_attr == 0:
        warn(
            "TopK(sorted=0) lowered to TFLite TOPK_V2 always returns descending-sorted output. "
            "Index order will not exactly match ONNX reference. "
            f"node={node.name}",
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

    k_for_topk_name = _maybe_constantize_topk_k(
        ctx=ctx,
        node=node,
        k_input_name=k_name,
        values_output_name=values_output_name,
    )
    if k_for_topk_name is None:
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
    runtime_dynamic_k = ctx.get_constant_array(k_for_topk_name) is None
    static_k_value = None if runtime_dynamic_k else _resolve_static_topk_k_value(ctx, k_for_topk_name)
    corrected_output_shape, corrected_output_signature = _infer_topk_output_shape_and_signature(
        input_shape=input_shape,
        input_signature=input_signature,
        axis=axis,
        static_k_value=static_k_value,
    )
    _set_tensor_shape_signature(
        ctx=ctx,
        tensor_name=values_output_name,
        shape=corrected_output_shape,
        signature=corrected_output_signature,
    )
    if indices_output_name != "":
        _set_tensor_shape_signature(
            ctx=ctx,
            tensor_name=indices_output_name,
            shape=corrected_output_shape,
            signature=corrected_output_signature,
        )

    topk_values_shape = (
        [int(corrected_output_shape[int(v)]) for v in perm_to_last]
        if perm_to_last is not None and len(values_output_shape) == len(perm_to_last)
        else [int(v) for v in corrected_output_shape]
    )
    topk_values_signature = (
        [int(corrected_output_signature[int(v)]) for v in perm_to_last]
        if perm_to_last is not None and len(corrected_output_signature) == len(perm_to_last)
        else [int(v) for v in corrected_output_signature]
    )
    topk_indices_shape = (
        [int(corrected_output_shape[int(v)]) for v in perm_to_last]
        if perm_to_last is not None and len(indices_output_shape) == len(perm_to_last)
        else [int(v) for v in corrected_output_shape]
    )
    topk_indices_signature = (
        [int(corrected_output_signature[int(v)]) for v in perm_to_last]
        if perm_to_last is not None and len(corrected_output_signature) == len(perm_to_last)
        else [int(v) for v in corrected_output_signature]
    )
    if runtime_dynamic_k:
        if len(topk_values_signature) > 0:
            topk_values_signature[-1] = -1
        if len(topk_indices_signature) > 0:
            topk_indices_signature[-1] = -1
        if int(axis) < len(corrected_output_signature):
            corrected_output_signature[int(axis)] = -1
        if values_output_tensor is not None:
            values_output_tensor.shape_signature = [int(v) for v in corrected_output_signature]
        if indices_output_tensor is not None:
            indices_output_tensor.shape_signature = [int(v) for v in corrected_output_signature]

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
            if perm_from_last is None and indices_output_dtype == "INT32"
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
            and indices_output_dtype == "INT32"
            else ctx.add_intermediate_tensor(
                f"{values_output_name}_topk_indices_axis_restored",
                dtype="INT32",
                shape=corrected_output_shape,
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
                indices_transposed_tensor.shape_signature = [int(v) for v in corrected_output_signature]
        indices_final_i32_name = indices_transposed_name

    if indices_output_name != "":
        indices_dtype = indices_output_dtype
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
        elif indices_final_i32_name != indices_output_name:
            shape_name = ctx.add_const_tensor(
                f"{indices_output_name}_topk_indices_identity_shape",
                np.asarray([int(v) for v in corrected_output_shape], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[indices_final_i32_name, shape_name],
                    outputs=[indices_output_name],
                    options={"newShape": [int(v) for v in corrected_output_shape]},
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
    output_dtype = _prefer_int32_index_output_dtype(
        ctx=ctx,
        tensor_name=output_name,
        requested_dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
    )
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
    def _rank_is_unknown_placeholder(tensor_name: str, shape: list[int]) -> bool:
        raw_shape = None
        if hasattr(ctx, "shape_map"):
            raw_shape = ctx.shape_map.get(str(tensor_name), None)
        if isinstance(raw_shape, (list, tuple)) and len(list(raw_shape)) > 0:
            # Rank is known even when dimensions are symbolic/unknown.
            return False
        return bool(
            len(shape) == 1
            and _is_unknown_rank_placeholder_tensor(ctx, tensor_name)
        )

    data_name = node.inputs[0].name
    indices_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(data_name)
    ctx.ensure_tensor(indices_name)
    ctx.ensure_tensor(output_name)

    data_shape = [int(v) for v in ctx.get_tensor_shape(data_name)]
    indices_shape = [int(v) for v in ctx.get_tensor_shape(indices_name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor is not None and output_tensor.shape_signature is not None
        else [int(v) for v in list(output_shape)]
    )
    data_tensor = ctx.model_ir.tensors.get(data_name, None)
    data_signature = (
        [int(v) for v in list(data_tensor.shape_signature)]
        if data_tensor is not None and data_tensor.shape_signature is not None
        else [int(v) for v in list(data_shape)]
    )
    indices_tensor = ctx.model_ir.tensors.get(indices_name, None)
    indices_signature = (
        [int(v) for v in list(indices_tensor.shape_signature)]
        if indices_tensor is not None and indices_tensor.shape_signature is not None
        else [int(v) for v in list(indices_shape)]
    )
    data_rank_unknown = _rank_is_unknown_placeholder(data_name, data_shape)
    indices_rank_unknown = _rank_is_unknown_placeholder(indices_name, indices_shape)
    output_rank_unknown = _rank_is_unknown_placeholder(output_name, output_shape)

    if data_rank_unknown and not indices_rank_unknown:
        data_shape = [int(v) for v in list(indices_shape)]
        data_signature = [int(v) for v in list(indices_signature)]
        if data_tensor is not None:
            data_tensor.shape = [int(v) for v in list(data_shape)]
            data_tensor.shape_signature = [int(v) for v in list(data_signature)]
        data_rank_unknown = False
    elif data_rank_unknown and not output_rank_unknown:
        data_shape = [int(v) for v in list(output_shape)]
        data_signature = [int(v) for v in list(output_signature)]
        if data_tensor is not None:
            data_tensor.shape = [int(v) for v in list(data_shape)]
            data_tensor.shape_signature = [int(v) for v in list(data_signature)]
        data_rank_unknown = False

    if indices_rank_unknown and not output_rank_unknown:
        indices_shape = [int(v) for v in list(output_shape)]
        indices_signature = [int(v) for v in list(output_signature)]
        if indices_tensor is not None:
            indices_tensor.shape = [int(v) for v in list(indices_shape)]
            indices_tensor.shape_signature = [int(v) for v in list(indices_signature)]
        indices_rank_unknown = False
    replace_to_pseudo_operators = getattr(ctx, "replace_to_pseudo_operators", set())
    rtpo_gathernd = "gathernd" in set(replace_to_pseudo_operators or set())
    if len(data_shape) != len(indices_shape) and not data_rank_unknown and not indices_rank_unknown:
        raise NotImplementedError(
            "GatherElements requires data and indices with the same rank in flatbuffer_direct. "
            f"op={node.name} data_shape={data_shape} indices_shape={indices_shape}"
        )
    if len(indices_shape) != len(output_shape) and not indices_rank_unknown and not output_rank_unknown:
        raise NotImplementedError(
            "GatherElements requires output rank equal to indices rank in flatbuffer_direct. "
            f"op={node.name} indices_shape={indices_shape} output_shape={output_shape}"
        )
    if data_rank_unknown and indices_rank_unknown and output_rank_unknown:
        raise NotImplementedError(
            "GatherElements requires resolvable rank in flatbuffer_direct. "
            f"op={node.name} data_shape={data_shape} indices_shape={indices_shape} output_shape={output_shape}"
        )
    if any(int(v) <= 0 for v in data_shape):
        raise NotImplementedError(
            "GatherElements requires fully static positive data shape in flatbuffer_direct. "
            f"op={node.name} data_shape={data_shape}"
        )

    rank = len(data_shape)
    if data_rank_unknown and not indices_rank_unknown:
        rank = len(indices_shape)
    elif data_rank_unknown and not output_rank_unknown:
        rank = len(output_shape)
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

    axis_coord_signature = [int(v) for v in output_signature] + [1]
    axis_coord_shape = [int(v) if int(v) > 0 else 1 for v in list(axis_coord_signature)]
    axis_coord_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_elements_axis_coord",
        dtype="INT32",
        shape=axis_coord_shape,
    )
    axis_coord_tensor = ctx.model_ir.tensors.get(axis_coord_name, None)
    if axis_coord_tensor is not None:
        axis_coord_tensor.shape_signature = [int(v) for v in axis_coord_signature]
    expand_axis_const = ctx.add_const_tensor(
        f"{output_name}_gather_elements_axis_coord_expand_axis",
        np.asarray(-1, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="EXPAND_DIMS",
            inputs=[indices_i32_name, expand_axis_const],
            outputs=[axis_coord_name],
        )
    )

    axis_is_dynamic = int(output_signature[axis]) < 0
    static_grid_shape = [int(v) for v in output_shape]
    if axis_is_dynamic:
        static_grid_shape[axis] = 1
    grid = np.indices(static_grid_shape, dtype=np.int32)
    coord_tensors: list[str] = []
    for dim in range(rank):
        if dim == axis:
            coord_tensors.append(axis_coord_name)
            continue
        if axis_is_dynamic:
            dim_signature = int(output_signature[dim]) if dim < len(output_signature) else int(output_shape[dim])
            dim_shape = int(output_shape[dim])
            if dim_signature == 1 or dim_shape == 1:
                coord_zero_name = ctx.add_intermediate_tensor(
                    f"{output_name}_gather_elements_coord_{dim}_zeros",
                    dtype="INT32",
                    shape=[int(v) for v in axis_coord_shape],
                )
                coord_zero_tensor = ctx.model_ir.tensors.get(coord_zero_name, None)
                if coord_zero_tensor is not None:
                    coord_zero_tensor.shape_signature = [int(v) for v in axis_coord_signature]
                ctx.add_operator(
                    OperatorIR(
                        op_type="SUB",
                        inputs=[axis_coord_name, axis_coord_name],
                        outputs=[coord_zero_name],
                        options={"fusedActivationFunction": "NONE"},
                    )
                )
                coord_tensors.append(coord_zero_name)
                continue
            raise NotImplementedError(
                "GatherElements with dynamic gather-axis currently supports only non-axis dimensions with size=1. "
                f"op={node.name} axis={axis} dim={dim} output_shape={output_shape} output_signature={output_signature}"
            )
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
        coords_tensor = ctx.model_ir.tensors.get(coords_name, None)
        if coords_tensor is not None:
            coords_tensor.shape_signature = [int(v) for v in output_signature] + [int(rank)]
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
    else:
        coords_tensor = ctx.model_ir.tensors.get(coords_name, None)
        if coords_tensor is not None:
            coords_tensor.shape_signature = [int(v) for v in output_signature] + [int(rank)]

    if not rtpo_gathernd:
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER_ND",
                inputs=[data_name, coords_name],
                outputs=[output_name],
            )
        )
        return

    # -rtpo GatherND: also pseudo-lower GatherElements-generated GATHER_ND path.
    data_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_elements_data_flat",
        dtype=str(ctx.get_tensor_dtype(data_name)).upper(),
        shape=[int(np.prod(np.asarray(data_shape, dtype=np.int64)))],
    )
    data_flat_shape_const = ctx.add_const_tensor(
        f"{output_name}_gather_elements_data_flat_shape",
        np.asarray([-1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[data_name, data_flat_shape_const],
            outputs=[data_flat_name],
            options={"newShape": [-1], "preserveDynamicShape": True},
        )
    )

    axis_steps = []
    for dim in range(rank):
        step = 1
        for next_dim in range(dim + 1, rank):
            step *= int(data_shape[next_dim])
        axis_steps.append(int(step))
    linear_signature = [int(v) for v in output_signature] + [1]
    linear_shape = [int(v) if int(v) > 0 else 1 for v in linear_signature]
    linear_acc_name: str | None = None
    for dim, coord_name in enumerate(coord_tensors):
        term_name = str(coord_name)
        step = int(axis_steps[dim])
        if step != 1:
            coord_const_arr = ctx.get_constant_array(term_name)
            if coord_const_arr is not None:
                # LiteRT.js WebGPU can reject MUL when both inputs are const.
                # Pre-fold const*const here and keep only one const tensor.
                term_name = ctx.add_const_tensor(
                    f"{output_name}_gather_elements_linear_mul_{dim}_const",
                    np.asarray(coord_const_arr, dtype=np.int32) * np.int32(step),
                )
            else:
                step_const_name = ctx.add_const_tensor(
                    f"{output_name}_gather_elements_axis_step_{dim}",
                    np.asarray(step, dtype=np.int32),
                )
                term_mul_name = ctx.add_intermediate_tensor(
                    f"{output_name}_gather_elements_linear_mul_{dim}",
                    dtype="INT32",
                    shape=[int(v) for v in linear_shape],
                )
                term_mul_tensor = ctx.model_ir.tensors.get(term_mul_name, None)
                if term_mul_tensor is not None:
                    term_mul_tensor.shape_signature = [int(v) for v in linear_signature]
                ctx.add_operator(
                    OperatorIR(
                        op_type="MUL",
                        inputs=[term_name, step_const_name],
                        outputs=[term_mul_name],
                        options={"fusedActivationFunction": "NONE"},
                    )
                )
                term_name = str(term_mul_name)

        if linear_acc_name is None:
            linear_acc_name = term_name
        else:
            acc_const_arr = ctx.get_constant_array(linear_acc_name)
            term_const_arr = ctx.get_constant_array(term_name)
            if acc_const_arr is not None and term_const_arr is not None:
                # LiteRT.js WebGPU can reject ADD with two const inputs.
                linear_acc_name = ctx.add_const_tensor(
                    f"{output_name}_gather_elements_linear_add_{dim}_const",
                    np.asarray(acc_const_arr, dtype=np.int32)
                    + np.asarray(term_const_arr, dtype=np.int32),
                )
            else:
                linear_add_name = ctx.add_intermediate_tensor(
                    f"{output_name}_gather_elements_linear_add_{dim}",
                    dtype="INT32",
                    shape=[int(v) for v in linear_shape],
                )
                linear_add_tensor = ctx.model_ir.tensors.get(linear_add_name, None)
                if linear_add_tensor is not None:
                    linear_add_tensor.shape_signature = [int(v) for v in linear_signature]
                ctx.add_operator(
                    OperatorIR(
                        op_type="ADD",
                        inputs=[linear_acc_name, term_name],
                        outputs=[linear_add_name],
                        options={"fusedActivationFunction": "NONE"},
                    )
                )
                linear_acc_name = str(linear_add_name)

    if linear_acc_name is None:
        raise NotImplementedError(
            f"GatherElements pseudo gathernd path failed to build linear indices. op={node.name}"
        )

    linear_index_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_elements_linear_index",
        dtype="INT32",
        shape=[int(v) for v in output_shape],
    )
    linear_index_tensor = ctx.model_ir.tensors.get(linear_index_name, None)
    if linear_index_tensor is not None:
        linear_index_tensor.shape_signature = [int(v) for v in output_signature]
    linear_index_shape_const = ctx.add_const_tensor(
        f"{output_name}_gather_elements_linear_index_shape",
        np.asarray([int(v) for v in output_shape], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[linear_acc_name, linear_index_shape_const],
            outputs=[linear_index_name],
            options={
                "newShape": [int(v) for v in output_shape],
                "preserveDynamicShape": True,
            },
        )
    )

    linear_index_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_elements_linear_index_flat",
        dtype="INT32",
        shape=[
            int(
                max(
                    1,
                    int(
                        np.prod(
                            np.asarray([int(v) for v in output_shape], dtype=np.int64)
                        )
                    ),
                )
            )
        ],
    )
    linear_index_flat_tensor = ctx.model_ir.tensors.get(linear_index_flat_name, None)
    if linear_index_flat_tensor is not None:
        flat_sig = -1 if any(int(v) < 0 for v in output_signature) else int(
            np.prod(np.asarray([int(v) for v in output_signature], dtype=np.int64))
        )
        linear_index_flat_tensor.shape_signature = [int(flat_sig)]
        linear_index_flat_tensor.shape = [1 if int(flat_sig) < 0 else int(flat_sig)]
    linear_index_flat_shape_const = ctx.add_const_tensor(
        f"{output_name}_gather_elements_linear_index_flat_shape",
        np.asarray([-1], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[linear_index_name, linear_index_flat_shape_const],
            outputs=[linear_index_flat_name],
            options={
                "newShape": [-1],
                "preserveDynamicShape": True,
            },
        )
    )

    gathered_flat_name = ctx.add_intermediate_tensor(
        f"{output_name}_gather_elements_gathered_flat",
        dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
        shape=[
            int(
                max(
                    1,
                    int(
                        np.prod(
                            np.asarray([int(v) for v in output_shape], dtype=np.int64)
                        )
                    ),
                )
            )
        ],
    )
    gathered_flat_tensor = ctx.model_ir.tensors.get(gathered_flat_name, None)
    if gathered_flat_tensor is not None:
        flat_sig = -1 if any(int(v) < 0 for v in output_signature) else int(
            np.prod(np.asarray([int(v) for v in output_signature], dtype=np.int64))
        )
        gathered_flat_tensor.shape_signature = [int(flat_sig)]
        gathered_flat_tensor.shape = [1 if int(flat_sig) < 0 else int(flat_sig)]
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[data_flat_name, linear_index_flat_name],
            outputs=[gathered_flat_name],
            options={"axis": 0},
        )
    )

    gathered_out_shape_const = ctx.add_const_tensor(
        f"{output_name}_gather_elements_gathered_out_shape",
        np.asarray([int(v) for v in output_shape], dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[gathered_flat_name, gathered_out_shape_const],
            outputs=[output_name],
            options={
                "newShape": [int(v) for v in output_shape],
                "preserveDynamicShape": True,
            },
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
    boxes_tensor = ctx.model_ir.tensors[boxes_name]
    boxes_signature = (
        [int(v) for v in list(boxes_tensor.shape_signature)]
        if boxes_tensor.shape_signature is not None
        else [int(v) for v in boxes_shape]
    )
    output_nms_with_argmax = bool(getattr(ctx, "output_nms_with_argmax", False))
    switch_nms_version = str(getattr(ctx, "switch_nms_version", "v4")).strip().lower()
    if switch_nms_version not in {"v4", "v5"}:
        switch_nms_version = "v4"
    use_nms_v5 = switch_nms_version == "v5"
    output_dtype = _prefer_int32_index_output_dtype(
        ctx=ctx,
        tensor_name=output_name,
        requested_dtype=str(ctx.get_tensor_dtype(output_name)).upper(),
    )
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

    num_boxes_static = None
    if len(boxes_signature) >= 2:
        # Prefer shape_signature for staticness. `shape` may contain placeholder 1
        # for unknown branch-local tensors, which must not be treated as static.
        if int(boxes_signature[1]) > 0:
            num_boxes_static = int(boxes_signature[1])
    elif len(boxes_shape) >= 2 and int(boxes_shape[1]) > 0:
        num_boxes_static = int(boxes_shape[1])
    num_boxes_dim = int(num_boxes_static) if num_boxes_static is not None else -1

    boxes_2d_name = ctx.add_intermediate_tensor(
        f"{output_name}_nms_boxes_2d",
        dtype=str(ctx.get_tensor_dtype(boxes_name)).upper(),
        shape=[num_boxes_dim, 4],
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
            shape=[1, num_boxes_dim],
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
            shape=[num_boxes_dim],
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
            shape=[1, 1, num_boxes_dim],
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
        shape=[num_boxes_dim],
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

    max_output_const = None
    if len(node.inputs) >= 3:
        max_output_arr = ctx.get_constant_array(node.inputs[2].name)
        if max_output_arr is not None:
            max_output_flat = np.asarray(max_output_arr).reshape(-1)
            if max_output_flat.size > 0:
                max_output_const = int(max_output_flat[0])

    max_output_static = None
    if num_boxes_static is not None:
        if max_output_const is None or int(max_output_const) <= 0:
            max_output_static = int(num_boxes_static)
        else:
            max_output_static = int(min(int(max_output_const), int(num_boxes_static)))
        max_output_size_name = ctx.add_const_tensor(
            f"{output_name}_nms_max_output_size",
            np.asarray(int(max_output_static), dtype=np.int32),
        )
        max_output_tensor = ctx.model_ir.tensors.get(max_output_size_name, None)
        if max_output_tensor is not None:
            max_output_tensor.shape = []
            max_output_tensor.shape_signature = []
    else:
        boxes_runtime_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_boxes_runtime_shape",
            dtype="INT32",
            shape=[2],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[boxes_2d_name],
                outputs=[boxes_runtime_shape_name],
                options={"outType": "INT32"},
            )
        )
        num_boxes_index_name = ctx.add_const_tensor(
            f"{output_name}_nms_num_boxes_index",
            np.asarray([0], dtype=np.int32),
        )
        num_boxes_vector_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_num_boxes_vector",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[boxes_runtime_shape_name, num_boxes_index_name],
                outputs=[num_boxes_vector_name],
                options={"axis": 0, "batchDims": 0},
            )
        )
        num_boxes_scalar_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_num_boxes_scalar",
            dtype="INT32",
            shape=[],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[num_boxes_vector_name],
                outputs=[num_boxes_scalar_name],
                options={"squeezeDims": [0]},
            )
        )
        num_boxes_scalar_tensor = ctx.model_ir.tensors.get(num_boxes_scalar_name, None)
        if num_boxes_scalar_tensor is not None:
            num_boxes_scalar_tensor.shape = []
            num_boxes_scalar_tensor.shape_signature = []

        if max_output_const is None or int(max_output_const) <= 0:
            max_output_size_name = num_boxes_scalar_name
        else:
            clipped_max_output = int(min(int(max_output_const), int(np.iinfo(np.int32).max)))
            max_output_const_name = ctx.add_const_tensor(
                f"{output_name}_nms_max_output_size_cap",
                np.asarray(clipped_max_output, dtype=np.int32),
            )
            max_output_const_tensor = ctx.model_ir.tensors.get(max_output_const_name, None)
            if max_output_const_tensor is not None:
                max_output_const_tensor.shape = []
                max_output_const_tensor.shape_signature = []
            max_output_size_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_max_output_size",
                dtype="INT32",
                shape=[],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MINIMUM",
                    inputs=[max_output_const_name, num_boxes_scalar_name],
                    outputs=[max_output_size_name],
                )
            )
            max_output_size_tensor = ctx.model_ir.tensors.get(max_output_size_name, None)
            if max_output_size_tensor is not None:
                max_output_size_tensor.shape = []
                max_output_size_tensor.shape_signature = []

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

    soft_nms_sigma_name = None
    if use_nms_v5:
        soft_nms_sigma_name = ctx.add_const_tensor(
            f"{output_name}_nms_soft_nms_sigma",
            np.asarray(0.0, dtype=np.float32),
        )
        soft_nms_sigma_tensor = ctx.model_ir.tensors.get(soft_nms_sigma_name, None)
        if soft_nms_sigma_tensor is not None:
            soft_nms_sigma_tensor.shape = []
            soft_nms_sigma_tensor.shape_signature = []

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
            shape=[int(max_output_static)] if max_output_static is not None else [-1],
        )
        nms_valid_count_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_valid_count{suffix}",
            dtype="INT32",
            shape=[],
        )
        nms_valid_count_tensor = ctx.model_ir.tensors.get(nms_valid_count_name, None)
        if nms_valid_count_tensor is not None:
            # NMS valid_count is scalar in LiteRT runtime. Keep rank-0 metadata so
            # downstream RESHAPE([1]) is preserved and SLICE size receives rank-1.
            nms_valid_count_tensor.shape = []
            nms_valid_count_tensor.shape_signature = []
        nms_inputs = [
            boxes_2d_name,
            class_scores_1d_name,
            max_output_size_name,
            iou_threshold_name,
            score_threshold_name,
        ]
        nms_outputs = [nms_selected_indices_name]
        if use_nms_v5:
            nms_selected_scores_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_selected_scores{suffix}",
                dtype=str(ctx.get_tensor_dtype(class_scores_1d_name)).upper(),
                shape=[int(max_output_static)] if max_output_static is not None else [-1],
            )
            nms_outputs.append(nms_selected_scores_name)
            nms_inputs.append(str(soft_nms_sigma_name))
        nms_outputs.append(nms_valid_count_name)
        ctx.add_operator(
            OperatorIR(
                op_type="NON_MAX_SUPPRESSION_V5" if use_nms_v5 else "NON_MAX_SUPPRESSION_V4",
                inputs=nms_inputs,
                outputs=nms_outputs,
            )
        )

        selected_indices_valid_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_selected_indices_valid{suffix}",
            dtype="INT32",
            shape=[-1],
        )
        valid_indices_range_start_name = ctx.add_const_tensor(
            f"{output_name}_nms_valid_indices_range_start{suffix}",
            np.asarray(0, dtype=np.int32),
        )
        valid_indices_range_start_tensor = ctx.model_ir.tensors.get(
            valid_indices_range_start_name,
            None,
        )
        if valid_indices_range_start_tensor is not None:
            valid_indices_range_start_tensor.shape = []
            valid_indices_range_start_tensor.shape_signature = []
        valid_indices_range_delta_name = ctx.add_const_tensor(
            f"{output_name}_nms_valid_indices_range_delta{suffix}",
            np.asarray(1, dtype=np.int32),
        )
        valid_indices_range_delta_tensor = ctx.model_ir.tensors.get(
            valid_indices_range_delta_name,
            None,
        )
        if valid_indices_range_delta_tensor is not None:
            valid_indices_range_delta_tensor.shape = []
            valid_indices_range_delta_tensor.shape_signature = []
        valid_indices_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_valid_indices{suffix}",
            dtype="INT32",
            shape=[-1],
        )
        nms_valid_count_scalar_shape_name = ctx.add_const_tensor(
            f"{output_name}_nms_valid_count_scalar_shape{suffix}",
            np.asarray([], dtype=np.int32),
        )
        nms_valid_count_scalar_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_valid_count_scalar{suffix}",
            dtype="INT32",
            shape=[],
        )
        nms_valid_count_scalar_tensor = ctx.model_ir.tensors.get(nms_valid_count_scalar_name, None)
        if nms_valid_count_scalar_tensor is not None:
            nms_valid_count_scalar_tensor.shape = []
            nms_valid_count_scalar_tensor.shape_signature = []
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[nms_valid_count_name, nms_valid_count_scalar_shape_name],
                outputs=[nms_valid_count_scalar_name],
                options={
                    "newShape": [],
                    "preserveDynamicShape": True,
                },
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RANGE",
                inputs=[
                    valid_indices_range_start_name,
                    nms_valid_count_scalar_name,
                    valid_indices_range_delta_name,
                ],
                outputs=[valid_indices_name],
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[nms_selected_indices_name, valid_indices_name],
                outputs=[selected_indices_valid_name],
                options={
                    "axis": 0,
                    "batchDims": 0,
                },
            )
        )

        selected_indices_col_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_selected_indices_col{suffix}",
            dtype="INT32",
            shape=[-1, 1],
        )
        selected_indices_valid_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_selected_indices_valid_shape{suffix}",
            dtype="INT32",
            shape=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[selected_indices_valid_name],
                outputs=[selected_indices_valid_shape_name],
                options={"outType": "INT32"},
            )
        )
        selected_indices_col_tail_dim_name = ctx.add_const_tensor(
            f"{output_name}_nms_selected_indices_col_tail_dim{suffix}",
            np.asarray([1], dtype=np.int32),
        )
        selected_indices_col_shape_name = ctx.add_intermediate_tensor(
            f"{output_name}_nms_selected_indices_col_shape{suffix}",
            dtype="INT32",
            shape=[2],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[selected_indices_valid_shape_name, selected_indices_col_tail_dim_name],
                outputs=[selected_indices_col_shape_name],
                options={
                    "axis": 0,
                    "fusedActivationFunction": "NONE",
                },
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[selected_indices_valid_name, selected_indices_col_shape_name],
                outputs=[selected_indices_col_name],
                options={},
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
            selected_class_ids_valid_shape_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_selected_class_ids_shape{suffix}",
                dtype="INT32",
                shape=[1],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SHAPE",
                    inputs=[selected_class_ids_valid_name],
                    outputs=[selected_class_ids_valid_shape_name],
                    options={"outType": "INT32"},
                )
            )
            class_ids_col_tail_dim_name = ctx.add_const_tensor(
                f"{output_name}_nms_selected_class_ids_col_tail_dim{suffix}",
                np.asarray([1], dtype=np.int32),
            )
            class_ids_col_shape_name = ctx.add_intermediate_tensor(
                f"{output_name}_nms_selected_class_ids_col_shape{suffix}",
                dtype="INT32",
                shape=[2],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=[selected_class_ids_valid_shape_name, class_ids_col_tail_dim_name],
                    outputs=[class_ids_col_shape_name],
                    options={
                        "axis": 0,
                        "fusedActivationFunction": "NONE",
                    },
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[selected_class_ids_valid_name, class_ids_col_shape_name],
                    outputs=[class_ids_col_name],
                    options={},
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
            shape=[num_boxes_dim],
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
                shape=[num_boxes_dim],
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
                shape=[num_classes, num_boxes_dim],
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
                    shape=[1, num_boxes_dim],
                )
                class_begin_name = ctx.add_const_tensor(
                    f"{output_name}_nms_scores_begin{class_suffix}",
                    np.asarray([int(class_idx), 0], dtype=np.int32),
                )
                class_size_name = ctx.add_const_tensor(
                    f"{output_name}_nms_scores_size{class_suffix}",
                    np.asarray([1, int(num_boxes_static) if num_boxes_static is not None else -1], dtype=np.int32),
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
                    shape=[num_boxes_dim],
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
    if str(output_name) in set(ctx.graph_output_names):
        dynamic_outputs = ctx.model_ir.metadata.setdefault(
            "onnx_dynamic_output_tensor_names",
            [],
        )
        if isinstance(dynamic_outputs, list) and str(output_name) not in set(
            str(v) for v in dynamic_outputs
        ):
            # NMS selected-indices length is runtime-dependent even when ONNX
            # declares a static max_output upper bound.
            dynamic_outputs.append(str(output_name))
