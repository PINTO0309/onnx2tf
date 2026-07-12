from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def add_rank_safe_negative_index_normalization(
    *,
    ctx: Any,
    indices_name: str,
    shape_prefix_name: str,
    name_prefix: str,
    indices_shape: list[int],
    indices_signature: list[int],
) -> str:
    """Normalize dynamic negative ScatterND indices without rank>4 broadcasts.

    TFLite's broadcast comparison implementation is limited to four dimensions.
    ScatterND indices may legally have any number of leading batch dimensions, so
    temporarily coalesce those dimensions while preserving the final index-vector
    dimension. The normalized indices are reshaped back before SCATTER_ND.
    """
    rank = int(len(indices_signature))
    if rank <= 0 or rank != int(len(indices_shape)):
        raise ValueError(
            "ScatterND indices must have a known positive rank. "
            f"shape={indices_shape} signature={indices_signature}"
        )
    k_dim = int(indices_signature[-1])
    if k_dim <= 0:
        raise ValueError(
            "ScatterND index-vector dimension must be positive. "
            f"signature={indices_signature}"
        )

    working_indices_name = str(indices_name)
    working_signature = [int(v) for v in indices_signature]
    restore_shape_name = ""
    if rank > 4:
        restore_shape_name = ctx.add_intermediate_tensor(
            f"{name_prefix}_original_shape",
            dtype="INT32",
            shape=[rank],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SHAPE",
                inputs=[indices_name],
                outputs=[restore_shape_name],
                options={"outType": "INT32"},
            )
        )
        flattened_shape_name = ctx.add_const_tensor(
            f"{name_prefix}_flattened_shape",
            np.asarray([-1, k_dim], dtype=np.int32),
        )
        working_indices_name = ctx.add_intermediate_tensor(
            f"{name_prefix}_flattened",
            dtype="INT32",
            shape=[-1, k_dim],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[indices_name, flattened_shape_name],
                outputs=[working_indices_name],
                options={
                    "newShape": [-1, k_dim],
                    "preserveDynamicShape": True,
                },
            )
        )
        working_signature = [-1, k_dim]

    zero_name = ctx.add_const_tensor(
        f"{name_prefix}_zero_i32",
        np.asarray(0, dtype=np.int32),
    )
    negative_mask_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_negative_mask",
        dtype="BOOL",
        shape=working_signature,
    )
    indices_plus_shape_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_plus_shape",
        dtype="INT32",
        shape=working_signature,
    )
    wrapped_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_wrapped",
        dtype="INT32",
        shape=working_signature,
    )
    normalized_working_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_normalized_working",
        dtype="INT32",
        shape=working_signature,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LESS",
            inputs=[working_indices_name, zero_name],
            outputs=[negative_mask_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[working_indices_name, shape_prefix_name],
            outputs=[indices_plus_shape_name],
            options={"fusedActivationFunction": "NONE"},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="FLOOR_MOD",
            inputs=[indices_plus_shape_name, shape_prefix_name],
            outputs=[wrapped_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT",
            inputs=[negative_mask_name, wrapped_name, working_indices_name],
            outputs=[normalized_working_name],
        )
    )

    if not restore_shape_name:
        return normalized_working_name

    normalized_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_normalized",
        dtype="INT32",
        shape=[int(v) for v in indices_shape],
    )
    normalized_tensor = ctx.model_ir.tensors.get(normalized_name, None)
    if normalized_tensor is not None:
        normalized_tensor.shape_signature = [int(v) for v in indices_signature]
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[normalized_working_name, restore_shape_name],
            outputs=[normalized_name],
            options={"newShape": [], "preserveDynamicShape": True},
        )
    )
    return normalized_name


def add_zero_safe_runtime_scatter_shape(
    *,
    ctx: Any,
    data_name: str,
    name_prefix: str,
    rank: int,
) -> str:
    """Clamp internal SCATTER_ND dimensions while preserving the public shape."""
    if int(rank) <= 0:
        raise ValueError(f"SCATTER_ND runtime shape rank must be positive. rank={rank}")
    runtime_shape_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_runtime_shape",
        dtype="INT32",
        shape=[int(rank)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[data_name],
            outputs=[runtime_shape_name],
            options={"outType": "INT32"},
        )
    )
    minimum_shape_name = ctx.add_const_tensor(
        f"{name_prefix}_minimum_shape",
        np.ones((int(rank),), dtype=np.int32),
    )
    safe_shape_name = ctx.add_intermediate_tensor(
        name_prefix,
        dtype="INT32",
        shape=[int(rank)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MAXIMUM",
            inputs=[runtime_shape_name, minimum_shape_name],
            outputs=[safe_shape_name],
        )
    )
    return safe_shape_name
