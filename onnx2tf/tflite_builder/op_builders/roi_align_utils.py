from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


def _add_binary(
    *,
    ctx: Any,
    op_type: str,
    lhs: str,
    rhs: str,
    output: str,
) -> None:
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[lhs, rhs],
            outputs=[output],
        )
    )


def _add_dynamic_reshape(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    shape: list[int],
) -> None:
    shape_name = ctx.add_const_tensor(
        f"{output_name}_shape",
        np.asarray(shape, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            options={"newShape": list(shape)},
        )
    )


def add_memory_efficient_roi_align_source(
    *,
    ctx: Any,
    input_name: str,
    batch_indices_name: str,
    output_name: str,
    compute_dtype: str,
    batch_meta: int,
    roi_count_meta: int,
    channels: int,
    height: int,
    width: int,
) -> Tuple[str, str]:
    """Flatten NCHW once and derive per-ROI global spatial offsets."""
    input_nhwc_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_input_nhwc",
        dtype=compute_dtype,
        shape=[int(batch_meta), int(height), int(width), int(channels)],
    )
    make_transpose(
        ctx=ctx,
        input_name=input_name,
        output_name=input_nhwc_name,
        perm_values=[0, 2, 3, 1],
    )
    flattened_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_input_flattened",
        dtype=compute_dtype,
        shape=[-1, int(channels)],
    )
    _add_dynamic_reshape(
        ctx=ctx,
        input_name=input_nhwc_name,
        output_name=flattened_name,
        shape=[-1, int(channels)],
    )
    batch_grid_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_batch_grid",
        dtype="INT32",
        shape=[int(roi_count_meta), 1, 1],
    )
    _add_dynamic_reshape(
        ctx=ctx,
        input_name=batch_indices_name,
        output_name=batch_grid_name,
        shape=[-1, 1, 1],
    )
    spatial_size_name = ctx.add_const_tensor(
        f"{output_name}_roialign_spatial_size",
        np.asarray(int(height * width), dtype=np.int32),
    )
    batch_offsets_name = ctx.add_intermediate_tensor(
        f"{output_name}_roialign_batch_offsets",
        dtype="INT32",
        shape=[int(roi_count_meta), 1, 1],
    )
    _add_binary(
        ctx=ctx,
        op_type="MUL",
        lhs=batch_grid_name,
        rhs=spatial_size_name,
        output=batch_offsets_name,
    )
    return flattened_name, batch_offsets_name


def add_masked_roi_align_neighbor(
    *,
    ctx: Any,
    flattened_input_name: str,
    batch_offsets_name: str,
    x_padded_index_name: str,
    y_padded_index_name: str,
    weight_name: str,
    output_name: str,
    tag: str,
    compute_dtype: str,
    roi_count_meta: int,
    pooled_h: int,
    pooled_w: int,
    channels: int,
    input_h: int,
    input_w: int,
) -> str:
    """Gather one bilinear neighbor without materializing a feature map per ROI."""
    prefix = f"{output_name}_roialign_{tag}"
    grid_shape = [int(roi_count_meta), int(pooled_h), int(pooled_w)]
    zero_name = ctx.add_const_tensor(
        f"{prefix}_zero_i32",
        np.asarray(0, dtype=np.int32),
    )
    one_name = ctx.add_const_tensor(
        f"{prefix}_one_i32",
        np.asarray(1, dtype=np.int32),
    )
    max_x_name = ctx.add_const_tensor(
        f"{prefix}_max_x_i32",
        np.asarray(int(input_w), dtype=np.int32),
    )
    max_y_name = ctx.add_const_tensor(
        f"{prefix}_max_y_i32",
        np.asarray(int(input_h), dtype=np.int32),
    )
    max_x_index_name = ctx.add_const_tensor(
        f"{prefix}_max_x_index_i32",
        np.asarray(int(input_w - 1), dtype=np.int32),
    )
    max_y_index_name = ctx.add_const_tensor(
        f"{prefix}_max_y_index_i32",
        np.asarray(int(input_h - 1), dtype=np.int32),
    )

    x_original_name = ctx.add_intermediate_tensor(
        f"{prefix}_x_original",
        dtype="INT32",
        shape=grid_shape,
    )
    y_original_name = ctx.add_intermediate_tensor(
        f"{prefix}_y_original",
        dtype="INT32",
        shape=grid_shape,
    )
    _add_binary(
        ctx=ctx,
        op_type="SUB",
        lhs=x_padded_index_name,
        rhs=one_name,
        output=x_original_name,
    )
    _add_binary(
        ctx=ctx,
        op_type="SUB",
        lhs=y_padded_index_name,
        rhs=one_name,
        output=y_original_name,
    )
    x_low_name = ctx.add_intermediate_tensor(
        f"{prefix}_x_low",
        dtype="INT32",
        shape=grid_shape,
    )
    y_low_name = ctx.add_intermediate_tensor(
        f"{prefix}_y_low",
        dtype="INT32",
        shape=grid_shape,
    )
    x_index_name = ctx.add_intermediate_tensor(
        f"{prefix}_x_index",
        dtype="INT32",
        shape=grid_shape,
    )
    y_index_name = ctx.add_intermediate_tensor(
        f"{prefix}_y_index",
        dtype="INT32",
        shape=grid_shape,
    )
    _add_binary(
        ctx=ctx,
        op_type="MAXIMUM",
        lhs=x_original_name,
        rhs=zero_name,
        output=x_low_name,
    )
    _add_binary(
        ctx=ctx,
        op_type="MAXIMUM",
        lhs=y_original_name,
        rhs=zero_name,
        output=y_low_name,
    )
    _add_binary(
        ctx=ctx,
        op_type="MINIMUM",
        lhs=x_low_name,
        rhs=max_x_index_name,
        output=x_index_name,
    )
    _add_binary(
        ctx=ctx,
        op_type="MINIMUM",
        lhs=y_low_name,
        rhs=max_y_index_name,
        output=y_index_name,
    )
    y_offset_name = ctx.add_intermediate_tensor(
        f"{prefix}_y_offset",
        dtype="INT32",
        shape=grid_shape,
    )
    spatial_index_name = ctx.add_intermediate_tensor(
        f"{prefix}_spatial_index",
        dtype="INT32",
        shape=grid_shape,
    )
    global_index_name = ctx.add_intermediate_tensor(
        f"{prefix}_global_index",
        dtype="INT32",
        shape=grid_shape,
    )
    width_name = ctx.add_const_tensor(
        f"{prefix}_width_i32",
        np.asarray(int(input_w), dtype=np.int32),
    )
    _add_binary(
        ctx=ctx,
        op_type="MUL",
        lhs=y_index_name,
        rhs=width_name,
        output=y_offset_name,
    )
    _add_binary(
        ctx=ctx,
        op_type="ADD",
        lhs=y_offset_name,
        rhs=x_index_name,
        output=spatial_index_name,
    )
    _add_binary(
        ctx=ctx,
        op_type="ADD",
        lhs=spatial_index_name,
        rhs=batch_offsets_name,
        output=global_index_name,
    )
    gathered_name = ctx.add_intermediate_tensor(
        f"{prefix}_gathered",
        dtype=compute_dtype,
        shape=[*grid_shape, int(channels)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[flattened_input_name, global_index_name],
            outputs=[gathered_name],
            options={"axis": 0, "batchDims": 0},
        )
    )

    def _compare(op_type: str, lhs: str, rhs: str, suffix: str) -> str:
        name = ctx.add_intermediate_tensor(
            f"{prefix}_{suffix}",
            dtype="BOOL",
            shape=grid_shape,
        )
        _add_binary(ctx=ctx, op_type=op_type, lhs=lhs, rhs=rhs, output=name)
        return name

    x_ge = _compare("GREATER_EQUAL", x_padded_index_name, one_name, "x_ge")
    x_le = _compare("LESS_EQUAL", x_padded_index_name, max_x_name, "x_le")
    y_ge = _compare("GREATER_EQUAL", y_padded_index_name, one_name, "y_ge")
    y_le = _compare("LESS_EQUAL", y_padded_index_name, max_y_name, "y_le")
    x_valid = _compare("LOGICAL_AND", x_ge, x_le, "x_valid")
    y_valid = _compare("LOGICAL_AND", y_ge, y_le, "y_valid")
    valid_name = _compare("LOGICAL_AND", x_valid, y_valid, "valid")
    valid_float_name = ctx.add_intermediate_tensor(
        f"{prefix}_valid_float",
        dtype=compute_dtype,
        shape=grid_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[valid_name],
            outputs=[valid_float_name],
            options={"inDataType": "BOOL", "outDataType": compute_dtype},
        )
    )
    masked_weight_name = ctx.add_intermediate_tensor(
        f"{prefix}_masked_weight",
        dtype=compute_dtype,
        shape=grid_shape,
    )
    _add_binary(
        ctx=ctx,
        op_type="MUL",
        lhs=weight_name,
        rhs=valid_float_name,
        output=masked_weight_name,
    )
    expanded_weight_name = ctx.add_intermediate_tensor(
        f"{prefix}_expanded_weight",
        dtype=compute_dtype,
        shape=[*grid_shape, 1],
    )
    _add_dynamic_reshape(
        ctx=ctx,
        input_name=masked_weight_name,
        output_name=expanded_weight_name,
        shape=[-1, int(pooled_h), int(pooled_w), 1],
    )
    weighted_name = ctx.add_intermediate_tensor(
        f"{prefix}_weighted",
        dtype=compute_dtype,
        shape=[*grid_shape, int(channels)],
    )
    _add_binary(
        ctx=ctx,
        op_type="MUL",
        lhs=gathered_name,
        rhs=expanded_weight_name,
        output=weighted_name,
    )
    return weighted_name
