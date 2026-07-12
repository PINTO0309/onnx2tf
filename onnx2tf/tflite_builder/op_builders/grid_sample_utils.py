from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import (
    _clone_quantization,
    make_transpose,
)


def _meta_shape(signature: Sequence[int]) -> list[int]:
    return [int(value) if int(value) > 0 else 1 for value in signature]


def _add_tensor(
    *,
    ctx: Any,
    name: str,
    dtype: str,
    signature: Sequence[int],
) -> str:
    tensor_name = ctx.add_intermediate_tensor(
        name,
        dtype=dtype,
        shape=_meta_shape(signature),
    )
    tensor = ctx.model_ir.tensors[tensor_name]
    tensor.shape = _meta_shape(signature)
    tensor.shape_signature = [int(value) for value in signature]
    return tensor_name


def _add_binary(
    *,
    ctx: Any,
    op_type: str,
    lhs: str,
    rhs: str,
    output: str,
) -> None:
    options = (
        {"fusedActivationFunction": "NONE"}
        if op_type in {"ADD", "SUB", "MUL", "DIV"}
        else {}
    )
    ctx.add_operator(
        OperatorIR(
            op_type=op_type,
            inputs=[lhs, rhs],
            outputs=[output],
            options=options,
        )
    )


def _add_cast(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    input_dtype: str,
    output_dtype: str,
) -> None:
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


def _add_dynamic_reshape(
    *,
    ctx: Any,
    input_name: str,
    shape_name: str,
    output_name: str,
) -> None:
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            options={"newShape": [], "preserveDynamicShape": True},
        )
    )


def build_dynamic_rank4_grid_sample(
    *,
    ctx: Any,
    image_name: str,
    grid_name: str,
    output_name: str,
    image_signature: Sequence[int],
    grid_signature: Sequence[int],
    output_signature: Sequence[int],
    image_dtype: str,
    grid_dtype: str,
    output_dtype: str,
    compute_dtype: str,
    align_corners: bool,
    interpolation_mode: str,
    padding_mode: str,
) -> None:
    """Lower dynamic rank-4 GridSample with runtime shape arithmetic.

    The image is flattened once as NHWC. Each sample gathers only its required
    neighbor values by a global spatial index, so dynamic H/W/C do not require
    per-output image copies or a statically sized padding buffer.
    """
    if len(image_signature) != 4 or len(grid_signature) != 4:
        raise NotImplementedError("dynamic GridSample requires rank-4 signatures")
    if int(grid_signature[-1]) not in {-1, 2}:
        raise NotImplementedError("dynamic GridSample grid last dimension must be 2")
    if interpolation_mode not in {"bilinear", "linear", "nearest"}:
        raise NotImplementedError(
            f"dynamic GridSample mode is unsupported: {interpolation_mode}"
        )
    if padding_mode not in {"zeros", "border"}:
        raise NotImplementedError(
            f"dynamic GridSample padding mode is unsupported: {padding_mode}"
        )

    np_dtype = np.float16 if compute_dtype == "FLOAT16" else np.float32
    grid_spatial_signature = [
        int(grid_signature[0]),
        int(grid_signature[1]),
        int(grid_signature[2]),
    ]
    channel_signature = int(image_signature[1])
    expected_output_signature = [
        int(grid_signature[0]),
        channel_signature,
        int(grid_signature[1]),
        int(grid_signature[2]),
    ]
    if len(output_signature) == 4:
        expected_output_signature = [
            int(expected_output_signature[idx])
            if int(value) <= 0
            else int(value)
            for idx, value in enumerate(output_signature)
        ]
    output_tensor = ctx.model_ir.tensors[output_name]
    output_tensor.shape = _meta_shape(expected_output_signature)
    output_tensor.shape_signature = list(expected_output_signature)

    image_compute_name = image_name
    if image_dtype != compute_dtype:
        image_compute_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_image_compute",
            dtype=compute_dtype,
            signature=image_signature,
        )
        _add_cast(
            ctx=ctx,
            input_name=image_name,
            output_name=image_compute_name,
            input_dtype=image_dtype,
            output_dtype=compute_dtype,
        )

    grid_compute_name = grid_name
    if grid_dtype != compute_dtype:
        grid_compute_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_grid_compute",
            dtype=compute_dtype,
            signature=grid_signature,
        )
        _add_cast(
            ctx=ctx,
            input_name=grid_name,
            output_name=grid_compute_name,
            input_dtype=grid_dtype,
            output_dtype=compute_dtype,
        )

    nan_mask_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_nan_mask",
        dtype="BOOL",
        signature=grid_signature,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="NOT_EQUAL",
            inputs=[grid_compute_name, grid_compute_name],
            outputs=[nan_mask_name],
        )
    )
    nan_replacement_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_nan_replacement",
        np.asarray(-1.0, dtype=np_dtype),
    )
    sanitized_grid_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_grid_sanitized",
        dtype=compute_dtype,
        signature=grid_signature,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[nan_mask_name, nan_replacement_name, grid_compute_name],
            outputs=[sanitized_grid_name],
        )
    )

    image_shape_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_image_shape",
        dtype="INT32",
        signature=[4],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[image_compute_name],
            outputs=[image_shape_name],
            options={"outType": "INT32"},
        )
    )

    def _shape_dim(index: int, tag: str) -> str:
        index_name = ctx.add_const_tensor(
            f"{output_name}_gridsample_dynamic_{tag}_index",
            np.asarray([index], dtype=np.int32),
        )
        dim_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}",
            dtype="INT32",
            signature=[1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[image_shape_name, index_name],
                outputs=[dim_name],
                options={"axis": 0, "batchDims": 0},
            )
        )
        return dim_name

    n_name = _shape_dim(0, "n")
    c_name = _shape_dim(1, "c")
    h_name = _shape_dim(2, "h")
    w_name = _shape_dim(3, "w")
    one_i32_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_one_i32",
        np.asarray([1], dtype=np.int32),
    )
    zero_i32_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_zero_i32",
        np.asarray([0], dtype=np.int32),
    )
    h_minus_one_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_h_minus_one",
        dtype="INT32",
        signature=[1],
    )
    w_minus_one_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_w_minus_one",
        dtype="INT32",
        signature=[1],
    )
    _add_binary(
        ctx=ctx,
        op_type="SUB",
        lhs=h_name,
        rhs=one_i32_name,
        output=h_minus_one_name,
    )
    _add_binary(
        ctx=ctx,
        op_type="SUB",
        lhs=w_name,
        rhs=one_i32_name,
        output=w_minus_one_name,
    )
    h_float_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_h_float",
        dtype=compute_dtype,
        signature=[1],
    )
    w_float_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_w_float",
        dtype=compute_dtype,
        signature=[1],
    )
    h_max_float_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_h_max_float",
        dtype=compute_dtype,
        signature=[1],
    )
    w_max_float_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_w_max_float",
        dtype=compute_dtype,
        signature=[1],
    )
    _add_cast(
        ctx=ctx,
        input_name=h_name,
        output_name=h_float_name,
        input_dtype="INT32",
        output_dtype=compute_dtype,
    )
    _add_cast(
        ctx=ctx,
        input_name=w_name,
        output_name=w_float_name,
        input_dtype="INT32",
        output_dtype=compute_dtype,
    )
    _add_cast(
        ctx=ctx,
        input_name=h_minus_one_name,
        output_name=h_max_float_name,
        input_dtype="INT32",
        output_dtype=compute_dtype,
    )
    _add_cast(
        ctx=ctx,
        input_name=w_minus_one_name,
        output_name=w_max_float_name,
        input_dtype="INT32",
        output_dtype=compute_dtype,
    )

    image_nhwc_signature = [
        int(image_signature[0]),
        int(image_signature[2]),
        int(image_signature[3]),
        int(image_signature[1]),
    ]
    image_nhwc_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_image_nhwc",
        dtype=compute_dtype,
        signature=image_nhwc_signature,
    )
    make_transpose(
        ctx=ctx,
        input_name=image_compute_name,
        output_name=image_nhwc_name,
        perm_values=[0, 2, 3, 1],
    )
    minus_one_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_minus_one_shape",
        np.asarray([-1], dtype=np.int32),
    )
    flattened_shape_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_flattened_shape",
        dtype="INT32",
        signature=[2],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[minus_one_name, c_name],
            outputs=[flattened_shape_name],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )
    flattened_image_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_image_flattened",
        dtype=compute_dtype,
        signature=[-1, channel_signature],
    )
    _add_dynamic_reshape(
        ctx=ctx,
        input_name=image_nhwc_name,
        shape_name=flattened_shape_name,
        output_name=flattened_image_name,
    )

    n_scalar_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_n_scalar",
        dtype="INT32",
        signature=[],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SQUEEZE",
            inputs=[n_name],
            outputs=[n_scalar_name],
            options={"squeezeDims": [0]},
        )
    )
    range_zero_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_range_zero",
        np.asarray(0, dtype=np.int32),
    )
    range_one_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_range_one",
        np.asarray(1, dtype=np.int32),
    )
    for scalar_name in [range_zero_name, range_one_name]:
        scalar_tensor = ctx.model_ir.tensors[scalar_name]
        scalar_tensor.shape = []
        scalar_tensor.shape_signature = []
    batch_range_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_batch_range",
        dtype="INT32",
        signature=[int(grid_signature[0])],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="RANGE",
            inputs=[range_zero_name, n_scalar_name, range_one_name],
            outputs=[batch_range_name],
        )
    )
    batch_grid_shape_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_batch_grid_shape",
        np.asarray([-1, 1, 1], dtype=np.int32),
    )
    batch_grid_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_batch_grid",
        dtype="INT32",
        signature=[int(grid_signature[0]), 1, 1],
    )
    _add_dynamic_reshape(
        ctx=ctx,
        input_name=batch_range_name,
        shape_name=batch_grid_shape_name,
        output_name=batch_grid_name,
    )
    spatial_size_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_spatial_size",
        dtype="INT32",
        signature=[1],
    )
    _add_binary(
        ctx=ctx,
        op_type="MUL",
        lhs=h_name,
        rhs=w_name,
        output=spatial_size_name,
    )
    batch_offsets_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_batch_offsets",
        dtype="INT32",
        signature=[int(grid_signature[0]), 1, 1],
    )
    _add_binary(
        ctx=ctx,
        op_type="MUL",
        lhs=batch_grid_name,
        rhs=spatial_size_name,
        output=batch_offsets_name,
    )

    def _coordinate(index: int, tag: str) -> str:
        index_name = ctx.add_const_tensor(
            f"{output_name}_gridsample_dynamic_{tag}_coordinate_index",
            np.asarray([index], dtype=np.int32),
        )
        with_last_dim_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_coordinate_4d",
            dtype=compute_dtype,
            signature=[*grid_spatial_signature, 1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[sanitized_grid_name, index_name],
                outputs=[with_last_dim_name],
                options={"axis": 3, "batchDims": 0},
            )
        )
        coordinate_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_coordinate",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[with_last_dim_name],
                outputs=[coordinate_name],
                options={"squeezeDims": [3]},
            )
        )
        return coordinate_name

    grid_x_name = _coordinate(0, "x")
    grid_y_name = _coordinate(1, "y")
    one_float_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_one_float",
        np.asarray(1.0, dtype=np_dtype),
    )
    half_float_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_half_float",
        np.asarray(0.5, dtype=np_dtype),
    )
    neg_one_float_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_neg_one_float",
        np.asarray(-1.0, dtype=np_dtype),
    )

    def _map_coordinate(
        coordinate_name: str,
        size_float_name: str,
        max_float_name: str,
        tag: str,
    ) -> str:
        plus_one_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_plus_one",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        scaled_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_scaled",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        mapped_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_mapped",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        _add_binary(
            ctx=ctx,
            op_type="ADD",
            lhs=coordinate_name,
            rhs=one_float_name,
            output=plus_one_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="MUL",
            lhs=plus_one_name,
            rhs=max_float_name if align_corners else size_float_name,
            output=scaled_name,
        )
        if align_corners:
            _add_binary(
                ctx=ctx,
                op_type="MUL",
                lhs=scaled_name,
                rhs=half_float_name,
                output=mapped_name,
            )
        else:
            half_scaled_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_half_scaled",
                dtype=compute_dtype,
                signature=grid_spatial_signature,
            )
            _add_binary(
                ctx=ctx,
                op_type="MUL",
                lhs=scaled_name,
                rhs=half_float_name,
                output=half_scaled_name,
            )
            _add_binary(
                ctx=ctx,
                op_type="SUB",
                lhs=half_scaled_name,
                rhs=half_float_name,
                output=mapped_name,
            )
        if padding_mode == "border":
            low_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_border_low",
                dtype=compute_dtype,
                signature=grid_spatial_signature,
            )
            clipped_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_border_clipped",
                dtype=compute_dtype,
                signature=grid_spatial_signature,
            )
            zero_float_name = ctx.add_const_tensor(
                f"{output_name}_gridsample_dynamic_zero_float",
                np.asarray(0.0, dtype=np_dtype),
            )
            _add_binary(
                ctx=ctx,
                op_type="MAXIMUM",
                lhs=mapped_name,
                rhs=zero_float_name,
                output=low_name,
            )
            _add_binary(
                ctx=ctx,
                op_type="MINIMUM",
                lhs=low_name,
                rhs=max_float_name,
                output=clipped_name,
            )
            return clipped_name
        low_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_zero_low",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        clipped_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_zero_clipped",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        _add_binary(
            ctx=ctx,
            op_type="MAXIMUM",
            lhs=mapped_name,
            rhs=neg_one_float_name,
            output=low_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="MINIMUM",
            lhs=low_name,
            rhs=size_float_name,
            output=clipped_name,
        )
        return clipped_name

    x_name = _map_coordinate(
        grid_x_name,
        w_float_name,
        w_max_float_name,
        "x",
    )
    y_name = _map_coordinate(
        grid_y_name,
        h_float_name,
        h_max_float_name,
        "y",
    )

    def _gather_neighbor(
        *,
        x_index_float_name: str,
        y_index_float_name: str,
        weight_name: str,
        tag: str,
    ) -> str:
        x_raw_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_x_raw",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        y_raw_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_y_raw",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        _add_cast(
            ctx=ctx,
            input_name=x_index_float_name,
            output_name=x_raw_name,
            input_dtype=compute_dtype,
            output_dtype="INT32",
        )
        _add_cast(
            ctx=ctx,
            input_name=y_index_float_name,
            output_name=y_raw_name,
            input_dtype=compute_dtype,
            output_dtype="INT32",
        )
        x_low_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_x_low",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        y_low_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_y_low",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        x_index_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_x_index",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        y_index_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_y_index",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        _add_binary(
            ctx=ctx,
            op_type="MAXIMUM",
            lhs=x_raw_name,
            rhs=zero_i32_name,
            output=x_low_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="MAXIMUM",
            lhs=y_raw_name,
            rhs=zero_i32_name,
            output=y_low_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="MINIMUM",
            lhs=x_low_name,
            rhs=w_minus_one_name,
            output=x_index_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="MINIMUM",
            lhs=y_low_name,
            rhs=h_minus_one_name,
            output=y_index_name,
        )
        y_offset_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_y_offset",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        spatial_index_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_spatial_index",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        global_index_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_global_index",
            dtype="INT32",
            signature=grid_spatial_signature,
        )
        _add_binary(
            ctx=ctx,
            op_type="MUL",
            lhs=y_index_name,
            rhs=w_name,
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
        gathered_signature = [*grid_spatial_signature, channel_signature]
        gathered_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_gathered",
            dtype=compute_dtype,
            signature=gathered_signature,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="GATHER",
                inputs=[flattened_image_name, global_index_name],
                outputs=[gathered_name],
                options={"axis": 0, "batchDims": 0},
            )
        )
        effective_weight_name = weight_name
        if padding_mode == "zeros":
            x_ge_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_x_ge",
                dtype="BOOL",
                signature=grid_spatial_signature,
            )
            x_lt_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_x_lt",
                dtype="BOOL",
                signature=grid_spatial_signature,
            )
            y_ge_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_y_ge",
                dtype="BOOL",
                signature=grid_spatial_signature,
            )
            y_lt_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_y_lt",
                dtype="BOOL",
                signature=grid_spatial_signature,
            )
            x_valid_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_x_valid",
                dtype="BOOL",
                signature=grid_spatial_signature,
            )
            y_valid_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_y_valid",
                dtype="BOOL",
                signature=grid_spatial_signature,
            )
            valid_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_valid",
                dtype="BOOL",
                signature=grid_spatial_signature,
            )
            for op_type, lhs, rhs, result in [
                ("GREATER_EQUAL", x_raw_name, zero_i32_name, x_ge_name),
                ("LESS", x_raw_name, w_name, x_lt_name),
                ("GREATER_EQUAL", y_raw_name, zero_i32_name, y_ge_name),
                ("LESS", y_raw_name, h_name, y_lt_name),
                ("LOGICAL_AND", x_ge_name, x_lt_name, x_valid_name),
                ("LOGICAL_AND", y_ge_name, y_lt_name, y_valid_name),
                ("LOGICAL_AND", x_valid_name, y_valid_name, valid_name),
            ]:
                _add_binary(
                    ctx=ctx,
                    op_type=op_type,
                    lhs=lhs,
                    rhs=rhs,
                    output=result,
                )
            valid_float_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_valid_float",
                dtype=compute_dtype,
                signature=grid_spatial_signature,
            )
            _add_cast(
                ctx=ctx,
                input_name=valid_name,
                output_name=valid_float_name,
                input_dtype="BOOL",
                output_dtype=compute_dtype,
            )
            effective_weight_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_{tag}_masked_weight",
                dtype=compute_dtype,
                signature=grid_spatial_signature,
            )
            _add_binary(
                ctx=ctx,
                op_type="MUL",
                lhs=weight_name,
                rhs=valid_float_name,
                output=effective_weight_name,
            )
        weight_shape_name = ctx.add_const_tensor(
            f"{output_name}_gridsample_dynamic_{tag}_weight_shape",
            np.asarray([-1, 1], dtype=np.int32),
        )
        expanded_weight_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_weight_expanded",
            dtype=compute_dtype,
            signature=[-1, 1],
        )
        _add_dynamic_reshape(
            ctx=ctx,
            input_name=effective_weight_name,
            shape_name=weight_shape_name,
            output_name=expanded_weight_name,
        )
        flattened_gather_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_gathered_flat",
            dtype=compute_dtype,
            signature=[-1, channel_signature],
        )
        # The second dimension must be runtime C, not one. Reuse the same
        # dynamic [-1,C] vector used to flatten the image.
        _add_dynamic_reshape(
            ctx=ctx,
            input_name=gathered_name,
            shape_name=flattened_shape_name,
            output_name=flattened_gather_name,
        )
        weighted_flat_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_{tag}_weighted_flat",
            dtype=compute_dtype,
            signature=[-1, channel_signature],
        )
        _add_binary(
            ctx=ctx,
            op_type="MUL",
            lhs=flattened_gather_name,
            rhs=expanded_weight_name,
            output=weighted_flat_name,
        )
        return weighted_flat_name

    one_minus_x_name: str
    one_minus_y_name: str
    if interpolation_mode == "nearest":
        x_nearest_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_x_nearest",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        y_nearest_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_y_nearest",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        ctx.add_operator(
            OperatorIR(op_type="ROUND", inputs=[x_name], outputs=[x_nearest_name])
        )
        ctx.add_operator(
            OperatorIR(op_type="ROUND", inputs=[y_name], outputs=[y_nearest_name])
        )
        unit_weight_name = ctx.add_const_tensor(
            f"{output_name}_gridsample_dynamic_unit_weight",
            np.asarray(1.0, dtype=np_dtype),
        )
        output_flat_name = _gather_neighbor(
            x_index_float_name=x_nearest_name,
            y_index_float_name=y_nearest_name,
            weight_name=unit_weight_name,
            tag="nearest",
        )
    else:
        x0_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_x0",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        y0_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_y0",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        x1_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_x1",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        y1_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_y1",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        ctx.add_operator(OperatorIR(op_type="FLOOR", inputs=[x_name], outputs=[x0_name]))
        ctx.add_operator(OperatorIR(op_type="FLOOR", inputs=[y_name], outputs=[y0_name]))
        _add_binary(
            ctx=ctx,
            op_type="ADD",
            lhs=x0_name,
            rhs=one_float_name,
            output=x1_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="ADD",
            lhs=y0_name,
            rhs=one_float_name,
            output=y1_name,
        )
        dx_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_dx",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        dy_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_dy",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        one_minus_x_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_one_minus_dx",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        one_minus_y_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_one_minus_dy",
            dtype=compute_dtype,
            signature=grid_spatial_signature,
        )
        _add_binary(
            ctx=ctx,
            op_type="SUB",
            lhs=x_name,
            rhs=x0_name,
            output=dx_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="SUB",
            lhs=y_name,
            rhs=y0_name,
            output=dy_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="SUB",
            lhs=one_float_name,
            rhs=dx_name,
            output=one_minus_x_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="SUB",
            lhs=one_float_name,
            rhs=dy_name,
            output=one_minus_y_name,
        )
        weights: dict[str, str] = {}
        for tag, x_weight, y_weight in [
            ("00", one_minus_x_name, one_minus_y_name),
            ("01", one_minus_x_name, dy_name),
            ("10", dx_name, one_minus_y_name),
            ("11", dx_name, dy_name),
        ]:
            weight_name = _add_tensor(
                ctx=ctx,
                name=f"{output_name}_gridsample_dynamic_weight_{tag}",
                dtype=compute_dtype,
                signature=grid_spatial_signature,
            )
            _add_binary(
                ctx=ctx,
                op_type="MUL",
                lhs=x_weight,
                rhs=y_weight,
                output=weight_name,
            )
            weights[tag] = weight_name
        terms = [
            _gather_neighbor(
                x_index_float_name=x0_name,
                y_index_float_name=y0_name,
                weight_name=weights["00"],
                tag="v00",
            ),
            _gather_neighbor(
                x_index_float_name=x0_name,
                y_index_float_name=y1_name,
                weight_name=weights["01"],
                tag="v01",
            ),
            _gather_neighbor(
                x_index_float_name=x1_name,
                y_index_float_name=y0_name,
                weight_name=weights["10"],
                tag="v10",
            ),
            _gather_neighbor(
                x_index_float_name=x1_name,
                y_index_float_name=y1_name,
                weight_name=weights["11"],
                tag="v11",
            ),
        ]
        sum_left_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_sum_left",
            dtype=compute_dtype,
            signature=[-1, channel_signature],
        )
        sum_right_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_sum_right",
            dtype=compute_dtype,
            signature=[-1, channel_signature],
        )
        output_flat_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_sum",
            dtype=compute_dtype,
            signature=[-1, channel_signature],
        )
        _add_binary(
            ctx=ctx,
            op_type="ADD",
            lhs=terms[0],
            rhs=terms[1],
            output=sum_left_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="ADD",
            lhs=terms[2],
            rhs=terms[3],
            output=sum_right_name,
        )
        _add_binary(
            ctx=ctx,
            op_type="ADD",
            lhs=sum_left_name,
            rhs=sum_right_name,
            output=output_flat_name,
        )

    nhwc_output_shape_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_nhwc_output_shape",
        dtype="INT32",
        signature=[4],
    )
    grid_shape_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_grid_shape",
        dtype="INT32",
        signature=[4],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[sanitized_grid_name],
            outputs=[grid_shape_name],
            options={"outType": "INT32"},
        )
    )
    grid_prefix_indices_name = ctx.add_const_tensor(
        f"{output_name}_gridsample_dynamic_grid_prefix_indices",
        np.asarray([0, 1, 2], dtype=np.int32),
    )
    grid_prefix_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_grid_prefix",
        dtype="INT32",
        signature=[3],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="GATHER",
            inputs=[grid_shape_name, grid_prefix_indices_name],
            outputs=[grid_prefix_name],
            options={"axis": 0, "batchDims": 0},
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[grid_prefix_name, c_name],
            outputs=[nhwc_output_shape_name],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )
    nhwc_output_signature = [
        int(grid_signature[0]),
        int(grid_signature[1]),
        int(grid_signature[2]),
        channel_signature,
    ]
    nhwc_output_name = _add_tensor(
        ctx=ctx,
        name=f"{output_name}_gridsample_dynamic_output_nhwc",
        dtype=compute_dtype,
        signature=nhwc_output_signature,
    )
    _add_dynamic_reshape(
        ctx=ctx,
        input_name=output_flat_name,
        shape_name=nhwc_output_shape_name,
        output_name=nhwc_output_name,
    )
    compute_output_name = output_name
    if output_dtype != compute_dtype:
        compute_output_name = _add_tensor(
            ctx=ctx,
            name=f"{output_name}_gridsample_dynamic_output_compute",
            dtype=compute_dtype,
            signature=expected_output_signature,
        )
    make_transpose(
        ctx=ctx,
        input_name=nhwc_output_name,
        output_name=compute_output_name,
        perm_values=[0, 3, 1, 2],
    )
    if output_dtype != compute_dtype:
        _add_cast(
            ctx=ctx,
            input_name=compute_output_name,
            output_name=output_name,
            input_dtype=compute_dtype,
            output_dtype=output_dtype,
        )
    input_quantization = ctx.model_ir.tensors[image_name].quantization
    if input_quantization is not None and output_tensor.quantization is None:
        output_tensor.quantization = _clone_quantization(input_quantization)
