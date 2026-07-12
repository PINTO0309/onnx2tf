from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def add_partial_pivot_swap(
    *,
    ctx: Any,
    matrix_name: str,
    inverse_name: str,
    current_row_mask_name: str,
    name_prefix: str,
    row_index: int,
    rows: int,
    dtype: str,
    prefix_shape: Sequence[int],
    matrix_shape: Sequence[int],
    row_shape: Sequence[int],
) -> tuple[str, str]:
    """Swap in the largest remaining pivot row for every batch matrix."""
    prefix = [int(value) for value in prefix_shape]
    matrix_dims = [int(value) for value in matrix_shape]
    row_dims = [int(value) for value in row_shape]
    rank = len(matrix_dims)
    np_dtype = np.float16 if str(dtype) == "FLOAT16" else np.float32

    def _tensor(name: str, *, tensor_dtype: str, shape: Sequence[int]) -> str:
        return ctx.add_intermediate_tensor(
            name,
            dtype=tensor_dtype,
            shape=[int(value) for value in shape],
        )

    def _binary(
        op_type: str,
        lhs: str,
        rhs: str,
        name: str,
        *,
        tensor_dtype: str,
        shape: Sequence[int],
    ) -> str:
        output_name = _tensor(name, tensor_dtype=tensor_dtype, shape=shape)
        options = (
            {"fusedActivationFunction": "NONE"}
            if op_type in {"ADD", "SUB", "MUL", "DIV"}
            else {}
        )
        ctx.add_operator(
            OperatorIR(
                op_type=op_type,
                inputs=[lhs, rhs],
                outputs=[output_name],
                options=options,
            )
        )
        return output_name

    begin = [0] * rank
    size = [-1] * rank
    begin[-1] = int(row_index)
    size[-1] = 1
    begin_name = ctx.add_const_tensor(
        f"{name_prefix}_column_begin",
        np.asarray(begin, dtype=np.int32),
    )
    size_name = ctx.add_const_tensor(
        f"{name_prefix}_column_size",
        np.asarray(size, dtype=np.int32),
    )
    column_shape = prefix + [int(rows), 1]
    column_name = _tensor(
        f"{name_prefix}_column",
        tensor_dtype=dtype,
        shape=column_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[matrix_name, begin_name, size_name],
            outputs=[column_name],
        )
    )

    absolute_name = _tensor(
        f"{name_prefix}_column_absolute",
        tensor_dtype=dtype,
        shape=column_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ABS",
            inputs=[column_name],
            outputs=[absolute_name],
        )
    )

    selector_shape = [1] * len(prefix) + [int(rows), 1]
    eligible = np.zeros(selector_shape, dtype=np.bool_)
    eligible[..., int(row_index) :, :] = True
    eligible_name = ctx.add_const_tensor(
        f"{name_prefix}_eligible",
        eligible,
    )
    negative_one_name = ctx.add_const_tensor(
        f"{name_prefix}_negative_one",
        np.asarray(-1.0, dtype=np_dtype),
    )
    eligible_absolute_name = _tensor(
        f"{name_prefix}_eligible_absolute",
        tensor_dtype=dtype,
        shape=column_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[eligible_name, absolute_name, negative_one_name],
            outputs=[eligible_absolute_name],
        )
    )

    axis_name = ctx.add_const_tensor(
        f"{name_prefix}_axis",
        np.asarray(-2, dtype=np.int32),
    )
    pivot_index_shape = prefix + [1]
    pivot_index_name = _tensor(
        f"{name_prefix}_index",
        tensor_dtype="INT32",
        shape=pivot_index_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ARG_MAX",
            inputs=[eligible_absolute_name, axis_name],
            outputs=[pivot_index_name],
            options={"outputType": "INT32"},
        )
    )

    expand_axis_name = ctx.add_const_tensor(
        f"{name_prefix}_expand_axis",
        np.asarray(-2, dtype=np.int32),
    )
    expanded_index_shape = prefix + [1, 1]
    expanded_index_name = _tensor(
        f"{name_prefix}_expanded_index",
        tensor_dtype="INT32",
        shape=expanded_index_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="EXPAND_DIMS",
            inputs=[pivot_index_name, expand_axis_name],
            outputs=[expanded_index_name],
        )
    )

    row_indices = np.arange(int(rows), dtype=np.int32).reshape(selector_shape)
    row_indices_name = ctx.add_const_tensor(
        f"{name_prefix}_row_indices",
        row_indices,
    )
    pivot_mask_bool_name = _binary(
        "EQUAL",
        row_indices_name,
        expanded_index_name,
        f"{name_prefix}_mask_bool",
        tensor_dtype="BOOL",
        shape=column_shape,
    )
    pivot_mask_name = _tensor(
        f"{name_prefix}_mask",
        tensor_dtype=dtype,
        shape=column_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="CAST",
            inputs=[pivot_mask_bool_name],
            outputs=[pivot_mask_name],
            options={"inDataType": "BOOL", "outDataType": dtype},
        )
    )

    def _selected_row(source_name: str, suffix: str) -> str:
        output_name = _tensor(
            f"{name_prefix}_{suffix}",
            tensor_dtype=dtype,
            shape=row_dims,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="BATCH_MATMUL",
                inputs=[pivot_mask_name, source_name],
                outputs=[output_name],
                options={
                    "adjX": True,
                    "adjY": False,
                    "asymmetricQuantizeInputs": False,
                },
            )
        )
        return output_name

    pivot_matrix_row = _selected_row(matrix_name, "matrix_row")
    pivot_inverse_row = _selected_row(inverse_name, "inverse_row")
    current_matrix_row = _selected_row_with_mask(
        ctx=ctx,
        mask_name=current_row_mask_name,
        source_name=matrix_name,
        output_name=f"{name_prefix}_current_matrix_row",
        dtype=dtype,
        row_shape=row_dims,
    )
    current_inverse_row = _selected_row_with_mask(
        ctx=ctx,
        mask_name=current_row_mask_name,
        source_name=inverse_name,
        output_name=f"{name_prefix}_current_inverse_row",
        dtype=dtype,
        row_shape=row_dims,
    )

    matrix_row_delta = _binary(
        "SUB",
        pivot_matrix_row,
        current_matrix_row,
        f"{name_prefix}_matrix_row_delta",
        tensor_dtype=dtype,
        shape=row_dims,
    )
    inverse_row_delta = _binary(
        "SUB",
        pivot_inverse_row,
        current_inverse_row,
        f"{name_prefix}_inverse_row_delta",
        tensor_dtype=dtype,
        shape=row_dims,
    )
    swap_mask_name = _binary(
        "SUB",
        current_row_mask_name,
        pivot_mask_name,
        f"{name_prefix}_swap_mask",
        tensor_dtype=dtype,
        shape=column_shape,
    )

    def _swap_update(row_delta_name: str, suffix: str) -> str:
        output_name = _tensor(
            f"{name_prefix}_{suffix}",
            tensor_dtype=dtype,
            shape=matrix_dims,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="BATCH_MATMUL",
                inputs=[swap_mask_name, row_delta_name],
                outputs=[output_name],
                options={
                    "adjX": False,
                    "adjY": False,
                    "asymmetricQuantizeInputs": False,
                },
            )
        )
        return output_name

    matrix_update_name = _swap_update(matrix_row_delta, "matrix_update")
    inverse_update_name = _swap_update(inverse_row_delta, "inverse_update")
    swapped_matrix_name = _binary(
        "ADD",
        matrix_name,
        matrix_update_name,
        f"{name_prefix}_matrix_swapped",
        tensor_dtype=dtype,
        shape=matrix_dims,
    )
    swapped_inverse_name = _binary(
        "ADD",
        inverse_name,
        inverse_update_name,
        f"{name_prefix}_inverse_swapped",
        tensor_dtype=dtype,
        shape=matrix_dims,
    )
    return swapped_matrix_name, swapped_inverse_name


def _selected_row_with_mask(
    *,
    ctx: Any,
    mask_name: str,
    source_name: str,
    output_name: str,
    dtype: str,
    row_shape: Sequence[int],
) -> str:
    selected_name = ctx.add_intermediate_tensor(
        output_name,
        dtype=dtype,
        shape=[int(value) for value in row_shape],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[mask_name, source_name],
            outputs=[selected_name],
            options={
                "adjX": True,
                "adjY": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )
    return selected_name


def add_sign_preserving_pivot_guard(
    *,
    ctx: Any,
    pivot_name: str,
    name_prefix: str,
    dtype: str,
    shape: Sequence[int],
    epsilon: float = 1e-6,
) -> str:
    """Replace only near-zero pivots while leaving normal pivots unchanged."""
    tensor_shape = [int(value) for value in shape]
    np_dtype = np.float16 if str(dtype) == "FLOAT16" else np.float32

    epsilon_name = ctx.add_const_tensor(
        f"{name_prefix}_eps",
        np.asarray(float(epsilon), dtype=np_dtype),
    )
    negative_epsilon_name = ctx.add_const_tensor(
        f"{name_prefix}_negative_eps",
        np.asarray(-float(epsilon), dtype=np_dtype),
    )
    zero_name = ctx.add_const_tensor(
        f"{name_prefix}_zero",
        np.asarray(0.0, dtype=np_dtype),
    )

    absolute_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_absolute",
        dtype=dtype,
        shape=tensor_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ABS",
            inputs=[pivot_name],
            outputs=[absolute_name],
        )
    )

    near_zero_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_near_zero",
        dtype="BOOL",
        shape=tensor_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LESS",
            inputs=[absolute_name, epsilon_name],
            outputs=[near_zero_name],
        )
    )

    negative_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_negative",
        dtype="BOOL",
        shape=tensor_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="LESS",
            inputs=[pivot_name, zero_name],
            outputs=[negative_name],
        )
    )

    signed_epsilon_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_signed_eps",
        dtype=dtype,
        shape=tensor_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[negative_name, negative_epsilon_name, epsilon_name],
            outputs=[signed_epsilon_name],
        )
    )

    guarded_name = ctx.add_intermediate_tensor(
        f"{name_prefix}_guarded",
        dtype=dtype,
        shape=tensor_shape,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SELECT_V2",
            inputs=[near_zero_name, signed_epsilon_name, pivot_name],
            outputs=[guarded_name],
        )
    )
    return guarded_name
