from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


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
