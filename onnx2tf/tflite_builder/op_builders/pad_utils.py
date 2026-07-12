from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


@dataclass(frozen=True)
class ZeroSafeBatchPadPlan:
    input_name: str
    pad_output_name: str
    desired_output_shape_name: str
    zero_value_name: str


def prepare_zero_safe_batch_pad(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    pads_begin: Sequence[int],
    pads_end: Sequence[int],
) -> Optional[ZeroSafeBatchPadPlan]:
    """Add a temporary batch element so PAD never receives a zero batch."""
    input_tensor = ctx.model_ir.tensors.get(str(input_name), None)
    output_tensor = ctx.model_ir.tensors.get(str(output_name), None)
    if input_tensor is None or output_tensor is None:
        return None
    shape = [int(value) for value in list(input_tensor.shape)]
    signature = (
        [int(value) for value in list(input_tensor.shape_signature)]
        if input_tensor.shape_signature is not None
        else list(shape)
    )
    rank = len(shape)
    dtype = str(input_tensor.dtype).upper()
    if (
        rank == 0
        or len(signature) != rank
        or len(pads_begin) != rank
        or len(pads_end) != rank
        or int(signature[0]) > 0
        or int(pads_begin[0]) != 0
        or int(pads_end[0]) != 0
        or dtype not in {"FLOAT16", "FLOAT32"}
        or input_tensor.quantization is not None
    ):
        return None

    prefix = f"{output_name}_zero_safe_pad"
    runtime_shape_name = ctx.add_intermediate_tensor(
        f"{prefix}_runtime_input_shape",
        dtype="INT32",
        shape=[rank],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[runtime_shape_name],
            options={"outType": "INT32"},
        )
    )
    keep_non_batch_name = ctx.add_const_tensor(
        f"{prefix}_keep_non_batch",
        np.asarray([0] + [1] * (rank - 1), dtype=np.int32),
    )
    filler_shape_base_name = ctx.add_intermediate_tensor(
        f"{prefix}_filler_shape_base",
        dtype="INT32",
        shape=[rank],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MUL",
            inputs=[runtime_shape_name, keep_non_batch_name],
            outputs=[filler_shape_base_name],
        )
    )
    minimum_batch_name = ctx.add_const_tensor(
        f"{prefix}_minimum_batch",
        np.asarray([1] + [0] * (rank - 1), dtype=np.int32),
    )
    filler_shape_name = ctx.add_intermediate_tensor(
        f"{prefix}_filler_shape",
        dtype="INT32",
        shape=[rank],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[filler_shape_base_name, minimum_batch_name],
            outputs=[filler_shape_name],
        )
    )
    np_dtype = {
        "FLOAT16": np.float16,
        "FLOAT32": np.float32,
    }[dtype]
    zero_value_name = ctx.add_const_tensor(
        f"{prefix}_zero",
        np.asarray(0, dtype=np_dtype),
    )
    zero_value_tensor = ctx.model_ir.tensors[zero_value_name]
    zero_value_tensor.shape = []
    zero_value_tensor.shape_signature = []
    filler_shape = [max(int(value), 1) for value in shape]
    filler_shape[0] = 1
    filler_name = ctx.add_intermediate_tensor(
        f"{prefix}_filler",
        dtype=dtype,
        shape=filler_shape,
    )
    filler_tensor = ctx.model_ir.tensors[filler_name]
    filler_tensor.shape_signature = [1, *signature[1:]]
    ctx.add_operator(
        OperatorIR(
            op_type="FILL",
            inputs=[filler_shape_name, zero_value_name],
            outputs=[filler_name],
        )
    )
    appended_shape = [max(int(value), 1) for value in shape]
    appended_shape[0] = max(int(shape[0]), 1) + 1
    appended_name = ctx.add_intermediate_tensor(
        f"{prefix}_appended",
        dtype=dtype,
        shape=appended_shape,
    )
    appended_tensor = ctx.model_ir.tensors[appended_name]
    appended_tensor.shape_signature = [-1, *signature[1:]]
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[input_name, filler_name],
            outputs=[appended_name],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )
    safe_shape_name = ctx.add_intermediate_tensor(
        f"{prefix}_safe_input_shape",
        dtype="INT32",
        shape=[rank],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="MAXIMUM",
            inputs=[runtime_shape_name, minimum_batch_name],
            outputs=[safe_shape_name],
        )
    )
    slice_begin_name = ctx.add_const_tensor(
        f"{prefix}_slice_begin",
        np.zeros((rank,), dtype=np.int32),
    )
    safe_input_shape = [max(int(value), 1) for value in shape]
    safe_input_name = ctx.add_intermediate_tensor(
        f"{prefix}_input",
        dtype=dtype,
        shape=safe_input_shape,
    )
    safe_input_tensor = ctx.model_ir.tensors[safe_input_name]
    safe_input_tensor.shape_signature = [-1, *signature[1:]]
    ctx.add_operator(
        OperatorIR(
            op_type="SLICE",
            inputs=[appended_name, slice_begin_name, safe_shape_name],
            outputs=[safe_input_name],
        )
    )

    total_pads_name = ctx.add_const_tensor(
        f"{prefix}_total_pads",
        np.asarray(
            [int(begin) + int(end) for begin, end in zip(pads_begin, pads_end)],
            dtype=np.int32,
        ),
    )
    desired_output_shape_name = ctx.add_intermediate_tensor(
        f"{prefix}_desired_output_shape",
        dtype="INT32",
        shape=[rank],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[runtime_shape_name, total_pads_name],
            outputs=[desired_output_shape_name],
        )
    )
    safe_output_shape = [max(int(value), 1) for value in output_tensor.shape]
    safe_output_shape[0] = 1
    safe_output_name = ctx.add_intermediate_tensor(
        f"{prefix}_output",
        dtype=dtype,
        shape=safe_output_shape,
    )
    safe_output_tensor = ctx.model_ir.tensors[safe_output_name]
    output_signature = (
        [int(value) for value in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else list(output_tensor.shape)
    )
    safe_output_tensor.shape_signature = [-1, *output_signature[1:]]
    return ZeroSafeBatchPadPlan(
        input_name=safe_input_name,
        pad_output_name=safe_output_name,
        desired_output_shape_name=desired_output_shape_name,
        zero_value_name=zero_value_name,
    )


def finish_zero_safe_batch_pad(
    *,
    ctx: Any,
    output_name: str,
    plan: ZeroSafeBatchPadPlan,
) -> None:
    """Broadcast the safe PAD result back to the true, possibly empty shape."""
    output_tensor = ctx.model_ir.tensors[str(output_name)]
    empty_shape_name = ctx.add_intermediate_tensor(
        f"{output_name}_zero_safe_pad_output_zeros",
        dtype=str(output_tensor.dtype),
        shape=[int(value) for value in output_tensor.shape],
    )
    empty_shape_tensor = ctx.model_ir.tensors[empty_shape_name]
    empty_shape_tensor.shape_signature = (
        [int(value) for value in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(value) for value in output_tensor.shape]
    )
    ctx.add_operator(
        OperatorIR(
            op_type="FILL",
            inputs=[plan.desired_output_shape_name, plan.zero_value_name],
            outputs=[empty_shape_name],
        )
    )
    ctx.add_operator(
        OperatorIR(
            op_type="ADD",
            inputs=[plan.pad_output_name, empty_shape_name],
            outputs=[output_name],
        )
    )


def add_zero_safe_constant_pad_operator(
    *,
    ctx: Any,
    input_name: str,
    pads_name: str,
    output_name: str,
    pads_begin: Sequence[int],
    pads_end: Sequence[int],
) -> None:
    """Emit PAD with zero-batch protection when its metadata requires it."""
    plan = prepare_zero_safe_batch_pad(
        ctx=ctx,
        input_name=input_name,
        output_name=output_name,
        pads_begin=pads_begin,
        pads_end=pads_end,
    )
    ctx.add_operator(
        OperatorIR(
            op_type="PAD",
            inputs=[plan.input_name if plan is not None else input_name, pads_name],
            outputs=[plan.pad_output_name if plan is not None else output_name],
        )
    )
    if plan is not None:
        finish_zero_safe_batch_pad(
            ctx=ctx,
            output_name=output_name,
            plan=plan,
        )
