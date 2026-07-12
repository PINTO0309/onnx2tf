from __future__ import annotations

from types import SimpleNamespace

import pytest

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.op_builders.shape import build_concat_op


def _node():
    return SimpleNamespace(
        name="shape_concat",
        op="Concat",
        attrs={"axis": 0},
        inputs=[SimpleNamespace(name="dynamic_dim"), SimpleNamespace(name="captured_dim")],
        outputs=[SimpleNamespace(name="shape_vector")],
    )


def _context(*, rank: int = 1, captured_shape=None) -> LoweringContext:
    shape = [1 for _ in range(rank)]
    captured_shape = shape if captured_shape is None else list(captured_shape)
    model_ir = ModelIR(name="concat_shape_dtype")
    model_ir.tensors = {
        "dynamic_dim": TensorIR(name="dynamic_dim", dtype="INT32", shape=shape),
        "captured_dim": TensorIR(
            name="captured_dim",
            dtype="FLOAT32",
            shape=captured_shape,
        ),
        "shape_vector": TensorIR(
            name="shape_vector",
            dtype="INT32",
            shape=[2] if rank == 1 else shape,
        ),
    }
    return LoweringContext(
        model_ir=model_ir,
        shape_map={name: list(tensor.shape) for name, tensor in model_ir.tensors.items()},
        dtype_map={name: str(tensor.dtype) for name, tensor in model_ir.tensors.items()},
        constants={},
    )


def test_rank1_integer_shape_concat_casts_captured_float_to_output_dtype() -> None:
    ctx = _context(captured_shape=[1, 1])
    build_concat_op(_node(), ctx)

    assert [str(op.op_type) for op in ctx.model_ir.operators] == [
        "CAST",
        "RESHAPE",
        "CONCATENATION",
    ]
    cast_op, reshape_op, concat_op = ctx.model_ir.operators
    assert cast_op.inputs == ["captured_dim"]
    assert cast_op.options == {"inDataType": "FLOAT32", "outDataType": "INT32"}
    assert concat_op.inputs[0] == "dynamic_dim"
    assert reshape_op.inputs == cast_op.outputs
    assert reshape_op.options == {"newShape": [1]}
    assert concat_op.inputs[1] == reshape_op.outputs[0]
    assert ctx.model_ir.tensors["shape_vector"].dtype == "INT32"


def test_mixed_dtype_data_concat_remains_rejected() -> None:
    ctx = _context(rank=2)
    with pytest.raises(NotImplementedError, match="Concat input dtypes must be compatible"):
        build_concat_op(_node(), ctx)
