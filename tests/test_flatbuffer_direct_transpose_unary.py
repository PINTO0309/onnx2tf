from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_unary_passthrough_chains,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=False,
    )


def _model(*, fanout: bool) -> ModelIR:
    model_ir = ModelIR("transpose_unary_passthrough")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"] + (["side"] if fanout else [])
    model_ir.tensors = {
        "input": _tensor("input", [1, 2, 3, 4]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "nchw": _tensor("nchw", [1, 4, 2, 3]),
        "relu_nchw": _tensor("relu_nchw", [1, 4, 2, 3]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "output": _tensor("output", [1, 2, 3, 4]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "to_nchw"], ["nchw"]),
        OperatorIR("RELU", ["nchw"], ["relu_nchw"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc"], ["output"]),
    ]
    if fanout:
        model_ir.tensors["side"] = _tensor("side", [1, 4, 2, 3])
        model_ir.operators.append(OperatorIR("IDENTITY", ["nchw"], ["side"]))
    return model_ir


def test_transpose_unary_passthrough_characterization() -> None:
    model_ir = _model(fanout=False)

    stats = _optimize_transpose_unary_passthrough_chains(model_ir)

    assert stats["rewritten_transpose_unary_passthrough_chains"] == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].inputs == ["input"]
    assert model_ir.operators[0].outputs == ["output"]
    assert model_ir.tensors["output"].shape == [1, 2, 3, 4]


def test_transpose_unary_passthrough_rejects_pre_fanout() -> None:
    model_ir = _model(fanout=True)

    stats = _optimize_transpose_unary_passthrough_chains(model_ir)

    assert stats["rewritten_transpose_unary_passthrough_chains"] == 0
    assert [op.op_type for op in model_ir.operators[:3]] == [
        "TRANSPOSE",
        "RELU",
        "TRANSPOSE",
    ]
