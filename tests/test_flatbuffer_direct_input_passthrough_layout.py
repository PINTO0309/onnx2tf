from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.input_passthrough_layout import (
    _optimize_leading_input_transpose_passthrough_chains,
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
        is_variable=data is None,
    )


def _make_mul_roundtrip(*, add_input_fanout: bool = False) -> ModelIR:
    model_ir = ModelIR("input_passthrough_mul")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"] + (["side"] if add_input_fanout else [])
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 3, 2, 2]),
        "scale": _tensor(
            "scale",
            [1, 3, 1, 1],
            data=np.arange(3, dtype=np.float32).reshape(1, 3, 1, 1),
        ),
        "mul_nchw": _tensor("mul_nchw", [1, 3, 2, 2]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 2, 2, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "to_nchw"],
            outputs=["x_nchw"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "scale"],
            outputs=["mul_nchw"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["mul_nchw", "to_nhwc"],
            outputs=["y"],
        ),
    ]
    if add_input_fanout:
        model_ir.tensors["side"] = _tensor("side", [1, 2, 2, 3])
        model_ir.operators.append(
            OperatorIR(op_type="IDENTITY", inputs=["x_nchw"], outputs=["side"])
        )
    return model_ir


def test_input_passthrough_moves_linear_binary_chain_to_nhwc() -> None:
    model_ir = _make_mul_roundtrip()

    stats = _optimize_leading_input_transpose_passthrough_chains(model_ir)

    assert stats == {
        "rewritten_leading_input_transpose_passthrough_chains": 1
    }
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == "MUL"
    assert model_ir.operators[0].inputs == ["x", "scale"]
    assert model_ir.operators[0].outputs == ["y"]
    assert model_ir.tensors["scale"].shape == [1, 1, 1, 3]
    assert model_ir.tensors["scale"].data is not None
    assert list(model_ir.tensors["scale"].data.shape) == [1, 1, 1, 3]
    assert "x_nchw" not in model_ir.tensors
    assert "mul_nchw" not in model_ir.tensors


def test_input_passthrough_preserves_main_path_fanout() -> None:
    model_ir = _make_mul_roundtrip(add_input_fanout=True)

    stats = _optimize_leading_input_transpose_passthrough_chains(model_ir)

    assert stats == {
        "rewritten_leading_input_transpose_passthrough_chains": 0
    }
    assert [operator.op_type for operator in model_ir.operators[:3]] == [
        "TRANSPOSE",
        "MUL",
        "TRANSPOSE",
    ]
    assert model_ir.operators[1].inputs == ["x_nchw", "scale"]
