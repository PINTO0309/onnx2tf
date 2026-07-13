from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_trailing_output_transpose_passthrough_chains,
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


def test_trailing_output_transpose_characterization_direct_terminal() -> None:
    model_ir = ModelIR("direct_terminal_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 4, 2, 3]),
        "before": _tensor("before", [1, 4, 2, 3]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "output": _tensor("output", [1, 2, 3, 4]),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["x"], ["before"]),
        OperatorIR("TRANSPOSE", ["before", "perm"], ["output"]),
    ]

    stats = _optimize_trailing_output_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].outputs == ["output"]
    assert model_ir.tensors["output"].shape == [1, 4, 2, 3]


def test_trailing_output_transpose_characterization_passthrough_chain() -> None:
    model_ir = ModelIR("terminal_transpose_passthrough_chain")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "transposed": _tensor("transposed", [1, 4, 2, 3]),
        "activated": _tensor("activated", [1, 4, 2, 3]),
        "scale": _tensor(
            "scale",
            [],
            data=np.asarray(0.5, dtype=np.float32),
        ),
        "output": _tensor("output", [1, 4, 2, 3]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["transposed"]),
        OperatorIR("RELU", ["transposed"], ["activated"]),
        OperatorIR("MUL", ["activated", "scale"], ["output"]),
    ]

    stats = _optimize_trailing_output_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["RELU", "MUL"]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.tensors["activated"].shape == [1, 2, 3, 4]
    assert model_ir.tensors["output"].shape == [1, 2, 3, 4]


def test_trailing_output_transpose_characterization_preserves_boundary() -> None:
    model_ir = ModelIR("protected_terminal_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 4, 2, 3]),
        "before": _tensor("before", [1, 4, 2, 3]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "output": _tensor("output", [1, 2, 3, 4]),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["x"], ["before"]),
        OperatorIR(
            "TRANSPOSE",
            ["before", "perm"],
            ["output"],
            options={"__preserve_layout_boundary__": True},
        ),
    ]

    stats = _optimize_trailing_output_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "TRANSPOSE",
    ]
