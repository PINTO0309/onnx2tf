from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    _optimize_maximum_minimum_relu0to1_chains,
    _optimize_duplicate_reshape_fanout,
    _optimize_duplicate_transpose_fanout,
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


def test_duplicate_transpose_cleanup_uses_one_incremental_index_refresh(
    monkeypatch,
) -> None:
    model_ir = ModelIR("duplicate_transpose_incremental_index")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm0": _tensor(
            "perm0",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "perm1": _tensor(
            "perm1",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "y0": _tensor("y0", [1, 3, 2, 2]),
        "y1": _tensor("y1", [1, 3, 2, 2]),
        "out": _tensor("out", [1, 3, 2, 2]),
    }
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm0"], outputs=["y0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm1"], outputs=["y1"]),
        OperatorIR(op_type="IDENTITY", inputs=["y1"], outputs=["out"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_duplicate_transpose_fanout(model_ir)

    assert stats == {"removed_duplicate_transpose_fanout": 1}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["TRANSPOSE", "IDENTITY"]
    assert model_ir.operators[1].inputs == ["y0"]
    assert "y1" not in model_ir.tensors


def test_duplicate_reshape_cleanup_uses_one_incremental_index_refresh(
    monkeypatch,
) -> None:
    model_ir = ModelIR("duplicate_reshape_incremental_index")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 6]),
        "shape0": _tensor(
            "shape0",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "shape1": _tensor(
            "shape1",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "y0": _tensor("y0", [2, 3]),
        "y1": _tensor("y1", [2, 3]),
        "out": _tensor("out", [2, 3]),
    }
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape0"], outputs=["y0"]),
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape1"], outputs=["y1"]),
        OperatorIR(op_type="IDENTITY", inputs=["y1"], outputs=["out"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_duplicate_reshape_fanout(model_ir)

    assert stats == {"removed_duplicate_reshape_fanout": 1}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["RESHAPE", "IDENTITY"]
    assert model_ir.operators[1].inputs == ["y0"]
    assert "y1" not in model_ir.tensors


def test_clamp_cleanup_uses_one_incremental_index_refresh(monkeypatch) -> None:
    model_ir = ModelIR("clamp_incremental_index")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3]),
        "zero": _tensor(
            "zero",
            [],
            data=np.asarray(0.0, dtype=np.float32),
        ),
        "maximum": _tensor("maximum", [1, 3]),
        "one": _tensor(
            "one",
            [],
            data=np.asarray(1.0, dtype=np.float32),
        ),
        "out": _tensor("out", [1, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="MAXIMUM",
            inputs=["x", "zero"],
            outputs=["maximum"],
        ),
        OperatorIR(
            op_type="MINIMUM",
            inputs=["maximum", "one"],
            outputs=["out"],
        ),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_maximum_minimum_relu0to1_chains(model_ir)

    assert stats == {"rewritten_maximum_minimum_relu0to1_chains": 1}
    assert refresh_count == 1
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == "RELU_0_TO_1"
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[0].outputs == ["out"]
    assert "maximum" not in model_ir.tensors
