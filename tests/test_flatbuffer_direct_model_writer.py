from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.model_writer import _sanitize_model_ir_for_serialization


def test_serialization_sanitizer_reuses_indexed_dead_operator_pruning(
    monkeypatch,
) -> None:
    model_ir = ModelIR("serialization_sanitize")
    model_ir.inputs = ["x", "constant"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1], [1]),
        "constant": TensorIR(
            "constant",
            "FLOAT32",
            [1],
            [1],
            data=np.asarray([2.0], dtype=np.float32),
        ),
        "mid": TensorIR("mid", "FLOAT32", [1], [1]),
        "out": TensorIR("out", "FLOAT32", [1], [1]),
        "dead": TensorIR("dead", "FLOAT32", [1], [1]),
    }
    live_add = OperatorIR("ADD", ["x", "constant"], ["mid"])
    live_relu = OperatorIR("RELU", ["mid"], ["out"])
    dead_op = OperatorIR("IDENTITY", ["x"], ["dead"])
    model_ir.operators = [dead_op, live_add, live_relu]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    sanitized = _sanitize_model_ir_for_serialization(model_ir)

    assert refresh_count == 1
    assert sanitized.operators == [live_add, live_relu]
    assert sanitized.inputs == ["x"]
    assert "dead" not in sanitized.tensors
    assert "constant" in sanitized.tensors
    assert model_ir.operators == [dead_op, live_add, live_relu]
    assert model_ir.inputs == ["x", "constant"]
    assert "dead" in model_ir.tensors
