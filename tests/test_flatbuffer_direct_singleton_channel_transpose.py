from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.singleton_reshape_layout import (
    run_singleton_channel_transpose_cleanup,
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


def _model(*, singleton_channel: bool) -> ModelIR:
    model_ir = ModelIR("singleton_channel_transpose")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    input_shape = [1, 2, 3, 1 if singleton_channel else 4]
    output_shape = [1, input_shape[3], 2, 3]
    model_ir.tensors = {
        "input": _tensor("input", input_shape),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "output": _tensor("output", output_shape),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "perm"], ["output"]),
    ]
    return model_ir


def test_singleton_channel_transpose_runner_uses_one_index(monkeypatch) -> None:
    model_ir = _model(singleton_channel=True)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_singleton_channel_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_singleton_channel_layout_transpose_to_reshape"] == 1
    assert model_ir.operators[0].op_type == "RESHAPE"
    assert model_ir.operators[0].inputs[0] == "input"
    assert len(model_ir.operators[0].inputs) == 2
    assert model_ir.operators[0].options["layoutTransposeAsReshape"] is True
    assert refresh_count == 1
    assert diagnostics[0]["code"] == "layout.singleton_channel_transpose_as_reshape"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_singleton_channel_transpose_runner_rejects_non_singleton_order_change() -> None:
    model_ir = _model(singleton_channel=False)
    diagnostics: list[dict] = []

    stats = run_singleton_channel_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert sum(stats.values()) == 0
    assert model_ir.operators[0].op_type == "TRANSPOSE"
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
