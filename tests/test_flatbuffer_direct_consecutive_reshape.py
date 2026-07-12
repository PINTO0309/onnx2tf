from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_consecutive_reshape_cleanup,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT32" if data is not None else "FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=False,
    )


def _chain_model() -> ModelIR:
    model_ir = ModelIR("consecutive_reshape_runner")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "input": _tensor("input", [1, 2, 3]),
        "source": _tensor("source", [1, 2, 3]),
        "shape_mid": _tensor(
            "shape_mid", [2], data=np.asarray([2, 3], dtype=np.int32)
        ),
        "middle": _tensor("middle", [2, 3]),
        "shape_output": _tensor(
            "shape_output", [2], data=np.asarray([3, 2], dtype=np.int32)
        ),
        "output": _tensor("output", [3, 2]),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["input"], ["source"]),
        OperatorIR(
            "RESHAPE",
            ["source", "shape_mid"],
            ["middle"],
            options={"newShape": [2, 3]},
        ),
        OperatorIR(
            "RESHAPE",
            ["middle", "shape_output"],
            ["output"],
            options={"newShape": [3, 2]},
        ),
    ]
    return model_ir


def _output_noop_model() -> ModelIR:
    model_ir = ModelIR("output_noop_reshape_runner")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    shape = [1, 2, 3]
    model_ir.tensors = {
        "input": _tensor("input", shape),
        "source": _tensor("source", shape),
        "shape": _tensor("shape", [3], data=np.asarray(shape, dtype=np.int32)),
        "output": _tensor("output", shape),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["input"], ["source"]),
        OperatorIR(
            "RESHAPE",
            ["source", "shape"],
            ["output"],
            options={"newShape": shape},
        ),
    ]
    return model_ir


def test_consecutive_reshape_runner_uses_one_index_build(monkeypatch) -> None:
    model_ir = _chain_model()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_consecutive_reshape_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["rewritten_consecutive_reshape_passthrough_chains"] == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU", "RESHAPE"]
    assert model_ir.operators[1].inputs[0] == "source"
    assert refresh_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "cleanup.consecutive_reshape_passthrough"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_consecutive_reshape_runner_preserves_output_name_for_noop() -> None:
    model_ir = _output_noop_model()
    diagnostics: list[dict] = []

    stats = run_consecutive_reshape_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["removed_noop_reshape_chains"] == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].outputs == ["output"]
    assert model_ir.outputs == ["output"]
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1
