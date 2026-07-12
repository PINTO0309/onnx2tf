from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_unary_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_unary_passthrough_cleanup,
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


def test_transpose_unary_runner_rewrites_with_one_index(monkeypatch) -> None:
    model_ir = _model(fanout=False)
    refresh_count = 0
    snapshot_count = 0
    original_refresh = ModelIRGraphIndex.refresh
    original_snapshot = ModelIRPassState.snapshot

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    def counted_snapshot(pass_state: ModelIRPassState) -> ModelIR:
        nonlocal snapshot_count
        snapshot_count += 1
        return original_snapshot(pass_state)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)
    diagnostics: list[dict] = []

    stats = run_transpose_unary_passthrough_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_transpose_unary_passthrough_chains"] == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].inputs == ["input"]
    assert model_ir.operators[0].outputs == ["output"]
    assert refresh_count == 1
    assert snapshot_count == 1
    assert diagnostics[0]["code"] == "layout.transpose_unary_passthrough"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_transpose_unary_runner_rejects_pre_fanout() -> None:
    model_ir = _model(fanout=True)
    diagnostics: list[dict] = []

    stats = run_transpose_unary_passthrough_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_transpose_unary_passthrough_chains"] == 0
    assert [op.op_type for op in model_ir.operators[:3]] == [
        "TRANSPOSE",
        "RELU",
        "TRANSPOSE",
    ]
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
