from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_trailing_output_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_trailing_output_transpose_cleanup,
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


def _direct_terminal_model(*, protected: bool) -> ModelIR:
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
        OperatorIR(
            "TRANSPOSE",
            ["before", "perm"],
            ["output"],
            options={"__preserve_layout_boundary__": True} if protected else {},
        ),
    ]
    return model_ir


def _passthrough_chain_model() -> ModelIR:
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
    return model_ir


def test_trailing_output_transpose_characterization_direct_terminal() -> None:
    model_ir = _direct_terminal_model(protected=False)

    stats = _optimize_trailing_output_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].outputs == ["output"]
    assert model_ir.tensors["output"].shape == [1, 4, 2, 3]


def test_trailing_output_transpose_characterization_passthrough_chain() -> None:
    model_ir = _passthrough_chain_model()

    stats = _optimize_trailing_output_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["RELU", "MUL"]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.tensors["activated"].shape == [1, 2, 3, 4]
    assert model_ir.tensors["output"].shape == [1, 2, 3, 4]


def test_trailing_output_transpose_characterization_preserves_boundary() -> None:
    model_ir = _direct_terminal_model(protected=True)

    stats = _optimize_trailing_output_transpose_passthrough_chains(model_ir)

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "TRANSPOSE",
    ]


def test_trailing_output_transpose_runner_rewrites_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _direct_terminal_model(protected=False)
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

    stats = run_trailing_output_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].outputs == ["output"]
    assert refresh_count == 1
    assert snapshot_count == 1
    assert diagnostics[0]["code"] == "layout.trailing_output_transpose_passthrough"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_trailing_output_transpose_runner_rejects_protected_boundary() -> None:
    model_ir = _direct_terminal_model(protected=True)
    diagnostics: list[dict] = []

    stats = run_trailing_output_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "TRANSPOSE",
    ]
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0


def test_trailing_output_transpose_runner_rewrites_passthrough_chain() -> None:
    model_ir = _passthrough_chain_model()
    diagnostics: list[dict] = []

    stats = run_trailing_output_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_trailing_output_transpose_passthrough_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["RELU", "MUL"]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.tensors["output"].shape == [1, 2, 3, 4]
    assert diagnostics[0]["status"] == "changed"
