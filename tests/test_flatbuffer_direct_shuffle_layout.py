from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_nchw_channel_shuffle_concat_gathers,
)
from onnx2tf.tflite_builder.passes.channel_shuffle import (
    run_stale_nchw_channel_shuffle_repair,
)


def _tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype="INT32" if data is not None else "FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _stale_concat_model(*, index_count: int) -> ModelIR:
    model_ir = ModelIR("stale_shuffle_concat")
    model_ir.inputs = ["left", "right"]
    model_ir.outputs = ["shuffled"]
    _tensor(model_ir, "left", [1, 4, 3, 3])
    _tensor(model_ir, "right", [1, 4, 3, 3])
    _tensor(model_ir, "concat", [1, 4, 3, 6])
    _tensor(
        model_ir,
        "indices",
        [index_count],
        data=np.arange(index_count, dtype=np.int32),
    )
    _tensor(model_ir, "shuffled", [1, index_count, 3, 6])
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["left", "right"],
            ["concat"],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "GATHER",
            ["concat", "indices"],
            ["shuffled"],
            {"axis": 1, "batchDims": 0},
        ),
    ]
    return model_ir


def test_repair_restores_channel_axis_for_nchw_shuffle_gather() -> None:
    model_ir = ModelIR("stale_shuffle_concat_axis")
    model_ir.inputs = ["left", "right"]
    model_ir.outputs = ["shuffled"]
    _tensor(model_ir, "left", [1, 232, 7, 7])
    _tensor(model_ir, "right", [1, 232, 7, 7])
    _tensor(model_ir, "concat", [1, 232, 7, 14])
    _tensor(
        model_ir,
        "indices",
        [464],
        data=np.arange(464, dtype=np.int32),
    )
    _tensor(model_ir, "shuffled", [1, 464, 7, 14])
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["left", "right"],
            ["concat"],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "GATHER",
            ["concat", "indices"],
            ["shuffled"],
            {"axis": 1, "batchDims": 0},
        ),
    ]

    stats = _repair_nchw_channel_shuffle_concat_gathers(model_ir)

    assert stats == {"repaired_nchw_channel_shuffle_concat_gathers": 1}
    assert model_ir.operators[0].options["axis"] == 1
    assert model_ir.tensors["concat"].shape == [1, 464, 7, 7]
    assert model_ir.tensors["shuffled"].shape == [1, 464, 7, 7]


def test_repair_rejects_mismatched_shuffle_index_count() -> None:
    model_ir = ModelIR("mismatched_shuffle_index_count")
    model_ir.inputs = ["left", "right"]
    model_ir.outputs = ["shuffled"]
    _tensor(model_ir, "left", [1, 4, 3, 3])
    _tensor(model_ir, "right", [1, 4, 3, 3])
    _tensor(model_ir, "concat", [1, 4, 3, 6])
    _tensor(
        model_ir,
        "indices",
        [7],
        data=np.arange(7, dtype=np.int32),
    )
    _tensor(model_ir, "shuffled", [1, 7, 3, 6])
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["left", "right"],
            ["concat"],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "GATHER",
            ["concat", "indices"],
            ["shuffled"],
            {"axis": 1, "batchDims": 0},
        ),
    ]

    stats = _repair_nchw_channel_shuffle_concat_gathers(model_ir)

    assert stats == {"repaired_nchw_channel_shuffle_concat_gathers": 0}
    assert model_ir.operators[0].options["axis"] == 3
    assert model_ir.tensors["concat"].shape == [1, 4, 3, 6]


def test_stale_channel_shuffle_runner_uses_one_index(monkeypatch) -> None:
    model_ir = _stale_concat_model(index_count=8)
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

    stats = run_stale_nchw_channel_shuffle_repair(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"repaired_nchw_channel_shuffle_concat_gathers": 1}
    assert model_ir.operators[0].options["axis"] == 1
    assert model_ir.tensors["concat"].shape == [1, 8, 3, 3]
    assert refresh_count == 1
    assert snapshot_count == 1
    assert diagnostics[0]["code"] == "layout.repair_nchw_channel_shuffle_concat"
    assert diagnostics[0]["status"] == "changed"


def test_stale_channel_shuffle_runner_rejects_index_count_before_snapshot() -> None:
    model_ir = _stale_concat_model(index_count=7)
    diagnostics: list[dict] = []

    stats = run_stale_nchw_channel_shuffle_repair(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"repaired_nchw_channel_shuffle_concat_gathers": 0}
    assert model_ir.operators[0].options["axis"] == 3
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
