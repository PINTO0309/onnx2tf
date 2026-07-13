from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.channel_shuffle import (
    run_nhwc_channel_shuffle_cleanup,
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


def _model(*, fanout_at: str | None) -> ModelIR:
    model_ir = ModelIR("nhwc_channel_shuffle")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nhwc"] + (["side"] if fanout_at else [])
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 3, 5, 8]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 8, 3, 5]),
        "shape_r1": _tensor(
            "shape_r1",
            [5],
            dtype="INT32",
            data=np.asarray([1, 2, 4, 3, 5], dtype=np.int32),
        ),
        "r1": _tensor("r1", [1, 2, 4, 3, 5]),
        "swap": _tensor(
            "swap",
            [5],
            dtype="INT32",
            data=np.asarray([0, 2, 1, 3, 4], dtype=np.int32),
        ),
        "t1": _tensor("t1", [1, 4, 2, 3, 5]),
        "shape_r2": _tensor(
            "shape_r2",
            [4],
            dtype="INT32",
            data=np.asarray([1, 8, 3, 5], dtype=np.int32),
        ),
        "r2": _tensor("r2", [1, 8, 3, 5]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y_nhwc": _tensor("y_nhwc", [1, 3, 5, 8]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "to_nchw"], ["x_nchw"]),
        OperatorIR("RESHAPE", ["x_nchw", "shape_r1"], ["r1"]),
        OperatorIR("TRANSPOSE", ["r1", "swap"], ["t1"]),
        OperatorIR("RESHAPE", ["t1", "shape_r2"], ["r2"]),
        OperatorIR("TRANSPOSE", ["r2", "to_nhwc"], ["y_nhwc"]),
    ]
    if fanout_at:
        side_shape = [1, 8, 3, 5] if fanout_at == "pre" else [1, 2, 4, 3, 5]
        source_name = "x_nchw" if fanout_at == "pre" else "r1"
        model_ir.tensors["side"] = _tensor("side", side_shape)
        model_ir.operators.append(OperatorIR("IDENTITY", [source_name], ["side"]))
    return model_ir


def test_nhwc_channel_shuffle_characterization_preserves_shared_pre() -> None:
    model_ir = _model(fanout_at="pre")

    stats = _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains(model_ir)

    assert stats["optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        "GATHER",
        "IDENTITY",
    ]
    gather_op = model_ir.operators[1]
    assert gather_op.inputs[0] == "x_nhwc"
    assert gather_op.outputs == ["y_nhwc"]
    assert gather_op.options == {"axis": 3, "batchDims": 0}
    np.testing.assert_array_equal(
        model_ir.tensors[gather_op.inputs[1]].data,
        np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int32),
    )


def test_nhwc_channel_shuffle_characterization_rejects_intermediate_fanout() -> None:
    model_ir = _model(fanout_at="r1")

    stats = _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains(model_ir)

    assert stats["optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators[:5]] == [
        "TRANSPOSE",
        "RESHAPE",
        "TRANSPOSE",
        "RESHAPE",
        "TRANSPOSE",
    ]


def test_nhwc_channel_shuffle_runner_uses_one_index(monkeypatch) -> None:
    model_ir = _model(fanout_at="pre")
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

    stats = run_nhwc_channel_shuffle_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        "GATHER",
        "IDENTITY",
    ]
    assert refresh_count == 1
    assert snapshot_count == 1
    assert diagnostics[0]["code"] == "canonicalize.nhwc_channel_shuffle_gather"
    assert diagnostics[0]["phase"] == "canonicalize"
    assert diagnostics[0]["status"] == "changed"


def test_nhwc_channel_shuffle_runner_rejects_intermediate_fanout() -> None:
    model_ir = _model(fanout_at="r1")
    diagnostics: list[dict] = []

    stats = run_nhwc_channel_shuffle_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains"] == 0
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
