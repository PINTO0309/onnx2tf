from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    ModelIRPassStateScope,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather,
)
from onnx2tf.tflite_builder.passes.channel_shuffle import (
    run_nchw_channel_shuffle_cleanup,
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
    model_ir = ModelIR("nchw_channel_shuffle")
    model_ir.inputs = ["x_nchw"]
    model_ir.outputs = ["y_nchw"] + (["side"] if fanout else [])
    model_ir.tensors = {
        "x_nchw": _tensor("x_nchw", [1, 8, 3, 5]),
        "shape_r1": _tensor(
            "shape_r1",
            [5],
            dtype="INT32",
            data=np.asarray([1, 2, 4, 3, 5], dtype=np.int32),
        ),
        "r1": _tensor("r1", [1, 2, 4, 3, 5]),
        "perm": _tensor(
            "perm",
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
        "y_nchw": _tensor("y_nchw", [1, 8, 3, 5]),
    }
    model_ir.operators = [
        OperatorIR("RESHAPE", ["x_nchw", "shape_r1"], ["r1"]),
        OperatorIR("TRANSPOSE", ["r1", "perm"], ["t1"]),
        OperatorIR("RESHAPE", ["t1", "shape_r2"], ["y_nchw"]),
    ]
    if fanout:
        model_ir.tensors["side"] = _tensor("side", [1, 2, 4, 3, 5])
        model_ir.operators.append(OperatorIR("IDENTITY", ["r1"], ["side"]))
    return model_ir


def test_nchw_channel_shuffle_characterization_rewrites_to_gather() -> None:
    model_ir = _model(fanout=False)

    stats = _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
        model_ir
    )

    assert stats[
        "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather"
    ] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["GATHER"]
    gather_op = model_ir.operators[0]
    assert gather_op.inputs[0] == "x_nchw"
    assert gather_op.outputs == ["y_nchw"]
    assert gather_op.options == {"axis": 1, "batchDims": 0}
    indices = np.asarray(model_ir.tensors[gather_op.inputs[1]].data).reshape(-1)
    np.testing.assert_array_equal(
        indices,
        np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int32),
    )


def test_nchw_channel_shuffle_characterization_rejects_intermediate_fanout() -> None:
    model_ir = _model(fanout=True)

    stats = _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
        model_ir
    )

    assert stats[
        "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather"
    ] == 0
    assert [operator.op_type for operator in model_ir.operators[:3]] == [
        "RESHAPE",
        "TRANSPOSE",
        "RESHAPE",
    ]


def test_nchw_channel_shuffle_runner_uses_one_index(monkeypatch) -> None:
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
    state_scope = ModelIRPassStateScope(model_ir)

    stats = run_nchw_channel_shuffle_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    pass_state, state_built = state_scope.acquire(
        model_ir=model_ir,
        layout_state=None,
    )

    assert stats[
        "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather"
    ] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["GATHER"]
    assert refresh_count == 1
    assert snapshot_count == 1
    assert state_built is False
    assert pass_state.graph_index.operator_indices("RESHAPE") == []
    assert pass_state.graph_index.operator_indices("GATHER") == [0]
    assert diagnostics[0]["code"] == "canonicalize.nchw_channel_shuffle_gather"
    assert diagnostics[0]["phase"] == "canonicalize"
    assert diagnostics[0]["status"] == "changed"


def test_nchw_channel_shuffle_runner_rejects_intermediate_fanout() -> None:
    model_ir = _model(fanout=True)
    diagnostics: list[dict] = []

    stats = run_nchw_channel_shuffle_cleanup(model_ir, diagnostics=diagnostics)

    assert stats[
        "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather"
    ] == 0
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
