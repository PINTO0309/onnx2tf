from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_gather_axis_cleanup,
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


def _model(*, batch_dims: int, shared_pre: bool) -> ModelIR:
    model_ir = ModelIR("transpose_gather_axis")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nhwc"] + (["side"] if shared_pre else [])
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 3, 5, 8]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 8, 3, 5]),
        "indices": _tensor(
            "indices",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 4, 6], dtype=np.int32),
        ),
        "g_nchw": _tensor("g_nchw", [1, 4, 3, 5]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y_nhwc": _tensor("y_nhwc", [1, 3, 5, 4]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "to_nchw"], ["x_nchw"]),
        OperatorIR(
            "GATHER",
            ["x_nchw", "indices"],
            ["g_nchw"],
            options={"axis": 1, "batchDims": batch_dims},
        ),
        OperatorIR("TRANSPOSE", ["g_nchw", "to_nhwc"], ["y_nhwc"]),
    ]
    if shared_pre:
        model_ir.tensors["side"] = _tensor("side", [1, 8, 3, 5])
        model_ir.operators.append(OperatorIR("RELU", ["x_nchw"], ["side"]))
    return model_ir


def test_transpose_gather_runner_rewrites_with_one_index(monkeypatch) -> None:
    model_ir = _model(batch_dims=0, shared_pre=True)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_transpose_gather_axis_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["optimized_transpose_gather_transpose_axis_remap_nhwc_chains"] == 1
    assert [op.op_type for op in model_ir.operators] == [
        "TRANSPOSE",
        "GATHER",
        "RELU",
    ]
    gather_op = model_ir.operators[1]
    assert gather_op.inputs[0] == "x_nhwc"
    assert gather_op.outputs == ["y_nhwc"]
    assert gather_op.options["axis"] == 3
    assert refresh_count == 1
    assert diagnostics[0]["code"] == "layout.transpose_gather_axis_nhwc"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_transpose_gather_runner_rejects_nonzero_batch_dims() -> None:
    model_ir = _model(batch_dims=1, shared_pre=False)
    diagnostics: list[dict] = []

    stats = run_transpose_gather_axis_cleanup(model_ir, diagnostics=diagnostics)

    assert sum(stats.values()) == 0
    assert [op.op_type for op in model_ir.operators] == [
        "TRANSPOSE",
        "GATHER",
        "TRANSPOSE",
    ]
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
