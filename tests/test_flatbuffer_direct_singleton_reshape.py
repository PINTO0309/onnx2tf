from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.singleton_reshape_layout import (
    run_singleton_reshape_layout_cleanup,
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
        is_variable=data is None,
    )


def _unary_model() -> ModelIR:
    model_ir = ModelIR("singleton_reshape_unary")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["output"]
    nhwc = [1, 1, 1, 8]
    nchw = [1, 8, 1, 1]
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", nhwc),
        "shape_nchw": _tensor(
            "shape_nchw", [4], data=np.asarray(nchw, dtype=np.int32)
        ),
        "x_nchw": _tensor("x_nchw", nchw),
        "relu_nchw": _tensor("relu_nchw", nchw),
        "shape_nhwc": _tensor(
            "shape_nhwc", [4], data=np.asarray(nhwc, dtype=np.int32)
        ),
        "relu_nhwc": _tensor("relu_nhwc", nhwc),
        "output": _tensor("output", nhwc),
    }
    model_ir.operators = [
        OperatorIR("RESHAPE", ["x_nhwc", "shape_nchw"], ["x_nchw"]),
        OperatorIR("RELU", ["x_nchw"], ["relu_nchw"]),
        OperatorIR("RESHAPE", ["relu_nchw", "shape_nhwc"], ["relu_nhwc"]),
        OperatorIR("IDENTITY", ["relu_nhwc"], ["output"]),
    ]
    return model_ir


def _inverse_pair_model() -> ModelIR:
    model_ir = ModelIR("inverse_singleton_reshapes")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    nhwc = [1, 1, 1, 8]
    nchw = [1, 8, 1, 1]
    model_ir.tensors = {
        "input": _tensor("input", nhwc),
        "source": _tensor("source", nhwc),
        "shape_nchw": _tensor(
            "shape_nchw", [4], data=np.asarray(nchw, dtype=np.int32)
        ),
        "middle": _tensor("middle", nchw),
        "shape_nhwc": _tensor(
            "shape_nhwc", [4], data=np.asarray(nhwc, dtype=np.int32)
        ),
        "output": _tensor("output", nhwc),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["input"], ["source"]),
        OperatorIR("RESHAPE", ["source", "shape_nchw"], ["middle"]),
        OperatorIR("RESHAPE", ["middle", "shape_nhwc"], ["output"]),
    ]
    return model_ir


def test_singleton_reshape_runner_shares_index_across_ordered_specs(monkeypatch) -> None:
    model_ir = _unary_model()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_singleton_reshape_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_singleton_layout_reshape_unary_passthrough_chains"] == 1
    assert sum(stats.values()) == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU", "IDENTITY"]
    assert model_ir.operators[0].inputs == ["x_nhwc"]
    assert model_ir.operators[0].outputs == ["relu_nhwc"]
    assert refresh_count == 1
    assert [event["code"] for event in diagnostics] == [
        "layout.singleton_reshape_unary_passthrough",
        "layout.consecutive_inverse_singleton_reshapes",
    ]
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1
    assert diagnostics[1]["status"] == "skipped"
    assert diagnostics[1]["metrics"]["snapshot_count"] == 0


def test_inverse_singleton_reshape_runner_preserves_graph_output() -> None:
    model_ir = _inverse_pair_model()
    diagnostics: list[dict] = []

    stats = run_singleton_reshape_layout_cleanup(
        model_ir,
        include_unary_passthrough=False,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_consecutive_inverse_singleton_layout_reshapes"] == 1
    assert sum(stats.values()) == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].outputs == ["output"]
    assert model_ir.outputs == ["output"]
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.consecutive_inverse_singleton_reshapes"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1
