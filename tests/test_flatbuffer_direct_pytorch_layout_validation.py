from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.pytorch_layout_validation import (
    _is_attention_like_softmax_op,
    _is_transpose_sandwiched_last_axis_softmax_op,
    _propagate_feature_last_tensor_names,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    layout: str = "UNKNOWN",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT32" if data is not None else "FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        logical_layout=layout,
    )


def test_attention_softmax_reuses_supplied_graph_index(monkeypatch) -> None:
    model_ir = ModelIR(name="attention_softmax")
    model_ir.tensors = {
        "scores": _tensor("scores", [1, 2, 4, 5]),
        "probabilities": _tensor("probabilities", [1, 2, 4, 5]),
        "values": _tensor("values", [1, 2, 5, 3]),
        "context": _tensor("context", [1, 2, 4, 3]),
    }
    softmax = OperatorIR("SOFTMAX", ["scores"], ["probabilities"], {"axis": -1})
    model_ir.operators = [
        softmax,
        OperatorIR("BATCH_MATMUL", ["probabilities", "values"], ["context"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    assert _is_attention_like_softmax_op(
        model_ir,
        softmax,
        graph_index=graph_index,
    ) is True
    assert refresh_count == 1


def _transpose_sandwich_model() -> tuple[ModelIR, OperatorIR]:
    model_ir = ModelIR(name="transpose_softmax_transpose")
    model_ir.tensors = {
        "x": _tensor("x", [1, 3, 4], layout="NCW"),
        "pre_perm": _tensor(
            "pre_perm",
            [3],
            data=np.asarray([0, 2, 1], dtype=np.int32),
        ),
        "scores": _tensor("scores", [1, 4, 3]),
        "probabilities": _tensor("probabilities", [1, 4, 3]),
        "post_perm": _tensor(
            "post_perm",
            [3],
            data=np.asarray([0, 2, 1], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 3, 4], layout="NCW"),
    }
    softmax = OperatorIR(
        "SOFTMAX",
        ["scores"],
        ["probabilities"],
        {"axis": -1},
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "pre_perm"], ["scores"]),
        softmax,
        OperatorIR("TRANSPOSE", ["probabilities", "post_perm"], ["y"]),
    ]
    return model_ir, softmax


def test_transpose_sandwiched_softmax_uses_indexed_edges() -> None:
    model_ir, softmax = _transpose_sandwich_model()
    graph_index = ModelIRGraphIndex(model_ir)

    assert _is_transpose_sandwiched_last_axis_softmax_op(
        model_ir,
        softmax,
        graph_index=graph_index,
    ) is True


def test_transpose_sandwiched_softmax_rejects_duplicate_producer() -> None:
    model_ir, softmax = _transpose_sandwich_model()
    model_ir.operators.insert(0, OperatorIR("IDENTITY", ["x"], ["scores"]))
    graph_index = ModelIRGraphIndex(model_ir)

    assert _is_transpose_sandwiched_last_axis_softmax_op(
        model_ir,
        softmax,
        graph_index=graph_index,
    ) is False


def test_feature_last_worklist_reaches_bidirectional_fixed_point(monkeypatch) -> None:
    model_ir = ModelIR(name="feature_last_worklist")
    model_ir.tensors = {
        name: _tensor(name, [1, 2, 3, 4])
        for name in ["x", "relu", "residual", "sum"]
    }
    model_ir.operators = [
        OperatorIR("RELU", ["x"], ["relu"]),
        OperatorIR("ADD", ["relu", "residual"], ["sum"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    preserved = _propagate_feature_last_tensor_names(
        model_ir,
        {"relu"},
        graph_index=graph_index,
    )

    assert preserved == {"x", "relu", "residual", "sum"}
    assert refresh_count == 1


def test_feature_last_worklist_stops_at_standard_layout_transpose() -> None:
    model_ir = ModelIR(name="feature_last_transpose_barrier")
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4], layout="NHWC"),
        "perm": _tensor(
            "perm",
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 4, 2, 3], layout="NCHW"),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["y"]),
    ]
    graph_index = ModelIRGraphIndex(model_ir)

    assert _propagate_feature_last_tensor_names(
        model_ir,
        {"x"},
        graph_index=graph_index,
    ) == {"x"}
    assert _propagate_feature_last_tensor_names(
        model_ir,
        {"y"},
        graph_index=graph_index,
    ) == {"y"}
