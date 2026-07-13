from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.singleton_reshape_layout import (
    run_flatten_concat_reshape_cleanup,
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


def _model(*, fanout: bool) -> ModelIR:
    model_ir = ModelIR("flatten_concat_expanddims")
    model_ir.inputs = ["a4d", "b2d"]
    model_ir.outputs = ["z4d"] + (["side"] if fanout else [])
    model_ir.tensors = {
        "a4d": _tensor("a4d", [1, 1, 1, 4]),
        "shape_a2d": _tensor(
            "shape_a2d", [2], data=np.asarray([1, 4], dtype=np.int32)
        ),
        "a2d": _tensor("a2d", [1, 4]),
        "b2d": _tensor("b2d", [1, 3]),
        "c2d": _tensor("c2d", [1, 7]),
        "shape_z4d": _tensor(
            "shape_z4d", [4], data=np.asarray([1, 1, 1, 7], dtype=np.int32)
        ),
        "z4d": _tensor("z4d", [1, 1, 1, 7]),
    }
    model_ir.operators = [
        OperatorIR("RESHAPE", ["a4d", "shape_a2d"], ["a2d"]),
        OperatorIR(
            "CONCATENATION",
            ["a2d", "b2d"],
            ["c2d"],
            options={"axis": 1},
        ),
        OperatorIR("RESHAPE", ["c2d", "shape_z4d"], ["z4d"]),
    ]
    if fanout:
        model_ir.tensors["side"] = _tensor("side", [1, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", ["a2d"], ["side"]))
    return model_ir


def test_flatten_concat_reshape_runner_uses_one_index_build(monkeypatch) -> None:
    model_ir = _model(fanout=False)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_flatten_concat_reshape_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_flatten_concat_expanddims_to_nhwc_concat"] == 1
    assert [op.op_type for op in model_ir.operators] == [
        "RESHAPE",
        "CONCATENATION",
    ]
    concat_op = model_ir.operators[1]
    assert concat_op.options["axis"] == 3
    assert concat_op.outputs == ["z4d"]
    assert refresh_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.flatten_concat_expanddims_nhwc"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_flatten_concat_reshape_runner_rejects_pre_concat_fanout() -> None:
    model_ir = _model(fanout=True)
    diagnostics: list[dict] = []

    stats = run_flatten_concat_reshape_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert sum(stats.values()) == 0
    assert [op.op_type for op in model_ir.operators[:3]] == [
        "RESHAPE",
        "CONCATENATION",
        "RESHAPE",
    ]
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
