from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.singleton_maxpool_layout import (
    run_singleton_maxpool_layout_cleanup,
)


def _tensor(
    name: str,
    dtype: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _model(*, post_pool_fanout: bool) -> ModelIR:
    model_ir = ModelIR("singleton_maxpool_binary_cast")
    model_ir.inputs = ["a_nchw"]
    model_ir.outputs = ["d_nchw"] + (["side"] if post_pool_fanout else [])
    nchw = [1, 1, 2, 3]
    nhwc = [1, 2, 3, 1]
    model_ir.tensors = {
        "a_nchw": _tensor("a_nchw", "FLOAT32", nchw),
        "nhwc_shape1": _tensor(
            "nhwc_shape1", "INT32", [4], data=np.asarray(nhwc, dtype=np.int32)
        ),
        "a_nhwc": _tensor("a_nhwc", "FLOAT32", nhwc),
        "b_nhwc": _tensor("b_nhwc", "FLOAT32", nhwc),
        "nchw_shape1": _tensor(
            "nchw_shape1", "INT32", [4], data=np.asarray(nchw, dtype=np.int32)
        ),
        "b_nchw": _tensor("b_nchw", "FLOAT32", nchw),
        "cmp_nchw": _tensor("cmp_nchw", "BOOL", nchw),
        "cast_nchw": _tensor("cast_nchw", "FLOAT32", nchw),
        "nhwc_shape2": _tensor(
            "nhwc_shape2", "INT32", [4], data=np.asarray(nhwc, dtype=np.int32)
        ),
        "cast_nhwc": _tensor("cast_nhwc", "FLOAT32", nhwc),
        "d_nhwc": _tensor("d_nhwc", "FLOAT32", nhwc),
        "nchw_shape2": _tensor(
            "nchw_shape2", "INT32", [4], data=np.asarray(nchw, dtype=np.int32)
        ),
        "d_nchw": _tensor("d_nchw", "FLOAT32", nchw),
    }
    model_ir.operators = [
        OperatorIR("RESHAPE", ["a_nchw", "nhwc_shape1"], ["a_nhwc"]),
        OperatorIR("MAX_POOL_2D", ["a_nhwc"], ["b_nhwc"]),
        OperatorIR("RESHAPE", ["b_nhwc", "nchw_shape1"], ["b_nchw"]),
        OperatorIR("EQUAL", ["a_nchw", "b_nchw"], ["cmp_nchw"]),
        OperatorIR("CAST", ["cmp_nchw"], ["cast_nchw"]),
        OperatorIR("RESHAPE", ["cast_nchw", "nhwc_shape2"], ["cast_nhwc"]),
        OperatorIR("MAX_POOL_2D", ["cast_nhwc"], ["d_nhwc"]),
        OperatorIR("RESHAPE", ["d_nhwc", "nchw_shape2"], ["d_nchw"]),
    ]
    if post_pool_fanout:
        model_ir.tensors["side"] = _tensor("side", "FLOAT32", nchw)
        model_ir.operators.append(OperatorIR("IDENTITY", ["b_nchw"], ["side"]))
    return model_ir


def _nms_prefix_model() -> ModelIR:
    model_ir = ModelIR("singleton_nms_prefix")
    model_ir.inputs = ["lhs", "rhs"]
    model_ir.outputs = ["output"]
    nhwc = [1, 2, 3, 1]
    nchw = [1, 1, 2, 3]
    model_ir.tensors = {
        "lhs": _tensor("lhs", "FLOAT32", nhwc),
        "rhs": _tensor("rhs", "FLOAT32", nhwc),
        "equal": _tensor("equal", "BOOL", nhwc),
        "cast": _tensor("cast", "FLOAT32", nhwc),
        "pooled": _tensor("pooled", "FLOAT32", nhwc),
        "nchw_shape": _tensor(
            "nchw_shape", "INT32", [4], data=np.asarray(nchw, dtype=np.int32)
        ),
        "post": _tensor("post", "FLOAT32", nchw),
        "output": _tensor("output", "FLOAT32", nchw),
    }
    model_ir.operators = [
        OperatorIR("EQUAL", ["lhs", "rhs"], ["equal"]),
        OperatorIR("CAST", ["equal"], ["cast"]),
        OperatorIR("MAX_POOL_2D", ["cast"], ["pooled"]),
        OperatorIR("RESHAPE", ["pooled", "nchw_shape"], ["post"]),
        OperatorIR("IDENTITY", ["post"], ["output"]),
    ]
    return model_ir


def test_singleton_maxpool_binary_cast_characterization(monkeypatch) -> None:
    model_ir = _model(post_pool_fanout=False)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_singleton_maxpool_layout_cleanup(
        model_ir,
        include_nms=False,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_singleton_layout_reshape_maxpool_binary_cast_chains"] == 1
    assert sum(stats.values()) == 1
    assert [op.op_type for op in model_ir.operators] == [
        "RESHAPE",
        "MAX_POOL_2D",
        "EQUAL",
        "CAST",
        "MAX_POOL_2D",
        "RESHAPE",
    ]
    equal_op = model_ir.operators[2]
    assert equal_op.inputs == ["a_nhwc", "b_nhwc"]
    assert model_ir.operators[4].inputs == ["cast_nchw"]
    assert model_ir.tensors["cmp_nchw"].shape == [1, 2, 3, 1]
    assert refresh_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.singleton_maxpool_binary_cast"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_singleton_maxpool_binary_cast_rejects_post_pool_fanout() -> None:
    model_ir = _model(post_pool_fanout=True)

    diagnostics: list[dict] = []
    stats = run_singleton_maxpool_layout_cleanup(
        model_ir,
        include_nms=False,
        diagnostics=diagnostics,
    )

    assert sum(stats.values()) == 0
    assert [op.op_type for op in model_ir.operators[:3]] == [
        "RESHAPE",
        "MAX_POOL_2D",
        "RESHAPE",
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0


def test_singleton_nms_prefix_reaches_transactional_matcher() -> None:
    model_ir = _nms_prefix_model()
    diagnostics: list[dict] = []

    stats = run_singleton_maxpool_layout_cleanup(
        model_ir,
        include_binary_cast=False,
        diagnostics=diagnostics,
    )

    assert sum(stats.values()) == 0
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.singleton_nms_maxpool_nhwc"
    assert diagnostics[0]["status"] == "unchanged"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1
