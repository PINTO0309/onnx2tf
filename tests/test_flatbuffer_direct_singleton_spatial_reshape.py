from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.singleton_reshape_layout import (
    run_singleton_spatial_reshape_cleanup,
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


def _spatial_flatten_model() -> ModelIR:
    model_ir = ModelIR("singleton_spatial_flatten")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 1, 1, 4]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 4, 1, 1]),
        "shape": _tensor(
            "shape", [2], dtype="INT32", data=np.asarray([1, 4], dtype=np.int32)
        ),
        "y": _tensor("y", [1, 4]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "perm"], ["x_nchw"]),
        OperatorIR("RESHAPE", ["x_nchw", "shape"], ["y"]),
    ]
    return model_ir


def _concat_model() -> ModelIR:
    model_ir = ModelIR("singleton_concat_post_transpose")
    model_ir.inputs = ["x_nchw"]
    model_ir.outputs = ["y_nhwc"]
    model_ir.tensors = {
        "x_nchw": _tensor("x_nchw", [1, 1, 8, 8]),
        "concat_nchw": _tensor("concat_nchw", [1, 3, 8, 8]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y_nhwc": _tensor("y_nhwc", [1, 8, 8, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["x_nchw", "x_nchw", "x_nchw"],
            ["concat_nchw"],
            options={"axis": 1},
        ),
        OperatorIR("TRANSPOSE", ["concat_nchw", "perm"], ["y_nhwc"]),
    ]
    return model_ir


def test_singleton_spatial_flatten_runner_rewrites_with_one_index(monkeypatch) -> None:
    model_ir = _spatial_flatten_model()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_singleton_spatial_reshape_cleanup(
        model_ir,
        include_concat_post_transpose=False,
        diagnostics=diagnostics,
    )

    assert stats["optimized_singleton_spatial_nhwc_transpose_reshape_flatten"] == 1
    assert [op.op_type for op in model_ir.operators] == ["RESHAPE"]
    assert model_ir.operators[0].inputs[0] == "x_nhwc"
    assert refresh_count == 1
    assert diagnostics[0]["code"] == "layout.singleton_spatial_flatten"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_singleton_concat_runner_reuses_one_inserted_adapter() -> None:
    model_ir = _concat_model()
    diagnostics: list[dict] = []

    stats = run_singleton_spatial_reshape_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_singleton_reshape_concat_post_transpose_nhwc_chains"] == 1
    assert [op.op_type for op in model_ir.operators] == [
        "RESHAPE",
        "CONCATENATION",
    ]
    adapter_name = model_ir.operators[0].outputs[0]
    assert model_ir.operators[1].inputs == [adapter_name, adapter_name, adapter_name]
    assert model_ir.operators[1].outputs == ["y_nhwc"]
    assert [event["status"] for event in diagnostics] == ["skipped", "changed"]
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
    assert diagnostics[1]["metrics"]["snapshot_count"] == 1
