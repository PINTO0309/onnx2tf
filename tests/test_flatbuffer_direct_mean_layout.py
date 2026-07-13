from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.mean_layout import (
    run_mean_mul_add_conv_layout_cleanup,
    run_transpose_mean_passthrough_cleanup,
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
    model_ir = ModelIR("mean_mul_reshape_add_conv")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["conv_out"] + (["side"] if fanout else [])
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 3, 5, 4]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 4, 3, 5]),
        "axes": _tensor(
            "axes",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "mean": _tensor("mean", [1, 4, 1, 1]),
        "mul_scale": _tensor(
            "mul_scale",
            [1, 4, 1, 1],
            data=np.ones([1, 4, 1, 1], dtype=np.float32),
        ),
        "mul": _tensor("mul", [1, 4, 1, 1]),
        "reshape_shape": _tensor(
            "reshape_shape",
            [4],
            dtype="INT32",
            data=np.asarray([1, 1, 1, 4], dtype=np.int32),
        ),
        "reshaped": _tensor("reshaped", [1, 1, 1, 4]),
        "add_bias": _tensor(
            "add_bias",
            [1, 1, 1, 4],
            data=np.zeros([1, 1, 1, 4], dtype=np.float32),
        ),
        "added": _tensor("added", [1, 1, 1, 4]),
        "conv_out": _tensor("conv_out", [1, 1, 1, 4]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "to_nchw"], ["x_nchw"]),
        OperatorIR(
            "MEAN",
            ["x_nchw", "axes"],
            ["mean"],
            options={"keepDims": True},
        ),
        OperatorIR("MUL", ["mean", "mul_scale"], ["mul"]),
        OperatorIR("RESHAPE", ["mul", "reshape_shape"], ["reshaped"]),
        OperatorIR("ADD", ["reshaped", "add_bias"], ["added"]),
        OperatorIR("CONV_2D", ["added"], ["conv_out"]),
    ]
    if fanout:
        model_ir.tensors["side"] = _tensor("side", [1, 4, 1, 1])
        model_ir.operators.append(OperatorIR("IDENTITY", ["mean"], ["side"]))
    return model_ir


def _prepost_model() -> ModelIR:
    model_ir = ModelIR("transpose_mean_prepost")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nhwc"]
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 3, 5, 4]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 4, 3, 5]),
        "axes": _tensor(
            "axes",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "mean_nchw": _tensor("mean_nchw", [1, 4, 1, 1]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y_nhwc": _tensor("y_nhwc", [1, 1, 1, 4]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "to_nchw"], ["x_nchw"]),
        OperatorIR(
            "MEAN",
            ["x_nchw", "axes"],
            ["mean_nchw"],
            options={"keepDims": True},
        ),
        OperatorIR("TRANSPOSE", ["mean_nchw", "to_nhwc"], ["y_nhwc"]),
    ]
    return model_ir


def test_mean_mul_reshape_add_conv_characterization() -> None:
    model_ir = _model(fanout=False)

    stats = _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains(model_ir)

    assert stats["optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "MEAN",
        "MUL",
        "ADD",
        "CONV_2D",
    ]
    assert model_ir.operators[0].inputs == ["x_nhwc", "axes"]
    np.testing.assert_array_equal(
        model_ir.tensors["axes"].data,
        np.asarray([1, 2], dtype=np.int32),
    )
    assert model_ir.tensors["mul_scale"].shape == [1, 1, 1, 4]
    assert model_ir.operators[2].inputs == ["mul", "add_bias"]


def test_mean_mul_reshape_add_conv_rejects_mean_fanout() -> None:
    model_ir = _model(fanout=True)

    stats = _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains(model_ir)

    assert stats["optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators[:6]] == [
        "TRANSPOSE",
        "MEAN",
        "MUL",
        "RESHAPE",
        "ADD",
        "CONV_2D",
    ]


def test_mean_mul_add_conv_runner_records_transaction(monkeypatch) -> None:
    model_ir = _model(fanout=False)
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_mean_mul_add_conv_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "MEAN",
        "MUL",
        "ADD",
        "CONV_2D",
    ]
    assert diagnostics[0]["code"] == "layout.mean_mul_add_conv_nhwc"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1
    assert refresh_count == 1


def test_transpose_mean_passthrough_runner_records_transaction(monkeypatch) -> None:
    model_ir = _prepost_model()
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_transpose_mean_passthrough_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["optimized_transpose_mean_prepost_nhwc_passthrough_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["MEAN"]
    assert model_ir.operators[0].inputs == ["x_nhwc", "axes"]
    assert model_ir.operators[0].outputs == ["y_nhwc"]
    np.testing.assert_array_equal(
        model_ir.tensors["axes"].data,
        np.asarray([1, 2], dtype=np.int32),
    )
    assert diagnostics[0]["code"] == "layout.transpose_mean_prepost"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1
    assert refresh_count == 1


def test_mean_mul_add_conv_runner_rejects_fanout_before_snapshot() -> None:
    model_ir = _model(fanout=True)
    diagnostics: list[dict[str, object]] = []

    stats = run_mean_mul_add_conv_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains"] == 0
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
