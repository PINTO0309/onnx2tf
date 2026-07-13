from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_se_conv_mul_prepost_nhwc_chains,
    _optimize_transpose_se_fc_mul_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.se_layout import (
    run_se_conv_layout_cleanup,
    run_se_fc_layout_cleanup,
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
        is_variable=False if data is not None else True,
    )


def _se_conv_model(*, gate_fanout: bool) -> ModelIR:
    model_ir = ModelIR("se_conv_layout")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"] + (["gate_side"] if gate_fanout else [])
    shapes = {
        "x_nhwc": [1, 4, 4, 8],
        "x_nchw": [1, 8, 4, 4],
        "sig1_nchw": [1, 8, 4, 4],
        "sw_nchw": [1, 8, 4, 4],
        "mean_nchw": [1, 8, 1, 1],
        "mean_nhwc": [1, 1, 1, 8],
        "gate_pre_nhwc": [1, 1, 1, 8],
        "gate_nchw": [1, 8, 1, 1],
        "sig2_nchw": [1, 8, 1, 1],
        "y_nchw": [1, 8, 4, 4],
        "y_nhwc": [1, 4, 4, 8],
        "z": [1, 4, 4, 8],
    }
    model_ir.tensors = {
        name: _tensor(name, shape) for name, shape in shapes.items()
    }
    model_ir.tensors.update(
        {
            "to_nchw": _tensor(
                "to_nchw",
                [4],
                dtype="INT32",
                data=np.asarray([0, 3, 1, 2], dtype=np.int32),
            ),
            "to_nhwc": _tensor(
                "to_nhwc",
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            "axes": _tensor(
                "axes",
                [2],
                dtype="INT32",
                data=np.asarray([2, 3], dtype=np.int32),
            ),
            "conv_w": _tensor(
                "conv_w",
                [1, 1, 8, 8],
                data=np.ones([1, 1, 8, 8], dtype=np.float32),
            ),
            "conv_b": _tensor(
                "conv_b",
                [8],
                data=np.zeros([8], dtype=np.float32),
            ),
        }
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "to_nchw"], ["x_nchw"]),
        OperatorIR("LOGISTIC", ["x_nchw"], ["sig1_nchw"]),
        OperatorIR("MUL", ["x_nchw", "sig1_nchw"], ["sw_nchw"]),
        OperatorIR(
            "MEAN",
            ["sw_nchw", "axes"],
            ["mean_nchw"],
            options={"keepDims": True, "axes": [2, 3]},
        ),
        OperatorIR("TRANSPOSE", ["mean_nchw", "to_nhwc"], ["mean_nhwc"]),
        OperatorIR(
            "CONV_2D",
            ["mean_nhwc", "conv_w", "conv_b"],
            ["gate_pre_nhwc"],
        ),
        OperatorIR("TRANSPOSE", ["gate_pre_nhwc", "to_nchw"], ["gate_nchw"]),
        OperatorIR("LOGISTIC", ["gate_nchw"], ["sig2_nchw"]),
        OperatorIR("MUL", ["sw_nchw", "sig2_nchw"], ["y_nchw"]),
        OperatorIR("TRANSPOSE", ["y_nchw", "to_nhwc"], ["y_nhwc"]),
        OperatorIR("RELU", ["y_nhwc"], ["z"]),
    ]
    if gate_fanout:
        model_ir.tensors["gate_side"] = _tensor(
            "gate_side",
            [1, 8, 1, 1],
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["sig2_nchw"], ["gate_side"])
        )
    return model_ir


def _se_fc_model(*, public_gate: bool, shared_pre: bool) -> ModelIR:
    model_ir = ModelIR("se_fc_layout")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y"]
    if public_gate:
        model_ir.outputs.append("gate_nchw")
    if shared_pre:
        model_ir.outputs.append("pre_side")
    shapes = {
        "x_nhwc": [1, 4, 4, 8],
        "x_nchw": [1, 8, 4, 4],
        "pool_nhwc": [1, 1, 1, 8],
        "pool_nchw": [1, 8, 1, 1],
        "fc_in": [1, 8],
        "fc_out": [1, 8],
        "gate_nchw": [1, 8, 1, 1],
        "mul_nchw": [1, 8, 4, 4],
        "mul_nhwc": [1, 4, 4, 8],
        "y": [1, 4, 4, 8],
    }
    model_ir.tensors = {
        name: _tensor(name, shape) for name, shape in shapes.items()
    }
    model_ir.tensors.update(
        {
            "to_nchw": _tensor(
                "to_nchw",
                [4],
                dtype="INT32",
                data=np.asarray([0, 3, 1, 2], dtype=np.int32),
            ),
            "to_nhwc": _tensor(
                "to_nhwc",
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            "flat_shape": _tensor(
                "flat_shape",
                [2],
                dtype="INT32",
                data=np.asarray([1, 8], dtype=np.int32),
            ),
            "gate_shape": _tensor(
                "gate_shape",
                [4],
                dtype="INT32",
                data=np.asarray([1, 8, 1, 1], dtype=np.int32),
            ),
            "fc_w": _tensor(
                "fc_w",
                [8, 8],
                data=np.ones([8, 8], dtype=np.float32),
            ),
            "fc_b": _tensor(
                "fc_b",
                [8],
                data=np.zeros([8], dtype=np.float32),
            ),
        }
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "to_nchw"], ["x_nchw"]),
        OperatorIR("AVERAGE_POOL_2D", ["x_nhwc"], ["pool_nhwc"]),
        OperatorIR("TRANSPOSE", ["pool_nhwc", "to_nchw"], ["pool_nchw"]),
        OperatorIR("RESHAPE", ["pool_nchw", "flat_shape"], ["fc_in"]),
        OperatorIR("FULLY_CONNECTED", ["fc_in", "fc_w", "fc_b"], ["fc_out"]),
        OperatorIR(
            "RESHAPE",
            ["fc_out", "gate_shape"],
            ["gate_nchw"],
            options={
                "newShape": [1, 8, 1, 1],
                "onnxRawNewShape": [1, 8, 1, 1],
            },
        ),
        OperatorIR("MUL", ["gate_nchw", "x_nchw"], ["mul_nchw"]),
        OperatorIR("TRANSPOSE", ["mul_nchw", "to_nhwc"], ["mul_nhwc"]),
        OperatorIR("RELU", ["mul_nhwc"], ["y"]),
    ]
    if shared_pre:
        model_ir.tensors["pre_side"] = _tensor(
            "pre_side",
            [1, 8, 4, 4],
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x_nchw"], ["pre_side"])
        )
    return model_ir


def test_se_conv_layout_rejects_gate_fanout() -> None:
    model_ir = _se_conv_model(gate_fanout=True)

    stats = _optimize_transpose_se_conv_mul_prepost_nhwc_chains(model_ir)

    assert stats["optimized_transpose_se_conv_mul_prepost_nhwc_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators].count(
        "TRANSPOSE"
    ) == 4
    np.testing.assert_array_equal(
        model_ir.tensors["axes"].data,
        np.asarray([2, 3], dtype=np.int32),
    )


def test_se_fc_layout_rejects_public_gate() -> None:
    model_ir = _se_fc_model(public_gate=True, shared_pre=False)

    stats = _optimize_transpose_se_fc_mul_prepost_nhwc_chains(model_ir)

    assert stats["optimized_transpose_se_fc_mul_prepost_nhwc_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators].count(
        "TRANSPOSE"
    ) == 3
    np.testing.assert_array_equal(
        model_ir.tensors["gate_shape"].data,
        np.asarray([1, 8, 1, 1], dtype=np.int32),
    )


def test_se_fc_layout_keeps_shared_input_transpose() -> None:
    model_ir = _se_fc_model(public_gate=False, shared_pre=True)

    stats = run_se_fc_layout_cleanup(model_ir)

    assert stats["optimized_transpose_se_fc_mul_prepost_nhwc_chains"] == 1
    transpose_ops = [
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE"
    ]
    assert len(transpose_ops) == 1
    assert transpose_ops[0].outputs == ["x_nchw"]
    mul_op = next(
        operator for operator in model_ir.operators if operator.op_type == "MUL"
    )
    assert "x_nhwc" in mul_op.inputs
    side_op = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "IDENTITY"
    )
    assert side_op.inputs == ["x_nchw"]


def test_se_conv_layout_runner_reuses_one_index(monkeypatch) -> None:
    model_ir = _se_conv_model(gate_fanout=False)
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_se_conv_layout_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["optimized_transpose_se_conv_mul_prepost_nhwc_chains"] == 1
    assert refresh_count == 1
    assert diagnostics[0]["code"] == "layout.se_conv_gate_nhwc"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_se_conv_layout_runner_rejects_gate_fanout_before_snapshot() -> None:
    model_ir = _se_conv_model(gate_fanout=True)
    diagnostics: list[dict[str, object]] = []

    stats = run_se_conv_layout_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["optimized_transpose_se_conv_mul_prepost_nhwc_chains"] == 0
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0


def test_se_fc_layout_runner_reuses_one_index(monkeypatch) -> None:
    model_ir = _se_fc_model(public_gate=False, shared_pre=False)
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_se_fc_layout_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["optimized_transpose_se_fc_mul_prepost_nhwc_chains"] == 1
    assert refresh_count == 1
    assert diagnostics[0]["code"] == "layout.se_fc_gate_nhwc"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_se_fc_layout_runner_rejects_public_gate_before_snapshot() -> None:
    model_ir = _se_fc_model(public_gate=True, shared_pre=False)
    diagnostics: list[dict[str, object]] = []

    stats = run_se_fc_layout_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["optimized_transpose_se_fc_mul_prepost_nhwc_chains"] == 0
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
