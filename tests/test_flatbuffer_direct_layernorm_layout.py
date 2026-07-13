from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains,
    _optimize_transpose_layernorm_stats_nhwc_propagation_chains,
)
from onnx2tf.tflite_builder.passes.layernorm_layout import (
    run_layernorm_statistics_layout_cleanup,
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


def _model(*, source: str, fanout: bool) -> ModelIR:
    model_ir = ModelIR(f"layernorm_stats_{source}")
    model_ir.inputs = ["x_nhwc" if source == "pre" else "x_nchw"]
    model_ir.outputs = ["y"] + (["side"] if fanout else [])
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 1, 5, 3]),
        "x_nchw": _tensor("x_nchw", [1, 3, 1, 5]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray(
                [0, 3, 1, 2] if source == "pre" else [0, 2, 3, 1],
                dtype=np.int32,
            ),
        ),
        "mean_axes": _tensor(
            "mean_axes",
            [1],
            dtype="INT32",
            data=np.asarray([1], dtype=np.int32),
        ),
        "var_axes": _tensor(
            "var_axes",
            [1],
            dtype="INT32",
            data=np.asarray([1], dtype=np.int32),
        ),
        "mean": _tensor("mean", [1, 1, 1, 5]),
        "centered": _tensor("centered", [1, 3, 1, 5]),
        "square": _tensor("square", [1, 3, 1, 5]),
        "variance": _tensor("variance", [1, 1, 1, 5]),
        "epsilon": _tensor(
            "epsilon",
            [],
            data=np.asarray(1e-6, dtype=np.float32),
        ),
        "y": _tensor("y", [1, 1, 1, 5]),
    }
    transpose = (
        OperatorIR("TRANSPOSE", ["x_nhwc", "perm"], ["x_nchw"])
        if source == "pre"
        else OperatorIR("TRANSPOSE", ["x_nchw", "perm"], ["x_nhwc"])
    )
    model_ir.operators = [
        transpose,
        OperatorIR(
            "MEAN",
            ["x_nchw", "mean_axes"],
            ["mean"],
            options={"keepDims": True},
        ),
        OperatorIR("SUB", ["x_nchw", "mean"], ["centered"]),
        OperatorIR("MUL", ["centered", "centered"], ["square"]),
        OperatorIR(
            "MEAN",
            ["square", "var_axes"],
            ["variance"],
            options={"keepDims": True},
        ),
        OperatorIR("ADD", ["variance", "epsilon"], ["y"]),
    ]
    if fanout:
        model_ir.tensors["side"] = _tensor("side", [1, 3, 1, 5])
        model_ir.operators.append(OperatorIR("IDENTITY", ["centered"], ["side"]))
    return model_ir


@pytest.mark.parametrize("source", ["pre", "post"])
def test_layernorm_statistics_layout_characterization(source: str) -> None:
    model_ir = _model(source=source, fanout=False)

    if source == "pre":
        stats = _optimize_transpose_layernorm_stats_nhwc_propagation_chains(
            model_ir
        )
        stats_key = "optimized_transpose_layernorm_stats_nhwc_propagation_chains"
    else:
        stats = _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains(
            model_ir
        )
        stats_key = (
            "optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains"
        )

    assert stats[stats_key] == 1
    assert [operator.op_type for operator in model_ir.operators] == (
        ["MEAN", "SUB", "MUL", "MEAN", "ADD"]
        if source == "pre"
        else ["TRANSPOSE", "MEAN", "SUB", "MUL", "MEAN", "ADD"]
    )
    offset = 0 if source == "pre" else 1
    assert model_ir.operators[offset].inputs == ["x_nhwc", "mean_axes"]
    assert model_ir.operators[offset + 1].inputs == ["x_nhwc", "mean"]
    np.testing.assert_array_equal(
        model_ir.tensors["mean_axes"].data,
        np.asarray([3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["var_axes"].data,
        np.asarray([3], dtype=np.int32),
    )


@pytest.mark.parametrize("source", ["pre", "post"])
def test_layernorm_statistics_layout_rejects_centered_fanout(source: str) -> None:
    model_ir = _model(source=source, fanout=True)

    if source == "pre":
        stats = _optimize_transpose_layernorm_stats_nhwc_propagation_chains(
            model_ir
        )
        stats_key = "optimized_transpose_layernorm_stats_nhwc_propagation_chains"
    else:
        stats = _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains(
            model_ir
        )
        stats_key = (
            "optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains"
        )

    assert stats[stats_key] == 0
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        "MEAN",
        "SUB",
        "MUL",
        "MEAN",
        "ADD",
        "IDENTITY",
    ]
    np.testing.assert_array_equal(
        model_ir.tensors["mean_axes"].data,
        np.asarray([1], dtype=np.int32),
    )


@pytest.mark.parametrize("source", ["pre", "post"])
def test_layernorm_statistics_runner_reuses_one_index(
    source: str,
    monkeypatch,
) -> None:
    model_ir = _model(source=source, fanout=False)
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_layernorm_statistics_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    expected_key = (
        "optimized_transpose_layernorm_stats_nhwc_propagation_chains"
        if source == "pre"
        else "optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains"
    )
    assert stats[expected_key] == 1
    assert refresh_count == 1
    assert len(diagnostics) == 2
    assert sum(bool(event["changed"]) for event in diagnostics) == 1
    assert sum(
        int(event["metrics"]["snapshot_count"])
        for event in diagnostics
    ) == 1


@pytest.mark.parametrize("source", ["pre", "post"])
def test_layernorm_statistics_runner_rejects_fanout_before_snapshot(
    source: str,
) -> None:
    model_ir = _model(source=source, fanout=True)
    diagnostics: list[dict[str, object]] = []

    stats = run_layernorm_statistics_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert sum(stats.values()) == 0
    assert len(diagnostics) == 2
    assert all(event["changed"] is False for event in diagnostics)
    assert all(
        event["metrics"]["snapshot_count"] == 0
        for event in diagnostics
    )
