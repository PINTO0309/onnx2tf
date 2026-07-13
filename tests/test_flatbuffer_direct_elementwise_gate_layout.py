from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.elementwise_gate_layout import (
    run_elementwise_gate_layout_cleanup,
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


def _sum_gate_model(*, reduction_fanout: bool) -> ModelIR:
    model_ir = ModelIR("sum_logistic_muladd")
    model_ir.inputs = ["a_nhwc", "b_nhwc", "c_nhwc"]
    model_ir.outputs = ["z"] + (["reduction_side"] if reduction_fanout else [])
    shapes = {
        "a_nhwc": [1, 3, 5, 4],
        "b_nhwc": [1, 3, 5, 4],
        "c_nhwc": [1, 3, 5, 4],
        "a_nchw": [1, 4, 3, 5],
        "b_nchw": [1, 4, 3, 5],
        "c_nchw": [1, 4, 3, 5],
        "sum_nchw": [1, 1, 3, 5],
        "sigmoid_nchw": [1, 1, 3, 5],
        "inverse_nchw": [1, 1, 3, 5],
        "mul_a_nchw": [1, 4, 3, 5],
        "mul_b_nchw": [1, 4, 3, 5],
        "add_nchw": [1, 4, 3, 5],
        "y_nhwc": [1, 3, 5, 4],
        "z": [1, 3, 5, 4],
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
            "sum_axes": _tensor(
                "sum_axes",
                [1],
                dtype="INT32",
                data=np.asarray([1], dtype=np.int32),
            ),
            "one": _tensor(
                "one",
                [],
                data=np.asarray(1.0, dtype=np.float32),
            ),
        }
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["a_nhwc", "to_nchw"], ["a_nchw"]),
        OperatorIR("TRANSPOSE", ["b_nhwc", "to_nchw"], ["b_nchw"]),
        OperatorIR("TRANSPOSE", ["c_nhwc", "to_nchw"], ["c_nchw"]),
        OperatorIR(
            "SUM",
            ["c_nchw", "sum_axes"],
            ["sum_nchw"],
            options={"keepDims": True, "axes": [1]},
        ),
        OperatorIR("LOGISTIC", ["sum_nchw"], ["sigmoid_nchw"]),
        OperatorIR("SUB", ["one", "sigmoid_nchw"], ["inverse_nchw"]),
        OperatorIR("MUL", ["inverse_nchw", "a_nchw"], ["mul_a_nchw"]),
        OperatorIR("MUL", ["sigmoid_nchw", "b_nchw"], ["mul_b_nchw"]),
        OperatorIR("ADD", ["mul_a_nchw", "mul_b_nchw"], ["add_nchw"]),
        OperatorIR("TRANSPOSE", ["add_nchw", "to_nhwc"], ["y_nhwc"]),
        OperatorIR("RELU", ["y_nhwc"], ["z"]),
    ]
    if reduction_fanout:
        model_ir.tensors["reduction_side"] = _tensor(
            "reduction_side",
            [1, 4, 3, 5],
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["c_nchw"], ["reduction_side"])
        )
    return model_ir


def test_sum_logistic_muladd_layout_characterization() -> None:
    model_ir = _sum_gate_model(reduction_fanout=False)

    stats = _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains(
        model_ir
    )

    assert stats["optimized_transpose_sum_logistic_muladd_prepost_nhwc_chains"] == 1
    assert all(operator.op_type != "TRANSPOSE" for operator in model_ir.operators)
    np.testing.assert_array_equal(
        model_ir.tensors["sum_axes"].data,
        np.asarray([3], dtype=np.int32),
    )
    add_op = next(
        operator for operator in model_ir.operators if operator.op_type == "ADD"
    )
    assert add_op.outputs == ["y_nhwc"]
    relu_op = next(
        operator for operator in model_ir.operators if operator.op_type == "RELU"
    )
    assert relu_op.inputs == ["y_nhwc"]


def test_sum_logistic_muladd_layout_rejects_reduction_fanout() -> None:
    model_ir = _sum_gate_model(reduction_fanout=True)

    stats = _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains(
        model_ir
    )

    assert stats["optimized_transpose_sum_logistic_muladd_prepost_nhwc_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators].count(
        "TRANSPOSE"
    ) == 4
    np.testing.assert_array_equal(
        model_ir.tensors["sum_axes"].data,
        np.asarray([1], dtype=np.int32),
    )


def test_elementwise_gate_runner_reuses_one_index(monkeypatch) -> None:
    model_ir = _sum_gate_model(reduction_fanout=False)
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_elementwise_gate_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["optimized_transpose_sum_logistic_muladd_prepost_nhwc_chains"] == 1
    assert refresh_count == 1
    assert len(diagnostics) == 4
    assert sum(bool(event["changed"]) for event in diagnostics) == 1
    assert sum(
        int(event["metrics"]["snapshot_count"])
        for event in diagnostics
    ) == 1


def test_elementwise_gate_runner_rejects_reduction_fanout_before_snapshot() -> None:
    model_ir = _sum_gate_model(reduction_fanout=True)
    diagnostics: list[dict[str, object]] = []

    stats = run_elementwise_gate_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert sum(stats.values()) == 0
    assert len(diagnostics) == 4
    assert all(event["changed"] is False for event in diagnostics)
    assert all(
        event["metrics"]["snapshot_count"] == 0
        for event in diagnostics
    )
