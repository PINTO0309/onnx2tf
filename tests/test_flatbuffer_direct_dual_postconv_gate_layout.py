from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.dual_postconv_gate_layout import (
    run_dual_postconv_gate_layout_cleanup,
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
        is_variable=data is None,
    )


def _model(
    *,
    gate_fanout: bool = False,
    data_fanout: bool = False,
    public_intermediate: bool = False,
) -> ModelIR:
    model_ir = ModelIR("dual_postconv_gate")
    model_ir.inputs = ["gate_nhwc", "a_nhwc", "b_nhwc"]
    model_ir.outputs = ["z0", "z1"]
    model_ir.tensors = {
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
        "one": _tensor(
            "one",
            [],
            data=np.asarray(1.0, dtype=np.float32),
        ),
    }
    for name in ["gate_nhwc", "a_nhwc", "b_nhwc", "y0", "y1", "z0", "z1"]:
        model_ir.tensors[name] = _tensor(name, [1, 3, 5, 4])
    for name in [
        "gate_nchw",
        "sigmoid_nchw",
        "inverse_nchw",
        "a_nchw",
        "b_nchw",
        "mul_sig_nchw",
        "mul_sub_nchw",
        "add_sig_nchw",
        "add_sub_nchw",
    ]:
        model_ir.tensors[name] = _tensor(name, [1, 4, 3, 5])
    for branch in range(2):
        model_ir.tensors[f"weight{branch}"] = _tensor(
            f"weight{branch}",
            [4, 1, 1, 4],
            data=np.ones([4, 1, 1, 4], dtype=np.float32),
        )
        model_ir.tensors[f"bias{branch}"] = _tensor(
            f"bias{branch}",
            [4],
            data=np.zeros([4], dtype=np.float32),
        )

    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["gate_nhwc", "to_nchw"], ["gate_nchw"]),
        OperatorIR("LOGISTIC", ["gate_nchw"], ["sigmoid_nchw"]),
        OperatorIR("SUB", ["one", "sigmoid_nchw"], ["inverse_nchw"]),
        OperatorIR("TRANSPOSE", ["a_nhwc", "to_nchw"], ["a_nchw"]),
        OperatorIR("TRANSPOSE", ["b_nhwc", "to_nchw"], ["b_nchw"]),
        OperatorIR("MUL", ["sigmoid_nchw", "a_nchw"], ["mul_sig_nchw"]),
        OperatorIR("MUL", ["inverse_nchw", "b_nchw"], ["mul_sub_nchw"]),
        OperatorIR("ADD", ["b_nchw", "mul_sig_nchw"], ["add_sig_nchw"]),
        OperatorIR("ADD", ["mul_sub_nchw", "a_nchw"], ["add_sub_nchw"]),
        OperatorIR("TRANSPOSE", ["add_sig_nchw", "to_nhwc"], ["y0"]),
        OperatorIR("TRANSPOSE", ["add_sub_nchw", "to_nhwc"], ["y1"]),
        OperatorIR("CONV_2D", ["y0", "weight0", "bias0"], ["z0"]),
        OperatorIR("CONV_2D", ["y1", "weight1", "bias1"], ["z1"]),
    ]

    if gate_fanout:
        model_ir.tensors["gate_side"] = _tensor("gate_side", [1, 4, 3, 5])
        model_ir.outputs.append("gate_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["sigmoid_nchw"], ["gate_side"])
        )
    if data_fanout:
        model_ir.tensors["data_side"] = _tensor("data_side", [1, 4, 3, 5])
        model_ir.outputs.append("data_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["a_nchw"], ["data_side"])
        )
    if public_intermediate:
        model_ir.outputs.append("add_sig_nchw")
    return model_ir


def test_dual_postconv_gate_layout_characterization() -> None:
    model_ir = _model()

    stats = _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains"
    ] == 1
    assert all(operator.op_type != "TRANSPOSE" for operator in model_ir.operators)
    assert next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["sigmoid_nchw"]
    ).inputs == ["gate_nhwc"]
    assert next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["mul_sig_nchw"]
    ).inputs == ["sigmoid_nchw", "a_nhwc"]
    assert next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["mul_sub_nchw"]
    ).inputs == ["inverse_nchw", "b_nhwc"]
    assert next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "ADD" and "mul_sig_nchw" in operator.inputs
    ).outputs == ["y0"]
    assert next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "ADD" and "mul_sub_nchw" in operator.inputs
    ).outputs == ["y1"]
    assert {
        tuple(operator.inputs)
        for operator in model_ir.operators
        if operator.op_type == "CONV_2D"
    } == {
        ("y0", "weight0", "bias0"),
        ("y1", "weight1", "bias1"),
    }


@pytest.mark.parametrize(
    "boundary",
    ["gate_fanout", "data_fanout", "public_intermediate"],
)
def test_dual_postconv_gate_layout_rejects_unsafe_boundary(
    boundary: str,
) -> None:
    model_ir = _model(**{boundary: True})
    original_operators = [
        (operator.op_type, list(operator.inputs), list(operator.outputs))
        for operator in model_ir.operators
    ]

    stats = _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains"
    ] == 0
    assert [
        (operator.op_type, list(operator.inputs), list(operator.outputs))
        for operator in model_ir.operators
    ] == original_operators
    assert [operator.op_type for operator in model_ir.operators].count(
        "TRANSPOSE"
    ) == 5


def test_dual_postconv_gate_layout_runner_reuses_one_index(monkeypatch) -> None:
    model_ir = _model()
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_dual_postconv_gate_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats[
        "optimized_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains"
    ] == 1
    assert refresh_count == 1
    assert diagnostics[0]["code"] == (
        "layout.dual_postconv_complementary_gate_nhwc"
    )
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


@pytest.mark.parametrize(
    "boundary",
    ["gate_fanout", "data_fanout", "public_intermediate"],
)
def test_dual_postconv_gate_layout_runner_rejects_before_snapshot(
    boundary: str,
) -> None:
    model_ir = _model(**{boundary: True})
    diagnostics: list[dict[str, object]] = []

    stats = run_dual_postconv_gate_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats[
        "optimized_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains"
    ] == 0
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
