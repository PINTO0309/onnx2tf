from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.boundary_input_chains import (
    _optimize_boundary_input_transpose_batchmatmul_chains,
    _optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains,
    run_boundary_input_batchmatmul_cleanup,
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


def test_boundary_mul_sum_reshape_chain_moves_to_nhwc() -> None:
    model_ir = ModelIR("boundary_mul_sum_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_onnx_ncx_internal": _tensor(
            "x_onnx_ncx_internal", [1, 3, 2, 2]
        ),
        "scale": _tensor(
            "scale",
            [1, 3, 1, 1],
            data=np.asarray([[[[1.0]], [[2.0]], [[3.0]]]], dtype=np.float32),
        ),
        "mul_out": _tensor("mul_out", [1, 3, 2, 2]),
        "axes": _tensor(
            "axes",
            [1],
            dtype="INT32",
            data=np.asarray([1], dtype=np.int32),
        ),
        "sum_out": _tensor("sum_out", [1, 1, 2, 2]),
        "reshape_shape": _tensor(
            "reshape_shape",
            [4],
            dtype="INT32",
            data=np.asarray([1, 2, 2, 1], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 2, 2, 1]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "perm"],
            outputs=["x_onnx_ncx_internal"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["x_onnx_ncx_internal", "scale"],
            outputs=["mul_out"],
        ),
        OperatorIR(
            op_type="SUM",
            inputs=["mul_out", "axes"],
            outputs=["sum_out"],
            options={"keepDims": True, "axes": [1]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["sum_out", "reshape_shape"],
            outputs=["y"],
        ),
    ]

    stats = _optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains(
        model_ir
    )

    assert stats == {
        "rewritten_boundary_input_transpose_mul_sum_reshape_nhwc_chains": 1
    }
    assert [operator.op_type for operator in model_ir.operators] == [
        "MUL",
        "SUM",
        "RESHAPE",
    ]
    assert model_ir.operators[0].inputs == ["x", "scale"]
    assert model_ir.tensors["scale"].shape == [1, 1, 1, 3]
    assert model_ir.tensors["scale"].data is not None
    assert list(model_ir.tensors["scale"].data.shape) == [1, 1, 1, 3]
    assert model_ir.tensors["axes"].data is not None
    assert model_ir.tensors["axes"].data.tolist() == [3]
    assert model_ir.tensors["mul_out"].shape == [1, 2, 2, 3]
    assert model_ir.tensors["sum_out"].shape == [1, 2, 2, 1]
    assert "x_onnx_ncx_internal" not in model_ir.tensors
    assert "perm" not in model_ir.tensors


def test_boundary_batchmatmul_chain_removes_exclusive_transpose(monkeypatch) -> None:
    model_ir = ModelIR("boundary_batchmatmul")
    model_ir.inputs = ["x", "rhs"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_internal": _tensor("x_internal", [1, 3, 2, 2]),
        "rhs": _tensor("rhs", [1, 3, 2, 4]),
        "y": _tensor("y", [1, 3, 2, 4]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "perm"],
            outputs=["x_internal"],
        ),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["x_internal", "rhs"],
            outputs=["y"],
        ),
    ]

    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_boundary_input_batchmatmul_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"rewritten_boundary_input_transpose_batchmatmul_chains": 1}
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == "BATCH_MATMUL"
    assert model_ir.operators[0].inputs == ["x", "rhs"]
    assert model_ir.tensors["x"].shape == [1, 3, 2, 2]
    assert "x_internal" not in model_ir.tensors
    assert "perm" not in model_ir.tensors
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.boundary_input_batchmatmul"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1
    assert refresh_count == 1


def test_boundary_mul_sum_reshape_chain_preserves_fanout() -> None:
    model_ir = ModelIR("boundary_mul_sum_reshape_fanout")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y", "side"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_onnx_ncx_internal": _tensor(
            "x_onnx_ncx_internal", [1, 3, 2, 2]
        ),
        "scale": _tensor(
            "scale",
            [1],
            data=np.asarray([2.0], dtype=np.float32),
        ),
        "mul_out": _tensor("mul_out", [1, 3, 2, 2]),
        "axes": _tensor(
            "axes",
            [1],
            dtype="INT32",
            data=np.asarray([1], dtype=np.int32),
        ),
        "sum_out": _tensor("sum_out", [1, 1, 2, 2]),
        "reshape_shape": _tensor(
            "reshape_shape",
            [4],
            dtype="INT32",
            data=np.asarray([1, 2, 2, 1], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 2, 2, 1]),
        "side": _tensor("side", [1, 3, 2, 2]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "perm"],
            outputs=["x_onnx_ncx_internal"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["x_onnx_ncx_internal", "scale"],
            outputs=["mul_out"],
        ),
        OperatorIR(
            op_type="IDENTITY",
            inputs=["x_onnx_ncx_internal"],
            outputs=["side"],
        ),
        OperatorIR(
            op_type="SUM",
            inputs=["mul_out", "axes"],
            outputs=["sum_out"],
            options={"keepDims": True},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["sum_out", "reshape_shape"],
            outputs=["y"],
        ),
    ]

    stats = _optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains(
        model_ir
    )

    assert stats == {
        "rewritten_boundary_input_transpose_mul_sum_reshape_nhwc_chains": 0
    }
    assert model_ir.operators[0].op_type == "TRANSPOSE"
    assert model_ir.operators[1].inputs[0] == "x_onnx_ncx_internal"


def test_boundary_batchmatmul_chain_preserves_shared_model_input() -> None:
    model_ir = ModelIR("boundary_batchmatmul_shared_input")
    model_ir.inputs = ["x", "rhs"]
    model_ir.outputs = ["y", "side"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_internal": _tensor("x_internal", [1, 3, 2, 2]),
        "rhs": _tensor("rhs", [1, 3, 2, 4]),
        "y": _tensor("y", [1, 3, 2, 4]),
        "side": _tensor("side", [1, 2, 2, 3]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "perm"],
            outputs=["x_internal"],
        ),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["x_internal", "rhs"],
            outputs=["y"],
        ),
        OperatorIR(op_type="IDENTITY", inputs=["x"], outputs=["side"]),
    ]

    stats = _optimize_boundary_input_transpose_batchmatmul_chains(model_ir)

    assert stats == {"rewritten_boundary_input_transpose_batchmatmul_chains": 0}
    assert model_ir.operators[0].op_type == "TRANSPOSE"
    assert model_ir.operators[1].inputs[0] == "x_internal"
