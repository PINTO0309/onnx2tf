from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains,
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


def _model(*, boundary: str | None = None) -> ModelIR:
    model_ir = ModelIR("three_dimensional_complementary_gate")
    model_ir.inputs = ["base_nhwc", "skip_ndhwc", "gate_ndhwc"]
    model_ir.outputs = ["y0", "y1"]
    bad_perm = boundary == "permutation"
    bad_reshape = boundary == "reshape_rank"
    reshape_values = [1, 8, 1, 6, 10] if not bad_reshape else [1, 8, 6, 10]
    model_ir.tensors = {
        "perm4_pre": _tensor(
            "perm4_pre",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "perm5_pre": _tensor(
            "perm5_pre",
            [5],
            dtype="INT32",
            data=np.asarray(
                [0, 1, 2, 3, 4] if bad_perm else [0, 4, 1, 2, 3],
                dtype=np.int32,
            ),
        ),
        "perm5_post": _tensor(
            "perm5_post",
            [5],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 4, 1], dtype=np.int32),
        ),
        "base_reshape_shape": _tensor(
            "base_reshape_shape",
            [len(reshape_values)],
            dtype="INT64",
            data=np.asarray(reshape_values, dtype=np.int64),
        ),
    }
    shapes = {
        "base_nhwc": [1, 6, 10, 8],
        "base_nchw": [1, 8, 6, 10],
        "base_ncdhw": [1, 8, 1, 6, 10],
        "skip_ndhwc": [1, 6, 6, 10, 8],
        "gate_ndhwc": [1, 6, 6, 10, 8],
        "add0_ndhwc": [1, 6, 6, 10, 8],
        "add1_ndhwc": [1, 6, 6, 10, 8],
        "y0": [1, 6, 6, 10, 8],
        "y1": [1, 6, 6, 10, 8],
    }
    for name in [
        "skip_ncdhw",
        "skip_leaky_ncdhw",
        "add0_ncdhw",
        "gate_ncdhw",
        "gate_sig_ncdhw",
        "mul1_ncdhw",
        "add1_ncdhw",
    ]:
        shapes[name] = [1, 8, 6, 6, 10]
    for name, shape in shapes.items():
        model_ir.tensors[name] = _tensor(name, shape)

    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["base_nhwc", "perm4_pre"], ["base_nchw"]),
        OperatorIR(
            "RESHAPE",
            ["base_nchw", "base_reshape_shape"],
            ["base_ncdhw"],
        ),
        OperatorIR("TRANSPOSE", ["skip_ndhwc", "perm5_pre"], ["skip_ncdhw"]),
        OperatorIR(
            "LEAKY_RELU",
            ["skip_ncdhw"],
            ["skip_leaky_ncdhw"],
            options={"alpha": 0.1},
        ),
        OperatorIR(
            "ADD",
            ["skip_leaky_ncdhw", "base_ncdhw"],
            ["add0_ncdhw"],
        ),
        OperatorIR("TRANSPOSE", ["add0_ncdhw", "perm5_post"], ["add0_ndhwc"]),
        OperatorIR("TRANSPOSE", ["gate_ndhwc", "perm5_pre"], ["gate_ncdhw"]),
        OperatorIR("LOGISTIC", ["gate_ncdhw"], ["gate_sig_ncdhw"]),
        OperatorIR("MUL", ["gate_sig_ncdhw", "base_ncdhw"], ["mul1_ncdhw"]),
        OperatorIR(
            "ADD",
            ["mul1_ncdhw", "skip_leaky_ncdhw"],
            ["add1_ncdhw"],
        ),
        OperatorIR("TRANSPOSE", ["add1_ncdhw", "perm5_post"], ["add1_ndhwc"]),
        OperatorIR("RELU", ["add0_ndhwc"], ["y0"]),
        OperatorIR("RELU", ["add1_ndhwc"], ["y1"]),
    ]
    side_sources = {
        "base_fanout": "base_ncdhw",
        "gate_fanout": "gate_sig_ncdhw",
    }
    if boundary in side_sources:
        source = side_sources[boundary]
        model_ir.tensors["side"] = _tensor(
            "side",
            list(model_ir.tensors[source].shape),
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    if boundary == "public_intermediate":
        model_ir.outputs.append("add1_ncdhw")
    return model_ir


def test_3d_gate_layout_characterization() -> None:
    model_ir = _model()

    stats = _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains(model_ir)

    assert stats["optimized_transpose_3d_leaky_logistic_muladd_ndhwc_chains"] == 1
    assert all(operator.op_type != "TRANSPOSE" for operator in model_ir.operators)
    reshape_op = next(
        operator for operator in model_ir.operators if operator.op_type == "RESHAPE"
    )
    assert reshape_op.inputs == ["base_nhwc", "base_reshape_shape"]
    np.testing.assert_array_equal(
        model_ir.tensors["base_reshape_shape"].data,
        np.asarray([1, 1, 6, 10, 8], dtype=np.int64),
    )
    assert next(
        operator for operator in model_ir.operators if operator.op_type == "LEAKY_RELU"
    ).inputs == ["skip_ndhwc"]
    assert next(
        operator for operator in model_ir.operators if operator.op_type == "LOGISTIC"
    ).inputs == ["gate_ndhwc"]
    assert {
        tuple(operator.outputs)
        for operator in model_ir.operators
        if operator.op_type == "ADD"
    } == {("add0_ndhwc",), ("add1_ndhwc",)}


@pytest.mark.parametrize(
    "boundary",
    [
        "base_fanout",
        "gate_fanout",
        "public_intermediate",
        "permutation",
        "reshape_rank",
    ],
)
def test_3d_gate_layout_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original_operators = [
        (operator.op_type, list(operator.inputs), list(operator.outputs))
        for operator in model_ir.operators
    ]

    stats = _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains(model_ir)

    assert stats["optimized_transpose_3d_leaky_logistic_muladd_ndhwc_chains"] == 0
    assert [
        (operator.op_type, list(operator.inputs), list(operator.outputs))
        for operator in model_ir.operators
    ] == original_operators
    assert [operator.op_type for operator in model_ir.operators].count(
        "TRANSPOSE"
    ) == 5
