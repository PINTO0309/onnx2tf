from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains,
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
    model_ir = ModelIR("postadd_complementary_gate")
    model_ir.inputs = ["gate_nhwc", "a_nhwc", "b_nhwc"]
    model_ir.outputs = ["z"]
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
        "weight": _tensor(
            "weight",
            [4, 1, 1, 4],
            data=np.ones([4, 1, 1, 4], dtype=np.float32),
        ),
        "bias": _tensor(
            "bias",
            [4],
            data=np.zeros([4], dtype=np.float32),
        ),
    }
    for name in [
        "gate_nhwc",
        "a_nhwc",
        "b_nhwc",
        "ma_nhwc",
        "mb_nhwc",
        "merged_nhwc",
        "z",
    ]:
        model_ir.tensors[name] = _tensor(name, [1, 3, 5, 4])
    for name in [
        "gate_nchw",
        "sigmoid_nchw",
        "inverse_nchw",
        "a_nchw",
        "b_nchw",
        "ma_nchw",
        "mb_nchw",
    ]:
        model_ir.tensors[name] = _tensor(name, [1, 4, 3, 5])

    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["gate_nhwc", "to_nchw"], ["gate_nchw"]),
        OperatorIR("LOGISTIC", ["gate_nchw"], ["sigmoid_nchw"]),
        OperatorIR("SUB", ["one", "sigmoid_nchw"], ["inverse_nchw"]),
        OperatorIR("TRANSPOSE", ["a_nhwc", "to_nchw"], ["a_nchw"]),
        OperatorIR("TRANSPOSE", ["b_nhwc", "to_nchw"], ["b_nchw"]),
        OperatorIR("MUL", ["sigmoid_nchw", "a_nchw"], ["ma_nchw"]),
        OperatorIR("MUL", ["inverse_nchw", "b_nchw"], ["mb_nchw"]),
        OperatorIR("TRANSPOSE", ["ma_nchw", "to_nhwc"], ["ma_nhwc"]),
        OperatorIR("TRANSPOSE", ["mb_nchw", "to_nhwc"], ["mb_nhwc"]),
        OperatorIR("ADD", ["ma_nhwc", "mb_nhwc"], ["merged_nhwc"]),
        OperatorIR(
            "CONV_2D",
            ["merged_nhwc", "weight", "bias"],
            ["z"],
        ),
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
        model_ir.outputs.append("ma_nchw")
    return model_ir


def test_postadd_gate_layout_characterization() -> None:
    model_ir = _model()

    stats = _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains(model_ir)

    assert stats[
        "optimized_transpose_logistic_sub_mul_postadd_nhwc_chains"
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
        if operator.outputs == ["ma_nhwc"]
    ).inputs == ["sigmoid_nchw", "a_nhwc"]
    assert next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["mb_nhwc"]
    ).inputs == ["inverse_nchw", "b_nhwc"]
    assert next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["merged_nhwc"]
    ).inputs == ["ma_nhwc", "mb_nhwc"]


@pytest.mark.parametrize(
    "boundary",
    ["gate_fanout", "data_fanout", "public_intermediate"],
)
def test_postadd_gate_layout_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(**{boundary: True})
    original_operators = [
        (operator.op_type, list(operator.inputs), list(operator.outputs))
        for operator in model_ir.operators
    ]

    stats = _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains(model_ir)

    assert stats[
        "optimized_transpose_logistic_sub_mul_postadd_nhwc_chains"
    ] == 0
    assert [
        (operator.op_type, list(operator.inputs), list(operator.outputs))
        for operator in model_ir.operators
    ] == original_operators
    assert [operator.op_type for operator in model_ir.operators].count(
        "TRANSPOSE"
    ) == 5
