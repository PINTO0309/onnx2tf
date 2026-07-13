from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_unary_mean_terminal_nhwc_chains,
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


def _model(*, fanout: str | None = None, inverse_tail: bool = False) -> ModelIR:
    model_ir = ModelIR("terminal_mean_layout")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y_nhwc" if inverse_tail else "y"]
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 4, 4, 8]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 8, 4, 4]),
        "unary": _tensor("unary", [1, 8, 4, 4]),
        "axes": _tensor(
            "axes",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "mean": _tensor("mean", [1, 8, 1, 1]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "to_nchw"], ["x_nchw"]),
        OperatorIR("RELU6", ["x_nchw"], ["unary"]),
        OperatorIR(
            "MEAN",
            ["unary", "axes"],
            ["mean"],
            options={"keepDims": True},
        ),
    ]
    if inverse_tail:
        model_ir.tensors["to_nhwc"] = _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        )
        model_ir.tensors["y_nhwc"] = _tensor("y_nhwc", [1, 1, 1, 8])
        model_ir.operators.append(
            OperatorIR("TRANSPOSE", ["mean", "to_nhwc"], ["y_nhwc"])
        )
    else:
        model_ir.tensors["reshape_shape"] = _tensor(
            "reshape_shape",
            [2],
            dtype="INT32",
            data=np.asarray([1, 8], dtype=np.int32),
        )
        model_ir.tensors["y"] = _tensor("y", [1, 8])
        model_ir.operators.append(
            OperatorIR("RESHAPE", ["mean", "reshape_shape"], ["y"])
        )
    if fanout is not None:
        fanout_input = "x_nchw" if fanout == "pre" else "unary"
        model_ir.tensors["side"] = _tensor(
            "side",
            [1, 8, 4, 4],
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", [fanout_input], ["side"])
        )
    return model_ir


def test_terminal_mean_layout_keeps_shared_pre_transpose() -> None:
    model_ir = _model(fanout="pre")

    stats = _optimize_transpose_pre_unary_mean_terminal_nhwc_chains(model_ir)

    assert stats["optimized_transpose_pre_unary_mean_terminal_nhwc_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        "RELU6",
        "MEAN",
        "RESHAPE",
        "IDENTITY",
    ]
    assert model_ir.operators[1].inputs == ["x_nhwc"]
    assert model_ir.operators[4].inputs == ["x_nchw"]


def test_terminal_mean_layout_rejects_unary_fanout() -> None:
    model_ir = _model(fanout="unary")

    stats = _optimize_transpose_pre_unary_mean_terminal_nhwc_chains(model_ir)

    assert stats["optimized_transpose_pre_unary_mean_terminal_nhwc_chains"] == 0
    assert model_ir.operators[1].inputs == ["x_nchw"]
    np.testing.assert_array_equal(
        model_ir.tensors["axes"].data,
        np.asarray([2, 3], dtype=np.int32),
    )


def test_terminal_mean_layout_defers_inverse_transpose_tail() -> None:
    model_ir = _model(inverse_tail=True)

    stats = _optimize_transpose_pre_unary_mean_terminal_nhwc_chains(model_ir)

    assert stats["optimized_transpose_pre_unary_mean_terminal_nhwc_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        "RELU6",
        "MEAN",
        "TRANSPOSE",
    ]
