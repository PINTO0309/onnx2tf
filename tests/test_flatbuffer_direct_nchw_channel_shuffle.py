from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather,
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
    model_ir = ModelIR("nchw_channel_shuffle")
    model_ir.inputs = ["x_nchw"]
    model_ir.outputs = ["y_nchw"] + (["side"] if fanout else [])
    model_ir.tensors = {
        "x_nchw": _tensor("x_nchw", [1, 8, 3, 5]),
        "shape_r1": _tensor(
            "shape_r1",
            [5],
            dtype="INT32",
            data=np.asarray([1, 2, 4, 3, 5], dtype=np.int32),
        ),
        "r1": _tensor("r1", [1, 2, 4, 3, 5]),
        "perm": _tensor(
            "perm",
            [5],
            dtype="INT32",
            data=np.asarray([0, 2, 1, 3, 4], dtype=np.int32),
        ),
        "t1": _tensor("t1", [1, 4, 2, 3, 5]),
        "shape_r2": _tensor(
            "shape_r2",
            [4],
            dtype="INT32",
            data=np.asarray([1, 8, 3, 5], dtype=np.int32),
        ),
        "y_nchw": _tensor("y_nchw", [1, 8, 3, 5]),
    }
    model_ir.operators = [
        OperatorIR("RESHAPE", ["x_nchw", "shape_r1"], ["r1"]),
        OperatorIR("TRANSPOSE", ["r1", "perm"], ["t1"]),
        OperatorIR("RESHAPE", ["t1", "shape_r2"], ["y_nchw"]),
    ]
    if fanout:
        model_ir.tensors["side"] = _tensor("side", [1, 2, 4, 3, 5])
        model_ir.operators.append(OperatorIR("IDENTITY", ["r1"], ["side"]))
    return model_ir


def test_nchw_channel_shuffle_characterization_rewrites_to_gather() -> None:
    model_ir = _model(fanout=False)

    stats = _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
        model_ir
    )

    assert stats[
        "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather"
    ] == 1
    assert [operator.op_type for operator in model_ir.operators] == ["GATHER"]
    gather_op = model_ir.operators[0]
    assert gather_op.inputs[0] == "x_nchw"
    assert gather_op.outputs == ["y_nchw"]
    assert gather_op.options == {"axis": 1, "batchDims": 0}
    indices = np.asarray(model_ir.tensors[gather_op.inputs[1]].data).reshape(-1)
    np.testing.assert_array_equal(
        indices,
        np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int32),
    )


def test_nchw_channel_shuffle_characterization_rejects_intermediate_fanout() -> None:
    model_ir = _model(fanout=True)

    stats = _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
        model_ir
    )

    assert stats[
        "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather"
    ] == 0
    assert [operator.op_type for operator in model_ir.operators[:3]] == [
        "RESHAPE",
        "TRANSPOSE",
        "RESHAPE",
    ]
