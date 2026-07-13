from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_gather_transpose_nhwc_channel_chains,
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


def _model(*, public_post: bool) -> ModelIR:
    model_ir = ModelIR("transpose_gather_channel_fanout")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["y0"] if public_post else ["out0", "out1"]
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 5, 7, 4]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_nchw": _tensor("x_nchw", [1, 4, 5, 7]),
        "indices": _tensor(
            "indices",
            [4],
            dtype="INT32",
            data=np.asarray([2, 0, 3, 1], dtype=np.int32),
        ),
        "g_nchw": _tensor("g_nchw", [1, 4, 5, 7]),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y0": _tensor("y0", [1, 5, 7, 4]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "to_nchw"], ["x_nchw"]),
        OperatorIR(
            "GATHER",
            ["x_nchw", "indices"],
            ["g_nchw"],
            options={"axis": 1, "batchDims": 0},
        ),
        OperatorIR("TRANSPOSE", ["g_nchw", "to_nhwc"], ["y0"]),
    ]
    if not public_post:
        model_ir.tensors.update(
            {
                "y1": _tensor("y1", [1, 5, 7, 4]),
                "out0": _tensor("out0", [1, 5, 7, 4]),
                "out1": _tensor("out1", [1, 5, 7, 4]),
            }
        )
        model_ir.operators.extend(
            [
                OperatorIR("TRANSPOSE", ["g_nchw", "to_nhwc"], ["y1"]),
                OperatorIR("RELU", ["y0"], ["out0"]),
                OperatorIR("RELU", ["y1"], ["out1"]),
            ]
        )
    return model_ir


def test_transpose_gather_channel_fanout_characterization() -> None:
    model_ir = _model(public_post=False)

    stats = _optimize_transpose_gather_transpose_nhwc_channel_chains(model_ir)

    assert stats["optimized_transpose_gather_transpose_nhwc_channel_chains"] == 1
    assert [operator.op_type for operator in model_ir.operators].count("TRANSPOSE") == 0
    gather_op = model_ir.operators[0]
    assert gather_op.op_type == "GATHER"
    assert gather_op.inputs == ["x_nhwc", "indices"]
    assert gather_op.outputs == ["y0"]
    assert gather_op.options["axis"] == 3
    assert model_ir.operators[2].inputs == ["y0"]


def test_transpose_gather_channel_fanout_rejects_public_post() -> None:
    model_ir = _model(public_post=True)

    stats = _optimize_transpose_gather_transpose_nhwc_channel_chains(model_ir)

    assert stats["optimized_transpose_gather_transpose_nhwc_channel_chains"] == 0
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        "GATHER",
        "TRANSPOSE",
    ]
