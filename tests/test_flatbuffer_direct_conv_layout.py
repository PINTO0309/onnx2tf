from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_singleton_nhwc_conv_input_reshapes,
)


def _tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def test_repair_removes_stale_singleton_reshape_before_nhwc_conv() -> None:
    model_ir = ModelIR("stale_singleton_conv_adapter")
    model_ir.inputs = ["source_nhwc"]
    model_ir.outputs = ["y_nchw"]
    _tensor(model_ir, "source_nhwc", [1, 1, 1, 64])
    _tensor(model_ir, "bad_adapter", [1, 1, 64, 1])
    _tensor(
        model_ir,
        "filter",
        [52, 1, 1, 64],
        data=np.ones((52, 1, 1, 64), dtype=np.float32),
    )
    _tensor(
        model_ir,
        "bias",
        [52],
        data=np.zeros((52,), dtype=np.float32),
    )
    _tensor(model_ir, "conv_out", [1, 1, 64, 52])
    _tensor(model_ir, "y_nchw", [1, 52, 1, 1])
    _tensor(
        model_ir,
        "bad_shape",
        [4],
        data=np.asarray([1, 1, 64, 1], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "output_shape",
        [4],
        data=np.asarray([1, 52, 1, 1], dtype=np.int32),
    )
    model_ir.operators = [
        OperatorIR("RESHAPE", ["source_nhwc", "bad_shape"], ["bad_adapter"]),
        OperatorIR(
            "CONV_2D",
            ["bad_adapter", "filter", "bias"],
            ["conv_out"],
            {"padding": "SAME", "strideH": 1, "strideW": 1},
        ),
        OperatorIR("RESHAPE", ["conv_out", "output_shape"], ["y_nchw"]),
    ]

    stats = _repair_singleton_nhwc_conv_input_reshapes(model_ir)

    assert stats == {"repaired_singleton_nhwc_conv_input_reshapes": 1}
    conv = next(op for op in model_ir.operators if op.op_type == "CONV_2D")
    assert conv.inputs[0] == "source_nhwc"
    assert model_ir.tensors["conv_out"].shape == [1, 1, 1, 52]
    assert "bad_adapter" not in model_ir.tensors
