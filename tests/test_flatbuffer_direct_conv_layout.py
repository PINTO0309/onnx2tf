from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_nchw_concat_transpose_conv_axes,
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


def test_repair_restores_nchw_concat_axis_before_conv_transpose() -> None:
    model_ir = ModelIR("stale_concat_conv_axis")
    model_ir.inputs = ["left", "right"]
    model_ir.outputs = ["conv_out"]
    _tensor(model_ir, "left", [1, 384, 7, 7])
    _tensor(model_ir, "right", [1, 384, 7, 7])
    _tensor(model_ir, "concat", [1, 384, 7, 14])
    _tensor(
        model_ir,
        "perm",
        [4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    _tensor(model_ir, "conv_input", [1, 7, 14, 384])
    _tensor(
        model_ir,
        "filter",
        [2048, 1, 1, 768],
        data=np.ones((2048, 1, 1, 768), dtype=np.float32),
    )
    _tensor(
        model_ir,
        "bias",
        [2048],
        data=np.zeros((2048,), dtype=np.float32),
    )
    _tensor(model_ir, "conv_out", [1, 7, 14, 2048])
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["left", "right"],
            ["concat"],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR("TRANSPOSE", ["concat", "perm"], ["conv_input"]),
        OperatorIR(
            "CONV_2D",
            ["conv_input", "filter", "bias"],
            ["conv_out"],
            {"padding": "SAME", "strideH": 1, "strideW": 1},
        ),
    ]

    stats = _repair_nchw_concat_transpose_conv_axes(model_ir)

    assert stats == {"repaired_nchw_concat_transpose_conv_axes": 1}
    assert model_ir.operators[0].options["axis"] == 1
    assert model_ir.tensors["concat"].shape == [1, 768, 7, 7]
    assert model_ir.tensors["conv_input"].shape == [1, 7, 7, 768]
    assert model_ir.tensors["conv_out"].shape == [1, 7, 7, 2048]
