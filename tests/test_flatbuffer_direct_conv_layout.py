from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_nchw_concat_global_pool_conv_axes,
    _repair_nchw_concat_transpose_conv_axes,
    _repair_mixed_nhwc_inputs_for_nchw_concat,
    _repair_singleton_nhwc_conv_input_reshapes,
    _repair_stale_nchw_to_nhwc_conv_input_transposes,
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


def test_repair_removes_stale_transpose_before_already_nhwc_conv() -> None:
    model_ir = ModelIR("stale_split_conv_adapter")
    model_ir.inputs = ["source_nhwc"]
    model_ir.outputs = ["conv_out"]
    _tensor(model_ir, "source_nhwc", [1, 64, 64, 72])
    _tensor(model_ir, "bad_adapter", [1, 64, 72, 64])
    _tensor(
        model_ir,
        "perm",
        [4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "filter",
        [36, 1, 1, 72],
        data=np.ones((36, 1, 1, 72), dtype=np.float32),
    )
    _tensor(
        model_ir,
        "bias",
        [36],
        data=np.zeros((36,), dtype=np.float32),
    )
    _tensor(model_ir, "conv_out", [1, 64, 72, 36])
    model_ir.operators = [
        OperatorIR(
            "TRANSPOSE",
            ["source_nhwc", "perm"],
            ["bad_adapter"],
        ),
        OperatorIR(
            "CONV_2D",
            ["bad_adapter", "filter", "bias"],
            ["conv_out"],
            {"padding": "SAME", "strideH": 1, "strideW": 1},
        ),
    ]

    stats = _repair_stale_nchw_to_nhwc_conv_input_transposes(model_ir)

    assert stats == {
        "repaired_stale_nchw_to_nhwc_conv_input_transposes": 1,
    }
    conv = next(op for op in model_ir.operators if op.op_type == "CONV_2D")
    assert conv.inputs[0] == "source_nhwc"
    assert model_ir.tensors["conv_out"].shape == [1, 64, 64, 36]
    assert "bad_adapter" not in model_ir.tensors


def test_repair_keeps_valid_nchw_to_nhwc_conv_transpose() -> None:
    model_ir = ModelIR("valid_conv_adapter")
    model_ir.inputs = ["source_nchw"]
    model_ir.outputs = ["conv_out"]
    _tensor(model_ir, "source_nchw", [1, 72, 64, 64])
    _tensor(model_ir, "conv_input", [1, 64, 64, 72])
    _tensor(
        model_ir,
        "perm",
        [4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "filter",
        [36, 1, 1, 72],
        data=np.ones((36, 1, 1, 72), dtype=np.float32),
    )
    _tensor(
        model_ir,
        "bias",
        [36],
        data=np.zeros((36,), dtype=np.float32),
    )
    _tensor(model_ir, "conv_out", [1, 64, 64, 36])
    model_ir.operators = [
        OperatorIR(
            "TRANSPOSE",
            ["source_nchw", "perm"],
            ["conv_input"],
        ),
        OperatorIR(
            "CONV_2D",
            ["conv_input", "filter", "bias"],
            ["conv_out"],
            {"padding": "SAME", "strideH": 1, "strideW": 1},
        ),
    ]

    stats = _repair_stale_nchw_to_nhwc_conv_input_transposes(model_ir)

    assert stats == {
        "repaired_stale_nchw_to_nhwc_conv_input_transposes": 0,
    }
    assert model_ir.operators[0].op_type == "TRANSPOSE"


def test_repair_adds_local_nchw_adapter_for_mixed_concat_input() -> None:
    model_ir = ModelIR("mixed_split_concat_layout")
    model_ir.inputs = ["split_keep_nhwc", "branch1", "branch2"]
    model_ir.outputs = ["stacked"]
    _tensor(model_ir, "split_keep_nhwc", [1, 64, 64, 72])
    _tensor(model_ir, "branch1", [1, 36, 64, 64])
    _tensor(model_ir, "branch2", [1, 36, 64, 64])
    _tensor(model_ir, "stacked", [1, 136, 64, 72])
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["split_keep_nhwc", "branch1", "branch2"],
            ["stacked"],
            {"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir)

    assert stats == {"repaired_mixed_nhwc_inputs_for_nchw_concat": 1}
    concat = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    adapter_name = concat.inputs[0]
    assert adapter_name != "split_keep_nhwc"
    assert model_ir.tensors[adapter_name].shape == [1, 72, 64, 64]
    assert model_ir.tensors["stacked"].shape == [1, 144, 64, 64]


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


def test_repair_restores_nchw_concat_axis_before_transpose_conv() -> None:
    model_ir = ModelIR("stale_concat_transpose_conv_axis")
    model_ir.inputs = ["left", "right"]
    model_ir.outputs = ["deconv_out"]
    _tensor(model_ir, "left", [1, 16, 8, 1])
    _tensor(model_ir, "right", [1, 16, 8, 1])
    _tensor(model_ir, "concat", [1, 16, 8, 2])
    _tensor(
        model_ir,
        "perm",
        [4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    _tensor(model_ir, "deconv_input", [1, 8, 2, 16])
    _tensor(
        model_ir,
        "output_shape",
        [4],
        data=np.asarray([1, 17, 1, 16], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "filter",
        [16, 3, 1, 32],
        data=np.ones((16, 3, 1, 32), dtype=np.float32),
    )
    _tensor(model_ir, "deconv_out", [1, 17, 1, 16])
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["left", "right"],
            ["concat"],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR("TRANSPOSE", ["concat", "perm"], ["deconv_input"]),
        OperatorIR(
            "TRANSPOSE_CONV",
            ["output_shape", "filter", "deconv_input"],
            ["deconv_out"],
            {"padding": "SAME", "strideH": 2, "strideW": 1},
        ),
    ]

    stats = _repair_nchw_concat_transpose_conv_axes(model_ir)

    assert stats == {"repaired_nchw_concat_transpose_conv_axes": 1}
    assert model_ir.operators[0].options["axis"] == 1
    assert model_ir.tensors["concat"].shape == [1, 32, 8, 1]
    assert model_ir.tensors["deconv_input"].shape == [1, 8, 1, 32]
    assert model_ir.tensors["deconv_out"].shape == [1, 17, 1, 16]


def test_repair_restores_nchw_concat_axis_before_global_pool_attention() -> None:
    model_ir = ModelIR("stale_concat_global_pool_conv_axis")
    model_ir.inputs = ["left", "right"]
    model_ir.outputs = ["conv_out"]
    _tensor(model_ir, "left", [1, 384, 8, 6])
    _tensor(model_ir, "right", [1, 384, 8, 6])
    _tensor(model_ir, "concat", [1, 384, 8, 12])
    _tensor(model_ir, "pool", [1, 384, 1, 1])
    _tensor(model_ir, "conv_input", [1, 1, 1, 384])
    _tensor(
        model_ir,
        "axes",
        [2],
        data=np.asarray([2, 3], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "reshape_shape",
        [4],
        data=np.asarray([1, 1, 1, 384], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "filter",
        [768, 1, 1, 768],
        data=np.ones((768, 1, 1, 768), dtype=np.float32),
    )
    _tensor(
        model_ir,
        "bias",
        [768],
        data=np.zeros((768,), dtype=np.float32),
    )
    _tensor(model_ir, "conv_out", [1, 1, 1, 768])
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["left", "right"],
            ["concat"],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR("MEAN", ["concat", "axes"], ["pool"], {"keepDims": True}),
        OperatorIR("RESHAPE", ["pool", "reshape_shape"], ["conv_input"]),
        OperatorIR(
            "CONV_2D",
            ["conv_input", "filter", "bias"],
            ["conv_out"],
            {"padding": "SAME", "strideH": 1, "strideW": 1},
        ),
    ]

    stats = _repair_nchw_concat_global_pool_conv_axes(model_ir)

    assert stats == {"repaired_nchw_concat_global_pool_conv_axes": 1}
    assert model_ir.operators[0].options["axis"] == 1
    assert model_ir.tensors["concat"].shape == [1, 768, 8, 6]
    assert model_ir.tensors["pool"].shape == [1, 768, 1, 1]
    assert model_ir.tensors["conv_input"].shape == [1, 1, 1, 768]
    assert np.asarray(model_ir.tensors["reshape_shape"].data).tolist() == [
        1,
        1,
        1,
        768,
    ]
