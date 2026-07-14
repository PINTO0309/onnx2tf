from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_package_selection import (
    _should_prefer_saved_model_backed_package,
    _should_prefer_tflite_backed_package,
)


def _append_ops(model_ir: ModelIR, op_type: str, count: int) -> None:
    for index in range(count):
        model_ir.operators.append(
            OperatorIR(
                op_type=op_type,
                inputs=[f"{op_type.lower()}_{index}_in"],
                outputs=[f"{op_type.lower()}_{index}_out"],
            )
        )


def _append_nhwc_named_tensors(model_ir: ModelIR, count: int) -> None:
    for index in range(count):
        name = f"feature_{index}_nhwc"
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 2, 2, 4],
            logical_layout="NHWC",
        )


def test_simple_graph_does_not_prefer_backed_packages() -> None:
    model_ir = ModelIR(name="simple")
    _append_ops(model_ir, "ADD", 1)

    assert not _should_prefer_tflite_backed_package(model_ir)
    assert not _should_prefer_saved_model_backed_package(model_ir)


@pytest.mark.parametrize(
    "guard_op_type",
    [
        "WHILE",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    ],
)
def test_recurrent_and_control_ops_disable_backed_package_preference(
    guard_op_type: str,
) -> None:
    model_ir = ModelIR(name="guarded")
    _append_ops(model_ir, guard_op_type, 1)
    _append_ops(model_ir, "TRANSPOSE_CONV", 1)

    assert not _should_prefer_tflite_backed_package(model_ir)


@pytest.mark.parametrize("input_name", ["length", "seq_len", "audio-lengths"])
def test_length_like_public_inputs_disable_backed_package_preference(
    input_name: str,
) -> None:
    model_ir = ModelIR(name="length_guard", inputs=[input_name])
    _append_ops(model_ir, "TRANSPOSE_CONV", 1)

    assert not _should_prefer_tflite_backed_package(model_ir)


def test_transpose_conv_and_channel_first_softmax_prefer_backed_package() -> None:
    transpose_model = ModelIR(name="transpose_conv")
    _append_ops(transpose_model, "TRANSPOSE_CONV", 1)
    assert _should_prefer_tflite_backed_package(transpose_model)

    softmax_model = ModelIR(name="channel_first_softmax")
    softmax_model.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        logical_layout="NCHW",
    )
    softmax_model.operators.append(
        OperatorIR(op_type="SOFTMAX", inputs=["x"], outputs=["y"])
    )
    assert _should_prefer_tflite_backed_package(softmax_model)


def test_large_rank3_detection_signature_uses_structural_counts() -> None:
    model_ir = ModelIR(name="large_rank3")
    model_ir.outputs = ["output"]
    model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[1, 16, 80],
        logical_layout="NCW",
    )
    _append_ops(model_ir, "CONV_2D", 20)
    _append_ops(model_ir, "STRIDED_SLICE", 4)
    _append_ops(model_ir, "CONCATENATION", 4)

    assert _should_prefer_tflite_backed_package(model_ir)


@pytest.mark.parametrize(
    ("conv_count", "nhwc_count", "resize_count", "extra_op"),
    [
        (40, 40, 0, "SOFTMAX"),
        (60, 80, 2, None),
        (15, 30, 3, None),
    ],
)
def test_large_nhwc_signatures_use_declared_thresholds(
    conv_count: int,
    nhwc_count: int,
    resize_count: int,
    extra_op: str | None,
) -> None:
    model_ir = ModelIR(name="large_nhwc")
    _append_ops(model_ir, "CONV_2D", conv_count)
    _append_ops(model_ir, "RESIZE_BILINEAR", resize_count)
    if extra_op is not None:
        _append_ops(model_ir, extra_op, 1)
    _append_nhwc_named_tensors(model_ir, nhwc_count)

    assert _should_prefer_tflite_backed_package(model_ir)
    assert _should_prefer_saved_model_backed_package(model_ir)
