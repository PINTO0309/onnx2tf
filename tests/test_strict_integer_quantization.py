import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.quantization import (
    StrictFullIntegerQuantizationError,
    TensorCalibrationRange,
    activation_qparams_from_range,
    build_full_integer_quantized_model_ir,
    fixed_activation_qparams_for_op,
    load_calibration_samples,
)


def test_activation_qparams_from_range_uint8_uses_calibrated_range() -> None:
    qparams = activation_qparams_from_range(
        min_value=0.0,
        max_value=255.0,
        dtype="uint8",
    )

    assert qparams.scale == [1.0]
    assert qparams.zero_point == [0]


def test_fixed_softmax_uint8_qparams_match_tflite_contract() -> None:
    qparams = fixed_activation_qparams_for_op(op_type="SOFTMAX", dtype="uint8")

    assert qparams is not None
    assert qparams.scale == [1.0 / 256.0]
    assert qparams.zero_point == [0]


def test_load_calibration_samples_requires_mean_std_entries() -> None:
    with pytest.raises(Exception, match="requires each calibration entry"):
        load_calibration_samples(
            custom_input_op_name_np_data_path=[["x", "unused.npy"]],
            input_names=["x"],
        )


def test_strict_full_integer_builder_quantizes_fc_softmax_ir() -> None:
    model_ir = ModelIR(name="fc_softmax")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["prob"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4])
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 4],
        data=np.asarray(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.1, -0.1, -0.2],
                [0.0, 0.3, 0.1, -0.4],
            ],
            dtype=np.float32,
        ),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[3],
        data=np.zeros((3,), dtype=np.float32),
    )
    model_ir.tensors["logits"] = TensorIR(name="logits", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["prob"] = TensorIR(name="prob", dtype="FLOAT32", shape=[1, 3])
    model_ir.operators = [
        OperatorIR(
            op_type="FULLY_CONNECTED",
            inputs=["x", "w", "b"],
            outputs=["logits"],
            options={"asymmetricQuantizeInputs": False},
        ),
        OperatorIR(op_type="SOFTMAX", inputs=["logits"], outputs=["prob"]),
    ]
    ranges = {
        "x": TensorCalibrationRange(0.0, 255.0, 2),
        "logits": TensorCalibrationRange(-2.0, 2.0, 2),
        "prob": TensorCalibrationRange(0.0, 1.0, 2),
    }

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        input_quant_dtype="uint8",
        output_quant_dtype="uint8",
        calibration_ranges=ranges,
    )

    assert quantized.tensors["x"].dtype == "UINT8"
    assert quantized.tensors["x"].quantization is not None
    assert quantized.tensors["x"].quantization.scale == [1.0]
    assert quantized.tensors["x"].quantization.zero_point == [0]
    assert quantized.tensors["w"].dtype == "INT8"
    assert quantized.tensors["b"].dtype == "INT32"
    assert quantized.tensors["prob"].dtype == "UINT8"
    assert quantized.tensors["prob"].quantization is not None
    assert quantized.tensors["prob"].quantization.scale == [1.0 / 256.0]
    assert quantized.tensors["prob"].quantization.zero_point == [0]


def test_strict_full_integer_builder_rejects_unsupported_float_op() -> None:
    model_ir = ModelIR(name="unsupported")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3])
    model_ir.operators = [
        OperatorIR(op_type="EXP", inputs=["x"], outputs=["y"]),
    ]

    with pytest.raises(StrictFullIntegerQuantizationError, match="Unsupported op"):
        build_full_integer_quantized_model_ir(
            model_ir,
            calibration_ranges={
                "x": TensorCalibrationRange(-1.0, 1.0, 1),
                "y": TensorCalibrationRange(0.0, 2.0, 1),
            },
        )
