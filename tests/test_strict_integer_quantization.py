import os
import tempfile

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.model_writer import write_model_file
from onnx2tf.tflite_builder.quantization import (
    StrictFullIntegerQuantizationError,
    TensorCalibrationRange,
    activation_qparams_from_range,
    build_full_integer_quantized_model_ir,
    fixed_activation_qparams_for_op,
    load_calibration_samples,
    strict_int16_activation_skip_reasons,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module


Interpreter = pytest.importorskip("ai_edge_litert.interpreter").Interpreter


def _calibration_ranges_for_float_activations(model_ir: ModelIR) -> dict[str, TensorCalibrationRange]:
    return {
        name: TensorCalibrationRange(-1.0, 1.0, 1)
        for name, tensor in model_ir.tensors.items()
        if str(tensor.dtype).upper() == "FLOAT32" and tensor.data is None
    }


def _assert_litert_allocates(model_ir: ModelIR) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        schema = load_schema_module(output_folder_path=tmpdir)
        tflite_path = os.path.join(tmpdir, f"{model_ir.name}.tflite")
        write_model_file(
            schema_tflite=schema,
            model_ir=model_ir,
            output_tflite_path=tflite_path,
            timing={},
        )
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()


def _quantize_input_array(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    dtype = np.dtype(detail["dtype"])
    if dtype == np.dtype(np.int8):
        qmin, qmax = -128, 127
    elif dtype == np.dtype(np.uint8):
        qmin, qmax = 0, 255
    elif dtype == np.dtype(np.int16):
        qmin, qmax = -32768, 32767
    else:
        raise AssertionError(f"Unexpected quantized input dtype: {dtype}")
    quantized = np.round(np.asarray(values, dtype=np.float32) / float(scale)) + int(zero_point)
    return np.clip(quantized, qmin, qmax).astype(dtype)


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
        OperatorIR(op_type="SIN", inputs=["x"], outputs=["y"]),
    ]

    with pytest.raises(StrictFullIntegerQuantizationError, match="Unsupported op"):
        build_full_integer_quantized_model_ir(
            model_ir,
            calibration_ranges={
                "x": TensorCalibrationRange(-1.0, 1.0, 1),
                "y": TensorCalibrationRange(0.0, 2.0, 1),
            },
        )


def test_fixed_tanh_qparams_match_tflite_contract() -> None:
    int8_qparams = fixed_activation_qparams_for_op(op_type="TANH", dtype="int8")
    uint8_qparams = fixed_activation_qparams_for_op(op_type="TANH", dtype="uint8")
    int16_qparams = fixed_activation_qparams_for_op(op_type="TANH", dtype="int16")

    assert int8_qparams is not None
    assert int8_qparams.scale == [1.0 / 128.0]
    assert int8_qparams.zero_point == [0]
    assert uint8_qparams is not None
    assert uint8_qparams.scale == [1.0 / 128.0]
    assert uint8_qparams.zero_point == [128]
    assert int16_qparams is not None
    assert int16_qparams.scale == [1.0 / 32768.0]
    assert int16_qparams.zero_point == [0]


def test_select_keeps_bool_condition_and_aligns_value_qparams() -> None:
    model_ir = ModelIR(name="select")
    model_ir.inputs = ["cond", "x", "z"]
    model_ir.outputs = ["y"]
    model_ir.tensors["cond"] = TensorIR(name="cond", dtype="BOOL", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.operators = [OperatorIR(op_type="SELECT_V2", inputs=["cond", "x", "z"], outputs=["y"])]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges={
            "x": TensorCalibrationRange(-2.0, 2.0, 1),
            "z": TensorCalibrationRange(-1.0, 1.0, 1),
            "y": TensorCalibrationRange(-0.5, 0.5, 1),
        },
    )

    assert quantized.tensors["cond"].dtype == "BOOL"
    assert quantized.tensors["cond"].quantization is None
    assert quantized.tensors["x"].quantization is not None
    assert quantized.tensors["z"].quantization is not None
    assert quantized.tensors["y"].quantization is not None
    assert quantized.tensors["x"].quantization.scale == quantized.tensors["y"].quantization.scale
    assert quantized.tensors["z"].quantization.scale == quantized.tensors["y"].quantization.scale
    _assert_litert_allocates(quantized)


def test_reduce_axes_remain_int32() -> None:
    model_ir = ModelIR(name="reduce_sum")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["axes"] = TensorIR(
        name="axes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.operators = [
        OperatorIR(op_type="SUM", inputs=["x", "axes"], outputs=["y"], options={"keepDims": False})
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    assert quantized.tensors["axes"].dtype == "INT32"
    assert quantized.tensors["axes"].quantization is None
    _assert_litert_allocates(quantized)


def test_transpose_conv_quantizes_weight_and_bias() -> None:
    model_ir = ModelIR(name="transpose_conv")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["out_shape"] = TensorIR(
        name="out_shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 4, 4, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[1, 2, 2, 1],
        shape_signature=[1, 2, 2, 1],
        data=np.ones((1, 2, 2, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.zeros((1,), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3, 3, 1], shape_signature=[1, 3, 3, 1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4, 4, 1], shape_signature=[1, 4, 4, 1])
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE_CONV",
            inputs=["out_shape", "w", "x", "b"],
            outputs=["y"],
            options={
                "padding": "VALID",
                "strideH": 1,
                "strideW": 1,
                "fusedActivationFunction": "NONE",
            },
        )
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    assert quantized.tensors["out_shape"].dtype == "INT32"
    assert quantized.tensors["w"].dtype == "INT8"
    assert quantized.tensors["w"].quantization is not None
    assert quantized.tensors["w"].quantization.quantized_dimension == 0
    assert quantized.tensors["b"].dtype == "INT32"
    assert quantized.tensors["b"].quantization is not None


def test_transpose_conv_allocates_as_strict_full_integer() -> None:
    model_ir = ModelIR(name="transpose_conv_allocate")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["out_shape"] = TensorIR(
        name="out_shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 4, 4, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[1, 2, 2, 1],
        shape_signature=[1, 2, 2, 1],
        data=np.ones((1, 2, 2, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3, 3, 1], shape_signature=[1, 3, 3, 1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4, 4, 1], shape_signature=[1, 4, 4, 1])
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE_CONV",
            inputs=["out_shape", "w", "x"],
            outputs=["y"],
            options={
                "padding": "VALID",
                "strideH": 1,
                "strideW": 1,
                "fusedActivationFunction": "NONE",
            },
        )
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    _assert_litert_allocates(quantized)


def test_identity_is_elided_before_strict_full_integer_quantization() -> None:
    model_ir = ModelIR(name="identity_elide")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["relu_out"] = TensorIR(name="relu_out", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.operators = [
        OperatorIR(op_type="RELU", inputs=["x"], outputs=["relu_out"]),
        OperatorIR(op_type="IDENTITY", inputs=["relu_out"], outputs=["y"]),
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges={
            "x": TensorCalibrationRange(-1.0, 1.0, 1),
            "relu_out": TensorCalibrationRange(0.0, 1.0, 1),
            "y": TensorCalibrationRange(0.0, 1.0, 1),
        },
    )

    assert [str(op.op_type) for op in quantized.operators] == ["RELU"]
    assert quantized.outputs == ["y"]
    assert quantized.operators[0].outputs == ["y"]
    _assert_litert_allocates(quantized)


def _make_single_op_model(op_type: str) -> ModelIR:
    model_ir = ModelIR(name=str(op_type).lower())
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    if op_type in {"MAXIMUM", "MINIMUM", "DIV"}:
        model_ir.tensors["c"] = TensorIR(
            name="c",
            dtype="FLOAT32",
            shape=[1, 4],
            shape_signature=[1, 4],
            data=np.ones((1, 4), dtype=np.float32),
            is_variable=False,
        )
        model_ir.operators = [OperatorIR(op_type=op_type, inputs=["x", "c"], outputs=["y"])]
    elif op_type == "PRELU":
        model_ir.tensors["alpha"] = TensorIR(
            name="alpha",
            dtype="FLOAT32",
            shape=[1, 4],
            shape_signature=[1, 4],
            data=np.full((1, 4), 0.25, dtype=np.float32),
            is_variable=False,
        )
        model_ir.operators = [OperatorIR(op_type=op_type, inputs=["x", "alpha"], outputs=["y"])]
    elif op_type == "LEAKY_RELU":
        model_ir.operators = [OperatorIR(op_type=op_type, inputs=["x"], outputs=["y"], options={"alpha": 0.1})]
    else:
        model_ir.operators = [OperatorIR(op_type=op_type, inputs=["x"], outputs=["y"])]
    return model_ir


def _make_shape_op_model(op_type: str) -> ModelIR:
    model_ir = ModelIR(name=str(op_type).lower())
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    if op_type == "SLICE":
        model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
        model_ir.tensors["begin"] = TensorIR(name="begin", dtype="INT32", shape=[2], shape_signature=[2], data=np.asarray([0, 1], dtype=np.int32), is_variable=False)
        model_ir.tensors["size"] = TensorIR(name="size", dtype="INT32", shape=[2], shape_signature=[2], data=np.asarray([1, 2], dtype=np.int32), is_variable=False)
        model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 2], shape_signature=[1, 2])
        model_ir.operators = [OperatorIR(op_type="SLICE", inputs=["x", "begin", "size"], outputs=["y"])]
    elif op_type == "STRIDED_SLICE":
        model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
        model_ir.tensors["begin"] = TensorIR(name="begin", dtype="INT32", shape=[2], shape_signature=[2], data=np.asarray([0, 0], dtype=np.int32), is_variable=False)
        model_ir.tensors["end"] = TensorIR(name="end", dtype="INT32", shape=[2], shape_signature=[2], data=np.asarray([1, 4], dtype=np.int32), is_variable=False)
        model_ir.tensors["stride"] = TensorIR(name="stride", dtype="INT32", shape=[2], shape_signature=[2], data=np.asarray([1, 1], dtype=np.int32), is_variable=False)
        model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
        model_ir.operators = [
            OperatorIR(
                op_type="STRIDED_SLICE",
                inputs=["x", "begin", "end", "stride"],
                outputs=["y"],
                options={"beginMask": 0, "endMask": 0, "ellipsisMask": 0, "newAxisMask": 0, "shrinkAxisMask": 0},
            )
        ]
    elif op_type == "SPLIT":
        model_ir.outputs = ["y0", "y1"]
        model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
        model_ir.tensors["axis"] = TensorIR(name="axis", dtype="INT32", shape=[1], shape_signature=[1], data=np.asarray([1], dtype=np.int32), is_variable=False)
        model_ir.tensors["y0"] = TensorIR(name="y0", dtype="FLOAT32", shape=[1, 2], shape_signature=[1, 2])
        model_ir.tensors["y1"] = TensorIR(name="y1", dtype="FLOAT32", shape=[1, 2], shape_signature=[1, 2])
        model_ir.operators = [OperatorIR(op_type="SPLIT", inputs=["axis", "x"], outputs=["y0", "y1"], options={"numSplits": 2})]
    elif op_type == "GATHER":
        model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
        model_ir.tensors["idx"] = TensorIR(name="idx", dtype="INT32", shape=[2], shape_signature=[2], data=np.asarray([0, 2], dtype=np.int32), is_variable=False)
        model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 2], shape_signature=[1, 2])
        model_ir.operators = [OperatorIR(op_type="GATHER", inputs=["x", "idx"], outputs=["y"], options={"axis": 1, "batchDims": 0})]
    elif op_type == "TILE":
        model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
        model_ir.tensors["reps"] = TensorIR(name="reps", dtype="INT32", shape=[2], shape_signature=[2], data=np.asarray([1, 2], dtype=np.int32), is_variable=False)
        model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 8], shape_signature=[1, 8])
        model_ir.operators = [OperatorIR(op_type="TILE", inputs=["x", "reps"], outputs=["y"])]
    elif op_type == "PAD":
        model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
        model_ir.tensors["pads"] = TensorIR(name="pads", dtype="INT32", shape=[2, 2], shape_signature=[2, 2], data=np.asarray([[0, 0], [1, 1]], dtype=np.int32), is_variable=False)
        model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 6], shape_signature=[1, 6])
        model_ir.operators = [OperatorIR(op_type="PAD", inputs=["x", "pads"], outputs=["y"])]
    elif op_type == "SPACE_TO_DEPTH":
        model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4, 4, 1], shape_signature=[1, 4, 4, 1])
        model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 2, 2, 4], shape_signature=[1, 2, 2, 4])
        model_ir.operators = [OperatorIR(op_type="SPACE_TO_DEPTH", inputs=["x"], outputs=["y"], options={"blockSize": 2})]
    elif op_type == "DEPTH_TO_SPACE":
        model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 2, 2, 4], shape_signature=[1, 2, 2, 4])
        model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4, 4, 1], shape_signature=[1, 4, 4, 1])
        model_ir.operators = [OperatorIR(op_type="DEPTH_TO_SPACE", inputs=["x"], outputs=["y"], options={"blockSize": 2})]
    else:
        raise AssertionError(f"Unexpected shape op: {op_type}")
    return model_ir


@pytest.mark.parametrize(
    "op_type",
    [
        "ABS",
        "NEG",
        "RELU_N1_TO_1",
        "TANH",
        "HARD_SWISH",
        "LEAKY_RELU",
        "MAXIMUM",
        "MINIMUM",
        "DIV",
        "EXP",
        "PRELU",
    ],
)
def test_additional_activation_ops_allocate_as_strict_full_integer(op_type: str) -> None:
    model_ir = _make_single_op_model(op_type)
    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    _assert_litert_allocates(quantized)


@pytest.mark.parametrize(
    "op_type",
    [
        "SLICE",
        "STRIDED_SLICE",
        "SPLIT",
        "GATHER",
        "TILE",
        "PAD",
        "SPACE_TO_DEPTH",
        "DEPTH_TO_SPACE",
    ],
)
def test_additional_forwarding_ops_allocate_as_strict_full_integer(op_type: str) -> None:
    model_ir = _make_shape_op_model(op_type)
    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    _assert_litert_allocates(quantized)


@pytest.mark.parametrize("op_type", ["SUM", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD"])
def test_additional_reduce_ops_allocate_as_strict_full_integer(op_type: str) -> None:
    model_ir = ModelIR(name=str(op_type).lower())
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["axes"] = TensorIR(
        name="axes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.operators = [
        OperatorIR(op_type=op_type, inputs=["x", "axes"], outputs=["y"], options={"keepDims": False})
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    _assert_litert_allocates(quantized)


@pytest.mark.parametrize("op_type", ["RESIZE_NEAREST_NEIGHBOR", "RESIZE_BILINEAR"])
def test_additional_resize_ops_allocate_as_strict_full_integer(op_type: str) -> None:
    model_ir = ModelIR(name=str(op_type).lower())
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 2, 2, 1], shape_signature=[1, 2, 2, 1])
    model_ir.tensors["size"] = TensorIR(
        name="size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([4, 4], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4, 4, 1], shape_signature=[1, 4, 4, 1])
    model_ir.operators = [
        OperatorIR(
            op_type=op_type,
            inputs=["x", "size"],
            outputs=["y"],
            options={"alignCorners": False, "halfPixelCenters": False},
        )
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    _assert_litert_allocates(quantized)


def test_gather_nd_keeps_indices_int32_and_aligns_data_qparams() -> None:
    model_ir = ModelIR(name="gather_nd_quantized")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[2, 3], shape_signature=[2, 3])
    model_ir.tensors["idx"] = TensorIR(
        name="idx",
        dtype="INT32",
        shape=[2, 1],
        shape_signature=[2, 1],
        data=np.asarray([[0], [1]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2, 3], shape_signature=[2, 3])
    model_ir.operators = [OperatorIR(op_type="GATHER_ND", inputs=["x", "idx"], outputs=["y"])]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    assert quantized.tensors["idx"].dtype == "INT32"
    assert quantized.tensors["idx"].quantization is None
    assert quantized.tensors["x"].dtype == "INT8"
    assert quantized.tensors["y"].dtype == "INT8"
    assert quantized.tensors["x"].quantization is not None
    assert quantized.tensors["y"].quantization is not None
    assert quantized.tensors["x"].quantization.scale == quantized.tensors["y"].quantization.scale
    assert quantized.tensors["x"].quantization.zero_point == quantized.tensors["y"].quantization.zero_point
    _assert_litert_allocates(quantized)


def test_scatter_nd_keeps_indices_shape_int32_and_aligns_update_qparams() -> None:
    model_ir = ModelIR(name="scatter_nd_quantized")
    model_ir.inputs = ["updates"]
    model_ir.outputs = ["y"]
    model_ir.tensors["idx"] = TensorIR(
        name="idx",
        dtype="INT32",
        shape=[2, 1],
        shape_signature=[2, 1],
        data=np.asarray([[0], [2]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["updates"] = TensorIR(
        name="updates",
        dtype="FLOAT32",
        shape=[2],
        shape_signature=[2],
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([4], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[4], shape_signature=[4])
    model_ir.operators = [OperatorIR(op_type="SCATTER_ND", inputs=["idx", "updates", "shape"], outputs=["y"])]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges=_calibration_ranges_for_float_activations(model_ir),
    )

    assert quantized.tensors["idx"].dtype == "INT32"
    assert quantized.tensors["shape"].dtype == "INT32"
    assert quantized.tensors["idx"].quantization is None
    assert quantized.tensors["shape"].quantization is None
    assert quantized.tensors["updates"].dtype == "INT8"
    assert quantized.tensors["y"].dtype == "INT8"
    assert quantized.tensors["updates"].quantization is not None
    assert quantized.tensors["y"].quantization is not None
    assert quantized.tensors["updates"].quantization.scale == quantized.tensors["y"].quantization.scale
    assert quantized.tensors["updates"].quantization.zero_point == quantized.tensors["y"].quantization.zero_point
    _assert_litert_allocates(quantized)


def test_scatter_nd_reports_int16_activation_skip_reason() -> None:
    model_ir = ModelIR(name="scatter_nd_int16_skip")
    model_ir.tensors["idx"] = TensorIR(
        name="idx",
        dtype="INT32",
        shape=[1, 1],
        data=np.asarray([[0]], dtype=np.int32),
    )
    model_ir.tensors["updates"] = TensorIR(name="updates", dtype="FLOAT32", shape=[1])
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1])
    model_ir.operators = [OperatorIR(op_type="SCATTER_ND", inputs=["idx", "updates", "shape"], outputs=["y"])]

    reasons = strict_int16_activation_skip_reasons(model_ir)

    assert len(reasons) == 1
    assert "SCATTER_ND" in reasons[0]
    assert "INT16 updates" in reasons[0]


def test_topk_v2_quantizes_values_and_keeps_indices_int32() -> None:
    model_ir = ModelIR(name="topk_v2_quantized")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["values", "indices"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["k"] = TensorIR(
        name="k",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(2, dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["values"] = TensorIR(
        name="values",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[1, 2],
    )
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1, 2],
        shape_signature=[1, 2],
    )
    model_ir.operators = [OperatorIR(op_type="TOPK_V2", inputs=["x", "k"], outputs=["values", "indices"])]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges={
            "x": TensorCalibrationRange(-1.0, 1.0, 1),
            "values": TensorCalibrationRange(-1.0, 1.0, 1),
        },
    )

    assert quantized.tensors["x"].dtype == "INT8"
    assert quantized.tensors["values"].dtype == "INT8"
    assert quantized.tensors["k"].dtype == "INT32"
    assert quantized.tensors["indices"].dtype == "INT32"
    assert quantized.tensors["indices"].quantization is None

    with tempfile.TemporaryDirectory() as tmpdir:
        schema = load_schema_module(output_folder_path=tmpdir)
        tflite_path = os.path.join(tmpdir, "topk_v2_quantized.tflite")
        write_model_file(schema_tflite=schema, model_ir=quantized, output_tflite_path=tflite_path, timing={})
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(
            int(input_detail["index"]),
            _quantize_input_array(np.asarray([[0.1, 0.8, -0.2, 0.5]], dtype=np.float32), input_detail),
        )
        interpreter.invoke()
        indices_detail = next(
            detail
            for detail in interpreter.get_output_details()
            if np.dtype(detail["dtype"]) == np.dtype(np.int32)
        )
        indices = interpreter.get_tensor(int(indices_detail["index"]))

    np.testing.assert_array_equal(indices, np.asarray([[1, 3]], dtype=np.int32))


def test_arg_max_quantizes_data_and_keeps_axis_output_integer() -> None:
    model_ir = ModelIR(name="arg_max_quantized")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["axis"] = TensorIR(
        name="axis",
        dtype="INT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1, dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="INT64", shape=[1], shape_signature=[1])
    model_ir.operators = [
        OperatorIR(op_type="ARG_MAX", inputs=["x", "axis"], outputs=["y"], options={"outputType": "INT64"})
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges={"x": TensorCalibrationRange(-1.0, 1.0, 1)},
    )

    assert quantized.tensors["x"].dtype == "INT8"
    assert quantized.tensors["axis"].dtype == "INT32"
    assert quantized.tensors["y"].dtype == "INT64"
    assert quantized.tensors["axis"].quantization is None
    assert quantized.tensors["y"].quantization is None

    with tempfile.TemporaryDirectory() as tmpdir:
        schema = load_schema_module(output_folder_path=tmpdir)
        tflite_path = os.path.join(tmpdir, "arg_max_quantized.tflite")
        write_model_file(schema_tflite=schema, model_ir=quantized, output_tflite_path=tflite_path, timing={})
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(
            int(input_detail["index"]),
            _quantize_input_array(np.asarray([[0.1, 0.9, -0.2, 0.5]], dtype=np.float32), input_detail),
        )
        interpreter.invoke()
        output_detail = interpreter.get_output_details()[0]
        output = interpreter.get_tensor(int(output_detail["index"]))

    np.testing.assert_array_equal(output, np.asarray([1], dtype=np.int64))


def test_logical_where_and_shape_keep_non_activation_outputs() -> None:
    model_ir = ModelIR(name="logical_where_shape_quantized")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["where_out", "shape_out"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[4], shape_signature=[4])
    model_ir.tensors["threshold"] = TensorIR(
        name="threshold",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.5], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["less"] = TensorIR(name="less", dtype="BOOL", shape=[4], shape_signature=[4])
    model_ir.tensors["not_out"] = TensorIR(name="not_out", dtype="BOOL", shape=[4], shape_signature=[4])
    model_ir.tensors["where_out"] = TensorIR(name="where_out", dtype="INT64", shape=[-1, 1], shape_signature=[-1, 1])
    model_ir.tensors["shape_out"] = TensorIR(name="shape_out", dtype="INT32", shape=[1], shape_signature=[1])
    model_ir.operators = [
        OperatorIR(op_type="LESS", inputs=["x", "threshold"], outputs=["less"]),
        OperatorIR(op_type="LOGICAL_NOT", inputs=["less"], outputs=["not_out"]),
        OperatorIR(op_type="WHERE", inputs=["not_out"], outputs=["where_out"]),
        OperatorIR(op_type="SHAPE", inputs=["x"], outputs=["shape_out"], options={"outType": "INT32"}),
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges={"x": TensorCalibrationRange(-1.0, 1.0, 1)},
    )

    assert quantized.tensors["x"].dtype == "INT8"
    assert quantized.tensors["threshold"].dtype == "INT8"
    assert quantized.tensors["less"].dtype == "BOOL"
    assert quantized.tensors["not_out"].dtype == "BOOL"
    assert quantized.tensors["where_out"].dtype == "INT64"
    assert quantized.tensors["shape_out"].dtype == "INT32"
    assert quantized.tensors["where_out"].quantization is None
    assert quantized.tensors["shape_out"].quantization is None
    _assert_litert_allocates(quantized)


def test_cast_integer_output_remains_unquantized() -> None:
    model_ir = ModelIR(name="cast_integer_quantized")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="INT64", shape=[3], shape_signature=[3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="INT32", shape=[3], shape_signature=[3])
    model_ir.operators = [
        OperatorIR(
            op_type="CAST",
            inputs=["x"],
            outputs=["y"],
            options={"inDataType": "INT64", "outDataType": "INT32"},
        )
    ]

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        calibration_ranges={"unused": TensorCalibrationRange(0.0, 1.0, 1)},
    )

    assert quantized.tensors["x"].dtype == "INT64"
    assert quantized.tensors["y"].dtype == "INT32"
    assert quantized.tensors["y"].quantization is None
    _assert_litert_allocates(quantized)
