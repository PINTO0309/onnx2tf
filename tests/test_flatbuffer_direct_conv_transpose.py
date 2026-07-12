from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper, numpy_helper

import onnx2tf
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def test_dynamic_conv_transpose_supports_symmetric_pad_with_output_padding() -> None:
    x = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, ["batch", 2, 4, 4]
    )
    y = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, ["batch", 3, 8, 8]
    )
    weights = numpy_helper.from_array(
        np.arange(2 * 3 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 3)
        / 100.0,
        name="weights",
    )
    node = helper.make_node(
        "ConvTranspose",
        ["x", "weights"],
        ["y"],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        output_padding=[1, 1],
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "dynamic_conv_transpose_output_padding",
            [x],
            [y],
            initializer=[weights],
        ),
        opset_imports=[helper.make_operatorsetid("", 17)],
    )

    model_ir = lower_onnx_to_ir(
        model,
        "dynamic_conv_transpose_output_padding",
        optimize_layout_transpose_chains=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "TRANSPOSE_CONV" in op_types
    assert "CUSTOM" not in op_types
    transpose_conv = next(
        op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE_CONV"
    )
    output_shape_name = str(transpose_conv.inputs[0])
    assert model_ir.tensors[output_shape_name].data is None
    assert any(
        str(op.op_type) == "CONCATENATION"
        and output_shape_name in [str(v) for v in op.outputs]
        for op in model_ir.operators
    )
    crop = next(
        op for op in model_ir.operators if str(op.op_type) == "STRIDED_SLICE"
    )
    crop_end_name = str(crop.inputs[2])
    assert model_ir.tensors[crop_end_name].data is None
    assert any(
        str(op.op_type) == "ADD"
        and crop_end_name in [str(v) for v in op.outputs]
        for op in model_ir.operators
    )


def test_dynamic_conv_transpose_crop_matches_onnx(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, ["batch", 2, 4, 4]
    )
    y = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, ["batch", 3, 8, 8]
    )
    weights = numpy_helper.from_array(
        np.arange(2 * 3 * 3 * 3, dtype=np.float32).reshape(2, 3, 3, 3)
        / 100.0,
        name="weights",
    )
    node = helper.make_node(
        "ConvTranspose",
        ["x", "weights"],
        ["y"],
        strides=[2, 2],
        pads=[1, 1, 1, 1],
        output_padding=[1, 1],
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "dynamic_conv_transpose_crop",
            [x],
            [y],
            initializer=[weights],
        ),
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 10
    model_path = tmp_path / "dynamic_conv_transpose_crop.onnx"
    onnx.save(model, model_path)

    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(tmp_path),
        disable_strict_mode=True,
        verbosity="error",
        tflite_backend="flatbuffer_direct",
    )

    input_value = np.arange(32, dtype=np.float32).reshape(1, 2, 4, 4) / 10.0
    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"x": input_value})[0]

    interpreter = Interpreter(
        model_path=str(tmp_path / "dynamic_conv_transpose_crop_float32.tflite"),
        num_threads=1,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    tflite_input = input_value
    if list(input_detail["shape"]) == [1, 4, 4, 2]:
        tflite_input = np.transpose(input_value, [0, 2, 3, 1])
    interpreter.set_tensor(input_detail["index"], tflite_input)
    interpreter.invoke()
    actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    if list(actual.shape) == [1, 8, 8, 3]:
        actual = np.transpose(actual, [0, 3, 1, 2])

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_dynamic_conv_transpose_preserves_empty_batch(tmp_path: Path) -> None:
    x = helper.make_tensor_value_info(
        "x", TensorProto.FLOAT, ["batch", 2, "height", "width"]
    )
    y = helper.make_tensor_value_info(
        "y", TensorProto.FLOAT, ["batch", 3, "out_height", "out_width"]
    )
    weights = numpy_helper.from_array(
        np.ones((2, 3, 2, 2), dtype=np.float32),
        name="weights",
    )
    bias = numpy_helper.from_array(
        np.zeros((3,), dtype=np.float32),
        name="bias",
    )
    node = helper.make_node(
        "ConvTranspose",
        ["x", "weights", "bias"],
        ["y"],
        strides=[2, 2],
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "dynamic_conv_transpose_empty_batch",
            [x],
            [y],
            initializer=[weights, bias],
        ),
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 10
    model_path = tmp_path / "dynamic_conv_transpose_empty_batch.onnx"
    onnx.save(model, model_path)

    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(tmp_path),
        disable_strict_mode=True,
        verbosity="error",
        tflite_backend="flatbuffer_direct",
    )

    interpreter = Interpreter(
        model_path=str(
            tmp_path / "dynamic_conv_transpose_empty_batch_float32.tflite"
        ),
        num_threads=1,
    )
    input_detail = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(input_detail["index"], [0, 4, 4, 2])
    interpreter.allocate_tensors()
    interpreter.set_tensor(
        input_detail["index"],
        np.empty((0, 4, 4, 2), dtype=np.float32),
    )
    interpreter.invoke()

    output_detail = interpreter.get_output_details()[0]
    assert output_detail["shape_signature"].tolist() == [-1, -1, -1, 3]
    assert interpreter.get_tensor(output_detail["index"]).shape == (0, 8, 8, 3)
