from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def _make_opset10_nearest_resize_model() -> onnx.ModelProto:
    image = helper.make_tensor_value_info(
        "image",
        TensorProto.FLOAT,
        [1, 1, 4, 4],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 1, 2, 2],
    )
    scales = numpy_helper.from_array(
        np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float32),
        name="scales",
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Resize",
                    ["image", "scales"],
                    ["output"],
                    mode="nearest",
                )
            ],
            "opset10_nearest_resize",
            [image],
            [output],
            initializer=[scales],
        ),
        opset_imports=[helper.make_operatorsetid("", 10)],
    )
    model.ir_version = 10
    return model


def test_opset10_nearest_resize_uses_asymmetric_coordinates(tmp_path: Path) -> None:
    model = _make_opset10_nearest_resize_model()
    model_ir = lower_onnx_to_ir(
        onnx_graph=model,
        output_file_name="opset10_nearest_resize",
        allow_custom_ops=False,
        optimize_layout_transpose_chains=False,
    )
    resize_op = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "RESIZE_NEAREST_NEIGHBOR"
    )
    assert resize_op.options["alignCorners"] is False
    assert resize_op.options["halfPixelCenters"] is False

    result = export_tflite_model_flatbuffer_direct(
        onnx_graph=model,
        output_folder_path=str(tmp_path),
        output_file_name="opset10_nearest_resize",
        flatbuffer_direct_allow_custom_ops=False,
    )
    image = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"image": image})[0]
    np.testing.assert_array_equal(
        expected,
        np.asarray([[[[0.0, 2.0], [8.0, 10.0]]]], dtype=np.float32),
    )

    interpreter = Interpreter(model_path=result["float32_tflite_path"])
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    tflite_input = image
    if tuple(input_detail["shape"]) == (1, 4, 4, 1):
        tflite_input = np.transpose(image, (0, 2, 3, 1))
    interpreter.set_tensor(input_detail["index"], tflite_input)
    interpreter.invoke()
    actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    if actual.shape == (1, 2, 2, 1):
        actual = np.transpose(actual, (0, 3, 1, 2))
    np.testing.assert_array_equal(actual, expected)
