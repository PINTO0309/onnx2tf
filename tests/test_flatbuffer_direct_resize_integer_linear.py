from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnx2tf
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.schema import schema_generated as schema


def _make_integer_linear_resize() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 1, 4, 4])
    roi = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="roi")
    scales = numpy_helper.from_array(
        np.asarray([], dtype=np.float32),
        name="scales",
    )
    sizes = numpy_helper.from_array(
        np.asarray([1, 1, 4, 4], dtype=np.int64),
        name="sizes",
    )
    resize = helper.make_node(
        "Resize",
        ["x", "roi", "scales", "sizes"],
        ["y"],
        name="IntegerLinearResize",
        mode="linear",
        coordinate_transformation_mode="half_pixel",
    )
    graph = helper.make_graph(
        [resize],
        "integer_linear_resize",
        [x],
        [y],
        initializer=[roi, scales, sizes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10
    return model


def test_integer_linear_resize_truncates_like_onnxruntime(
    tmp_path: Path,
) -> None:
    model = _make_integer_linear_resize()
    model_path = tmp_path / "integer_resize.onnx"
    onnx.save(model, model_path)

    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(tmp_path),
        disable_strict_mode=True,
        verbosity="error",
        tflite_backend="flatbuffer_direct",
    )
    tflite_path = tmp_path / "integer_resize_float32.tflite"

    tflite_model = schema.Model.GetRootAsModel(tflite_path.read_bytes(), 0)
    builtin_codes = {
        int(tflite_model.OperatorCodes(index).BuiltinCode())
        for index in range(tflite_model.OperatorCodesLength())
    }
    assert schema.BuiltinOperator.RESIZE_BILINEAR in builtin_codes
    assert schema.BuiltinOperator.CAST in builtin_codes
    assert schema.BuiltinOperator.CUSTOM not in builtin_codes

    input_value = np.asarray(
        [[[[-19, -19], [-18, -17]]]],
        dtype=np.int8,
    )
    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"x": input_value})[0]

    interpreter = Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    tflite_input = input_value
    if list(input_detail["shape"]) == [1, 2, 2, 1]:
        tflite_input = np.transpose(input_value, [0, 2, 3, 1])
    interpreter.set_tensor(input_detail["index"], tflite_input)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    if list(output.shape) == [1, 4, 4, 1]:
        output = np.transpose(output, [0, 3, 1, 2])

    # The fractional negative values distinguish truncation toward zero from
    # LiteRT's native quantized bilinear rounding-to-nearest behavior.
    assert int(expected[0, 0, 1, 0]) == -18
    np.testing.assert_array_equal(output, expected)
