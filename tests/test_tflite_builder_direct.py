import os
import shutil
import tempfile

import numpy as np
import onnx
import onnx2tf
import pytest
from onnx import TensorProto, helper, numpy_helper

Interpreter = pytest.importorskip("ai_edge_litert.interpreter").Interpreter


def _save_model(tmpdir: str, name: str, model: onnx.ModelProto) -> str:
    model_path = os.path.join(tmpdir, f"{name}.onnx")
    onnx.save(model, model_path)
    return model_path


def _convert(
    model_path: str,
    output_dir: str,
    backend: str,
    output_dynamic_range_quantized_tflite: bool = False,
    output_integer_quantized_tflite: bool = False,
) -> str:
    onnx2tf.convert(
        input_onnx_file_path=model_path,
        output_folder_path=output_dir,
        disable_strict_mode=True,
        verbosity="error",
        tflite_backend=backend,
        output_dynamic_range_quantized_tflite=output_dynamic_range_quantized_tflite,
        output_integer_quantized_tflite=output_integer_quantized_tflite,
    )
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(output_dir, f"{model_name}_float32.tflite")


def _run_add_inference(tflite_path: str) -> np.ndarray:
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    by_name = {detail["name"]: detail for detail in input_details}
    if "x" in by_name and "y" in by_name:
        interpreter.set_tensor(by_name["x"]["index"], x)
        interpreter.set_tensor(by_name["y"]["index"], y)
    else:
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.set_tensor(input_details[1]["index"], y)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def _make_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvNode",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "conv_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node(
        "AveragePool",
        ["x"],
        ["y"],
        name="PoolNode",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([node], "pool_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gemm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((3, 4), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="B")
    node = helper.make_node("Gemm", ["x", "W", "B"], ["y"], name="GemmNode", transB=1)
    graph = helper.make_graph([node], "gemm_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _requires_flatbuffer_tools() -> bool:
    return shutil.which("flatc") is not None and shutil.which("curl") is not None


def test_tflite_backend_matrix_add() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add", model)

        tf_out = os.path.join(tmpdir, "tf_converter")
        tf_tflite = _convert(model_path, tf_out, "tf_converter")
        tf_pred = _run_add_inference(tf_tflite)

        if not _requires_flatbuffer_tools():
            pytest.skip("flatbuffer_direct requires flatc and curl")

        fb_out = os.path.join(tmpdir, "flatbuffer_direct")
        fb_tflite = _convert(model_path, fb_out, "flatbuffer_direct")
        fb_pred = _run_add_inference(fb_tflite)

        np.testing.assert_allclose(tf_pred, np.array([[5.0, 7.0, 9.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(fb_pred, tf_pred, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
@pytest.mark.parametrize(
    "name, model_factory",
    [
        ("conv", _make_conv_model),
        ("pool", _make_pool_model),
        ("gemm", _make_gemm_model),
    ],
)
def test_flatbuffer_direct_operator_smoke(name: str, model_factory) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = model_factory()
        model_path = _save_model(tmpdir, name, model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_quantized_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_dq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
        )
        tflite_path = os.path.join(out_dir, "gemm_dq_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        assert y.shape == (1, 3)
