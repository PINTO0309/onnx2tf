import glob
import json
import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import onnx
import onnx2tf
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx2tf.tflite_builder.schema_loader import load_schema_module

Interpreter = pytest.importorskip("ai_edge_litert.interpreter").Interpreter
pytest.importorskip("onnxruntime")


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
    quant_type: str = "per-channel",
    input_quant_dtype: str = "int8",
    output_quant_dtype: str = "int8",
    eval_with_onnx: bool = False,
    eval_num_samples: int = 10,
    eval_rtol: float = 0.0,
    eval_atol: float = 1e-4,
    eval_fail_on_threshold: bool = False,
    eval_target_tflite: str = "float32",
    eval_compare_mode: str = "auto",
    eval_split_models: bool = False,
    eval_split_reference: str = "unsplit_tflite",
    eval_split_fail_on_threshold: bool = False,
    auto_split_tflite_by_size: bool = False,
    report_op_coverage: bool = False,
    tflite_split_max_bytes: int = 1073741824,
    tflite_split_target_bytes: int = 1060000000,
) -> str:
    onnx2tf.convert(
        input_onnx_file_path=model_path,
        output_folder_path=output_dir,
        disable_strict_mode=True,
        verbosity="error",
        tflite_backend=backend,
        output_dynamic_range_quantized_tflite=output_dynamic_range_quantized_tflite,
        output_integer_quantized_tflite=output_integer_quantized_tflite,
        quant_type=quant_type,
        input_quant_dtype=input_quant_dtype,
        output_quant_dtype=output_quant_dtype,
        eval_with_onnx=eval_with_onnx,
        eval_num_samples=eval_num_samples,
        eval_rtol=eval_rtol,
        eval_atol=eval_atol,
        eval_fail_on_threshold=eval_fail_on_threshold,
        eval_target_tflite=eval_target_tflite,
        eval_compare_mode=eval_compare_mode,
        eval_split_models=eval_split_models,
        eval_split_reference=eval_split_reference,
        eval_split_fail_on_threshold=eval_split_fail_on_threshold,
        auto_split_tflite_by_size=auto_split_tflite_by_size,
        report_op_coverage=report_op_coverage,
        tflite_split_max_bytes=tflite_split_max_bytes,
        tflite_split_target_bytes=tflite_split_target_bytes,
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


def _make_add_const_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    c = numpy_helper.from_array(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        name="c",
    )
    node = helper.make_node("Add", ["x", "c"], ["y"], name="AddConstNode")
    graph = helper.make_graph([node], "add_const_graph", [x], [y], initializer=[c])
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


def _make_add_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    n0 = helper.make_node("Add", ["x", "y"], ["a0"], name="AddChain0")
    n1 = helper.make_node("Add", ["a0", "y"], ["a1"], name="AddChain1")
    n2 = helper.make_node("Add", ["a1", "y"], ["z"], name="AddChain2")
    graph = helper.make_graph([n0, n1, n2], "add_chain_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_relu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", ["x"], ["y"], name="ReluNode")
    graph = helper.make_graph([node], "relu_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _requires_flatbuffer_tools() -> bool:
    return shutil.which("flatc") is not None and shutil.which("curl") is not None


@contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key, None)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _collect_int8_quant_scale_lengths(tflite_path: str) -> list[int]:
    output_dir = os.path.dirname(tflite_path)
    schema = load_schema_module(output_dir)
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    model = schema["ModelT"].InitFromObj(schema["Model"].GetRootAs(model_bytes, 0))
    subgraph = model.subgraphs[0]
    int8_type = getattr(schema["TensorType"], "INT8")
    lengths: list[int] = []
    for tensor in subgraph.tensors:
        if tensor.type != int8_type:
            continue
        if tensor.quantization is None or tensor.quantization.scale is None:
            continue
        lengths.append(len(list(tensor.quantization.scale)))
    return lengths


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
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.ones(input_details[0]["shape"], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])


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


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_quantized_add_const_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_const_model()
        model_path = _save_model(tmpdir, "add_const_dq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
        )
        tflite_path = os.path.join(out_dir, "add_const_dq_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_allclose(
            y,
            np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
            rtol=0.0,
            atol=5e-2,
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
@pytest.mark.parametrize(
    "quant_type, expected_multi_scale",
    [
        ("per-channel", True),
        ("per-tensor", False),
    ],
)
def test_flatbuffer_direct_dynamic_range_quantized_fc_quant_type(
    quant_type: str,
    expected_multi_scale: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, f"gemm_{quant_type}", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
            quant_type=quant_type,
        )
        tflite_path = os.path.join(
            out_dir,
            f"gemm_{quant_type}_dynamic_range_quant.tflite",
        )
        assert os.path.isfile(tflite_path)

        scale_lengths = _collect_int8_quant_scale_lengths(tflite_path)
        assert len(scale_lengths) > 0
        if expected_multi_scale:
            assert any(length > 1 for length in scale_lengths)
        else:
            assert all(length == 1 for length in scale_lengths)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integer_quantized_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_iq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
            input_quant_dtype="int8",
            output_quant_dtype="int8",
        )

        integer_tflite = os.path.join(out_dir, "gemm_iq_integer_quant.tflite")
        full_integer_tflite = os.path.join(out_dir, "gemm_iq_full_integer_quant.tflite")
        integer_i16_tflite = os.path.join(out_dir, "gemm_iq_integer_quant_with_int16_act.tflite")
        full_integer_i16_tflite = os.path.join(out_dir, "gemm_iq_full_integer_quant_with_int16_act.tflite")
        assert os.path.isfile(integer_tflite)
        assert os.path.isfile(full_integer_tflite)
        assert os.path.isfile(integer_i16_tflite)
        assert os.path.isfile(full_integer_i16_tflite)

        # integer_quant: float input/output path
        interpreter = Interpreter(model_path=integer_tflite)
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter.set_tensor(in_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out_details[0]["index"])
        assert y.shape == (1, 3)

        # full_integer_quant: quantized io path
        interpreter2 = Interpreter(model_path=full_integer_tflite)
        interpreter2.allocate_tensors()
        in2 = interpreter2.get_input_details()
        out2 = interpreter2.get_output_details()
        assert in2[0]["dtype"] == np.int8
        assert out2[0]["dtype"] == np.int8
        xq = np.zeros(in2[0]["shape"], dtype=np.int8)
        interpreter2.set_tensor(in2[0]["index"], xq)
        interpreter2.invoke()
        yq = interpreter2.get_tensor(out2[0]["index"])
        assert yq.dtype == np.int8

        # integer_quant_with_int16_act: float input/output path
        interpreter3 = Interpreter(model_path=integer_i16_tflite)
        interpreter3.allocate_tensors()
        in3 = interpreter3.get_input_details()
        out3 = interpreter3.get_output_details()
        x3 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter3.set_tensor(in3[0]["index"], x3)
        interpreter3.invoke()
        y3 = interpreter3.get_tensor(out3[0]["index"])
        assert y3.shape == (1, 3)

        # full_integer_quant_with_int16_act: int16 input/output path
        interpreter4 = Interpreter(model_path=full_integer_i16_tflite)
        interpreter4.allocate_tensors()
        in4 = interpreter4.get_input_details()
        out4 = interpreter4.get_output_details()
        assert in4[0]["dtype"] == np.int16
        assert out4[0]["dtype"] == np.int16
        x4 = np.zeros(in4[0]["shape"], dtype=np.int16)
        interpreter4.set_tensor(in4[0]["index"], x4)
        interpreter4.invoke()
        y4 = interpreter4.get_tensor(out4[0]["index"])
        assert y4.dtype == np.int16


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_percentile_calibration_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_pct", model)
        out_dir = os.path.join(tmpdir, "out")
        with _temporary_env(
            {
                "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_METHOD": "percentile",
                "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_PERCENTILE": "99.0",
            }
        ):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                output_dynamic_range_quantized_tflite=True,
            )
        tflite_path = os.path.join(out_dir, "gemm_pct_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_threshold_control() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_th", model)
        out_dir = os.path.join(tmpdir, "out")
        with _temporary_env(
            {
                "ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_ABS_MAX": "1000.0",
            }
        ):
            with pytest.raises(NotImplementedError):
                _convert(
                    model_path,
                    out_dir,
                    "flatbuffer_direct",
                    output_dynamic_range_quantized_tflite=True,
                )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add_eval", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            eval_with_onnx=True,
            eval_num_samples=3,
        )

        report_path = os.path.join(out_dir, "add_eval_accuracy_report.json")
        assert os.path.isfile(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["schema_version"] == 1
        assert report["num_samples"] == 3
        assert report["seed"] == 0
        assert report["inputs_source"] == "seeded_random"
        assert report["compare_mode"] == "raw"
        assert report["evaluation_pass"] is True
        assert report["allclose_summary"]["pass"] is True
        assert "overall_metrics" in report
        assert "per_output_metrics" in report
        assert "z" in report["per_output_metrics"]
        assert report["overall_metrics"]["max_abs"] <= 1e-6


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_quant_dequant_mode() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_eval_q", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_integer_quantized_tflite=True,
            eval_with_onnx=True,
            eval_num_samples=3,
            eval_target_tflite="full_integer_quant",
            eval_compare_mode="dequant",
        )

        report_path = os.path.join(out_dir, "gemm_eval_q_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["compare_mode"] == "dequant"
        assert report["has_quantized_outputs"] is True
        assert "metric_threshold_judgement" in report


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_fail_on_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_eval_fail", model)
        out_dir = os.path.join(tmpdir, "out")
        with pytest.raises(RuntimeError):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                output_integer_quantized_tflite=True,
                eval_with_onnx=True,
                eval_num_samples=2,
                eval_target_tflite="full_integer_quant",
                eval_compare_mode="raw",
                eval_fail_on_threshold=True,
            )
        report_path = os.path.join(out_dir, "gemm_eval_fail_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["evaluation_pass"] is False


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_plan_report_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_split_plan", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=9_000_000,
        )
        report_path = os.path.join(out_dir, "gemm_split_plan_split_plan.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["schema_version"] == 1
        assert report["plan_valid"] is True
        assert report["total_estimated_bytes"] > 0
        assert len(report["partitions"]) >= 1


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_manifest_and_partition_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_split", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
        )

        manifest_path = os.path.join(out_dir, "add_chain_split_split_manifest.json")
        assert os.path.isfile(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["schema_version"] == 1
        assert manifest["base_model"] == "add_chain_split.tflite"
        assert manifest["target_max_bytes"] == 1
        assert len(manifest["partitions"]) >= 1
        assert "edges" in manifest

        part_files = sorted(
            glob.glob(os.path.join(out_dir, "add_chain_split_[0-9][0-9][0-9][0-9].tflite"))
        )
        assert len(part_files) >= 1
        for part_file in part_files:
            interpreter = Interpreter(model_path=part_file)
            interpreter.allocate_tensors()


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_accuracy_report_with_unsplit_reference() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_eval_split", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
            eval_split_models=True,
            eval_split_reference="unsplit_tflite",
            eval_num_samples=3,
        )
        report_path = os.path.join(out_dir, "add_chain_eval_split_split_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["reference_mode"] == "unsplit_tflite"
        assert report["evaluation_pass"] is True
        assert report["allclose_summary"]["pass"] is True


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_split_accuracy_report_fail_on_threshold() -> None:
    from onnx2tf.tflite_builder.split_accuracy_evaluator import evaluate_split_manifest_outputs

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_eval_split_fail", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
        )
        split_manifest_path = os.path.join(out_dir, "add_chain_eval_split_fail_split_manifest.json")
        reference_tflite_path = os.path.join(out_dir, "add_chain_eval_split_fail_float32.tflite")
        onnx_graph = onnx.load(model_path)
        report_path = os.path.join(out_dir, "add_chain_eval_split_fail_split_accuracy_report.json")
        with pytest.raises(RuntimeError):
            evaluate_split_manifest_outputs(
                onnx_graph=onnx_graph,
                split_manifest_path=split_manifest_path,
                reference_mode="unsplit_tflite",
                reference_tflite_path=reference_tflite_path,
                output_report_path=report_path,
                num_samples=2,
                fail_on_threshold=True,
                metric_thresholds={
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "rmse": 0.0,
                    "cosine_similarity": 1.1,
                },
            )
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["evaluation_pass"] is False


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_op_coverage_report_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add_cov", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        report_path = os.path.join(out_dir, "add_cov_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["schema_version"] == 1
        assert report["conversion_error"] is None
        assert "Add" in report["graph_ops"]
        assert report["graph_summary"]["unsupported_nodes"] == 0
        assert report["graph_summary"]["coverage_ratio"] == 1.0


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_op_coverage_report_on_unsupported_op() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_relu_model()
        model_path = _save_model(tmpdir, "relu_cov", model)
        out_dir = os.path.join(tmpdir, "out")
        with pytest.raises(NotImplementedError):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                report_op_coverage=True,
            )
        report_path = os.path.join(out_dir, "relu_cov_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is not None
        assert report["graph_summary"]["unsupported_nodes"] == 1
        assert report["unsupported_reason_counts"]["unsupported_onnx_op"] == 1
        assert report["unsupported_nodes"][0]["onnx_op"] == "Relu"
