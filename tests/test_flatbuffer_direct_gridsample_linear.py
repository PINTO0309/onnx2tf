from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper

from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.preprocess import (
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)


def _make_standard_gridsample_model(*, mode: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 2, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node(
        "GridSample",
        ["x", "grid"],
        ["y"],
        name="standard_linear_gridsample",
        mode=mode,
        padding_mode="zeros",
        align_corners=0,
    )
    model = helper.make_model(
        helper.make_graph([node], "standard_linear_gridsample_graph", [x, grid], [y]),
        opset_imports=[helper.make_operatorsetid("", 20)],
    )
    model.ir_version = 10
    return model


@pytest.mark.parametrize("mode", ["linear", "nearest"])
def test_standard_gridsample_uses_builtin_lowering(mode: str) -> None:
    model = _make_standard_gridsample_model(mode=mode)
    register_default_preprocess_rules()
    preprocessed, _ = run_preprocess_pipeline(onnx_graph=model)

    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed,
        output_file_name="standard_linear_gridsample",
        allow_custom_ops=False,
    )

    assert all(str(op.op_type) != "CUSTOM" for op in model_ir.operators)
    assert any(str(op.op_type) == "GATHER" for op in model_ir.operators)


@pytest.mark.parametrize("mode", ["linear", "nearest"])
def test_standard_gridsample_tflite_matches_onnx(tmp_path: Path, mode: str) -> None:
    model = _make_standard_gridsample_model(mode=mode)
    result = export_tflite_model_flatbuffer_direct(
        onnx_graph=model,
        output_folder_path=str(tmp_path),
        output_file_name="standard_linear_gridsample",
        flatbuffer_direct_allow_custom_ops=False,
        keep_shape_absolutely_input_names=["grid"],
    )
    x = np.asarray([[[[1.0, 2.0], [3.0, 5.0]]]], dtype=np.float32)
    grid = np.asarray(
        [[[[ -0.75, -0.75], [0.25, -0.5]], [[-0.5, 0.5], [0.75, 0.75]]]],
        dtype=np.float32,
    )
    grid[0, 0, 0] = np.nan
    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"x": x, "grid": grid})[0]

    interpreter = Interpreter(model_path=result["float32_tflite_path"])
    interpreter.allocate_tensors()
    details = {
        str(detail["name"]).split(":")[0]: detail
        for detail in interpreter.get_input_details()
    }
    x_value = np.transpose(x, (0, 2, 3, 1)) if list(details["x"]["shape"]) == [1, 2, 2, 1] else x
    interpreter.set_tensor(int(details["x"]["index"]), x_value)
    interpreter.set_tensor(int(details["grid"]["index"]), grid)
    interpreter.invoke()
    actual = interpreter.get_tensor(int(interpreter.get_output_details()[0]["index"]))

    if list(actual.shape) == [1, 2, 2, 1]:
        expected = np.transpose(expected, (0, 2, 3, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
