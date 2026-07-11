from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.model_writer import serialize_model
from onnx2tf.tflite_builder.preprocess import (
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module


Interpreter = pytest.importorskip("ai_edge_litert.interpreter").Interpreter


def _make_dynamic_fractional_resize_model():
    x = helper.make_tensor_value_info(
        "x",
        TensorProto.FLOAT,
        [1, 1, "height", "width"],
    )
    y = helper.make_tensor_value_info(
        "y",
        TensorProto.FLOAT,
        [1, 1, "output_height", "output_width"],
    )
    scales = numpy_helper.from_array(
        np.asarray([1.0, 1.0, 0.5, 0.25], dtype=np.float32),
        name="scales",
    )
    node = helper.make_node(
        "Resize",
        ["x", "", "scales"],
        ["y"],
        name="DynamicFractionalResize",
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph([node], "dynamic_resize", [x], [y], initializer=[scales])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


def test_dynamic_fractional_resize_builds_runtime_floor_size_and_runs(
    tmp_path,
) -> None:
    model = _make_dynamic_fractional_resize_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="dynamic_fractional_resize_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "SHAPE" in op_types
    assert "SLICE" in op_types
    assert op_types.count("CAST") >= 2
    assert "MUL" in op_types
    assert "FLOOR" in op_types
    assert "RESIZE_NEAREST_NEIGHBOR" in op_types
    assert "CUSTOM" not in op_types

    model_bytes = serialize_model(
        schema_tflite=load_schema_module(str(tmp_path)),
        model_ir=model_ir,
    )
    tflite_path = tmp_path / "dynamic_fractional_resize.tflite"
    tflite_path.write_bytes(model_bytes)
    interpreter = Interpreter(model_path=str(tflite_path))
    input_detail = interpreter.get_input_details()[0]
    input_signature = [int(value) for value in input_detail["shape_signature"]]
    assert input_signature == [1, 1, -1, -1]
    interpreter.resize_tensor_input(int(input_detail["index"]), [1, 1, 6, 8])
    interpreter.allocate_tensors()

    input_value = np.arange(48, dtype=np.float32).reshape(1, 1, 6, 8)
    interpreter.set_tensor(int(input_detail["index"]), input_value)
    interpreter.invoke()
    output_detail = interpreter.get_output_details()[0]
    actual = np.asarray(interpreter.get_tensor(int(output_detail["index"])))

    expected = np.transpose(input_value[:, :, ::2, ::4], [0, 2, 3, 1])
    assert actual.shape == (1, 3, 2, 1)
    np.testing.assert_array_equal(actual, expected)
