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


def _make_edge_pad_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [5, 6])
    pads = numpy_helper.from_array(
        np.asarray([1, 2, 2, 1], dtype=np.int64),
        name="pads",
    )
    node = helper.make_node(
        "Pad",
        ["x", "pads"],
        ["y"],
        name="EdgePad",
        mode="edge",
    )
    graph = helper.make_graph([node], "edge_pad", [x], [y], initializer=[pads])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


def test_edge_pad_lowers_to_builtin_slice_tile_concat_and_matches_numpy(
    tmp_path,
) -> None:
    model = _make_edge_pad_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="edge_pad_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("STRIDED_SLICE") == 4
    assert op_types.count("TILE") == 2
    assert op_types.count("CONCATENATION") == 2
    assert "CUSTOM" not in op_types

    model_bytes = serialize_model(
        schema_tflite=load_schema_module(str(tmp_path)),
        model_ir=model_ir,
    )
    tflite_path = tmp_path / "edge_pad.tflite"
    tflite_path.write_bytes(model_bytes)
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_value = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    interpreter.set_tensor(int(input_detail["index"]), input_value)
    interpreter.invoke()
    actual = np.asarray(interpreter.get_tensor(int(output_detail["index"])))

    expected = np.pad(input_value, ((1, 2), (2, 1)), mode="edge")
    np.testing.assert_array_equal(actual, expected)
