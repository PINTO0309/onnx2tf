from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper

from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def _make_dynamic_rank5_indices_model() -> onnx.ModelProto:
    data = helper.make_tensor_value_info(
        "data",
        TensorProto.FLOAT,
        [2, 3, 4, 5],
    )
    indices = helper.make_tensor_value_info(
        "indices",
        TensorProto.INT64,
        ["count", 1, 1, 1, 4],
    )
    updates = helper.make_tensor_value_info(
        "updates",
        TensorProto.FLOAT,
        ["count", 1, 1, 1],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [2, 3, 4, 5],
    )
    node = helper.make_node(
        "ScatterND",
        ["data", "indices", "updates"],
        ["output"],
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "dynamic_rank5_scatternd_indices",
            [data, indices, updates],
            [output],
        ),
        opset_imports=[helper.make_operatorsetid("", 16)],
    )
    model.ir_version = 10
    return model


def test_dynamic_rank5_scatternd_indices_match_onnx(tmp_path: Path) -> None:
    model = _make_dynamic_rank5_indices_model()
    model_ir = lower_onnx_to_ir(
        onnx_graph=model,
        output_file_name="dynamic_rank5_scatternd_indices",
        allow_custom_ops=False,
        optimize_layout_transpose_chains=False,
    )
    less_op = next(op for op in model_ir.operators if str(op.op_type) == "LESS")
    less_input = model_ir.tensors[str(less_op.inputs[0])]
    assert list(less_input.shape_signature or []) == [-1, 4]
    scatter_indices = {
        str(op.inputs[0])
        for op in model_ir.operators
        if str(op.op_type) == "SCATTER_ND"
    }
    assert len(scatter_indices) == 1
    normalized = model_ir.tensors[scatter_indices.pop()]
    assert list(normalized.shape_signature or []) == [-1, 1, 1, 1, 4]

    result = export_tflite_model_flatbuffer_direct(
        onnx_graph=model,
        output_folder_path=str(tmp_path),
        output_file_name="dynamic_rank5_scatternd_indices",
        flatbuffer_direct_allow_custom_ops=False,
        keep_shape_absolutely_input_names=["data", "indices", "updates"],
    )
    data = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
    indices = np.asarray(
        [
            [[[[0, 0, 0, -1]]]],
            [[[[-1, -1, -1, 0]]]],
        ],
        dtype=np.int64,
    )
    updates = np.asarray([[[[1000.0]]], [[[2000.0]]]], dtype=np.float32)
    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"data": data, "indices": indices, "updates": updates})[0]

    interpreter = Interpreter(model_path=result["float32_tflite_path"])
    details = {
        str(detail["name"]).split(":")[0]: detail
        for detail in interpreter.get_input_details()
    }
    interpreter.resize_tensor_input(details["indices"]["index"], indices.shape)
    interpreter.resize_tensor_input(details["updates"]["index"], updates.shape)
    interpreter.allocate_tensors()
    details = {
        str(detail["name"]).split(":")[0]: detail
        for detail in interpreter.get_input_details()
    }
    interpreter.set_tensor(details["data"]["index"], data)
    interpreter.set_tensor(details["indices"]["index"], indices)
    interpreter.set_tensor(details["updates"]["index"], updates)
    interpreter.invoke()
    actual = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
    np.testing.assert_array_equal(actual, expected)
