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


def _make_dynamic_gridsample_model(
    *,
    mode: str,
    padding_mode: str,
    align_corners: int,
) -> onnx.ModelProto:
    image = helper.make_tensor_value_info(
        "image",
        TensorProto.FLOAT,
        [1, 1, "height", "width"],
    )
    grid = helper.make_tensor_value_info(
        "grid",
        TensorProto.FLOAT,
        [1, 2, 3, 2],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 1, 2, 3],
    )
    node = helper.make_node(
        "GridSample",
        ["image", "grid"],
        ["output"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "dynamic_gridsample",
            [image, grid],
            [output],
        ),
        opset_imports=[helper.make_operatorsetid("", 16)],
    )
    model.ir_version = 10
    return model


@pytest.mark.parametrize(
    ("mode", "padding_mode", "align_corners"),
    [
        ("bilinear", "zeros", 1),
        ("bilinear", "border", 0),
        ("nearest", "zeros", 0),
    ],
)
def test_dynamic_gridsample_matches_onnx(
    tmp_path: Path,
    mode: str,
    padding_mode: str,
    align_corners: int,
) -> None:
    model = _make_dynamic_gridsample_model(
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    model_ir = lower_onnx_to_ir(
        onnx_graph=model,
        output_file_name="dynamic_gridsample",
        allow_custom_ops=False,
        optimize_layout_transpose_chains=False,
    )
    assert all(str(op.op_type) != "CUSTOM" for op in model_ir.operators)
    assert any(str(op.op_type) == "RANGE" for op in model_ir.operators)

    result = export_tflite_model_flatbuffer_direct(
        onnx_graph=model,
        output_folder_path=str(tmp_path),
        output_file_name=f"dynamic_gridsample_{mode}_{padding_mode}",
        flatbuffer_direct_allow_custom_ops=False,
        keep_shape_absolutely_input_names=["image", "grid"],
    )
    image = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4) / 5.0
    grid = np.asarray(
        [
            [
                [[-1.2, -1.1], [-0.25, -0.5], [1.1, -0.8]],
                [[-0.9, 0.7], [0.3, 0.2], [1.2, 1.1]],
            ]
        ],
        dtype=np.float32,
    )
    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"image": image, "grid": grid})[0]

    interpreter = Interpreter(model_path=result["float32_tflite_path"])
    input_details = {
        str(detail["name"]).split(":")[0]: detail
        for detail in interpreter.get_input_details()
    }
    interpreter.resize_tensor_input(
        int(input_details["image"]["index"]),
        list(image.shape),
    )
    interpreter.allocate_tensors()
    input_details = {
        str(detail["name"]).split(":")[0]: detail
        for detail in interpreter.get_input_details()
    }
    interpreter.set_tensor(int(input_details["image"]["index"]), image)
    interpreter.set_tensor(int(input_details["grid"]["index"]), grid)
    interpreter.invoke()
    actual = interpreter.get_tensor(
        int(interpreter.get_output_details()[0]["index"])
    )

    if actual.shape == (1, 2, 3, 1):
        expected = np.transpose(expected, (0, 2, 3, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
