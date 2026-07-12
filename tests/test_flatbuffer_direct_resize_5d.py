from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.preprocess import (
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)


def _make_resize_5d_model(*, mode: str = "linear") -> onnx.ModelProto:
    input_shape = [1, 2, 2, 3, 4]
    output_shape = [1, 2, 3, 4, 5]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)
    sizes = numpy_helper.from_array(np.asarray(output_shape, dtype=np.int64), name="sizes")
    resize = helper.make_node(
        "Resize",
        ["x", "", "", "sizes"],
        ["y"],
        name="resize_5d",
        mode=mode,
        coordinate_transformation_mode="half_pixel",
        nearest_mode="round_prefer_ceil",
    )
    model = helper.make_model(
        helper.make_graph([resize], "resize_5d_graph", [x], [y], initializer=[sizes]),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10
    return model


@pytest.mark.parametrize(
    ("mode", "builtin_op"),
    [("linear", "RESIZE_BILINEAR"), ("nearest", "RESIZE_NEAREST_NEIGHBOR")],
)
def test_resize_5d_lowers_to_two_builtin_resizes(mode: str, builtin_op: str) -> None:
    model = _make_resize_5d_model(mode=mode)
    register_default_preprocess_rules()
    preprocessed, _ = run_preprocess_pipeline(onnx_graph=model)

    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed,
        output_file_name="resize_5d",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count(builtin_op) == 2
    assert op_types.count("CUSTOM") == 0


@pytest.mark.parametrize("mode", ["linear", "nearest"])
def test_resize_5d_tflite_matches_onnx(tmp_path: Path, mode: str) -> None:
    model = _make_resize_5d_model(mode=mode)
    result = export_tflite_model_flatbuffer_direct(
        onnx_graph=model,
        output_folder_path=str(tmp_path),
        output_file_name="resize_5d",
        flatbuffer_direct_allow_custom_ops=False,
    )
    sample = np.linspace(-1.0, 1.0, num=48, dtype=np.float32).reshape(1, 2, 2, 3, 4)
    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"x": sample})[0]

    interpreter = Interpreter(model_path=result["float32_tflite_path"])
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    sample_ndhwc = np.transpose(sample, (0, 2, 3, 4, 1))
    assert list(input_detail["shape"]) == list(sample_ndhwc.shape)
    interpreter.set_tensor(int(input_detail["index"]), sample_ndhwc)
    interpreter.invoke()
    actual = interpreter.get_tensor(int(interpreter.get_output_details()[0]["index"]))

    # The separable lowering restores the ONNX NCDHW boundary explicitly.
    assert list(actual.shape) == list(expected.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
