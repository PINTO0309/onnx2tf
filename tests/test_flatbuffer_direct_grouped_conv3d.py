from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.preprocess import (
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)


def _make_grouped_conv3d_model() -> onnx.ModelProto:
    input_shape = [1, 2, 2, 2, 2]
    output_shape = [1, 4, 2, 2, 2]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)
    weights = numpy_helper.from_array(
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(4, 1, 1, 1, 1),
        name="weights",
    )
    bias = numpy_helper.from_array(
        np.asarray([0.25, -0.5, 1.0, -2.0], dtype=np.float32),
        name="bias",
    )
    node = helper.make_node(
        "Conv",
        ["x", "weights", "bias"],
        ["y"],
        name="grouped_conv3d",
        group=2,
        pads=[0, 0, 0, 0, 0, 0],
        strides=[1, 1, 1],
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "grouped_conv3d_graph",
            [x],
            [y],
            initializer=[weights, bias],
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10
    return model


def test_grouped_conv3d_lowers_to_sliced_builtin_convolutions() -> None:
    model = _make_grouped_conv3d_model()
    register_default_preprocess_rules()
    preprocessed, _ = run_preprocess_pipeline(onnx_graph=model)

    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed,
        output_file_name="grouped_conv3d",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    # The post-lowering cleanup canonicalizes the adjacent channel slices to
    # one SPLIT while preserving the grouped-convolution partition.
    assert op_types.count("SPLIT") == 1
    assert op_types.count("CONV_3D") == 2
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("CUSTOM") == 0
    concat = next(op for op in model_ir.operators if op.op_type == "CONCATENATION")
    assert concat.options["axis"] == 4


def test_grouped_conv3d_tflite_matches_onnx(tmp_path: Path) -> None:
    model = _make_grouped_conv3d_model()
    result = export_tflite_model_flatbuffer_direct(
        onnx_graph=model,
        output_folder_path=str(tmp_path),
        output_file_name="grouped_conv3d",
        flatbuffer_direct_allow_custom_ops=False,
    )
    sample = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 2, 2) / 8.0

    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"x": sample})[0]

    interpreter = Interpreter(model_path=result["float32_tflite_path"])
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    assert list(input_detail["shape"]) == list(sample.shape)
    # Direct TFLite public spatial tensors use NDHWC rather than ONNX NCDHW.
    sample_ndhwc = np.transpose(sample, (0, 2, 3, 4, 1))
    interpreter.set_tensor(int(input_detail["index"]), sample_ndhwc)
    interpreter.invoke()
    actual = interpreter.get_tensor(int(interpreter.get_output_details()[0]["index"]))

    expected_ndhwc = np.transpose(expected, (0, 2, 3, 4, 1))
    np.testing.assert_allclose(actual, expected_ndhwc, rtol=1e-6, atol=1e-6)
