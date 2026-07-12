from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from ai_edge_litert.interpreter import Interpreter
from onnx import TensorProto, helper, numpy_helper

import onnx2tf
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.preprocess import (
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)


def _roi_align_model() -> onnx.ModelProto:
    feature = helper.make_tensor_value_info(
        "feature",
        TensorProto.FLOAT,
        [1, 1, 4, 4],
    )
    output = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 1, 2, 2],
    )
    rois = numpy_helper.from_array(
        np.asarray([[0.0, 0.0, 3.0, 3.0]], dtype=np.float32),
        name="rois",
    )
    batch_indices = numpy_helper.from_array(
        np.asarray([0], dtype=np.int64),
        name="batch_indices",
    )
    node = helper.make_node(
        "RoiAlign",
        ["feature", "rois", "batch_indices"],
        ["output"],
        name="roi_align",
        coordinate_transformation_mode="output_half_pixel",
        mode="avg",
        output_height=2,
        output_width=2,
        sampling_ratio=1,
        spatial_scale=1.0,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "roi_align_graph",
            [feature],
            [output],
            initializer=[rois, batch_indices],
        ),
        opset_imports=[helper.make_operatorsetid("", 16)],
    )
    model.ir_version = 10
    return model


def test_roi_align_gathers_only_sampled_pixels() -> None:
    register_default_preprocess_rules()
    preprocessed, _ = run_preprocess_pipeline(onnx_graph=_roi_align_model())
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed,
        output_file_name="roi_align_memory_test",
        allow_custom_ops=False,
    )

    tensor_names = set(model_ir.tensors)
    assert not any(name.endswith("_roialign_input_gathered") for name in tensor_names)
    assert not any(
        str(operator.op_type) == "PAD"
        and "roialign" in str(operator.outputs[0]).lower()
        for operator in model_ir.operators
    )
    assert any(
        str(operator.op_type) == "GATHER"
        and "roialign_input_flattened" in str(operator.inputs[0])
        and int(operator.options.get("axis", -1)) == 0
        for operator in model_ir.operators
    )


def test_roi_align_memory_efficient_lowering_is_numerically_correct(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "roi_align.onnx"
    output_dir = tmp_path / "out"
    onnx.save(_roi_align_model(), model_path)
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        disable_strict_mode=True,
        verbosity="error",
        tflite_backend="flatbuffer_direct",
    )

    interpreter = Interpreter(
        model_path=str(output_dir / "roi_align_float32.tflite"),
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    feature = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    if list(input_detail["shape"]) == [1, 4, 4, 1]:
        feature = np.transpose(feature, (0, 2, 3, 1))
    interpreter.set_tensor(int(input_detail["index"]), feature)
    interpreter.invoke()
    output_detail = interpreter.get_output_details()[0]
    actual = interpreter.get_tensor(int(output_detail["index"]))

    np.testing.assert_allclose(
        actual.reshape(-1),
        np.asarray([3.75, 5.25, 9.75, 11.25], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
