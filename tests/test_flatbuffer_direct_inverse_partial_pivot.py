from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from onnx2tf.tflite_builder.accuracy_evaluator import (
    evaluate_onnx_tflite_outputs,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def _make_batched_inverse_model() -> onnx.ModelProto:
    input_value = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 2, 8, 8],
    )
    output_value = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 2, 8, 8],
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Inverse",
                    ["input"],
                    ["output"],
                    name="Inverse8x8",
                    domain="com.microsoft",
                )
            ],
            "batched_inverse_with_row_swap",
            [input_value],
            [output_value],
        ),
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    return model


def _make_row_swap_inputs() -> np.ndarray:
    matrices = np.stack(
        [
            np.eye(8, dtype=np.float32),
            np.eye(8, dtype=np.float32),
        ],
        axis=0,
    )
    matrices[0, [0, 1]] = matrices[0, [1, 0]]
    matrices[0, 0, 2] = np.float32(0.25)
    matrices[0, 1, 3] = np.float32(-0.5)
    matrices[1, 0, 0] = np.float32(-2.0)
    matrices[1, 0, 1] = np.float32(0.75)
    return matrices[None, ...]


def test_inverse_8x8_uses_partial_pivoting_for_zero_and_negative_pivots(
    tmp_path: Path,
) -> None:
    model = _make_batched_inverse_model()
    model_ir = lower_onnx_to_ir(
        onnx_graph=model,
        output_file_name="inverse_partial_pivot",
        allow_custom_ops=False,
        optimize_layout_transpose_chains=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "CUSTOM" not in op_types
    assert op_types.count("ARG_MAX") == 8
    assert op_types.count("SELECT_V2") >= 8

    result = export_tflite_model_flatbuffer_direct(
        onnx_graph=model,
        output_folder_path=str(tmp_path),
        output_file_name="inverse_partial_pivot",
        flatbuffer_direct_allow_custom_ops=False,
    )
    input_path = tmp_path / "inverse_inputs.npy"
    np.save(input_path, _make_row_swap_inputs())
    report = evaluate_onnx_tflite_outputs(
        onnx_graph=model,
        tflite_path=result["float32_tflite_path"],
        output_report_path=str(tmp_path / "inverse_accuracy.json"),
        num_samples=1,
        seed=0,
        custom_input_op_name_np_data_path=[["input", str(input_path)]],
    )

    assert report["evaluation_pass"] is True
    assert report["allclose_summary"]["pass"] is True
    assert report["overall_metrics"]["max_abs"] <= 1.0e-6
