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


def _make_integer_topk_model(*, dtype: np.dtype, largest: int) -> onnx.ModelProto:
    tensor_type = TensorProto.INT64 if dtype == np.dtype(np.int64) else TensorProto.INT32
    x = helper.make_tensor_value_info("x", tensor_type, [2, 5])
    values = helper.make_tensor_value_info("values", tensor_type, [2, 3])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3])
    k = numpy_helper.from_array(np.asarray([3], dtype=np.int64), name="k")
    topk = helper.make_node(
        "TopK",
        ["x", "k"],
        ["values", "indices"],
        name="integer_topk",
        axis=1,
        largest=largest,
        sorted=1,
    )
    model = helper.make_model(
        helper.make_graph(
            [topk],
            "integer_topk_graph",
            [x],
            [values, indices],
            initializer=[k],
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 10
    return model


@pytest.mark.parametrize("dtype", [np.dtype(np.int32), np.dtype(np.int64)])
@pytest.mark.parametrize("largest", [0, 1])
def test_integer_topk_lowers_to_builtin(dtype: np.dtype, largest: int) -> None:
    model = _make_integer_topk_model(dtype=dtype, largest=largest)
    register_default_preprocess_rules()
    preprocessed, _ = run_preprocess_pipeline(onnx_graph=model)

    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed,
        output_file_name="integer_topk",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TOPK_V2") == 1
    assert op_types.count("CUSTOM") == 0


@pytest.mark.parametrize("dtype", [np.dtype(np.int32), np.dtype(np.int64)])
@pytest.mark.parametrize("largest", [0, 1])
def test_integer_topk_tflite_matches_onnx(
    tmp_path: Path,
    dtype: np.dtype,
    largest: int,
) -> None:
    model = _make_integer_topk_model(dtype=dtype, largest=largest)
    result = export_tflite_model_flatbuffer_direct(
        onnx_graph=model,
        output_folder_path=str(tmp_path),
        output_file_name=f"integer_topk_{dtype.name}_{largest}",
        flatbuffer_direct_allow_custom_ops=False,
    )
    sample = np.asarray(
        [[9, -4, 7, 2, 0], [3, 8, -5, 6, 1]],
        dtype=dtype,
    )
    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"x": sample})

    interpreter = Interpreter(model_path=result["float32_tflite_path"])
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(int(input_detail["index"]), sample)
    interpreter.invoke()
    actual = {
        str(detail["name"]).split(":")[0]: interpreter.get_tensor(int(detail["index"]))
        for detail in interpreter.get_output_details()
    }

    np.testing.assert_array_equal(actual["values"], expected[0])
    np.testing.assert_array_equal(actual["indices"], expected[1])
