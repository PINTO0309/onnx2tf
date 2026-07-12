import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.utils.onnx_reference_compat import create_reference_evaluator


def test_reference_lstm_three_outputs_matches_onnxruntime() -> None:
    rng = np.random.default_rng(0)
    hidden_size = 3
    input_size = 2
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 1, input_size])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 1, 1, hidden_size])
    y_h = helper.make_tensor_value_info("y_h", TensorProto.FLOAT, [1, 1, hidden_size])
    y_c = helper.make_tensor_value_info("y_c", TensorProto.FLOAT, [1, 1, hidden_size])
    w = numpy_helper.from_array(
        rng.normal(0.0, 0.1, size=(1, 4 * hidden_size, input_size)).astype(np.float32),
        name="w",
    )
    r = numpy_helper.from_array(
        rng.normal(0.0, 0.1, size=(1, 4 * hidden_size, hidden_size)).astype(np.float32),
        name="r",
    )
    b = numpy_helper.from_array(
        rng.normal(0.0, 0.1, size=(1, 8 * hidden_size)).astype(np.float32),
        name="b",
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "LSTM",
                    ["x", "w", "r", "b"],
                    ["y", "y_h", "y_c"],
                    hidden_size=hidden_size,
                )
            ],
            "three_output_lstm",
            [x],
            [y, y_h, y_c],
            [w, r, b],
        ),
        opset_imports=[helper.make_operatorsetid("", 16)],
    )
    sample = rng.normal(0.0, 0.2, size=(4, 1, input_size)).astype(np.float32)

    expected = ort.InferenceSession(
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(None, {"x": sample})
    actual = create_reference_evaluator(model).run(None, {"x": sample})

    assert len(actual) == 3
    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-6)
