from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def _make_sequence_lstm_model():
    hidden_size = 2
    feature_size = 3
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 1, feature_size])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 1, hidden_size])
    w = numpy_helper.from_array(
        np.zeros([1, 4 * hidden_size, feature_size], dtype=np.float32),
        name="w",
    )
    r = numpy_helper.from_array(
        np.zeros([1, 4 * hidden_size, hidden_size], dtype=np.float32),
        name="r",
    )
    b = numpy_helper.from_array(
        np.zeros([1, 8 * hidden_size], dtype=np.float32),
        name="b",
    )
    node = helper.make_node(
        "LSTM",
        ["x", "w", "r", "b"],
        ["y"],
        name="SequenceLSTM",
        hidden_size=hidden_size,
    )
    graph = helper.make_graph([node], "sequence_lstm", [x], [y], initializer=[w, r, b])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


def test_recurrent_input_keeps_onnx_sequence_layout_when_nhwc_inputs_enabled() -> None:
    model_ir = lower_onnx_to_ir(
        _make_sequence_lstm_model(),
        output_file_name="recurrent_input_layout",
        transpose_inputs_to_nhwc=True,
    )

    assert model_ir.inputs == ["x"]
    assert model_ir.tensors["x"].shape == [2, 1, 3]
    assert model_ir.tensors["x"].shape_signature == [2, 1, 3]
    assert "x_onnx_ncx_internal" not in model_ir.tensors
    lstm_op = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "UNIDIRECTIONAL_SEQUENCE_LSTM"
    )
    assert lstm_op.inputs[0] == "x"
