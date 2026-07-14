from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_recurrent_codegen_policy import (
    _require_constant_array_from_model_ir,
    _sequence_lstm_bias_array_for_model_ir,
)


def test_sequence_lstm_omitted_biases_create_zero_synthetic_tensor() -> None:
    model_ir = ModelIR(name="omitted_lstm_biases")
    op = OperatorIR(
        op_type="UNIDIRECTIONAL_SEQUENCE_LSTM",
        inputs=["input", "", "", "", ""],
        outputs=["output"],
    )

    bias_name = _sequence_lstm_bias_array_for_model_ir(
        model_ir=model_ir,
        op=op,
        indices=[1, 2, 3, 4],
        hidden_size=2,
        dtype="FLOAT32",
        base_name="lstm_bias",
        synthetic_tensor_serial_ref=[0],
    )

    assert bias_name in model_ir.tensors
    np.testing.assert_array_equal(
        model_ir.tensors[bias_name].data,
        np.zeros((8,), dtype=np.float32),
    )


def test_sequence_lstm_present_biases_preserve_gate_order() -> None:
    model_ir = ModelIR(
        name="present_lstm_biases",
        tensors={
            "bias_0": TensorIR(
                name="bias_0",
                dtype="FLOAT32",
                shape=[2],
                data=np.asarray([1.0, 2.0], dtype=np.float32),
            ),
            "bias_1": TensorIR(
                name="bias_1",
                dtype="FLOAT32",
                shape=[2],
                data=np.asarray([3.0, 4.0], dtype=np.float32),
            ),
        },
    )
    op = OperatorIR(
        op_type="UNIDIRECTIONAL_SEQUENCE_LSTM",
        inputs=["input", "bias_0", "bias_1"],
        outputs=["output"],
    )

    bias_name = _sequence_lstm_bias_array_for_model_ir(
        model_ir=model_ir,
        op=op,
        indices=[1, 2],
        hidden_size=1,
        dtype="FLOAT32",
        base_name="lstm_bias",
        synthetic_tensor_serial_ref=[0],
    )

    np.testing.assert_array_equal(
        _require_constant_array_from_model_ir(
            model_ir=model_ir,
            tensor_name=bias_name,
            context="test LSTM bias",
        ),
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
