from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes import pytorch_recurrent
from onnx2tf.tflite_builder.passes.pytorch_recurrent import (
    _can_direct_codegen_sequence_lstm_op,
    _can_direct_codegen_sequence_rnn_op,
    _repair_orphan_recurrent_step_tensors,
    _rewrite_recurrent_ops_for_native_export,
    _sequence_lstm_index_spec,
)
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
)


def _constant_tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=np.zeros(shape, dtype=np.float32),
    )


def _direct_rnn_model_ir() -> tuple[ModelIR, OperatorIR]:
    model_ir = ModelIR(name="direct_rnn")
    model_ir.tensors["x"] = TensorIR("x", "FLOAT32", [2, 1, 3], [2, 1, 3])
    model_ir.tensors["weight"] = _constant_tensor("weight", [4, 3])
    model_ir.tensors["recurrent"] = _constant_tensor("recurrent", [4, 4])
    model_ir.tensors["bias"] = _constant_tensor("bias", [4])
    model_ir.tensors["y"] = TensorIR("y", "FLOAT32", [2, 1, 4], [2, 1, 4])
    op = OperatorIR(
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        ["x", "weight", "recurrent", "bias"],
        ["y"],
        {"timeMajor": True, "fusedActivationFunction": "TANH"},
    )
    model_ir.operators.append(op)
    return model_ir, op


def _direct_lstm_15_input_model_ir() -> tuple[ModelIR, OperatorIR]:
    model_ir = ModelIR(name="direct_lstm")
    input_names = ["x"] + [f"const_{index}" for index in range(1, 13)] + [
        "hidden_state",
        "cell_state",
    ]
    model_ir.tensors["x"] = TensorIR("x", "FLOAT32", [2, 1, 3], [2, 1, 3])
    for index in range(1, 13):
        model_ir.tensors[f"const_{index}"] = _constant_tensor(
            f"const_{index}",
            [4, 3] if index <= 4 else [4, 4] if index <= 8 else [4],
        )
    model_ir.tensors["hidden_state"] = TensorIR(
        "hidden_state", "FLOAT32", [1, 4], [1, 4]
    )
    model_ir.tensors["cell_state"] = TensorIR(
        "cell_state", "FLOAT32", [1, 4], [1, 4]
    )
    model_ir.tensors["y"] = TensorIR("y", "FLOAT32", [2, 1, 4], [2, 1, 4])
    op = OperatorIR(
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        input_names,
        ["y"],
        {
            "timeMajor": True,
            "fusedActivationFunction": "TANH",
            "cellClip": 0.0,
            "projClip": 0.0,
        },
    )
    model_ir.operators.append(op)
    return model_ir, op


def test_recurrent_index_specs_cover_supported_legacy_arities() -> None:
    assert _sequence_lstm_index_spec(
        OperatorIR("UNIDIRECTIONAL_SEQUENCE_LSTM", [""] * 15, ["y"])
    ) is not None
    assert _sequence_lstm_index_spec(
        OperatorIR("UNIDIRECTIONAL_SEQUENCE_LSTM", [""] * 24, ["y"])
    ) is not None
    assert _sequence_lstm_index_spec(
        OperatorIR("BIDIRECTIONAL_SEQUENCE_LSTM", [""] * 29, ["y"])
    ) is not None
    assert _sequence_lstm_index_spec(
        OperatorIR("BIDIRECTIONAL_SEQUENCE_LSTM", [""] * 48, ["y"])
    ) is not None
    assert _sequence_lstm_index_spec(
        OperatorIR("UNIDIRECTIONAL_SEQUENCE_LSTM", [""] * 14, ["y"])
    ) is None


def test_direct_recurrent_capabilities_accept_complete_constant_contracts() -> None:
    rnn_model, rnn_op = _direct_rnn_model_ir()
    lstm_model, lstm_op = _direct_lstm_15_input_model_ir()

    assert _can_direct_codegen_sequence_rnn_op(rnn_model, rnn_op) is True
    assert _can_direct_codegen_sequence_lstm_op(lstm_model, lstm_op) is True
    assert _rewrite_recurrent_ops_for_native_export(rnn_model) is rnn_model
    assert _rewrite_recurrent_ops_for_native_export(lstm_model) is lstm_model


def test_direct_recurrent_preflight_scans_root_once() -> None:
    class _CountingOperatorList(list):
        def __init__(self, values):
            super().__init__(values)
            self.iteration_count = 0

        def __iter__(self):
            self.iteration_count += 1
            return super().__iter__()

    model_ir, _ = _direct_rnn_model_ir()
    operators = _CountingOperatorList(model_ir.operators)
    model_ir.operators = operators

    rewritten = _rewrite_recurrent_ops_for_native_export(model_ir)

    assert rewritten is model_ir
    assert operators.iteration_count == 1


def test_direct_rnn_capability_rejects_dynamic_weight_and_non_time_major() -> None:
    model_ir, op = _direct_rnn_model_ir()
    model_ir.tensors["weight"].data = None
    assert _can_direct_codegen_sequence_rnn_op(model_ir, op) is False

    model_ir.tensors["weight"] = _constant_tensor("weight", [4, 3])
    op.options["timeMajor"] = False
    assert _can_direct_codegen_sequence_rnn_op(model_ir, op) is False


def test_recurrent_unroll_failure_is_normalized(monkeypatch) -> None:
    model_ir, op = _direct_rnn_model_ir()
    op.options["timeMajor"] = False

    def fail_unroll(*, model_ir: ModelIR):
        raise ValueError(model_ir.name)

    monkeypatch.setattr(
        pytorch_recurrent,
        "rewrite_model_ir_unroll_recurrent_ops",
        fail_unroll,
    )

    with pytest.raises(ModelIRPyTorchExportError, match="could not rewrite"):
        _rewrite_recurrent_ops_for_native_export(model_ir)


def test_orphan_recurrent_step_repair_uses_one_differential_index(monkeypatch) -> None:
    model_ir = ModelIR(name="orphan_recurrent_step")
    model_ir.inputs = ["source"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "source": TensorIR("source", "FLOAT32", [1, 4], [1, 4]),
        "state_h_step_shape_3": TensorIR(
            "state_h_step_shape_3",
            "INT32",
            [2],
            [2],
            data=np.asarray([1, 4], dtype=np.int32),
        ),
        "state_h_step_3": TensorIR("state_h_step_3", "FLOAT32", [1, 4], [1, 4]),
        "reshaped_state": TensorIR("reshaped_state", "FLOAT32", [1, 4], [1, 4]),
        "y": TensorIR("y", "FLOAT32", [1, 4], [1, 4]),
    }
    model_ir.operators = [
        OperatorIR(
            "RESHAPE",
            ["source", "state_h_step_shape_3"],
            ["reshaped_state"],
        ),
        OperatorIR("ADD", ["state_h_step_3", "reshaped_state"], ["y"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    _repair_orphan_recurrent_step_tensors(model_ir)

    assert refresh_count == 1
    assert model_ir.operators[1].inputs == ["reshaped_state", "reshaped_state"]
    assert "state_h_step_3" not in model_ir.tensors


def test_orphan_recurrent_step_repair_skips_index_without_candidates(
    monkeypatch,
) -> None:
    model_ir, _ = _direct_rnn_model_ir()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    _repair_orphan_recurrent_step_tensors(model_ir)

    assert refresh_count == 0
