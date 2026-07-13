from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
)
from onnx2tf.tflite_builder.split_planner import (
    rewrite_model_ir_unroll_recurrent_ops,
)


def _sequence_lstm_input_name(op: OperatorIR, index: int) -> str:
    if int(index) < 0 or int(index) >= len(op.inputs):
        return ""
    return str(op.inputs[int(index)]).strip()


def _tensor_has_constant_data(model_ir: ModelIR, tensor_name: str) -> bool:
    if str(tensor_name).strip() == "":
        return False
    tensor = model_ir.tensors.get(str(tensor_name), None)
    return tensor is not None and isinstance(tensor.data, np.ndarray)


def _sequence_lstm_bias_inputs_supported(
    model_ir: ModelIR,
    op: OperatorIR,
    indices: Sequence[int],
) -> bool:
    bias_names = [_sequence_lstm_input_name(op, int(index)) for index in list(indices)]
    non_empty_bias_names = [name for name in bias_names if name != ""]
    if len(non_empty_bias_names) == 0:
        return True
    if len(non_empty_bias_names) != len(bias_names):
        return False
    return all(_tensor_has_constant_data(model_ir, name) for name in non_empty_bias_names)


def _sequence_lstm_index_spec(op: OperatorIR) -> Optional[Dict[str, Any]]:
    op_type = str(op.op_type)
    input_count = int(len(op.inputs))
    if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
        if input_count >= 24:
            return {
                "required_const_indices": [1, 2, 3, 4, 5, 6, 7, 8],
                "unsupported_optional_indices": [9, 10, 11, 16, 17, 20, 21, 22, 23],
                "weight_input_indices": [1, 2, 3, 4],
                "recurrent_input_indices": [5, 6, 7, 8],
                "bias_indices": [12, 13, 14, 15],
                "state_indices": [18, 19],
            }
        if input_count == 15:
            return {
                "required_const_indices": [1, 2, 3, 4, 5, 6, 7, 8],
                "unsupported_optional_indices": [],
                "weight_input_indices": [1, 2, 3, 4],
                "recurrent_input_indices": [5, 6, 7, 8],
                "bias_indices": [9, 10, 11, 12],
                "state_indices": [13, 14],
            }
        return None
    if op_type == "BIDIRECTIONAL_SEQUENCE_LSTM":
        if input_count >= 48:
            return {
                "required_const_indices": [
                    1, 2, 3, 4, 5, 6, 7, 8,
                    18, 19, 20, 21, 22, 23, 24, 25,
                ],
                "unsupported_optional_indices": [9, 10, 11, 16, 17, 26, 27, 28, 33, 34, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                "fw_weight_input_indices": [1, 2, 3, 4],
                "fw_recurrent_input_indices": [5, 6, 7, 8],
                "fw_bias_indices": [12, 13, 14, 15],
                "bw_weight_input_indices": [18, 19, 20, 21],
                "bw_recurrent_input_indices": [22, 23, 24, 25],
                "bw_bias_indices": [29, 30, 31, 32],
                "state_indices": [35, 36, 37, 38],
            }
        if input_count == 29:
            return {
                "required_const_indices": [
                    1, 2, 3, 4, 5, 6, 7, 8,
                    13, 14, 15, 16, 17, 18, 19, 20,
                ],
                "unsupported_optional_indices": [],
                "fw_weight_input_indices": [1, 2, 3, 4],
                "fw_recurrent_input_indices": [5, 6, 7, 8],
                "fw_bias_indices": [9, 10, 11, 12],
                "bw_weight_input_indices": [13, 14, 15, 16],
                "bw_recurrent_input_indices": [17, 18, 19, 20],
                "bw_bias_indices": [21, 22, 23, 24],
                "state_indices": [25, 26, 27, 28],
            }
        return None
    return None


def _can_direct_codegen_sequence_lstm_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    if op_type not in {"UNIDIRECTIONAL_SEQUENCE_LSTM", "BIDIRECTIONAL_SEQUENCE_LSTM"}:
        return False
    index_spec = _sequence_lstm_index_spec(op)
    if index_spec is None:
        return False
    options = dict(op.options)
    if not bool(options.get("timeMajor", True)):
        return False
    if str(options.get("fusedActivationFunction", "TANH")).upper() != "TANH":
        return False
    if abs(float(options.get("cellClip", 0.0))) > 1e-12:
        return False
    if abs(float(options.get("projClip", 0.0))) > 1e-12:
        return False
    if len(op.outputs) != 1:
        return False

    required_const_indices = list(index_spec["required_const_indices"])
    unsupported_optional_indices = list(index_spec["unsupported_optional_indices"])
    if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
        bias_indices = list(index_spec["bias_indices"])
    else:
        bias_indices = list(index_spec["fw_bias_indices"]) + list(index_spec["bw_bias_indices"])

    if any(not _tensor_has_constant_data(model_ir, _sequence_lstm_input_name(op, idx)) for idx in required_const_indices):
        return False
    if any(_sequence_lstm_input_name(op, idx) != "" for idx in unsupported_optional_indices):
        return False
    if not _sequence_lstm_bias_inputs_supported(model_ir, op, bias_indices):
        return False
    return True


def _can_direct_codegen_sequence_rnn_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> bool:
    if str(op.op_type) != "UNIDIRECTIONAL_SEQUENCE_RNN":
        return False
    options = dict(op.options)
    if not bool(options.get("timeMajor", True)):
        return False
    if str(options.get("fusedActivationFunction", "TANH")).upper() not in {"TANH", "RELU"}:
        return False
    if len(op.outputs) != 1 or len(op.inputs) < 4:
        return False
    required_const_indices = [1, 2, 3]
    if any(
        not _tensor_has_constant_data(model_ir, _sequence_lstm_input_name(op, idx))
        for idx in required_const_indices
    ):
        return False
    weight_name = _sequence_lstm_input_name(op, 1)
    recurrent_name = _sequence_lstm_input_name(op, 2)
    bias_name = _sequence_lstm_input_name(op, 3)
    if weight_name == "" or recurrent_name == "" or bias_name == "":
        return False
    weight_tensor = model_ir.tensors.get(weight_name, None)
    recurrent_tensor = model_ir.tensors.get(recurrent_name, None)
    bias_tensor = model_ir.tensors.get(bias_name, None)
    if weight_tensor is None or recurrent_tensor is None or bias_tensor is None:
        return False
    weight_shape = [int(v) for v in list(weight_tensor.shape)]
    recurrent_shape = [int(v) for v in list(recurrent_tensor.shape)]
    bias_shape = [int(v) for v in list(bias_tensor.shape)]
    if len(weight_shape) != 2 or len(recurrent_shape) != 2 or len(bias_shape) != 1:
        return False
    hidden_size = int(weight_shape[0])
    return (
        hidden_size > 0
        and int(recurrent_shape[0]) == hidden_size
        and int(recurrent_shape[1]) == hidden_size
        and int(bias_shape[0]) == hidden_size
    )


def _rewrite_recurrent_ops_for_native_export(model_ir: ModelIR) -> ModelIR:
    recurrent_op_types = {
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if not any(str(op.op_type) in recurrent_op_types for op in model_ir.operators):
        return model_ir
    if all(
        (
            str(op.op_type) == "UNIDIRECTIONAL_SEQUENCE_RNN"
            and _can_direct_codegen_sequence_rnn_op(model_ir, op)
        )
        or (
            str(op.op_type) in {"UNIDIRECTIONAL_SEQUENCE_LSTM", "BIDIRECTIONAL_SEQUENCE_LSTM"}
            and _can_direct_codegen_sequence_lstm_op(model_ir, op)
        )
        or str(op.op_type) not in recurrent_op_types
        for op in model_ir.operators
    ):
        return model_ir
    try:
        rewritten_model_ir, _ = rewrite_model_ir_unroll_recurrent_ops(
            model_ir=model_ir,
        )
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter could not rewrite recurrent sequence ops "
            "for native export."
        ) from ex
    return rewritten_model_ir

