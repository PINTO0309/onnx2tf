from __future__ import annotations

from typing import List, Sequence

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.pytorch_recurrent import (
    _sequence_lstm_input_name,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _add_synthetic_tensor_to_model_ir,
)
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
)


def _require_constant_array_from_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
    context: str,
) -> np.ndarray:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        raise ModelIRPyTorchExportError(
            f"Native PyTorch-like model.py codegen requires constant tensor data for {context}. "
            f"tensor={tensor_name}"
        )
    return np.asarray(tensor.data)

def _sequence_lstm_bias_array_for_model_ir(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    indices: Sequence[int],
    hidden_size: int,
    dtype: str,
    base_name: str,
    synthetic_tensor_serial_ref: List[int],
) -> str:
    bias_names = [_sequence_lstm_input_name(op, int(index)) for index in list(indices)]
    if all(name == "" for name in bias_names):
        return _add_synthetic_tensor_to_model_ir(
            model_ir=model_ir,
            base_name=base_name,
            data=np.zeros((4 * int(hidden_size),), dtype=np.float32),
            dtype=str(dtype),
            synthetic_tensor_serial_ref=synthetic_tensor_serial_ref,
        )
    if any(name == "" for name in bias_names):
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen requires LSTM gate biases to be either all present or all omitted."
        )
    concatenated = np.concatenate(
        [
            _require_constant_array_from_model_ir(
                model_ir=model_ir,
                tensor_name=name,
                context=f"LSTM bias gate {index}",
            ).reshape(-1)
            for index, name in enumerate(bias_names)
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return _add_synthetic_tensor_to_model_ir(
        model_ir=model_ir,
        base_name=base_name,
        data=concatenated,
        dtype=str(dtype),
        synthetic_tensor_serial_ref=synthetic_tensor_serial_ref,
    )
