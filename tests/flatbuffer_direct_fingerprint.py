from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import onnx

from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def model_ir_fingerprint(model: onnx.ModelProto, fixture_name: str) -> str:
    model_ir = lower_onnx_to_ir(
        model,
        output_file_name=f"{fixture_name}_family_fingerprint",
        allow_custom_ops=False,
    )
    payload = {
        "inputs": model_ir.inputs,
        "outputs": model_ir.outputs,
        "operators": [
            {
                "type": operator.op_type,
                "inputs": operator.inputs,
                "outputs": operator.outputs,
                "options": _normalize(operator.options),
                "version": operator.version,
                "custom": getattr(operator, "custom_code", None),
            }
            for operator in model_ir.operators
        ],
        "tensors": {
            name: {
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "signature": tensor.shape_signature,
                "data": _normalize(tensor.data),
                "quant": _normalize(
                    None
                    if tensor.quantization is None
                    else vars(tensor.quantization)
                ),
            }
            for name, tensor in sorted(model_ir.tensors.items())
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()
