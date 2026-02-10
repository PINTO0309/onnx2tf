from __future__ import annotations

from typing import Dict, List

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR


_DYNAMIC_RANGE_TARGET_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
}


def _clone_model_ir(model_ir: ModelIR) -> ModelIR:
    clone = ModelIR(
        name=model_ir.name,
        description=model_ir.description,
    )
    clone.inputs = list(model_ir.inputs)
    clone.outputs = list(model_ir.outputs)
    clone.operators = [
        OperatorIR(
            op_type=op.op_type,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            options=dict(op.options),
            version=op.version,
        )
        for op in model_ir.operators
    ]
    for name, tensor in model_ir.tensors.items():
        clone.tensors[name] = TensorIR(
            name=tensor.name,
            dtype=tensor.dtype,
            shape=list(tensor.shape),
            shape_signature=list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else None,
            data=tensor.data.copy() if isinstance(tensor.data, np.ndarray) else tensor.data,
            is_variable=tensor.is_variable,
            quantization=(
                dict(tensor.quantization)
                if isinstance(tensor.quantization, dict)
                else QuantParamIR(
                    scale=list(tensor.quantization.scale),
                    zero_point=list(tensor.quantization.zero_point),
                    quantized_dimension=int(tensor.quantization.quantized_dimension),
                    min=list(tensor.quantization.min)
                    if tensor.quantization.min is not None
                    else None,
                    max=list(tensor.quantization.max)
                    if tensor.quantization.max is not None
                    else None,
                )
                if isinstance(tensor.quantization, QuantParamIR)
                else tensor.quantization
            ),
        )
    return clone


def _symmetric_int8_quantize(data: np.ndarray) -> tuple[np.ndarray, float]:
    data = np.asarray(data, dtype=np.float32)
    max_abs = float(np.max(np.abs(data))) if data.size > 0 else 0.0
    if max_abs == 0.0:
        scale = 1.0
        q = np.zeros_like(data, dtype=np.int8)
        return q, scale
    scale = max_abs / 127.0
    q = np.round(data / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, float(scale)


def build_dynamic_range_quantized_model_ir(model_ir: ModelIR) -> ModelIR:
    clone = _clone_model_ir(model_ir)

    quantized_weight_names: List[str] = []
    for op in clone.operators:
        if op.op_type not in _DYNAMIC_RANGE_TARGET_OPS:
            continue
        if len(op.inputs) < 2:
            continue
        weight_name = op.inputs[1]
        if weight_name not in clone.tensors:
            continue
        tensor = clone.tensors[weight_name]
        if tensor.dtype != "FLOAT32":
            continue
        if not isinstance(tensor.data, np.ndarray):
            continue

        q_data, scale = _symmetric_int8_quantize(tensor.data)
        tensor.dtype = "INT8"
        tensor.data = q_data
        tensor.quantization = QuantParamIR(
            scale=[float(scale)],
            zero_point=[0],
            min=[float(np.min(tensor.data.astype(np.float32) * scale))],
            max=[float(np.max(tensor.data.astype(np.float32) * scale))],
            quantized_dimension=0,
        )
        quantized_weight_names.append(weight_name)

    if len(quantized_weight_names) == 0:
        raise NotImplementedError(
            "flatbuffer_direct dynamic-range quantization requires at least one supported weight tensor "
            "(CONV_2D/DEPTHWISE_CONV_2D/FULLY_CONNECTED)."
        )

    clone.description = f"{clone.description} (dynamic_range_quantized)"
    return clone
