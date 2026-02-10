from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR


_DYNAMIC_RANGE_KERNEL_OPS = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "FULLY_CONNECTED",
}

_DYNAMIC_RANGE_CONST_DEQUANT_OPS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "CONCATENATION",
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


def _symmetric_int8_quantize(data: np.ndarray) -> Tuple[np.ndarray, float]:
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


def _symmetric_int8_quantize_per_channel(
    data: np.ndarray,
    axis: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=np.float32)
    if axis < 0:
        axis += data.ndim
    if axis < 0 or axis >= data.ndim:
        raise ValueError(f"Invalid quantization axis {axis} for data rank {data.ndim}")

    reduce_axes = tuple(i for i in range(data.ndim) if i != axis)
    max_abs = np.max(np.abs(data), axis=reduce_axes, keepdims=True)
    scales = max_abs / 127.0
    scales = np.where(scales == 0.0, 1.0, scales).astype(np.float32)
    q = np.round(data / scales)
    q = np.clip(q, -127, 127).astype(np.int8)
    scales_1d = np.squeeze(scales, axis=reduce_axes).astype(np.float32)
    return q, scales_1d


def _is_float_constant_tensor(
    *,
    tensor: TensorIR,
    graph_input_names: Set[str],
) -> bool:
    if tensor.name in graph_input_names:
        return False
    if tensor.is_variable:
        return False
    if tensor.dtype != "FLOAT32":
        return False
    if not isinstance(tensor.data, np.ndarray):
        return False
    return True


def _quantize_tensor_inplace(
    *,
    tensor: TensorIR,
    quant_mode: str = "per-tensor",
    quantized_dimension: int = 0,
) -> bool:
    if tensor.dtype == "INT8" and isinstance(tensor.quantization, QuantParamIR):
        return True
    if tensor.dtype != "FLOAT32":
        return False
    if not isinstance(tensor.data, np.ndarray):
        return False

    if quant_mode == "per-tensor":
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
    elif quant_mode == "per-channel":
        q_data, scales = _symmetric_int8_quantize_per_channel(
            tensor.data,
            axis=quantized_dimension,
        )
        tensor.dtype = "INT8"
        tensor.data = q_data
        tensor.quantization = QuantParamIR(
            scale=[float(v) for v in scales.reshape(-1).tolist()],
            zero_point=[0 for _ in range(int(scales.size))],
            min=None,
            max=None,
            quantized_dimension=int(quantized_dimension),
        )
    else:
        raise ValueError(f"Unsupported quantization mode: {quant_mode}")
    return True


def _make_unique_tensor_name(base: str, tensors: Dict[str, TensorIR]) -> str:
    if base not in tensors:
        return base
    serial = 1
    while f"{base}_{serial}" in tensors:
        serial += 1
    return f"{base}_{serial}"


def _kernel_weight_quant_axis(op_type: str, tensor: TensorIR) -> int:
    if op_type in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
        return len(tensor.shape) - 1
    if op_type == "FULLY_CONNECTED":
        return 0
    return 0


def build_dynamic_range_quantized_model_ir(
    model_ir: ModelIR,
    quant_type: str = "per-channel",
) -> ModelIR:
    clone = _clone_model_ir(model_ir)
    graph_input_names = set(clone.inputs)
    quant_type = str(quant_type).strip().lower()
    if quant_type not in ["per-channel", "per-tensor"]:
        raise ValueError(
            f"flatbuffer_direct dynamic-range quantization quant_type must be one of "
            f"[\"per-channel\", \"per-tensor\"]. got: {quant_type}"
        )

    quantized_tensor_names: Set[str] = set()
    dequantized_tensor_map: Dict[str, str] = {}

    rewritten_ops: List[OperatorIR] = []

    def ensure_dequantized_tensor(quant_tensor_name: str) -> Optional[str]:
        if quant_tensor_name in dequantized_tensor_map:
            return dequantized_tensor_map[quant_tensor_name]

        if quant_tensor_name not in clone.tensors:
            return None
        q_tensor = clone.tensors[quant_tensor_name]
        if q_tensor.dtype != "INT8":
            return None

        deq_name = _make_unique_tensor_name(f"{quant_tensor_name}_dequantized", clone.tensors)
        clone.tensors[deq_name] = TensorIR(
            name=deq_name,
            dtype="FLOAT32",
            shape=list(q_tensor.shape),
            shape_signature=list(q_tensor.shape_signature)
            if q_tensor.shape_signature is not None
            else list(q_tensor.shape),
            data=None,
            is_variable=False,
            quantization=None,
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="DEQUANTIZE",
                inputs=[quant_tensor_name],
                outputs=[deq_name],
                options={},
            )
        )
        dequantized_tensor_map[quant_tensor_name] = deq_name
        return deq_name

    for op in clone.operators:
        new_op = OperatorIR(
            op_type=op.op_type,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            options=dict(op.options),
            version=op.version,
        )

        if new_op.op_type in _DYNAMIC_RANGE_KERNEL_OPS and len(new_op.inputs) >= 2:
            weight_name = new_op.inputs[1]
            tensor = clone.tensors.get(weight_name)
            if tensor is not None and _is_float_constant_tensor(
                tensor=tensor,
                graph_input_names=graph_input_names,
            ):
                quant_mode = "per-channel" if quant_type == "per-channel" else "per-tensor"
                quant_axis = _kernel_weight_quant_axis(new_op.op_type, tensor)
                if _quantize_tensor_inplace(
                    tensor=tensor,
                    quant_mode=quant_mode,
                    quantized_dimension=quant_axis,
                ):
                    quantized_tensor_names.add(weight_name)

        if new_op.op_type in _DYNAMIC_RANGE_CONST_DEQUANT_OPS:
            for idx, input_name in enumerate(list(new_op.inputs)):
                tensor = clone.tensors.get(input_name)
                if tensor is None:
                    continue
                if _is_float_constant_tensor(tensor=tensor, graph_input_names=graph_input_names):
                    if _quantize_tensor_inplace(tensor=tensor, quant_mode="per-tensor"):
                        quantized_tensor_names.add(input_name)
                tensor = clone.tensors.get(input_name)
                if tensor is None or tensor.dtype != "INT8":
                    continue
                deq_name = ensure_dequantized_tensor(input_name)
                if deq_name is not None:
                    new_op.inputs[idx] = deq_name

        rewritten_ops.append(new_op)

    clone.operators = rewritten_ops

    if len(quantized_tensor_names) == 0:
        raise NotImplementedError(
            "flatbuffer_direct dynamic-range quantization requires at least one quantizable float32 constant tensor "
            "(kernel weights for CONV_2D/DEPTHWISE_CONV_2D/FULLY_CONNECTED or constants used by "
            "ADD/SUB/MUL/DIV/CONCATENATION)."
        )

    clone.description = f"{clone.description} (dynamic_range_quantized)"
    return clone
