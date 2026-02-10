from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class QuantParamIR:
    scale: List[float]
    zero_point: List[int]
    quantized_dimension: int = 0
    min: Optional[List[float]] = None
    max: Optional[List[float]] = None


@dataclass
class TensorIR:
    name: str
    dtype: str
    shape: List[int]
    shape_signature: Optional[List[int]] = None
    data: Optional[np.ndarray] = None
    is_variable: bool = False
    quantization: Optional[Union[Dict[str, Any], QuantParamIR]] = None


@dataclass
class OperatorIR:
    op_type: str
    inputs: List[str]
    outputs: List[str]
    options: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


@dataclass
class ModelIR:
    name: str
    description: str = "onnx2tf flatbuffer_direct"
    tensors: Dict[str, TensorIR] = field(default_factory=dict)
    operators: List[OperatorIR] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


def normalize_dim_to_shape_and_signature(dim: Any) -> Tuple[int, int]:
    if isinstance(dim, (int, np.integer)):
        if int(dim) >= 0:
            return int(dim), int(dim)
    return 1, -1


def normalize_onnx_shape(shape: Optional[List[Any]]) -> Tuple[List[int], List[int]]:
    if shape is None:
        return [1], [-1]
    norm_shape: List[int] = []
    signature: List[int] = []
    for dim in shape:
        s, sig = normalize_dim_to_shape_and_signature(dim)
        norm_shape.append(s)
        signature.append(sig)
    if len(norm_shape) == 0:
        # Scalar tensors are represented as rank-1 in many tflite paths.
        return [1], [1]
    return norm_shape, signature


def clone_model_ir_with_float16(model_ir: ModelIR) -> ModelIR:
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
        ) for op in model_ir.operators
    ]
    for name, tensor in model_ir.tensors.items():
        new_data = tensor.data
        new_dtype = tensor.dtype
        if tensor.dtype == "FLOAT32":
            new_dtype = "FLOAT16"
            if tensor.data is not None:
                new_data = tensor.data.astype(np.float16)
        clone.tensors[name] = TensorIR(
            name=tensor.name,
            dtype=new_dtype,
            shape=list(tensor.shape),
            shape_signature=list(tensor.shape_signature) if tensor.shape_signature is not None else None,
            data=new_data.copy() if isinstance(new_data, np.ndarray) else new_data,
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
