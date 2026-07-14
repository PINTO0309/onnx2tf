from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError
from onnx2tf.tflite_builder.pytorch_layout_utils import _normalize_constant_pad_pairs


def _is_small_inline_constant_tensor(tensor: TensorIR) -> bool:
    if tensor.data is None:
        return False
    arr = np.asarray(tensor.data)
    if arr.size > 32:
        return False
    if arr.ndim > 2:
        return False
    return str(tensor.dtype).upper() in {
        "BOOL",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "FLOAT16",
        "FLOAT32",
        "FLOAT64",
    }


def _python_literal_for_constant_tensor(tensor: TensorIR) -> Optional[str]:
    if not _is_small_inline_constant_tensor(tensor):
        return None

    def _python_literal_value(value: Any) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            value = float(value)
            if math.isnan(value):
                return "float('nan')"
            if math.isinf(value):
                return "float('inf')" if value > 0.0 else "float('-inf')"
            return repr(value)
        if isinstance(value, list):
            return "[" + ", ".join(_python_literal_value(item) for item in value) + "]"
        return repr(value)

    arr = np.asarray(tensor.data)
    if arr.ndim == 0:
        return _python_literal_value(arr.reshape(-1)[0].item())
    return _python_literal_value(arr.tolist())


def _torch_pad_literal_for_constant_tensor(
    tensor: Optional[TensorIR],
    *,
    axis_permutation: Optional[Sequence[int]] = None,
) -> Optional[str]:
    if tensor is None or tensor.data is None:
        return None
    pads = _normalize_constant_pad_pairs(np.asarray(tensor.data))
    if pads is None:
        return None
    if axis_permutation is not None:
        perm = [int(v) for v in list(axis_permutation)]
        if len(perm) == len(pads):
            pads = [pads[idx] for idx in perm]
    torch_pad: List[int] = []
    for before, after in reversed(pads):
        torch_pad.extend([int(before), int(after)])
    while len(torch_pad) >= 2 and int(torch_pad[-2]) == 0 and int(torch_pad[-1]) == 0:
        torch_pad = torch_pad[:-2]
    return repr(torch_pad)


def _scalar_literal_for_constant_tensor(tensor: Optional[TensorIR]) -> Optional[str]:
    if tensor is None or tensor.data is None:
        return None
    flat = np.asarray(tensor.data).reshape(-1)
    if int(flat.size) != 1:
        return None
    value = flat[0].item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        value = float(value)
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0.0 else "float('-inf')"
        return repr(value)
    return repr(value)


def _torch_dtype_literal(dtype_name: str) -> str:
    mapping = {
        "BOOL": "torch.bool",
        "INT8": "torch.int8",
        "INT16": "torch.int16",
        "INT32": "torch.int32",
        "INT64": "torch.int64",
        "UINT8": "torch.uint8",
        "FLOAT16": "torch.float16",
        "FLOAT32": "torch.float32",
        "FLOAT64": "torch.float64",
    }
    key = str(dtype_name).upper()
    if key not in mapping:
        raise ModelIRPyTorchExportError(
            f"Unsupported dtype for native PyTorch-like model.py codegen: {dtype_name}"
        )
    return str(mapping[key])


def _conv_block_activation_config(op: OperatorIR) -> Tuple[str, Optional[float]]:
    op_type = str(op.op_type)
    if op_type == "LEAKY_RELU":
        return ("leaky_relu", float(op.options.get("alpha", 0.2)))
    if op_type == "RELU":
        return ("relu", None)
    if op_type == "RELU6":
        return ("relu6", None)
    if op_type == "RELU_N1_TO_1":
        return ("relu_n1_to_1", None)
    if op_type == "RELU_0_TO_1":
        return ("relu_0_to_1", None)
    if op_type == "TANH":
        return ("tanh", None)
    if op_type == "LOGISTIC":
        return ("sigmoid", None)
    return ("none", None)


def _conv_block_activation_config_from_fused_name(
    fused_name: str,
    *,
    alpha: Optional[float] = None,
) -> Tuple[str, Optional[float]]:
    key = str(fused_name).upper()
    if key == "LEAKY_RELU":
        return ("leaky_relu", float(0.2 if alpha is None else alpha))
    if key == "RELU":
        return ("relu", None)
    if key == "RELU6":
        return ("relu6", None)
    if key == "RELU_N1_TO_1":
        return ("relu_n1_to_1", None)
    if key == "RELU_0_TO_1":
        return ("relu_0_to_1", None)
    if key == "TANH":
        return ("tanh", None)
    if key == "LOGISTIC":
        return ("sigmoid", None)
    return ("none", None)
