from __future__ import annotations

import struct
from typing import Dict, List, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import QuantParamIR, TensorIR


_NP_DTYPE_TO_TFLITE_DTYPE = {
    np.dtype(np.float16): "FLOAT16",
    np.dtype(np.float32): "FLOAT32",
    np.dtype(np.float64): "FLOAT64",
    np.dtype(np.int8): "INT8",
    np.dtype(np.int16): "INT16",
    np.dtype(np.int32): "INT32",
    np.dtype(np.int64): "INT64",
    np.dtype(np.uint8): "UINT8",
    np.dtype(np.uint16): "UINT16",
    np.dtype(np.uint32): "UINT32",
    np.dtype(np.uint64): "UINT64",
    np.dtype(np.bool_): "BOOL",
    np.dtype(np.object_): "STRING",
}


_INT32_MAX = np.iinfo(np.int32).max


def tflite_dtype_from_numpy(np_dtype: np.dtype) -> str:
    np_dtype = np.dtype(np_dtype)
    if np_dtype not in _NP_DTYPE_TO_TFLITE_DTYPE:
        if np_dtype.kind in {"U", "S", "O"}:
            return "STRING"
        raise NotImplementedError(f"Unsupported numpy dtype for flatbuffer_direct: {np_dtype}")
    return _NP_DTYPE_TO_TFLITE_DTYPE[np_dtype]


def _serialize_tflite_string_buffer(values: np.ndarray) -> bytes:
    flat_values = np.asarray(values, dtype=object).reshape(-1).tolist()
    encoded: List[bytes] = []
    for value in flat_values:
        if isinstance(value, bytes):
            encoded.append(bytes(value))
        else:
            encoded.append(str(value).encode("utf-8"))

    num_strings = int(len(encoded))
    header_size = int(4 * (num_strings + 2))
    offsets: List[int] = [header_size]
    cursor = int(header_size)
    for item in encoded:
        cursor += int(len(item))
        offsets.append(cursor)

    buffer = bytearray()
    buffer.extend(struct.pack("<i", num_strings))
    for off in offsets:
        buffer.extend(struct.pack("<i", int(off)))
    for item in encoded:
        buffer.extend(item)
    return bytes(buffer)


def _sanitize_runtime_shape(shape: List[int]) -> List[int]:
    # TFLite Tensor.shape must be concrete non-negative int32 dims.
    sanitized: List[int] = []
    for dim in shape:
        value = int(dim)
        if value < 0 or value > _INT32_MAX:
            sanitized.append(1)
        else:
            sanitized.append(value)
    return sanitized


def build_tensors_and_buffers(
    schema_tflite: Dict,
    tensors: Dict[str, TensorIR],
) -> Tuple[List[object], List[object], Dict[str, int]]:
    buffer_table = []
    tensor_table = []
    tensor_index_map: Dict[str, int] = {}

    # Buffer[0] must be empty.
    empty_buffer = schema_tflite["BufferT"]()
    empty_buffer.data = bytes()
    buffer_table.append(empty_buffer)

    for tensor_name, tensor in tensors.items():
        tensor_obj = schema_tflite["TensorT"]()
        tensor_obj.name = tensor.name
        original_shape = [int(v) for v in list(tensor.shape)]
        tensor_obj.shape = _sanitize_runtime_shape(original_shape)
        tensor_obj.shapeSignature = (
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else list(original_shape)
        )
        tensor_obj.type = getattr(schema_tflite["TensorType"], tensor.dtype)
        tensor_obj.isVariable = bool(tensor.is_variable)
        quant_params = tensor.quantization
        if isinstance(quant_params, QuantParamIR):
            quant_params = {
                "scale": list(quant_params.scale),
                "zero_point": list(quant_params.zero_point),
                "min": list(quant_params.min) if quant_params.min is not None else None,
                "max": list(quant_params.max) if quant_params.max is not None else None,
                "quantized_dimension": int(quant_params.quantized_dimension),
            }

        if isinstance(quant_params, dict):
            q = schema_tflite["QuantizationParametersT"]()
            if "scale" in quant_params:
                q.scale = [float(v) for v in quant_params["scale"]]
            if "zero_point" in quant_params:
                q.zeroPoint = [int(v) for v in quant_params["zero_point"]]
            if "min" in quant_params and quant_params["min"] is not None:
                q.min = [float(v) for v in quant_params["min"]]
            if "max" in quant_params and quant_params["max"] is not None:
                q.max = [float(v) for v in quant_params["max"]]
            q.quantizedDimension = int(quant_params.get("quantized_dimension", 0))
            tensor_obj.quantization = q

        if tensor.data is not None:
            b = schema_tflite["BufferT"]()
            if isinstance(tensor.data, np.ndarray):
                if str(tensor.dtype).upper() == "STRING":
                    b.data = _serialize_tflite_string_buffer(tensor.data)
                else:
                    b.data = tensor.data.tobytes()
            else:
                b.data = tensor.data if isinstance(tensor.data, bytes) else bytes(tensor.data)
            buffer_index = len(buffer_table)
            buffer_table.append(b)
            tensor_obj.buffer = buffer_index
        else:
            tensor_obj.buffer = 0

        tensor_index_map[tensor_name] = len(tensor_table)
        tensor_table.append(tensor_obj)

    return tensor_table, buffer_table, tensor_index_map
