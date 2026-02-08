from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import TensorIR


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
}


def tflite_dtype_from_numpy(np_dtype: np.dtype) -> str:
    np_dtype = np.dtype(np_dtype)
    if np_dtype not in _NP_DTYPE_TO_TFLITE_DTYPE:
        raise NotImplementedError(f"Unsupported numpy dtype for flatbuffer_direct: {np_dtype}")
    return _NP_DTYPE_TO_TFLITE_DTYPE[np_dtype]


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
        tensor_obj.shape = list(tensor.shape)
        tensor_obj.shapeSignature = (
            list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else list(tensor.shape)
        )
        tensor_obj.type = getattr(schema_tflite["TensorType"], tensor.dtype)
        tensor_obj.isVariable = bool(tensor.is_variable)

        if tensor.data is not None:
            b = schema_tflite["BufferT"]()
            if isinstance(tensor.data, np.ndarray):
                b.data = bytes(tensor.data.tobytes())
            else:
                b.data = bytes(tensor.data)
            buffer_index = len(buffer_table)
            buffer_table.append(b)
            tensor_obj.buffer = buffer_index
        else:
            tensor_obj.buffer = 0

        tensor_index_map[tensor_name] = len(tensor_table)
        tensor_table.append(tensor_obj)

    return tensor_table, buffer_table, tensor_index_map
