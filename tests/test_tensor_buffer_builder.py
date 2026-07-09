import numpy as np

from onnx2tf.tflite_builder.ir import TensorIR
from onnx2tf.tflite_builder.tensor_buffer_builder import build_tensors_and_buffers


class _BufferT:
    def __init__(self) -> None:
        self.data = None


class _TensorT:
    def __init__(self) -> None:
        self.name = None
        self.shape = None
        self.shapeSignature = None
        self.type = None
        self.isVariable = None
        self.quantization = None
        self.buffer = None


class _TensorType:
    INT8 = 9


class _QuantizationParametersT:
    pass


_SCHEMA_TFLITE_STUB = {
    "BufferT": _BufferT,
    "TensorT": _TensorT,
    "TensorType": _TensorType,
    "QuantizationParametersT": _QuantizationParametersT,
}


def test_build_tensors_and_buffers_serializes_numpy_scalar_int8_data() -> None:
    tensors = {
        "neg": TensorIR(
            name="neg",
            dtype="INT8",
            shape=[1],
            data=np.int8(-127),
        ),
        "pos": TensorIR(
            name="pos",
            dtype="INT8",
            shape=[1],
            data=np.int8(5),
        ),
    }

    _, buffers, _ = build_tensors_and_buffers(
        schema_tflite=_SCHEMA_TFLITE_STUB,
        tensors=tensors,
    )

    assert getattr(buffers[1], "data") == np.int8(-127).tobytes()
    assert getattr(buffers[2], "data") == np.int8(5).tobytes()
