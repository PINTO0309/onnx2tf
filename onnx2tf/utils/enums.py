import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from onnx import TensorProto

ONNX_DTYPES_TO_TF_DTYPES = {
    TensorProto.FLOAT16: tf.float16,
    TensorProto.FLOAT: tf.float32,
    TensorProto.DOUBLE: tf.float64,

    TensorProto.UINT8: tf.uint8,
    TensorProto.UINT16: tf.uint16,
    TensorProto.UINT32: tf.uint32,
    TensorProto.UINT64: tf.uint64,

    TensorProto.INT8: tf.int8,
    TensorProto.INT16: tf.int16,
    TensorProto.INT32: tf.int32,
    TensorProto.INT64: tf.int64,

    TensorProto.BOOL: tf.bool,

    TensorProto.STRING: tf.string,

    # tf.qint8
    # tf.qint16
    # tf.qint32
    # tf.quint8
    # tf.quint16

    # tf.complex64
    # tf.complex128
}

NUMPY_DTYPES_TO_TF_DTYPES = {
    np.dtype('float16'): tf.float16,
    np.dtype('float32'): tf.float32,
    np.dtype('float64'): tf.float64,

    np.dtype('uint8'): tf.uint8,
    np.dtype('uint16'): tf.uint16,
    np.dtype('uint32'): tf.uint32,
    np.dtype('uint64'): tf.uint64,

    np.dtype('int8'): tf.int8,
    np.dtype('int16'): tf.int16,
    np.dtype('int32'): tf.int32,
    np.dtype('int64'): tf.int64,

    np.dtype('bool_'): tf.bool,
}

TF_DTYPES_TO_NUMPY_DTYPES = {
    tf.float16: np.dtype('float16'),
    tf.float32: np.dtype('float32'),
    tf.float64: np.dtype('float64'),

    tf.uint8: np.dtype('uint8'),
    tf.uint16: np.dtype('uint16'),
    tf.uint32: np.dtype('uint32'),
    tf.uint64: np.dtype('uint64'),

    tf.int8: np.dtype('int8'),
    tf.int16: np.dtype('int16'),
    tf.int32: np.dtype('int32'),
    tf.int64: np.dtype('int64'),

    tf.bool: np.dtype('bool_'),
}
