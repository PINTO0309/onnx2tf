from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.pytorch_onnx_utils import (
    _onnx_evaluate_constant_binary_elementwise,
    _onnx_evaluate_constant_reshape,
    _onnx_evaluate_constant_scatter_nd,
    _onnx_fold_constant_binary_elementwise_in_place,
    _onnx_fold_constant_reshape_in_place,
    _onnx_get_initializer_array,
)


def test_pytorch_onnx_constant_evaluators() -> None:
    scattered = _onnx_evaluate_constant_scatter_nd(
        data_array=np.zeros([3], dtype=np.float32),
        indices_array=np.asarray([[1]], dtype=np.int64),
        updates_array=np.asarray([4.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(scattered, np.asarray([0.0, 4.0, 0.0]))

    reshaped = _onnx_evaluate_constant_reshape(
        data_array=np.arange(6, dtype=np.float32),
        shape_array=np.asarray([2, 3], dtype=np.int64),
    )
    assert list(np.asarray(reshaped).shape) == [2, 3]

    multiplied = _onnx_evaluate_constant_binary_elementwise(
        lhs_array=np.asarray([1.0, 2.0], dtype=np.float32),
        rhs_array=np.asarray(3.0, dtype=np.float32),
        op_type="Mul",
    )
    np.testing.assert_array_equal(multiplied, np.asarray([3.0, 6.0]))


def test_pytorch_onnx_constant_fold_pipeline() -> None:
    data = numpy_helper.from_array(np.arange(6, dtype=np.float32), name="data")
    shape = numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), name="shape")
    bias = numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), name="bias")
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["data", "shape"], ["reshaped"]),
            helper.make_node("Add", ["reshaped", "bias"], ["output"]),
        ],
        "constant_fold",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])],
        initializer=[data, shape, bias],
    )

    _onnx_fold_constant_reshape_in_place(graph)
    _onnx_fold_constant_binary_elementwise_in_place(graph)

    assert list(graph.node) == []
    output = _onnx_get_initializer_array(graph, "output")
    np.testing.assert_array_equal(
        output,
        np.arange(6, dtype=np.float32).reshape(2, 3) + 2.0,
    )
