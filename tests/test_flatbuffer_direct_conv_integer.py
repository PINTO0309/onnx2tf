from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tests.flatbuffer_direct_fingerprint import model_ir_fingerprint


def _make_conv_integer_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 3, 4, 4])
    x_zero = helper.make_tensor_value_info("x_zero", TensorProto.UINT8, [])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [1, 2, 4, 4])
    weights = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [[1, 2, 0], [2, 1, 1], [0, 1, 2]],
                    [[0, 1, 1], [1, 2, 1], [1, 0, 1]],
                    [[1, 0, 2], [1, 1, 1], [2, 1, 0]],
                ],
                [
                    [[2, 1, 1], [1, 0, 1], [1, 2, 0]],
                    [[1, 1, 0], [0, 1, 2], [2, 1, 1]],
                    [[0, 2, 1], [1, 1, 0], [1, 0, 2]],
                ],
            ],
            dtype=np.uint8,
        ),
        "ci_w",
    )
    weight_zero = numpy_helper.from_array(
        np.asarray(1, dtype=np.uint8),
        "ci_w_zero",
    )
    node = helper.make_node(
        "ConvInteger",
        ["x", "ci_w", "x_zero", "ci_w_zero"],
        ["y"],
        name="ConvIntegerNode",
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    return helper.make_model(
        helper.make_graph(
            [node],
            "conv_integer_graph",
            [x, x_zero],
            [y],
            initializer=[weights, weight_zero],
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


def test_conv_integer_family_preserves_pre_extraction_model_ir_fingerprint() -> None:
    assert model_ir_fingerprint(
        _make_conv_integer_model(),
        "ConvInteger",
    ) == "587f53091ce42815e43946d7b73324fe31ec7d5aeb1c3d2d749097351106dfb5"
