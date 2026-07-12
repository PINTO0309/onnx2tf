from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tests.flatbuffer_direct_fingerprint import model_ir_fingerprint


def _make_qlinear_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])
    initializers = [
        numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), "mc_x_scale"),
        numpy_helper.from_array(np.asarray([128], dtype=np.uint8), "mc_x_zero"),
        numpy_helper.from_array(
            np.asarray(
                [
                    [[[1]], [[2]], [[3]]],
                    [[[1]], [[0]], [[-1]]],
                    [[[2]], [[1]], [[0]]],
                    [[[0]], [[1]], [[2]]],
                ],
                dtype=np.int8,
            ),
            "mc_w",
        ),
        numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), "mc_w_scale"),
        numpy_helper.from_array(np.asarray([0], dtype=np.int8), "mc_w_zero"),
        numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), "mc_y_scale"),
        numpy_helper.from_array(np.asarray([127], dtype=np.uint8), "mc_y_zero"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "mc_x_scale", "mc_x_zero"],
            ["x_q"],
            name="QMC0",
        ),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "mc_x_scale",
                "mc_x_zero",
                "mc_w",
                "mc_w_scale",
                "mc_w_zero",
                "mc_y_scale",
                "mc_y_zero",
            ],
            ["y_q"],
            name="QConvMC",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "mc_y_scale", "mc_y_zero"],
            ["y"],
            name="DQMC0",
        ),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "qlinear_conv_multichannel_graph",
            [x],
            [y],
            initializer=initializers,
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


def test_qlinear_conv_family_preserves_pre_extraction_model_ir_fingerprint() -> None:
    assert model_ir_fingerprint(
        _make_qlinear_conv_model(),
        "QLinearConv",
    ) == "c752a5b1e31744e65d483733f55a688f2189d6bf11436cabd498cfc6a2ef5019"
