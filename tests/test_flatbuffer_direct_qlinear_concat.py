from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tests.flatbuffer_direct_fingerprint import model_ir_fingerprint


def _make_qlinear_concat_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
    initializers = [
        numpy_helper.from_array(
            np.asarray([0.25], dtype=np.float32), "qcat_x_scale"
        ),
        numpy_helper.from_array(np.asarray([0], dtype=np.int8), "qcat_x_zero"),
        numpy_helper.from_array(
            np.asarray([0.25], dtype=np.float32), "qcat_y_scale"
        ),
        numpy_helper.from_array(np.asarray([0], dtype=np.int8), "qcat_y_zero"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "qcat_x_scale", "qcat_x_zero"],
            ["x_q"],
            name="QCatQ0",
        ),
        helper.make_node(
            "QLinearConcat",
            [
                "qcat_y_scale",
                "qcat_y_zero",
                "x_q",
                "qcat_x_scale",
                "qcat_x_zero",
                "x_q",
                "qcat_x_scale",
                "qcat_x_zero",
            ],
            ["y_q"],
            name="QCatNode",
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "qcat_y_scale", "qcat_y_zero"],
            ["y"],
            name="QCatDQ0",
        ),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "qlinear_concat_graph",
            [x],
            [y],
            initializer=initializers,
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


def test_qlinear_concat_family_preserves_pre_extraction_model_ir_fingerprint() -> None:
    assert model_ir_fingerprint(
        _make_qlinear_concat_model(),
        "QLinearConcat",
    ) == "924e1470c62f93ba44dde277144d84bf796f40c5123839b59b44e4cd89c5b927"
