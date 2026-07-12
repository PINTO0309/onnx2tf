from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.flatbuffer_direct_fingerprint import model_ir_fingerprint


def _make_qlinear_activation_model(op_type: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [2, 3])
    initializers = [
        numpy_helper.from_array(np.asarray(0.125, dtype=np.float32), "xs"),
        numpy_helper.from_array(np.asarray(-8, dtype=np.int8), "xz"),
        numpy_helper.from_array(np.asarray(1.0 / 256.0, dtype=np.float32), "ys"),
        numpy_helper.from_array(np.asarray(-128, dtype=np.int8), "yz"),
    ]
    attributes = {}
    if op_type == "QLinearLeakyRelu":
        attributes["alpha"] = 0.125
    elif op_type == "QLinearSoftmax":
        attributes.update({"axis": -1, "opset": 13})
    node = helper.make_node(
        op_type,
        ["x", "xs", "xz", "ys", "yz"],
        ["y"],
        name=f"{op_type}Node",
        domain="com.microsoft",
        **attributes,
    )
    return helper.make_model(
        helper.make_graph(
            [node],
            f"{op_type}Graph",
            [x],
            [y],
            initializer=initializers,
        ),
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )


@pytest.mark.parametrize(
    ("op_type", "expected_fingerprint"),
    [
        (
            "QLinearSigmoid",
            "67e5b3d23cf2cfe03ae8ef1a006ac5fecf221f328553d3c1904ceebad9a7d902",
        ),
        (
            "QLinearLeakyRelu",
            "f1d0b1b74e6f0f056ca595912efcceb2827da416b059dc12992fd06ed137ab09",
        ),
        (
            "QLinearSoftmax",
            "56aef3cabbed33cabcaba95d36058a37b6a12428102f7e83b0aef334eadbb4ec",
        ),
    ],
)
def test_qlinear_activation_family_preserves_pre_extraction_model_ir_fingerprint(
    op_type: str,
    expected_fingerprint: str,
) -> None:
    assert model_ir_fingerprint(
        _make_qlinear_activation_model(op_type),
        op_type,
    ) == expected_fingerprint
