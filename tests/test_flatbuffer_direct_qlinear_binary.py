from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.flatbuffer_direct_fingerprint import model_ir_fingerprint


def _make_qlinear_binary_model(op_type: str) -> onnx.ModelProto:
    a = helper.make_tensor_value_info("a", TensorProto.INT8, [2, 3])
    b = helper.make_tensor_value_info("b", TensorProto.INT8, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [2, 3])
    initializers = [
        numpy_helper.from_array(np.asarray(0.03, dtype=np.float32), "as"),
        numpy_helper.from_array(np.asarray(-7, dtype=np.int8), "az"),
        numpy_helper.from_array(np.asarray(0.07, dtype=np.float32), "bs"),
        numpy_helper.from_array(np.asarray(11, dtype=np.int8), "bz"),
        numpy_helper.from_array(np.asarray(0.04, dtype=np.float32), "ys"),
        numpy_helper.from_array(np.asarray(-3, dtype=np.int8), "yz"),
    ]
    node = helper.make_node(
        op_type,
        ["a", "as", "az", "b", "bs", "bz", "ys", "yz"],
        ["y"],
        name=f"{op_type}Node",
        domain="com.microsoft",
    )
    return helper.make_model(
        helper.make_graph(
            [node],
            f"{op_type}Graph",
            [a, b],
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
            "QLinearAdd",
            "d2f0714a44b2dc376827b845269a217c1df894986f3957128994a2913d611c24",
        ),
        (
            "QLinearMul",
            "b4d9d1a39202474faf52ab43fbde4938fe892a0a38c5739a87b6da2d9b882b34",
        ),
    ],
)
def test_qlinear_binary_family_preserves_pre_extraction_model_ir_fingerprint(
    op_type: str,
    expected_fingerprint: str,
) -> None:
    assert model_ir_fingerprint(
        _make_qlinear_binary_model(op_type),
        op_type,
    ) == expected_fingerprint
