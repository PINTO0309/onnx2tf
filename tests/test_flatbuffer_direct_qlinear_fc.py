from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.flatbuffer_direct_fingerprint import model_ir_fingerprint


def _make_qlinear_fc_model(op_type: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [2, 4])
    initializers = [
        numpy_helper.from_array(np.asarray(0.25, dtype=np.float32), "as"),
        numpy_helper.from_array(np.asarray(-3, dtype=np.int8), "az"),
        numpy_helper.from_array(
            np.arange(12, dtype=np.int8).reshape(3, 4) - 6,
            "w",
        ),
        numpy_helper.from_array(np.asarray(0.125, dtype=np.float32), "bs"),
        numpy_helper.from_array(np.asarray(1, dtype=np.int8), "bz"),
        numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "ys"),
        numpy_helper.from_array(np.asarray(-2, dtype=np.int8), "yz"),
    ]
    inputs = ["x", "as", "az", "w", "bs", "bz"]
    domain = ""
    opsets = [helper.make_operatorsetid("", 13)]
    if op_type == "QGemm":
        initializers.append(
            numpy_helper.from_array(
                np.asarray([1, -2, 3, -4], dtype=np.int32),
                "bias",
            )
        )
        inputs.extend(["bias", "ys", "yz"])
        domain = "com.microsoft"
        opsets.append(helper.make_operatorsetid(domain, 1))
    else:
        inputs.extend(["ys", "yz"])
    node = helper.make_node(
        op_type,
        inputs,
        ["y"],
        name=f"{op_type}Node",
        domain=domain,
    )
    return helper.make_model(
        helper.make_graph(
            [node],
            f"{op_type}Graph",
            [x],
            [y],
            initializer=initializers,
        ),
        opset_imports=opsets,
    )


@pytest.mark.parametrize(
    ("op_type", "expected_fingerprint"),
    [
        (
            "QLinearMatMul",
            "633d083445fcf765023a948c038c0956c7a0b7646b73bdac0bb65cf4c14173c8",
        ),
        (
            "QGemm",
            "bf71085f2cc3a5981b209b6d5b02cc65ea55a41251465229a5ef1636a319f70f",
        ),
    ],
)
def test_qlinear_fc_family_preserves_pre_extractionmodel_ir_fingerprint(
    op_type: str,
    expected_fingerprint: str,
) -> None:
    assert model_ir_fingerprint(
        _make_qlinear_fc_model(op_type),
        op_type,
    ) == expected_fingerprint
