from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tests.flatbuffer_direct_fingerprint import model_ir_fingerprint


def _make_qlinear_pool_model(op_type: str) -> onnx.ModelProto:
    is_global = op_type == "QLinearGlobalAveragePool"
    input_shape = [1, 2, 4, 4] if is_global else [1, 1, 2, 2]
    output_shape = [1, 2, 1, 1] if is_global else [1, 1, 1, 1]
    prefix = "qgap" if is_global else "qavg"
    node_prefix = "QGap" if is_global else "QAvg"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)
    initializers = [
        numpy_helper.from_array(
            np.asarray([0.25], dtype=np.float32), f"{prefix}_x_scale"
        ),
        numpy_helper.from_array(np.asarray([0], dtype=np.int8), f"{prefix}_x_zero"),
        numpy_helper.from_array(
            np.asarray([0.125], dtype=np.float32), f"{prefix}_y_scale"
        ),
        numpy_helper.from_array(np.asarray([0], dtype=np.int8), f"{prefix}_y_zero"),
    ]
    x_scale, x_zero, y_scale, y_zero = [
        initializer.name for initializer in initializers
    ]
    attributes = (
        {"channels_last": 0}
        if is_global
        else {
            "kernel_shape": [2, 2],
            "strides": [2, 2],
        }
    )
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", x_scale, x_zero],
            ["x_q"],
            name=f"{node_prefix}Q0",
        ),
        helper.make_node(
            op_type,
            ["x_q", x_scale, x_zero, y_scale, y_zero],
            ["y_q"],
            name=f"{node_prefix}Node",
            **attributes,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", y_scale, y_zero],
            ["y"],
            name=f"{node_prefix}DQ0",
        ),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            (
                "qlinear_global_average_pool_graph"
                if is_global
                else "qlinear_average_pool_graph"
            ),
            [x],
            [y],
            initializer=initializers,
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


@pytest.mark.parametrize(
    ("op_type", "expected_fingerprint"),
    [
        (
            "QLinearAveragePool",
            "0bb8b9064ae208810addbcebb27846b05873d817e947a5af212f3fd8ee4a6b7c",
        ),
        (
            "QLinearGlobalAveragePool",
            "1b066e8245cb45f79df76dbc052ecf7485f07d7910fb789cff38b47c298b7f19",
        ),
    ],
)
def test_qlinear_pool_family_preserves_pre_extractionmodel_ir_fingerprint(
    op_type: str,
    expected_fingerprint: str,
) -> None:
    assert (
        model_ir_fingerprint(
            _make_qlinear_pool_model(op_type),
            op_type,
        )
        == expected_fingerprint
    )
