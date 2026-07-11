from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.pytorch_onnx_layout_passes import (
    _onnx_convert_pads_nhwc_to_nchw,
    _onnx_fold_inverse_transpose_pairs_in_place,
    _onnx_fold_relu_layout_bridges_in_place,
)


def _rank4_value(name: str):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3, 4, 5])


def test_pytorch_onnx_pad_conversion_is_deterministic() -> None:
    converted = _onnx_convert_pads_nhwc_to_nchw(
        np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        converted,
        np.asarray([0, 3, 1, 2, 4, 7, 5, 6], dtype=np.int64),
    )


def test_pytorch_onnx_inverse_transpose_pair_is_removed() -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Transpose", ["x"], ["nhwc"], name="to_nhwc", perm=[0, 2, 3, 1]
            ),
            helper.make_node(
                "Transpose", ["nhwc"], ["restored"], name="to_nchw", perm=[0, 3, 1, 2]
            ),
            helper.make_node("Identity", ["restored"], ["y"], name="consumer"),
        ],
        "inverse_transpose",
        [_rank4_value("x")],
        [_rank4_value("y")],
    )
    _onnx_fold_inverse_transpose_pairs_in_place(graph)
    assert [node.op_type for node in graph.node] == ["Identity"]
    assert list(graph.node[0].input) == ["x"]


def test_pytorch_onnx_unary_layout_bridge_is_folded() -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "Transpose", ["x"], ["nhwc"], name="to_nhwc", perm=[0, 2, 3, 1]
            ),
            helper.make_node("Relu", ["nhwc"], ["activated"], name="relu"),
            helper.make_node(
                "Transpose", ["activated"], ["y"], name="to_nchw", perm=[0, 3, 1, 2]
            ),
        ],
        "relu_bridge",
        [_rank4_value("x")],
        [_rank4_value("y")],
    )
    _onnx_fold_relu_layout_bridges_in_place(graph)
    assert [node.op_type for node in graph.node] == ["Relu"]
    assert list(graph.node[0].input) == ["x"]
    assert list(graph.node[0].output) == ["y"]
