from __future__ import annotations

from onnx import TensorProto, helper

from onnx2tf.tflite_builder.pytorch_onnx_bridge_passes import (
    _onnx_fold_concat_layout_bridges_in_place,
    _onnx_fold_softmax_layout_bridges_in_place,
)


def _value(name: str, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def test_pytorch_onnx_concat_layout_bridge_is_folded() -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["x1"], ["x1_nhwc"], name="t1", perm=[0, 2, 3, 1]),
            helper.make_node("Transpose", ["x2"], ["x2_nhwc"], name="t2", perm=[0, 2, 3, 1]),
            helper.make_node("Concat", ["x1_nhwc", "x2_nhwc"], ["joined"], name="concat", axis=3),
            helper.make_node("Transpose", ["joined"], ["restored"], name="back", perm=[0, 3, 1, 2]),
            helper.make_node("Identity", ["restored"], ["y"], name="consumer"),
        ],
        "concat_bridge",
        [_value("x1", [1, 2, 3, 4]), _value("x2", [1, 5, 3, 4])],
        [_value("y", [1, 7, 3, 4])],
    )
    _onnx_fold_concat_layout_bridges_in_place(graph)
    assert [node.op_type for node in graph.node] == ["Concat", "Identity"]
    concat = graph.node[0]
    assert list(concat.input) == ["x1", "x2"]
    assert helper.get_attribute_value(concat.attribute[0]) == 1
    assert list(graph.node[1].input) == ["joined"]


def test_pytorch_onnx_softmax_layout_bridge_is_folded() -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Transpose", ["x"], ["nhwc"], name="to_nhwc", perm=[0, 2, 3, 1]),
            helper.make_node("Softmax", ["nhwc"], ["softmax"], name="softmax", axis=3),
            helper.make_node("Transpose", ["softmax"], ["restored"], name="to_nchw", perm=[0, 3, 1, 2]),
            helper.make_node("Identity", ["restored"], ["y"], name="consumer"),
        ],
        "softmax_bridge",
        [_value("x", [1, 3, 4, 5])],
        [_value("y", [1, 3, 4, 5])],
    )
    _onnx_fold_softmax_layout_bridges_in_place(graph)
    assert [node.op_type for node in graph.node] == ["Softmax", "Identity"]
    softmax = graph.node[0]
    assert list(softmax.input) == ["x"]
    assert helper.get_attribute_value(softmax.attribute[0]) == 1
    assert list(graph.node[1].input) == ["softmax"]
