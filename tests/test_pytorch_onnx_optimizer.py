from __future__ import annotations

from onnx import TensorProto, helper

from onnx2tf.tflite_builder.pytorch_onnx_optimizer import (
    _optimize_dynamo_exported_onnx_in_place,
)


def test_pytorch_onnx_optimizer_is_deterministic_and_idempotent() -> None:
    value = lambda name: helper.make_tensor_value_info(
        name, TensorProto.FLOAT, [1, 3, 4, 5]
    )
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
        "optimizer",
        [value("x")],
        [value("y")],
    )
    model = helper.make_model(graph)

    _optimize_dynamo_exported_onnx_in_place(model)
    first = model.SerializeToString()
    assert [node.op_type for node in model.graph.node] == ["Relu"]
    assert list(model.graph.node[0].input) == ["x"]
    assert list(model.graph.node[0].output) == ["y"]

    _optimize_dynamo_exported_onnx_in_place(model)
    assert model.SerializeToString() == first
