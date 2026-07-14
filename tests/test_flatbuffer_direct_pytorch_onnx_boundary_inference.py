from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.pytorch_onnx_artifact_support import (
    _infer_batchless_rank3_image_boundaries_from_onnx_graph,
    _infer_public_layouts_from_onnx_graph,
    _is_onnx_boundary_layout_passthrough_node,
    _read_onnx_transpose_perm,
)


def test_onnx_transpose_perm_and_passthrough_classification() -> None:
    transpose = helper.make_node("Transpose", ["x"], ["y"], perm=[0, 3, 1, 2])
    relu = helper.make_node("Relu", ["x"], ["y"])
    conv = helper.make_node("Conv", ["x", "w"], ["y"])

    assert _read_onnx_transpose_perm(transpose) == [0, 3, 1, 2]
    assert _read_onnx_transpose_perm(relu) is None
    assert _is_onnx_boundary_layout_passthrough_node(
        node=relu,
        source_tensor_name="x",
    )
    assert not _is_onnx_boundary_layout_passthrough_node(
        node=conv,
        source_tensor_name="x",
    )


def _layout_boundary_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["x_id"]),
            helper.make_node("Transpose", ["x_id"], ["core"], perm=[0, 3, 1, 2]),
            helper.make_node("Transpose", ["core"], ["y_id"], perm=[0, 2, 3, 1]),
            helper.make_node("Identity", ["y_id"], ["y"]),
        ],
        "layout_boundaries",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 8, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 3])],
    )
    return helper.make_model(graph)


def test_public_layout_inference_walks_short_passthrough_boundary_chains() -> None:
    assert _infer_public_layouts_from_onnx_graph(_layout_boundary_model()) == {
        "x": "NHWC",
        "y": "NHWC",
    }


def test_public_layout_inference_rejects_input_fanout() -> None:
    model = _layout_boundary_model()
    model.graph.node.append(helper.make_node("Relu", ["x"], ["side"]))

    inferred = _infer_public_layouts_from_onnx_graph(model)

    assert "x" not in inferred
    assert inferred["y"] == "NHWC"


def _batchless_boundary_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["image"], ["image_relu"]),
            helper.make_node("Unsqueeze", ["image_relu"], ["batched"], axes=[0]),
            helper.make_node("Squeeze", ["core"], ["squeezed"], axes=[0]),
            helper.make_node("Identity", ["squeezed"], ["image_out"]),
        ],
        "batchless_boundaries",
        [
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [3, 8, 8]),
            helper.make_tensor_value_info("core", TensorProto.FLOAT, [1, 3, 8, 8]),
        ],
        [helper.make_tensor_value_info("image_out", TensorProto.FLOAT, [3, 8, 8])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def test_batchless_rank3_boundary_inference_handles_input_and_output_chains() -> None:
    assert _infer_batchless_rank3_image_boundaries_from_onnx_graph(
        _batchless_boundary_model()
    ) == {"image", "image_out"}


def test_boundary_inference_returns_empty_without_graph() -> None:
    assert _infer_public_layouts_from_onnx_graph(None) == {}
    assert _infer_batchless_rank3_image_boundaries_from_onnx_graph(None) == set()
