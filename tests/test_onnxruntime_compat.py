from __future__ import annotations

from onnx import TensorProto, helper

from onnx2tf.utils.onnxruntime_compat import prepare_onnx_graph_for_onnxruntime


def _grid_sample_model(*, opset: int, include_inverse: bool):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 2])
    grid = helper.make_tensor_value_info("grid", TensorProto.FLOAT, [1, 1, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1, 1])
    nodes = []
    grid_input = "x"
    if include_inverse:
        nodes.append(helper.make_node("Inverse", ["x"], ["x_inv"], name="inverse"))
        grid_input = "x_inv"
    nodes.append(helper.make_node("GridSample", [grid_input, "grid"], ["y"], name="grid"))
    return helper.make_model(
        helper.make_graph(nodes, "grid_sample", [x, grid], [y]),
        opset_imports=[helper.make_operatorsetid("", int(opset))],
    )


def test_prepare_onnxruntime_graph_redomains_legacy_grid_sample_and_inverse() -> None:
    model = _grid_sample_model(opset=10, include_inverse=True)

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"GridSample": 1, "Inverse": 1}
    assert [node.domain for node in prepared.graph.node] == [
        "com.microsoft",
        "com.microsoft",
    ]
    assert [node.domain for node in model.graph.node] == ["", ""]
    assert any(opset.domain == "com.microsoft" for opset in prepared.opset_import)


def test_prepare_onnxruntime_graph_keeps_standard_grid_sample_at_opset_16() -> None:
    model = _grid_sample_model(opset=16, include_inverse=False)

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {}
    assert prepared.graph.node[0].domain == ""
