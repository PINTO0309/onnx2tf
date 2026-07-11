from __future__ import annotations

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

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


def test_prepare_onnxruntime_graph_decomposes_group_norm_with_swish() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 4])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 4])
    scale = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    bias = np.asarray([-0.2, -0.1, 0.1, 0.2], dtype=np.float32)
    node = helper.make_node(
        "GroupNorm",
        ["x", "scale", "bias"],
        ["y"],
        name="group_norm",
        activation=1,
        epsilon=1e-5,
        groups=2,
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "group_norm",
            [x_info],
            [y_info],
            [
                numpy_helper.from_array(scale, name="scale"),
                numpy_helper.from_array(bias, name="bias"),
            ],
        ),
        opset_imports=[
            helper.make_operatorsetid("", 14),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"GroupNorm": 1}
    assert all(node.op_type != "GroupNorm" for node in prepared.graph.node)
    assert model.graph.node[0].op_type == "GroupNorm"
    x = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4) / 8.0
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(["y"], {"x": x})[0]
    grouped = x.reshape(1, 2, 2, 2, 2)
    mean = np.mean(grouped, axis=(1, 2, 4), keepdims=True)
    variance = np.mean(np.square(grouped - mean), axis=(1, 2, 4), keepdims=True)
    normalized = ((grouped - mean) / np.sqrt(variance + 1e-5)).reshape(x.shape)
    affine = normalized * scale.reshape(1, 1, 1, 4) + bias.reshape(1, 1, 1, 4)
    expected = affine / (1.0 + np.exp(-affine))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
