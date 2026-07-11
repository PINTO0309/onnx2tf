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


def test_prepare_onnxruntime_graph_repairs_if_sequenceconstruct_tensor_alias() -> None:
    cond_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x1_info = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 2])
    x2_info = helper.make_tensor_value_info("x2", TensorProto.FLOAT, [2, 2])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])

    def branch(*, name: str, include_x2: bool):
        one = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32))
        nodes = [
            helper.make_node(
                "Constant",
                [],
                [f"{name}_one"],
                name=f"{name}_constant",
                value=one,
            ),
            helper.make_node(
                "Add",
                ["x1", f"{name}_one"],
                [f"{name}_x1"],
                name=f"{name}_add_x1",
            ),
        ]
        sequence_inputs = [f"{name}_x1"]
        value_infos = [
            helper.make_tensor_value_info(
                f"{name}_x1", TensorProto.FLOAT, [1, 2]
            )
        ]
        if include_x2:
            nodes.append(
                helper.make_node(
                    "Add",
                    ["x2", f"{name}_one"],
                    [f"{name}_x2"],
                    name=f"{name}_add_x2",
                )
            )
            sequence_inputs.append(f"{name}_x2")
            value_infos.append(
                helper.make_tensor_value_info(
                    f"{name}_x2", TensorProto.FLOAT, [2, 2]
                )
            )
        sequence_output = f"{name}_sequence"
        nodes.append(
            helper.make_node(
                "SequenceConstruct",
                sequence_inputs,
                [sequence_output],
                name=f"{name}_sequence_node",
            )
        )
        graph = helper.make_graph(
            nodes,
            name,
            [],
            [helper.make_tensor_value_info(sequence_output, TensorProto.FLOAT, [])],
            value_info=value_infos,
        )
        return graph

    if_node = helper.make_node(
        "If",
        ["cond"],
        ["y"],
        name="if_sequence",
        then_branch=branch(name="then", include_x2=False),
        else_branch=branch(name="else", include_x2=True),
    )
    model = helper.make_model(
        helper.make_graph(
            [if_node],
            "if_sequence",
            [cond_info, x1_info, x2_info],
            [y_info],
        ),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"IfSequenceConstruct": 1}
    prepared_if = prepared.graph.node[0]
    branch_terminals = {
        attribute.name: attribute.g.node[-1].op_type
        for attribute in prepared_if.attribute
    }
    assert branch_terminals == {
        "else_branch": "Concat",
        "then_branch": "Identity",
    }
    session = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    x1 = np.asarray([[1.0, 2.0]], dtype=np.float32)
    x2 = np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    actual_then = session.run(["y"], {"cond": np.asarray(True), "x1": x1, "x2": x2})[0]
    actual_else = session.run(["y"], {"cond": np.asarray(False), "x1": x1, "x2": x2})[0]
    np.testing.assert_array_equal(actual_then, x1 + 1.0)
    np.testing.assert_array_equal(actual_else, np.concatenate([x1 + 1.0, x2 + 1.0], axis=0))


def test_prepare_onnxruntime_graph_rewrites_integer_matmul_like_direct_lowerer() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 2])
    y_info = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 2])
    weights = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    node = helper.make_node("MatMul", ["x", "weights"], ["y"], name="matmul")
    model = helper.make_model(
        helper.make_graph(
            [node],
            "integer_matmul",
            [x_info],
            [y_info],
            [numpy_helper.from_array(weights, name="weights")],
        ),
        opset_imports=[helper.make_operatorsetid("", 14)],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"IntegerMatMul": 1}
    assert [node.op_type for node in prepared.graph.node] == [
        "Cast",
        "Cast",
        "MatMul",
        "Cast",
    ]
    x = np.asarray([[2, 3], [4, 5]], dtype=np.uint8)
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(["y"], {"x": x})[0]
    expected = np.matmul(x.astype(np.float32), weights.astype(np.float32)).astype(
        np.uint8
    )
    np.testing.assert_array_equal(actual, expected)


def test_prepare_onnxruntime_graph_folds_optional_has_element_tensor_alias() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y_info = helper.make_tensor_value_info("y", TensorProto.BOOL, [])
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "OptionalHasElement",
                    ["x"],
                    ["has_element"],
                    name="optional_has_element",
                ),
                helper.make_node(
                    "Identity", ["has_element"], ["y"], name="identity"
                ),
            ],
            "optional_tensor_alias",
            [x_info],
            [y_info],
        ),
        opset_imports=[helper.make_operatorsetid("", 15)],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"TensorOptionalHasElement": 1}
    assert prepared.graph.node[0].op_type == "Constant"
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(["y"], {"x": np.asarray([1.0, 2.0], dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, np.asarray(True, dtype=np.bool_))


def test_prepare_onnxruntime_graph_materializes_unknown_rank_fused_conv_io() -> None:
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [])
    weights = np.ones((4, 3, 3, 3), dtype=np.float32)
    bias = np.zeros((4,), dtype=np.float32)
    node = helper.make_node(
        "FusedConv",
        ["x", "weights", "bias"],
        ["y"],
        name="fused_conv",
        activation="Relu",
        pads=[1, 1, 1, 1],
    )
    model = helper.make_model(
        helper.make_graph(
            [node],
            "unknown_rank_fused_conv",
            [x_info],
            [y_info],
            [
                numpy_helper.from_array(weights, name="weights"),
                numpy_helper.from_array(bias, name="bias"),
            ],
        ),
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )

    prepared, rewritten = prepare_onnx_graph_for_onnxruntime(model)

    assert rewritten == {"FusedConv": 1, "UnknownRankConv": 1}
    input_dims = prepared.graph.input[0].type.tensor_type.shape.dim
    output_dims = prepared.graph.output[0].type.tensor_type.shape.dim
    assert len(input_dims) == 4
    assert input_dims[1].dim_value == 3
    assert len(output_dims) == 4
    assert output_dims[1].dim_value == 4
    actual = ort.InferenceSession(
        prepared.SerializeToString(),
        providers=["CPUExecutionProvider"],
    ).run(["y"], {"x": np.ones((1, 3, 1, 1), dtype=np.float32)})[0]
    np.testing.assert_array_equal(actual, np.full((1, 4, 1, 1), 3.0, dtype=np.float32))
