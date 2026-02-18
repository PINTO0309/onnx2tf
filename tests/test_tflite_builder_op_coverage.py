import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.lower_from_onnx2tf import build_op_coverage_report


def _make_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_axis_non_last_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("Softmax", ["x"], ["y"], name="SoftmaxNode", axis=1)
    graph = helper.make_graph([node], "softmax_axis_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_dynamic_quantize_linear_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])
    y_scale = helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, [])
    y_zero = helper.make_tensor_value_info("y_zero", TensorProto.UINT8, [])
    node = helper.make_node(
        "DynamicQuantizeLinear",
        ["x"],
        ["y", "y_scale", "y_zero"],
        name="DynamicQuantizeLinearNode",
    )
    graph = helper.make_graph([node], "dynamic_quantize_linear_graph", [x], [y, y_scale, y_zero])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_shape_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [2])
    node = helper.make_node(
        "Shape",
        ["x"],
        ["y"],
        name="ShapeNode",
        start=1,
        end=3,
    )
    graph = helper.make_graph([node], "shape_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 15)])


def _make_constant_of_shape_model() -> onnx.ModelProto:
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    shape = numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), name="cos_shape")
    value = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="cos_value")
    node = helper.make_node(
        "ConstantOfShape",
        ["cos_shape"],
        ["y"],
        name="ConstantOfShapeNode",
        value=value,
    )
    graph = helper.make_graph([node], "constant_of_shape_graph", [], [y], initializer=[shape])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_fused_matmul_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    w = numpy_helper.from_array(
        np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        ),
        name="fmm_w",
    )
    node = helper.make_node(
        "FusedMatMul",
        ["x", "fmm_w"],
        ["y"],
        name="FusedMatMulNode",
        alpha=0.125,
        transA=0,
        transB=1,
    )
    graph = helper.make_graph([node], "fused_matmul_graph", [x], [y], initializer=[w])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )


def _make_qlinear_concat_slice_softmax_unknown_rank_model() -> onnx.ModelProto:
    x0_q = helper.make_tensor_value_info("x0_q", TensorProto.UINT8, [])
    x1_q = helper.make_tensor_value_info("x1_q", TensorProto.UINT8, [])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 4, 2])
    part = helper.make_tensor_value_info("part", TensorProto.FLOAT, [1, 4, 1])

    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="x_scale")
    x_zero = numpy_helper.from_array(np.asarray([128], dtype=np.uint8), name="x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="y_scale")
    y_zero = numpy_helper.from_array(np.asarray([128], dtype=np.uint8), name="y_zero")
    starts = numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="starts")
    ends = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="ends")
    axes = numpy_helper.from_array(np.asarray([2], dtype=np.int64), name="axes")
    steps = numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="steps")

    nodes = [
        helper.make_node(
            "QLinearConcat",
            [
                "y_scale",
                "y_zero",
                "x0_q",
                "x_scale",
                "x_zero",
                "x1_q",
                "x_scale",
                "x_zero",
            ],
            ["y_q"],
            name="QCatUnknownRank",
            axis=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "y_scale", "y_zero"],
            ["y"],
            name="DQUnknownRank",
        ),
        helper.make_node(
            "Softmax",
            ["y"],
            ["scores"],
            name="SoftmaxUnknownRank",
            axis=2,
        ),
        helper.make_node(
            "Slice",
            ["y", "starts", "ends", "axes", "steps"],
            ["part"],
            name="SliceUnknownRank",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_concat_slice_softmax_unknown_rank_graph",
        [x0_q, x1_q],
        [scores, part],
        initializer=[x_scale, x_zero, y_scale, y_zero, starts, ends, axes, steps],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_axes_nonconst_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    axes = helper.make_tensor_value_info("axes", TensorProto.INT64, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    node = helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="ReduceNode", keepdims=1)
    graph = helper.make_graph([node], "reduce_nonconst_axes_graph", [x, axes], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_global_average_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 5, 7])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 1, 1])
    node = helper.make_node("GlobalAveragePool", ["x"], ["y"], name="GlobalAveragePoolNode")
    graph = helper.make_graph([node], "global_average_pool_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_grouped_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 5, 5])
    w = numpy_helper.from_array(
        np.ones((6, 2, 3, 3), dtype=np.float32),
        name="W",
    )
    b = numpy_helper.from_array(np.zeros((6,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="GroupedConvNode",
        group=2,
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "grouped_conv_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_nonconst_rhs_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumNode",
        equation="ij,jk->ik",
    )
    graph = helper.make_graph([node], "einsum_nonconst_rhs_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_custom_candidate_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumCustomNode",
        equation="ij,jk->kj",
    )
    graph = helper.make_graph([node], "einsum_custom_candidate_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_const_rhs_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    w = numpy_helper.from_array(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    node = helper.make_node(
        "Einsum",
        ["x", "W"],
        ["z"],
        name="EinsumConstNode",
        equation="ij,jk->ik",
    )
    graph = helper.make_graph([node], "einsum_const_rhs_graph", [x], [z], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_erf_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Erf", ["x"], ["y"], name="ErfNode")
    graph = helper.make_graph([node], "erf_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_tile_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 6])
    repeats = numpy_helper.from_array(np.asarray([2, 2], dtype=np.int64), name="repeats")
    node = helper.make_node("Tile", ["x", "repeats"], ["y"], name="TileNode")
    graph = helper.make_graph([node], "tile_graph", [x], [y], initializer=[repeats])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_scatter_nd_model() -> onnx.ModelProto:
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 3])
    indices = numpy_helper.from_array(
        np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        name="indices",
    )
    node = helper.make_node(
        "ScatterND",
        ["data", "indices", "updates"],
        ["output"],
        name="ScatterNDNode",
    )
    graph = helper.make_graph(
        [node],
        "scatter_nd_graph",
        [data, updates],
        [output],
        initializer=[indices],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)])


def test_op_coverage_report_keys_compatibility_snapshot() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_add_model(),
        output_file_name="add_cov_snapshot",
    )
    assert sorted(report.keys()) == [
        "conversion_error",
        "custom_lowered_nodes",
        "custom_op_candidate_ops",
        "custom_op_policy",
        "graph_custom_ops",
        "graph_node_reports",
        "graph_ops",
        "graph_summary",
        "graph_supported_ops",
        "graph_unsupported_ops",
        "preprocess_report",
        "registry_extra_outside_schema_range",
        "registry_missing_from_schema_range",
        "schema_onnx_ops_target_range",
        "schema_policy_counts",
        "schema_policy_matrix",
        "schema_unresolved_ops",
        "schema_version",
        "supported_onnx_ops_registry",
        "target_opset_max",
        "target_opset_min",
        "unsupported_nodes",
        "unsupported_reason_counts",
    ]
    assert sorted(report["custom_op_policy"].keys()) == [
        "allow_custom_ops",
        "allowlist_builtin_supported_ops",
        "allowlist_custom_candidate_ops",
        "allowlist_unknown_ops",
        "candidate_count",
        "candidate_count_excluding_builtin_supported",
        "candidate_ops_now_builtin_supported",
        "custom_op_allowlist",
    ]
    assert sorted(report["preprocess_report"].keys()) == [
        "applied_rules",
        "enabled_rule_ids",
        "pipeline_version",
        "registered_rule_ids",
        "schema_version",
        "summary",
    ]
    assert sorted(report["preprocess_report"]["summary"].keys()) == [
        "changed_rule_count",
        "enabled_rule_count",
        "executed_rule_count",
        "registered_rule_count",
        "total_matched_nodes",
        "total_rewritten_nodes",
    ]


def test_op_coverage_reason_code_snapshot_validation_failures() -> None:
    softmax_report = build_op_coverage_report(
        onnx_graph=_make_softmax_axis_non_last_model(),
        output_file_name="softmax_reason_snapshot",
    )
    assert softmax_report["unsupported_reason_counts"] == {}
    assert softmax_report["graph_summary"]["unsupported_nodes"] == 0
    assert softmax_report["graph_summary"]["supported_nodes"] == 1
    assert softmax_report["graph_node_reports"][0]["onnx_op"] == "Softmax"
    assert softmax_report["graph_node_reports"][0]["dispatch_mode"] == "builtin"

    reduce_report = build_op_coverage_report(
        onnx_graph=_make_reduce_axes_nonconst_model(),
        output_file_name="reduce_reason_snapshot",
    )
    assert reduce_report["unsupported_reason_counts"] == {"requires_constant_input": 1}
    assert reduce_report["unsupported_nodes"][0]["reason_code"] == "requires_constant_input"
    assert reduce_report["unsupported_nodes"][0]["onnx_op"] == "ReduceSum"


def test_op_coverage_dynamic_quantize_linear_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_dynamic_quantize_linear_model(),
        output_file_name="dynamic_quantize_linear_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "DynamicQuantizeLinear"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_shape_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_shape_model(),
        output_file_name="shape_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "Shape"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_constant_of_shape_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_constant_of_shape_model(),
        output_file_name="constant_of_shape_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "ConstantOfShape"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_fused_matmul_builtin_dispatch() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_fused_matmul_model(),
        output_file_name="fused_matmul_builtin_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["supported_nodes"] == 1
    assert report["graph_node_reports"][0]["onnx_op"] == "FusedMatMul"
    assert report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_axis_rank_inference_avoids_custom_fallback_on_unknown_shapes() -> None:
    report = build_op_coverage_report(
        onnx_graph=_make_qlinear_concat_slice_softmax_unknown_rank_model(),
        output_file_name="axis_rank_inference_snapshot",
    )
    assert report["unsupported_reason_counts"] == {}
    assert report["graph_summary"]["unsupported_nodes"] == 0
    assert report["graph_summary"]["custom_lowered_nodes"] == 0
    assert report["graph_custom_ops"] == []
    assert all(
        node.get("dispatch_mode") == "builtin"
        for node in report["graph_node_reports"]
    )


def test_op_coverage_reason_code_snapshot_custom_policy_paths() -> None:
    default_report = build_op_coverage_report(
        onnx_graph=_make_einsum_custom_candidate_model(),
        output_file_name="einsum_custom_disabled_snapshot",
    )
    assert default_report["unsupported_reason_counts"] == {"custom_op_candidate_disabled": 1}
    assert default_report["unsupported_nodes"][0]["reason_code"] == "custom_op_candidate_disabled"

    allowlist_block_report = build_op_coverage_report(
        onnx_graph=_make_einsum_custom_candidate_model(),
        output_file_name="einsum_custom_allowlist_block_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["TopK"],
    )
    assert allowlist_block_report["unsupported_reason_counts"] == {"custom_op_not_in_allowlist": 1}
    assert allowlist_block_report["unsupported_nodes"][0]["reason_code"] == "custom_op_not_in_allowlist"

    allowlist_custom_report = build_op_coverage_report(
        onnx_graph=_make_einsum_custom_candidate_model(),
        output_file_name="einsum_custom_allowed_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["Einsum"],
    )
    assert allowlist_custom_report["unsupported_reason_counts"] == {}
    assert allowlist_custom_report["graph_summary"]["custom_lowered_nodes"] == 1
    assert allowlist_custom_report["graph_custom_ops"] == ["Einsum"]

    builtin_report = build_op_coverage_report(
        onnx_graph=_make_einsum_nonconst_rhs_model(),
        output_file_name="einsum_builtin_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["Einsum"],
    )
    assert builtin_report["unsupported_reason_counts"] == {}
    assert builtin_report["graph_summary"]["custom_lowered_nodes"] == 0
    assert builtin_report["graph_custom_ops"] == []
    assert any(
        node["onnx_op"] == "Einsum" and node.get("dispatch_mode") == "builtin"
        for node in builtin_report["graph_node_reports"]
    )


def test_op_coverage_global_average_pool_and_grouped_conv_builtin_dispatch() -> None:
    global_avg_report = build_op_coverage_report(
        onnx_graph=_make_global_average_pool_model(),
        output_file_name="global_average_pool_builtin_snapshot",
    )
    assert global_avg_report["unsupported_reason_counts"] == {}
    assert global_avg_report["graph_summary"]["unsupported_nodes"] == 0
    assert global_avg_report["graph_summary"]["custom_lowered_nodes"] == 0
    assert global_avg_report["graph_custom_ops"] == []
    assert global_avg_report["graph_node_reports"][0]["onnx_op"] == "GlobalAveragePool"
    assert global_avg_report["graph_node_reports"][0]["dispatch_mode"] == "builtin"

    grouped_conv_default_report = build_op_coverage_report(
        onnx_graph=_make_grouped_conv_model(),
        output_file_name="grouped_conv_default_snapshot",
    )
    assert grouped_conv_default_report["unsupported_reason_counts"] == {}
    assert grouped_conv_default_report["graph_summary"]["unsupported_nodes"] == 0

    grouped_conv_report = build_op_coverage_report(
        onnx_graph=_make_grouped_conv_model(),
        output_file_name="grouped_conv_builtin_snapshot",
        disable_group_convolution=True,
    )
    assert grouped_conv_report["unsupported_reason_counts"] == {}
    assert grouped_conv_report["graph_summary"]["unsupported_nodes"] == 0
    assert grouped_conv_report["graph_summary"]["custom_lowered_nodes"] == 0
    assert grouped_conv_report["graph_custom_ops"] == []
    assert grouped_conv_report["graph_node_reports"][0]["onnx_op"] == "Conv"
    assert grouped_conv_report["graph_node_reports"][0]["dispatch_mode"] == "builtin"


def test_op_coverage_erf_tile_scatternd_builtin_dispatch() -> None:
    cases = [
        (_make_erf_model(), "Erf"),
        (_make_tile_model(), "Tile"),
        (_make_scatter_nd_model(), "ScatterND"),
    ]
    for model, op_name in cases:
        report = build_op_coverage_report(
            onnx_graph=model,
            output_file_name=f"{op_name.lower()}_builtin_snapshot",
        )
        assert report["unsupported_reason_counts"] == {}
        assert report["graph_summary"]["unsupported_nodes"] == 0
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert report["graph_custom_ops"] == []
        assert any(
            node["onnx_op"] == op_name and node.get("dispatch_mode") == "builtin"
            for node in report["graph_node_reports"]
        )
