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


def _make_reduce_axes_nonconst_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    axes = helper.make_tensor_value_info("axes", TensorProto.INT64, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    node = helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="ReduceNode", keepdims=1)
    graph = helper.make_graph([node], "reduce_nonconst_axes_graph", [x, axes], [y])
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
    assert softmax_report["unsupported_reason_counts"] == {"unsupported_attribute_value": 1}
    assert softmax_report["unsupported_nodes"][0]["reason_code"] == "unsupported_attribute_value"
    assert softmax_report["unsupported_nodes"][0]["onnx_op"] == "Softmax"

    reduce_report = build_op_coverage_report(
        onnx_graph=_make_reduce_axes_nonconst_model(),
        output_file_name="reduce_reason_snapshot",
    )
    assert reduce_report["unsupported_reason_counts"] == {"requires_constant_input": 1}
    assert reduce_report["unsupported_nodes"][0]["reason_code"] == "requires_constant_input"
    assert reduce_report["unsupported_nodes"][0]["onnx_op"] == "ReduceSum"


def test_op_coverage_reason_code_snapshot_custom_policy_paths() -> None:
    default_report = build_op_coverage_report(
        onnx_graph=_make_einsum_nonconst_rhs_model(),
        output_file_name="einsum_custom_disabled_snapshot",
    )
    assert default_report["unsupported_reason_counts"] == {"custom_op_candidate_disabled": 1}
    assert default_report["unsupported_nodes"][0]["reason_code"] == "custom_op_candidate_disabled"

    allowlist_block_report = build_op_coverage_report(
        onnx_graph=_make_einsum_nonconst_rhs_model(),
        output_file_name="einsum_custom_allowlist_block_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["TopK"],
    )
    assert allowlist_block_report["unsupported_reason_counts"] == {"custom_op_not_in_allowlist": 1}
    assert allowlist_block_report["unsupported_nodes"][0]["reason_code"] == "custom_op_not_in_allowlist"

    allowlist_custom_report = build_op_coverage_report(
        onnx_graph=_make_einsum_nonconst_rhs_model(),
        output_file_name="einsum_custom_allowed_snapshot",
        allow_custom_ops=True,
        custom_op_allowlist=["Einsum"],
    )
    assert allowlist_custom_report["unsupported_reason_counts"] == {}
    assert allowlist_custom_report["graph_summary"]["custom_lowered_nodes"] == 1
    assert allowlist_custom_report["graph_custom_ops"] == ["Einsum"]

    builtin_report = build_op_coverage_report(
        onnx_graph=_make_einsum_const_rhs_model(),
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
