from __future__ import annotations

import ast
from pathlib import Path

import pytest
from onnx import helper

from onnx2tf.tflite_builder.core.session import ConversionSession
from onnx2tf.tflite_builder.ir import ModelIR


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_core_cleanup_dynamic_reshape_stats",
    "_no_layout_safe_transpose_reduction_stats",
    "_very_late_broadcast_static_shape_stats",
    "_late_binary_repair_static_shape_stats",
    "_late_binary_layout_recovery_static_shape_stats",
    "_terminal_expand_squeeze_static_shape_stats",
    "_very_late_static_shape_stats",
    "_fallback_norm_static_shape_stats",
    "_fallback_dynamic_rank1_topology_layout_stats",
    "_fallback_broadcast_static_shape_stats",
    "_fallback_broadcast_topology_layout_stats",
    "_fallback_se_fc_gather_static_shape_stats",
    "_fallback_placeholder_matmul_static_shape_stats",
    "_fallback_post_placeholder_topology_stats",
    "_fallback_conv_input_static_shape_stats",
    "_fallback_mixed_concat_static_shape_stats",
    "_fallback_concat_axis_static_shape_stats",
    "_fallback_binary_layout_static_shape_stats",
    "_fallback_post_layout_repair_topology_stats",
    "_fallback_high_rank_bmm_static_shape_stats",
    "_fallback_topology_layout_validation_stats",
    "_primary_post_lowering_topology_stats",
    "_no_layout_post_reduction_topology_stats",
    "_absolute_final_topology_layout_stats",
    "_final_convinteger_static_shape_stats",
    "_final_convinteger_topology_layout_stats",
    "_final_instancenorm_static_shape_stats",
    "_final_instancenorm_topology_layout_stats",
    "_final_broadcast_static_shape_stats",
    "_final_broadcast_topology_layout_stats",
    "_final_mixed_singleton_concat_static_shape_stats",
    "_final_placeholder_binary_static_shape_stats",
    "_final_placeholder_topology_stats",
    "_final_se_fc_gather_static_shape_stats",
    "_final_prelu_static_shape_stats",
    "_final_consecutive_reshape_static_shape_stats",
    "_final_sinet_late_residual_static_shape_stats",
    "_final_sinet_preadd_fanout_static_shape_stats",
    "_final_sinet_dual_resize_static_shape_stats",
    "_final_sinet_shared_post_static_shape_stats",
    "_final_sinet_deep_skip_static_shape_stats",
    "_final_sinet_concat_resize_static_shape_stats",
    "_final_high_rank_bmm_static_shape_stats",
    "_final_pad_layout_static_shape_stats",
    "_final_conv_input_static_shape_stats",
    "_final_mixed_concat_static_shape_stats",
    "_final_concat_axis_static_shape_stats",
    "_final_binary_layout_static_shape_stats",
    "_terminal_topology_layout_validation_stats",
)
EXPECTED_OWNERS = (
    "_resolve_dynamic_reshape_shapes",
    "_apply_safe_transpose_reduction_lite",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "run_static_shape_topology_reconciliation",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_topologically_sort_operators",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_topologically_sort_operators",
    "run_static_shape_topology_reconciliation",
    "run_topology_layout_validation",
    "_topologically_sort_operators",
    "_topologically_sort_operators",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_topologically_sort_operators",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_topology_layout_validation",
)
EXPECTED_MODEL_ARGUMENTS = (
    *("model_ir",) * 7,
    *("fallback_ir",) * 14,
    *("model_ir",) * 28,
)
EXPECTED_PHASE_IDS = (
    "shape_resolution.core.dynamic_reshape",
    "layout.no_layout.safe_transpose_reduction",
    "shape_reconciliation.primary.very_late_broadcast",
    "shape_reconciliation.primary.late_binary_repair",
    "shape_reconciliation.primary.late_binary_layout_recovery",
    "shape_reconciliation.terminal.expand_squeeze",
    "shape_reconciliation.primary.very_late_final",
    "shape_topology.fallback.norm",
    "topology_layout.fallback.post_dynamic_rank1",
    "shape_reconciliation.fallback.broadcast",
    "topology_layout.fallback.broadcast",
    "shape_reconciliation.fallback.se_fc_gather",
    "shape_reconciliation.fallback.placeholder_matmul",
    "topology.fallback.post_placeholder",
    "shape_reconciliation.fallback.conv_input",
    "shape_reconciliation.fallback.mixed_concat",
    "shape_reconciliation.fallback.concat_axis",
    "shape_reconciliation.fallback.binary_layout",
    "topology.fallback.post_layout_repair",
    "shape_topology.fallback.high_rank_batch_matmul",
    "layout_validation.fallback.terminal",
    "topology.primary.post_lowering",
    "topology.primary.no_layout_post_reduction",
    "topology_layout.primary.absolute_final",
    "shape_reconciliation.primary.final_convinteger",
    "topology_layout.primary.final_convinteger",
    "shape_reconciliation.primary.final_instancenorm",
    "topology_layout.primary.final_instancenorm",
    "shape_reconciliation.primary.final_broadcast",
    "topology_layout.primary.final_broadcast",
    "shape_reconciliation.primary.final_mixed_singleton_concat",
    "shape_reconciliation.primary.final_placeholder_binary",
    "topology.primary.final_placeholder",
    "shape_reconciliation.primary.final_se_fc_gather",
    "shape_reconciliation.primary.final_prelu",
    "shape_reconciliation.primary.final_consecutive_reshape",
    "shape_reconciliation.primary.final_sinet_late_residual",
    "shape_reconciliation.primary.final_sinet_preadd_fanout",
    "shape_reconciliation.primary.final_sinet_dual_resize",
    "shape_reconciliation.primary.final_sinet_shared_post",
    "shape_reconciliation.primary.final_sinet_deep_skip",
    "shape_reconciliation.primary.final_sinet_concat_resize",
    "shape_topology.primary.final_high_rank_batch_matmul",
    "shape_topology.primary.final_pad_layout",
    "shape_topology.primary.final_conv_input",
    "shape_topology.primary.final_mixed_concat",
    "shape_topology.primary.final_concat_axis",
    "shape_topology.primary.final_binary_layout",
    "layout_validation.primary.terminal",
)


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _is_phase_result_record(statement: ast.stmt) -> bool:
    call = _statement_call(statement)
    return bool(
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
    )


def _session() -> ConversionSession:
    graph = helper.make_graph([], "phase_results", [], [])
    return ConversionSession(
        onnx_model=helper.make_model(graph),
        model_ir=ModelIR("phase_results"),
        shape_map={},
        dtype_map={},
        constants={},
    )


def test_forty_nine_observations_use_the_bounded_session_store() -> None:
    lowerer = _lowerer()
    records = sorted(
        [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr) and _is_phase_result_record(node)
        ],
        key=lambda node: node.lineno,
    )

    assert len(records) == 49
    assert tuple(
        ast.literal_eval(_statement_call(node).args[0]) for node in records
    ) == EXPECTED_PHASE_IDS
    nested_calls = tuple(_statement_call(node).args[1] for node in records)
    assert all(isinstance(call, ast.Call) for call in nested_calls)
    assert tuple(
        call.func.id
        for call in nested_calls
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ) == EXPECTED_OWNERS
    assert tuple(
        ast.unparse(call.args[0])
        for call in nested_calls
        if isinstance(call, ast.Call)
    ) == EXPECTED_MODEL_ARGUMENTS
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_phase_result_store_is_bounded_integer_only_and_snapshot_isolated() -> None:
    session = _session()
    source = {"changed": 1, "cycle_detected": 0}

    assert session.phase_results_snapshot() == {}
    session.record_phase_result("topology.primary", source)
    source["changed"] = 99
    assert session.phase_results_snapshot() == {
        "topology.primary": {"changed": 1, "cycle_detected": 0}
    }

    snapshot = session.phase_results_snapshot()
    snapshot["topology.primary"]["changed"] = 77
    assert session.phase_results_snapshot()["topology.primary"]["changed"] == 1

    with pytest.raises(ValueError):
        session.record_phase_result("", {"changed": 1})
    with pytest.raises(ValueError):
        session.record_phase_result(
            "too_many_counters",
            {f"counter_{index}": index for index in range(33)},
        )
    with pytest.raises(TypeError):
        session.record_phase_result("non_integer", {"details": []})

    for index in range(1, 128):
        session.record_phase_result(f"phase_{index}", {"changed": index})
    with pytest.raises(ValueError):
        session.record_phase_result("phase_128", {"changed": 128})
