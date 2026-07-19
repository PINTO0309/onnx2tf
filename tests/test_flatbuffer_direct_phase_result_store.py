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
    "_fallback_dynamic_rank1_topology_layout_stats",
    "_fallback_broadcast_topology_layout_stats",
    "_fallback_post_placeholder_topology_stats",
    "_fallback_post_layout_repair_topology_stats",
    "_fallback_topology_layout_validation_stats",
    "_primary_post_lowering_topology_stats",
    "_no_layout_post_reduction_topology_stats",
    "_absolute_final_topology_layout_stats",
    "_final_convinteger_topology_layout_stats",
    "_final_instancenorm_topology_layout_stats",
    "_final_broadcast_topology_layout_stats",
    "_final_placeholder_topology_stats",
    "_terminal_topology_layout_validation_stats",
)
EXPECTED_OWNERS = (
    "run_topology_layout_refresh",
    "run_topology_layout_refresh",
    "_topologically_sort_operators",
    "_topologically_sort_operators",
    "run_topology_layout_validation",
    "_topologically_sort_operators",
    "_topologically_sort_operators",
    "run_topology_layout_refresh",
    "run_topology_layout_refresh",
    "run_topology_layout_refresh",
    "run_topology_layout_refresh",
    "_topologically_sort_operators",
    "run_topology_layout_validation",
)
EXPECTED_MODEL_ARGUMENTS = (
    "fallback_ir",
    "fallback_ir",
    "fallback_ir",
    "fallback_ir",
    "fallback_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
)
EXPECTED_PHASE_IDS = (
    "topology_layout.fallback.post_dynamic_rank1",
    "topology_layout.fallback.broadcast",
    "topology.fallback.post_placeholder",
    "topology.fallback.post_layout_repair",
    "layout_validation.fallback.terminal",
    "topology.primary.post_lowering",
    "topology.primary.no_layout_post_reduction",
    "topology_layout.primary.absolute_final",
    "topology_layout.primary.final_convinteger",
    "topology_layout.primary.final_instancenorm",
    "topology_layout.primary.final_broadcast",
    "topology.primary.final_placeholder",
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


def test_thirteen_topology_observations_use_the_bounded_session_store() -> None:
    lowerer = _lowerer()
    records = sorted(
        [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr) and _is_phase_result_record(node)
        ],
        key=lambda node: node.lineno,
    )

    assert len(records) == 13
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
