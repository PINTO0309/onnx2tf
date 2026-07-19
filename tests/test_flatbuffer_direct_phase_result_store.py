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
    "_fallback_post_placeholder_topology_stats",
    "_fallback_post_layout_repair_topology_stats",
    "_fallback_topology_layout_validation_stats",
    "_primary_post_lowering_topology_stats",
    "_no_layout_post_reduction_topology_stats",
    "_final_placeholder_topology_stats",
    "_terminal_topology_layout_validation_stats",
)
EXPECTED_OWNERS = (
    "_topologically_sort_operators",
    "_topologically_sort_operators",
    "run_topology_layout_validation",
    "_topologically_sort_operators",
    "_topologically_sort_operators",
    "_topologically_sort_operators",
    "run_topology_layout_validation",
)
EXPECTED_MODEL_ARGUMENTS = (
    "fallback_ir",
    "fallback_ir",
    "fallback_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
)


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _session() -> ConversionSession:
    graph = helper.make_graph([], "phase_results", [], [])
    return ConversionSession(
        onnx_model=helper.make_model(graph),
        model_ir=ModelIR("phase_results"),
        shape_map={},
        dtype_map={},
        constants={},
    )


def test_seven_topology_observation_locals_are_explicit_and_unconsumed() -> None:
    lowerer = _lowerer()
    assignments = sorted(
        [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Assign)
            and _single_target(node) in EXPECTED_RESULT_TARGETS
        ],
        key=lambda node: node.lineno,
    )

    assert len(assignments) == 7
    assert tuple(_single_target(node) for node in assignments) == (
        EXPECTED_RESULT_TARGETS
    )
    assert tuple(_call_name(node) for node in assignments) == EXPECTED_OWNERS
    assert tuple(
        ast.unparse(_statement_call(node).args[0]) for node in assignments
    ) == EXPECTED_MODEL_ARGUMENTS
    for target in EXPECTED_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="bounded ConversionSession phase-result store is not implemented",
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
