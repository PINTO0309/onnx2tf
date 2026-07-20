from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import late_window_layout_orchestration
from onnx2tf.tflite_builder.passes.late_window_layout_orchestration import (
    LATE_WINDOW_LAYOUT_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_window_layout_orchestration.py"
)
OWNER = "run_late_window_layout_cleanup"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_reshape_shuffle_attention_window_orchestration.py"
)
COMPOSITE_OWNER = "run_late_reshape_shuffle_attention_window_cleanup"
COMPOSITE_TARGET = "_late_final_shape_boundary_results"
RESULT_TARGET = "_late_window_layout_results"
PREDECESSOR_TARGET = "_late_affine_optional_fanout_results"
SUCCESSOR_TARGET = "_terminal_fanout_singleton_results"
OLD_RESULT_TARGETS = (
    "_late_window_partition_stats",
    "_late_window_reverse_stats",
)
PASS_IDS = (
    "_optimize_window_partition_reshape_transpose_to_space_to_depth_chains",
    "_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


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


def _composite_calls() -> list[ast.Call]:
    owner = _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]


def test_late_window_layout_pair_uses_one_composite_result_outside_store() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_final_shape_boundary_cleanup("
        "late_final_shape_boundary_context)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert _single_target(predecessor) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.Assign)
    assert _single_target(successor) == SUCCESSOR_TARGET
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_phase_result"
        and any(
            isinstance(child, ast.Name) and child.id == OWNER
            for child in ast.walk(node)
        )
        for node in ast.walk(lowerer)
    )


def test_late_window_layout_pair_uses_one_composite_owner() -> None:
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PASS_IDS
    ]
    assert owner_calls == list(PASS_IDS)

    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_final_shape_boundary_cleanup("
        "late_final_shape_boundary_context)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert _single_target(predecessor) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.Assign)
    assert _single_target(successor) == SUCCESSOR_TARGET
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_late_window_layout_owner_preserves_arguments_and_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_window_layout_owner")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[str, object, object]] = []
    expected_results = (
        {"window_partition": 1},
        {"window_reverse": 2},
    )

    def _callback(name: str, result: dict[str, int]):
        def _run(
            candidate: ModelIR,
            *,
            layout_state: object,
        ) -> dict[str, int]:
            observed.append((name, candidate, layout_state))
            return dict(result)

        return _run

    for pass_id, result in zip(PASS_IDS, expected_results, strict=True):
        monkeypatch.setattr(
            late_window_layout_orchestration,
            pass_id,
            _callback(pass_id, result),
        )

    assert late_window_layout_orchestration.run_late_window_layout_cleanup(
        context
    ) == expected_results
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert all(entry[2] is context.layout_state for entry in observed)
    assert LATE_WINDOW_LAYOUT_PASS_IDS == PASS_IDS
