from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_reconcile_static_tensor_shapes"
EXPECTED_RESULT_TARGETS = (
    "_very_late_broadcast_static_shape_stats",
    "_very_late_static_shape_stats",
)
EXPECTED_PHASE_IDS = (
    "shape_reconciliation.primary.very_late_broadcast",
    "shape_reconciliation.primary.very_late_final",
)
EXPECTED_PREDECESSOR_TARGETS = (
    "_very_late_broadcast_repair_stats",
    "_very_late_dynamic_rank1_reshape_stats",
)
EXPECTED_SUCCESSOR_TARGETS = (
    "shared_late_tensor_count",
    "split_fallback_stats",
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


def _phase_result_owner(statement: ast.stmt) -> ast.Call | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
        or not isinstance(call.args[1], ast.Call)
    ):
        return None
    return call.args[1]


def test_two_very_late_reconciliations_are_unconditional_and_bounded() -> None:
    lowerer = _lowerer()
    indices = tuple(
        next(
            index
            for index, statement in enumerate(lowerer.body)
            if _single_target(statement) == target
        )
        for target in EXPECTED_RESULT_TARGETS
    )

    assert tuple(
        _single_target(lowerer.body[index - 1]) for index in indices
    ) == EXPECTED_PREDECESSOR_TARGETS
    assert tuple(
        _single_target(lowerer.body[index + 1]) for index in indices
    ) == EXPECTED_SUCCESSOR_TARGETS
    for index in indices:
        observation = lowerer.body[index]
        call = _statement_call(observation)
        assert call is not None
        assert isinstance(call.func, ast.Name)
        assert call.func.id == OWNER
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"include_mutation_count": "True"}
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


@pytest.mark.xfail(
    strict=True,
    reason="unconditional very-late reconciliations have not moved to phase results",
)
def test_two_very_late_reconciliations_use_phase_results() -> None:
    lowerer = _lowerer()
    records = sorted(
        (
            node
            for node in lowerer.body
            if isinstance(node, ast.Expr)
            and (owner := _phase_result_owner(node)) is not None
            and isinstance(owner.func, ast.Name)
            and owner.func.id == OWNER
            and ast.literal_eval(_statement_call(node).args[0])
            in EXPECTED_PHASE_IDS
        ),
        key=lambda node: node.lineno,
    )

    assert tuple(
        ast.literal_eval(_statement_call(node).args[0]) for node in records
    ) == EXPECTED_PHASE_IDS
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
