from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_layout_pass_set_2_squeeze_reshape_identity_stats",
    "_layout_pass_set_2_prune_reconcile_stats",
)
EXPECTED_PHASE_IDS = (
    "cleanup.layout_pass_set_2.squeeze_reshape_identity",
    "cleanup.layout_pass_set_2.prune_reconcile",
)
EXPECTED_OWNER_EXPRESSIONS = (
    (
        "run_squeeze_reshape_identity_cleanup(model_ir, "
        "include_unary_passthrough=True, layout_state=session.layout_state, "
        "diagnostics=session.diagnostics)"
    ),
    (
        "run_indexed_prune_reconcile_cleanup(model_ir, "
        "layout_state=session.layout_state)"
    ),
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


def _phase_id(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        node
        for node in lowerer.body
        if isinstance(node, ast.If)
        and any(
            _single_target(statement) in EXPECTED_RESULT_TARGETS
            or _phase_id(statement) in EXPECTED_PHASE_IDS
            for statement in node.body
        )
    )


def test_layout_pass_set_2_cleanup_results_are_guarded_and_unconsumed() -> None:
    lowerer = _lowerer()
    guard = _guard(lowerer)
    records = [
        statement
        for statement in guard.body
        if isinstance(statement, ast.Expr)
        and _phase_id(statement) in EXPECTED_PHASE_IDS
    ]
    indices = [guard.body.index(statement) for statement in records]

    assert ast.unparse(guard.test) == "optimize_layout_transpose_chains"
    assert guard.orelse == []
    assert tuple(_phase_id(statement) for statement in records) == EXPECTED_PHASE_IDS
    assert tuple(
        ast.unparse(_statement_call(statement).args[1]) for statement in records
    ) == EXPECTED_OWNER_EXPRESSIONS
    convergence = guard.body[indices[0] - 1]
    assert isinstance(convergence, ast.For)
    assert ast.unparse(convergence.target) == "_"
    assert ast.unparse(convergence.iter) == "range(2)"
    assert "rewritten_instnorm" in ast.unparse(convergence)
    assert ast.unparse(guard.body[indices[-1] + 1]) == "_advance_post_progress()"
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


def test_layout_pass_set_2_cleanup_result_locals_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
