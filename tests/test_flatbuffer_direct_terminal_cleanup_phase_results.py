from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_terminal_cleanup_terminal_dequant_stats",
    "_terminal_cleanup_terminal_qdq_stats",
    "_terminal_cleanup_conv_affine_stats",
    "_terminal_cleanup_conv_activation_stats",
)
EXPECTED_PHASE_IDS = (
    "cleanup.terminal.dequant",
    "cleanup.terminal.qdq",
    "cleanup.terminal.conv_affine",
    "cleanup.terminal.conv_activation",
)
EXPECTED_OWNER_EXPRESSIONS = (
    "_sanitize_terminal_transpose_before_dequantize(model_ir)",
    (
        "run_terminal_quantize_dequantize_cleanup(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    (
        "_optimize_fold_conv_mul_add_affine_chains(model_ir, "
        "enable_conv_add_only_fold=True, layout_state=session.layout_state)"
    ),
    (
        "_optimize_fuse_conv_activation_chains(model_ir, "
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


def _terminal_records(lowerer: ast.FunctionDef) -> list[ast.Expr]:
    return [
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Expr)
        and _phase_id(statement) in EXPECTED_PHASE_IDS
    ]


def test_terminal_cleanup_results_are_unconditional_and_unconsumed() -> None:
    lowerer = _lowerer()
    records = _terminal_records(lowerer)
    indices = [lowerer.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == EXPECTED_PHASE_IDS
    assert tuple(
        ast.unparse(_statement_call(statement).args[1]) for statement in records
    ) == EXPECTED_OWNER_EXPRESSIONS
    assert ast.unparse(lowerer.body[indices[0] - 1]) == (
        "_set_post_progress_desc('terminal cleanup passes')"
    )
    assert _single_target(lowerer.body[indices[-1] + 1]) == (
        "_terminal_pre_argmax_stats"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


def test_terminal_cleanup_result_locals_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
