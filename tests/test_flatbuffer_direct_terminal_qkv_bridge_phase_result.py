from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGET = "_terminal_qkv_split_conv_concat_bridge_stats"
EXPECTED_PHASE_ID = "cleanup.terminal.qkv_split_conv_concat_bridge"
EXPECTED_OWNER_EXPRESSION = (
    "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw("
    "model_ir, layout_state=session.layout_state)"
)
PREDECESSOR_TARGET = "_terminal_qkv_attention_results"
SUCCESSOR_TARGET = "_terminal_singleton_reshape_results"


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


def _terminal_layout_guard(lowerer: ast.FunctionDef) -> ast.If:
    guards = [
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            _single_target(child) == PREDECESSOR_TARGET for child in statement.body
        )
        and any(
            _single_target(child) == SUCCESSOR_TARGET for child in statement.body
        )
    ]
    assert len(guards) == 1
    return guards[0]


def test_terminal_qkv_bridge_result_uses_phase_result_store() -> None:
    lowerer = _lowerer()
    guard = _terminal_layout_guard(lowerer)
    records = [
        statement
        for statement in guard.body
        if _phase_id(statement) == EXPECTED_PHASE_ID
    ]
    assert len(records) == 1
    statement = records[0]
    index = guard.body.index(statement)
    call = _statement_call(statement)

    assert call is not None
    assert ast.unparse(call.args[1]) == EXPECTED_OWNER_EXPRESSION
    assert _single_target(guard.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(guard.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id == EXPECTED_RESULT_TARGET
        for node in ast.walk(lowerer)
    )
