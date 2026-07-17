from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"


def _lowerer_body() -> list[ast.stmt]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    return lowerer.body


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if isinstance(statement, (ast.Assign, ast.Expr)) and isinstance(
        statement.value,
        ast.Call,
    ):
        return statement.value
    return None


def _call_name(call: ast.Call | None) -> str | None:
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _call_index(body: list[ast.stmt], function_name: str) -> int:
    return next(
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == function_name
    )


@pytest.mark.xfail(
    strict=True,
    reason="primary layout validation precedes terminal graph mutations",
)
def test_primary_path_validates_terminal_layout_and_clears_stale_errors() -> None:
    body = _lowerer_body()
    convergence_index = _call_index(
        body,
        "_run_indexed_binary_layout_convergence",
    )
    coalesce_index = _call_index(
        body,
        "coalesce_static_high_rank_binary_operators",
    )
    realign_index = _call_index(
        body,
        "_realign_dynamic_boundary_shape_signature_map",
    )
    terminal_sort_index = next(
        index
        for index in range(realign_index + 1, len(body))
        if _call_name(_statement_call(body[index]))
        == "_topologically_sort_operators"
    )
    validation_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "layout_problems"
    )

    assert convergence_index < coalesce_index < realign_index < terminal_sort_index
    assert validation_index == terminal_sort_index + 1
    validation = body[validation_index]
    assert isinstance(validation, ast.Assign)
    assert ast.unparse(validation.value) == (
        "validate_model_ir_layout_annotations(model_ir)"
    )

    guard = body[validation_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "len(layout_problems) > 0"
    assert len(guard.body) == 1
    assert ast.unparse(guard.body[0]) == (
        "model_ir.metadata['logical_layout_validation_errors'] = "
        "list(layout_problems)"
    )
    assert len(guard.orelse) == 1
    assert ast.unparse(guard.orelse[0]) == (
        "model_ir.metadata.pop('logical_layout_validation_errors', None)"
    )

    terminal = body[validation_index + 2]
    assert isinstance(terminal, ast.Return)
    assert ast.unparse(terminal.value) == "_finalize_model_ir(model_ir)"
