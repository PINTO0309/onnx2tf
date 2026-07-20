from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.binary_bridge_layout import (
    optimize_transpose_binary_bridges,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_optimize_transpose_binary_bridges"
RESULT_TARGET = "_layout_pass_set_1_transpose_binary_bridge_stats"


def _lowerer() -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(LOWERER_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    call = statement.value if isinstance(statement.value, ast.Call) else None
    if (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        return call.args[1]
    return call


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _layout_and_binary_guards() -> tuple[ast.If, int, ast.If]:
    lowerer = _lowerer()
    for statement in lowerer.body:
        if not (
            isinstance(statement, ast.If)
            and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        ):
            continue
        for index, candidate in enumerate(statement.body):
            if (
                isinstance(candidate, ast.If)
                and ast.unparse(candidate.test)
                == "enable_transpose_binary_bridge_optimizations"
                and any(_call_name(child) == OWNER for child in candidate.body)
            ):
                return statement, index, candidate
    raise AssertionError("binary bridge guard not found")


def test_binary_bridge_schema_guard_and_call_are_explicit() -> None:
    assert optimize_transpose_binary_bridges(ModelIR("binary_bridge_schema")) == {
        "removed_transpose_binary_bridges": 0,
        "removed_transpose_binary_asymmetric_bridges": 0,
    }

    lowerer = _lowerer()
    layout_guard, guard_index, binary_guard = _layout_and_binary_guards()
    assert len(binary_guard.body) == 1
    assert binary_guard.orelse == []
    call = _statement_call(binary_guard.body[0])
    assert call is not None
    assert isinstance(call.func, ast.Name)
    assert call.func.id == OWNER
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"layout_state": "session.layout_state"}
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(lowerer)
    ) == 1
    assert _single_target(layout_guard.body[guard_index - 1]) == (
        "_layout_pass_set_1_quantized_activation_binary_results"
    )
    assert _call_name(layout_guard.body[guard_index + 1]) == (
        "run_duplicate_fanout_cleanup"
    )


def test_guarded_binary_bridge_result_is_retained_observation_only() -> None:
    lowerer = _lowerer()
    _, _, binary_guard = _layout_and_binary_guards()
    assert _single_target(binary_guard.body[0]) is None
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
