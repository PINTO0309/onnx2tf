from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_transpose.py"
)
LATE_BINARY_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_binary_layout_recovery.py"
)
LATE_CONCAT_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_concat_layout_orchestration.py"
)
VERY_LATE_LAYOUT_BROADCAST_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_layout_broadcast_orchestration.py"
)
OWNER = "run_layout_transpose_cleanup"
INNER_OWNER = "_optimize_layout_transpose_chains"
RESULT_TARGET = "_layout_pass_set_1_layout_transpose_cleanup_stats"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


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


def _direct_lowerer_calls(lowerer: ast.FunctionDef) -> list[ast.stmt]:
    return [
        statement
        for root in lowerer.body
        for statement in ast.walk(root)
        if isinstance(statement, (ast.Assign, ast.Expr))
        and _call_name(statement) == OWNER
    ]


def test_layout_transpose_schema_and_all_owner_occurrences_are_explicit() -> None:
    functions = _functions(OWNER_PATH)
    owner = functions[OWNER]
    default_details = next(
        statement
        for statement in owner.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "default_details"
            for target in statement.targets
        )
    )
    assert isinstance(default_details.value, ast.Dict)
    assert [ast.literal_eval(key) for key in default_details.value.keys] == [
        "iterations",
        "removed_identity_transpose",
        "removed_inverse_transpose_pairs",
        "removed_inverse_transpose_fanout_branches",
        "composed_consecutive_transpose_pairs",
    ]
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{str(key): int(value) for key, value in details.items()}"
    )

    inner_owner = functions[INNER_OWNER]
    inner_source = ast.get_source_segment(
        OWNER_PATH.read_text(encoding="utf-8"), inner_owner
    )
    assert inner_source is not None
    assert "_prune_unused_tensors(model_ir, layout_state=layout_state)" in (
        inner_source
    )
    assert "layout_state.sync_from_model_ir(model_ir)" in inner_source

    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_calls = _direct_lowerer_calls(lowerer)
    very_late_owner = _functions(VERY_LATE_LAYOUT_BROADCAST_PATH)[
        "run_very_late_layout_broadcast_cleanup"
    ]
    very_late_calls = [
        node
        for node in ast.walk(very_late_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert len(direct_calls) + len(very_late_calls) == 2
    assert [_single_target(statement) for statement in direct_calls] == [
        None,
    ]
    assert all(
        [ast.unparse(argument) for argument in _statement_call(statement).args]
        == ["model_ir"]
        for statement in direct_calls
        if _statement_call(statement) is not None
    )
    keyword_contracts = [
        {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        }
        for statement in direct_calls
        if (call := _statement_call(statement)) is not None
    ]
    assert keyword_contracts == [
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
        },
    ]
    very_late_call = very_late_calls[0]
    assert [ast.unparse(argument) for argument in very_late_call.args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in very_late_call.keywords
    } == {
        "layout_state": "context.layout_state",
        "diagnostics": "context.diagnostics",
    }

    late_concat = _functions(LATE_CONCAT_PATH)[
        "run_late_concat_layout_cleanup"
    ]
    late_concat_calls = [
        node
        for node in ast.walk(late_concat)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert len(late_concat_calls) == 1
    late_concat_call = late_concat_calls[0]
    assert [ast.unparse(argument) for argument in late_concat_call.args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in late_concat_call.keywords
    } == {
        "layout_state": "context.layout_state",
        "diagnostics": "context.diagnostics",
        "state_scope": "state_scope",
    }

    late_binary = _functions(LATE_BINARY_PATH)["run_late_binary_layout_recovery"]
    nested_calls = [
        node
        for node in ast.walk(late_binary)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert len(nested_calls) == 1
    nested_call = nested_calls[0]
    assert [ast.unparse(argument) for argument in nested_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in nested_call.keywords
    } == {
        "layout_state": "layout_state",
        "diagnostics": "diagnostics",
    }


def test_primary_layout_transpose_result_is_retained_observation_only() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    layout_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(_call_name(child) == OWNER for child in statement.body)
    )
    result_index = next(
        index
        for index, statement in enumerate(layout_guard.body)
        if _call_name(statement) == OWNER
    )
    result = layout_guard.body[result_index]
    assert _single_target(result) is None
    call = _statement_call(result)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }
    assert _single_target(layout_guard.body[result_index - 1]) == (
        "enable_duplicate_transpose_fanout_optimizations"
    )
    assert _single_target(layout_guard.body[result_index + 1]) == (
        "_layout_pass_set_1_initial_attention_recovery_results"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
