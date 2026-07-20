from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_duplicate_fanout_cleanup,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "graph_cleanup.py"
)
ORCHESTRATION_PATHS = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "duplicate_quantized_prelu_orchestration.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "singleton_consecutive_reshape_orchestration.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "singleton_reshape_orchestration.py",
)
OWNER = "run_duplicate_fanout_cleanup"
RESULT_TARGET = "_layout_pass_set_1_duplicate_fanout_stats"


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


def test_duplicate_fanout_policy_schema_and_selections_are_explicit() -> None:
    reshape_only = run_duplicate_fanout_cleanup(
        ModelIR("duplicate_fanout_reshape_only_schema"),
        include_transpose=False,
    )
    full = run_duplicate_fanout_cleanup(
        ModelIR("duplicate_fanout_full_schema"),
        include_transpose=True,
    )
    assert reshape_only == {"removed_duplicate_reshape_fanout": 0}
    assert full == {
        "removed_duplicate_reshape_fanout": 0,
        "removed_duplicate_transpose_fanout": 0,
    }

    owner = _functions(OWNER_PATH)[OWNER]
    owner_source = ast.get_source_segment(
        OWNER_PATH.read_text(encoding="utf-8"), owner
    )
    assert owner_source is not None
    assert 'pass_id="cleanup.duplicate_transpose_fanout"' in owner_source
    assert 'pass_id="cleanup.duplicate_reshape_fanout"' in owner_source
    assert owner_source.count("transactional=True") == 2
    assert 'if include_transpose:' in owner_source
    assert 'default_details["removed_duplicate_transpose_fanout"] = 0' in (
        owner_source
    )

    for path in ORCHESTRATION_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selections = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == OWNER
        ]
        assert len(selections) == 1


def test_primary_duplicate_fanout_result_is_retained_observation_only() -> None:
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
        "include_transpose": "enable_duplicate_transpose_fanout_optimizations",
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }

    policy = next(
        statement
        for statement in layout_guard.body
        if _single_target(statement)
        == "enable_duplicate_transpose_fanout_optimizations"
    )
    assert ast.unparse(policy.value) == "not has_qdq_ops"
    predecessor = layout_guard.body[result_index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == (
        "enable_transpose_binary_bridge_optimizations"
    )
    assert _single_target(layout_guard.body[result_index + 1]) == (
        "_layout_pass_set_1_post_binary_attention_recovery_results"
    )
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(lowerer)
    ) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
