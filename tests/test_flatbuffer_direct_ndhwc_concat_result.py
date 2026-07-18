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
    / "ndhwc_concat_layout.py"
)
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_recovery_orchestration.py"
)
NDHWC_CONCAT = "run_ndhwc_concat_layout_cleanup"
INNER_OWNER = "_optimize_transpose_pre_concat_ndhwc_chains"
RESULT_TARGET = "_layout_opt_ndhwc_concat_stats"
PREVIOUS_TARGET = "_layout_opt_pre_concat_stats"
FOLLOWING = "_optimize_transpose_stridedslice_pre_concat_nhwc_chains"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _direct_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    function = statement.value.func
    if not isinstance(function, ast.Name) or function.id != NDHWC_CONCAT:
        return None
    return statement.value


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _containing_body(root: ast.AST, target: ast.stmt) -> list[ast.stmt]:
    for node in ast.walk(root):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and target in value:
                return value
    raise AssertionError("statement is not contained by an AST body")


def test_ndhwc_concat_schema_and_positive_cleanup_are_explicit() -> None:
    functions = _functions(OWNER_PATH)
    inner_owner = functions[INNER_OWNER]
    cleanup_guards = [
        statement
        for statement in inner_owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimized > 0"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(cleanup_guards) == 1
    inner_return = inner_owner.body[-1]
    assert isinstance(inner_return, ast.Return)
    assert ast.unparse(inner_return.value) == (
        "{_STATS_KEY: int(optimized)}"
    )

    runner = functions[NDHWC_CONCAT]
    runner_return = runner.body[-1]
    assert isinstance(runner_return, ast.Return)
    assert ast.unparse(runner_return.value) == (
        "{str(key): int(value) for key, value in details.items()}"
    )
    assert any(
        isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Name) and key.id == "_STATS_KEY"
            for key in node.keys
        )
        for node in ast.walk(runner)
    )


def test_direct_ndhwc_concat_result_is_retained_observation_only() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_results = sorted(
        (
            statement
            for statement in ast.walk(lowerer)
            if isinstance(statement, (ast.Assign, ast.Expr))
            and _direct_call(statement) is not None
        ),
        key=lambda statement: statement.lineno,
    )
    assert len(direct_results) == 1
    result = direct_results[0]
    assert _single_target(result) == RESULT_TARGET
    call = _direct_call(result)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )

    body = _containing_body(lowerer, result)
    result_index = body.index(result)
    assert _single_target(body[result_index - 1]) == PREVIOUS_TARGET
    following = body[result_index + 1]
    assert isinstance(following, ast.Expr)
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == FOLLOWING


def test_layout_recovery_keeps_independent_ndhwc_concat_selection() -> None:
    orchestration = _functions(ORCHESTRATION_PATH)[
        "build_layout_recovery_invocations"
    ]
    selected = [
        node
        for node in ast.walk(orchestration)
        if isinstance(node, ast.Name) and node.id == NDHWC_CONCAT
    ]
    assert len(selected) == 1
