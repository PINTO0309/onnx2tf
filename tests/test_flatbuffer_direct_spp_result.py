from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "spp_layout.py"
ORCHESTRATION_PATHS = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_recovery_orchestration.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_layout_mean_spp_gather_constant_cast_orchestration.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_spp_concat_unary_conv_orchestration.py",
)
SPP = "run_spp_layout_cleanup"
INNER_OWNER = "_optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"
RESULT_TARGET = "_layout_opt_spp_stats"
PREVIOUS = "_optimize_transpose_elementwise_concat_conv_nhwc_groups"
FOLLOWING_TARGET = "_layout_opt_pre_concat_stats"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


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


def test_spp_schema_positive_cleanup_and_selections_are_explicit() -> None:
    functions = _functions(OWNER_PATH)
    owner = functions[INNER_OWNER]
    cleanup_guards = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "rewritten > 0"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(cleanup_guards) == 1
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == "{_STATS_KEY: int(rewritten)}"

    runner = functions[SPP]
    runner_return = runner.body[-1]
    assert isinstance(runner_return, ast.Return)
    assert ast.unparse(runner_return.value) == (
        "{str(key): int(value) for key, value in details.items()}"
    )

    selected = 0
    for path in ORCHESTRATION_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        path_selections = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == SPP
        ]
        assert len(path_selections) == 1
        selected += len(path_selections)
    assert selected == 3


@pytest.mark.xfail(
    strict=True,
    reason="the sole direct SPP layout-cleanup result is not retained yet",
)
def test_direct_spp_result_is_retained_observation_only() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    layout_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(_call_name(child) == SPP for child in statement.body)
    )
    result_index = next(
        index
        for index, statement in enumerate(layout_guard.body)
        if _call_name(statement) == SPP
    )
    result = layout_guard.body[result_index]
    assert _single_target(result) == RESULT_TARGET
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
    assert sum(
        1
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SPP
    ) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )

    assert _call_name(layout_guard.body[result_index - 1]) == PREVIOUS
    assert _single_target(layout_guard.body[result_index + 1]) == FOLLOWING_TARGET
