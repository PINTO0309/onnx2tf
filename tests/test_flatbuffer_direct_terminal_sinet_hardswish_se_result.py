from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "hardswish_se_layout.py"
)
HARDSWISH_SE = (
    "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
)
OWNER_NAME = (
    "optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
)
SINET_TERMINAL_TARGET = "_terminal_sinet_layout_recovery_results"
RESULT_TARGET = "_terminal_sinet_hardswish_se_stats"
DEQUANT_TARGET = "_terminal_dequant_hardsigmoid_bridge_stats"
LATE_RESULT_TARGET = "_terminal_hardswish_se_stats"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _direct_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    value = statement.value
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return None
    return value if value.func.id == HARDSWISH_SE else None


def test_hardswish_se_wrapper_schema_and_cleanup_are_explicit() -> None:
    wrapper = _functions(LOWERER_PATH)[HARDSWISH_SE]
    assert len(wrapper.body) == 1
    wrapper_return = wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{HARDSWISH_SE}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir"
    ]
    assert wrapper_return.value.keywords == []

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    top_level_cleanup = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_prune_unused_tensors"
    ]
    assert len(top_level_cleanup) == 1
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains': "
        "int(rewritten)}"
    )


def test_hardswish_se_has_exactly_two_production_forms() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    owner_statements = [
        statement
        for statement in lowerer.body
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == HARDSWISH_SE
            for node in ast.walk(statement)
        )
    ]
    assert len(owner_statements) == 2

    first_call = next(
        node
        for node in ast.walk(owner_statements[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == HARDSWISH_SE
    )
    assert [ast.unparse(argument) for argument in first_call.args] == ["model_ir"]
    assert first_call.keywords == []

    late_statement = owner_statements[1]
    assert _single_target(late_statement) == LATE_RESULT_TARGET
    assert isinstance(late_statement, ast.Assign)
    assert isinstance(late_statement.value, ast.Dict)
    assert len(late_statement.value.keys) == 2
    assert late_statement.value.keys[0] is None
    prune_key = late_statement.value.keys[1]
    assert isinstance(prune_key, ast.Constant)
    assert prune_key.value == "pruned_unused_tensors"


@pytest.mark.xfail(
    strict=True,
    reason="the terminal-SiNet HardSwish-SE result is not retained yet",
)
def test_terminal_sinet_hardswish_se_result_is_retained_observation_only() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    result_statement = next(
        statement
        for statement in lowerer.body
        if _direct_call(statement) is not None
    )
    assert _single_target(result_statement) == RESULT_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )

    result_index = lowerer.body.index(result_statement)
    assert _single_target(lowerer.body[result_index - 1]) == SINET_TERMINAL_TARGET
    assert _single_target(lowerer.body[result_index + 1]) == DEQUANT_TARGET
