from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "hardswish_se_layout.py"
)
RAW_WRAPPER = (
    "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_"
    "nhwc_chains"
)
RAW_OWNER = (
    "optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_"
    "nhwc_chains"
)
SUMMARY_OWNER = "run_hardswish_se_layout_summary"
COUNT_TARGET = "terminal_hardswish_se_tensor_count"
SUMMARY_TARGET = "_terminal_hardswish_se_stats"
PREDECESSOR_TARGET = "_terminal_split_conv_concat_bridge_stats"
SUCCESSOR_TARGET = "late_hard_activation_tensor_count"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def test_terminal_hardswish_se_prune_aware_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    count = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COUNT_TARGET
    )
    index = lowerer.body.index(count)
    summary = lowerer.body[index + 1]
    assert isinstance(count, ast.Assign)
    assert ast.unparse(count.value) == "len(model_ir.tensors)"
    assert _single_target(summary) == SUMMARY_TARGET
    assert isinstance(summary, ast.Assign)
    assert isinstance(summary.value, ast.Dict)
    assert len(summary.value.keys) == 2
    assert summary.value.keys[0] is None
    assert ast.unparse(summary.value.values[0]) == f"{RAW_WRAPPER}(model_ir)"
    assert ast.unparse(summary.value.keys[1]) == "'pruned_unused_tensors'"
    assert ast.unparse(summary.value.values[1]) == (
        "max(0, int(terminal_hardswish_se_tensor_count - "
        "len(model_ir.tensors)))"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET
    assert sum(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    ) == 2
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == SUMMARY_TARGET
        for node in ast.walk(lowerer)
    )


@pytest.mark.xfail(
    strict=True,
    reason="terminal HardSwish/SE lacks one prune-aware summary owner",
)
def test_terminal_hardswish_se_uses_one_prune_aware_summary_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    raw_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RAW_OWNER
    ]
    assert len(raw_calls) == 1
    initial_count = next(
        statement
        for statement in owner.body
        if _single_target(statement) == "initial_tensor_count"
    )
    assert isinstance(initial_count, ast.Assign)
    assert ast.unparse(initial_count.value) == "len(model_ir.tensors)"

    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == SUMMARY_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == f"{SUMMARY_OWNER}(model_ir)"
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )

    wrapper = _functions(LOWERER_PATH)[RAW_WRAPPER]
    assert ast.unparse(wrapper.body[0]) == f"return {RAW_WRAPPER}_pass(model_ir)"
