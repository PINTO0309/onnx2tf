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
    / "conv_input_adapter_repair.py"
)
RAW_OWNER = "_repair_stale_nchw_to_nhwc_conv_input_transposes"
SUMMARY_OWNER = "run_stale_conv_input_adapter_repair_summary"
COUNT_TARGET = "final_conv_input_tensor_count"
SUMMARY_TARGET = "final_conv_input_stats"
PREDECESSOR_TARGET = "final_pad_layout_stats"
SUCCESSOR_TARGET = "final_concat_layout_stats"


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


def test_final_stale_conv_input_prune_aware_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    count = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COUNT_TARGET
    )
    index = lowerer.body.index(count)
    stats = lowerer.body[index + 1]
    assert isinstance(count, ast.Assign)
    assert ast.unparse(count.value) == "len(model_ir.tensors)"
    assert _single_target(stats) == SUMMARY_TARGET
    assert isinstance(stats, ast.Assign)
    assert isinstance(stats.value, ast.Dict)
    assert len(stats.value.keys) == 2
    assert stats.value.keys[0] is None
    assert ast.unparse(stats.value.values[0]) == f"{RAW_OWNER}(model_ir)"
    assert ast.unparse(stats.value.keys[1]) == "'pruned_unused_tensors'"
    assert ast.unparse(stats.value.values[1]) == (
        f"max(0, {COUNT_TARGET} - len(model_ir.tensors))"
    )

    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == PREDECESSOR_TARGET
        for node in ast.walk(predecessor.test)
    )
    guard = lowerer.body[index + 2]
    assert isinstance(guard, ast.If)
    assert any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == SUMMARY_TARGET
        for node in ast.walk(guard.test)
    )
    assert _single_target(lowerer.body[index + 3]) == SUCCESSOR_TARGET


@pytest.mark.xfail(
    strict=True,
    reason="final stale Conv-input site lacks its prune-aware summary owner",
)
def test_final_stale_conv_input_uses_dedicated_prune_aware_summary_owner() -> None:
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
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert any(
        isinstance(node, ast.Name)
        and node.id == PREDECESSOR_TARGET
        for node in ast.walk(predecessor.test)
    )
    guard = lowerer.body[index + 1]
    assert isinstance(guard, ast.If)
    assert any(
        isinstance(node, ast.Name)
        and node.id == SUMMARY_TARGET
        for node in ast.walk(guard.test)
    )
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET

    wrapper = _functions(LOWERER_PATH)[RAW_OWNER]
    assert len(wrapper.body) == 1
    assert ast.unparse(wrapper.body[0]) == (
        f"return {RAW_OWNER}_pass(model_ir, graph_index=graph_index)"
    )
