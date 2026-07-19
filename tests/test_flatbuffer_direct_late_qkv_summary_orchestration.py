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
    / "qkv_attention_orchestration.py"
)
RAW_WRAPPER = "_run_qkv_attention_layout_pass_cluster"
RAW_OWNER = "run_qkv_attention"
SUMMARY_OWNER = "run_qkv_attention_summary"
SUMMARY_FUNCTION = "summarize_qkv_attention_mutations"
COUNT_TARGET = "late_qkv_tensor_count"
RAW_TARGET = "late_qkv_results"
SUMMARY_TARGET = "_late_qkv_stats"
PREDECESSOR_TARGET = "_late_pre_qkv_shape_extract_stats"
SUCCESSOR_TARGET = "_terminal_split_conv_concat_bridge_stats"


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


def test_late_qkv_prune_aware_summary_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    count = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COUNT_TARGET
    )
    index = lowerer.body.index(count)
    raw = lowerer.body[index + 1]
    summary = lowerer.body[index + 2]
    assert isinstance(count, ast.Assign)
    assert ast.unparse(count.value) == "len(model_ir.tensors)"
    assert _single_target(raw) == RAW_TARGET
    assert isinstance(raw, ast.Assign)
    assert ast.unparse(raw.value) == (
        f"{RAW_WRAPPER}(include_layout_transpose="
        "optimize_layout_transpose_chains, include_prefix=False)"
    )
    assert _single_target(summary) == SUMMARY_TARGET
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_FUNCTION}({RAW_TARGET}, include_layout_transpose="
        "optimize_layout_transpose_chains, include_prefix=False, "
        "pruned_unused_tensors=max(0, int(late_qkv_tensor_count - "
        "len(model_ir.tensors))))"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 3]) == SUCCESSOR_TARGET
    assert sum(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    ) == 2
    assert sum(
        isinstance(node, ast.Name) and node.id == RAW_TARGET
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
    reason="late QKV lacks one prune-aware summary owner",
)
def test_late_qkv_uses_one_prune_aware_summary_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {RAW_OWNER, SUMMARY_FUNCTION}
    ]
    assert owner_calls.count(RAW_OWNER) == 1
    assert owner_calls.count(SUMMARY_FUNCTION) == 1
    initial_count = next(
        statement
        for statement in owner.body
        if _single_target(statement) == "initial_tensor_count"
    )
    assert isinstance(initial_count, ast.Assign)
    assert ast.unparse(initial_count.value) == "len(context.model_ir.tensors)"

    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == SUMMARY_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(qkv_attention_context, include_layout_transpose="
        "optimize_layout_transpose_chains, include_prefix=False)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in {COUNT_TARGET, RAW_TARGET}
        for node in ast.walk(lowerer)
    )

    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RAW_OWNER
        for node in ast.walk(wrapper)
    )
