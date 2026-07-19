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
    / "late_spp_concat_unary_conv_orchestration.py"
)
RAW_WRAPPER = "_run_late_spp_concat_unary_conv_pass_pair"
RAW_OWNER = "run_late_spp_concat_unary_conv"
SUMMARY_OWNER = "run_late_spp_concat_unary_conv_summary"
SUMMARY_FUNCTION = "summarize_late_spp_concat_unary_conv_mutations"
RAW_TARGET = "late_spp_results"
SUMMARY_TARGET = "_late_spp_stats"
PREDECESSOR_TARGET = "_terminal_slice_pad_concat_stats"
SUCCESSOR_TARGET = "_late_pre_qkv_shape_extract_stats"


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


def test_late_spp_raw_summary_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    raw = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RAW_TARGET
    )
    index = lowerer.body.index(raw)
    summary = lowerer.body[index + 1]
    assert isinstance(raw, ast.Assign)
    assert ast.unparse(raw.value) == f"{RAW_WRAPPER}()"
    assert _single_target(summary) == SUMMARY_TARGET
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == f"{SUMMARY_FUNCTION}({RAW_TARGET})"
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET
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

    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert ast.unparse(wrapper.body[0]) == (
        f"return {RAW_OWNER}(late_spp_concat_unary_conv_context)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="late SPP/Concat/Unary lacks one direct summary owner",
)
def test_late_spp_uses_one_direct_summary_owner() -> None:
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

    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == SUMMARY_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(late_spp_concat_unary_conv_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id == RAW_TARGET
        for node in ast.walk(lowerer)
    )

    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert ast.unparse(wrapper.body[0]) == (
        f"return {RAW_OWNER}(late_spp_concat_unary_conv_context)"
    )
