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
    / "terminal_affine_concat_split_recovery_orchestration.py"
)
RAW_WRAPPER = "_run_terminal_affine_concat_split_recovery_sequence"
RAW_OWNER = "run_terminal_affine_concat_split_recovery"
SUMMARY_OWNER = "run_terminal_affine_concat_split_recovery_summary"
SUMMARY_TARGETS = (
    "_pre_terminal_affine_stats",
    "_terminal_affine_stats",
)
COUNT_TARGETS = (
    "pre_terminal_affine_tensor_count",
    "terminal_affine_tensor_count",
)
RAW_RESULT_TARGETS = (
    "pre_terminal_affine_results",
    "terminal_affine_results",
)
PREDECESSOR_TARGETS = (
    "_pre_terminal_instancenorm_layout_results",
    "_pre_terminal_affine_slice_pad_concat_stats",
)
SUCCESSOR_TARGETS = (
    "pre_terminal_pre_add_tensor_count",
    "_terminal_slice_pad_concat_stats",
)


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


def test_terminal_affine_recovery_evidence_triples_are_fixed() -> None:
    lowerer = _lowerer()
    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert ast.unparse(wrapper.body[0]) == (
        "return run_terminal_affine_concat_split_recovery("
        "terminal_affine_concat_split_recovery_context)"
    )

    for position, summary_target in enumerate(SUMMARY_TARGETS):
        summary = next(
            statement
            for statement in lowerer.body
            if _single_target(statement) == summary_target
        )
        index = lowerer.body.index(summary)
        count = lowerer.body[index - 2]
        raw_result = lowerer.body[index - 1]
        assert _single_target(count) == COUNT_TARGETS[position]
        assert isinstance(count, ast.Assign)
        assert ast.unparse(count.value) == "len(model_ir.tensors)"
        assert _single_target(raw_result) == RAW_RESULT_TARGETS[position]
        assert isinstance(raw_result, ast.Assign)
        assert ast.unparse(raw_result.value) == f"{RAW_WRAPPER}()"
        assert isinstance(summary, ast.Assign)
        assert ast.unparse(summary.value) == (
            "summarize_terminal_affine_concat_split_mutations("
            f"{RAW_RESULT_TARGETS[position]}, "
            "pruned_unused_tensors=max(0, "
            f"int({COUNT_TARGETS[position]} - len(model_ir.tensors))))"
        )
        assert _single_target(lowerer.body[index - 3]) == (
            PREDECESSOR_TARGETS[position]
        )
        assert _single_target(lowerer.body[index + 1]) == (
            SUCCESSOR_TARGETS[position]
        )


@pytest.mark.xfail(
    strict=True,
    reason="terminal-affine recovery lacks one prune-aware summary owner",
)
def test_terminal_affine_recovery_uses_one_summary_owner_twice() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {RAW_OWNER, "summarize_terminal_affine_concat_split_mutations"}
    ]
    assert owner_calls.count(RAW_OWNER) == 1
    assert owner_calls.count(
        "summarize_terminal_affine_concat_split_mutations"
    ) == 1

    lowerer = _lowerer()
    for position, summary_target in enumerate(SUMMARY_TARGETS):
        summary = next(
            statement
            for statement in lowerer.body
            if _single_target(statement) == summary_target
        )
        index = lowerer.body.index(summary)
        assert isinstance(summary, ast.Assign)
        assert ast.unparse(summary.value) == (
            "run_terminal_affine_concat_split_recovery_summary("
            "terminal_affine_concat_split_recovery_context)"
        )
        assert _single_target(lowerer.body[index - 1]) == (
            PREDECESSOR_TARGETS[position]
        )
        assert _single_target(lowerer.body[index + 1]) == (
            SUCCESSOR_TARGETS[position]
        )

    assert not any(
        isinstance(node, ast.Name)
        and node.id in (*COUNT_TARGETS, *RAW_RESULT_TARGETS)
        for node in ast.walk(lowerer)
    )
    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert ast.unparse(wrapper.body[0]) == (
        "return run_terminal_affine_concat_split_recovery("
        "terminal_affine_concat_split_recovery_context)"
    )
