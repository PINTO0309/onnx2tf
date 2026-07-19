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
    / "pre_terminal_pre_add_orchestration.py"
)
OWNER = "run_pre_terminal_pre_add_cleanup"
RESULT_TARGET = "_pre_terminal_pre_add_stats"
COUNT_TARGET = "pre_terminal_pre_add_tensor_count"
PASS_ID = "_optimize_transpose_pre_add_nhwc_chains"
PREDECESSOR_TARGET = "_pre_terminal_affine_stats"
SUCCESSOR_TARGET = "channel_slice_pad_mul_results"


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


def test_pre_terminal_pre_add_prune_evidence_is_fixed() -> None:
    lowerer = _lowerer()
    result = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(result)
    count = lowerer.body[index - 1]
    assert _single_target(count) == COUNT_TARGET
    assert isinstance(count, ast.Assign)
    assert ast.unparse(count.value) == "len(model_ir.tensors)"
    assert isinstance(result, ast.Assign)
    assert ast.unparse(result.value) == (
        "{**_optimize_transpose_pre_add_nhwc_chains("
        "model_ir, layout_state=session.layout_state), "
        "'pruned_unused_tensors': max(0, "
        "int(pre_terminal_pre_add_tensor_count - len(model_ir.tensors)))}"
    )
    assert _single_target(lowerer.body[index - 2]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert sum(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    ) == 2


@pytest.mark.xfail(
    strict=True,
    reason="pre-terminal pre-add lacks one prune-aware owner",
)
def test_pre_terminal_pre_add_uses_one_prune_aware_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    pass_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PASS_ID.removeprefix("_")
    ]
    assert len(pass_calls) == 1
    assert ast.unparse(pass_calls[0].args[0]) == "context.model_ir"

    lowerer = _lowerer()
    result = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(result)
    assert isinstance(result, ast.Assign)
    assert ast.unparse(result.value) == (
        "run_pre_terminal_pre_add_cleanup(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )
