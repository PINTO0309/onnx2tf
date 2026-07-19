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
    / "static_shape_signature_sanitization.py"
)
REALIGN_OWNER = "realign_dynamic_boundary_shape_signature_map"
SANITIZE_OWNER = "sanitize_static_shape_signature_consistency"
REALIGN_WRAPPER = "_realign_dynamic_boundary_shape_signature_map"
SANITIZE_WRAPPER = "_sanitize_static_shape_signature_consistency"
SUMMARY_OWNER = "run_boundary_shape_signature_cleanup"
OLD_TARGETS = (
    "_absolute_final_boundary_signature_stats",
    "_absolute_final_static_signature_stats",
)
SUMMARY_TARGET = "_absolute_final_boundary_signature_results"
SUCCESSOR_TARGET = "_absolute_final_affine_post_add_stats"


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


def test_absolute_final_boundary_signature_pair_is_fixed() -> None:
    lowerer = _lowerer()
    realign = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == OLD_TARGETS[0]
    )
    index = lowerer.body.index(realign)
    sanitize = lowerer.body[index + 1]
    assert _single_target(sanitize) == OLD_TARGETS[1]
    assert isinstance(realign, ast.Assign)
    assert ast.unparse(realign.value) == f"{REALIGN_WRAPPER}(model_ir)"
    assert isinstance(sanitize, ast.Assign)
    assert ast.unparse(sanitize.value) == f"{SANITIZE_WRAPPER}(model_ir)"
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET

    realign_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == REALIGN_WRAPPER
    ]
    sanitize_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SANITIZE_WRAPPER
    ]
    assert len(realign_calls) == 2
    assert len(sanitize_calls) == 1


@pytest.mark.xfail(
    strict=True,
    reason="absolute-final boundary signature pair lacks one ordered owner",
)
def test_absolute_final_boundary_signature_pair_uses_one_ordered_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {REALIGN_OWNER, SANITIZE_OWNER}
    ]
    assert owner_calls == [REALIGN_OWNER, SANITIZE_OWNER]

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
        isinstance(node, ast.Name) and node.id in OLD_TARGETS
        for node in ast.walk(lowerer)
    )
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET

    functions = _functions(LOWERER_PATH)
    for wrapper_name in (REALIGN_WRAPPER, SANITIZE_WRAPPER):
        assert wrapper_name in functions
