from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import static_shape_signature_sanitization


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
    assert len(realign_calls) == 1
    assert len(sanitize_calls) == 0


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


def test_boundary_shape_signature_cleanup_preserves_order_and_raw_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("boundary_shape_signature_cleanup")
    results = (
        {"realigned_dynamic_boundary_shape_signature_map": 2},
        {
            "sanitized_static_shape_signature_consistency": 3,
            "preserved_dynamic_boundary_shape_signature": 4,
            "preserved_dynamic_leading_axis_shape_signature": 5,
            "preserved_dynamic_lineage_shape_signature": 6,
        },
    )
    events: list[tuple[str, ModelIR]] = []

    def _recorder(name: str, result: dict[str, int]):
        def _run(candidate: ModelIR) -> dict[str, int]:
            events.append((name, candidate))
            return dict(result)

        return _run

    monkeypatch.setattr(
        static_shape_signature_sanitization,
        REALIGN_OWNER,
        _recorder(REALIGN_OWNER, results[0]),
    )
    monkeypatch.setattr(
        static_shape_signature_sanitization,
        SANITIZE_OWNER,
        _recorder(SANITIZE_OWNER, results[1]),
    )

    assert (
        static_shape_signature_sanitization.run_boundary_shape_signature_cleanup(
            model_ir
        )
        == results
    )
    assert events == [
        (REALIGN_OWNER, model_ir),
        (SANITIZE_OWNER, model_ir),
    ]
