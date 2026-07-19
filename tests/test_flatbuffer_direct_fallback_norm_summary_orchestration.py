from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import pad_layout


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pad_layout.py"
)
RAW_OWNER = "run_pad_layout_cleanup"
SUMMARY_OWNER = "run_norm_subgraph_pad_layout_summary"
COUNT_TARGET = "fallback_norm_tensor_count"
SUMMARY_TARGET = "fallback_norm_stats"
PREDECESSOR_TARGET = "fallback_ir"
SUCCESSOR_TARGET = "_fallback_dynamic_rank1_stats"


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


def _containing_body(root: ast.AST, target: ast.stmt) -> list[ast.stmt]:
    for node in ast.walk(root):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and target in value:
                return value
    raise AssertionError("statement is not contained by an AST body")


def test_fallback_norm_prune_aware_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    stats = next(
        statement
        for statement in ast.walk(lowerer)
        if _single_target(statement) == SUMMARY_TARGET
    )
    body = _containing_body(lowerer, stats)
    index = body.index(stats)
    assert isinstance(stats, ast.Assign)
    assert ast.unparse(stats.value) == (
        f"{SUMMARY_OWNER}(fallback_ir, diagnostics=session.diagnostics)"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )
    assert _single_target(body[index - 1]) == PREDECESSOR_TARGET
    guard = body[index + 1]
    assert isinstance(guard, ast.If)
    assert any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == SUMMARY_TARGET
        for node in ast.walk(guard.test)
    )
    assert _single_target(body[index + 2]) == SUCCESSOR_TARGET


def test_fallback_norm_uses_dedicated_prune_aware_summary_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    raw_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RAW_OWNER
    ]
    assert len(raw_calls) == 1
    raw_call = raw_calls[0]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in raw_call.keywords
    } == {
        "include_pad": "False",
        "include_unary": "False",
        "include_norm": "True",
        "diagnostics": "diagnostics",
    }
    initial_count = next(
        statement
        for statement in owner.body
        if _single_target(statement) == "initial_tensor_count"
    )
    assert isinstance(initial_count, ast.Assign)
    assert ast.unparse(initial_count.value) == "len(model_ir.tensors)"

    lowerer = _lowerer()
    stats = next(
        statement
        for statement in ast.walk(lowerer)
        if _single_target(statement) == SUMMARY_TARGET
    )
    body = _containing_body(lowerer, stats)
    index = body.index(stats)
    assert isinstance(stats, ast.Assign)
    assert ast.unparse(stats.value) == (
        f"{SUMMARY_OWNER}(fallback_ir, diagnostics=session.diagnostics)"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )
    assert _single_target(body[index - 1]) == PREDECESSOR_TARGET
    guard = body[index + 1]
    assert isinstance(guard, ast.If)
    assert any(
        isinstance(node, ast.Name) and node.id == SUMMARY_TARGET
        for node in ast.walk(guard.test)
    )
    assert _single_target(body[index + 2]) == SUCCESSOR_TARGET


@pytest.mark.parametrize("prune", (False, True))
def test_fallback_norm_summary_preserves_flags_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    prune: bool,
) -> None:
    model_ir = ModelIR("fallback_norm_summary")
    model_ir.tensors["probe"] = TensorIR(
        name="probe",
        dtype="float32",
        shape=[1],
    )
    diagnostics: list[dict[str, object]] = []
    raw_result = {
        "optimized_transpose_pad_prepost_nhwc_chains": 0,
        "optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains": 0,
        "optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains": 3,
    }
    observed: list[tuple[ModelIR, dict[str, object]]] = []

    def _run(candidate: ModelIR, **kwargs: object) -> dict[str, int]:
        observed.append((candidate, kwargs))
        if prune:
            del candidate.tensors["probe"]
        return raw_result

    monkeypatch.setattr(pad_layout, RAW_OWNER, _run)

    assert pad_layout.run_norm_subgraph_pad_layout_summary(
        model_ir,
        diagnostics=diagnostics,
    ) == {
        **raw_result,
        "pruned_unused_tensors": int(prune),
    }
    assert observed == [
        (
            model_ir,
            {
                "include_pad": False,
                "include_unary": False,
                "include_norm": True,
                "diagnostics": diagnostics,
            },
        )
    ]
