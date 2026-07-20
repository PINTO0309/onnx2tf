from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import prelu_passthrough_layout


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "prelu_passthrough_layout.py"
)
RAW_OWNER = "optimize_prelu_transpose_passthrough_chains"
RAW_WRAPPER = "_optimize_prelu_transpose_passthrough_chains"
SUMMARY_OWNER = "run_prelu_transpose_passthrough_summary"
COUNT_TARGET = "final_prelu_tensor_count"
SUMMARY_TARGET = "final_prelu_stats"
PREDECESSOR_TARGET = "final_gather_stats"
SUCCESSOR_TARGET = "final_consecutive_reshape_stats"


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


def test_final_prelu_rewrite_or_prune_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    stats = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == SUMMARY_TARGET
    )
    index = lowerer.body.index(stats)
    assert isinstance(stats, ast.Assign)
    assert ast.unparse(stats.value) == (
        f"{SUMMARY_OWNER}(model_ir, layout_state=session.layout_state)"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )

    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == PREDECESSOR_TARGET
        for node in ast.walk(predecessor.test)
    )
    guard = lowerer.body[index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "_stats_have_positive_count(final_prelu_stats)"
    )
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET


def test_final_prelu_uses_dedicated_prune_aware_summary_owner() -> None:
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
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(model_ir, layout_state=session.layout_state)"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert any(
        isinstance(node, ast.Name) and node.id == PREDECESSOR_TARGET
        for node in ast.walk(predecessor.test)
    )
    guard = lowerer.body[index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "_stats_have_positive_count(final_prelu_stats)"
    )
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET

    wrapper = _functions(LOWERER_PATH)[RAW_WRAPPER]
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == f"{RAW_WRAPPER}_pass"
        for node in ast.walk(wrapper)
    )


@pytest.mark.parametrize("prune", (False, True))
def test_final_prelu_summary_preserves_layout_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    prune: bool,
) -> None:
    model_ir = ModelIR("final_prelu_summary")
    model_ir.tensors["probe"] = TensorIR(
        name="probe",
        dtype="float32",
        shape=[1],
    )
    layout_state = LayoutState.from_model_ir(model_ir)
    raw_result = {"rewritten_prelu_transpose_passthrough_chains": 3}
    observed: list[tuple[ModelIR, LayoutState | None]] = []

    def _run(
        candidate: ModelIR,
        *,
        layout_state: LayoutState | None = None,
    ) -> dict[str, int]:
        observed.append((candidate, layout_state))
        if prune:
            del candidate.tensors["probe"]
        return raw_result

    monkeypatch.setattr(prelu_passthrough_layout, RAW_OWNER, _run)

    assert prelu_passthrough_layout.run_prelu_transpose_passthrough_summary(
        model_ir,
        layout_state=layout_state,
    ) == {
        **raw_result,
        "pruned_unused_tensors": int(prune),
    }
    assert observed == [(model_ir, layout_state)]
