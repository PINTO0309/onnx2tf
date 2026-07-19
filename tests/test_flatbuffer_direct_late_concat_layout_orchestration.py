from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import late_concat_layout_orchestration


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_concat_layout_orchestration.py"
)
OWNER = "run_late_concat_layout_cleanup"
RESULT_TARGET = "_late_concat_layout_results"
SCOPE_TARGET = "late_concat_layout_state_scope"
PREDECESSOR_TARGET = "_late_cost_volume_conv_affine_stats"
OLD_RESULT_TARGETS = (
    "_late_concat_axis3_const_layout_stats",
    "_late_concat_dequant_quantize_layout_stats",
    "_late_concat_layernorm_layout_stats",
    "_late_concat_transpose_layout_stats",
)
PASS_IDS = (
    "run_axis3_const_concat_layout_cleanup",
    "run_dequant_concat_quantize_layout_cleanup",
    "run_layernorm_statistics_layout_cleanup",
    "run_layout_transpose_cleanup",
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


def test_late_concat_layout_cluster_uses_one_composite_result_outside_store() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_concat_layout_cleanup(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == "optimize_layout_transpose_chains"
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {SCOPE_TARGET, *OLD_RESULT_TARGETS}
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_phase_result"
        and any(
            isinstance(child, ast.Name) and child.id == OWNER
            for child in ast.walk(node)
        )
        for node in ast.walk(lowerer)
    )


def test_late_concat_layout_cluster_uses_one_composite_owner() -> None:
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"ModelIRPassStateScope", *PASS_IDS}
    ]
    assert owner_calls == ["ModelIRPassStateScope", *PASS_IDS]

    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_concat_layout_cleanup(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert isinstance(lowerer.body[index + 1], ast.If)
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {SCOPE_TARGET, *OLD_RESULT_TARGETS}
        for node in ast.walk(lowerer)
    )


def test_late_concat_layout_owner_shares_scope_and_preserves_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_concat_layout_owner")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[str, object, object, object, object]] = []

    def _callback(name: str, result: dict[str, int]):
        def _run(
            candidate: ModelIR,
            *,
            layout_state: object,
            diagnostics: object,
            state_scope: object,
        ) -> dict[str, int]:
            observed.append(
                (name, candidate, layout_state, diagnostics, state_scope)
            )
            return dict(result)

        return _run

    expected_results = tuple(
        {f"result_{index}": index}
        for index in range(1, len(PASS_IDS) + 1)
    )
    for pass_id, result in zip(PASS_IDS, expected_results, strict=True):
        monkeypatch.setattr(
            late_concat_layout_orchestration,
            pass_id,
            _callback(pass_id, result),
        )

    assert late_concat_layout_orchestration.run_late_concat_layout_cleanup(
        context
    ) == expected_results
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    for _, candidate, layout_state, diagnostics, _ in observed:
        assert candidate is context.model_ir
        assert layout_state is context.layout_state
        assert diagnostics is context.diagnostics
    scopes = [entry[4] for entry in observed]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
