from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import no_layout_final_cleanup_orchestration


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "no_layout_final_cleanup_orchestration.py"
)
GUARD = "apply_safe_transpose_reduction_lite_on_no_layout_opt"
SE_FC_OWNER = "run_se_fc_layout_cleanup"
AFFINE_OWNER = "optimize_transpose_mul_add_const_prepost_nhwc_chains"
AFFINE_WRAPPER = "_optimize_transpose_mul_add_const_prepost_nhwc_chains"
SUMMARY_OWNER = "run_no_layout_final_cleanup"
OLD_TARGETS = (
    "_no_layout_final_se_fc_stats",
    "_no_layout_final_affine_prepost_stats",
)
SUMMARY_TARGET = "_no_layout_final_cleanup_results"
TOPOLOGY_PHASE = "topology.primary.no_layout_post_reduction"
PREDECESSOR_PHASE = "topology.primary.post_lowering"
SUCCESSOR_TARGET = "_absolute_final_boundary_signature_results"


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


def _assert_topology_phase(statement: ast.stmt, phase_id: str) -> None:
    assert isinstance(statement, ast.Expr)
    assert ast.unparse(statement) == (
        f"session.record_phase_result('{phase_id}', "
        "_topologically_sort_operators(model_ir))"
    )


def _guard(lowerer: ast.FunctionDef) -> tuple[int, ast.If]:
    index, guard = next(
        (index, statement)
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == GUARD
    )
    return index, guard


def test_no_layout_final_cleanup_raw_pair_and_boundaries_are_fixed() -> None:
    lowerer = _lowerer()
    guard_index, guard = _guard(lowerer)
    assert len(guard.body) == 2
    summary = guard.body[0]
    assert _single_target(summary) == SUMMARY_TARGET
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(shared_model_ir_pass_context)"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_TARGETS
        for node in ast.walk(lowerer)
    )
    _assert_topology_phase(guard.body[1], TOPOLOGY_PHASE)
    _assert_topology_phase(lowerer.body[guard_index - 1], PREDECESSOR_PHASE)
    assert _single_target(lowerer.body[guard_index + 1]) == SUCCESSOR_TARGET


def test_no_layout_final_cleanup_uses_one_ordered_context_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {SE_FC_OWNER, AFFINE_OWNER}
    ]
    assert owner_calls == [SE_FC_OWNER, AFFINE_OWNER]

    lowerer = _lowerer()
    guard_index, guard = _guard(lowerer)
    assert len(guard.body) == 2
    summary = guard.body[0]
    assert _single_target(summary) == SUMMARY_TARGET
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(shared_model_ir_pass_context)"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_TARGETS
        for node in ast.walk(lowerer)
    )
    _assert_topology_phase(guard.body[1], TOPOLOGY_PHASE)
    _assert_topology_phase(lowerer.body[guard_index - 1], PREDECESSOR_PHASE)
    assert _single_target(lowerer.body[guard_index + 1]) == SUCCESSOR_TARGET
    assert AFFINE_WRAPPER in _functions(LOWERER_PATH)

    se_layout_import = next(
        node
        for node in ast.parse(LOWERER_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.passes.se_layout"
    )
    assert SE_FC_OWNER in {alias.name for alias in se_layout_import.names}


def test_no_layout_final_cleanup_preserves_context_order_and_raw_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("no_layout_final_cleanup")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict[str, object]] = []
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    results = (
        {"optimized_transpose_se_fc_mul_prepost_nhwc_chains": 2},
        {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 3},
    )
    events: list[tuple[str, ModelIR, dict[str, object]]] = []

    def _run_se_fc(candidate: ModelIR, **kwargs: object) -> dict[str, int]:
        events.append((SE_FC_OWNER, candidate, dict(kwargs)))
        return dict(results[0])

    def _run_affine(candidate: ModelIR, **kwargs: object) -> dict[str, int]:
        events.append((AFFINE_OWNER, candidate, dict(kwargs)))
        return dict(results[1])

    monkeypatch.setattr(
        no_layout_final_cleanup_orchestration,
        SE_FC_OWNER,
        _run_se_fc,
    )
    monkeypatch.setattr(
        no_layout_final_cleanup_orchestration,
        AFFINE_OWNER,
        _run_affine,
    )

    assert no_layout_final_cleanup_orchestration.run_no_layout_final_cleanup(
        context
    ) == results
    assert events == [
        (
            SE_FC_OWNER,
            model_ir,
            {
                "layout_state": layout_state,
                "diagnostics": diagnostics,
            },
        ),
        (
            AFFINE_OWNER,
            model_ir,
            {"layout_state": layout_state},
        ),
    ]
