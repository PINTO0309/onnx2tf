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
    assert len(guard.body) == 3
    assert tuple(_single_target(statement) for statement in guard.body[:2]) == (
        OLD_TARGETS
    )
    se_fc = guard.body[0]
    affine = guard.body[1]
    assert isinstance(se_fc, ast.Assign)
    assert ast.unparse(se_fc.value) == (
        "run_se_fc_layout_cleanup(model_ir, "
        "layout_state=session.layout_state, "
        "diagnostics=session.diagnostics)"
    )
    assert isinstance(affine, ast.Assign)
    assert ast.unparse(affine.value) == (
        f"{AFFINE_WRAPPER}(model_ir, layout_state=session.layout_state)"
    )
    _assert_topology_phase(guard.body[2], TOPOLOGY_PHASE)
    _assert_topology_phase(lowerer.body[guard_index - 1], PREDECESSOR_PHASE)
    assert _single_target(lowerer.body[guard_index + 1]) == SUCCESSOR_TARGET


@pytest.mark.xfail(
    strict=True,
    reason="no-layout final SE-FC/affine pair lacks one ordered owner",
)
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
