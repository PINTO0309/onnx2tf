from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import (
    late_layout_mean_spp_gather_constant_cast_orchestration,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_layout_mean_spp_gather_constant_cast_orchestration.py"
)
RAW_WRAPPER = "_run_late_layout_mean_spp_gather_constant_cast_pass_cluster"
RAW_OWNER = "run_late_layout_mean_spp_gather_constant_cast"
SUMMARY_OWNER = "run_late_layout_mean_spp_gather_constant_cast_summary"
SUMMARY_FUNCTION = (
    "summarize_late_layout_mean_spp_gather_constant_cast_mutations"
)
COUNT_TARGET = "late_layout_cluster_tensor_count"
RAW_TARGET = "late_layout_cluster_results"
SUMMARY_TARGET = "_late_layout_cluster_stats"
PREDECESSOR_TARGET = "_late_pre_layout_cluster_shape_extract_stats"
SUCCESSOR_TARGET = "_terminal_expand_squeeze_stats"


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


def test_late_layout_cluster_prune_aware_summary_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == SUMMARY_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(late_layout_mean_spp_gather_constant_cast_context, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in {COUNT_TARGET, RAW_TARGET}
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == SUMMARY_TARGET
        for node in ast.walk(lowerer)
    )


def test_late_layout_cluster_uses_one_prune_aware_summary_owner() -> None:
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
    initial_count = next(
        statement
        for statement in owner.body
        if _single_target(statement) == "initial_tensor_count"
    )
    assert isinstance(initial_count, ast.Assign)
    assert ast.unparse(initial_count.value) == "len(context.model_ir.tensors)"

    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == SUMMARY_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(late_layout_mean_spp_gather_constant_cast_context, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in {COUNT_TARGET, RAW_TARGET}
        for node in ast.walk(lowerer)
    )

    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RAW_OWNER
        for node in ast.walk(wrapper)
    )


@pytest.mark.parametrize(
    ("include_layout_transpose", "prune"),
    ((False, False), (True, False), (True, True)),
)
def test_late_layout_cluster_summary_preserves_flags_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    prune: bool,
) -> None:
    model_ir = ModelIR("late_layout_cluster_summary")
    model_ir.tensors["probe"] = TensorIR(
        name="probe",
        dtype="float32",
        shape=[1],
    )
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    required_results = (
        {"mean_mutations": 1},
        {"spp_mutations": 2},
        {"gather_mutations": 3},
        {"constant_fold_mutations": 4},
        {"cast_mutations": 5},
    )
    raw_results = (
        (
            {
                "removed_identity_transpose": 6,
                "removed_inverse_transpose_pairs": 7,
                "removed_inverse_transpose_fanout_branches": 8,
                "composed_consecutive_transpose_pairs": 9,
            },
            *required_results,
        )
        if include_layout_transpose
        else required_results
    )
    expected = (
        late_layout_mean_spp_gather_constant_cast_orchestration
        .summarize_late_layout_mean_spp_gather_constant_cast_mutations(
            raw_results,
            include_layout_transpose=include_layout_transpose,
            pruned_unused_tensors=int(prune),
        )
    )
    observed: list[tuple[ModelIRPassContext, bool]] = []

    def _run(
        candidate: ModelIRPassContext,
        *,
        include_layout_transpose: bool,
    ) -> tuple[dict[str, int], ...]:
        observed.append((candidate, include_layout_transpose))
        if prune:
            del candidate.model_ir.tensors["probe"]
        return raw_results

    monkeypatch.setattr(
        late_layout_mean_spp_gather_constant_cast_orchestration,
        RAW_OWNER,
        _run,
    )

    assert (
        late_layout_mean_spp_gather_constant_cast_orchestration
        .run_late_layout_mean_spp_gather_constant_cast_summary(
            context,
            include_layout_transpose=include_layout_transpose,
        )
        == expected
    )
    assert observed == [(context, include_layout_transpose)]
