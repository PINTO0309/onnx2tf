from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import (
    late_hard_activation_layout_orchestration,
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
    / "late_hard_activation_layout_orchestration.py"
)
TERMINAL_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_activation_bridge_orchestration.py"
)
TERMINAL_OWNER = "run_terminal_activation_bridge_cleanup"
RAW_WRAPPER = "_run_late_hard_activation_layout_pass_pair"
RAW_OWNER = "run_late_hard_activation_layout"
SUMMARY_OWNER = "run_late_hard_activation_layout_summary"
SUMMARY_FUNCTION = "summarize_late_hard_activation_layout_mutations"
COUNT_TARGET = "late_hard_activation_tensor_count"
RAW_TARGET = "late_hard_activation_results"
SUMMARY_TARGET = "_late_hard_activation_stats"
COMPOSITE_TARGET = "_terminal_activation_bridge_results"
PREDECESSOR_TARGET = "_terminal_qkv_shape_attention_results"
SUCCESSOR_TARGET = "_terminal_layout_shape_results"
OUTER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_activation_bridge_orchestration.py"
)
OUTER_OWNER = "run_terminal_qkv_activation_bridge_cleanup"
OUTER_TARGET = "_terminal_qkv_activation_bridge_results"
OUTER_PREDECESSOR_TARGET = "_pre_terminal_affine_slice_spp_results"
TOP_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_activation_layout_shape_orchestration.py"
)
TOP_OWNER = "run_terminal_qkv_activation_layout_shape_cleanup"
TOP_TARGET = "_terminal_qkv_activation_layout_shape_results"
LOWERER_OWNER = "run_terminal_affine_qkv_layout_shape_cleanup"
LOWERER_TARGET = "_terminal_affine_qkv_layout_shape_results"
LOWERER_PREDECESSOR_GUARD = (
    "_late_binary_layout_recovery_requires_reconciliation"
)
TOP_SUCCESSOR_PHASE_ID = "shape_reconciliation.terminal.expand_squeeze"


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


def _outer_calls() -> list[ast.Call]:
    owner = _functions(OUTER_PATH)[OUTER_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_OWNER
    ]


def _top_calls() -> list[ast.Call]:
    owner = _functions(TOP_PATH)[TOP_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OUTER_OWNER
    ]


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(
        statement.value, ast.Call
    ):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or ast.unparse(call.func) != "session.record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_late_hard_activation_prune_aware_summary_boundary_is_fixed() -> None:
    terminal_owner = _functions(TERMINAL_OWNER_PATH)[TERMINAL_OWNER]
    summary_call = next(
        node
        for node in ast.walk(terminal_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SUMMARY_OWNER
    )
    assert ast.unparse(summary_call) == (
        f"{SUMMARY_OWNER}(context, "
        "include_layout_transpose=include_layout_transpose)"
    )

    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == LOWERER_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value).startswith(f"{LOWERER_OWNER}(")
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == LOWERER_PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == TOP_SUCCESSOR_PHASE_ID
    assert len(_top_calls()) == 1
    assert len(_outer_calls()) == 1
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


def test_late_hard_activation_uses_one_prune_aware_summary_owner() -> None:
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
        if _single_target(statement) == LOWERER_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value).startswith(f"{LOWERER_OWNER}(")
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == LOWERER_PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == TOP_SUCCESSOR_PHASE_ID
    assert len(_top_calls()) == 1
    assert len(_outer_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in {COUNT_TARGET, RAW_TARGET}
        for node in ast.walk(lowerer)
    )
    terminal_owner = _functions(TERMINAL_OWNER_PATH)[TERMINAL_OWNER]
    summary_calls = [
        node
        for node in ast.walk(terminal_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SUMMARY_OWNER
    ]
    assert len(summary_calls) == 1

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
def test_late_hard_activation_summary_preserves_flags_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    prune: bool,
) -> None:
    model_ir = ModelIR("late_hard_activation_summary")
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
    raw_results = (
        ({"rewritten_hardsigmoid_transpose_passthrough_chains": 3},)
        if not include_layout_transpose
        else (
            {"rewritten_hardsigmoid_transpose_passthrough_chains": 3},
            {
                "removed_identity_transpose": 4,
                "removed_inverse_transpose_pairs": 5,
                "removed_inverse_transpose_fanout_branches": 6,
                "composed_consecutive_transpose_pairs": 7,
            },
        )
    )
    expected = (
        late_hard_activation_layout_orchestration
        .summarize_late_hard_activation_layout_mutations(
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
        late_hard_activation_layout_orchestration,
        RAW_OWNER,
        _run,
    )

    assert (
        late_hard_activation_layout_orchestration
        .run_late_hard_activation_layout_summary(
            context,
            include_layout_transpose=include_layout_transpose,
        )
        == expected
    )
    assert observed == [(context, include_layout_transpose)]
