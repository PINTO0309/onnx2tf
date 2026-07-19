from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import (
    terminal_affine_concat_split_recovery_orchestration as recovery,
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
    / "terminal_affine_concat_split_recovery_orchestration.py"
)
RAW_WRAPPER = "_run_terminal_affine_concat_split_recovery_sequence"
RAW_OWNER = "run_terminal_affine_concat_split_recovery"
SUMMARY_OWNER = "run_terminal_affine_concat_split_recovery_summary"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_cleanup_orchestration.py"
)
COMPOSITE_OWNER = "run_pre_terminal_cleanup"
COMPOSITE_TARGET = "_pre_terminal_cleanup_results"
SUMMARY_TARGETS = (
    "_pre_terminal_affine_stats",
    "_terminal_affine_stats",
)
COUNT_TARGETS = (
    "pre_terminal_affine_tensor_count",
    "terminal_affine_tensor_count",
)
RAW_RESULT_TARGETS = (
    "pre_terminal_affine_results",
    "terminal_affine_results",
)
PREDECESSOR_TARGETS = (
    "_pre_terminal_instancenorm_layout_results",
    "_pre_terminal_affine_tail_results",
)
SUCCESSOR_TARGETS = (
    "_pre_terminal_pre_add_stats",
    "_terminal_slice_pad_concat_stats",
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


def _composite_summary_calls() -> list[ast.Call]:
    owner = _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SUMMARY_OWNER
    ]


def _terminal_summary(lowerer: ast.FunctionDef) -> ast.Assign:
    return next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == "_terminal_affine_stats"
        and isinstance(statement, ast.Assign)
    )


def test_terminal_affine_recovery_summary_boundaries_are_fixed() -> None:
    lowerer = _lowerer()
    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert ast.unparse(wrapper.body[0]) == (
        "return run_terminal_affine_concat_split_recovery("
        "terminal_affine_concat_split_recovery_context)"
    )

    composite_calls = _composite_summary_calls()
    assert len(composite_calls) == 1
    assert [ast.unparse(argument) for argument in composite_calls[0].args] == [
        "context"
    ]
    summary = _terminal_summary(lowerer)
    index = lowerer.body.index(summary)
    assert ast.unparse(summary.value) == (
        "run_terminal_affine_concat_split_recovery_summary("
        "terminal_affine_concat_split_recovery_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == COMPOSITE_TARGET
    assert _single_target(lowerer.body[index + 1]) == (
        "_terminal_slice_pad_concat_stats"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in (*COUNT_TARGETS, *RAW_RESULT_TARGETS)
        for node in ast.walk(lowerer)
    )


def test_terminal_affine_recovery_uses_one_summary_owner_twice() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {RAW_OWNER, "summarize_terminal_affine_concat_split_mutations"}
    ]
    assert owner_calls.count(RAW_OWNER) == 1
    assert owner_calls.count(
        "summarize_terminal_affine_concat_split_mutations"
    ) == 1

    lowerer = _lowerer()
    assert len(_composite_summary_calls()) == 1
    summary = _terminal_summary(lowerer)
    index = lowerer.body.index(summary)
    assert ast.unparse(summary.value) == (
        "run_terminal_affine_concat_split_recovery_summary("
        "terminal_affine_concat_split_recovery_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == COMPOSITE_TARGET
    assert _single_target(lowerer.body[index + 1]) == (
        "_terminal_slice_pad_concat_stats"
    )

    assert not any(
        isinstance(node, ast.Name)
        and node.id in (*COUNT_TARGETS, *RAW_RESULT_TARGETS)
        for node in ast.walk(lowerer)
    )
    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert ast.unparse(wrapper.body[0]) == (
        "return run_terminal_affine_concat_split_recovery("
        "terminal_affine_concat_split_recovery_context)"
    )


@pytest.mark.parametrize("pruned", [False, True])
def test_terminal_affine_recovery_summary_owner_preserves_prune_evidence(
    monkeypatch: pytest.MonkeyPatch,
    pruned: bool,
) -> None:
    model_ir = ModelIR("terminal_affine_recovery_summary")
    model_ir.tensors["probe"] = TensorIR(
        name="probe",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    raw_results = ({"raw": 1},)
    summary_result = {"summary": 2}
    observed: list[tuple[str, object, object]] = []

    def _raw_owner(candidate: ModelIRPassContext):
        observed.append((RAW_OWNER, candidate, candidate.model_ir))
        if pruned:
            assert candidate.model_ir.tensors.pop("probe", None) is not None
        return raw_results

    def _summary_owner(
        candidate_results: tuple[dict[str, int], ...],
        *,
        pruned_unused_tensors: int,
    ) -> dict[str, int]:
        observed.append(
            (
                "summarize_terminal_affine_concat_split_mutations",
                candidate_results,
                pruned_unused_tensors,
            )
        )
        return summary_result

    monkeypatch.setattr(
        recovery,
        RAW_OWNER,
        _raw_owner,
    )
    monkeypatch.setattr(
        recovery,
        "summarize_terminal_affine_concat_split_mutations",
        _summary_owner,
    )

    assert (
        recovery.run_terminal_affine_concat_split_recovery_summary(context)
        == summary_result
    )
    assert observed == [
        (RAW_OWNER, context, context.model_ir),
        (
            "summarize_terminal_affine_concat_split_mutations",
            raw_results,
            int(pruned),
        ),
    ]
