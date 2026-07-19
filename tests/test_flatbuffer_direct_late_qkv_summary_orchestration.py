from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import qkv_attention_orchestration

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "qkv_attention_orchestration.py"
)
RAW_WRAPPER = "_run_qkv_attention_layout_pass_cluster"
RAW_OWNER = "run_qkv_attention"
SUMMARY_OWNER = "run_qkv_attention_summary"
SUMMARY_FUNCTION = "summarize_qkv_attention_mutations"
COUNT_TARGET = "late_qkv_tensor_count"
RAW_TARGET = "late_qkv_results"
SUMMARY_TARGET = "_late_qkv_stats"
PREDECESSOR_TARGET = "_late_pre_qkv_shape_extract_stats"
SUCCESSOR_TARGET = "_terminal_split_conv_concat_bridge_stats"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_shape_attention_orchestration.py"
)
COMPOSITE_OWNER = "run_terminal_qkv_shape_attention_cleanup"
COMPOSITE_TARGET = "_terminal_qkv_shape_attention_results"


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


def test_late_qkv_prune_aware_summary_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{COMPOSITE_OWNER}(shared_model_ir_pass_context, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )
    assert _single_target(lowerer.body[index - 1]) == (
        "_terminal_affine_slice_spp_results"
    )
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    calls = _composite_summary_calls()
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {COUNT_TARGET, RAW_TARGET}
        for node in ast.walk(lowerer)
    )


def test_late_qkv_uses_one_prune_aware_summary_owner() -> None:
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
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{COMPOSITE_OWNER}(shared_model_ir_pass_context, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )
    assert _single_target(lowerer.body[index - 1]) == (
        "_terminal_affine_slice_spp_results"
    )
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert len(_composite_summary_calls()) == 1
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
    ("include_layout_transpose", "include_prefix", "prune"),
    (
        (False, False, False),
        (True, False, True),
        (False, True, True),
    ),
)
def test_qkv_summary_owner_preserves_flags_context_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    include_prefix: bool,
    prune: bool,
) -> None:
    model_ir = ModelIR("late_qkv_summary")
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
    result_count = 1 + int(include_layout_transpose) + int(include_prefix)
    raw_results = tuple({} for _ in range(result_count))
    expected = qkv_attention_orchestration.summarize_qkv_attention_mutations(
        raw_results,
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
        pruned_unused_tensors=int(prune),
    )
    observed: list[tuple[object, bool, bool]] = []

    def _run(
        candidate: ModelIRPassContext,
        *,
        include_layout_transpose: bool,
        include_prefix: bool,
    ) -> tuple[dict[str, int], ...]:
        observed.append(
            (candidate, include_layout_transpose, include_prefix)
        )
        if prune:
            del candidate.model_ir.tensors["probe"]
        return raw_results

    monkeypatch.setattr(qkv_attention_orchestration, RAW_OWNER, _run)

    assert qkv_attention_orchestration.run_qkv_attention_summary(
        context,
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
    ) == expected
    assert observed == [
        (context, include_layout_transpose, include_prefix)
    ]
