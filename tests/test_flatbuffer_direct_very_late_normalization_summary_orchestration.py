from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import (
    very_late_gather_constant_normalization_orchestration,
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
    / "very_late_gather_constant_normalization_orchestration.py"
)
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_input_affine_normalization_orchestration.py"
)
COMPOSITE_OWNER = "run_late_input_affine_normalization_cleanup"
FINAL_COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_input_dynamic_orchestration.py"
)
FINAL_COMPOSITE_OWNER = "run_final_input_dynamic_cleanup"
FINAL_COMPOSITE_TARGET = "_final_input_dynamic_results"
FINAL_SHAPE_OWNER_EXPRESSION = (
    "run_final_input_dynamic_shape_cleanup("
    "shared_model_ir_pass_context, "
    "shape_reconciler=_reconcile_static_tensor_shapes)[1]"
)
RAW_WRAPPER = "_run_very_late_gather_constant_normalization_pass_cluster"
RAW_OWNER = "run_very_late_gather_constant_normalization"
SUMMARY_OWNER = "run_very_late_gather_constant_normalization_summary"
SUMMARY_FUNCTION = "summarize_very_late_gather_constant_normalization_mutations"
COUNT_TARGET = "very_late_normalization_tensor_count"
RAW_TARGET = "very_late_normalization_results"
SUMMARY_TARGET = "_very_late_normalization_stats"
SUCCESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_final"


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


def _final_composite_calls() -> list[ast.Call]:
    owner = _functions(FINAL_COMPOSITE_PATH)[FINAL_COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == COMPOSITE_OWNER
    ]


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(
        statement.value, ast.Call
    ):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_very_late_normalization_prune_aware_summary_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    record = next(
        statement
        for statement in lowerer.body
        if _phase_id(statement) == SUCCESSOR_PHASE_ID
    )
    index = lowerer.body.index(record)
    assert isinstance(record, ast.Expr)
    assert isinstance(record.value, ast.Call)
    assert ast.unparse(record.value.args[1]) == FINAL_SHAPE_OWNER_EXPRESSION
    assert ast.unparse(lowerer.body[index - 1]) == "_advance_post_progress()"
    assert _single_target(lowerer.body[index + 1]) == "split_fallback_stats"
    assert len(_final_composite_calls()) == 1
    summary_calls = _composite_summary_calls()
    assert len(summary_calls) == 1
    assert [ast.unparse(argument) for argument in summary_calls[0].args] == [
        "context"
    ]
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
    assert not any(
        isinstance(node, ast.Name) and node.id == FINAL_COMPOSITE_TARGET
        for node in ast.walk(lowerer)
    )


def test_very_late_normalization_uses_one_prune_aware_summary_owner() -> None:
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
    record = next(
        statement
        for statement in lowerer.body
        if _phase_id(statement) == SUCCESSOR_PHASE_ID
    )
    index = lowerer.body.index(record)
    assert isinstance(record, ast.Expr)
    assert isinstance(record.value, ast.Call)
    assert ast.unparse(record.value.args[1]) == FINAL_SHAPE_OWNER_EXPRESSION
    assert ast.unparse(lowerer.body[index - 1]) == "_advance_post_progress()"
    assert _single_target(lowerer.body[index + 1]) == "split_fallback_stats"
    assert len(_final_composite_calls()) == 1
    assert len(_composite_summary_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in {COUNT_TARGET, RAW_TARGET}
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == FINAL_COMPOSITE_TARGET
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


@pytest.mark.parametrize("prune", (False, True))
def test_very_late_normalization_summary_preserves_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    prune: bool,
) -> None:
    model_ir = ModelIR("very_late_normalization_summary")
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
        {"optimized_transpose_gather_transpose_axis_remap_nhwc_chains": 1},
        {
            "optimized_constant_input_pad_chains": 2,
            "optimized_constant_input_pool_chains": 3,
            "optimized_constant_input_cast_chains": 4,
        },
        {
            "optimized_redundant_int32_to_int64_passthrough_cast_chains": 5,
            "optimized_redundant_int64_to_int32_cast_chains": 6,
        },
        {
            "optimized_transpose_instancenorm_pad_prepost_nhwc_chains": 7,
            "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 8,
        },
    )
    expected = (
        very_late_gather_constant_normalization_orchestration
        .summarize_very_late_gather_constant_normalization_mutations(
            raw_results,
            pruned_unused_tensors=int(prune),
        )
    )
    observed: list[ModelIRPassContext] = []

    def _run(
        candidate: ModelIRPassContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append(candidate)
        if prune:
            del candidate.model_ir.tensors["probe"]
        return raw_results

    monkeypatch.setattr(
        very_late_gather_constant_normalization_orchestration,
        RAW_OWNER,
        _run,
    )

    assert (
        very_late_gather_constant_normalization_orchestration
        .run_very_late_gather_constant_normalization_summary(context)
        == expected
    )
    assert observed == [context]
