from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_input_affine_normalization_orchestration,
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
    / "late_input_affine_normalization_orchestration.py"
)
OWNER = "run_late_input_affine_normalization_cleanup"
PASS_OWNERS = (
    "repair_orphan_recurrent_step_tensors_summary",
    "repair_unbound_nonconstant_operator_inputs_with_layout_transpose",
    "optimize_transpose_mul_posttranspose_add_nhwc_chains",
    "run_very_late_gather_constant_normalization_summary",
)
RESULT_TARGETS = (
    "_late_orphan_recurrent_repair_stats",
    "_late_unbound_input_repair_stats",
    "_very_late_affine_post_add_stats",
    "_very_late_normalization_stats",
)
COMPOSITE_TARGET = "_late_input_affine_normalization_results"
EXPECTED_EMPTY_RESULTS = (
    {"repaired_orphan_recurrent_step_tensors": 0},
    {"repaired_unbound_nonconstant_inputs_with_layout_transpose": 0},
    {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 0},
    {
        "optimized_transpose_gather_transpose_axis_remap_nhwc_chains": 0,
        "optimized_constant_input_pad_chains": 0,
        "optimized_constant_input_pool_chains": 0,
        "optimized_constant_input_cast_chains": 0,
        "optimized_redundant_int32_to_int64_passthrough_cast_chains": 0,
        "optimized_redundant_int64_to_int32_cast_chains": 0,
        "optimized_transpose_instancenorm_pad_prepost_nhwc_chains": 0,
        "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 0,
        "pruned_unused_tensors": 0,
    },
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def test_late_input_affine_normalization_current_boundary_and_schemas() -> None:
    lowerer = _lowerer()
    index, invocation = next(
        (index, statement)
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == COMPOSITE_TARGET
    )
    assert _call_name(invocation) == OWNER
    assert _call_name(lowerer.body[index - 1]) == "_advance_post_progress"
    assert (
        _single_target(lowerer.body[index + 1])
        == "_very_late_dynamic_adapter_results"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("late_input_affine_normalization_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        late_input_affine_normalization_orchestration.run_late_input_affine_normalization_cleanup(
            context
        )
        == EXPECTED_EMPTY_RESULTS
    )


def test_late_input_affine_normalization_has_one_context_owner() -> None:
    owner = _functions(OWNER_PATH)[OWNER]
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in PASS_OWNERS
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [call.func.id for call in calls] == list(PASS_OWNERS)
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "context.model_ir"
    ]
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.model_ir"
    ]
    assert [ast.unparse(argument) for argument in calls[2].args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[2].keywords
    } == {"layout_state": "context.layout_state"}
    assert [ast.unparse(argument) for argument in calls[3].args] == ["context"]

    lowerer = _lowerer()
    index, invocation = next(
        (index, statement)
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == COMPOSITE_TARGET
    )
    assert _call_name(invocation) == OWNER
    call = _call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []
    assert _call_name(lowerer.body[index - 1]) == "_advance_post_progress"
    assert (
        _single_target(lowerer.body[index + 1])
        == "_very_late_dynamic_adapter_results"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_late_input_affine_normalization_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_input_affine_normalization_runtime")
    layout_state = LayoutState.from_model_ir(model_ir)
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=[],
    )
    results = tuple({"stage": index} for index in range(4))
    calls: list[tuple[object, ...]] = []

    def recurrent(active_model_ir: ModelIR) -> dict[str, int]:
        calls.append((PASS_OWNERS[0], active_model_ir))
        return results[0]

    def unbound(active_model_ir: ModelIR) -> dict[str, int]:
        calls.append((PASS_OWNERS[1], active_model_ir))
        return results[1]

    def affine(
        active_model_ir: ModelIR,
        *,
        layout_state: LayoutState | None = None,
    ) -> dict[str, int]:
        calls.append((PASS_OWNERS[2], active_model_ir, layout_state))
        return results[2]

    def normalization(active_context: ModelIRPassContext) -> dict[str, int]:
        calls.append((PASS_OWNERS[3], active_context))
        return results[3]

    for name, callback in zip(
        PASS_OWNERS,
        (recurrent, unbound, affine, normalization),
    ):
        monkeypatch.setattr(
            late_input_affine_normalization_orchestration,
            name,
            callback,
        )

    actual = late_input_affine_normalization_orchestration.run_late_input_affine_normalization_cleanup(
        context
    )

    assert actual == results
    assert all(actual[index] is results[index] for index in range(4))
    assert calls == [
        (PASS_OWNERS[0], model_ir),
        (PASS_OWNERS[1], model_ir),
        (PASS_OWNERS[2], model_ir, layout_state),
        (PASS_OWNERS[3], context),
    ]
