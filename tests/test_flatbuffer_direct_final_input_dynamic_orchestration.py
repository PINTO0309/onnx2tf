from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    final_input_dynamic_orchestration,
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
    / "final_input_dynamic_orchestration.py"
)
OWNER = "run_final_input_dynamic_cleanup"
CHILD_OWNERS = (
    "run_late_input_affine_normalization_cleanup",
    "run_very_late_dynamic_adapter_cleanup",
)
RESULT_TARGETS = (
    "_late_input_affine_normalization_results",
    "_very_late_dynamic_adapter_results",
)
COMPOSITE_TARGET = "_final_input_dynamic_results"
SUCCESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_final"
EXPECTED_SCHEMAS = (
    (
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
    ),
    (
        {"resolved_dynamic_reshape_shapes": 0},
        {
            "repaired_singleton_nhwc_conv_input_reshapes": 0,
            "repaired_stale_nchw_to_nhwc_conv_input_transposes": 0,
            "pruned_unused_tensors": 0,
        },
        {"repaired_nchw_channel_shuffle_concat_gathers": 0},
        {"repaired_nchw_concat_transpose_conv_axes": 0},
        {"repaired_nchw_concat_global_pool_conv_axes": 0},
        {"rewritten_dynamic_rank1_unsqueeze_reshape_shape_inputs": 0},
    ),
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


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _phase_id(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_final_input_dynamic_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []
    assert _call_name(lowerer.body[index - 1]) == "_advance_post_progress"
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert _single_target(lowerer.body[index + 2]) == "split_fallback_stats"
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("final_input_dynamic_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        final_input_dynamic_orchestration.run_final_input_dynamic_cleanup(
            context
        )
        == EXPECTED_SCHEMAS
    )


def test_final_input_dynamic_has_one_context_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in CHILD_OWNERS
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [call.func.id for call in calls] == list(CHILD_OWNERS)
    for call in calls:
        assert [ast.unparse(argument) for argument in call.args] == ["context"]
        assert call.keywords == []

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []
    assert _call_name(lowerer.body[index - 1]) == "_advance_post_progress"
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert _single_target(lowerer.body[index + 2]) == "split_fallback_stats"
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_final_input_dynamic_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("final_input_dynamic_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = (
        tuple({f"input_stage_{index}": index} for index in range(4)),
        tuple({f"dynamic_stage_{index}": index} for index in range(6)),
    )
    observed: list[tuple[str, ModelIRPassContext]] = []

    def _input(
        active_context: ModelIRPassContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[0], active_context))
        return expected_results[0]

    def _dynamic(
        active_context: ModelIRPassContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[1], active_context))
        return expected_results[1]

    monkeypatch.setattr(
        final_input_dynamic_orchestration,
        CHILD_OWNERS[0],
        _input,
    )
    monkeypatch.setattr(
        final_input_dynamic_orchestration,
        CHILD_OWNERS[1],
        _dynamic,
    )

    actual = final_input_dynamic_orchestration.run_final_input_dynamic_cleanup(
        context
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], context),
        (CHILD_OWNERS[1], context),
    ]
