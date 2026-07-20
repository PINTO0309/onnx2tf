from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_affine_concat_orchestration,
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
    / "late_affine_concat_orchestration.py"
)
OWNER = "run_late_affine_concat_cleanup"
CHILD_OWNERS = (
    "optimize_fold_conv_mul_add_affine_chains",
    "run_late_concat_layout_cleanup",
)
RESULT_TARGETS = (
    "_late_cost_volume_conv_affine_stats",
    "_late_concat_layout_results",
)
LOWERER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_affine_optional_fanout_orchestration.py"
)
LOWERER_OWNER = "run_late_affine_optional_fanout_cleanup"
COMPOSITE_TARGET = "_late_affine_optional_fanout_results"
PREDECESSOR_PHASE_ID = "cleanup.late.ndhwc_cost_volume"
SUCCESSOR_TARGET = "_late_final_shape_terminal_fanout_results"
SUCCESSOR_OWNER = "run_late_final_shape_terminal_fanout_cleanup"
EXPECTED_SCHEMAS = (
    {
        "folded_conv_mul_add_affine_chains": 0,
        "folded_conv_add_only_affine_chains": 0,
        "folded_conv_mul_only_affine_chains": 0,
        "folded_conv_mul_add_only_affine_chains": 0,
    },
    (
        {"optimized_transpose_axis3_const_concat_bridge_nhwc_chains": 0},
        {"optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains": 0},
        {
            "optimized_transpose_layernorm_stats_nhwc_propagation_chains": 0,
            "optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains": 0,
        },
        {
            "iterations": 0,
            "removed_identity_transpose": 0,
            "removed_inverse_transpose_pairs": 0,
            "removed_inverse_transpose_fanout_branches": 0,
            "composed_consecutive_transpose_pairs": 0,
        },
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


def test_late_affine_concat_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == LOWERER_OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_elementwise_fanout": "optimize_layout_transpose_chains"}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    successor = lowerer.body[index + 1]
    assert _single_target(successor) == SUCCESSOR_TARGET
    assert _call_name(successor) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    lowerer_owner = _functions(LOWERER_OWNER_PATH)[LOWERER_OWNER]
    child_calls = [
        node
        for node in ast.walk(lowerer_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert len(child_calls) == 1
    assert [ast.unparse(argument) for argument in child_calls[0].args] == [
        "context"
    ]

    model_ir = ModelIR("late_affine_concat_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        late_affine_concat_orchestration.run_late_affine_concat_cleanup(
            context
        )
        == EXPECTED_SCHEMAS
    )


def test_late_affine_concat_has_one_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
    } == {
        "enable_conv_add_only_fold": "True",
        "layout_state": "context.layout_state",
    }
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
    assert calls[1].keywords == []

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == LOWERER_OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_elementwise_fanout": "optimize_layout_transpose_chains"}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    successor = lowerer.body[index + 1]
    assert _single_target(successor) == SUCCESSOR_TARGET
    assert _call_name(successor) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    lowerer_owner = _functions(LOWERER_OWNER_PATH)[LOWERER_OWNER]
    child_calls = [
        node
        for node in ast.walk(lowerer_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert len(child_calls) == 1


def test_late_affine_concat_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_affine_concat_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    affine_result = {"affine": 1}
    concat_result = tuple({f"concat_{index}": index} for index in range(4))
    observed: list[tuple[str, object, dict[str, object]]] = []

    def _affine(
        active_model_ir: ModelIR,
        **kwargs: object,
    ) -> dict[str, int]:
        observed.append((CHILD_OWNERS[0], active_model_ir, kwargs))
        return affine_result

    def _concat(
        active_context: ModelIRPassContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[1], active_context, {}))
        return concat_result

    monkeypatch.setattr(
        late_affine_concat_orchestration,
        CHILD_OWNERS[0],
        _affine,
    )
    monkeypatch.setattr(
        late_affine_concat_orchestration,
        CHILD_OWNERS[1],
        _concat,
    )

    actual = late_affine_concat_orchestration.run_late_affine_concat_cleanup(
        context
    )
    assert actual == (affine_result, concat_result)
    assert actual[0] is affine_result
    assert actual[1] is concat_result
    assert observed == [
        (
            CHILD_OWNERS[0],
            context.model_ir,
            {
                "enable_conv_add_only_fold": True,
                "layout_state": context.layout_state,
            },
        ),
        (CHILD_OWNERS[1], context, {}),
    ]
