from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.late_affine_optional_fanout_orchestration import (
    run_late_affine_optional_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_final_shape_boundary_orchestration import (
    LateFinalShapeBoundaryContext,
)
from onnx2tf.tflite_builder.passes.late_final_shape_terminal_fanout_orchestration import (
    run_late_final_shape_terminal_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
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
    / "late_affine_final_shape_terminal_orchestration.py"
)
OWNER = "run_late_affine_final_shape_terminal_cleanup"
CHILD_OWNERS = (
    "run_late_affine_optional_fanout_cleanup",
    "run_late_final_shape_terminal_fanout_cleanup",
)
RESULT_TARGETS = (
    "_late_affine_optional_fanout_results",
    "_late_final_shape_terminal_fanout_results",
)
COMPOSITE_TARGET = "_late_affine_final_shape_terminal_results"
PREDECESSOR_PHASE_ID = "cleanup.late.ndhwc_cost_volume"
SUCCESSOR_TARGET = "_terminal_convpool_output_passthrough_stats"
SUCCESSOR_OWNER = (
    "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
)
GUARD = "optimize_layout_transpose_chains"
CONTEXT_TARGET = "late_final_shape_boundary_context"

AFFINE_SCHEMA = (
    "folded_conv_mul_add_affine_chains",
    "folded_conv_add_only_affine_chains",
    "folded_conv_mul_only_affine_chains",
    "folded_conv_mul_add_only_affine_chains",
)
CONCAT_SCHEMA = (
    ("optimized_transpose_axis3_const_concat_bridge_nhwc_chains",),
    ("optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains",),
    (
        "optimized_transpose_layernorm_stats_nhwc_propagation_chains",
        "optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains",
    ),
    (
        "iterations",
        "removed_identity_transpose",
        "removed_inverse_transpose_pairs",
        "removed_inverse_transpose_fanout_branches",
        "composed_consecutive_transpose_pairs",
    ),
)
CONVERGENCE_SCHEMA = (
    "removed_dead_operators",
    "resolved_dynamic_reshape_shapes",
    "reconciled_static_tensor_shapes",
    "sanitized_hardswish_tensor_shapes",
    "fused_conv_activation_chains",
    "fused_add_activation_chains",
    "fused_sub_activation_chains",
    "fused_mul_activation_chains",
    "fused_div_activation_chains",
    "fused_binary_activation_chains",
    "fused_activation_chains_total",
)
SINGLETON_SCHEMA = (
    (
        "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains",
        "rewritten_singleton_nms_maxpool_nhwc_chains",
    ),
    (
        "removed_noop_reshape_chains",
        "rewritten_consecutive_reshape_passthrough_chains",
        "rewritten_fanout_bypass_reshape_passthrough_chains",
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


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr, ast.Return)):
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


def _phase_id(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or ast.unparse(call.func) != "session.record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _context(name: str) -> LateFinalShapeBoundaryContext:
    model_ir = ModelIR(name)
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    terminal_context = TerminalSliceConcatRecoveryContext(
        pass_context=pass_context,
        channel_slice_pad_mul_cluster=lambda: (),
    )
    return LateFinalShapeBoundaryContext(
        pass_context=pass_context,
        terminal_slice_concat_context=terminal_context,
    )


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_late_affine_final_shape_terminal_current_contract() -> None:
    lowerer = _lowerer()
    affine_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGETS[0]
    )
    index = lowerer.body.index(affine_assignment)
    assert _call_name(affine_assignment) == CHILD_OWNERS[0]
    affine_call = _call(affine_assignment)
    assert affine_call is not None
    assert [ast.unparse(argument) for argument in affine_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in affine_call.keywords
    } == {"include_elementwise_fanout": GUARD}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID

    final_assignment = lowerer.body[index + 1]
    assert _single_target(final_assignment) == RESULT_TARGETS[1]
    assert _call_name(final_assignment) == CHILD_OWNERS[1]
    final_call = _call(final_assignment)
    assert final_call is not None
    assert [ast.unparse(argument) for argument in final_call.args] == [
        CONTEXT_TARGET
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in final_call.keywords
    } == {"include_elementwise_fanout": GUARD}

    successor_guard = lowerer.body[index + 2]
    assert isinstance(successor_guard, ast.If)
    assert ast.unparse(successor_guard.test) == GUARD
    assert _single_target(successor_guard.body[0]) == SUCCESSOR_TARGET
    assert _call_name(successor_guard.body[0]) == SUCCESSOR_OWNER
    assert len(successor_guard.orelse) == 1
    assert isinstance(successor_guard.orelse[0], ast.If)
    assert ast.unparse(successor_guard.orelse[0].test) == (
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    context_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == CONTEXT_TARGET
    )
    assert isinstance(context_assignment.value, ast.Call)
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in context_assignment.value.keywords
    }["pass_context"] == "shared_model_ir_pass_context"


@pytest.mark.parametrize("include_elementwise_fanout", [False, True])
def test_late_affine_final_shape_terminal_child_schemas_and_contexts(
    include_elementwise_fanout: bool,
) -> None:
    context = _context(
        f"late_affine_final_shape_terminal_{include_elementwise_fanout}"
    )
    affine_results = run_late_affine_optional_fanout_cleanup(
        context.pass_context,
        include_elementwise_fanout=include_elementwise_fanout,
    )
    final_results = run_late_final_shape_terminal_fanout_cleanup(
        context,
        include_elementwise_fanout=include_elementwise_fanout,
    )

    assert tuple(affine_results[0][0]) == AFFINE_SCHEMA
    assert _dict_schema(affine_results[0][1]) == CONCAT_SCHEMA
    if include_elementwise_fanout:
        assert tuple(affine_results[1] or ()) == (
            "optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
        )
    else:
        assert affine_results[1] is None

    late_results, terminal_results = final_results
    assert tuple(len(result) for result in late_results[0]) == (3, 2, 4, 2)
    assert tuple(late_results[1]) == CONVERGENCE_SCHEMA
    assert tuple(len(result) for result in late_results[2]) == (3, 14, 2, 6)
    if include_elementwise_fanout:
        assert tuple(terminal_results[0] or ()) == (
            "optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
        )
    else:
        assert terminal_results[0] is None
    assert _dict_schema(terminal_results[1]) == SINGLETON_SCHEMA
    assert context.terminal_slice_concat_context.pass_context is (
        context.pass_context
    )


@pytest.mark.xfail(
    strict=True,
    reason="late affine/final-shape/terminal owner is not implemented",
)
def test_late_affine_final_shape_terminal_has_one_context_owner() -> None:
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
        "context.pass_context"
    ]
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context"
    ]
    assert all(
        {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        }
        == {"include_elementwise_fanout": "include_elementwise_fanout"}
        for call in calls
    )

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
        CONTEXT_TARGET
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_elementwise_fanout": GUARD}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    successor_guard = lowerer.body[index + 1]
    assert isinstance(successor_guard, ast.If)
    assert ast.unparse(successor_guard.test) == GUARD
    assert _single_target(successor_guard.body[0]) == SUCCESSOR_TARGET
    assert _call_name(successor_guard.body[0]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )
