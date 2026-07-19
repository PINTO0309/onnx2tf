from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _run_indexed_final_shape_activation_convergence,
)
from onnx2tf.tflite_builder.passes.final_boundary_slice_concat_orchestration import (
    run_final_boundary_slice_concat_cleanup,
)
from onnx2tf.tflite_builder.passes.late_reshape_shuffle_attention_window_orchestration import (
    run_late_reshape_shuffle_attention_window_cleanup,
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
    / "late_final_shape_boundary_orchestration.py"
)
CONVERGENCE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "indexed_final_shape_activation_convergence.py"
)
OWNER = "run_late_final_shape_boundary_cleanup"
CONVERGENCE_OWNER = "run_indexed_final_shape_activation_convergence"
CHILD_OWNERS = (
    "run_late_reshape_shuffle_attention_window_cleanup",
    CONVERGENCE_OWNER,
    "run_final_boundary_slice_concat_cleanup",
)
RESULT_TARGETS = (
    "_late_reshape_shuffle_attention_window_results",
    "_late_final_shape_activation_convergence_stats",
    "_final_boundary_slice_concat_results",
)
COMPOSITE_TARGET = "_late_final_shape_boundary_results"
PREDECESSOR_GUARD = "optimize_layout_transpose_chains"
PREDECESSOR_TARGET = "_late_concat_elementwise_fanout_stats"
SUCCESSOR_GUARD = "optimize_layout_transpose_chains"
SUCCESSOR_TARGET = "_terminal_elementwise_fanout_stats"
CONVERGENCE_KEYS = (
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


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


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


def test_late_final_shape_boundary_current_order_context_and_schema() -> None:
    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) in RESULT_TARGETS
    ]
    assert [_single_target(statement) for statement in assignments] == list(
        RESULT_TARGETS
    )
    assert [_call_name(statement) for statement in assignments] == [
        CHILD_OWNERS[0],
        "_run_indexed_final_shape_activation_convergence",
        CHILD_OWNERS[2],
    ]
    indices = [lowerer.body.index(statement) for statement in assignments]
    assert indices == list(range(indices[0], indices[0] + 3))

    late_call = _call(assignments[0])
    convergence_call = _call(assignments[1])
    final_call = _call(assignments[2])
    assert late_call is not None
    assert convergence_call is not None
    assert final_call is not None
    assert [ast.unparse(argument) for argument in late_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert late_call.keywords == []
    assert [ast.unparse(argument) for argument in convergence_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in convergence_call.keywords
    } == {"layout_state": "session.layout_state"}
    assert [ast.unparse(argument) for argument in final_call.args] == [
        "terminal_slice_concat_recovery_context"
    ]
    assert final_call.keywords == []

    predecessor = lowerer.body[indices[0] - 1]
    successor = lowerer.body[indices[-1] + 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert [_single_target(statement) for statement in predecessor.body] == [
        PREDECESSOR_TARGET
    ]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == SUCCESSOR_GUARD
    assert [_single_target(statement) for statement in successor.body] == [
        SUCCESSOR_TARGET
    ]

    shared_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == "shared_model_ir_pass_context"
    )
    terminal_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement)
        == "terminal_slice_concat_recovery_context"
    )
    assert ast.unparse(shared_assignment.value) == "session.model_ir_pass_context"
    assert isinstance(terminal_assignment.value, ast.Call)
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in terminal_assignment.value.keywords
    }["pass_context"] == "session.model_ir_pass_context"

    model_ir = ModelIR("late_final_shape_boundary_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    recovery_context = TerminalSliceConcatRecoveryContext(
        pass_context=pass_context,
        channel_slice_pad_mul_cluster=lambda: (),
    )
    results = (
        run_late_reshape_shuffle_attention_window_cleanup(pass_context),
        _run_indexed_final_shape_activation_convergence(
            model_ir,
            layout_state=pass_context.layout_state,
        ),
        run_final_boundary_slice_concat_cleanup(recovery_context),
    )
    assert tuple(type(result) for result in results) == (tuple, dict, tuple)
    assert tuple(len(result) for result in results[0]) == (3, 2, 4, 2)
    assert tuple(results[1]) == CONVERGENCE_KEYS
    assert tuple(len(result) for result in results[2]) == (3, 14, 2, 6)


@pytest.mark.xfail(
    strict=True,
    reason="late final shape/boundary composite owner is not implemented",
)
def test_late_final_shape_boundary_has_one_context_owner() -> None:
    assert OWNER_PATH.exists()
    assert CONVERGENCE_OWNER_PATH.exists()
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
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.pass_context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {"layout_state": "context.pass_context.layout_state"}
    assert [ast.unparse(argument) for argument in calls[2].args] == [
        "context.terminal_slice_concat_context"
    ]
    assert calls[2].keywords == []

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
        "late_final_shape_boundary_context"
    ]
    assert call.keywords == []
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert isinstance(lowerer.body[index + 1], ast.If)
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
