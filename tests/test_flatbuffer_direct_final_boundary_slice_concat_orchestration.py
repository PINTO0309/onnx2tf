from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.final_boundary_channel_layout_orchestration import (
    run_final_boundary_channel_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.final_slice_pre_concat_layout_orchestration import (
    run_final_slice_pre_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_concat_bridge_layout_orchestration import (
    run_terminal_concat_bridge_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
    run_terminal_slice_concat_recovery,
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
    / "final_boundary_slice_concat_orchestration.py"
)
OWNER = "run_final_boundary_slice_concat_cleanup"
CHILD_OWNERS = (
    "run_final_boundary_channel_layout_cleanup",
    "run_terminal_slice_concat_recovery",
    "run_final_slice_pre_concat_layout_cleanup",
    "run_terminal_concat_bridge_layout_cleanup",
)
WRAPPER = "_run_terminal_slice_concat_layout_recovery_sequence"
CURRENT_CHILD_OWNERS = (
    CHILD_OWNERS[0],
    WRAPPER,
    CHILD_OWNERS[2],
    CHILD_OWNERS[3],
)
RESULT_TARGETS = (
    "_final_boundary_channel_layout_results",
    "_final_slice_concat_recovery_results",
    "_final_slice_pre_concat_layout_results",
    "_terminal_concat_bridge_layout_results",
)
COMPOSITE_TARGET = "_final_boundary_slice_concat_results"
PREDECESSOR_TARGET = "_late_final_shape_activation_convergence_stats"
SUCCESSOR_GUARD = "optimize_layout_transpose_chains"
SUCCESSOR_TARGET = "_terminal_elementwise_fanout_stats"


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


def test_final_boundary_slice_concat_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    indices = tuple(
        next(
            index
            for index, statement in enumerate(lowerer.body)
            if _single_target(statement) == target
        )
        for target in RESULT_TARGETS
    )
    assert indices == tuple(range(indices[0], indices[0] + len(indices)))
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    successor = lowerer.body[indices[-1] + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == SUCCESSOR_GUARD
    assert [_single_target(statement) for statement in successor.body] == [
        SUCCESSOR_TARGET
    ]
    assert tuple(_call_name(lowerer.body[index]) for index in indices) == (
        CURRENT_CHILD_OWNERS
    )
    calls = tuple(_call(lowerer.body[index]) for index in indices)
    assert all(call is not None for call in calls)
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "shared_model_ir_pass_context"
    ]
    assert calls[0].keywords == []
    assert calls[1].args == []
    assert calls[1].keywords == []
    for call in calls[2:]:
        assert [ast.unparse(argument) for argument in call.args] == [
            "shared_model_ir_pass_context"
        ]
        assert call.keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("final_boundary_slice_concat_schema")
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
        run_final_boundary_channel_layout_cleanup(pass_context),
        run_terminal_slice_concat_recovery(recovery_context),
        run_final_slice_pre_concat_layout_cleanup(pass_context),
        run_terminal_concat_bridge_layout_cleanup(pass_context),
    )
    assert tuple(type(result) for result in results) == (tuple,) * 4
    assert tuple(len(result) for result in results) == (3, 14, 2, 6)
    assert tuple(type(result) for result in results[1]) == (
        tuple,
        *((dict,) * 13),
    )


def test_terminal_slice_concat_wrapper_independent_route_is_fixed() -> None:
    lowerer = _lowerer()
    calls = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == WRAPPER
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert len(calls) == 2
    assert all(call.args == [] and call.keywords == [] for call in calls)
    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == WRAPPER
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    call = wrapper.body[0].value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == CHILD_OWNERS[1]
    assert [ast.unparse(argument) for argument in call.args] == [
        "terminal_slice_concat_recovery_context"
    ]
    assert call.keywords == []


@pytest.mark.xfail(
    strict=True,
    reason="final boundary/Slice/Concat sequence has no composite owner",
)
def test_final_boundary_slice_concat_has_one_context_owner() -> None:
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
        [ast.unparse(argument) for argument in call.args]
        == ["context.pass_context"]
        for call in calls[2:]
    )
    assert all(call.keywords == [] for call in calls)

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
        "terminal_slice_concat_recovery_context"
    ]
    assert call.keywords == []
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == SUCCESSOR_GUARD
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
