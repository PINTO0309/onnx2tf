from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
    run_sinet_terminal_layout_recovery,
)
from onnx2tf.tflite_builder.passes.terminal_clamp_unary_relu_orchestration import (
    run_terminal_clamp_unary_relu,
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
    / "terminal_clamp_sinet_layout_orchestration.py"
)
OWNER = "run_terminal_clamp_sinet_layout_cleanup"
CHILD_OWNERS = (
    "run_terminal_clamp_unary_relu",
    "run_sinet_terminal_layout_recovery",
)
CURRENT_CHILD_OWNERS = (
    "_run_terminal_clamp_unary_relu_pass_cluster",
    "_run_sinet_terminal_layout_recovery_sequence",
)
RESULT_TARGETS = (
    "_terminal_clamp_unary_relu_results",
    "_terminal_sinet_layout_recovery_results",
)
COMPOSITE_TARGET = "_terminal_clamp_sinet_layout_results"
PREDECESSOR_GUARD = "optimize_layout_transpose_chains"
SUCCESSOR_PHASE_ID = "cleanup.terminal.sinet_hardswish_se"


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


def test_terminal_clamp_sinet_layout_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) in RESULT_TARGETS
    ]
    assert [_single_target(statement) for statement in assignments] == list(
        RESULT_TARGETS
    )
    assert [_call_name(statement) for statement in assignments] == list(
        CURRENT_CHILD_OWNERS
    )
    indices = [lowerer.body.index(statement) for statement in assignments]
    assert indices[1] == indices[0] + 1
    assert all(_call(statement).args == [] for statement in assignments)
    assert all(_call(statement).keywords == [] for statement in assignments)

    predecessor = lowerer.body[indices[0] - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[indices[1] + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    context_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == "sinet_terminal_layout_recovery_context"
    )
    context_call = _call(context_assignment)
    assert context_call is not None
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in context_call.keywords
    } == {
        "pass_context": "session.model_ir_pass_context",
        "preadd_resize_recovery": "_run_sinet_preadd_resize_recovery_sequence",
    }

    model_ir = ModelIR("terminal_clamp_sinet_layout_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    sinet_context = SINetTerminalLayoutRecoveryContext(
        pass_context=pass_context,
        preadd_resize_recovery=lambda: (),
    )
    results = (
        run_terminal_clamp_unary_relu(pass_context),
        run_sinet_terminal_layout_recovery(sinet_context),
    )
    assert tuple(type(result) for result in results) == (tuple, tuple)
    assert tuple(len(result) for result in results) == (3, 3)
    assert tuple(type(result) for result in results[0]) == (dict,) * 3
    assert tuple(type(result) for result in results[1]) == (dict, tuple, dict)


@pytest.mark.xfail(
    strict=True,
    reason="terminal clamp/SiNet-layout composite owner is not implemented",
)
def test_terminal_clamp_sinet_layout_has_one_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
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
        "sinet_terminal_layout_recovery_context"
    ]
    assert call.keywords == []
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
