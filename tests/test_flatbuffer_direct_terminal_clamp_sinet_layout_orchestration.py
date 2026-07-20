from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_clamp_sinet_layout_orchestration,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
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
RESULT_TARGETS = (
    "_terminal_clamp_unary_relu_results",
    "_terminal_sinet_layout_recovery_results",
)
LOWERER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_singleton_clamp_sinet_orchestration.py"
)
LOWERER_OWNER = "run_terminal_singleton_clamp_sinet_cleanup"
COMPOSITE_TARGET = "_terminal_singleton_clamp_sinet_results"
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
        "sinet_terminal_layout_recovery_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_terminal_singleton": PREDECESSOR_GUARD}

    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    lowerer_owner = _functions(LOWERER_OWNER_PATH)[LOWERER_OWNER]
    owner_call = next(
        node
        for node in ast.walk(lowerer_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    )
    assert [ast.unparse(argument) for argument in owner_call.args] == ["context"]
    assert owner_call.keywords == []

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
        terminal_clamp_sinet_layout_orchestration.run_terminal_clamp_sinet_layout_cleanup(
            sinet_context
        )
    )
    assert tuple(type(result) for result in results) == (tuple, tuple)
    assert tuple(len(result) for result in results) == (3, 3)
    assert tuple(type(result) for result in results[0]) == (dict,) * 3
    assert tuple(type(result) for result in results[1]) == (dict, tuple, dict)


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
    assert _call_name(assignment) == LOWERER_OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "sinet_terminal_layout_recovery_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_terminal_singleton": PREDECESSOR_GUARD}
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    lowerer_owner = _functions(LOWERER_OWNER_PATH)[LOWERER_OWNER]
    owner_calls = [
        node
        for node in ast.walk(lowerer_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert len(owner_calls) == 1
    assert [ast.unparse(argument) for argument in owner_calls[0].args] == [
        "context"
    ]
    assert owner_calls[0].keywords == []


def test_terminal_clamp_sinet_layout_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("terminal_clamp_sinet_layout_runtime")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    def callback() -> tuple[()]:
        return ()

    context = SINetTerminalLayoutRecoveryContext(
        pass_context=pass_context,
        preadd_resize_recovery=callback,
    )
    expected_results = (
        tuple({f"clamp_{index}": index} for index in range(3)),
        tuple({f"sinet_{index}": index} for index in range(3)),
    )
    observed: list[tuple[str, object]] = []

    def _clamp(
        active_context: ModelIRPassContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[0], active_context))
        return expected_results[0]

    def _sinet(
        active_context: SINetTerminalLayoutRecoveryContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[1], active_context))
        return expected_results[1]

    monkeypatch.setattr(
        terminal_clamp_sinet_layout_orchestration,
        CHILD_OWNERS[0],
        _clamp,
    )
    monkeypatch.setattr(
        terminal_clamp_sinet_layout_orchestration,
        CHILD_OWNERS[1],
        _sinet,
    )

    actual = (
        terminal_clamp_sinet_layout_orchestration.run_terminal_clamp_sinet_layout_cleanup(
            context
        )
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], pass_context),
        (CHILD_OWNERS[1], context),
    ]
    assert context.preadd_resize_recovery is callback
