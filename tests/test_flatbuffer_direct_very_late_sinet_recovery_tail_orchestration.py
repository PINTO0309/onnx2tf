from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.sinet_preadd_resize_recovery_orchestration import (
    run_sinet_preadd_resize_recovery,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes import (
    very_late_sinet_recovery_tail_orchestration,
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
    / "very_late_sinet_recovery_tail_orchestration.py"
)
OWNER = "run_very_late_sinet_recovery_tail_cleanup"
CHILD_OWNERS = (
    "run_sinet_terminal_layout_recovery",
    "preadd_resize_recovery",
)
RESULT_TARGETS = (
    "_very_late_sinet_layout_recovery_results",
    "_very_late_sinet_preadd_resize_results",
)
CURRENT_TARGET = "_very_late_sinet_recovery_tail_results"
PREDECESSOR_PHASE_ID = "shape_topology.terminal.indexed_convergence"
SUCCESSOR_PHASE_ID = "cleanup.very_late.residual_affine_prelu"
SUCCESSOR_OWNER_EXPRESSION = (
    "run_very_late_sinet_residual_affine_prelu_cleanup("
    "sinet_terminal_layout_recovery_context)[1]"
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


def test_very_late_sinet_recovery_tail_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    record = next(
        statement
        for statement in lowerer.body
        if _phase_id(statement) == SUCCESSOR_PHASE_ID
    )
    index = lowerer.body.index(record)
    call = _call(record)
    assert call is not None
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert ast.unparse(call.args[1]) == SUCCESSOR_OWNER_EXPRESSION
    assert _phase_id(lowerer.body[index + 1]) == (
        "cleanup.very_late.residual_affine_fanout"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
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

    model_ir = ModelIR("very_late_sinet_recovery_tail_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )

    def preadd_resize_recovery() -> tuple[dict[str, int], ...]:
        return run_sinet_preadd_resize_recovery(pass_context)

    context = SINetTerminalLayoutRecoveryContext(
        pass_context=pass_context,
        preadd_resize_recovery=preadd_resize_recovery,
    )
    results = (
        very_late_sinet_recovery_tail_orchestration.run_very_late_sinet_recovery_tail_cleanup(
            context
        )
    )
    assert tuple(type(result) for result in results) == (tuple, tuple)
    assert tuple(len(result) for result in results) == (3, 6)
    assert tuple(type(result) for result in results[0]) == (dict, tuple, dict)
    assert tuple(type(result) for result in results[0][1]) == (dict,) * 6
    assert tuple(type(result) for result in results[1]) == (dict,) * 6


def test_very_late_sinet_recovery_tail_has_one_context_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id == CHILD_OWNERS[0]
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and ast.unparse(node.func) == "context.preadd_resize_recovery"
                )
            )
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [ast.unparse(call.func) for call in calls] == [
        CHILD_OWNERS[0],
        "context.preadd_resize_recovery",
    ]
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[1].args == []
    assert all(call.keywords == [] for call in calls)

    lowerer = _lowerer()
    record = next(
        statement
        for statement in lowerer.body
        if _phase_id(statement) == SUCCESSOR_PHASE_ID
    )
    index = lowerer.body.index(record)
    call = _call(record)
    assert call is not None
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert ast.unparse(call.args[1]) == SUCCESSOR_OWNER_EXPRESSION
    assert _phase_id(lowerer.body[index + 1]) == (
        "cleanup.very_late.residual_affine_fanout"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_very_late_sinet_recovery_tail_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("very_late_sinet_recovery_tail_runtime")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = (
        tuple({f"terminal_{index}": index} for index in range(3)),
        tuple({f"preadd_{index}": index} for index in range(6)),
    )
    observed: list[tuple[str, object]] = []

    def preadd_resize_recovery() -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[1], pass_context))
        return expected_results[1]

    context = SINetTerminalLayoutRecoveryContext(
        pass_context=pass_context,
        preadd_resize_recovery=preadd_resize_recovery,
    )

    def terminal_layout_recovery(
        active_context: SINetTerminalLayoutRecoveryContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[0], active_context))
        return expected_results[0]

    monkeypatch.setattr(
        very_late_sinet_recovery_tail_orchestration,
        CHILD_OWNERS[0],
        terminal_layout_recovery,
    )

    actual = (
        very_late_sinet_recovery_tail_orchestration.run_very_late_sinet_recovery_tail_cleanup(
            context
        )
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], context),
        (CHILD_OWNERS[1], pass_context),
    ]
    assert context.preadd_resize_recovery is preadd_resize_recovery
