from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.indexed_final_shape_activation_convergence import (
    run_indexed_shape_convergence_cleanup,
)
from onnx2tf.tflite_builder.passes import (
    terminal_sinet_singleton_reshape_convergence_orchestration,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes.terminal_sinet_singleton_reshape_orchestration import (
    run_terminal_sinet_singleton_reshape_cleanup,
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
    / "terminal_sinet_singleton_reshape_convergence_orchestration.py"
)
OWNER = "run_terminal_sinet_singleton_reshape_convergence_cleanup"
CHILD_OWNERS = (
    "run_terminal_sinet_singleton_reshape_cleanup",
    "run_indexed_shape_convergence_cleanup",
)
CURRENT_TARGET = "_terminal_sinet_singleton_reshape_results"
CURRENT_INDEXED_WRAPPER = "_run_indexed_shape_convergence_cleanup"
PHASE_ID = "shape_topology.terminal.indexed_convergence"
PREDECESSOR_PHASE_ID = "cleanup.terminal.dequant_hardsigmoid_bridge"
SUCCESSOR_OWNER = "run_very_late_sinet_recovery_tail_cleanup"
SUCCESSOR_TARGET = "_very_late_sinet_recovery_tail_results"
FUTURE_OWNER_EXPRESSION = (
    "run_terminal_sinet_singleton_reshape_convergence_cleanup("
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


def _phase_record(lowerer: ast.FunctionDef) -> ast.Expr:
    records = [
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Expr) and _phase_id(statement) == PHASE_ID
    ]
    assert len(records) == 1
    return records[0]


def test_terminal_sinet_singleton_reshape_convergence_current_contract() -> None:
    lowerer = _lowerer()
    record = _phase_record(lowerer)
    index = lowerer.body.index(record)
    predecessor = lowerer.body[index - 1]
    successor = lowerer.body[index + 1]

    assert _phase_id(predecessor) == PREDECESSOR_PHASE_ID

    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == FUTURE_OWNER_EXPRESSION

    assert _single_target(successor) == SUCCESSOR_TARGET
    assert _call_name(successor) == SUCCESSOR_OWNER
    successor_call = _call(successor)
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "sinet_terminal_layout_recovery_context"
    ]
    assert successor_call.keywords == []
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )


def test_terminal_sinet_singleton_reshape_convergence_schemas_are_fixed() -> None:
    model_ir = ModelIR("terminal_sinet_singleton_reshape_convergence_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )

    terminal_results = run_terminal_sinet_singleton_reshape_cleanup(context)
    convergence_results = run_indexed_shape_convergence_cleanup(
        model_ir,
        layout_state=context.layout_state,
    )

    assert tuple(type(result) for result in terminal_results) == (tuple, tuple)
    assert tuple(len(result) for result in terminal_results) == (6, 8)
    assert tuple(type(result) for result in terminal_results[0]) == (dict,) * 6
    assert tuple(type(result) for result in terminal_results[1]) == (dict,) * 8
    assert tuple(convergence_results) == (
        "removed_dead_operators",
        "resolved_dynamic_reshape_shapes",
        "reconciled_static_tensor_shapes",
    )
    assert all(type(value) is int for value in convergence_results.values())


def test_terminal_sinet_singleton_reshape_convergence_wrapper_is_retained() -> None:
    wrapper = _functions(LOWERER_PATH)[CURRENT_INDEXED_WRAPPER]
    assert len(wrapper.body) == 1
    statement = wrapper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == f"{CURRENT_INDEXED_WRAPPER}_pass"
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {
        "layout_state": "layout_state",
        "graph_index": "graph_index",
    }


def test_terminal_sinet_singleton_reshape_convergence_has_one_context_owner() -> None:
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
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.pass_context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {"layout_state": "context.pass_context.layout_state"}
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "(terminal_results, convergence_results)"
    )

    lowerer = _lowerer()
    record = _phase_record(lowerer)
    index = lowerer.body.index(record)
    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == FUTURE_OWNER_EXPRESSION
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert _call_name(lowerer.body[index + 1]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


def test_terminal_sinet_singleton_reshape_convergence_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("terminal_sinet_singleton_reshape_convergence_runtime")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    context = SINetTerminalLayoutRecoveryContext(
        pass_context=pass_context,
        preadd_resize_recovery=lambda: (),
    )
    terminal_results = (({"terminal": 1},), ({"reshape": 2},))
    convergence_results = {"convergence": 3}
    observed: list[tuple[str, object, dict[str, object]]] = []

    def terminal(active_context: object) -> object:
        observed.append((CHILD_OWNERS[0], active_context, {}))
        return terminal_results

    def convergence(active_model_ir: object, **options: object) -> object:
        observed.append((CHILD_OWNERS[1], active_model_ir, options))
        return convergence_results

    monkeypatch.setattr(
        terminal_sinet_singleton_reshape_convergence_orchestration,
        CHILD_OWNERS[0],
        terminal,
    )
    monkeypatch.setattr(
        terminal_sinet_singleton_reshape_convergence_orchestration,
        CHILD_OWNERS[1],
        convergence,
    )

    actual = terminal_sinet_singleton_reshape_convergence_orchestration.run_terminal_sinet_singleton_reshape_convergence_cleanup(
        context
    )
    assert actual[0] is terminal_results
    assert actual[1] is convergence_results
    assert observed == [
        (CHILD_OWNERS[0], pass_context, {}),
        (
            CHILD_OWNERS[1],
            model_ir,
            {"layout_state": pass_context.layout_state},
        ),
    ]
