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
from onnx2tf.tflite_builder.passes.singleton_reshape_orchestration import (
    run_singleton_reshape,
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
    / "terminal_sinet_singleton_reshape_orchestration.py"
)
OWNER = "run_terminal_sinet_singleton_reshape_cleanup"
CHILD_OWNERS = (
    "run_sinet_preadd_resize_recovery",
    "run_singleton_reshape",
)
CURRENT_CHILD_OWNERS = (
    "_run_sinet_preadd_resize_recovery_sequence",
    "_run_singleton_reshape_layout_pass_cluster",
)
RESULT_TARGETS = (
    "_terminal_sinet_preadd_resize_results",
    "_post_terminal_singleton_reshape_results",
)
COMPOSITE_TARGET = "_terminal_sinet_singleton_reshape_results"
PREDECESSOR_PHASE_ID = "cleanup.terminal.dequant_hardsigmoid_bridge"
SUCCESSOR_PHASE_ID = "shape_topology.terminal.indexed_convergence"
SINGLETON_POLICY = {
    "include_duplicate_fanout": True,
    "include_spatial_concat_post_transpose": False,
}


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


def test_terminal_sinet_singleton_reshape_current_boundary_and_schema() -> None:
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
    assert assignments[0].value.args == []
    assert assignments[0].value.keywords == []
    assert assignments[1].value.args == []
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in assignments[1].value.keywords
    } == SINGLETON_POLICY
    assert _phase_id(lowerer.body[indices[0] - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(lowerer.body[indices[1] + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    shared_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == "shared_model_ir_pass_context"
    )
    assert ast.unparse(shared_assignment.value) == "session.model_ir_pass_context"
    for context_name in (
        "sinet_preadd_resize_recovery_context",
        "singleton_reshape_context",
    ):
        context_assignment = next(
            statement
            for statement in lowerer.body
            if _single_target(statement) == context_name
        )
        assert ast.unparse(context_assignment.value) == (
            "shared_model_ir_pass_context"
        )

    model_ir = ModelIR("terminal_sinet_singleton_reshape_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = (
        run_sinet_preadd_resize_recovery(context),
        run_singleton_reshape(context, **SINGLETON_POLICY),
    )
    assert tuple(type(result) for result in results) == (tuple, tuple)
    assert tuple(len(result) for result in results) == (6, 8)
    assert tuple(type(result) for result in results[0]) == (dict,) * 6
    assert tuple(type(result) for result in results[1]) == (dict,) * 8


@pytest.mark.xfail(
    strict=True,
    reason="terminal SiNet/singleton-reshape composite owner is not implemented",
)
def test_terminal_sinet_singleton_reshape_has_one_shared_context_owner() -> None:
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
    assert all(
        [ast.unparse(argument) for argument in call.args] == ["context"]
        for call in calls
    )
    assert calls[0].keywords == []
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[1].keywords
    } == SINGLETON_POLICY

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
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
