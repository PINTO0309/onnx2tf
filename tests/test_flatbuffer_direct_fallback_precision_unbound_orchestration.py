from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    fallback_precision_unbound_orchestration,
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
    / "fallback_precision_unbound_orchestration.py"
)
OWNER = "run_fallback_precision_unbound_cleanup"
CHILD_OWNERS = (
    "run_precision_cleanup_sequence",
    "repair_unbound_nonconstant_operator_inputs_with_layout_transpose",
)
CURRENT_CHILD_OWNERS = (
    "_run_precision_cleanup_sequence",
    "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose",
)
RESULT_TARGETS = (
    "_fallback_precision_cleanup_results",
    "_fallback_unbound_repair_stats",
)
COMPOSITE_TARGET = "_fallback_precision_unbound_results"
FALLBACK_GUARD = (
    "optimize_layout_transpose_chains and len(unbound_inputs) > 0"
)
PREDECESSOR_PHASE_ID = "topology.fallback.post_placeholder"
SUCCESSOR_TARGET = "fallback_conv_input_stats"
SUCCESSOR_OWNER = "run_indexed_conv_input_adapter_repairs_summary"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _fallback_body() -> list[ast.stmt]:
    lowerer = _lowerer()
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == FALLBACK_GUARD
    )
    return guard.body


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


def test_fallback_precision_unbound_current_boundary_and_schema() -> None:
    body = _fallback_body()
    assignment = next(
        statement
        for statement in body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "fallback_precision_unbound_context"
    ]
    assert call.keywords == []
    assert _phase_id(body[index - 1]) == PREDECESSOR_PHASE_ID
    successor = body[index + 1]
    assert _single_target(successor) == SUCCESSOR_TARGET
    assert _call_name(successor) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for statement in body
        for node in ast.walk(statement)
    )

    model_ir = ModelIR("fallback_precision_unbound_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=None,
        diagnostics=[],
    )
    results = fallback_precision_unbound_orchestration.run_fallback_precision_unbound_cleanup(
        context
    )
    assert tuple(type(result) for result in results) == (tuple, dict)
    assert tuple(type(result) for result in results[0]) == (dict,) * 3
    assert tuple(results[0][0]) == ("rewritten_constant_div_to_mul",)
    assert tuple(results[0][1]) == (
        "optimized_fold_consecutive_mul_constants_chains",
    )
    assert tuple(results[0][2]) == (
        "restored_precision_sensitive_reciprocal_divisions",
    )
    assert tuple(results[1]) == (
        "repaired_unbound_nonconstant_inputs_with_layout_transpose",
    )


def test_fallback_precision_unbound_has_one_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.model_ir"
    ]
    assert calls[1].keywords == []

    body = _fallback_body()
    assignment = next(
        statement
        for statement in body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "fallback_precision_unbound_context"
    ]
    assert call.keywords == []
    assert _phase_id(body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _single_target(body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for statement in body
        for node in ast.walk(statement)
    )

    context_assignment = next(
        statement
        for statement in body
        if _single_target(statement) == "fallback_precision_unbound_context"
    )
    assert isinstance(context_assignment.value, ast.Call)
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in context_assignment.value.keywords
    } == {
        "model_ir": "fallback_ir",
        "layout_state": "None",
        "diagnostics": "session.diagnostics",
    }


def test_fallback_precision_unbound_runtime_order_context_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("fallback_precision_unbound_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=None,
        diagnostics=[],
    )
    expected_results = (
        tuple({f"precision_{index}": index} for index in range(3)),
        {"unbound": 4},
    )
    observed: list[tuple[str, object]] = []

    def precision(active_context: object) -> object:
        observed.append((CHILD_OWNERS[0], active_context))
        return expected_results[0]

    def unbound(active_model_ir: object) -> object:
        observed.append((CHILD_OWNERS[1], active_model_ir))
        return expected_results[1]

    monkeypatch.setattr(
        fallback_precision_unbound_orchestration,
        CHILD_OWNERS[0],
        precision,
    )
    monkeypatch.setattr(
        fallback_precision_unbound_orchestration,
        CHILD_OWNERS[1],
        unbound,
    )

    actual = fallback_precision_unbound_orchestration.run_fallback_precision_unbound_cleanup(
        context
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], context),
        (CHILD_OWNERS[1], model_ir),
    ]
