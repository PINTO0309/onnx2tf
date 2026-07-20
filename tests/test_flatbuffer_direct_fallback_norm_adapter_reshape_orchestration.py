from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    fallback_norm_adapter_reshape_orchestration,
)
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    run_indexed_binary_layout_adapter_cleanup,
)
from onnx2tf.tflite_builder.passes.singleton_consecutive_reshape_orchestration import (
    run_singleton_consecutive_reshape,
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
    / "fallback_norm_adapter_reshape_orchestration.py"
)
OWNER = "run_fallback_norm_adapter_reshape_cleanup"
CHILD_OWNERS = (
    "run_indexed_binary_layout_adapter_cleanup",
    "run_singleton_consecutive_reshape",
)
CURRENT_RESHAPE_WRAPPER = "_run_singleton_consecutive_reshape_pass_cluster"
RESULT_TARGETS = (
    "_fallback_binary_adapter_stats",
    "_fallback_singleton_adapter_stats",
    "_fallback_singleton_consecutive_reshape_results",
)
COMPOSITE_TARGET = "_fallback_norm_adapter_reshape_results"
CONTEXT_TARGET = "fallback_precision_unbound_context"
GUARD_EXPRESSION = (
    "int(fallback_norm_stats.get("
    "'optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains', 0)) > 0"
)
SUCCESSOR_PHASE_ID = "shape_topology.fallback.norm"
ADAPTER_SCHEMA = (
    ("inserted_rank4_binary_layout_fix_transpose",),
    ("repaired_rank4_binary_singleton_broadcast_layout_mismatch",),
)
RESHAPE_SCHEMA = (
    ("rewritten_singleton_channel_layout_transpose_to_reshape",),
    ("removed_duplicate_reshape_fanout",),
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


def _assignment_targets(statement: ast.stmt) -> tuple[str, ...]:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return ()
    target = statement.targets[0]
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            element.id
            for element in target.elts
            if isinstance(element, ast.Name)
        )
    return ()


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


def _guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == GUARD_EXPRESSION
        and any(
            set(_assignment_targets(statement))
            & {*RESULT_TARGETS, COMPOSITE_TARGET}
            for statement in node.body
        )
    )


def _context() -> ModelIRPassContext:
    return ModelIRPassContext(
        model_ir=ModelIR("fallback_norm_adapter_reshape_schema"),
        layout_state=None,
        diagnostics=[],
    )


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_fallback_norm_adapter_reshape_current_contract() -> None:
    lowerer = _lowerer()
    guard = _guard(lowerer)
    assert guard.orelse == []
    assert len(guard.body) == 2

    assignment = guard.body[0]
    assert _single_target(assignment) == COMPOSITE_TARGET
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        CONTEXT_TARGET
    ]
    assert call.keywords == []
    assert _phase_id(guard.body[1]) == SUCCESSOR_PHASE_ID

    context_assignment = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Assign)
        and _single_target(node) == CONTEXT_TARGET
    )
    assert isinstance(context_assignment.value, ast.Call)
    assert ast.unparse(context_assignment.value.func) == "ModelIRPassContext"
    assert context_assignment.value.args == []
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in context_assignment.value.keywords
    } == {
        "model_ir": "fallback_ir",
        "layout_state": "None",
        "diagnostics": "session.diagnostics",
    }
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_fallback_norm_adapter_reshape_child_schemas_and_wrapper() -> None:
    context = _context()
    adapter_results = run_indexed_binary_layout_adapter_cleanup(
        context.model_ir
    )
    reshape_results = run_singleton_consecutive_reshape(context)
    assert _dict_schema(adapter_results) == ADAPTER_SCHEMA
    assert _dict_schema(reshape_results) == RESHAPE_SCHEMA

    lowerer = _lowerer()
    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == CURRENT_RESHAPE_WRAPPER
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    assert _call_name(wrapper.body[0]) == CHILD_OWNERS[1]
    wrapper_call = _call(wrapper.body[0])
    assert wrapper_call is not None
    assert ast.unparse(wrapper_call.args[0]) == (
        "ModelIRPassContext(model_ir=target_model_ir, "
        "layout_state=target_layout_state, diagnostics=session.diagnostics)"
    )
    assert wrapper_call.keywords == []


def test_fallback_norm_adapter_reshape_has_one_context_owner() -> None:
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
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
    assert calls[1].keywords == []

    lowerer = _lowerer()
    guard = _guard(lowerer)
    assert len(guard.body) == 2
    assignment = guard.body[0]
    assert _single_target(assignment) == COMPOSITE_TARGET
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        CONTEXT_TARGET
    ]
    assert call.keywords == []
    assert _phase_id(guard.body[1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == CURRENT_RESHAPE_WRAPPER
        for node in lowerer.body
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


def test_fallback_norm_adapter_reshape_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    adapter_results = ({"binary": 1}, {"singleton": 2})
    reshape_results = (
        {"channel": 3},
        {"fanout": 4},
        {"reshape": 5},
    )
    observed: list[tuple[str, object]] = []

    def _adapter(active_model_ir: ModelIR) -> object:
        observed.append((CHILD_OWNERS[0], active_model_ir))
        return adapter_results

    def _reshape(active_context: ModelIRPassContext) -> object:
        observed.append((CHILD_OWNERS[1], active_context))
        return reshape_results

    monkeypatch.setattr(
        fallback_norm_adapter_reshape_orchestration,
        CHILD_OWNERS[0],
        _adapter,
    )
    monkeypatch.setattr(
        fallback_norm_adapter_reshape_orchestration,
        CHILD_OWNERS[1],
        _reshape,
    )

    actual = fallback_norm_adapter_reshape_orchestration.run_fallback_norm_adapter_reshape_cleanup(
        context
    )
    assert actual[0] is adapter_results
    assert actual[1] is reshape_results
    assert observed == [
        (CHILD_OWNERS[0], context.model_ir),
        (CHILD_OWNERS[1], context),
    ]
