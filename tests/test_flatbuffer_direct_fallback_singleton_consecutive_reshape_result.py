from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.singleton_consecutive_reshape_orchestration import (
    SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS,
    build_singleton_consecutive_reshape_invocations,
    run_singleton_consecutive_reshape,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_run_singleton_consecutive_reshape_pass_cluster"
RESULT_TARGET = "_fallback_singleton_consecutive_reshape_results"
GUARD_EXPRESSION = (
    "int(fallback_norm_stats.get("
    "'optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains', 0)) > 0"
)
RESULT_SCHEMA = (
    {"rewritten_singleton_channel_layout_transpose_to_reshape": 0},
    {"removed_duplicate_reshape_fanout": 0},
    {
        "removed_noop_reshape_chains": 0,
        "rewritten_consecutive_reshape_passthrough_chains": 0,
        "rewritten_fanout_bypass_reshape_passthrough_chains": 0,
    },
)


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _fallback_location() -> tuple[ast.FunctionDef, ast.If, int]:
    lowerer = _lowerer()
    guard = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == GUARD_EXPRESSION
        and any(
            _call_name(statement) == OWNER
            for statement in node.body
        )
    )
    index = next(
        index
        for index, statement in enumerate(guard.body)
        if _call_name(statement) == OWNER
    )
    return lowerer, guard, index


def test_fallback_singleton_consecutive_schema_and_scope_are_explicit() -> None:
    assert SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS == (
        "run_singleton_channel_transpose_cleanup",
        "run_duplicate_fanout_cleanup",
        "run_consecutive_reshape_cleanup",
    )
    model_ir = ModelIR("fallback_singleton_consecutive_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=None,
        diagnostics=[],
    )
    invocations = build_singleton_consecutive_reshape_invocations(context)
    assert tuple(invocation.pass_id for invocation in invocations) == (
        SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS
    )
    assert all(invocation.args == (model_ir,) for invocation in invocations)
    keyword_args = [dict(invocation.keyword_args) for invocation in invocations]
    scopes = [keywords["state_scope"] for keywords in keyword_args]
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert all(scope is scopes[0] for scope in scopes)
    assert keyword_args[1]["include_transpose"] is False
    assert all(keywords["layout_state"] is None for keywords in keyword_args)
    assert run_singleton_consecutive_reshape(context) == RESULT_SCHEMA


def test_fallback_singleton_consecutive_guard_and_boundaries_are_explicit() -> None:
    lowerer, guard, index = _fallback_location()
    invocation = guard.body[index]
    assert _single_target(invocation) == RESULT_TARGET
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "fallback_ir",
        "None",
    ]
    assert call.keywords == []
    previous = guard.body[index - 1]
    assert _call_name(previous) == "run_indexed_binary_layout_adapter_cleanup"
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Tuple)
    assert [
        element.id
        for element in previous.targets[0].elts
        if isinstance(element, ast.Name)
    ] == [
        "_fallback_binary_adapter_stats",
        "_fallback_singleton_adapter_stats",
    ]
    assert ast.unparse(guard.body[index + 1]) == (
        "session.record_phase_result('shape_topology.fallback.norm', "
        "run_static_shape_topology_reconciliation(fallback_ir))"
    )

    fallback_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        and [ast.unparse(argument) for argument in node.args]
        == ["fallback_ir", "None"]
    ]
    assert fallback_calls == [call]


def test_fallback_singleton_consecutive_result_is_retained_for_observation() -> None:
    lowerer, guard, index = _fallback_location()
    assert _single_target(guard.body[index]) == RESULT_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
