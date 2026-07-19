from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes.singleton_reshape_orchestration import (
    run_singleton_reshape,
)
from onnx2tf.tflite_builder.passes.terminal_clamp_sinet_layout_orchestration import (
    run_terminal_clamp_sinet_layout_cleanup,
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
    / "terminal_singleton_clamp_sinet_orchestration.py"
)
OWNER = "run_terminal_singleton_clamp_sinet_cleanup"
CHILD_OWNERS = (
    "run_singleton_reshape",
    "run_terminal_clamp_sinet_layout_cleanup",
)
CURRENT_CHILD_OWNERS = (
    "_run_singleton_reshape_layout_pass_cluster",
    "run_terminal_clamp_sinet_layout_cleanup",
)
RESULT_TARGETS = (
    "_terminal_singleton_reshape_results",
    "_terminal_clamp_sinet_layout_results",
)
COMPOSITE_TARGET = "_terminal_singleton_clamp_sinet_results"
PREDECESSOR_PHASE_ID = "cleanup.terminal.qkv_split_conv_concat_bridge"
SUCCESSOR_PHASE_ID = "cleanup.terminal.sinet_hardswish_se"
GUARD = "optimize_layout_transpose_chains"

SINGLETON_SCHEMA = (
    (
        "iterations",
        "removed_identity_transpose",
        "removed_inverse_transpose_pairs",
        "removed_inverse_transpose_fanout_branches",
        "composed_consecutive_transpose_pairs",
    ),
    ("rewritten_singleton_channel_layout_transpose_to_reshape",),
    (
        "rewritten_singleton_layout_reshape_unary_passthrough_chains",
        "rewritten_consecutive_inverse_singleton_layout_reshapes",
    ),
    (
        "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains",
        "rewritten_singleton_nms_maxpool_nhwc_chains",
    ),
    ("rewritten_flatten_concat_expanddims_to_nhwc_concat",),
    (
        "removed_noop_reshape_chains",
        "rewritten_consecutive_reshape_passthrough_chains",
        "rewritten_fanout_bypass_reshape_passthrough_chains",
    ),
    ("optimized_squeeze_reshape_identity_chains",),
    (
        "optimized_singleton_spatial_nhwc_transpose_reshape_flatten",
        "rewritten_singleton_reshape_concat_post_transpose_nhwc_chains",
    ),
    ("optimized_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains",),
)
CLAMP_SCHEMA = (
    ("rewritten_maximum_minimum_relu0to1_chains",),
    ("rewritten_transpose_unary_passthrough_chains",),
    ("rewritten_maximum_with_zero_input2_to_relu",),
)
SINET_SCHEMA = (
    ("optimized_sinet_shuffle_residual_transpose_chains",),
    None,
    (
        "optimized_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains",
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


def _terminal_guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == GUARD
        and any(
            _single_target(candidate) == RESULT_TARGETS[0]
            or _phase_id(candidate) == PREDECESSOR_PHASE_ID
            for candidate in statement.body
        )
    )


def _context() -> tuple[
    SINetTerminalLayoutRecoveryContext,
    tuple[dict[str, int], ...],
]:
    model_ir = ModelIR("terminal_singleton_clamp_sinet_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    preadd_results = ({"preadd_resize": 1},)
    return (
        SINetTerminalLayoutRecoveryContext(
            pass_context=pass_context,
            preadd_resize_recovery=lambda: preadd_results,
        ),
        preadd_results,
    )


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_terminal_singleton_clamp_sinet_current_contract() -> None:
    lowerer = _lowerer()
    guard = _terminal_guard(lowerer)
    assert guard.orelse == []
    singleton = guard.body[-1]
    assert _single_target(singleton) == RESULT_TARGETS[0]
    assert _call_name(singleton) == CURRENT_CHILD_OWNERS[0]
    singleton_call = _call(singleton)
    assert singleton_call is not None
    assert singleton_call.args == []
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in singleton_call.keywords
    } == {
        "include_layout_transpose": True,
        "include_multi_branch_gate": True,
    }
    assert _phase_id(guard.body[-2]) == PREDECESSOR_PHASE_ID

    guard_index = lowerer.body.index(guard)
    clamp_sinet = lowerer.body[guard_index + 1]
    assert _single_target(clamp_sinet) == RESULT_TARGETS[1]
    assert _call_name(clamp_sinet) == CURRENT_CHILD_OWNERS[1]
    clamp_sinet_call = _call(clamp_sinet)
    assert clamp_sinet_call is not None
    assert [ast.unparse(argument) for argument in clamp_sinet_call.args] == [
        "sinet_terminal_layout_recovery_context"
    ]
    assert clamp_sinet_call.keywords == []
    assert _phase_id(lowerer.body[guard_index + 2]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    lowerer_functions = {
        node.name: node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
    }
    singleton_helper = lowerer_functions[CURRENT_CHILD_OWNERS[0]]
    singleton_delegate = singleton_helper.body[0]
    assert isinstance(singleton_delegate, ast.Return)
    assert isinstance(singleton_delegate.value, ast.Call)
    assert ast.unparse(singleton_delegate.value.args[0]) == (
        "singleton_reshape_context"
    )
    assert "_run_sinet_terminal_layout_recovery_sequence" in lowerer_functions

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


def test_terminal_singleton_clamp_sinet_child_schemas() -> None:
    context, preadd_results = _context()
    singleton_results = run_singleton_reshape(
        context.pass_context,
        include_layout_transpose=True,
        include_multi_branch_gate=True,
    )
    clamp_sinet_results = run_terminal_clamp_sinet_layout_cleanup(context)

    assert _dict_schema(singleton_results) == SINGLETON_SCHEMA
    assert _dict_schema(clamp_sinet_results[0]) == CLAMP_SCHEMA
    sinet_results = clamp_sinet_results[1]
    assert tuple(
        tuple(result) if isinstance(result, dict) else None
        for result in sinet_results
    ) == SINET_SCHEMA
    assert sinet_results[1] is preadd_results


@pytest.mark.xfail(
    strict=True,
    reason="terminal singleton/Clamp-SiNet owner is not implemented",
)
def test_terminal_singleton_clamp_sinet_has_one_optional_context_owner() -> None:
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
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[0].keywords
    } == {
        "include_layout_transpose": True,
        "include_multi_branch_gate": True,
    }
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
    assert calls[1].keywords == []
    singleton_guard = next(
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "include_terminal_singleton"
    )
    assert calls[0] in list(ast.walk(singleton_guard))
    assert calls[1] not in list(ast.walk(singleton_guard))

    lowerer = _lowerer()
    guard = _terminal_guard(lowerer)
    guard_index = lowerer.body.index(guard)
    assert _phase_id(guard.body[-1]) == PREDECESSOR_PHASE_ID
    assignment = lowerer.body[guard_index + 1]
    assert _single_target(assignment) == COMPOSITE_TARGET
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "sinet_terminal_layout_recovery_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_terminal_singleton": GUARD}
    assert _phase_id(lowerer.body[guard_index + 2]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    nested_functions = {
        node.name
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
    }
    assert CURRENT_CHILD_OWNERS[0] in nested_functions
    assert "_run_sinet_terminal_layout_recovery_sequence" in nested_functions
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )
