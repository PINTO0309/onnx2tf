from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    PREADD_MEAN_ATTENTION_PASS_IDS,
    AttentionRecoveryContext,
    run_preadd_mean_attention_recovery,
)
from onnx2tf.tflite_builder.passes.channel_shuffle_gather_orchestration import (
    CHANNEL_SHUFFLE_GATHER_PASS_IDS,
    run_channel_shuffle_gather,
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
    / "layout_pass_set_2_channel_preadd_orchestration.py"
)
OWNER = "run_layout_pass_set_2_channel_preadd_recovery"
CHILD_OWNERS = (
    "run_channel_shuffle_gather",
    "run_preadd_mean_attention_recovery",
)
CURRENT_CHILD_OWNERS = (
    "_run_channel_shuffle_gather_layout_pass_cluster",
    "_run_preadd_mean_attention_recovery_sequence",
)
RESULT_TARGETS = (
    "_layout_opt_channel_shuffle_gather_results",
    "_layout_opt_preadd_mean_attention_results",
)
COMPOSITE_TARGET = "_layout_pass_set_2_channel_preadd_results"
PREDECESSOR_PHASE_ID = (
    "cleanup.layout_pass_set_2.slice_logistic_concat_tail"
)
SUCCESSOR_PHASE_ID = "cleanup.layout_pass_set_2.sa_pa_mirrorpad"
GUARD = "optimize_layout_transpose_chains"

CHANNEL_SCHEMA = (
    ("optimized_shufflenet_transpose_shuffle_chains",),
    ("optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains",),
    ("optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather",),
    ("optimized_transpose_gather_transpose_axis_remap_nhwc_chains",),
    (
        "iterations",
        "removed_identity_transpose",
        "removed_inverse_transpose_pairs",
        "removed_inverse_transpose_fanout_branches",
        "composed_consecutive_transpose_pairs",
    ),
    ("rewritten_transpose_unary_fanout_inverse_post_bridges",),
    ("rewritten_transpose_unary_binary_full_post_fanout_bridges",),
)
PREADD_SCHEMA = (
    ("optimized_transpose_pre_add_nhwc_chains",),
    ("optimized_transpose_pre_add_mul_add_prelu_nhwc_chains",),
    (
        "optimized_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
    ),
    ("optimized_transpose_mul_add_const_prepost_nhwc_chains",),
    (
        "optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
    ),
    ("optimized_transpose_mean_mul_add_const_prepost_nhwc_chains",),
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


def _guard_body() -> list[ast.stmt]:
    guard = next(
        statement
        for statement in _lowerer().body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == GUARD
        and any(
            _single_target(candidate)
            in (*RESULT_TARGETS, COMPOSITE_TARGET)
            for candidate in statement.body
        )
    )
    assert guard.orelse == []
    return guard.body


def _context() -> tuple[
    AttentionRecoveryContext,
    tuple[dict[str, int], ...],
]:
    model_ir = ModelIR("layout_pass_set_2_channel_preadd_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    mean_results = ({"mean_attention": 1},)
    return (
        AttentionRecoveryContext(
            pass_context=pass_context,
            mean_attention_cluster=lambda: mean_results,
            gate_layout_cluster=lambda: None,
            transpose_unary_fanout_cluster=lambda: None,
        ),
        mean_results,
    )


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_layout_pass_set_2_channel_preadd_current_contract() -> None:
    body = _guard_body()
    assignments = [
        statement
        for statement in body
        if _single_target(statement) in RESULT_TARGETS
    ]
    assert [_single_target(statement) for statement in assignments] == list(
        RESULT_TARGETS
    )
    assert [_call_name(statement) for statement in assignments] == list(
        CURRENT_CHILD_OWNERS
    )
    indexes = [body.index(statement) for statement in assignments]
    assert indexes[1] == indexes[0] + 1
    assert _phase_id(body[indexes[0] - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(body[indexes[-1] + 1]) == SUCCESSOR_PHASE_ID

    first_call = _call(assignments[0])
    second_call = _call(assignments[1])
    assert first_call is not None
    assert second_call is not None
    assert first_call.args == []
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in first_call.keywords
    } == {"include_post_gather_cleanup": True}
    assert second_call.args == []
    assert second_call.keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(_lowerer())
    )

    lowerer_functions = {
        node.name: node
        for node in _lowerer().body
        if isinstance(node, ast.FunctionDef)
    }
    assert all(name in lowerer_functions for name in CURRENT_CHILD_OWNERS)
    channel_helper = lowerer_functions[CURRENT_CHILD_OWNERS[0]]
    preadd_helper = lowerer_functions[CURRENT_CHILD_OWNERS[1]]
    assert ast.unparse(channel_helper.body[0].value.args[0]) == (
        "channel_shuffle_gather_context"
    )
    assert ast.unparse(preadd_helper.body[0].value.args[0]) == (
        "attention_recovery_context"
    )


def test_layout_pass_set_2_channel_preadd_child_schemas() -> None:
    context, mean_results = _context()
    channel_results = run_channel_shuffle_gather(
        context.pass_context,
        include_post_gather_cleanup=True,
    )
    preadd_results = run_preadd_mean_attention_recovery(context)

    assert len(channel_results) == len(CHANNEL_SHUFFLE_GATHER_PASS_IDS) == 7
    assert _dict_schema(channel_results) == CHANNEL_SCHEMA
    assert len(preadd_results) == len(PREADD_MEAN_ATTENTION_PASS_IDS) == 7
    assert _dict_schema(preadd_results[:6]) == PREADD_SCHEMA
    assert preadd_results[6] is mean_results


@pytest.mark.xfail(
    strict=True,
    reason="layout-pass-set-2 channel/pre-add owner is not implemented",
)
def test_layout_pass_set_2_channel_preadd_has_one_context_owner() -> None:
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
    } == {"include_post_gather_cleanup": True}
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
    assert calls[1].keywords == []

    body = _guard_body()
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
        "attention_recovery_context"
    ]
    assert call.keywords == []
    assert _phase_id(body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(_lowerer())
    )

    lowerer_owner = _lowerer()
    nested_functions = {
        node.name
        for node in lowerer_owner.body
        if isinstance(node, ast.FunctionDef)
    }
    assert all(name in nested_functions for name in CURRENT_CHILD_OWNERS)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )
