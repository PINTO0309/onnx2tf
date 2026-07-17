from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LAYOUT_RECOVERY_PASS_IDS,
    LayoutRecoveryContext,
    build_layout_recovery_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
CHANNEL_SHUFFLE_GATHER = "_run_channel_shuffle_gather_layout_pass_cluster"
FULL_OWNER_IDS = (
    "run_two_way_channel_shuffle_cleanup",
    "run_nhwc_channel_shuffle_cleanup",
    "run_nchw_channel_shuffle_cleanup",
    "run_transpose_gather_axis_cleanup",
    "run_layout_transpose_cleanup",
    "run_transpose_unary_fanout_bridge_cleanup",
    "run_transpose_unary_binary_fanout_bridge_cleanup",
)
BASE_OWNER_IDS = FULL_OWNER_IDS[2:4]
POST_OWNER_IDS = FULL_OWNER_IDS[4:]


def _lowerer_and_helper() -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == CHANNEL_SHUFFLE_GATHER
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected expression: {ast.dump(node)}")


def _direct_call_name(statement: ast.stmt) -> str:
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    return statement.value.func.id


def _ordered_owner_calls(helper: ast.FunctionDef) -> list[ast.Call]:
    calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in FULL_OWNER_IDS
    ]
    return sorted(calls, key=lambda call: call.lineno)


def test_channel_shuffle_gather_signature_defaults_and_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_two_way_shuffle",
        "include_nhwc_shuffle",
        "include_post_gather_cleanup",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        True,
        True,
        False,
    ]
    assert helper.args.defaults == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert not any(
        isinstance(
            node,
            (
                ast.AsyncFor,
                ast.AsyncWith,
                ast.For,
                ast.Match,
                ast.Try,
                ast.While,
                ast.With,
            ),
        )
        for node in ast.walk(helper)
    )

    scope_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
    ]
    assert len(scope_calls) == 1
    assert tuple(_expression_path(argument) for argument in scope_calls[0].args) == (
        "model_ir",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in scope_calls[0].keywords
    } == {"layout_state": "session.layout_state"}


def test_channel_shuffle_gather_preserves_owner_contracts_and_guards() -> None:
    _, helper = _lowerer_and_helper()
    calls = _ordered_owner_calls(helper)

    assert tuple(call.func.id for call in calls) == FULL_OWNER_IDS
    shared_contract = {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    for call in calls:
        assert tuple(_expression_path(argument) for argument in call.args) == (
            "model_ir",
        )
        assert {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        } == shared_contract

    conditionals = [
        statement for statement in helper.body if isinstance(statement, ast.If)
    ]
    assert [_expression_path(conditional.test) for conditional in conditionals] == [
        "include_two_way_shuffle",
        "include_nhwc_shuffle",
        "include_post_gather_cleanup",
    ]
    assert all(conditional.orelse == [] for conditional in conditionals)
    assert [
        tuple(_direct_call_name(statement) for statement in conditional.body)
        for conditional in conditionals
    ] == [
        (FULL_OWNER_IDS[0],),
        (FULL_OWNER_IDS[1],),
        POST_OWNER_IDS,
    ]
    guarded_nodes = {
        id(node) for conditional in conditionals for node in ast.walk(conditional)
    }
    assert (
        tuple(call.func.id for call in calls if id(call) not in guarded_nodes)
        == BASE_OWNER_IDS
    )


def test_channel_shuffle_gather_preserves_full_post_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == CHANNEL_SHUFFLE_GATHER
            and any(
                keyword.arg == "include_post_gather_cleanup"
                for keyword in node.keywords
            )
            for node in ast.walk(statement)
        )
    )
    invocation_index = next(
        index
        for index, statement in enumerate(guard.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == CHANNEL_SHUFFLE_GATHER
    )
    invocation = guard.body[invocation_index]

    assert isinstance(guard.test, ast.Name)
    assert guard.test.id == "optimize_layout_transpose_chains"
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_post_gather_cleanup": True}
    assert _direct_call_name(guard.body[invocation_index - 1]) == (
        "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"
    )
    assert _direct_call_name(guard.body[invocation_index + 1]) == (
        "_run_preadd_mean_attention_recovery_sequence"
    )


def test_channel_shuffle_gather_preserves_late_base_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == CHANNEL_SHUFFLE_GATHER
    )
    invocation = lowerer.body[invocation_index]

    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {
        "include_two_way_shuffle": False,
        "include_nhwc_shuffle": False,
    }
    assert _direct_call_name(lowerer.body[invocation_index - 1]) == (
        "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains"
    )
    assert _direct_call_name(lowerer.body[invocation_index + 1]) == (
        "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains"
    )


def test_channel_shuffle_gather_preserves_argument_free_default_callback() -> None:
    lowerer, helper = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "layout_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    callback_keyword = next(
        keyword
        for keyword in context_assignment.value.keywords
        if keyword.arg == "channel_shuffle_gather_cluster"
    )
    assert _expression_path(callback_keyword.value) == CHANNEL_SHUFFLE_GATHER

    def callback() -> None:
        return None

    context = LayoutRecoveryContext(
        model_ir=ModelIR("channel_shuffle_gather_callback_test"),
        layout_state=None,
        diagnostics=[],
        boundary_batchmatmul_unary_cluster=lambda: None,
        pre_concat_cleanup=lambda *args, **kwargs: None,
        channel_shuffle_gather_cluster=callback,
    )
    invocation = build_layout_recovery_invocations(context)[-1]

    assert LAYOUT_RECOVERY_PASS_IDS[-1] == CHANNEL_SHUFFLE_GATHER
    assert invocation.callback is callback
    assert invocation.args == ()
    assert invocation.keyword_args == ()
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_two_way_shuffle",
        "include_nhwc_shuffle",
        "include_post_gather_cleanup",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        True,
        True,
        False,
    ]

    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHANNEL_SHUFFLE_GATHER
    ]
    assert len(direct_invocations) == 2
