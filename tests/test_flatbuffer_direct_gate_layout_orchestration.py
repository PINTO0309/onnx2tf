from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    ATTENTION_GATE_QDQ_PASS_IDS,
    AttentionRecoveryContext,
    build_attention_gate_qdq_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
GATE_LAYOUT = "_run_gate_layout_pass_cluster"
FULL_OWNER_IDS = (
    "run_mixed_attention_layout_cleanup",
    "run_elementwise_gate_layout_cleanup",
    "run_pad_layout_cleanup",
    "run_dual_postconv_gate_layout_cleanup",
    "run_ndhwc_gate_layout_cleanup",
    "run_cost_volume_scatter_layout_cleanup",
    "run_add_concat_suffix_layout_cleanup",
    "run_dual_mul_concat_layout_cleanup",
)
REQUIRED_OWNER_IDS = FULL_OWNER_IDS[1:]


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
        if isinstance(node, ast.FunctionDef) and node.name == GATE_LAYOUT
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


def test_gate_layout_signature_default_and_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_mixed_attention"
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [True]
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


def test_gate_layout_preserves_all_owner_contracts() -> None:
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


def test_gate_layout_optional_owner_has_one_exact_guard() -> None:
    _, helper = _lowerer_and_helper()
    conditionals = [
        statement for statement in helper.body if isinstance(statement, ast.If)
    ]

    assert len(conditionals) == 1
    conditional = conditionals[0]
    assert isinstance(conditional.test, ast.Name)
    assert conditional.test.id == "include_mixed_attention"
    assert conditional.orelse == []
    assert len(conditional.body) == 1
    statement = conditional.body[0]
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    assert statement.value.func.id == FULL_OWNER_IDS[0]
    assert tuple(call.func.id for call in _ordered_owner_calls(helper)[1:]) == (
        REQUIRED_OWNER_IDS
    )
    assert all(
        call not in [node for node in ast.walk(conditional)]
        for call in _ordered_owner_calls(helper)[1:]
    )


def test_gate_layout_preserves_direct_reduced_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    direct_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(candidate, ast.Expr)
            and isinstance(candidate.value, ast.Call)
            and isinstance(candidate.value.func, ast.Name)
            and candidate.value.func.id == GATE_LAYOUT
            for candidate in statement.body
        )
    )
    invocation_index = next(
        index
        for index, statement in enumerate(direct_guard.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == GATE_LAYOUT
    )
    invocation = direct_guard.body[invocation_index]

    assert isinstance(direct_guard.test, ast.Name)
    assert direct_guard.test.id == "optimize_layout_transpose_chains"
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_mixed_attention": False}
    assert _direct_call_name(direct_guard.body[invocation_index - 1]) == (
        "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"
    )
    following = direct_guard.body[invocation_index + 1]
    assert isinstance(following, ast.For)
    assert isinstance(following.target, ast.Name)
    assert following.target.id == "_"
    assert isinstance(following.iter, ast.Call)
    assert isinstance(following.iter.func, ast.Name)
    assert following.iter.func.id == "range"
    assert len(following.iter.args) == 1
    assert _expression_path(following.iter.args[0]) == 2


def test_gate_layout_preserves_argument_free_full_policy_callback() -> None:
    lowerer, helper = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "attention_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    gate_keyword = next(
        keyword
        for keyword in context_assignment.value.keywords
        if keyword.arg == "gate_layout_cluster"
    )
    assert _expression_path(gate_keyword.value) == GATE_LAYOUT

    def callback() -> None:
        return None

    context = AttentionRecoveryContext(
        model_ir=ModelIR("gate_layout_callback_test"),
        layout_state=None,
        diagnostics=[],
        mean_attention_cluster=lambda: None,
        gate_layout_cluster=callback,
        transpose_unary_fanout_cluster=lambda: None,
    )
    invocation = build_attention_gate_qdq_invocations(context)[2]

    assert ATTENTION_GATE_QDQ_PASS_IDS[2] == GATE_LAYOUT
    assert invocation.callback is callback
    assert invocation.args == ()
    assert invocation.keyword_args == ()
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_mixed_attention"
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [True]
