from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    PREADD_MEAN_ATTENTION_PASS_IDS,
    AttentionRecoveryContext,
    build_preadd_mean_attention_invocations,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS,
    LayoutAttentionQuantizedSuffixContext,
    build_layout_attention_quantized_suffix_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
MEAN_ATTENTION = "_run_mean_attention_layout_pass_cluster"
FULL_OWNER_IDS = (
    "run_transpose_mean_passthrough_cleanup",
    "run_mean_mul_add_conv_layout_cleanup",
    "run_layernorm_statistics_layout_cleanup",
    "run_terminal_mean_layout_cleanup",
    "run_se_conv_layout_cleanup",
    "run_se_fc_layout_cleanup",
    "run_conv_attention_layout_cleanup",
)
BASE_OWNER_IDS = (
    FULL_OWNER_IDS[0],
    FULL_OWNER_IDS[1],
    *FULL_OWNER_IDS[3:6],
)


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
        if isinstance(node, ast.FunctionDef) and node.name == MEAN_ATTENTION
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


def test_mean_attention_signature_defaults_and_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_layernorm",
        "include_conv_attention",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
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


def test_mean_attention_preserves_owner_contracts_and_guards() -> None:
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
        "include_layernorm",
        "include_conv_attention",
    ]
    assert all(conditional.orelse == [] for conditional in conditionals)
    assert [
        tuple(_direct_call_name(statement) for statement in conditional.body)
        for conditional in conditionals
    ] == [
        (FULL_OWNER_IDS[2],),
        (FULL_OWNER_IDS[6],),
    ]
    guarded_nodes = {
        id(node) for conditional in conditionals for node in ast.walk(conditional)
    }
    assert (
        tuple(call.func.id for call in calls if id(call) not in guarded_nodes)
        == BASE_OWNER_IDS
    )


def test_mean_attention_preserves_layernorm_conv_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == MEAN_ATTENTION
            and any(keyword.arg == "include_layernorm" for keyword in node.keywords)
            for node in ast.walk(statement)
        )
    )
    invocation_index = next(
        index
        for index, statement in enumerate(guard.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == MEAN_ATTENTION
        and any(
            keyword.arg == "include_layernorm" for keyword in statement.value.keywords
        )
    )
    invocation = guard.body[invocation_index]

    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_layernorm": True}
    assert _direct_call_name(guard.body[invocation_index - 1]) == (
        "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains"
    )
    assert _direct_call_name(guard.body[invocation_index + 1]) == (
        "_run_attention_gate_qdq_recovery_sequence"
    )


def test_mean_attention_preserves_terminal_base_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    outer_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == MEAN_ATTENTION
            and any(
                keyword.arg == "include_conv_attention" for keyword in node.keywords
            )
            for node in ast.walk(statement)
        )
    )
    guard = lowerer.body[outer_index]
    assert isinstance(guard, ast.If)
    invocation_index = next(
        index
        for index, statement in enumerate(guard.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == MEAN_ATTENTION
    )
    invocation = guard.body[invocation_index]

    assert invocation_index == 0
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_conv_attention": False}
    assert _direct_call_name(lowerer.body[outer_index - 1]) == (
        "_run_terminal_boundary_layout_pass_cluster"
    )
    assert _direct_call_name(guard.body[invocation_index + 1]) == (
        "_optimize_batchmatmul_affine_transpose_input_chains"
    )


def test_mean_attention_preserves_both_argument_free_default_callbacks() -> None:
    lowerer, helper = _lowerer_and_helper()
    assignments = {
        target.id: statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        and target.id
        in {
            "attention_recovery_context",
            "layout_attention_quantized_suffix_context",
        }
    }
    assert set(assignments) == {
        "attention_recovery_context",
        "layout_attention_quantized_suffix_context",
    }
    for assignment in assignments.values():
        assert isinstance(assignment.value, ast.Call)
        callback_keyword = next(
            keyword
            for keyword in assignment.value.keywords
            if keyword.arg == "mean_attention_cluster"
        )
        assert _expression_path(callback_keyword.value) == MEAN_ATTENTION

    def callback() -> None:
        return None

    attention_context = AttentionRecoveryContext(
        model_ir=ModelIR("mean_attention_callback_test"),
        layout_state=None,
        diagnostics=[],
        mean_attention_cluster=callback,
        gate_layout_cluster=lambda: None,
        transpose_unary_fanout_cluster=lambda: None,
    )
    attention_invocation = build_preadd_mean_attention_invocations(attention_context)[
        -1
    ]
    suffix_context = LayoutAttentionQuantizedSuffixContext(
        model_ir=attention_context.model_ir,
        layout_state=None,
        diagnostics=[],
        mean_attention_cluster=callback,
        attention_gate_qdq_recovery=lambda: None,
        duplicate_quantized_prelu_cluster=lambda **kwargs: None,
    )
    suffix_invocation = build_layout_attention_quantized_suffix_invocations(
        suffix_context,
        include_duplicate_transpose=True,
    )[3]

    assert PREADD_MEAN_ATTENTION_PASS_IDS[-1] == MEAN_ATTENTION
    assert LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[3] == MEAN_ATTENTION
    for invocation in (attention_invocation, suffix_invocation):
        assert invocation.callback is callback
        assert invocation.args == ()
        assert invocation.keyword_args == ()
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_layernorm",
        "include_conv_attention",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
    ]

    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == MEAN_ATTENTION
    ]
    assert len(direct_invocations) == 2
