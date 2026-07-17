from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SINGLETON_RESHAPE = "_run_singleton_reshape_layout_pass_cluster"
FULL_OWNER_IDS = (
    "run_layout_transpose_cleanup",
    "run_singleton_channel_transpose_cleanup",
    "run_duplicate_fanout_cleanup",
    "run_singleton_reshape_layout_cleanup",
    "run_singleton_maxpool_layout_cleanup",
    "run_flatten_concat_reshape_cleanup",
    "run_consecutive_reshape_cleanup",
    "run_squeeze_reshape_identity_cleanup",
    "run_singleton_spatial_reshape_cleanup",
    "run_multi_branch_gate_layout_cleanup",
)
BASE_OWNER_IDS = FULL_OWNER_IDS[1:2] + FULL_OWNER_IDS[3:9]


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
        if isinstance(node, ast.FunctionDef) and node.name == SINGLETON_RESHAPE
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


def test_singleton_reshape_signature_defaults_and_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_layout_transpose",
        "include_duplicate_fanout",
        "include_multi_branch_gate",
        "include_spatial_concat_post_transpose",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        False,
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


def test_singleton_reshape_preserves_owner_contracts_and_guards() -> None:
    _, helper = _lowerer_and_helper()
    calls = _ordered_owner_calls(helper)

    assert tuple(call.func.id for call in calls) == FULL_OWNER_IDS
    shared_contract = {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    for call in calls:
        contract = {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        }
        if call.func.id == "run_duplicate_fanout_cleanup":
            assert contract.pop("include_transpose") is False
        if call.func.id == "run_singleton_spatial_reshape_cleanup":
            assert contract.pop("include_concat_post_transpose") == (
                "include_spatial_concat_post_transpose"
            )
        assert tuple(_expression_path(argument) for argument in call.args) == (
            "model_ir",
        )
        assert contract == shared_contract

    conditionals = [
        statement for statement in helper.body if isinstance(statement, ast.If)
    ]
    assert [_expression_path(conditional.test) for conditional in conditionals] == [
        "include_layout_transpose",
        "include_duplicate_fanout",
        "include_multi_branch_gate",
    ]
    assert all(conditional.orelse == [] for conditional in conditionals)
    assert [
        tuple(_direct_call_name(statement) for statement in conditional.body)
        for conditional in conditionals
    ] == [
        (FULL_OWNER_IDS[0],),
        (FULL_OWNER_IDS[2],),
        (FULL_OWNER_IDS[9],),
    ]
    guarded_nodes = {
        id(node) for conditional in conditionals for node in ast.walk(conditional)
    }
    assert (
        tuple(call.func.id for call in calls if id(call) not in guarded_nodes)
        == BASE_OWNER_IDS
    )


def test_singleton_reshape_preserves_layout_multi_policy_and_boundaries() -> None:
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
            and node.func.id == SINGLETON_RESHAPE
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
        and statement.value.func.id == SINGLETON_RESHAPE
    )
    invocation = guard.body[invocation_index]

    assert invocation_index == len(guard.body) - 1
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {
        "include_layout_transpose": True,
        "include_multi_branch_gate": True,
    }
    assert _direct_call_name(guard.body[invocation_index - 1]) == (
        "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )
    assert _direct_call_name(lowerer.body[outer_index + 1]) == (
        "_run_terminal_clamp_unary_relu_pass_cluster"
    )


def test_singleton_reshape_preserves_duplicate_spatial_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINGLETON_RESHAPE
    )
    invocation = lowerer.body[invocation_index]

    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {
        "include_duplicate_fanout": True,
        "include_spatial_concat_post_transpose": False,
    }
    assert _direct_call_name(lowerer.body[invocation_index - 1]) == (
        "_run_sinet_preadd_resize_recovery_sequence"
    )
    assert _direct_call_name(lowerer.body[invocation_index + 1]) == (
        "_run_indexed_shape_convergence_cleanup"
    )


def test_singleton_reshape_preserves_both_active_owner_sequences() -> None:
    lowerer, helper = _lowerer_and_helper()
    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SINGLETON_RESHAPE
    ]
    direct_invocations.sort(key=lambda call: call.lineno)

    assert len(direct_invocations) == 2
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        False,
        False,
        True,
    ]
    assert (
        FULL_OWNER_IDS[0],
        *BASE_OWNER_IDS,
        FULL_OWNER_IDS[9],
    ) == (
        "run_layout_transpose_cleanup",
        "run_singleton_channel_transpose_cleanup",
        "run_singleton_reshape_layout_cleanup",
        "run_singleton_maxpool_layout_cleanup",
        "run_flatten_concat_reshape_cleanup",
        "run_consecutive_reshape_cleanup",
        "run_squeeze_reshape_identity_cleanup",
        "run_singleton_spatial_reshape_cleanup",
        "run_multi_branch_gate_layout_cleanup",
    )
    assert (
        BASE_OWNER_IDS[0],
        FULL_OWNER_IDS[2],
        *BASE_OWNER_IDS[1:],
    ) == (
        "run_singleton_channel_transpose_cleanup",
        "run_duplicate_fanout_cleanup",
        "run_singleton_reshape_layout_cleanup",
        "run_singleton_maxpool_layout_cleanup",
        "run_flatten_concat_reshape_cleanup",
        "run_consecutive_reshape_cleanup",
        "run_squeeze_reshape_identity_cleanup",
        "run_singleton_spatial_reshape_cleanup",
    )
