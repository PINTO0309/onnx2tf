from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SINGLETON_CONSECUTIVE = "_run_singleton_consecutive_reshape_pass_cluster"
OWNER_IDS = (
    "run_singleton_channel_transpose_cleanup",
    "run_duplicate_fanout_cleanup",
    "run_consecutive_reshape_cleanup",
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
        if isinstance(node, ast.FunctionDef) and node.name == SINGLETON_CONSECUTIVE
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


def _main_invocation_indexes(lowerer: ast.FunctionDef) -> list[int]:
    return [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINGLETON_CONSECUTIVE
    ]


def test_singleton_consecutive_signature_and_target_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert [argument.arg for argument in helper.args.args] == [
        "target_model_ir",
        "target_layout_state",
    ]
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
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
                ast.If,
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
        "target_model_ir",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in scope_calls[0].keywords
    } == {"layout_state": "target_layout_state"}


def test_singleton_consecutive_preserves_all_owner_contracts() -> None:
    _, helper = _lowerer_and_helper()
    calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in OWNER_IDS
    ]
    calls.sort(key=lambda call: call.lineno)

    assert tuple(call.func.id for call in calls) == OWNER_IDS
    shared_contract = {
        "layout_state": "target_layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    for call in (calls[0], calls[2]):
        assert tuple(_expression_path(argument) for argument in call.args) == (
            "target_model_ir",
        )
        assert {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        } == shared_contract
    assert tuple(_expression_path(argument) for argument in calls[1].args) == (
        "target_model_ir",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in calls[1].keywords
    } == {"include_transpose": False, **shared_contract}


def test_singleton_consecutive_preserves_all_three_target_forms() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SINGLETON_CONSECUTIVE
    ]
    invocations.sort(key=lambda call: call.lineno)

    assert [
        tuple(_expression_path(argument) for argument in invocation.args)
        for invocation in invocations
    ] == [
        ("model_ir", "session.layout_state"),
        ("model_ir", "session.layout_state"),
        ("fallback_ir", None),
    ]
    assert all(invocation.keywords == [] for invocation in invocations)


def test_singleton_consecutive_preserves_both_main_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = _main_invocation_indexes(lowerer)

    assert len(invocation_indexes) == 2
    first_index, second_index = invocation_indexes
    assert _direct_call_name(lowerer.body[first_index - 1]) == (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )
    first_following = lowerer.body[first_index + 1]
    assert isinstance(first_following, ast.If)
    assert isinstance(first_following.test, ast.Name)
    assert first_following.test.id == "optimize_layout_transpose_chains"

    assert _direct_call_name(lowerer.body[second_index - 1]) == (
        "_repair_rank4_binary_singleton_broadcast_layout_mismatch"
    )
    assert _direct_call_name(lowerer.body[second_index + 1]) == (
        "_reconcile_static_tensor_shapes"
    )


def test_singleton_consecutive_preserves_fallback_guard_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    fallback_guard = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == SINGLETON_CONSECUTIVE
            for statement in node.body
        )
    )
    invocation_index = next(
        index
        for index, statement in enumerate(fallback_guard.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINGLETON_CONSECUTIVE
    )

    assert ast.unparse(fallback_guard.test) == (
        "int(fallback_norm_stats.get("
        "'optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains', 0)) > 0"
    )
    assert _direct_call_name(fallback_guard.body[invocation_index - 1]) == (
        "_repair_rank4_binary_singleton_broadcast_layout_mismatch"
    )
    assert _direct_call_name(fallback_guard.body[invocation_index + 1]) == (
        "_reconcile_static_tensor_shapes"
    )
