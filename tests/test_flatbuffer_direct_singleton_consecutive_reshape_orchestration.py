from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    singleton_consecutive_reshape_orchestration,
)
from onnx2tf.tflite_builder.passes.singleton_consecutive_reshape_orchestration import (
    SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS,
    SingletonConsecutiveReshapeContext,
    build_singleton_consecutive_reshape_invocations,
    run_singleton_consecutive_reshape,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PHASE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "singleton_consecutive_reshape_orchestration.py"
)
SINGLETON_CONSECUTIVE = "_run_singleton_consecutive_reshape_pass_cluster"


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


def _statement_call_name(statement: ast.stmt) -> str:
    value = (
        statement.value
        if isinstance(statement, (ast.Expr, ast.Assign))
        else None
    )
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    return value.func.id


def _main_invocation_indexes(lowerer: ast.FunctionDef) -> list[int]:
    return [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, (ast.Expr, ast.Assign))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINGLETON_CONSECUTIVE
    ]


def _context(*, use_layout_state: bool) -> SingletonConsecutiveReshapeContext:
    model_ir = ModelIR("singleton_consecutive_reshape_test")
    return SingletonConsecutiveReshapeContext(
        model_ir=model_ir,
        layout_state=(
            LayoutState.from_model_ir(model_ir) if use_layout_state else None
        ),
        diagnostics=[],
    )


def _normalize_contract(
    invocation: singleton_consecutive_reshape_orchestration.RecoveryInvocation,
    context: SingletonConsecutiveReshapeContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "target_model_ir"
        if value is context.layout_state:
            return "target_layout_state"
        if value is context.diagnostics:
            return "session.diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_singleton_consecutive_context_and_delegate_are_explicit() -> None:
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
    assert len(helper.body) == 1
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
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )

    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_singleton_consecutive_reshape"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Call)
    assert isinstance(call.args[0].func, ast.Name)
    assert call.args[0].func.id == "ModelIRPassContext"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in call.args[0].keywords
    } == {
        "model_ir": "target_model_ir",
        "layout_state": "target_layout_state",
        "diagnostics": "session.diagnostics",
    }
    assert call.keywords == []


@pytest.mark.parametrize("use_layout_state", [False, True])
def test_singleton_consecutive_preserves_both_target_contracts(
    use_layout_state: bool,
) -> None:
    context = _context(use_layout_state=use_layout_state)
    invocations = build_singleton_consecutive_reshape_invocations(context)

    assert (
        tuple(invocation.pass_id for invocation in invocations)
        == SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS
    )
    shared_contract = (
        ("target_model_ir",),
        {
            "layout_state": "target_layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        invocation.pass_id: _normalize_contract(invocation, context)
        for invocation in invocations
    } == {
        SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS[0]: shared_contract,
        SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS[1]: (
            ("target_model_ir",),
            {
                "include_transpose": False,
                "layout_state": "target_layout_state",
                "diagnostics": "session.diagnostics",
                "state_scope": "state_scope",
            },
        ),
        SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS[2]: shared_contract,
    }

    scopes = [
        dict(invocation.keyword_args)["state_scope"] for invocation in invocations
    ]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_singleton_consecutive_reshape_invocations(context)[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize("use_layout_state", [False, True])
def test_singleton_consecutive_runner_preserves_both_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    use_layout_state: bool,
) -> None:
    context = _context(use_layout_state=use_layout_state)
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    for pass_id in SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS:
        monkeypatch.setattr(
            singleton_consecutive_reshape_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_singleton_consecutive_reshape(context)

    assert [pass_id for pass_id, _ in events] == list(
        SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS
    )
    assert all(scope is events[0][1] for _, scope in events)


def test_singleton_consecutive_runner_returns_three_ordered_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(use_layout_state=True)
    expected_results = (
        {"singleton_changes": 1},
        {"duplicate_changes": 2},
        {"reshape_changes": 3},
    )

    for pass_id, expected_result in zip(
        SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS,
        expected_results,
        strict=True,
    ):
        monkeypatch.setattr(
            singleton_consecutive_reshape_orchestration,
            pass_id,
            lambda *args, _result=expected_result, **kwargs: dict(_result),
        )

    assert run_singleton_consecutive_reshape(context) == expected_results


def test_singleton_consecutive_empty_model_returns_only_zero_mutation_counts() -> None:
    context = _context(use_layout_state=True)

    assert run_singleton_consecutive_reshape(context) == (
        {"rewritten_singleton_channel_layout_transpose_to_reshape": 0},
        {"removed_duplicate_reshape_fanout": 0},
        {
            "removed_noop_reshape_chains": 0,
            "rewritten_consecutive_reshape_passthrough_chains": 0,
            "rewritten_fanout_bypass_reshape_passthrough_chains": 0,
        },
    )


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


def test_singleton_consecutive_retains_very_late_main_results() -> None:
    lowerer, _ = _lowerer_and_helper()
    first_index, second_index = _main_invocation_indexes(lowerer)

    first = lowerer.body[first_index]
    assert isinstance(first, ast.Assign)
    assert len(first.targets) == 1
    assert isinstance(first.targets[0], ast.Name)
    assert first.targets[0].id == (
        "_very_late_singleton_consecutive_reshape_results"
    )
    assert isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name)
    assert first.value.func.id == SINGLETON_CONSECUTIVE
    assert [_expression_path(argument) for argument in first.value.args] == [
        "model_ir",
        "session.layout_state",
    ]
    assert first.value.keywords == []

    predecessor = lowerer.body[first_index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert len(predecessor.targets) == 1
    assert isinstance(predecessor.targets[0], ast.Name)
    assert predecessor.targets[0].id == (
        "_very_late_pad_instancenorm_layout_results"
    )

    successor = lowerer.body[first_index + 1]
    assert isinstance(successor, ast.Assign)
    assert isinstance(successor.targets[0], ast.Name)
    assert successor.targets[0].id == "_very_late_layout_broadcast_results"
    assert _statement_call_name(successor) == (
        "run_very_late_layout_broadcast_cleanup"
    )

    second = lowerer.body[second_index]
    assert isinstance(second, ast.Assign)
    assert len(second.targets) == 1
    second_target = second.targets[0]
    assert isinstance(second_target, ast.Tuple)
    assert [
        element.id
        for element in second_target.elts
        if isinstance(element, ast.Name)
    ] == [
        "shared_singleton_channel_stats",
        "shared_duplicate_fanout_stats",
        "shared_consecutive_reshape_stats",
    ]

    fallback_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SINGLETON_CONSECUTIVE
        and [_expression_path(argument) for argument in node.args]
        == ["fallback_ir", None]
    ]
    assert len(fallback_calls) == 1
    fallback_call = fallback_calls[0]
    assert any(
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_fallback_singleton_consecutive_reshape_results"
        and statement.value is fallback_call
        for statement in ast.walk(lowerer)
    )


def test_singleton_consecutive_preserves_both_main_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = _main_invocation_indexes(lowerer)

    assert len(invocation_indexes) == 2
    first_index, second_index = invocation_indexes
    first_preceding = lowerer.body[first_index - 1]
    assert isinstance(first_preceding, ast.Assign)
    assert len(first_preceding.targets) == 1
    assert isinstance(first_preceding.targets[0], ast.Name)
    assert first_preceding.targets[0].id == (
        "_very_late_pad_instancenorm_layout_results"
    )
    assert _statement_call_name(first_preceding) == (
        "run_very_late_pad_instancenorm_layout_cleanup"
    )
    first_following = lowerer.body[first_index + 1]
    assert isinstance(first_following, ast.Assign)
    assert isinstance(first_following.targets[0], ast.Name)
    assert first_following.targets[0].id == (
        "_very_late_layout_broadcast_results"
    )
    assert _statement_call_name(first_following) == (
        "run_very_late_layout_broadcast_cleanup"
    )

    assert _statement_call_name(lowerer.body[second_index - 1]) == (
        "run_indexed_binary_layout_adapter_cleanup"
    )
    second_following = lowerer.body[second_index + 1]
    assert isinstance(second_following, ast.If)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_stats_have_positive_count"
        for node in ast.walk(second_following.test)
    )


def test_singleton_consecutive_preserves_fallback_guard_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    fallback_guard = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, (ast.Assign, ast.Expr))
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == SINGLETON_CONSECUTIVE
            for statement in node.body
        )
    )
    invocation_index = next(
        index
        for index, statement in enumerate(fallback_guard.body)
        if isinstance(statement, (ast.Assign, ast.Expr))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINGLETON_CONSECUTIVE
    )

    invocation = fallback_guard.body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        "_fallback_singleton_consecutive_reshape_results"
    )

    assert ast.unparse(fallback_guard.test) == (
        "int(fallback_norm_stats.get("
        "'optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains', 0)) > 0"
    )
    assert _statement_call_name(fallback_guard.body[invocation_index - 1]) == (
        "run_indexed_binary_layout_adapter_cleanup"
    )
    reconciliation = fallback_guard.body[invocation_index + 1]
    assert isinstance(reconciliation, ast.Expr)
    assert ast.unparse(reconciliation) == (
        "session.record_phase_result('shape_topology.fallback.norm', "
        "run_static_shape_topology_reconciliation(fallback_ir))"
    )


def test_singleton_consecutive_phase_imports_owners_without_lowerer() -> None:
    tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
    assert {
        "onnx2tf.tflite_builder.passes.graph_cleanup",
        "onnx2tf.tflite_builder.passes.singleton_reshape_layout",
    } <= imported_modules
