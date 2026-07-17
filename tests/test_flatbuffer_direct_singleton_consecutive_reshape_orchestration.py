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


def _main_invocation_indexes(lowerer: ast.FunctionDef) -> list[int]:
    return [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
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
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_singleton_consecutive_reshape"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Call)
    assert isinstance(call.args[0].func, ast.Name)
    assert call.args[0].func.id == "SingletonConsecutiveReshapeContext"
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
