from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import constant_fold_cast_orchestration
from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
    ConstantFoldCastContext,
    build_constant_fold_cast_invocations,
    run_constant_fold_cast,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
CONSTANT_FOLD_CAST = "_run_constant_fold_cast_cleanup_pass_cluster"
LATE_LAYOUT_PARENT = "_run_late_layout_mean_spp_gather_constant_cast_pass_cluster"
VERY_LATE_MODULE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_gather_constant_normalization_orchestration.py"
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
        if isinstance(node, ast.FunctionDef) and node.name == CONSTANT_FOLD_CAST
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


def _parent(lowerer: ast.FunctionDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _direct_invocation_index(parent: ast.FunctionDef) -> int:
    return next(
        index
        for index, statement in enumerate(parent.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == CONSTANT_FOLD_CAST
    )


def _context() -> ConstantFoldCastContext:
    model_ir = ModelIR("constant_fold_cast_test")
    return ConstantFoldCastContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: constant_fold_cast_orchestration.RecoveryInvocation,
    context: ConstantFoldCastContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        if value is context.diagnostics:
            return "session.diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_constant_fold_cast_signature_and_delegate_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == ["state_scope"]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [None]
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
    assert call.func.id == "run_constant_fold_cast"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "constant_fold_cast_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {"state_scope": "state_scope"}


@pytest.mark.parametrize("use_external_scope", [False, True])
def test_constant_fold_cast_preserves_both_scope_forms(
    use_external_scope: bool,
) -> None:
    context = _context()
    external_scope = (
        ModelIRPassStateScope(
            context.model_ir,
            layout_state=context.layout_state,
        )
        if use_external_scope
        else None
    )
    invocations = build_constant_fold_cast_invocations(
        context,
        state_scope=external_scope,
    )

    assert tuple(step.pass_id for step in invocations) == CONSTANT_FOLD_CAST_PASS_IDS
    expected_contract = (
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {pass_id: expected_contract for pass_id in CONSTANT_FOLD_CAST_PASS_IDS}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert scopes[0] is scopes[1]
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    if external_scope is not None:
        assert scopes[0] is external_scope
    else:
        rebuilt_scope = dict(
            build_constant_fold_cast_invocations(context)[0].keyword_args
        )["state_scope"]
        assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize("use_external_scope", [False, True])
def test_constant_fold_cast_runner_preserves_both_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    use_external_scope: bool,
) -> None:
    context = _context()
    external_scope = (
        ModelIRPassStateScope(
            context.model_ir,
            layout_state=context.layout_state,
        )
        if use_external_scope
        else None
    )
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    for pass_id in CONSTANT_FOLD_CAST_PASS_IDS:
        monkeypatch.setattr(
            constant_fold_cast_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_constant_fold_cast(
        context,
        state_scope=external_scope,
    )

    assert [pass_id for pass_id, _ in events] == list(CONSTANT_FOLD_CAST_PASS_IDS)
    assert events[0][1] is events[1][1]
    if external_scope is not None:
        assert events[0][1] is external_scope
    else:
        assert isinstance(events[0][1], ModelIRPassStateScope)


def test_constant_fold_cast_has_one_remaining_external_scope_delegate_call() -> None:
    lowerer, _ = _lowerer_and_helper()
    owners: list[tuple[str, ast.Call]] = []
    for parent in lowerer.body:
        if not isinstance(parent, ast.FunctionDef):
            continue
        for statement in parent.body:
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == CONSTANT_FOLD_CAST
            ):
                continue
            owners.append((parent.name, statement.value))

    assert [name for name, _ in owners] == [LATE_LAYOUT_PARENT]
    for _, invocation in owners:
        assert invocation.args == []
        assert {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in invocation.keywords
        } == {"state_scope": "state_scope"}


def test_constant_fold_cast_builder_is_composed_by_very_late_phase() -> None:
    tree = ast.parse(VERY_LATE_MODULE_PATH.read_text(encoding="utf-8"))
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_very_late_gather_constant_normalization_invocations"
    )
    calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_constant_fold_cast_invocations"
    ]

    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Call)
    assert isinstance(call.args[0].func, ast.Name)
    assert call.args[0].func.id == "ConstantFoldCastContext"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in call.args[0].keywords
    } == {
        "model_ir": "context.model_ir",
        "layout_state": "context.layout_state",
        "diagnostics": "context.diagnostics",
    }
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {"state_scope": "state_scope"}


def test_constant_fold_cast_preserves_late_layout_parent_boundary() -> None:
    lowerer, _ = _lowerer_and_helper()
    parent = _parent(lowerer, LATE_LAYOUT_PARENT)
    invocation_index = _direct_invocation_index(parent)

    assert invocation_index == len(parent.body) - 1
    assert _direct_call_name(parent.body[invocation_index - 1]) == (
        "run_transpose_gather_axis_cleanup"
    )


def test_constant_fold_cast_context_is_explicit() -> None:
    lowerer, _ = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "constant_fold_cast_context"
            for target in statement.targets
        )
    )

    assert isinstance(context_assignment.value, ast.Call)
    assert isinstance(context_assignment.value.func, ast.Name)
    assert context_assignment.value.func.id == "ConstantFoldCastContext"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in context_assignment.value.keywords
    } == {
        "model_ir": "model_ir",
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }


def test_constant_fold_cast_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "constant_fold_cast_orchestration.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.lower_from_onnx2tf"
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "onnx2tf.tflite_builder.lower_from_onnx2tf"
            for alias in node.names
        )
        for node in tree.body
    )
