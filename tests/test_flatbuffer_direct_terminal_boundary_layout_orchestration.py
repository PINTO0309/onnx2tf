from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import terminal_boundary_layout_orchestration
from onnx2tf.tflite_builder.passes.terminal_boundary_layout_orchestration import (
    TERMINAL_BOUNDARY_LAYOUT_PASS_IDS,
    TerminalBoundaryLayoutContext,
    build_terminal_boundary_layout_invocations,
    run_terminal_boundary_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PHASE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_boundary_layout_orchestration.py"
)
TERMINAL_BOUNDARY = "_run_terminal_boundary_layout_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == TERMINAL_BOUNDARY
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


def _context(*, use_layout_state: bool) -> TerminalBoundaryLayoutContext:
    model_ir = ModelIR("terminal_boundary_layout_test")
    return TerminalBoundaryLayoutContext(
        model_ir=model_ir,
        layout_state=(
            LayoutState.from_model_ir(model_ir) if use_layout_state else None
        ),
        diagnostics=[],
    )


def _normalize_contract(
    invocation: terminal_boundary_layout_orchestration.RecoveryInvocation,
    context: TerminalBoundaryLayoutContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "layout_state"
        if value is context.diagnostics:
            return "diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_terminal_boundary_context_and_delegate_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert helper.args.kwonlyargs == []
    assert helper.args.defaults == []
    assert helper.args.kw_defaults == []
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
    assert statement.value is not None
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_terminal_boundary_layout"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "terminal_boundary_layout_context"
    assert call.keywords == []

    context_assignment = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "terminal_boundary_layout_context"
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize("use_layout_state", [False, True])
def test_terminal_boundary_preserves_all_owner_contracts(
    use_layout_state: bool,
) -> None:
    context = _context(use_layout_state=use_layout_state)
    invocations = build_terminal_boundary_layout_invocations(context)

    assert (
        tuple(invocation.pass_id for invocation in invocations)
        == TERMINAL_BOUNDARY_LAYOUT_PASS_IDS
    )
    expected_contract = (
        ("model_ir",),
        {
            "layout_state": "layout_state",
            "diagnostics": "diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        invocation.pass_id: _normalize_contract(invocation, context)
        for invocation in invocations
    } == {pass_id: expected_contract for pass_id in TERMINAL_BOUNDARY_LAYOUT_PASS_IDS}

    scopes = [
        dict(invocation.keyword_args)["state_scope"] for invocation in invocations
    ]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_terminal_boundary_layout_invocations(context)[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize("use_layout_state", [False, True])
def test_terminal_boundary_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
    use_layout_state: bool,
) -> None:
    context = _context(use_layout_state=use_layout_state)
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    for pass_id in TERMINAL_BOUNDARY_LAYOUT_PASS_IDS:
        monkeypatch.setattr(
            terminal_boundary_layout_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_terminal_boundary_layout(context)

    assert [pass_id for pass_id, _ in events] == list(TERMINAL_BOUNDARY_LAYOUT_PASS_IDS)
    assert all(scope is events[0][1] for _, scope in events)


def test_terminal_boundary_propagates_ordered_results_to_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(use_layout_state=True)
    expected = tuple(
        {f"{pass_id}_mutations": index}
        for index, pass_id in enumerate(TERMINAL_BOUNDARY_LAYOUT_PASS_IDS, start=1)
    )
    for pass_id, result in zip(TERMINAL_BOUNDARY_LAYOUT_PASS_IDS, expected):
        monkeypatch.setattr(
            terminal_boundary_layout_orchestration,
            pass_id,
            lambda *args, result=result, **kwargs: result,
        )

    assert run_terminal_boundary_layout(context) == expected

    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    delegate = helper.body[0]
    assert isinstance(delegate, ast.Return)
    assert isinstance(delegate.value, ast.Call)
    assert isinstance(delegate.value.func, ast.Name)
    assert delegate.value.func.id == "run_terminal_boundary_layout"
    assert [ast.unparse(argument) for argument in delegate.value.args] == [
        "terminal_boundary_layout_context"
    ]
    assert delegate.value.keywords == []

    invocations = [
        (index, statement)
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_boundary_layout_results"
    ]
    assert len(invocations) == 1
    invocation_index, invocation = invocations[0]
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == TERMINAL_BOUNDARY
    assert invocation.value.args == []
    assert invocation.value.keywords == []

    predecessor = lowerer.body[invocation_index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert len(predecessor.targets) == 1
    assert isinstance(predecessor.targets[0], ast.Name)
    assert predecessor.targets[0].id == "_terminal_instancenorm_dualstats_stats"
    successor = lowerer.body[invocation_index + 1]
    assert isinstance(successor, ast.If)
    assert isinstance(successor.test, ast.Name)
    assert successor.test.id == "optimize_layout_transpose_chains"


def test_terminal_boundary_has_one_argument_free_production_call() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_BOUNDARY
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert invocations[0].keywords == []


def test_terminal_boundary_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_boundary_layout_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TERMINAL_BOUNDARY
    )

    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Assign)
    assert len(previous_boundary.targets) == 1
    assert isinstance(previous_boundary.targets[0], ast.Name)
    assert previous_boundary.targets[0].id == (
        "_terminal_instancenorm_dualstats_stats"
    )
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert previous_boundary.value.func.id == (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )

    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.If)
    assert isinstance(next_boundary.test, ast.Name)
    assert next_boundary.test.id == "optimize_layout_transpose_chains"


def test_terminal_boundary_phase_imports_owners_without_lowerer() -> None:
    tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
    assert {
        "onnx2tf.tflite_builder.passes.boundary_input_layout",
        "onnx2tf.tflite_builder.passes.dual_mul_concat_layout",
        "onnx2tf.tflite_builder.passes.layout_transpose",
        "onnx2tf.tflite_builder.passes.pad_layout",
    } <= imported_modules
