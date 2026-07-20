from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_singleton_maxpool_reshape_orchestration,
)
from onnx2tf.tflite_builder.passes.terminal_singleton_maxpool_reshape_orchestration import (
    TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS,
    TerminalSingletonMaxPoolReshapeContext,
    build_terminal_singleton_maxpool_reshape_invocations,
    run_terminal_singleton_maxpool_reshape,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TERMINAL_SINGLETON_MAXPOOL_RESHAPE = "_run_terminal_singleton_maxpool_reshape_pass_pair"
OUTER_OWNER = "run_terminal_fanout_singleton_cleanup"
LOWERER_OWNER = "run_late_final_shape_terminal_fanout_cleanup"
OUTER_TARGET = "_late_final_shape_terminal_fanout_results"
OUTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_fanout_singleton_orchestration.py"
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
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_SINGLETON_MAXPOOL_RESHAPE
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context() -> TerminalSingletonMaxPoolReshapeContext:
    model_ir = ModelIR("terminal_singleton_maxpool_reshape_test")
    return TerminalSingletonMaxPoolReshapeContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: terminal_singleton_maxpool_reshape_orchestration.RecoveryInvocation,
    context: TerminalSingletonMaxPoolReshapeContext,
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


def test_terminal_singleton_maxpool_reshape_is_straight_line_scoped() -> None:
    _, helper = _lowerer_and_helper()
    control_flow_nodes = (
        ast.AsyncFor,
        ast.AsyncWith,
        ast.For,
        ast.If,
        ast.Match,
        ast.Try,
        ast.While,
        ast.With,
    )

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 6
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    assert not any(isinstance(node, control_flow_nodes) for node in ast.walk(helper))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )

    called_names = {
        node.func.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    loaded_data_names = {
        node.id
        for statement in helper.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in called_names
    }
    assert loaded_data_names == {"terminal_singleton_maxpool_reshape_context"}


def test_terminal_singleton_maxpool_reshape_preserves_cleanup_contracts() -> None:
    context = _context()
    invocations = build_terminal_singleton_maxpool_reshape_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS
    )
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
    } == {
        pass_id: expected_contract
        for pass_id in TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS
    }

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_terminal_singleton_maxpool_reshape_invocations(context)[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_terminal_singleton_maxpool_reshape_invocation_is_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_SINGLETON_MAXPOOL_RESHAPE
    ]

    assert invocations == []
    owner = next(
        node
        for node in ast.parse(OUTER_OWNER_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == OUTER_OWNER
    )
    owner_invocations = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_terminal_singleton_maxpool_reshape"
    ]
    assert len(owner_invocations) == 1
    assert [ast.unparse(argument) for argument in owner_invocations[0].args] == [
        "context"
    ]
    assert owner_invocations[0].keywords == []


def test_terminal_singleton_maxpool_reshape_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == LOWERER_OWNER
    )
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        OUTER_TARGET
    )

    previous = lowerer.body[invocation_index - 1]
    assert isinstance(previous, ast.Assign)
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_late_affine_optional_fanout_results"

    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.If)
    assert isinstance(following.test, ast.Name)
    assert following.test.id == "optimize_layout_transpose_chains"
    following_call = following.body[0]
    assert isinstance(following_call, ast.Assign)
    assert len(following_call.targets) == 1
    assert isinstance(following_call.targets[0], ast.Name)
    assert following_call.targets[0].id == (
        "_terminal_convpool_output_passthrough_stats"
    )
    assert isinstance(following_call.value, ast.Call)
    assert isinstance(following_call.value.func, ast.Name)
    assert (
        following_call.value.func.id
        == "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
    )


def test_terminal_singleton_maxpool_reshape_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_terminal_singleton_maxpool_reshape"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "terminal_singleton_maxpool_reshape_context"
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "terminal_singleton_maxpool_reshape_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_terminal_singleton_maxpool_reshape_runner_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    probe_steps = build_terminal_singleton_maxpool_reshape_invocations(context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for step in probe_steps:
        module_name = next(
            name
            for name, value in vars(
                terminal_singleton_maxpool_reshape_orchestration
            ).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            terminal_singleton_maxpool_reshape_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_terminal_singleton_maxpool_reshape(context)

    assert events == list(TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS)


def test_terminal_singleton_maxpool_reshape_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_singleton_maxpool_reshape_orchestration.py"
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
