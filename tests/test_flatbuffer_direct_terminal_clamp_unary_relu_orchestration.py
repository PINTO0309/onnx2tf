from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_clamp_unary_relu_orchestration,
)
from onnx2tf.tflite_builder.passes.terminal_clamp_unary_relu_orchestration import (
    TERMINAL_CLAMP_UNARY_RELU_PASS_IDS,
    TerminalClampUnaryReLUContext,
    build_terminal_clamp_unary_relu_invocations,
    run_terminal_clamp_unary_relu,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_clamp_unary_relu_orchestration.py"
)
GRAPH_CLEANUP_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "graph_cleanup.py"
)
LAYOUT_TRANSPOSE_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "layout_transpose.py"
)
TERMINAL_CLAMP_UNARY_RELU = "_run_terminal_clamp_unary_relu_pass_cluster"
RESULT_TARGET = "_terminal_clamp_unary_relu_results"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _direct_call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    function = statement.value.func
    return function.id if isinstance(function, ast.Name) else None


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


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
        if isinstance(node, ast.FunctionDef) and node.name == TERMINAL_CLAMP_UNARY_RELU
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


def _context() -> TerminalClampUnaryReLUContext:
    model_ir = ModelIR("terminal_clamp_unary_relu_test")
    return TerminalClampUnaryReLUContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: terminal_clamp_unary_relu_orchestration.RecoveryInvocation,
    context: TerminalClampUnaryReLUContext,
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


def test_terminal_clamp_unary_relu_is_a_straight_line_scoped_cluster() -> None:
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

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
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
        for statement in helper.body
        for node in ast.walk(statement)
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
    assert loaded_data_names == {"terminal_clamp_unary_relu_context"}


def test_terminal_clamp_unary_relu_preserves_all_cleanup_contracts() -> None:
    context = _context()
    invocations = build_terminal_clamp_unary_relu_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == TERMINAL_CLAMP_UNARY_RELU_PASS_IDS
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
    } == {pass_id: expected_contract for pass_id in TERMINAL_CLAMP_UNARY_RELU_PASS_IDS}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_terminal_clamp_unary_relu_invocations(context)[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_terminal_clamp_unary_relu_child_schemas_and_pruning_are_explicit() -> None:
    context = _context()
    assert tuple(
        invocation.run()
        for invocation in build_terminal_clamp_unary_relu_invocations(context)
    ) == (
        {"rewritten_maximum_minimum_relu0to1_chains": 0},
        {"rewritten_transpose_unary_passthrough_chains": 0},
        {"rewritten_maximum_with_zero_input2_to_relu": 0},
    )

    graph_functions = _functions(GRAPH_CLEANUP_PATH)
    clamp_owner = graph_functions[
        "_optimize_maximum_minimum_relu0to1_chains"
    ]
    unary_owner = _functions(LAYOUT_TRANSPOSE_PATH)[
        "_optimize_transpose_unary_passthrough_chains"
    ]
    maximum_owner = graph_functions[
        "_optimize_maximum_with_zero_input2_to_relu"
    ]

    for owner in (clamp_owner, unary_owner):
        assert sum(
            1
            for statement in owner.body
            if _direct_call_name(statement) == "_prune_unused_tensors"
        ) == 1

    assert not any(
        _direct_call_name(statement) == "_prune_unused_tensors"
        for statement in maximum_owner.body
    )
    guarded_cleanup = [
        statement
        for statement in maximum_owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "rewritten > 0"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(guarded_cleanup) == 1


def test_terminal_clamp_unary_relu_invocation_remains_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_CLAMP_UNARY_RELU
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert invocations[0].keywords == []


def test_terminal_clamp_unary_relu_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and _single_target(statement) == RESULT_TARGET
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TERMINAL_CLAMP_UNARY_RELU
    )

    previous = lowerer.body[invocation_index - 1]
    assert isinstance(previous, ast.If)
    assert isinstance(previous.test, ast.Name)
    assert previous.test.id == "optimize_layout_transpose_chains"
    previous_call = previous.body[-1]
    assert isinstance(previous_call, ast.Assign)
    assert len(previous_call.targets) == 1
    assert isinstance(previous_call.targets[0], ast.Name)
    assert previous_call.targets[0].id == "_terminal_singleton_reshape_results"
    assert isinstance(previous_call.value, ast.Call)
    assert isinstance(previous_call.value.func, ast.Name)
    assert previous_call.value.func.id == "_run_singleton_reshape_layout_pass_cluster"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in previous_call.value.keywords
    } == {
        "include_layout_transpose": True,
        "include_multi_branch_gate": True,
    }

    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_terminal_sinet_layout_recovery_results"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == "_run_sinet_terminal_layout_recovery_sequence"


def test_terminal_clamp_unary_relu_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_terminal_clamp_unary_relu"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "terminal_clamp_unary_relu_context"
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "terminal_clamp_unary_relu_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_terminal_clamp_unary_relu_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    probe_steps = build_terminal_clamp_unary_relu_invocations(context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for step in probe_steps:
        module_name = next(
            name
            for name, value in vars(terminal_clamp_unary_relu_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            terminal_clamp_unary_relu_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_terminal_clamp_unary_relu(context)

    assert events == list(TERMINAL_CLAMP_UNARY_RELU_PASS_IDS)


def test_terminal_clamp_unary_relu_propagates_and_retains_ordered_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    expected_results = (
        {"rewritten_maximum_minimum_relu0to1_chains": 1},
        {"rewritten_transpose_unary_passthrough_chains": 2},
        {"rewritten_maximum_with_zero_input2_to_relu": 3},
    )
    probe_steps = build_terminal_clamp_unary_relu_invocations(context)
    for step, expected in zip(probe_steps, expected_results):
        module_name = next(
            name
            for name, value in vars(
                terminal_clamp_unary_relu_orchestration
            ).items()
            if value is step.callback
        )

        def result(*args: Any, _expected: dict[str, int] = expected, **kwargs: Any):
            return dict(_expected)

        monkeypatch.setattr(
            terminal_clamp_unary_relu_orchestration,
            module_name,
            result,
        )

    assert run_terminal_clamp_unary_relu(context) == expected_results

    runner = _functions(ORCHESTRATION_PATH)["run_terminal_clamp_unary_relu"]
    assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
    assert len(runner.body) == 1
    assert isinstance(runner.body[0], ast.Return)

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    invocation_statement = next(
        statement
        for statement in lowerer.body
        if _direct_call_name(statement) == TERMINAL_CLAMP_UNARY_RELU
    )
    assert _single_target(invocation_statement) == RESULT_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


def test_terminal_clamp_unary_relu_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_clamp_unary_relu_orchestration.py"
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
