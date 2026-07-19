from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    sinet_terminal_layout_recovery_orchestration,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS,
    SINetTerminalLayoutRecoveryContext,
    build_sinet_terminal_layout_recovery_invocations,
    run_sinet_terminal_layout_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PHASE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "sinet_terminal_layout_recovery_orchestration.py"
)
SINET_PREADD_RESIZE = "_run_sinet_preadd_resize_recovery_sequence"
SINET_TERMINAL = "_run_sinet_terminal_layout_recovery_sequence"


def _noop_recovery() -> None:
    pass


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
        if isinstance(node, ast.FunctionDef) and node.name == SINET_TERMINAL
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


def _direct_call_name(statement: ast.stmt) -> str:
    assert isinstance(statement, (ast.Assign, ast.Expr))
    assert isinstance(statement.value, ast.Call)
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        call = call.args[1]
    assert isinstance(call.func, ast.Name)
    return call.func.id


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _context(
    *,
    preadd_resize_recovery: Callable[[], Any] | None = None,
) -> SINetTerminalLayoutRecoveryContext:
    model_ir = ModelIR("sinet_terminal_layout_recovery_test")
    if preadd_resize_recovery is None:
        preadd_resize_recovery = _noop_recovery
    return SINetTerminalLayoutRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        preadd_resize_recovery=preadd_resize_recovery,
    )


def _normalize_new_contract(
    invocation: sinet_terminal_layout_recovery_orchestration.RecoveryInvocation,
    context: SINetTerminalLayoutRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.pass_context.model_ir:
            return "model_ir"
        if value is context.pass_context.layout_state:
            return "session.layout_state"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_sinet_terminal_layout_recovery_is_a_straight_line_closure() -> None:
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
    assert helper.end_lineno - helper.lineno + 1 == 4
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
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
        for node in ast.walk(helper)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in called_names
    }
    assert loaded_data_names == {
        "Any",
        "Tuple",
        "sinet_terminal_layout_recovery_context",
    }


def test_sinet_terminal_layout_recovery_preserves_all_call_contracts() -> None:
    context = _context()
    invocations = build_sinet_terminal_layout_recovery_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        "_optimize_sinet_shuffle_residual_transpose_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        SINET_PREADD_RESIZE: ((), {}),
        "_optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains": (
            ("model_ir",),
            {},
        ),
    }
    assert invocations[1].callback is context.preadd_resize_recovery


def test_sinet_terminal_layout_recovery_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SINET_TERMINAL
    ]

    assert len(invocations) == 2
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)


def test_sinet_terminal_layout_recovery_preserves_all_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        in {
            "_terminal_sinet_layout_recovery_results",
            "_very_late_sinet_layout_recovery_results",
        }
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINET_TERMINAL
    ]

    assert len(invocation_indexes) == 2
    assert [
        lowerer.body[index].targets[0].id
        for index in invocation_indexes
        if isinstance(lowerer.body[index], ast.Assign)
        and isinstance(lowerer.body[index].targets[0], ast.Name)
    ] == [
        "_terminal_sinet_layout_recovery_results",
        "_very_late_sinet_layout_recovery_results",
    ]
    observed: list[tuple[str, str]] = []
    assigned_boundary_targets: list[str] = []
    for index in invocation_indexes:
        previous = lowerer.body[index - 1]
        following = lowerer.body[index + 1]
        for boundary in (previous, following):
            assert isinstance(boundary, (ast.Assign, ast.Expr))
            assert isinstance(boundary.value, ast.Call)
            _direct_call_name(boundary)
            if isinstance(boundary, ast.Assign):
                assert len(boundary.targets) == 1
                assert isinstance(boundary.targets[0], ast.Name)
                assigned_boundary_targets.append(boundary.targets[0].id)
        observed.append((_direct_call_name(previous), _direct_call_name(following)))

    assert _phase_id(lowerer.body[invocation_indexes[0] + 1]) == (
        "cleanup.terminal.sinet_hardswish_se"
    )

    assert observed == [
        (
            "_run_terminal_clamp_unary_relu_pass_cluster",
            "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains",
        ),
        (
            "_run_indexed_shape_convergence_cleanup",
            SINET_PREADD_RESIZE,
        ),
    ]
    assert assigned_boundary_targets == [
        "_terminal_clamp_unary_relu_results",
        "_post_terminal_indexed_shape_convergence_stats",
        "_very_late_sinet_preadd_resize_results",
    ]


def test_sinet_terminal_layout_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_sinet_terminal_layout_recovery"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "sinet_terminal_layout_recovery_context"
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "sinet_terminal_layout_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    assert isinstance(context_assignment.value.func, ast.Name)
    assert context_assignment.value.func.id == "SINetTerminalLayoutRecoveryContext"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in context_assignment.value.keywords
    } == {
        "pass_context": "session.model_ir_pass_context",
        "preadd_resize_recovery": SINET_PREADD_RESIZE,
    }


def test_sinet_terminal_layout_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    context = _context(
        preadd_resize_recovery=recorder(SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS[1])
    )
    probe_steps = build_sinet_terminal_layout_recovery_invocations(context)
    for step in (probe_steps[0], probe_steps[2]):
        module_name = next(
            name
            for name, value in vars(
                sinet_terminal_layout_recovery_orchestration
            ).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            sinet_terminal_layout_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_sinet_terminal_layout_recovery(context)

    assert events == list(SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS)


def test_sinet_terminal_layout_propagates_and_retains_ordered_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_results = (
        {"shuffle_residual_mutations": 1},
        {"preadd_resize_mutations": 2},
        {"terminal_affine_prelu_mutations": 3},
    )
    context = _context(
        preadd_resize_recovery=lambda: dict(expected_results[1]),
    )
    monkeypatch.setattr(
        sinet_terminal_layout_recovery_orchestration,
        "optimize_sinet_shuffle_residual_transpose_chains",
        lambda *args, **kwargs: dict(expected_results[0]),
    )
    monkeypatch.setattr(
        sinet_terminal_layout_recovery_orchestration,
        "optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains",
        lambda *args, **kwargs: dict(expected_results[2]),
    )

    assert run_sinet_terminal_layout_recovery(context) == expected_results

    phase_tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
    phase_runner = next(
        node
        for node in phase_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_sinet_terminal_layout_recovery"
    )
    assert ast.unparse(phase_runner.returns) == "Tuple[Any, ...]"
    assert len(phase_runner.body) == 1
    assert isinstance(phase_runner.body[0], ast.Return)

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Any, ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)

    direct_results = sorted(
        (
            statement
            for statement in ast.walk(lowerer)
            if isinstance(statement, (ast.Assign, ast.Expr))
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == SINET_TERMINAL
        ),
        key=lambda statement: statement.lineno,
    )
    assert len(direct_results) == 2
    assert all(isinstance(statement, ast.Assign) for statement in direct_results)
    assert [
        statement.targets[0].id
        for statement in direct_results
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ] == [
        "_terminal_sinet_layout_recovery_results",
        "_very_late_sinet_layout_recovery_results",
    ]
    assert all(statement.value.args == [] for statement in direct_results)
    assert all(statement.value.keywords == [] for statement in direct_results)

    first_index = lowerer.body.index(direct_results[0])
    second_index = lowerer.body.index(direct_results[1])
    first_previous = lowerer.body[first_index - 1]
    first_following = lowerer.body[first_index + 1]
    second_previous = lowerer.body[second_index - 1]
    second_following = lowerer.body[second_index + 1]
    assert _direct_call_name(first_previous) == (
        "_run_terminal_clamp_unary_relu_pass_cluster"
    )
    assert _direct_call_name(first_following) == (
        "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
    )
    assert isinstance(second_previous, ast.Assign)
    assert len(second_previous.targets) == 1
    assert isinstance(second_previous.targets[0], ast.Name)
    assert second_previous.targets[0].id == (
        "_post_terminal_indexed_shape_convergence_stats"
    )
    assert _direct_call_name(second_following) == SINET_PREADD_RESIZE


def test_sinet_terminal_layout_module_does_not_import_lowerer() -> None:
    tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
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
