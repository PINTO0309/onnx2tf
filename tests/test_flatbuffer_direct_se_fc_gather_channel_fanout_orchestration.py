from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    se_fc_gather_channel_fanout_orchestration,
)
from onnx2tf.tflite_builder.passes.se_fc_gather_channel_fanout_orchestration import (
    SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS,
    SEFCGatherChannelFanoutContext,
    build_se_fc_gather_channel_fanout_invocations,
    run_se_fc_gather_channel_fanout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SE_FC_GATHER = "_run_se_fc_gather_channel_fanout_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == SE_FC_GATHER
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
    assert isinstance(statement, (ast.Expr, ast.Assign))
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    return statement.value.func.id


def _direct_invocation_index(statements: list[ast.stmt]) -> int:
    return next(
        index
        for index, statement in enumerate(statements)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SE_FC_GATHER
    )


def _context(*, use_layout_state: bool) -> SEFCGatherChannelFanoutContext:
    model_ir = ModelIR("se_fc_gather_channel_fanout_test")
    return SEFCGatherChannelFanoutContext(
        model_ir=model_ir,
        layout_state=(
            LayoutState.from_model_ir(model_ir) if use_layout_state else None
        ),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: se_fc_gather_channel_fanout_orchestration.RecoveryInvocation,
    context: SEFCGatherChannelFanoutContext,
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


def test_se_fc_gather_signature_context_and_delegate_are_explicit() -> None:
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
    assert call.func.id == "run_se_fc_gather_channel_fanout"
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
def test_se_fc_gather_preserves_both_target_contract_forms(
    use_layout_state: bool,
) -> None:
    context = _context(use_layout_state=use_layout_state)
    invocations = build_se_fc_gather_channel_fanout_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS
    )
    expected_contract = (
        ("target_model_ir",),
        {
            "layout_state": "target_layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        pass_id: expected_contract for pass_id in SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS
    }

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert scopes[0] is scopes[1]
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_se_fc_gather_channel_fanout_invocations(context)[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize("use_layout_state", [False, True])
def test_se_fc_gather_runner_preserves_both_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    use_layout_state: bool,
) -> None:
    context = _context(use_layout_state=use_layout_state)
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    for pass_id in SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS:
        monkeypatch.setattr(
            se_fc_gather_channel_fanout_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_se_fc_gather_channel_fanout(context)

    assert [pass_id for pass_id, _ in events] == list(
        SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS
    )
    assert events[0][1] is events[1][1]


def test_se_fc_gather_runner_returns_both_ordered_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(use_layout_state=True)
    expected_results = (
        {"optimized_transpose_se_fc_mul_prepost_nhwc_chains": 1},
        {"optimized_transpose_gather_transpose_nhwc_channel_chains": 2},
    )

    for pass_id, expected_result in zip(
        SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS,
        expected_results,
        strict=True,
    ):
        monkeypatch.setattr(
            se_fc_gather_channel_fanout_orchestration,
            pass_id,
            lambda *args, _result=expected_result, **kwargs: dict(_result),
        )

    assert run_se_fc_gather_channel_fanout(context) == expected_results


def test_se_fc_gather_preserves_both_production_target_forms() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SE_FC_GATHER
    ]
    invocations.sort(key=lambda call: call.lineno)

    assert [
        tuple(_expression_path(argument) for argument in invocation.args)
        for invocation in invocations
    ] == [
        ("fallback_ir", None),
        ("model_ir", "session.layout_state"),
    ]
    assert all(invocation.keywords == [] for invocation in invocations)


def test_se_fc_gather_preserves_fallback_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    fallback_block = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == SE_FC_GATHER
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "fallback_ir"
            for node in ast.walk(statement)
        )
    )
    invocation_index = _direct_invocation_index(fallback_block.body)

    assert _direct_call_name(fallback_block.body[invocation_index - 1]) == (
        "_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains"
    )
    guard = fallback_block.body[invocation_index + 1]
    assert isinstance(guard, ast.If)
    assert len(guard.body) == 1
    assert _direct_call_name(guard.body[0]) == "_reconcile_static_tensor_shapes"


def test_se_fc_gather_preserves_main_model_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = _direct_invocation_index(lowerer.body)

    assert _direct_call_name(lowerer.body[invocation_index - 1]) == (
        "_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains"
    )
    guard = lowerer.body[invocation_index + 1]
    assert isinstance(guard, ast.If)
    assert len(guard.body) == 1
    assert _direct_call_name(guard.body[0]) == "_reconcile_static_tensor_shapes"


def test_terminal_se_fc_gather_reconciles_only_after_change_or_prune() -> None:
    lowerer, _ = _lowerer_and_helper()
    helper_name = "_run_se_fc_gather_channel_fanout_pass_cluster"
    sinet_owner = (
        "_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains"
    )
    expected_counters = {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains",
        "optimized_transpose_se_fc_mul_prepost_nhwc_chains",
        "optimized_transpose_gather_transpose_nhwc_channel_chains",
    }

    fallback_block = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "fallback_ir"
            for node in ast.walk(statement)
        )
    )

    def assert_boundary(statements: list[ast.stmt], model_name: str) -> None:
        sinet_index = next(
            index
            for index, statement in enumerate(statements)
            if isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == sinet_owner
            and len(statement.value.args) >= 1
            and isinstance(statement.value.args[0], ast.Name)
            and statement.value.args[0].id == model_name
        )

        count_assignment = statements[sinet_index - 1]
        assert isinstance(count_assignment, ast.Assign)
        assert len(count_assignment.targets) == 1
        count_target = count_assignment.targets[0]
        assert isinstance(count_target, ast.Name)
        assert isinstance(count_assignment.value, ast.Call)
        assert isinstance(count_assignment.value.func, ast.Name)
        assert count_assignment.value.func.id == "len"

        cluster_assignment = statements[sinet_index + 1]
        assert isinstance(cluster_assignment, ast.Assign)
        assert isinstance(cluster_assignment.value, ast.Call)
        assert isinstance(cluster_assignment.value.func, ast.Name)
        assert cluster_assignment.value.func.id == helper_name
        assert len(cluster_assignment.targets) == 1
        assert isinstance(cluster_assignment.targets[0], ast.Tuple)
        assert len(cluster_assignment.targets[0].elts) == 2

        guard = statements[sinet_index + 2]
        assert isinstance(guard, ast.If)
        assert guard.orelse == []
        get_calls = [
            node
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        ]
        assert {
            str(call.args[0].value)
            for call in get_calls
            if len(call.args) >= 1 and isinstance(call.args[0], ast.Constant)
        } == expected_counters
        assert len(get_calls) == len(expected_counters)
        guard_names = {
            node.id for node in ast.walk(guard.test) if isinstance(node, ast.Name)
        }
        assert count_target.id in guard_names
        tensor_len_calls = [
            node
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
        ]
        assert len(tensor_len_calls) == 1
        assert len(guard.body) == 1
        reconcile = guard.body[0]
        assert isinstance(reconcile, ast.Expr)
        assert isinstance(reconcile.value, ast.Call)
        assert isinstance(reconcile.value.func, ast.Name)
        assert reconcile.value.func.id == "_reconcile_static_tensor_shapes"
        assert len(reconcile.value.args) == 1
        assert isinstance(reconcile.value.args[0], ast.Name)
        assert reconcile.value.args[0].id == model_name

    assert_boundary(fallback_block.body, "fallback_ir")
    assert_boundary(lowerer.body, "model_ir")


def test_se_fc_gather_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "se_fc_gather_channel_fanout_orchestration.py"
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
