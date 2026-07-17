from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import channel_shuffle_gather_orchestration
from onnx2tf.tflite_builder.passes.channel_shuffle_gather_orchestration import (
    CHANNEL_SHUFFLE_GATHER_BASE_PASS_IDS,
    CHANNEL_SHUFFLE_GATHER_DEFAULT_PASS_IDS,
    CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS,
    CHANNEL_SHUFFLE_GATHER_PASS_IDS,
    CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS,
    ChannelShuffleGatherContext,
    build_channel_shuffle_gather_invocations,
    run_channel_shuffle_gather,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LAYOUT_RECOVERY_PASS_IDS,
    LayoutRecoveryContext,
    build_layout_recovery_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PHASE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "channel_shuffle_gather_orchestration.py"
)
CHANNEL_SHUFFLE_GATHER = "_run_channel_shuffle_gather_layout_pass_cluster"
POLICIES = (
    (False, False, False),
    (False, False, True),
    (False, True, False),
    (False, True, True),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True),
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
        if isinstance(node, ast.FunctionDef) and node.name == CHANNEL_SHUFFLE_GATHER
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


def _context(*, use_layout_state: bool = False) -> ChannelShuffleGatherContext:
    model_ir = ModelIR("channel_shuffle_gather_test")
    return ChannelShuffleGatherContext(
        model_ir=model_ir,
        layout_state=(
            LayoutState.from_model_ir(model_ir) if use_layout_state else None
        ),
        diagnostics=[],
    )


def _expected_ids(
    include_two_way_shuffle: bool,
    include_nhwc_shuffle: bool,
    include_post_gather_cleanup: bool,
) -> tuple[str, ...]:
    return (
        *(
            (CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS[0],)
            if include_two_way_shuffle
            else ()
        ),
        *(
            (CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS[1],)
            if include_nhwc_shuffle
            else ()
        ),
        *CHANNEL_SHUFFLE_GATHER_BASE_PASS_IDS,
        *(CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS if include_post_gather_cleanup else ()),
    )


def _normalize_contract(
    invocation: channel_shuffle_gather_orchestration.RecoveryInvocation,
    context: ChannelShuffleGatherContext,
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


def test_channel_shuffle_gather_context_and_delegate_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_two_way_shuffle",
        "include_nhwc_shuffle",
        "include_post_gather_cleanup",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        True,
        True,
        False,
    ]
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
    assert call.func.id == "run_channel_shuffle_gather"
    assert tuple(_expression_path(argument) for argument in call.args) == (
        "channel_shuffle_gather_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {
        "include_two_way_shuffle": "include_two_way_shuffle",
        "include_nhwc_shuffle": "include_nhwc_shuffle",
        "include_post_gather_cleanup": "include_post_gather_cleanup",
    }

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "channel_shuffle_gather_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize(
    (
        "include_two_way_shuffle",
        "include_nhwc_shuffle",
        "include_post_gather_cleanup",
    ),
    POLICIES,
)
def test_channel_shuffle_gather_preserves_all_policy_contracts(
    include_two_way_shuffle: bool,
    include_nhwc_shuffle: bool,
    include_post_gather_cleanup: bool,
) -> None:
    context = _context()
    invocations = build_channel_shuffle_gather_invocations(
        context,
        include_two_way_shuffle=include_two_way_shuffle,
        include_nhwc_shuffle=include_nhwc_shuffle,
        include_post_gather_cleanup=include_post_gather_cleanup,
    )
    expected_ids = _expected_ids(
        include_two_way_shuffle,
        include_nhwc_shuffle,
        include_post_gather_cleanup,
    )

    assert tuple(invocation.pass_id for invocation in invocations) == expected_ids
    expected_contract = (
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        invocation.pass_id: _normalize_contract(invocation, context)
        for invocation in invocations
    } == {pass_id: expected_contract for pass_id in expected_ids}

    scopes = [
        dict(invocation.keyword_args)["state_scope"] for invocation in invocations
    ]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_channel_shuffle_gather_invocations(
            context,
            include_two_way_shuffle=include_two_way_shuffle,
            include_nhwc_shuffle=include_nhwc_shuffle,
            include_post_gather_cleanup=include_post_gather_cleanup,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_channel_shuffle_gather_preserves_layout_state_contract() -> None:
    context = _context(use_layout_state=True)
    invocations = build_channel_shuffle_gather_invocations(context)
    scopes = [
        dict(invocation.keyword_args)["state_scope"] for invocation in invocations
    ]

    assert CHANNEL_SHUFFLE_GATHER_DEFAULT_PASS_IDS == (
        *CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS,
        *CHANNEL_SHUFFLE_GATHER_BASE_PASS_IDS,
    )
    assert CHANNEL_SHUFFLE_GATHER_PASS_IDS == (
        *CHANNEL_SHUFFLE_GATHER_DEFAULT_PASS_IDS,
        *CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS,
    )
    assert all(scope.layout_state is context.layout_state for scope in scopes)


@pytest.mark.parametrize(
    (
        "include_two_way_shuffle",
        "include_nhwc_shuffle",
        "include_post_gather_cleanup",
    ),
    POLICIES,
)
def test_channel_shuffle_gather_runner_preserves_all_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_two_way_shuffle: bool,
    include_nhwc_shuffle: bool,
    include_post_gather_cleanup: bool,
) -> None:
    context = _context(use_layout_state=True)
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    for pass_id in CHANNEL_SHUFFLE_GATHER_PASS_IDS:
        monkeypatch.setattr(
            channel_shuffle_gather_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_channel_shuffle_gather(
        context,
        include_two_way_shuffle=include_two_way_shuffle,
        include_nhwc_shuffle=include_nhwc_shuffle,
        include_post_gather_cleanup=include_post_gather_cleanup,
    )
    expected_ids = _expected_ids(
        include_two_way_shuffle,
        include_nhwc_shuffle,
        include_post_gather_cleanup,
    )

    assert [pass_id for pass_id, _ in events] == list(expected_ids)
    assert all(scope is events[0][1] for _, scope in events)


def test_channel_shuffle_gather_preserves_full_post_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == CHANNEL_SHUFFLE_GATHER
            and any(
                keyword.arg == "include_post_gather_cleanup"
                for keyword in node.keywords
            )
            for node in ast.walk(statement)
        )
    )
    invocation_index = next(
        index
        for index, statement in enumerate(guard.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == CHANNEL_SHUFFLE_GATHER
    )
    invocation = guard.body[invocation_index]

    assert isinstance(guard.test, ast.Name)
    assert guard.test.id == "optimize_layout_transpose_chains"
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_post_gather_cleanup": True}
    assert _direct_call_name(guard.body[invocation_index - 1]) == (
        "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"
    )
    assert _direct_call_name(guard.body[invocation_index + 1]) == (
        "_run_preadd_mean_attention_recovery_sequence"
    )


def test_channel_shuffle_gather_preserves_late_base_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == CHANNEL_SHUFFLE_GATHER
    )
    invocation = lowerer.body[invocation_index]

    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {
        "include_two_way_shuffle": False,
        "include_nhwc_shuffle": False,
    }
    assert _direct_call_name(lowerer.body[invocation_index - 1]) == (
        "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains"
    )
    assert _direct_call_name(lowerer.body[invocation_index + 1]) == (
        "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains"
    )


def test_channel_shuffle_gather_preserves_argument_free_default_callback() -> None:
    lowerer, helper = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "layout_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    callback_keyword = next(
        keyword
        for keyword in context_assignment.value.keywords
        if keyword.arg == "channel_shuffle_gather_cluster"
    )
    assert _expression_path(callback_keyword.value) == CHANNEL_SHUFFLE_GATHER

    def callback() -> None:
        return None

    model_ir = ModelIR("channel_shuffle_gather_callback_test")
    context = LayoutRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=[],
        ),
        boundary_batchmatmul_unary_cluster=lambda: None,
        pre_concat_cleanup=lambda *args, **kwargs: None,
        channel_shuffle_gather_cluster=callback,
    )
    invocation = build_layout_recovery_invocations(context)[-1]

    assert LAYOUT_RECOVERY_PASS_IDS[-1] == CHANNEL_SHUFFLE_GATHER
    assert invocation.callback is callback
    assert invocation.args == ()
    assert invocation.keyword_args == ()
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_two_way_shuffle",
        "include_nhwc_shuffle",
        "include_post_gather_cleanup",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        True,
        True,
        False,
    ]

    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHANNEL_SHUFFLE_GATHER
    ]
    assert len(direct_invocations) == 2


def test_channel_shuffle_gather_phase_imports_owners_without_lowerer() -> None:
    tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
    assert {
        "onnx2tf.tflite_builder.passes.channel_shuffle",
        "onnx2tf.tflite_builder.passes.layout_transpose",
    } <= imported_modules
