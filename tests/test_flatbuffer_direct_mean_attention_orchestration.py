from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import mean_attention_orchestration
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    PREADD_MEAN_ATTENTION_PASS_IDS,
    AttentionRecoveryContext,
    build_preadd_mean_attention_invocations,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS,
    LayoutAttentionQuantizedSuffixContext,
    build_layout_attention_quantized_suffix_invocations,
)
from onnx2tf.tflite_builder.passes.mean_attention_orchestration import (
    MEAN_ATTENTION_BASE_PASS_IDS,
    MEAN_ATTENTION_BASE_TAIL_PASS_IDS,
    MEAN_ATTENTION_CONV_PASS_IDS,
    MEAN_ATTENTION_DEFAULT_PASS_IDS,
    MEAN_ATTENTION_LAYERNORM_PASS_IDS,
    MEAN_ATTENTION_PASS_IDS,
    MEAN_ATTENTION_PREFIX_PASS_IDS,
    MeanAttentionContext,
    build_mean_attention_invocations,
    run_mean_attention,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PHASE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "mean_attention_orchestration.py"
)
LAYOUT_PASS_SET_1_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_pass_set_1_mean_attention_gate_orchestration.py"
)
MEAN_ATTENTION = "_run_mean_attention_layout_pass_cluster"
POLICIES = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
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
        if isinstance(node, ast.FunctionDef) and node.name == MEAN_ATTENTION
    )
    return lowerer, helper


def _layout_pass_set_1_owner_calls(child_owner: str) -> list[ast.Call]:
    tree = ast.parse(
        LAYOUT_PASS_SET_1_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_layout_pass_set_1_mean_attention_gate_cleanup"
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == child_owner
    ]


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected expression: {ast.dump(node)}")


def _direct_call_name(statement: ast.stmt) -> str:
    assert isinstance(statement, (ast.Assign, ast.Expr))
    assert isinstance(statement.value, ast.Call)
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and ast.unparse(call.func) == "session.record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        call = call.args[1]
    assert isinstance(call.func, ast.Name)
    return call.func.id


def _context(*, use_layout_state: bool = False) -> MeanAttentionContext:
    model_ir = ModelIR("mean_attention_test")
    return MeanAttentionContext(
        model_ir=model_ir,
        layout_state=(
            LayoutState.from_model_ir(model_ir) if use_layout_state else None
        ),
        diagnostics=[],
    )


def _expected_ids(
    include_layernorm: bool,
    include_conv_attention: bool,
) -> tuple[str, ...]:
    return (
        *MEAN_ATTENTION_PREFIX_PASS_IDS,
        *(MEAN_ATTENTION_LAYERNORM_PASS_IDS if include_layernorm else ()),
        *MEAN_ATTENTION_BASE_TAIL_PASS_IDS,
        *(MEAN_ATTENTION_CONV_PASS_IDS if include_conv_attention else ()),
    )


def _normalize_contract(
    invocation: mean_attention_orchestration.RecoveryInvocation,
    context: MeanAttentionContext,
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


def test_mean_attention_context_and_delegate_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_layernorm",
        "include_conv_attention",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
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
    assert isinstance(statement, ast.Return)
    assert statement.value is not None
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_mean_attention"
    assert tuple(_expression_path(argument) for argument in call.args) == (
        "mean_attention_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {
        "include_layernorm": "include_layernorm",
        "include_conv_attention": "include_conv_attention",
    }

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "mean_attention_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize(
    ("include_layernorm", "include_conv_attention"),
    POLICIES,
)
def test_mean_attention_preserves_all_policy_contracts(
    include_layernorm: bool,
    include_conv_attention: bool,
) -> None:
    context = _context()
    invocations = build_mean_attention_invocations(
        context,
        include_layernorm=include_layernorm,
        include_conv_attention=include_conv_attention,
    )
    expected_ids = _expected_ids(include_layernorm, include_conv_attention)

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
        build_mean_attention_invocations(
            context,
            include_layernorm=include_layernorm,
            include_conv_attention=include_conv_attention,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_mean_attention_preserves_layout_state_and_named_sequences() -> None:
    context = _context(use_layout_state=True)
    invocations = build_mean_attention_invocations(context)
    scopes = [
        dict(invocation.keyword_args)["state_scope"] for invocation in invocations
    ]

    assert MEAN_ATTENTION_BASE_PASS_IDS == (
        *MEAN_ATTENTION_PREFIX_PASS_IDS,
        *MEAN_ATTENTION_BASE_TAIL_PASS_IDS,
    )
    assert MEAN_ATTENTION_DEFAULT_PASS_IDS == (
        *MEAN_ATTENTION_BASE_PASS_IDS,
        *MEAN_ATTENTION_CONV_PASS_IDS,
    )
    assert MEAN_ATTENTION_PASS_IDS == (
        *MEAN_ATTENTION_PREFIX_PASS_IDS,
        *MEAN_ATTENTION_LAYERNORM_PASS_IDS,
        *MEAN_ATTENTION_BASE_TAIL_PASS_IDS,
        *MEAN_ATTENTION_CONV_PASS_IDS,
    )
    assert all(scope.layout_state is context.layout_state for scope in scopes)


@pytest.mark.parametrize(
    ("include_layernorm", "include_conv_attention"),
    POLICIES,
)
def test_mean_attention_runner_preserves_all_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_layernorm: bool,
    include_conv_attention: bool,
) -> None:
    context = _context(use_layout_state=True)
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    for pass_id in MEAN_ATTENTION_PASS_IDS:
        monkeypatch.setattr(
            mean_attention_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_mean_attention(
        context,
        include_layernorm=include_layernorm,
        include_conv_attention=include_conv_attention,
    )
    expected_ids = _expected_ids(include_layernorm, include_conv_attention)

    assert [pass_id for pass_id, _ in events] == list(expected_ids)
    assert all(scope is events[0][1] for _, scope in events)


def test_mean_attention_propagates_policy_results_to_primary_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(use_layout_state=True)
    result_by_pass_id = {
        pass_id: {f"{pass_id}_mutations": index}
        for index, pass_id in enumerate(MEAN_ATTENTION_PASS_IDS, start=1)
    }
    for pass_id, result in result_by_pass_id.items():
        monkeypatch.setattr(
            mean_attention_orchestration,
            pass_id,
            lambda *args, result=result, **kwargs: result,
        )

    for include_layernorm, include_conv_attention in POLICIES:
        expected_ids = _expected_ids(include_layernorm, include_conv_attention)
        assert run_mean_attention(
            context,
            include_layernorm=include_layernorm,
            include_conv_attention=include_conv_attention,
        ) == tuple(result_by_pass_id[pass_id] for pass_id in expected_ids)

    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    delegate = helper.body[0]
    assert isinstance(delegate, ast.Return)
    assert isinstance(delegate.value, ast.Call)
    assert isinstance(delegate.value.func, ast.Name)
    assert delegate.value.func.id == "run_mean_attention"

    direct_results = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == MEAN_ATTENTION
        ),
        key=lambda node: node.lineno,
    )
    assert len(direct_results) == 1
    assert [result.targets[0].id for result in direct_results] == [
        "_terminal_mean_attention_results",
    ]
    assert [
        {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in result.value.keywords
        }
        for result in direct_results
    ] == [
        {"include_conv_attention": "False"},
    ]
    owner_calls = _layout_pass_set_1_owner_calls("run_mean_attention")
    assert len(owner_calls) == 1
    assert [ast.unparse(argument) for argument in owner_calls[0].args] == [
        "context.pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in owner_calls[0].keywords
    } == {"include_layernorm": "True"}

    callback_assignments = {
        target.id: statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        and target.id
        in {
            "attention_recovery_context",
            "layout_attention_quantized_suffix_context",
        }
    }
    assert set(callback_assignments) == {
        "attention_recovery_context",
        "layout_attention_quantized_suffix_context",
    }
    assert all(
        any(
            keyword.arg == "mean_attention_cluster"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == MEAN_ATTENTION
            for keyword in assignment.value.keywords
        )
        for assignment in callback_assignments.values()
        if isinstance(assignment.value, ast.Call)
    )


def test_mean_attention_preserves_layernorm_conv_policy_and_boundaries() -> None:
    owner_calls = _layout_pass_set_1_owner_calls("run_mean_attention")
    assert len(owner_calls) == 1
    invocation = owner_calls[0]
    assert [ast.unparse(argument) for argument in invocation.args] == [
        "context.pass_context"
    ]
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.keywords
    } == {"include_layernorm": True}


def test_mean_attention_preserves_terminal_base_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    outer_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == MEAN_ATTENTION
            and any(
                keyword.arg == "include_conv_attention" for keyword in node.keywords
            )
            for node in ast.walk(statement)
        )
    )
    guard = lowerer.body[outer_index]
    assert isinstance(guard, ast.If)
    invocation_index = next(
        index
        for index, statement in enumerate(guard.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_mean_attention_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == MEAN_ATTENTION
    )
    invocation = guard.body[invocation_index]

    assert invocation_index == 0
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_conv_attention": False}
    assert _direct_call_name(lowerer.body[outer_index - 1]) == (
        "_run_terminal_boundary_layout_pass_cluster"
    )
    assert _direct_call_name(guard.body[invocation_index + 1]) == (
        "_optimize_batchmatmul_affine_transpose_input_chains"
    )


def test_mean_attention_preserves_both_argument_free_default_callbacks() -> None:
    lowerer, helper = _lowerer_and_helper()
    assignments = {
        target.id: statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
        and target.id
        in {
            "attention_recovery_context",
            "layout_attention_quantized_suffix_context",
        }
    }
    assert set(assignments) == {
        "attention_recovery_context",
        "layout_attention_quantized_suffix_context",
    }
    for assignment in assignments.values():
        assert isinstance(assignment.value, ast.Call)
        callback_keyword = next(
            keyword
            for keyword in assignment.value.keywords
            if keyword.arg == "mean_attention_cluster"
        )
        assert _expression_path(callback_keyword.value) == MEAN_ATTENTION

    def callback() -> None:
        return None

    pass_context = ModelIRPassContext(
        model_ir=ModelIR("mean_attention_callback_test"),
        layout_state=None,
        diagnostics=[],
    )
    attention_context = AttentionRecoveryContext(
        pass_context=pass_context,
        mean_attention_cluster=callback,
        gate_layout_cluster=lambda: None,
        transpose_unary_fanout_cluster=lambda: None,
    )
    attention_invocation = build_preadd_mean_attention_invocations(attention_context)[
        -1
    ]
    suffix_context = LayoutAttentionQuantizedSuffixContext(
        pass_context=pass_context,
        mean_attention_cluster=callback,
        attention_gate_qdq_recovery=lambda: None,
        duplicate_quantized_prelu_cluster=lambda **kwargs: None,
    )
    suffix_invocation = build_layout_attention_quantized_suffix_invocations(
        suffix_context,
        include_duplicate_transpose=True,
    )[3]

    assert PREADD_MEAN_ATTENTION_PASS_IDS[-1] == MEAN_ATTENTION
    assert LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[3] == MEAN_ATTENTION
    for invocation in (attention_invocation, suffix_invocation):
        assert invocation.callback is callback
        assert invocation.args == ()
        assert invocation.keyword_args == ()
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_layernorm",
        "include_conv_attention",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
    ]

    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == MEAN_ATTENTION
    ]
    owner_invocations = _layout_pass_set_1_owner_calls("run_mean_attention")
    assert len(direct_invocations) == 1
    assert len(owner_invocations) == 1


def test_mean_attention_phase_imports_owners_without_lowerer() -> None:
    tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
    assert {
        "onnx2tf.tflite_builder.passes.attention_layout",
        "onnx2tf.tflite_builder.passes.layernorm_layout",
        "onnx2tf.tflite_builder.passes.mean_layout",
        "onnx2tf.tflite_builder.passes.se_layout",
        "onnx2tf.tflite_builder.passes.terminal_mean_layout",
    } <= imported_modules
