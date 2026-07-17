from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import gate_layout_orchestration
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    ATTENTION_GATE_QDQ_PASS_IDS,
    AttentionRecoveryContext,
    build_attention_gate_qdq_invocations,
)
from onnx2tf.tflite_builder.passes.gate_layout_orchestration import (
    GATE_LAYOUT_PASS_IDS,
    GATE_LAYOUT_REQUIRED_PASS_IDS,
    GateLayoutContext,
    build_gate_layout_invocations,
    run_gate_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PHASE_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "gate_layout_orchestration.py"
)
GATE_LAYOUT = "_run_gate_layout_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == GATE_LAYOUT
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


def _context(*, use_layout_state: bool) -> GateLayoutContext:
    model_ir = ModelIR("gate_layout_test")
    return GateLayoutContext(
        model_ir=model_ir,
        layout_state=(
            LayoutState.from_model_ir(model_ir) if use_layout_state else None
        ),
        diagnostics=[],
    )


def _normalize_contract(
    invocation: gate_layout_orchestration.RecoveryInvocation,
    context: GateLayoutContext,
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


def test_gate_layout_context_and_delegate_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_mixed_attention"
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [True]
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
    assert call.func.id == "run_gate_layout"
    assert tuple(_expression_path(argument) for argument in call.args) == (
        "gate_layout_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {"include_mixed_attention": "include_mixed_attention"}

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gate_layout_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize("use_layout_state", [False, True])
@pytest.mark.parametrize("include_mixed_attention", [False, True])
def test_gate_layout_preserves_both_policy_contracts(
    use_layout_state: bool,
    include_mixed_attention: bool,
) -> None:
    context = _context(use_layout_state=use_layout_state)
    invocations = build_gate_layout_invocations(
        context,
        include_mixed_attention=include_mixed_attention,
    )
    expected_ids = (
        GATE_LAYOUT_PASS_IDS
        if include_mixed_attention
        else GATE_LAYOUT_REQUIRED_PASS_IDS
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
        build_gate_layout_invocations(
            context,
            include_mixed_attention=include_mixed_attention,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize("include_mixed_attention", [False, True])
def test_gate_layout_runner_preserves_both_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_mixed_attention: bool,
) -> None:
    context = _context(use_layout_state=True)
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    for pass_id in GATE_LAYOUT_PASS_IDS:
        monkeypatch.setattr(
            gate_layout_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_gate_layout(
        context,
        include_mixed_attention=include_mixed_attention,
    )
    expected_ids = (
        GATE_LAYOUT_PASS_IDS
        if include_mixed_attention
        else GATE_LAYOUT_REQUIRED_PASS_IDS
    )

    assert [pass_id for pass_id, _ in events] == list(expected_ids)
    assert all(scope is events[0][1] for _, scope in events)


def test_gate_layout_preserves_direct_reduced_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    direct_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(candidate, ast.Expr)
            and isinstance(candidate.value, ast.Call)
            and isinstance(candidate.value.func, ast.Name)
            and candidate.value.func.id == GATE_LAYOUT
            for candidate in statement.body
        )
    )
    invocation_index = next(
        index
        for index, statement in enumerate(direct_guard.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == GATE_LAYOUT
    )
    invocation = direct_guard.body[invocation_index]

    assert isinstance(direct_guard.test, ast.Name)
    assert direct_guard.test.id == "optimize_layout_transpose_chains"
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_mixed_attention": False}
    assert _direct_call_name(direct_guard.body[invocation_index - 1]) == (
        "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"
    )
    following = direct_guard.body[invocation_index + 1]
    assert isinstance(following, ast.For)
    assert isinstance(following.target, ast.Name)
    assert following.target.id == "_"
    assert isinstance(following.iter, ast.Call)
    assert isinstance(following.iter.func, ast.Name)
    assert following.iter.func.id == "range"
    assert len(following.iter.args) == 1
    assert _expression_path(following.iter.args[0]) == 2


def test_gate_layout_preserves_argument_free_full_policy_callback() -> None:
    lowerer, helper = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "attention_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    gate_keyword = next(
        keyword
        for keyword in context_assignment.value.keywords
        if keyword.arg == "gate_layout_cluster"
    )
    assert _expression_path(gate_keyword.value) == GATE_LAYOUT

    def callback() -> None:
        return None

    model_ir = ModelIR("gate_layout_callback_test")
    context = AttentionRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=[],
        ),
        mean_attention_cluster=lambda: None,
        gate_layout_cluster=callback,
        transpose_unary_fanout_cluster=lambda: None,
    )
    invocation = build_attention_gate_qdq_invocations(context)[2]

    assert ATTENTION_GATE_QDQ_PASS_IDS[2] == GATE_LAYOUT
    assert invocation.callback is callback
    assert invocation.args == ()
    assert invocation.keyword_args == ()
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_mixed_attention"
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [True]


def test_gate_layout_phase_imports_owners_without_lowerer() -> None:
    tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
    assert {
        "onnx2tf.tflite_builder.passes.add_concat_suffix_layout",
        "onnx2tf.tflite_builder.passes.attention_layout",
        "onnx2tf.tflite_builder.passes.cost_volume_scatter_layout",
        "onnx2tf.tflite_builder.passes.dual_mul_concat_layout",
        "onnx2tf.tflite_builder.passes.dual_postconv_gate_layout",
        "onnx2tf.tflite_builder.passes.elementwise_gate_layout",
        "onnx2tf.tflite_builder.passes.ndhwc_gate_layout",
        "onnx2tf.tflite_builder.passes.pad_layout",
    } <= imported_modules
