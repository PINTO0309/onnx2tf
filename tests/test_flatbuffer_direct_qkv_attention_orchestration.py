from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import qkv_attention_orchestration
from onnx2tf.tflite_builder.passes.qkv_attention_orchestration import (
    QKV_ATTENTION_PASS_IDS,
    QKVAttentionContext,
    active_qkv_attention_pass_ids,
    build_qkv_attention_invocations,
    run_qkv_attention,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
QKV_ATTENTION = "_run_qkv_attention_layout_pass_cluster"
DEFAULT_PASS_IDS = QKV_ATTENTION_PASS_IDS[1:]
BRIDGE_ONLY_PASS_IDS = QKV_ATTENTION_PASS_IDS[2:]
LAYOUT_BRIDGE_PASS_IDS = (
    QKV_ATTENTION_PASS_IDS[0],
    QKV_ATTENTION_PASS_IDS[2],
)
PRODUCTION_FORMS = [
    (False, True, DEFAULT_PASS_IDS),
    (False, False, BRIDGE_ONLY_PASS_IDS),
    (True, False, LAYOUT_BRIDGE_PASS_IDS),
]


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
        if isinstance(node, ast.FunctionDef) and node.name == QKV_ATTENTION
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


def _call_name(statement: ast.stmt) -> str:
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    return statement.value.func.id


def _context() -> QKVAttentionContext:
    model_ir = ModelIR("qkv_attention_test")
    return QKVAttentionContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: qkv_attention_orchestration.RecoveryInvocation,
    context: QKVAttentionContext,
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


def test_qkv_attention_signature_and_delegate_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == [
        "include_layout_transpose",
        "include_prefix",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
    ]
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
    assert call.func.id == "run_qkv_attention"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "qkv_attention_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {
        "include_layout_transpose": "include_layout_transpose",
        "include_prefix": "include_prefix",
    }


@pytest.mark.parametrize(
    ("include_layout_transpose", "include_prefix", "expected_ids"),
    PRODUCTION_FORMS,
)
def test_qkv_attention_preserves_all_production_cleanup_forms(
    include_layout_transpose: bool,
    include_prefix: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    invocations = build_qkv_attention_invocations(
        context,
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
    )

    assert (
        active_qkv_attention_pass_ids(
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
        )
        == expected_ids
    )
    assert tuple(step.pass_id for step in invocations) == expected_ids
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
    } == {pass_id: expected_contract for pass_id in expected_ids}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_qkv_attention_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize(
    ("include_layout_transpose", "include_prefix", "expected_ids"),
    PRODUCTION_FORMS,
)
def test_qkv_attention_runner_preserves_all_production_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    include_prefix: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for pass_id in expected_ids:
        monkeypatch.setattr(
            qkv_attention_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_qkv_attention(
        context,
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
    )

    assert events == list(expected_ids)


def test_qkv_attention_preserves_both_invocation_forms() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == QKV_ATTENTION
    ]

    assert len(invocations) == 3
    default_invocations = [call for call in invocations if call.keywords == []]
    assert len(default_invocations) == 2
    assert all(call.args == [] for call in default_invocations)
    late_invocation = next(call for call in invocations if call.keywords)
    assert late_invocation.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in late_invocation.keywords
    } == {
        "include_layout_transpose": "optimize_layout_transpose_chains",
        "include_prefix": False,
    }


def test_qkv_attention_preserves_both_default_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    layout_block = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == QKV_ATTENTION
            for node in ast.walk(statement)
        )
    )
    nested_index = next(
        index
        for index, statement in enumerate(layout_block.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == QKV_ATTENTION
    )
    assert (
        _call_name(layout_block.body[nested_index - 1])
        == "_optimize_batchmatmul_transpose_input_to_adj_flags"
    )
    assert (
        _call_name(layout_block.body[nested_index + 1])
        == "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )

    top_level_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == QKV_ATTENTION
        and statement.value.keywords == []
    )
    assert (
        _call_name(lowerer.body[top_level_index - 1])
        == "_optimize_batchmatmul_transpose_input_to_adj_flags"
    )
    assert (
        _call_name(lowerer.body[top_level_index + 1])
        == "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains"
    )


def test_qkv_attention_preserves_late_bridge_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    late_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == QKV_ATTENTION
        and statement.value.keywords
    )

    assert (
        _call_name(lowerer.body[late_index - 1])
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )
    assert (
        _call_name(lowerer.body[late_index + 1])
        == "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )


def test_qkv_attention_context_is_explicit() -> None:
    lowerer, _ = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "qkv_attention_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_qkv_attention_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "qkv_attention_orchestration.py"
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
