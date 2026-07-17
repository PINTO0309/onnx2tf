from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import duplicate_quantized_prelu_orchestration
from onnx2tf.tflite_builder.passes.duplicate_quantized_prelu_orchestration import (
    DUPLICATE_QUANTIZED_PRELU_PASS_IDS,
    DuplicateQuantizedPReLUContext,
    build_duplicate_quantized_prelu_invocations,
    run_duplicate_quantized_prelu,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS,
    LayoutAttentionQuantizedSuffixContext,
    build_layout_attention_quantized_suffix_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
DUPLICATE_QUANTIZED_PRELU = "_run_duplicate_quantized_prelu_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == DUPLICATE_QUANTIZED_PRELU
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


def _context() -> DuplicateQuantizedPReLUContext:
    model_ir = ModelIR("duplicate_quantized_prelu_test")
    return DuplicateQuantizedPReLUContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: duplicate_quantized_prelu_orchestration.RecoveryInvocation,
    context: DuplicateQuantizedPReLUContext,
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


def test_duplicate_quantized_prelu_signature_and_delegate_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == ["include_transpose"]
    assert helper.args.kw_defaults == [None]
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
    assert call.func.id == "run_duplicate_quantized_prelu"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "duplicate_quantized_prelu_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {"include_transpose": "include_transpose"}


@pytest.mark.parametrize("include_transpose", [False, True])
def test_duplicate_quantized_prelu_preserves_both_cleanup_contracts(
    include_transpose: bool,
) -> None:
    context = _context()
    invocations = build_duplicate_quantized_prelu_invocations(
        context,
        include_transpose=include_transpose,
    )

    assert (
        tuple(step.pass_id for step in invocations)
        == DUPLICATE_QUANTIZED_PRELU_PASS_IDS
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        DUPLICATE_QUANTIZED_PRELU_PASS_IDS[0]: (
            ("model_ir",),
            {
                "include_transpose": include_transpose,
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
                "state_scope": "state_scope",
            },
        ),
        DUPLICATE_QUANTIZED_PRELU_PASS_IDS[1]: (
            ("model_ir",),
            {
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
                "state_scope": "state_scope",
            },
        ),
    }

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert scopes[0] is scopes[1]
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_duplicate_quantized_prelu_invocations(
            context,
            include_transpose=include_transpose,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize("include_transpose", [False, True])
def test_duplicate_quantized_prelu_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
    include_transpose: bool,
) -> None:
    context = _context()
    events: list[tuple[str, Any]] = []

    def duplicate_recorder(*args: Any, **kwargs: Any) -> None:
        events.append(
            (
                DUPLICATE_QUANTIZED_PRELU_PASS_IDS[0],
                kwargs["include_transpose"],
            )
        )

    def prelu_recorder(*args: Any, **kwargs: Any) -> None:
        events.append((DUPLICATE_QUANTIZED_PRELU_PASS_IDS[1], None))

    monkeypatch.setattr(
        duplicate_quantized_prelu_orchestration,
        "run_duplicate_fanout_cleanup",
        duplicate_recorder,
    )
    monkeypatch.setattr(
        duplicate_quantized_prelu_orchestration,
        "run_quantized_prelu_cleanup",
        prelu_recorder,
    )

    run_duplicate_quantized_prelu(
        context,
        include_transpose=include_transpose,
    )

    assert events == [
        (DUPLICATE_QUANTIZED_PRELU_PASS_IDS[0], include_transpose),
        (DUPLICATE_QUANTIZED_PRELU_PASS_IDS[1], None),
    ]


def test_duplicate_quantized_prelu_is_owned_only_by_suffix_callback() -> None:
    lowerer, _ = _lowerer_and_helper()
    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == DUPLICATE_QUANTIZED_PRELU
    ]
    assert direct_invocations == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "layout_attention_quantized_suffix_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    callback = next(
        keyword
        for keyword in context_assignment.value.keywords
        if keyword.arg == "duplicate_quantized_prelu_cluster"
    )
    assert _expression_path(callback.value) == DUPLICATE_QUANTIZED_PRELU


def test_duplicate_quantized_prelu_preserves_stable_callback_boundaries() -> None:
    callback_index = LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS.index(
        DUPLICATE_QUANTIZED_PRELU
    )

    assert (
        LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS.count(DUPLICATE_QUANTIZED_PRELU) == 1
    )
    assert LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[callback_index - 1] == (
        "_run_attention_gate_qdq_recovery_sequence"
    )
    assert LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[callback_index + 1] == (
        "_optimize_dequant_transposeconv_quantize_chains"
    )


def test_duplicate_quantized_prelu_suffix_callback_forwards_required_option() -> None:
    model_ir = ModelIR("duplicate_quantized_prelu_suffix_callback_test")

    def no_op(*args: Any, **kwargs: Any) -> None:
        return None

    context = LayoutAttentionQuantizedSuffixContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
        mean_attention_cluster=no_op,
        attention_gate_qdq_recovery=no_op,
        duplicate_quantized_prelu_cluster=no_op,
    )
    include_transpose = object()
    invocations = build_layout_attention_quantized_suffix_invocations(
        context,
        include_duplicate_transpose=include_transpose,  # type: ignore[arg-type]
    )
    callback = next(
        invocation
        for invocation in invocations
        if invocation.pass_id == DUPLICATE_QUANTIZED_PRELU
    )

    assert callback.callback is context.duplicate_quantized_prelu_cluster
    assert callback.args == ()
    assert callback.keyword_args == (("include_transpose", include_transpose),)


def test_duplicate_quantized_prelu_context_is_explicit() -> None:
    lowerer, _ = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "duplicate_quantized_prelu_context"
            for target in statement.targets
        )
    )

    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_duplicate_quantized_prelu_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "duplicate_quantized_prelu_orchestration.py"
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
