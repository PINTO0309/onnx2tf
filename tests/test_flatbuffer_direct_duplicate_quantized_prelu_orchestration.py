from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
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


def test_duplicate_quantized_prelu_signature_and_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 21
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == ["include_transpose"]
    assert helper.args.kw_defaults == [None]
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
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

    scope_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
    ]
    assert len(scope_calls) == 1
    assert tuple(_expression_path(arg) for arg in scope_calls[0].args) == ("model_ir",)
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in scope_calls[0].keywords
    } == {"layout_state": "session.layout_state"}


def test_duplicate_quantized_prelu_preserves_both_cleanup_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_names = (
        "run_duplicate_fanout_cleanup",
        "run_quantized_prelu_cleanup",
    )
    calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in cleanup_names
    ]
    calls.sort(key=lambda call: call.lineno)

    assert tuple(call.func.id for call in calls) == cleanup_names
    assert tuple(_expression_path(arg) for arg in calls[0].args) == ("model_ir",)
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in calls[0].keywords
    } == {
        "include_transpose": "include_transpose",
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    assert tuple(_expression_path(arg) for arg in calls[1].args) == ("model_ir",)
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in calls[1].keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }


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
