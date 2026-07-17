from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    AttentionRecoveryContext,
    build_attention_gate_qdq_invocations,
    build_preadd_mean_attention_invocations,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LayoutAttentionQuantizedSuffixContext,
    build_layout_attention_quantized_suffix_invocations,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LayoutRecoveryContext,
    build_layout_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
    build_terminal_slice_concat_recovery_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PASSES_ROOT = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
CALLBACK_CONTEXT_TYPES = (
    (
        "attention_recovery_orchestration",
        "AttentionRecoveryContext",
        (
            "mean_attention_cluster",
            "gate_layout_cluster",
            "transpose_unary_fanout_cluster",
        ),
    ),
    (
        "layout_attention_quantized_suffix_orchestration",
        "LayoutAttentionQuantizedSuffixContext",
        (
            "mean_attention_cluster",
            "attention_gate_qdq_recovery",
            "duplicate_quantized_prelu_cluster",
        ),
    ),
    (
        "layout_recovery_orchestration",
        "LayoutRecoveryContext",
        (
            "boundary_batchmatmul_unary_cluster",
            "pre_concat_cleanup",
            "channel_shuffle_gather_cluster",
        ),
    ),
    (
        "terminal_slice_concat_recovery_orchestration",
        "TerminalSliceConcatRecoveryContext",
        ("channel_slice_pad_mul_cluster",),
    ),
)
LOWERER_CALLBACK_CONTRACTS = {
    "AttentionRecoveryContext": {
        "mean_attention_cluster": "_run_mean_attention_layout_pass_cluster",
        "gate_layout_cluster": "_run_gate_layout_pass_cluster",
        "transpose_unary_fanout_cluster": (
            "_run_transpose_unary_fanout_layout_pass_cluster"
        ),
    },
    "LayoutAttentionQuantizedSuffixContext": {
        "mean_attention_cluster": "_run_mean_attention_layout_pass_cluster",
        "attention_gate_qdq_recovery": ("_run_attention_gate_qdq_recovery_sequence"),
        "duplicate_quantized_prelu_cluster": (
            "_run_duplicate_quantized_prelu_pass_cluster"
        ),
    },
    "LayoutRecoveryContext": {
        "boundary_batchmatmul_unary_cluster": (
            "_run_boundary_batchmatmul_unary_layout_pass_cluster"
        ),
        "pre_concat_cleanup": "_optimize_transpose_pre_concat_nhwc_chains",
        "channel_shuffle_gather_cluster": (
            "_run_channel_shuffle_gather_layout_pass_cluster"
        ),
    },
    "TerminalSliceConcatRecoveryContext": {
        "channel_slice_pad_mul_cluster": (
            "_run_channel_slice_pad_mul_layout_pass_cluster"
        ),
    },
}


def _expression_path(expression: ast.expr) -> object:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return f"{_expression_path(expression.value)}.{expression.attr}"
    if isinstance(expression, ast.Constant):
        return expression.value
    return type(expression).__name__


def _callback(name: str) -> Callable[..., Any]:
    def callback(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict]:
        return name, args, kwargs

    return callback


@pytest.mark.parametrize(
    ("module_name", "context_name", "callback_fields"),
    CALLBACK_CONTEXT_TYPES,
)
def test_callback_contexts_share_one_frozen_base_identity_contract(
    module_name: str,
    context_name: str,
    callback_fields: tuple[str, ...],
) -> None:
    module = import_module(f"onnx2tf.tflite_builder.passes.{module_name}")
    context_type = getattr(module, context_name)
    model_ir = ModelIR(f"callback_context_{module_name}")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []
    callbacks = {name: _callback(name) for name in callback_fields}

    assert is_dataclass(context_type)
    assert tuple(field.name for field in fields(context_type)) == (
        "model_ir",
        "layout_state",
        "diagnostics",
        *callback_fields,
    )
    context = context_type(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
        **callbacks,
    )
    assert context.model_ir is model_ir
    assert context.layout_state is layout_state
    assert context.diagnostics is diagnostics
    assert all(
        getattr(context, name) is callback for name, callback in callbacks.items()
    )
    with pytest.raises(FrozenInstanceError):
        context.model_ir = ModelIR("replacement")


def test_lowerer_callback_context_wiring_is_explicit() -> None:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    context_names = set(LOWERER_CALLBACK_CONTRACTS)
    calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in context_names
    ]

    assert len(calls) == 4
    assert {call.func.id for call in calls} == context_names
    for call in calls:
        assert isinstance(call.func, ast.Name)
        assert call.args == []
        contract = {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        }
        assert contract == {
            "model_ir": "model_ir",
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            **LOWERER_CALLBACK_CONTRACTS[call.func.id],
        }


def test_callback_invocation_argument_contracts_are_preserved() -> None:
    model_ir = ModelIR("callback_invocation_contracts")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []
    mean = _callback("mean")
    gate = _callback("gate")
    unary = _callback("unary")
    attention_gate = _callback("attention_gate")
    duplicate = _callback("duplicate")
    boundary = _callback("boundary")
    pre_concat = _callback("pre_concat")
    channel_shuffle = _callback("channel_shuffle")
    channel_slice = _callback("channel_slice")

    attention_context = AttentionRecoveryContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
        mean_attention_cluster=mean,
        gate_layout_cluster=gate,
        transpose_unary_fanout_cluster=unary,
    )
    preadd = build_preadd_mean_attention_invocations(attention_context)
    attention = build_attention_gate_qdq_invocations(attention_context)
    assert next(item for item in preadd if item.callback is mean).args == ()
    assert next(item for item in attention if item.callback is gate).args == ()
    assert next(item for item in attention if item.callback is unary).args == ()

    suffix = build_layout_attention_quantized_suffix_invocations(
        LayoutAttentionQuantizedSuffixContext(
            model_ir=model_ir,
            layout_state=layout_state,
            diagnostics=diagnostics,
            mean_attention_cluster=mean,
            attention_gate_qdq_recovery=attention_gate,
            duplicate_quantized_prelu_cluster=duplicate,
        ),
        include_duplicate_transpose=True,
    )
    assert next(item for item in suffix if item.callback is mean).args == ()
    assert next(item for item in suffix if item.callback is attention_gate).args == ()
    duplicate_invocation = next(item for item in suffix if item.callback is duplicate)
    assert duplicate_invocation.args == ()
    assert dict(duplicate_invocation.keyword_args) == {"include_transpose": True}

    layout = build_layout_recovery_invocations(
        LayoutRecoveryContext(
            model_ir=model_ir,
            layout_state=layout_state,
            diagnostics=diagnostics,
            boundary_batchmatmul_unary_cluster=boundary,
            pre_concat_cleanup=pre_concat,
            channel_shuffle_gather_cluster=channel_shuffle,
        )
    )
    assert next(item for item in layout if item.callback is boundary).args == ()
    pre_concat_invocation = next(item for item in layout if item.callback is pre_concat)
    assert pre_concat_invocation.args == (model_ir,)
    assert dict(pre_concat_invocation.keyword_args) == {
        "layout_state": layout_state,
        "diagnostics": diagnostics,
    }
    assert next(item for item in layout if item.callback is channel_shuffle).args == ()

    terminal = build_terminal_slice_concat_recovery_invocations(
        TerminalSliceConcatRecoveryContext(
            model_ir=model_ir,
            layout_state=layout_state,
            diagnostics=diagnostics,
            channel_slice_pad_mul_cluster=channel_slice,
        )
    )
    assert next(item for item in terminal if item.callback is channel_slice).args == ()


def test_sinet_terminal_partial_context_remains_outside_the_shared_boundary() -> None:
    module = import_module(
        "onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration"
    )
    context_type = module.SINetTerminalLayoutRecoveryContext

    assert tuple(field.name for field in fields(context_type)) == (
        "model_ir",
        "layout_state",
        "preadd_resize_recovery",
    )
    assert "diagnostics" not in {field.name for field in fields(context_type)}


def test_callback_context_modules_do_not_import_the_lowerer() -> None:
    module_names = [module_name for module_name, _, _ in CALLBACK_CONTEXT_TYPES]
    module_names.append("sinet_terminal_layout_recovery_orchestration")
    for module_name in module_names:
        tree = ast.parse(
            (PASSES_ROOT / f"{module_name}.py").read_text(encoding="utf-8")
        )
        imported_modules = {
            str(node.module)
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
