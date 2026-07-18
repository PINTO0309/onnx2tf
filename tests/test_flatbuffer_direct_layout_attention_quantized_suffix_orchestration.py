from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    layout_attention_quantized_suffix_orchestration,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS,
    LayoutAttentionQuantizedSuffixContext,
    build_layout_attention_quantized_suffix_invocations,
    run_layout_attention_quantized_suffix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SUFFIX = "_run_layout_attention_quantized_recovery_suffix"


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
        if isinstance(node, ast.FunctionDef) and node.name == SUFFIX
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


def _context() -> LayoutAttentionQuantizedSuffixContext:
    model_ir = ModelIR("layout_attention_quantized_suffix_test")

    def no_op(*args: Any, **kwargs: Any) -> None:
        return None

    return LayoutAttentionQuantizedSuffixContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        mean_attention_cluster=no_op,
        attention_gate_qdq_recovery=no_op,
        duplicate_quantized_prelu_cluster=no_op,
    )


def _normalize_new_contract(
    invocation: layout_attention_quantized_suffix_orchestration.RecoveryInvocation,
    context: LayoutAttentionQuantizedSuffixContext,
    include_duplicate_transpose: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.pass_context.model_ir:
            return "model_ir"
        if value is context.pass_context.layout_state:
            return "session.layout_state"
        if value is context.pass_context.diagnostics:
            return "session.diagnostics"
        if value is include_duplicate_transpose:
            return "include_duplicate_transpose"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_layout_attention_quantized_suffix_is_a_straight_line_closure() -> None:
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
    assert helper.end_lineno - helper.lineno + 1 == 8
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_duplicate_transpose"
    ]
    assert helper.args.kw_defaults == [None]
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
        and node.id != "bool"
    }
    assert loaded_data_names == {
        "include_duplicate_transpose",
        "layout_attention_quantized_suffix_context",
    }


def test_layout_attention_quantized_suffix_preserves_all_call_contracts() -> None:
    context = _context()
    include_duplicate_transpose = object()
    invocations = build_layout_attention_quantized_suffix_invocations(
        context,
        include_duplicate_transpose=include_duplicate_transpose,  # type: ignore[arg-type]
    )

    assert (
        tuple(step.pass_id for step in invocations)
        == LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS
    )
    contracts = {
        step.pass_id: _normalize_new_contract(
            step,
            context,
            include_duplicate_transpose,
        )
        for step in invocations
    }
    assert contracts == {
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_run_mean_attention_layout_pass_cluster": ((), {}),
        "_run_attention_gate_qdq_recovery_sequence": ((), {}),
        "_run_duplicate_quantized_prelu_pass_cluster": (
            (),
            {"include_transpose": "include_duplicate_transpose"},
        ),
        "_optimize_dequant_transposeconv_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "run_quantized_reshape_cleanup": (
            ("model_ir",),
            {
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            },
        ),
        "_optimize_dequant_hardsigmoid_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_maxpool_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_softmax_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_logistic_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_canonicalize_softmax_transpose_chains": (("model_ir",), {}),
    }
    assert invocations[3].callback is context.mean_attention_cluster
    assert invocations[4].callback is context.attention_gate_qdq_recovery
    assert invocations[5].callback is context.duplicate_quantized_prelu_cluster


def test_layout_attention_quantized_suffix_invocations_preserve_option() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SUFFIX
    ]

    assert len(invocations) == 2
    assert all(call.args == [] for call in invocations)
    for invocation in invocations:
        assert len(invocation.keywords) == 1
        keyword = invocation.keywords[0]
        assert keyword.arg == "include_duplicate_transpose"
        assert isinstance(keyword.value, ast.Name)
        assert keyword.value.id == "enable_duplicate_transpose_fanout_optimizations"


def test_layout_attention_quantized_suffix_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    boundaries: list[tuple[str, str]] = []
    following_targets: list[str | None] = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if not (
                isinstance(candidate, ast.Expr)
                and isinstance(candidate.value, ast.Call)
                and isinstance(candidate.value.func, ast.Name)
                and candidate.value.func.id == SUFFIX
            ):
                continue
            previous = statement.body[index - 1]
            following = statement.body[index + 1]
            assert isinstance(previous, ast.Expr)
            assert isinstance(previous.value, ast.Call)
            assert isinstance(previous.value.func, ast.Name)
            assert isinstance(following, (ast.Assign, ast.Expr))
            assert isinstance(following.value, ast.Call)
            assert isinstance(following.value.func, ast.Name)
            boundaries.append((previous.value.func.id, following.value.func.id))
            following_targets.append(
                following.targets[0].id
                if isinstance(following, ast.Assign)
                and len(following.targets) == 1
                and isinstance(following.targets[0], ast.Name)
                else None
            )

    assert boundaries == [
        (
            "_optimize_fold_mul_add_mul_affine_chains",
            "_run_safe_binary_bridge_recovery_sequence",
        ),
        (
            "run_squeeze_reshape_identity_cleanup",
            "_run_transpose_unary_fanout_layout_pass_cluster",
        ),
    ]
    assert following_targets == [
        "_layout_pass_set_1_safe_binary_results",
        None,
    ]


def test_layout_attention_quantized_suffix_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    statement = helper.body[0]
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_layout_attention_quantized_suffix"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "layout_attention_quantized_suffix_context"
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {"include_duplicate_transpose": "include_duplicate_transpose"}

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
    assert isinstance(context_assignment.value.func, ast.Name)
    assert context_assignment.value.func.id == "LayoutAttentionQuantizedSuffixContext"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in context_assignment.value.keywords
    } == {
        "pass_context": "session.model_ir_pass_context",
        "mean_attention_cluster": "_run_mean_attention_layout_pass_cluster",
        "attention_gate_qdq_recovery": ("_run_attention_gate_qdq_recovery_sequence"),
        "duplicate_quantized_prelu_cluster": (
            "_run_duplicate_quantized_prelu_pass_cluster"
        ),
    }


def test_layout_attention_quantized_suffix_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    model_ir = ModelIR("layout_attention_quantized_suffix_order_test")
    context = LayoutAttentionQuantizedSuffixContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        mean_attention_cluster=recorder("_run_mean_attention_layout_pass_cluster"),
        attention_gate_qdq_recovery=recorder(
            "_run_attention_gate_qdq_recovery_sequence"
        ),
        duplicate_quantized_prelu_cluster=recorder(
            "_run_duplicate_quantized_prelu_pass_cluster"
        ),
    )
    probe_steps = build_layout_attention_quantized_suffix_invocations(
        context,
        include_duplicate_transpose=True,
    )
    injected_callbacks = {
        context.mean_attention_cluster,
        context.attention_gate_qdq_recovery,
        context.duplicate_quantized_prelu_cluster,
    }
    for step in probe_steps:
        if step.callback in injected_callbacks:
            continue
        module_name = next(
            name
            for name, value in vars(
                layout_attention_quantized_suffix_orchestration
            ).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            layout_attention_quantized_suffix_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_layout_attention_quantized_suffix(
        context,
        include_duplicate_transpose=True,
    )

    assert events == list(LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS)


def test_layout_attention_quantized_suffix_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "layout_attention_quantized_suffix_orchestration.py"
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
