from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    AttentionRecoveryContext,
    run_attention_gate_qdq_recovery,
)
from onnx2tf.tflite_builder.passes.duplicate_quantized_prelu_orchestration import (
    run_duplicate_quantized_prelu,
)
from onnx2tf.tflite_builder.passes.gate_layout_orchestration import (
    run_gate_layout,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LayoutAttentionQuantizedSuffixContext,
    run_layout_attention_quantized_suffix,
)
from onnx2tf.tflite_builder.passes.mean_attention_orchestration import (
    run_mean_attention,
)
from onnx2tf.tflite_builder.passes.quantized_recovery_orchestration import (
    run_safe_binary_recovery,
)
from onnx2tf.tflite_builder.passes.transpose_unary_fanout_orchestration import (
    run_transpose_unary_fanout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_pass_set_1_attention_quantized_safe_binary_orchestration.py"
)
OWNER = "run_layout_pass_set_1_attention_quantized_safe_binary_cleanup"
CHILD_OWNERS = (
    "run_layout_attention_quantized_suffix",
    "run_safe_binary_recovery",
)
CURRENT_CHILD_OWNERS = (
    "_run_layout_attention_quantized_recovery_suffix",
    "_run_safe_binary_bridge_recovery_sequence",
)
RESULT_TARGETS = (
    "_layout_pass_set_1_attention_quantized_suffix_results",
    "_layout_pass_set_1_safe_binary_results",
)
COMPOSITE_TARGET = (
    "_layout_pass_set_1_attention_quantized_safe_binary_results"
)
LATER_RESULT_TARGETS = (
    "_layout_pass_set_1_final_attention_quantized_suffix_results",
    "_layout_pass_set_1_transpose_unary_fanout_results",
    "_layout_pass_set_1_final_safe_binary_results",
)
PREDECESSOR_PHASE_ID = (
    "cleanup.layout_pass_set_1.post_binary_affine_chain_fold"
)
SUCCESSOR_PHASE_ID = "cleanup.layout_pass_set_1.dequant_mean_quantize"
OPTION = "enable_duplicate_transpose_fanout_optimizations"
GUARD = "optimize_layout_transpose_chains"

MEAN_SCHEMA = (
    ("optimized_transpose_mean_prepost_nhwc_passthrough_chains",),
    ("optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains",),
    ("optimized_transpose_pre_unary_mean_terminal_nhwc_chains",),
    ("optimized_transpose_se_conv_mul_prepost_nhwc_chains",),
    ("optimized_transpose_se_fc_mul_prepost_nhwc_chains",),
    ("optimized_transpose_conv_attention_nhwc_propagation_chains",),
)
ATTENTION_SCHEMA = (
    ("optimized_transpose_sa_pa_mirrorpad_nhwc_propagation_chains",),
    ("optimized_sinet_mix_attention_double_logistic_nhwc_chains",),
    (
        ("optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains",),
        (
            "optimized_transpose_sum_logistic_muladd_prepost_nhwc_chains",
            "optimized_transpose_weighted_add_swish_prepost_nhwc_chains",
            "optimized_transpose_nested_weighted_add_swish_prepost_nhwc_chains",
            "optimized_transpose_logistic_muladd_prepost_nhwc_chains",
        ),
        (
            "optimized_transpose_pad_prepost_nhwc_chains",
            "optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains",
            "optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains",
        ),
        (
            "optimized_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains",
            "optimized_transpose_logistic_sub_mul_postadd_nhwc_chains",
        ),
        (
            "optimized_transpose_3d_leaky_logistic_muladd_ndhwc_chains",
            "optimized_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains",
        ),
        ("optimized_transpose_cost_volume_scatter_ndhwc_chains",),
        ("optimized_transpose_add_concat_const_suffix_nhwc_chains",),
        ("optimized_transpose_dual_mul_concat_prepost_nhwc_chains",),
    ),
    ("rewritten_transposeconv_output_nhwc_passthrough_chains",),
    ("rewritten_transposeconv_output_channel1_terminal_transpose_chains",),
    (
        ("rewritten_transpose_unary_passthrough_chains",),
        ("rewritten_transpose_unary_fanout_inverse_post_bridges",),
        ("rewritten_transpose_unary_binary_full_post_fanout_bridges",),
    ),
    ("removed_transpose_dequant_relu_quantize_bridges",),
    ("removed_transpose_dequant_hardsigmoid_quantize_bridges",),
    ("rewritten_trailing_output_transpose_passthrough_chains",),
    ("removed_transpose_dequant_mul_add_prelu_quantize_bridges",),
)
SAFE_BINARY_SCHEMA = (
    (
        "rewritten_transpose_binary_symmetric_legacy_only_bridges_safe",
        "rewritten_transpose_binary_single_post_bridges_safe",
        "rewritten_transpose_binary_mixed_fanout_bridges_safe",
        "rewritten_transpose_binary_asymmetric_fanout_bridges",
        "rewritten_transpose_binary_full_post_fanout_bridges",
    ),
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _phase_id(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or ast.unparse(call.func) != "session.record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _guard_body() -> list[ast.stmt]:
    lowerer = _lowerer()
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == GUARD
        and any(
            _single_target(candidate) in RESULT_TARGETS
            for candidate in statement.body
        )
    )
    assert guard.orelse == []
    return guard.body


def _schema(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(value)
    if isinstance(value, tuple):
        return tuple(_schema(item) for item in value)
    raise AssertionError(f"unexpected result type: {type(value)!r}")


def _expected_suffix_schema(*, include_duplicate_transpose: bool) -> tuple[Any, ...]:
    duplicate_keys = ("removed_duplicate_reshape_fanout",)
    if include_duplicate_transpose:
        duplicate_keys = (
            *duplicate_keys,
            "removed_duplicate_transpose_fanout",
        )
    return (
        ("optimized_transpose_mul_add_const_prepost_nhwc_chains",),
        (
            "optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
        ),
        ("optimized_transpose_mean_mul_add_const_prepost_nhwc_chains",),
        MEAN_SCHEMA,
        ATTENTION_SCHEMA,
        (
            duplicate_keys,
            (
                "removed_transpose_dequant_prelu_quantize_bridges",
                "removed_transpose_dequant_prelu_transpose_bridges",
                "folded_dequant_prelu_quantize_chains",
                "folded_dequant_prelu_depthwise_quantize_chains",
            ),
        ),
        ("folded_dequant_transposeconv_quantize_chains",),
        ("folded_dequant_reshape_quantize_chains",),
        ("folded_dequant_hardsigmoid_quantize_chains",),
        ("folded_dequant_maxpool_quantize_chains",),
        ("folded_dequant_softmax_quantize_chains",),
        ("folded_dequant_logistic_quantize_chains",),
        ("canonicalized_softmax_transpose_chains",),
    )


def _context() -> LayoutAttentionQuantizedSuffixContext:
    model_ir = ModelIR("layout_pass_set_1_attention_quantized_safe_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    attention_context = AttentionRecoveryContext(
        pass_context=pass_context,
        mean_attention_cluster=lambda: run_mean_attention(pass_context),
        gate_layout_cluster=lambda: run_gate_layout(pass_context),
        transpose_unary_fanout_cluster=lambda: run_transpose_unary_fanout(
            pass_context
        ),
    )
    return LayoutAttentionQuantizedSuffixContext(
        pass_context=pass_context,
        mean_attention_cluster=lambda: run_mean_attention(pass_context),
        attention_gate_qdq_recovery=lambda: run_attention_gate_qdq_recovery(
            attention_context
        ),
        duplicate_quantized_prelu_cluster=(
            lambda *, include_transpose: run_duplicate_quantized_prelu(
                pass_context,
                include_transpose=include_transpose,
            )
        ),
    )


@pytest.mark.parametrize("include_duplicate_transpose", [False, True])
def test_layout_pass_set_1_attention_quantized_safe_current_contract(
    include_duplicate_transpose: bool,
) -> None:
    body = _guard_body()
    assignments = [
        statement
        for statement in body
        if _single_target(statement) in RESULT_TARGETS
    ]
    assert [_single_target(statement) for statement in assignments] == list(
        RESULT_TARGETS
    )
    assert [_call_name(statement) for statement in assignments] == list(
        CURRENT_CHILD_OWNERS
    )
    indices = [body.index(statement) for statement in assignments]
    assert indices[1] == indices[0] + 1
    assert _phase_id(body[indices[0] - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(body[indices[-1] + 1]) == SUCCESSOR_PHASE_ID

    suffix_call = _call(assignments[0])
    safe_call = _call(assignments[1])
    assert suffix_call is not None
    assert safe_call is not None
    assert suffix_call.args == []
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in suffix_call.keywords
    } == {"include_duplicate_transpose": OPTION}
    assert safe_call.args == []
    assert safe_call.keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for statement in body
        for node in ast.walk(statement)
    )

    lowerer = _lowerer()
    assignments_by_target = {
        _single_target(statement): statement
        for statement in lowerer.body
        if _single_target(statement)
        in {
            "shared_model_ir_pass_context",
            "quantized_recovery_context",
            "layout_attention_quantized_suffix_context",
        }
    }
    assert ast.unparse(
        assignments_by_target["shared_model_ir_pass_context"].value
    ) == "session.model_ir_pass_context"
    assert ast.unparse(
        assignments_by_target["quantized_recovery_context"].value
    ) == "shared_model_ir_pass_context"
    suffix_context_assignment = assignments_by_target[
        "layout_attention_quantized_suffix_context"
    ]
    assert isinstance(suffix_context_assignment.value, ast.Call)
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in suffix_context_assignment.value.keywords
    } == {
        "pass_context": "session.model_ir_pass_context",
        "mean_attention_cluster": "_run_mean_attention_layout_pass_cluster",
        "attention_gate_qdq_recovery": (
            "_run_attention_gate_qdq_recovery_sequence"
        ),
        "duplicate_quantized_prelu_cluster": (
            "_run_duplicate_quantized_prelu_pass_cluster"
        ),
    }

    context = _context()
    results = (
        run_layout_attention_quantized_suffix(
            context,
            include_duplicate_transpose=include_duplicate_transpose,
        ),
        run_safe_binary_recovery(context.pass_context),
    )
    assert _schema(results) == (
        _expected_suffix_schema(
            include_duplicate_transpose=include_duplicate_transpose
        ),
        SAFE_BINARY_SCHEMA,
    )

    later_targets = [
        _single_target(statement)
        for statement in body
        if _single_target(statement) in LATER_RESULT_TARGETS
    ]
    assert later_targets == list(LATER_RESULT_TARGETS)
    later_indices = [
        next(
            index
            for index, statement in enumerate(body)
            if _single_target(statement) == target
        )
        for target in LATER_RESULT_TARGETS
    ]
    assert later_indices == list(
        range(later_indices[0], later_indices[0] + len(LATER_RESULT_TARGETS))
    )


@pytest.mark.xfail(
    strict=True,
    reason="layout-pass-set-1 quantized suffix/safe owner is not implemented",
)
def test_layout_pass_set_1_attention_quantized_safe_has_one_context_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in CHILD_OWNERS
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [call.func.id for call in calls] == list(CHILD_OWNERS)
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
    } == {"include_duplicate_transpose": "include_duplicate_transpose"}
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.pass_context"
    ]
    assert calls[1].keywords == []

    body = _guard_body()
    assignment = next(
        statement
        for statement in body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "layout_attention_quantized_suffix_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_duplicate_transpose": OPTION}
    assert _phase_id(body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(_lowerer())
    )

    lowerer_functions = {
        node.name: node
        for node in _lowerer().body
        if isinstance(node, ast.FunctionDef)
    }
    assert all(name in lowerer_functions for name in CURRENT_CHILD_OWNERS)
