from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PREADD_MEAN_ATTENTION = "_run_preadd_mean_attention_recovery_sequence"
ATTENTION_GATE_QDQ = "_run_attention_gate_qdq_recovery_sequence"


def _lowerer_and_helper(helper_name: str) -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
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


def _ordered_call_contracts(
    helper: ast.FunctionDef,
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    contracts: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    for statement in helper.body:
        assert isinstance(statement, ast.Expr)
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        contracts.append(
            (
                call.func.id,
                tuple(_expression_path(argument) for argument in call.args),
                {
                    str(keyword.arg): _expression_path(keyword.value)
                    for keyword in call.keywords
                },
            )
        )
    return contracts


def test_attention_recovery_sequences_are_straight_line_closures() -> None:
    expected_lines = {
        PREADD_MEAN_ATTENTION: 14,
        ATTENTION_GATE_QDQ: 27,
    }
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
    for helper_name, line_count in expected_lines.items():
        _, helper = _lowerer_and_helper(helper_name)

        assert helper.end_lineno is not None
        assert helper.end_lineno - helper.lineno + 1 == line_count
        assert helper.args.args == []
        assert helper.args.posonlyargs == []
        assert helper.args.kwonlyargs == []
        assert helper.args.vararg is None
        assert helper.args.kwarg is None
        assert not any(
            isinstance(node, control_flow_nodes) for node in ast.walk(helper)
        )
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
        }
        assert loaded_data_names == {"model_ir", "session"}


def test_preadd_mean_attention_preserves_exact_order_and_arguments() -> None:
    _, helper = _lowerer_and_helper(PREADD_MEAN_ATTENTION)

    assert _ordered_call_contracts(helper) == [
        (
            "_optimize_transpose_pre_add_nhwc_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
            ("model_ir",),
            {},
        ),
        (
            "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
            ("model_ir",),
            {},
        ),
        (
            "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
            ("model_ir",),
            {},
        ),
        (
            "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains",
            ("model_ir",),
            {},
        ),
        ("_run_mean_attention_layout_pass_cluster", (), {}),
    ]


def test_attention_gate_qdq_preserves_exact_order_and_arguments() -> None:
    _, helper = _lowerer_and_helper(ATTENTION_GATE_QDQ)

    assert _ordered_call_contracts(helper) == [
        (
            "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_sinet_mix_attention_double_logistic_nhwc_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        ("_run_gate_layout_pass_cluster", (), {}),
        (
            "_optimize_transposeconv_output_nhwc_passthrough_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_transposeconv_output_channel1_terminal_transpose_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        ("_run_transpose_unary_fanout_layout_pass_cluster", (), {}),
        (
            "_optimize_transpose_dequant_relu_quantize_bridges",
            ("model_ir",),
            {},
        ),
        (
            "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
            ("model_ir",),
            {},
        ),
        (
            "run_trailing_output_transpose_cleanup",
            ("model_ir",),
            {
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            },
        ),
        (
            "_optimize_transpose_dequant_mul_add_prelu_quantize_bridges",
            ("model_ir",),
            {},
        ),
    ]


def test_attention_recovery_invocation_boundaries_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper(PREADD_MEAN_ATTENTION)
    expected_counts = {
        PREADD_MEAN_ATTENTION: 2,
        ATTENTION_GATE_QDQ: 3,
    }

    for helper_name, expected_count in expected_counts.items():
        invocations = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
        ]
        assert len(invocations) == expected_count
        assert all(call.args == [] for call in invocations)
        assert all(call.keywords == [] for call in invocations)

    quantized_suffix = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_layout_attention_quantized_recovery_suffix"
    )
    suffix_calls = [
        statement.value.func.id
        for statement in quantized_suffix.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    attention_index = suffix_calls.index(ATTENTION_GATE_QDQ)
    assert suffix_calls[attention_index - 1 : attention_index + 2] == [
        "_run_mean_attention_layout_pass_cluster",
        ATTENTION_GATE_QDQ,
        "_run_duplicate_quantized_prelu_pass_cluster",
    ]
