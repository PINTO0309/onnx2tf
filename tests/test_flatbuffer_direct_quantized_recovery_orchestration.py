from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SAFE_BINARY = "_run_safe_binary_bridge_recovery_sequence"
QUANTIZED_ACTIVATION_BINARY = (
    "_run_quantized_activation_binary_bridge_recovery_sequence"
)


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


def test_quantized_recovery_sequences_are_straight_line_closures() -> None:
    expected_lines = {
        SAFE_BINARY: 5,
        QUANTIZED_ACTIVATION_BINARY: 19,
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


def test_safe_binary_recovery_preserves_exact_arguments() -> None:
    _, helper = _lowerer_and_helper(SAFE_BINARY)

    assert _ordered_call_contracts(helper) == [
        (
            "_run_safe_binary_bridge_recovery_pass",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
    ]


def test_quantized_activation_binary_preserves_exact_order_and_arguments() -> None:
    _, helper = _lowerer_and_helper(QUANTIZED_ACTIVATION_BINARY)

    assert _ordered_call_contracts(helper) == [
        (
            "_optimize_dequant_hardsigmoid_quantize_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_dequant_maxpool_quantize_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_dequant_softmax_quantize_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_dequant_logistic_quantize_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        ("_canonicalize_softmax_transpose_chains", ("model_ir",), {}),
        (SAFE_BINARY, (), {}),
    ]


def test_quantized_recovery_invocation_boundaries_remain_zero_argument() -> None:
    lowerer, quantized_helper = _lowerer_and_helper(QUANTIZED_ACTIVATION_BINARY)
    expected_counts = {
        SAFE_BINARY: 3,
        QUANTIZED_ACTIVATION_BINARY: 2,
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

    nested_call = quantized_helper.body[-1]
    assert isinstance(nested_call, ast.Expr)
    assert isinstance(nested_call.value, ast.Call)
    assert isinstance(nested_call.value.func, ast.Name)
    assert nested_call.value.func.id == SAFE_BINARY
