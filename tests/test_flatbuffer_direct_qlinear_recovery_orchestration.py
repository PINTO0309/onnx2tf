from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
QLINEAR_MEAN_CONCAT = "_run_qlinear_mean_concat_recovery_sequence"


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
        if isinstance(node, ast.FunctionDef) and node.name == QLINEAR_MEAN_CONCAT
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


def test_qlinear_recovery_sequence_is_a_straight_line_closure() -> None:
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
    assert helper.end_lineno - helper.lineno + 1 == 6
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
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
    }
    assert loaded_data_names == {"model_ir"}


def test_qlinear_recovery_preserves_exact_order_and_arguments() -> None:
    _, helper = _lowerer_and_helper()

    assert _ordered_call_contracts(helper) == [
        (
            "_optimize_transpose_mean_hardsigmoid_muladd_chains",
            ("model_ir",),
            {},
        ),
        ("_optimize_nhwc_prefix_qlinear_silu_chains", ("model_ir",), {}),
        (
            "_optimize_nhwc_propagation_qlinear_concat_conv",
            ("model_ir",),
            {},
        ),
        ("_optimize_concat_pre_quantize_dequantize", ("model_ir",), {}),
        (
            "_optimize_transpose_mean_maxpool_concat_conv_chains",
            ("model_ir",),
            {},
        ),
    ]


def test_qlinear_recovery_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == QLINEAR_MEAN_CONCAT
    ]

    assert len(invocations) == 2
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)


def test_qlinear_recovery_preserves_both_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    boundaries: list[tuple[str, str]] = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if not (
                isinstance(candidate, ast.Expr)
                and isinstance(candidate.value, ast.Call)
                and isinstance(candidate.value.func, ast.Name)
                and candidate.value.func.id == QLINEAR_MEAN_CONCAT
            ):
                continue
            previous = statement.body[index - 1]
            following = statement.body[index + 1]
            assert isinstance(previous, ast.Expr)
            assert isinstance(previous.value, ast.Call)
            assert isinstance(previous.value.func, ast.Name)
            assert isinstance(following, ast.Expr)
            assert isinstance(following.value, ast.Call)
            assert isinstance(following.value.func, ast.Name)
            boundaries.append((previous.value.func.id, following.value.func.id))

    assert boundaries == [
        (
            "_optimize_transpose_dequantize_mean_quantize_bridges",
            "_run_layout_reshape_attention_recovery_prefix",
        ),
        (
            "_set_post_progress_desc",
            "_run_layout_recovery_prefix_pass_sequence",
        ),
    ]
