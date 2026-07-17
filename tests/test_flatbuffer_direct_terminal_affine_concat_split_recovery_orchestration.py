from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TERMINAL_AFFINE_CONCAT_SPLIT = "_run_terminal_affine_concat_split_recovery_sequence"


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
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_AFFINE_CONCAT_SPLIT
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


def test_terminal_affine_concat_split_recovery_is_straight_line() -> None:
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
    assert helper.end_lineno - helper.lineno + 1 == 30
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
    assert loaded_data_names == {"model_ir", "session"}


def test_terminal_affine_concat_split_preserves_all_call_contracts() -> None:
    _, helper = _lowerer_and_helper()

    assert _ordered_call_contracts(helper) == [
        (
            "_optimize_fold_mul_add_mul_affine_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_concat_mul_add_transpose_nhwc_bridge_chains",
            ("model_ir",),
            {},
        ),
        (
            "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
            ("model_ir",),
            {},
        ),
        (
            "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
            ("model_ir",),
            {},
        ),
        (
            "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains",
            ("model_ir",),
            {},
        ),
        (
            "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_transpose_unary_split_concat_single_post_nchw",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_transpose_split_channelwise_tail_to_single_post_nchw",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        ("_sanitize_probable_nhwc_axis_sensitive_ops", ("model_ir",), {}),
    ]


def test_terminal_affine_concat_split_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
    ]

    assert len(invocations) == 2
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)


def test_terminal_affine_concat_split_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
    ]

    assert len(invocation_indexes) == 2
    observed: list[tuple[str, str]] = []
    for index in invocation_indexes:
        previous = lowerer.body[index - 1]
        following = lowerer.body[index + 1]
        for boundary in (previous, following):
            assert isinstance(boundary, ast.Expr)
            assert isinstance(boundary.value, ast.Call)
            assert isinstance(boundary.value.func, ast.Name)
        observed.append((previous.value.func.id, following.value.func.id))

    assert observed == [
        (
            "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
            "_optimize_transpose_pre_add_nhwc_chains",
        ),
        (
            "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
            "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
        ),
    ]
