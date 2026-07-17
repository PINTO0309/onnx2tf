from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SINET_PREADD_RESIZE = "_run_sinet_preadd_resize_recovery_sequence"
SINET_TERMINAL = "_run_sinet_terminal_layout_recovery_sequence"


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
        if isinstance(node, ast.FunctionDef) and node.name == SINET_PREADD_RESIZE
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


def test_sinet_preadd_resize_recovery_is_a_straight_line_closure() -> None:
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
    assert helper.end_lineno - helper.lineno + 1 == 20
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


def test_sinet_preadd_resize_recovery_preserves_all_call_contracts() -> None:
    _, helper = _lowerer_and_helper()

    assert _ordered_call_contracts(helper) == [
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
            "_optimize_sinet_concat_resize_affine_transpose_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_sinet_dual_resize_affine_transpose_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_sinet_concat_resize_affine_tail_concat_transpose_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "_optimize_sinet_softmax_mask_residual_nhwc_tail_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
    ]


def test_sinet_preadd_resize_recovery_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SINET_PREADD_RESIZE
    ]

    assert len(invocations) == 4
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)


def test_sinet_preadd_resize_recovery_preserves_all_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    terminal_helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == SINET_TERMINAL
    )
    terminal_calls = [
        statement.value
        for statement in terminal_helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    terminal_names = [call.func.id for call in terminal_calls]
    nested_index = terminal_names.index(SINET_PREADD_RESIZE)
    assert terminal_names[nested_index - 1 : nested_index + 2] == [
        "_optimize_sinet_shuffle_residual_transpose_chains",
        SINET_PREADD_RESIZE,
        "_optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains",
    ]

    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINET_PREADD_RESIZE
    ]
    assert len(invocation_indexes) == 3
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
            "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
            "_run_singleton_reshape_layout_pass_cluster",
        ),
        (
            SINET_TERMINAL,
            "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
        ),
        (
            "_reconcile_static_tensor_shapes",
            "_optimize_transpose_csp_attention_nhwc_chains",
        ),
    ]
