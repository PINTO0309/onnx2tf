from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


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
    assert helper.end_lineno - helper.lineno + 1 == 41
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
        "model_ir",
        "session",
    }


def test_layout_attention_quantized_suffix_preserves_all_call_contracts() -> None:
    _, helper = _lowerer_and_helper()

    assert _ordered_call_contracts(helper) == [
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
        ("_run_attention_gate_qdq_recovery_sequence", (), {}),
        (
            "_run_duplicate_quantized_prelu_pass_cluster",
            (),
            {"include_transpose": "include_duplicate_transpose"},
        ),
        (
            "_optimize_dequant_transposeconv_quantize_chains",
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        (
            "run_quantized_reshape_cleanup",
            ("model_ir",),
            {
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            },
        ),
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
    ]


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
            assert isinstance(following, ast.Expr)
            assert isinstance(following.value, ast.Call)
            assert isinstance(following.value.func, ast.Name)
            boundaries.append((previous.value.func.id, following.value.func.id))

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
