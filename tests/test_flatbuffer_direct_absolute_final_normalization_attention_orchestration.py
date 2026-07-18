from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    absolute_final_normalization_attention_orchestration,
)
from onnx2tf.tflite_builder.passes.absolute_final_normalization_attention_orchestration import (
    ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS,
    AbsoluteFinalNormalizationAttentionContext,
    build_absolute_final_normalization_attention_invocations,
    run_absolute_final_normalization_attention,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ABSOLUTE_FINAL_NORMALIZATION_ATTENTION = (
    "_run_absolute_final_normalization_attention_pass_pair"
)


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
        and node.name == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context() -> AbsoluteFinalNormalizationAttentionContext:
    model_ir = ModelIR("absolute_final_normalization_attention_test")
    return AbsoluteFinalNormalizationAttentionContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: absolute_final_normalization_attention_orchestration.RecoveryInvocation,
    context: AbsoluteFinalNormalizationAttentionContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        if value is context.diagnostics:
            return "session.diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_absolute_final_normalization_attention_is_a_straight_line_delegate() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 4
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert len(helper.body) == 1
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
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )


def test_absolute_final_normalization_attention_preserves_cleanup_contracts() -> None:
    context = _context()
    invocations = build_absolute_final_normalization_attention_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS
    )
    expected_contracts = {
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS[0]: (
            ("model_ir",),
            {
                "include_instance": False,
                "include_flatten": True,
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
                "state_scope": "state_scope",
            },
        ),
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS[1]: (
            ("model_ir",),
            {
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
                "state_scope": "state_scope",
            },
        ),
    }
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == expected_contracts

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_absolute_final_normalization_attention_invocations(context)[
            0
        ].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_absolute_final_normalization_attention_invocation_is_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert invocations[0].keywords == []


def test_absolute_final_normalization_attention_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_absolute_final_normalization_attention_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    )

    previous = lowerer.body[invocation_index - 1]
    invocation = lowerer.body[invocation_index]
    following = lowerer.body[invocation_index + 1]
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_absolute_final_instancenorm_post_bias_stats"
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_absolute_final_dynamic_rank1_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert (
        following.value.func.id
        == "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    )


def test_absolute_final_post_bias_captures_complete_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    normalization_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_absolute_final_normalization_attention_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    )
    invocation = lowerer.body[normalization_index - 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        "_absolute_final_instancenorm_post_bias_stats"
    )
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == (
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"

    previous = lowerer.body[normalization_index - 2]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_absolute_final_affine_post_add_stats"
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )
    following = lowerer.body[normalization_index]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_absolute_final_normalization_attention_results"
    )
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION

    direct_statements = [
        statement
        for statement in lowerer.body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
            for node in ast.walk(statement)
        )
    ]
    assert len(direct_statements) == 4
    assert isinstance(direct_statements[0], ast.Assign)
    assert isinstance(direct_statements[0].targets[0], ast.Name)
    assert direct_statements[0].targets[0].id == (
        "_terminal_instancenorm_post_bias_stats"
    )
    assert isinstance(direct_statements[1], ast.Expr)
    assert isinstance(direct_statements[2], ast.Assign)
    third_target = direct_statements[2].targets[0]
    assert isinstance(third_target, ast.Name)
    assert third_target.id == "_pre_terminal_affine_instancenorm_post_bias_stats"
    assert direct_statements[3] is invocation


def test_absolute_final_affine_post_add_captures_complete_mutation_evidence() -> (
    None
):
    lowerer, _ = _lowerer_and_helper()
    post_bias_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_absolute_final_instancenorm_post_bias_stats"
    )
    invocation = lowerer.body[post_bias_index - 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == "_absolute_final_affine_post_add_stats"
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == (
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"

    previous = lowerer.body[post_bias_index - 2]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_absolute_final_static_signature_stats"
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == "_sanitize_static_shape_signature_consistency"
    realign = lowerer.body[post_bias_index - 3]
    assert isinstance(realign, ast.Assign)
    assert len(realign.targets) == 1
    assert isinstance(realign.targets[0], ast.Name)
    assert realign.targets[0].id == "_absolute_final_boundary_signature_stats"
    assert isinstance(realign.value, ast.Call)
    assert isinstance(realign.value.func, ast.Name)
    assert realign.value.func.id == (
        "_realign_dynamic_boundary_shape_signature_map"
    )
    following = lowerer.body[post_bias_index]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_absolute_final_instancenorm_post_bias_stats"
    )

    direct_statements = [
        statement
        for statement in lowerer.body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
            for node in ast.walk(statement)
        )
    ]
    assert len(direct_statements) == 3
    assert isinstance(direct_statements[0], ast.Assign)
    first_target = direct_statements[0].targets[0]
    assert isinstance(first_target, ast.Name)
    assert first_target.id == "_pre_terminal_affine_post_add_stats"
    assert isinstance(direct_statements[1], ast.Assign)
    second_target = direct_statements[1].targets[0]
    assert isinstance(second_target, ast.Name)
    assert second_target.id == "_very_late_affine_post_add_stats"
    assert direct_statements[2] is invocation


def test_absolute_final_normalization_attention_context_and_wrapper_are_explicit() -> (
    None
):
    lowerer, helper = _lowerer_and_helper()
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_absolute_final_normalization_attention"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "absolute_final_normalization_attention_context",
    )
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "absolute_final_normalization_attention_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_absolute_final_normalization_attention_runner_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for pass_id in ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS:
        monkeypatch.setattr(
            absolute_final_normalization_attention_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_absolute_final_normalization_attention(context)

    assert events == list(ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS)


def test_absolute_final_normalization_attention_runner_returns_ordered_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    expected_by_pass_id = {
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS[0]: {
            "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 1,
        },
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS[1]: {
            "optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains": 2,
        },
    }

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> dict[str, int]:
            return dict(expected_by_pass_id[pass_id])

        return record

    for pass_id in ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS:
        monkeypatch.setattr(
            absolute_final_normalization_attention_orchestration,
            pass_id,
            recorder(pass_id),
        )

    assert run_absolute_final_normalization_attention(context) == tuple(
        expected_by_pass_id[pass_id]
        for pass_id in ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS
    )


def test_absolute_final_normalization_attention_lowerer_captures_results() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    helper_statement = helper.body[0]
    assert isinstance(helper_statement, ast.Return)
    assert isinstance(helper_statement.value, ast.Call)
    assert isinstance(helper_statement.value.func, ast.Name)
    assert helper_statement.value.func.id == "run_absolute_final_normalization_attention"

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_absolute_final_normalization_attention_results"
    )
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    assert invocation.value.args == []
    assert invocation.value.keywords == []

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    assert isinstance(previous, ast.Assign)
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_absolute_final_instancenorm_post_bias_stats"
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_absolute_final_dynamic_rank1_stats"


def test_absolute_final_normalization_attention_module_does_not_import_lowerer() -> (
    None
):
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "absolute_final_normalization_attention_orchestration.py"
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
