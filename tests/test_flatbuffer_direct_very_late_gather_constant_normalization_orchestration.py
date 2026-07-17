from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import constant_fold_cast_orchestration
from onnx2tf.tflite_builder.passes import (
    very_late_gather_constant_normalization_orchestration,
)
from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.very_late_gather_constant_normalization_orchestration import (
    VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS,
    VeryLateGatherConstantNormalizationContext,
    build_very_late_gather_constant_normalization_invocations,
    run_very_late_gather_constant_normalization,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
VERY_LATE = "_run_very_late_gather_constant_normalization_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == VERY_LATE
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected expression: {ast.dump(node)}")


def _context() -> VeryLateGatherConstantNormalizationContext:
    model_ir = ModelIR("very_late_gather_constant_normalization_test")
    return VeryLateGatherConstantNormalizationContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: very_late_gather_constant_normalization_orchestration.RecoveryInvocation,
    context: VeryLateGatherConstantNormalizationContext,
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


def test_very_late_is_a_straight_line_delegate() -> None:
    _, helper = _lowerer_and_helper()

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

    statement = helper.body[0]
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_very_late_gather_constant_normalization"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "very_late_gather_constant_normalization_context",
    )
    assert call.keywords == []


def test_very_late_preserves_all_four_effective_owner_contracts() -> None:
    context = _context()
    invocations = build_very_late_gather_constant_normalization_invocations(context)

    assert VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS == (
        "run_transpose_gather_axis_cleanup",
        *CONSTANT_FOLD_CAST_PASS_IDS,
        "run_normalization_pad_layout_cleanup",
    )
    assert (
        tuple(step.pass_id for step in invocations)
        == VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS
    )
    shared_contract = (
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[0]: shared_contract,
        CONSTANT_FOLD_CAST_PASS_IDS[0]: shared_contract,
        CONSTANT_FOLD_CAST_PASS_IDS[1]: shared_contract,
        VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[-1]: (
            ("model_ir",),
            {
                "include_instance": False,
                "include_flatten": True,
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
                "state_scope": "state_scope",
            },
        ),
    }


def test_very_late_composes_one_fresh_scope_across_both_builders() -> None:
    context = _context()
    invocations = build_very_late_gather_constant_normalization_invocations(context)
    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]

    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_very_late_gather_constant_normalization_invocations(context)[
            0
        ].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_very_late_runner_preserves_instrumented_effective_order(
    monkeypatch,
) -> None:
    context = _context()
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    monkeypatch.setattr(
        very_late_gather_constant_normalization_orchestration,
        "run_transpose_gather_axis_cleanup",
        recorder(VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[0]),
    )
    for pass_id in CONSTANT_FOLD_CAST_PASS_IDS:
        monkeypatch.setattr(
            constant_fold_cast_orchestration,
            pass_id,
            recorder(pass_id),
        )
    monkeypatch.setattr(
        very_late_gather_constant_normalization_orchestration,
        "run_normalization_pad_layout_cleanup",
        recorder(VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[-1]),
    )

    run_very_late_gather_constant_normalization(context)

    assert [pass_id for pass_id, _ in events] == list(
        VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS
    )
    assert all(scope is events[0][1] for _, scope in events)


def test_very_late_preserves_sole_terminal_invocation_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == VERY_LATE
    ]

    assert len(invocation_indexes) == 1
    invocation_index = invocation_indexes[0]
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert invocation.value.keywords == []

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    for boundary in (previous, following):
        assert isinstance(boundary, ast.Expr)
        assert isinstance(boundary.value, ast.Call)
        assert isinstance(boundary.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )
    assert following.value.func.id == "_resolve_dynamic_reshape_shapes"


@pytest.mark.xfail(
    strict=True,
    reason="the very-late affine post-ADD result is discarded",
)
def test_very_late_affine_post_add_captures_complete_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == VERY_LATE
    )
    invocation = lowerer.body[invocation_index - 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == "_very_late_affine_post_add_stats"
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

    previous = lowerer.body[invocation_index - 2]
    assert isinstance(previous, ast.Expr)
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
    )
    following = lowerer.body[invocation_index]
    assert isinstance(following, ast.Expr)
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == VERY_LATE

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
    assert direct_statements[1] is invocation
    assert isinstance(direct_statements[2], ast.Assign)
    third_target = direct_statements[2].targets[0]
    assert isinstance(third_target, ast.Name)
    assert third_target.id == "_absolute_final_affine_post_add_stats"


def test_very_late_context_is_explicit() -> None:
    lowerer, _ = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "very_late_gather_constant_normalization_context"
            for target in statement.targets
        )
    )

    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_very_late_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "very_late_gather_constant_normalization_orchestration.py"
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
