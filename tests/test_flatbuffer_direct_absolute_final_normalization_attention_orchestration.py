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
from onnx2tf.tflite_builder.passes.pre_terminal_affine_tail_orchestration import (
    PRE_TERMINAL_AFFINE_TAIL_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ABSOLUTE_FINAL_NORMALIZATION_ATTENTION = (
    "run_absolute_final_normalization_attention_rank1_cleanup"
)
ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "absolute_final_normalization_attention_orchestration.py"
)
VERY_LATE_PAD_INSTANCENORM = (
    "run_very_late_pad_instancenorm_layout_cleanup"
)
PRE_TERMINAL_INSTANCENORM_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_instancenorm_layout_orchestration.py"
)
PRE_TERMINAL_INSTANCENORM = (
    "run_pre_terminal_instancenorm_layout_cleanup"
)
ABSOLUTE_FINAL_AFFINE_INSTANCENORM_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "absolute_final_affine_instancenorm_orchestration.py"
)
ABSOLUTE_FINAL_AFFINE_INSTANCENORM = (
    "run_absolute_final_affine_instancenorm_cleanup"
)
ABSOLUTE_FINAL_CLEANUP_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "absolute_final_cleanup_orchestration.py"
)
ABSOLUTE_FINAL_CLEANUP = "run_absolute_final_cleanup"
ABSOLUTE_FINAL_CLEANUP_TARGET = "_absolute_final_cleanup_results"


def _lowerer_and_helper() -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    owner_tree = ast.parse(
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PATH.read_text(encoding="utf-8")
    )
    helper = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    )
    return lowerer, helper


def _absolute_final_cleanup_owner() -> ast.FunctionDef:
    tree = ast.parse(
        ABSOLUTE_FINAL_CLEANUP_PATH.read_text(encoding="utf-8")
    )
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == ABSOLUTE_FINAL_CLEANUP
    )


def _absolute_final_cleanup_calls(function_name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(_absolute_final_cleanup_owner())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _very_late_pad_instancenorm_call_count(lowerer: ast.FunctionDef) -> int:
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == VERY_LATE_PAD_INSTANCENORM
        for node in ast.walk(lowerer)
    )


def _pre_terminal_instancenorm_call_count(function_name: str) -> int:
    tree = ast.parse(
        PRE_TERMINAL_INSTANCENORM_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == PRE_TERMINAL_INSTANCENORM
    )
    owner_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
        for node in ast.walk(owner)
    )


def _absolute_final_affine_instancenorm_calls(
    function_name: str,
) -> list[ast.Call]:
    tree = ast.parse(
        ABSOLUTE_FINAL_AFFINE_INSTANCENORM_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == ABSOLUTE_FINAL_AFFINE_INSTANCENORM
    )
    owner_name = function_name.removeprefix("_")
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
    ]


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _phase_result_owner(statement: ast.stmt, phase_id: str) -> ast.Call:
    assert isinstance(statement, ast.Expr)
    record = statement.value
    assert isinstance(record, ast.Call)
    assert isinstance(record.func, ast.Attribute)
    assert isinstance(record.func.value, ast.Name)
    assert record.func.value.id == "session"
    assert record.func.attr == "record_phase_result"
    assert len(record.args) == 2
    assert ast.literal_eval(record.args[0]) == phase_id
    owner = record.args[1]
    assert isinstance(owner, ast.Call)
    return owner


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
    assert helper.end_lineno - helper.lineno + 1 == 10
    assert [argument.arg for argument in helper.args.args] == ["context"]
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


def test_absolute_final_normalization_attention_invocation_uses_shared_context() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = _absolute_final_cleanup_calls(
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    )

    assert len(invocations) == 1
    assert [ast.unparse(argument) for argument in invocations[0].args] == [
        "context"
    ]
    assert invocations[0].keywords == []

    top_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == ABSOLUTE_FINAL_CLEANUP
    ]
    assert len(top_invocations) == 1
    assert [ast.unparse(argument) for argument in top_invocations[0].args] == [
        "shared_model_ir_pass_context"
    ]
    assert top_invocations[0].keywords == []


def test_absolute_final_normalization_attention_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == ABSOLUTE_FINAL_CLEANUP_TARGET
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == ABSOLUTE_FINAL_CLEANUP
    )

    invocation = lowerer.body[invocation_index]
    following = lowerer.body[invocation_index + 1]
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == ABSOLUTE_FINAL_CLEANUP
    assert [ast.unparse(argument) for argument in invocation.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert invocation.value.keywords == []
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following) == (
        "session.record_phase_result('topology_layout.primary.absolute_final', "
        "run_topology_layout_refresh(model_ir))"
    )

    calls = [
        node
        for node in ast.walk(_absolute_final_cleanup_owner())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "run_boundary_shape_signature_cleanup",
            ABSOLUTE_FINAL_AFFINE_INSTANCENORM,
            ABSOLUTE_FINAL_NORMALIZATION_ATTENTION,
        }
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    assert [call.func.id for call in calls] == [
        "run_boundary_shape_signature_cleanup",
        ABSOLUTE_FINAL_AFFINE_INSTANCENORM,
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION,
    ]


def test_absolute_final_post_bias_captures_complete_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    top_calls = [
        node
        for node in ast.walk(_absolute_final_cleanup_owner())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            ABSOLUTE_FINAL_AFFINE_INSTANCENORM,
            ABSOLUTE_FINAL_NORMALIZATION_ATTENTION,
        }
    ]
    top_calls.sort(key=lambda node: (node.lineno, node.col_offset))
    assert [call.func.id for call in top_calls] == [
        ABSOLUTE_FINAL_AFFINE_INSTANCENORM,
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION,
    ]
    assert all(ast.unparse(call.args[0]) == "context" for call in top_calls)

    owner_calls = _absolute_final_affine_instancenorm_calls(
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    assert len(owner_calls) == 1
    assert [ast.unparse(argument) for argument in owner_calls[0].args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in owner_calls[0].keywords
    } == {"layout_state": "context.layout_state"}
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
    assert (
        len(direct_statements)
        + _very_late_pad_instancenorm_call_count(lowerer)
        + _pre_terminal_instancenorm_call_count(
            "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
        )
        + len(owner_calls)
        == 4
    )
    terminal_owner = _phase_result_owner(
        direct_statements[0],
        "cleanup.terminal.instancenorm_post_bias",
    )
    assert isinstance(terminal_owner.func, ast.Name)
    assert terminal_owner.func.id == (
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )


def test_absolute_final_affine_post_add_captures_complete_mutation_evidence() -> (
    None
):
    lowerer, _ = _lowerer_and_helper()
    top_calls = [
        node
        for node in ast.walk(_absolute_final_cleanup_owner())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "run_boundary_shape_signature_cleanup",
            ABSOLUTE_FINAL_AFFINE_INSTANCENORM,
            ABSOLUTE_FINAL_NORMALIZATION_ATTENTION,
        }
    ]
    top_calls.sort(key=lambda node: (node.lineno, node.col_offset))
    assert [call.func.id for call in top_calls] == [
        "run_boundary_shape_signature_cleanup",
        ABSOLUTE_FINAL_AFFINE_INSTANCENORM,
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION,
    ]
    assert ast.unparse(top_calls[1]) == (
        f"{ABSOLUTE_FINAL_AFFINE_INSTANCENORM}(context)"
    )

    owner_calls = _absolute_final_affine_instancenorm_calls(
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )
    assert len(owner_calls) == 1
    assert [ast.unparse(argument) for argument in owner_calls[0].args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in owner_calls[0].keywords
    } == {"layout_state": "context.layout_state"}

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
    assert (
        len(direct_statements)
        + PRE_TERMINAL_AFFINE_TAIL_PASS_IDS.count(
            "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
        )
        + TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS.count(
            "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
        )
        + len(owner_calls)
        == 3
    )
    assert direct_statements == []


def test_absolute_final_normalization_attention_context_and_composite_are_explicit() -> (
    None
):
    lowerer, helper = _lowerer_and_helper()
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    assert isinstance(statement.value, ast.Tuple)
    assert len(statement.value.elts) == 2
    normalization_call = statement.value.elts[0]
    dynamic_call = statement.value.elts[1]
    assert isinstance(normalization_call, ast.Call)
    assert isinstance(normalization_call.func, ast.Name)
    assert normalization_call.func.id == "run_absolute_final_normalization_attention"
    assert tuple(_expression_path(arg) for arg in normalization_call.args) == (
        "context",
    )
    assert normalization_call.keywords == []
    assert isinstance(dynamic_call, ast.Call)
    assert isinstance(dynamic_call.func, ast.Name)
    assert dynamic_call.func.id == (
        "rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    )
    assert tuple(_expression_path(arg) for arg in dynamic_call.args) == (
        "context.model_ir",
    )
    assert {
        keyword.arg: _expression_path(keyword.value)
        for keyword in dynamic_call.keywords
    } == {"layout_state": "context.layout_state"}

    top_calls = _absolute_final_cleanup_calls(
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    )
    assert len(top_calls) == 1
    assert ast.unparse(top_calls[0]) == (
        "run_absolute_final_normalization_attention_rank1_cleanup(context)"
    )
    invocation = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == ABSOLUTE_FINAL_CLEANUP_TARGET
    )
    assert ast.unparse(invocation.value) == (
        "run_absolute_final_cleanup(shared_model_ir_pass_context)"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "absolute_final_normalization_attention_context"
        for node in ast.walk(lowerer)
    )


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
    assert isinstance(helper_statement.value, ast.Tuple)
    assert len(helper_statement.value.elts) == 2
    raw_call = helper_statement.value.elts[0]
    assert isinstance(raw_call, ast.Call)
    assert isinstance(raw_call.func, ast.Name)
    assert raw_call.func.id == "run_absolute_final_normalization_attention"

    top_calls = _absolute_final_cleanup_calls(
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION
    )
    assert len(top_calls) == 1
    assert ast.unparse(top_calls[0]) == (
        "run_absolute_final_normalization_attention_rank1_cleanup(context)"
    )

    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == ABSOLUTE_FINAL_CLEANUP_TARGET
    )
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == ABSOLUTE_FINAL_CLEANUP
    assert [ast.unparse(argument) for argument in invocation.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert invocation.value.keywords == []

    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following) == (
        "session.record_phase_result('topology_layout.primary.absolute_final', "
        "run_topology_layout_refresh(model_ir))"
    )


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
