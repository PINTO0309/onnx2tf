from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.mean_attention_orchestration import (
    run_mean_attention,
)
from onnx2tf.tflite_builder.passes.terminal_boundary_layout_orchestration import (
    run_terminal_boundary_layout,
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
    / "terminal_boundary_mean_attention_orchestration.py"
)
OWNER = "run_terminal_boundary_mean_attention_cleanup"
CHILD_OWNERS = (
    "run_terminal_boundary_layout",
    "run_mean_attention",
)
CURRENT_CHILD_OWNERS = (
    "_run_terminal_boundary_layout_pass_cluster",
    "_run_mean_attention_layout_pass_cluster",
)
RESULT_TARGETS = (
    "_terminal_boundary_layout_results",
    "_terminal_mean_attention_results",
)
COMPOSITE_TARGET = "_terminal_boundary_mean_attention_results"
PREDECESSOR_PHASE_ID = "cleanup.terminal.instancenorm_dualstats"
SUCCESSOR_PHASE_ID = "cleanup.terminal.batchmatmul_affine_input"
AFTER_GUARD_TARGET = "_terminal_clamp_sinet_layout_results"
GUARD = "optimize_layout_transpose_chains"

BOUNDARY_SCHEMA = (
    ("optimized_transpose_dual_mul_concat_prepost_nhwc_chains",),
    ("removed_boundary_input_layout_transpose",),
    (
        "optimized_transpose_pad_prepost_nhwc_chains",
        "optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains",
        "optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains",
    ),
    (
        "iterations",
        "removed_identity_transpose",
        "removed_inverse_transpose_pairs",
        "removed_inverse_transpose_fanout_branches",
        "composed_consecutive_transpose_pairs",
    ),
    ("optimized_transpose_gather_transpose_nhwc_channel_chains",),
)
MEAN_SCHEMA = (
    ("optimized_transpose_mean_prepost_nhwc_passthrough_chains",),
    ("optimized_transpose_mean_mul_reshape_add_conv_nhwc_chains",),
    ("optimized_transpose_pre_unary_mean_terminal_nhwc_chains",),
    ("optimized_transpose_se_conv_mul_prepost_nhwc_chains",),
    ("optimized_transpose_se_fc_mul_prepost_nhwc_chains",),
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


def _guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == GUARD
        and any(
            _single_target(candidate) == RESULT_TARGETS[1]
            or _phase_id(candidate) == SUCCESSOR_PHASE_ID
            for candidate in statement.body
        )
    )


def _context() -> ModelIRPassContext:
    model_ir = ModelIR("terminal_boundary_mean_attention_schema")
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_terminal_boundary_mean_attention_current_contract() -> None:
    lowerer = _lowerer()
    boundary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGETS[0]
    )
    boundary_index = lowerer.body.index(boundary)
    assert _call_name(boundary) == CURRENT_CHILD_OWNERS[0]
    boundary_call = _call(boundary)
    assert boundary_call is not None
    assert boundary_call.args == []
    assert boundary_call.keywords == []
    assert _phase_id(lowerer.body[boundary_index - 1]) == PREDECESSOR_PHASE_ID

    guard = _guard(lowerer)
    assert lowerer.body[boundary_index + 1] is guard
    assert guard.orelse == []
    mean = guard.body[0]
    assert _single_target(mean) == RESULT_TARGETS[1]
    assert _call_name(mean) == CURRENT_CHILD_OWNERS[1]
    mean_call = _call(mean)
    assert mean_call is not None
    assert mean_call.args == []
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in mean_call.keywords
    } == {"include_conv_attention": False}
    assert _phase_id(guard.body[1]) == SUCCESSOR_PHASE_ID
    assert _single_target(lowerer.body[lowerer.body.index(guard) + 1]) == (
        AFTER_GUARD_TARGET
    )
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    lowerer_functions = {
        node.name: node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
    }
    assert all(name in lowerer_functions for name in CURRENT_CHILD_OWNERS)
    assert ast.unparse(
        lowerer_functions[CURRENT_CHILD_OWNERS[0]].body[0].value.args[0]
    ) == "terminal_boundary_layout_context"
    assert ast.unparse(
        lowerer_functions[CURRENT_CHILD_OWNERS[1]].body[0].value.args[0]
    ) == "mean_attention_context"


def test_terminal_boundary_mean_attention_child_schemas() -> None:
    context = _context()
    boundary_results = run_terminal_boundary_layout(context)
    mean_results = run_mean_attention(
        context,
        include_conv_attention=False,
    )

    assert _dict_schema(boundary_results) == BOUNDARY_SCHEMA
    assert _dict_schema(mean_results) == MEAN_SCHEMA


@pytest.mark.xfail(
    strict=True,
    reason="terminal boundary/optional mean-attention owner is not implemented",
)
def test_terminal_boundary_mean_attention_has_one_optional_context_owner() -> None:
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
    assert all(
        [ast.unparse(argument) for argument in call.args] == ["context"]
        for call in calls
    )
    assert calls[0].keywords == []
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[1].keywords
    } == {"include_conv_attention": False}
    mean_guard = next(
        node
        for node in owner.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "include_mean_attention"
    )
    assert calls[1] in list(ast.walk(mean_guard))

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_mean_attention": GUARD}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    guard = lowerer.body[index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == GUARD
    assert guard.orelse == []
    assert _phase_id(guard.body[0]) == SUCCESSOR_PHASE_ID
    assert _single_target(lowerer.body[index + 2]) == AFTER_GUARD_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    nested_functions = {
        node.name
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
    }
    assert all(name in nested_functions for name in CURRENT_CHILD_OWNERS)
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )
