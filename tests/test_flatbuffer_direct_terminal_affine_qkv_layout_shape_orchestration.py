from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.pre_terminal_affine_slice_spp_orchestration import (
    run_pre_terminal_affine_slice_spp_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_qkv_activation_layout_shape_orchestration import (
    run_terminal_qkv_activation_layout_shape_cleanup,
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
    / "terminal_affine_qkv_layout_shape_orchestration.py"
)
OWNER = "run_terminal_affine_qkv_layout_shape_cleanup"
CHILD_OWNERS = (
    "run_pre_terminal_affine_slice_spp_cleanup",
    "run_terminal_qkv_activation_layout_shape_cleanup",
)
RESULT_TARGETS = (
    "_pre_terminal_affine_slice_spp_results",
    "_terminal_qkv_activation_layout_shape_results",
)
COMPOSITE_TARGET = "_terminal_affine_qkv_layout_shape_results"
PREDECESSOR_GUARD = "_late_binary_layout_recovery_requires_reconciliation"
SUCCESSOR_PHASE_ID = "shape_reconciliation.terminal.expand_squeeze"


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


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_terminal_affine_qkv_layout_shape_current_boundary_and_schema(
    include_layout_transpose: bool,
) -> None:
    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) in RESULT_TARGETS
    ]
    assert [_single_target(statement) for statement in assignments] == list(
        RESULT_TARGETS
    )
    assert [_call_name(statement) for statement in assignments] == list(
        CHILD_OWNERS
    )
    indices = [lowerer.body.index(statement) for statement in assignments]
    assert indices[1] == indices[0] + 1

    first_call = _call(assignments[0])
    second_call = _call(assignments[1])
    assert first_call is not None
    assert second_call is not None
    assert [ast.unparse(argument) for argument in first_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert first_call.keywords == []
    assert [ast.unparse(argument) for argument in second_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in second_call.keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}

    predecessor = lowerer.body[indices[0] - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[indices[1] + 1]) == SUCCESSOR_PHASE_ID
    assert _call_name(lowerer.body[indices[1] + 2]) == "_advance_post_progress"
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("terminal_affine_qkv_layout_shape_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = (
        run_pre_terminal_affine_slice_spp_cleanup(context),
        run_terminal_qkv_activation_layout_shape_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
    assert tuple(type(result) for result in results) == (tuple, tuple)
    assert tuple(len(result) for result in results) == (2, 2)
    assert tuple(len(result) for result in results[0]) == (5, 3)
    assert tuple(len(result) for result in results[1]) == (2, 4)


@pytest.mark.xfail(
    strict=True,
    reason="terminal affine/QKV/layout-shape composite owner is not implemented",
)
def test_terminal_affine_qkv_layout_shape_has_one_context_owner() -> None:
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
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {"include_layout_transpose": "include_layout_transpose"}

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
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert _call_name(lowerer.body[index + 2]) == "_advance_post_progress"
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
