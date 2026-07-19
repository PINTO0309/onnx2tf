from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.late_conv1d_decoder_layout_orchestration import (
    run_late_conv1d_decoder_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.singleton_consecutive_reshape_orchestration import (
    run_singleton_consecutive_reshape,
)
from onnx2tf.tflite_builder.passes.very_late_layout_broadcast_orchestration import (
    run_very_late_layout_broadcast_cleanup,
)
from onnx2tf.tflite_builder.passes.very_late_pad_instancenorm_layout_orchestration import (
    run_very_late_pad_instancenorm_layout_cleanup,
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
    / "very_late_layout_tail_orchestration.py"
)
OWNER = "run_very_late_layout_tail_cleanup"
CHILD_OWNERS = (
    "run_late_conv1d_decoder_layout_cleanup",
    "run_very_late_pad_instancenorm_layout_cleanup",
    "run_singleton_consecutive_reshape",
    "run_very_late_layout_broadcast_cleanup",
)
SINGLETON_WRAPPER = "_run_singleton_consecutive_reshape_pass_cluster"
CURRENT_CHILD_OWNERS = (
    CHILD_OWNERS[0],
    CHILD_OWNERS[1],
    SINGLETON_WRAPPER,
    CHILD_OWNERS[3],
)
RESULT_TARGETS = (
    "_late_conv1d_decoder_layout_results",
    "_very_late_pad_instancenorm_layout_results",
    "_very_late_singleton_consecutive_reshape_results",
    "_very_late_layout_broadcast_results",
)
COMPOSITE_TARGET = "_very_late_layout_tail_results"
PREDECESSOR_TARGET = "_late_swish_transpose_passthrough_stats"
SUCCESSOR_PHASE = "shape_reconciliation.primary.very_late_broadcast"


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
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[0], ast.Constant)
    ):
        return None
    return str(call.args[0].value)


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_very_late_layout_tail_current_boundary_and_schema(
    include_layout_transpose: bool,
) -> None:
    lowerer = _lowerer()
    indices = tuple(
        next(
            index
            for index, statement in enumerate(lowerer.body)
            if _single_target(statement) == target
        )
        for target in RESULT_TARGETS
    )
    assert indices == tuple(range(indices[0], indices[0] + len(indices)))
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _phase_id(lowerer.body[indices[-1] + 1]) == SUCCESSOR_PHASE
    assert tuple(_call_name(lowerer.body[index]) for index in indices) == (
        CURRENT_CHILD_OWNERS
    )
    calls = tuple(_call(lowerer.body[index]) for index in indices)
    assert all(call is not None for call in calls)
    for call in calls[:2]:
        assert [ast.unparse(argument) for argument in call.args] == [
            "shared_model_ir_pass_context"
        ]
        assert call.keywords == []
    assert [ast.unparse(argument) for argument in calls[2].args] == [
        "model_ir",
        "session.layout_state",
    ]
    assert calls[2].keywords == []
    assert [ast.unparse(argument) for argument in calls[3].args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[3].keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("very_late_layout_tail_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = (
        run_late_conv1d_decoder_layout_cleanup(context),
        run_very_late_pad_instancenorm_layout_cleanup(context),
        run_singleton_consecutive_reshape(context),
        run_very_late_layout_broadcast_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
    assert tuple(type(result) for result in results) == (tuple,) * 4
    assert tuple(len(result) for result in results) == (8, 4, 3, 2)
    if include_layout_transpose:
        assert isinstance(results[3][0], dict)
    else:
        assert results[3][0] is None
    assert isinstance(results[3][1], dict)


def test_singleton_consecutive_wrapper_independent_route_is_fixed() -> None:
    lowerer = _lowerer()
    calls = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == SINGLETON_WRAPPER
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert len(calls) == 2
    assert tuple(
        [ast.unparse(argument) for argument in call.args] for call in calls
    ) == (
        ["model_ir", "session.layout_state"],
        ["fallback_ir", "None"],
    )
    assert all(call.keywords == [] for call in calls)
    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == SINGLETON_WRAPPER
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    call = wrapper.body[0].value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == CHILD_OWNERS[2]


@pytest.mark.xfail(
    strict=True,
    reason="very-late Conv1D/Pad/Reshape/Broadcast tail has no context owner",
)
def test_very_late_layout_tail_has_one_context_owner() -> None:
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
    assert all(call.keywords == [] for call in calls[:3])
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[3].keywords
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
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
