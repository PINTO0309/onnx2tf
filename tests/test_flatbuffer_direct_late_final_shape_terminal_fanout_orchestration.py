from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_final_shape_terminal_fanout_orchestration,
)
from onnx2tf.tflite_builder.passes.late_final_shape_boundary_orchestration import (
    LateFinalShapeBoundaryContext,
    run_late_final_shape_boundary_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_fanout_singleton_orchestration import (
    run_terminal_fanout_singleton_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
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
    / "late_final_shape_terminal_fanout_orchestration.py"
)
OWNER = "run_late_final_shape_terminal_fanout_cleanup"
CHILD_OWNERS = (
    "run_late_final_shape_boundary_cleanup",
    "run_terminal_fanout_singleton_cleanup",
)
RESULT_TARGETS = (
    "_late_final_shape_boundary_results",
    "_terminal_fanout_singleton_results",
)
COMPOSITE_TARGET = "_late_final_shape_terminal_fanout_results"
PREDECESSOR_TARGET = "_late_affine_optional_fanout_results"
SUCCESSOR_TARGET = "_terminal_convpool_output_passthrough_stats"
SUCCESSOR_OWNER = (
    "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
)
GUARD = "optimize_layout_transpose_chains"
LATE_CONTEXT_TARGET = "late_final_shape_boundary_context"
PASS_CONTEXT_TARGET = "shared_model_ir_pass_context"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr, ast.Return)):
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


def _context(name: str) -> LateFinalShapeBoundaryContext:
    model_ir = ModelIR(name)
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    terminal_context = TerminalSliceConcatRecoveryContext(
        pass_context=pass_context,
        channel_slice_pad_mul_cluster=lambda: (),
    )
    return LateFinalShapeBoundaryContext(
        pass_context=pass_context,
        terminal_slice_concat_context=terminal_context,
    )


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_late_final_shape_terminal_fanout_current_contract() -> None:
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
        LATE_CONTEXT_TARGET
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_elementwise_fanout": GUARD}

    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    successor_guard = lowerer.body[index + 1]
    assert isinstance(successor_guard, ast.If)
    assert ast.unparse(successor_guard.test) == GUARD
    assert _single_target(successor_guard.body[0]) == SUCCESSOR_TARGET
    assert _call_name(successor_guard.body[0]) == SUCCESSOR_OWNER
    assert len(successor_guard.orelse) == 1
    assert isinstance(successor_guard.orelse[0], ast.If)
    assert ast.unparse(successor_guard.orelse[0].test) == (
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    shared_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == PASS_CONTEXT_TARGET
    )
    assert ast.unparse(shared_assignment.value) == "session.model_ir_pass_context"
    late_context_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == LATE_CONTEXT_TARGET
    )
    assert isinstance(late_context_assignment.value, ast.Call)
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in late_context_assignment.value.keywords
    } == {
        "pass_context": PASS_CONTEXT_TARGET,
        "terminal_slice_concat_context": (
            "terminal_slice_concat_recovery_context"
        ),
    }


@pytest.mark.parametrize("include_elementwise_fanout", [False, True])
def test_late_final_shape_terminal_fanout_child_schemas_and_contexts(
    include_elementwise_fanout: bool,
) -> None:
    context = _context(
        f"late_final_shape_terminal_fanout_{include_elementwise_fanout}"
    )
    late_results = run_late_final_shape_boundary_cleanup(context)
    terminal_results = run_terminal_fanout_singleton_cleanup(
        context.pass_context,
        include_elementwise_fanout=include_elementwise_fanout,
    )

    assert tuple(type(result) for result in late_results) == (
        tuple,
        dict,
        tuple,
    )
    assert tuple(len(result) for result in late_results[0]) == (3, 2, 4, 2)
    assert len(late_results[1]) == 11
    assert tuple(len(result) for result in late_results[2]) == (3, 14, 2, 6)
    if include_elementwise_fanout:
        assert tuple(terminal_results[0] or ()) == (
            "optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
        )
    else:
        assert terminal_results[0] is None
    assert _dict_schema(terminal_results[1]) == (
        (
            "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains",
            "rewritten_singleton_nms_maxpool_nhwc_chains",
        ),
        (
            "removed_noop_reshape_chains",
            "rewritten_consecutive_reshape_passthrough_chains",
            "rewritten_fanout_bypass_reshape_passthrough_chains",
        ),
    )
    assert context.terminal_slice_concat_context.pass_context is (
        context.pass_context
    )


def test_late_final_shape_terminal_fanout_has_one_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "context"
    ]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {"include_elementwise_fanout": "include_elementwise_fanout"}

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
        LATE_CONTEXT_TARGET
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_elementwise_fanout": GUARD}
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    successor_guard = lowerer.body[index + 1]
    assert isinstance(successor_guard, ast.If)
    assert ast.unparse(successor_guard.test) == GUARD
    assert _single_target(successor_guard.body[0]) == SUCCESSOR_TARGET
    assert _call_name(successor_guard.body[0]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


@pytest.mark.parametrize("include_elementwise_fanout", [False, True])
def test_late_final_shape_terminal_fanout_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    include_elementwise_fanout: bool,
) -> None:
    context = _context(
        f"late_final_shape_terminal_fanout_runtime_{include_elementwise_fanout}"
    )
    late_results = ({"late": 1},)
    terminal_results = ({"fanout": 2}, ({"singleton": 3},))
    expected_results = (late_results, terminal_results)
    observed: list[tuple[str, object, dict[str, object]]] = []

    def late(active_context: object) -> object:
        observed.append((CHILD_OWNERS[0], active_context, {}))
        return late_results

    def terminal(
        active_context: object,
        **options: object,
    ) -> object:
        observed.append((CHILD_OWNERS[1], active_context, options))
        return terminal_results

    monkeypatch.setattr(
        late_final_shape_terminal_fanout_orchestration,
        CHILD_OWNERS[0],
        late,
    )
    monkeypatch.setattr(
        late_final_shape_terminal_fanout_orchestration,
        CHILD_OWNERS[1],
        terminal,
    )

    actual = late_final_shape_terminal_fanout_orchestration.run_late_final_shape_terminal_fanout_cleanup(
        context,
        include_elementwise_fanout=include_elementwise_fanout,
    )
    assert actual == expected_results
    assert actual[0] is late_results
    assert actual[1] is terminal_results
    assert observed == [
        (CHILD_OWNERS[0], context, {}),
        (
            CHILD_OWNERS[1],
            context.pass_context,
            {"include_elementwise_fanout": include_elementwise_fanout},
        ),
    ]
