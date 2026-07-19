from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_reshape_shuffle_attention_window_orchestration,
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
    / "late_reshape_shuffle_attention_window_orchestration.py"
)
OWNER = "run_late_reshape_shuffle_attention_window_cleanup"
FULL_POST_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_pass_set_2_channel_preadd_orchestration.py"
)
FULL_POST_OWNER = "run_layout_pass_set_2_channel_preadd_recovery"
CHILD_OWNERS = (
    "run_late_reshape_layout_cleanup",
    "run_channel_shuffle_gather",
    "run_late_attention_layout_cleanup",
    "run_late_window_layout_cleanup",
)
CURRENT_CHILD_OWNERS = (
    CHILD_OWNERS[0],
    "_run_channel_shuffle_gather_layout_pass_cluster",
    CHILD_OWNERS[2],
    CHILD_OWNERS[3],
)
RESULT_TARGETS = (
    "_late_reshape_layout_results",
    "_late_channel_shuffle_gather_results",
    "_late_attention_layout_results",
    "_late_window_layout_results",
)
COMPOSITE_TARGET = "_late_final_shape_boundary_results"
OUTER_OWNER = "run_late_final_shape_boundary_cleanup"
PREDECESSOR_TARGET = "_late_concat_elementwise_fanout_stats"
SUCCESSOR_TARGET = "_terminal_elementwise_fanout_stats"


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


def test_late_reshape_shuffle_attention_window_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OUTER_OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "late_final_shape_boundary_context"
    ]
    assert call.keywords == []
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == "optimize_layout_transpose_chains"
    assert [_single_target(statement) for statement in predecessor.body] == [
        PREDECESSOR_TARGET
    ]
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("late_reshape_shuffle_attention_window_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = late_reshape_shuffle_attention_window_orchestration.run_late_reshape_shuffle_attention_window_cleanup(
        context
    )
    assert tuple(type(result) for result in results) == (tuple,) * 4
    assert tuple(len(result) for result in results) == (3, 2, 4, 2)
    assert all(
        isinstance(child_result, dict)
        for result in results
        for child_result in result
    )


def test_late_reshape_shuffle_attention_window_independent_policy_is_fixed() -> None:
    lowerer = _lowerer()
    helper_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CURRENT_CHILD_OWNERS[1]
    ]
    assert helper_calls == []

    full_post_owner = _functions(FULL_POST_OWNER_PATH)[FULL_POST_OWNER]
    full_post_calls = [
        node
        for node in ast.walk(full_post_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHILD_OWNERS[1]
    ]
    assert len(full_post_calls) == 1
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in full_post_calls[0].keywords
    } == {"include_post_gather_cleanup": "True"}

    owner = _functions(OWNER_PATH)[OWNER]
    channel_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHILD_OWNERS[1]
    ]
    assert len(channel_calls) == 1
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in channel_calls[0].keywords
    } == {
        "include_two_way_shuffle": "False",
        "include_nhwc_shuffle": "False",
    }


def test_late_reshape_shuffle_attention_window_has_one_context_owner() -> None:
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
    } == {
        "include_two_way_shuffle": "False",
        "include_nhwc_shuffle": "False",
    }
    assert all(call.keywords == [] for call in calls[2:])

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OUTER_OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "late_final_shape_boundary_context"
    ]
    assert call.keywords == []
    assert isinstance(lowerer.body[index - 1], ast.If)
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_late_reshape_shuffle_attention_window_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_reshape_shuffle_attention_window_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = tuple({"stage": index} for index in range(len(CHILD_OWNERS)))
    calls: list[
        tuple[str, ModelIRPassContext, dict[str, bool]]
    ] = []

    def callback(index: int):
        def run(
            active_context: ModelIRPassContext,
            **kwargs: bool,
        ) -> dict[str, int]:
            calls.append((CHILD_OWNERS[index], active_context, kwargs))
            return results[index]

        return run

    for index, name in enumerate(CHILD_OWNERS):
        monkeypatch.setattr(
            late_reshape_shuffle_attention_window_orchestration,
            name,
            callback(index),
        )

    actual = late_reshape_shuffle_attention_window_orchestration.run_late_reshape_shuffle_attention_window_cleanup(
        context
    )

    assert actual == results
    assert all(actual[index] is results[index] for index in range(len(results)))
    assert calls == [
        (CHILD_OWNERS[0], context, {}),
        (
            CHILD_OWNERS[1],
            context,
            {
                "include_two_way_shuffle": False,
                "include_nhwc_shuffle": False,
            },
        ),
        (CHILD_OWNERS[2], context, {}),
        (CHILD_OWNERS[3], context, {}),
    ]
