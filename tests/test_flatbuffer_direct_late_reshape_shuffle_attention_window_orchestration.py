from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.channel_shuffle_gather_orchestration import (
    run_channel_shuffle_gather,
)
from onnx2tf.tflite_builder.passes.late_attention_layout_orchestration import (
    run_late_attention_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_reshape_layout_orchestration import (
    run_late_reshape_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_window_layout_orchestration import (
    run_late_window_layout_cleanup,
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
COMPOSITE_TARGET = "_late_reshape_shuffle_attention_window_results"
PREDECESSOR_TARGET = "_late_concat_elementwise_fanout_stats"
SUCCESSOR_TARGET = "_late_final_shape_activation_convergence_stats"


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
    indices = tuple(
        next(
            index
            for index, statement in enumerate(lowerer.body)
            if _single_target(statement) == target
        )
        for target in RESULT_TARGETS
    )
    assert indices == tuple(range(indices[0], indices[0] + len(indices)))
    predecessor = lowerer.body[indices[0] - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == "optimize_layout_transpose_chains"
    assert [_single_target(statement) for statement in predecessor.body] == [
        PREDECESSOR_TARGET
    ]
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET

    calls = tuple(_call(lowerer.body[index]) for index in indices)
    assert all(call is not None for call in calls)
    assert tuple(_call_name(lowerer.body[index]) for index in indices) == (
        CURRENT_CHILD_OWNERS
    )
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "shared_model_ir_pass_context"
    ]
    assert calls[0].keywords == []
    assert calls[1].args == []
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {
        "include_two_way_shuffle": "False",
        "include_nhwc_shuffle": "False",
    }
    for call in calls[2:]:
        assert [ast.unparse(argument) for argument in call.args] == [
            "shared_model_ir_pass_context"
        ]
        assert call.keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("late_reshape_shuffle_attention_window_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = (
        run_late_reshape_layout_cleanup(context),
        run_channel_shuffle_gather(
            context,
            include_two_way_shuffle=False,
            include_nhwc_shuffle=False,
        ),
        run_late_attention_layout_cleanup(context),
        run_late_window_layout_cleanup(context),
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
    assert len(helper_calls) == 2
    policies = tuple(
        {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        }
        for call in sorted(helper_calls, key=lambda call: call.lineno)
    )
    assert policies == (
        {"include_post_gather_cleanup": "True"},
        {
            "include_two_way_shuffle": "False",
            "include_nhwc_shuffle": "False",
        },
    )


@pytest.mark.xfail(
    strict=True,
    reason="late reshape/shuffle/attention/window sequence has no context owner",
)
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
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
