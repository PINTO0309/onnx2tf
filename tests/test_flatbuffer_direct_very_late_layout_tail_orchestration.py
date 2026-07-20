from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    very_late_layout_tail_orchestration,
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
OUTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_swish_layout_tail_orchestration.py"
)
OWNER = "run_very_late_layout_tail_cleanup"
OUTER_OWNER = "run_late_swish_layout_tail_cleanup"
FALLBACK_NORM_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "fallback_norm_adapter_reshape_orchestration.py"
)
FALLBACK_NORM_OWNER = "run_fallback_norm_adapter_reshape_cleanup"
LOWERER_OWNER = "run_late_dequant_swish_layout_tail_cleanup"
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
COMPOSITE_TARGET = "_late_dequant_swish_layout_tail_results"
PREDECESSOR_GUARD = "optimize_layout_transpose_chains"
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
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == LOWERER_OWNER
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
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("very_late_layout_tail_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = very_late_layout_tail_orchestration.run_very_late_layout_tail_cleanup(
        context,
        include_layout_transpose=include_layout_transpose,
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
    assert calls == []
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
    fallback_owner = _functions(FALLBACK_NORM_OWNER_PATH)[FALLBACK_NORM_OWNER]
    fallback_calls = [
        node
        for node in ast.walk(fallback_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHILD_OWNERS[2]
    ]
    assert len(fallback_calls) == 1
    assert [ast.unparse(argument) for argument in fallback_calls[0].args] == [
        "context"
    ]
    assert fallback_calls[0].keywords == []


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

    outer_owner = _functions(OUTER_OWNER_PATH)[OUTER_OWNER]
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(outer_owner)
    ) == 1

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == LOWERER_OWNER
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
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_very_late_layout_tail_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("very_late_layout_tail_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = tuple({"stage": index} for index in range(len(CHILD_OWNERS)))
    calls: list[tuple[str, ModelIRPassContext, dict[str, bool]]] = []

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
            very_late_layout_tail_orchestration,
            name,
            callback(index),
        )

    actual = very_late_layout_tail_orchestration.run_very_late_layout_tail_cleanup(
        context,
        include_layout_transpose=True,
    )

    assert actual == results
    assert all(actual[index] is results[index] for index in range(len(results)))
    assert calls == [
        (CHILD_OWNERS[0], context, {}),
        (CHILD_OWNERS[1], context, {}),
        (CHILD_OWNERS[2], context, {}),
        (
            CHILD_OWNERS[3],
            context,
            {"include_layout_transpose": True},
        ),
    ]
