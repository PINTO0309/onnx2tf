from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_dequant_swish_layout_tail_orchestration,
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
    / "late_dequant_swish_layout_tail_orchestration.py"
)
OWNER = "run_late_dequant_swish_layout_tail_cleanup"
CHILD_OWNERS = (
    "run_late_dequant_hardsigmoid_unary_cleanup",
    "run_late_swish_layout_tail_cleanup",
)
RESULT_TARGETS = (
    "_late_dequant_hardsigmoid_unary_results",
    "_late_swish_layout_tail_results",
)
COMPOSITE_TARGET = "_late_dequant_swish_layout_tail_results"
PREDECESSOR_GUARD = "optimize_layout_transpose_chains"
SUCCESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_broadcast"


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
def test_late_dequant_swish_layout_tail_current_boundary_and_schema(
    include_layout_transpose: bool,
) -> None:
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
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("late_dequant_swish_layout_tail_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = (
        late_dequant_swish_layout_tail_orchestration.run_late_dequant_swish_layout_tail_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        )
    )
    assert tuple(type(result) for result in results) == (tuple, tuple)
    assert tuple(len(result) for result in results) == (2, 2)
    assert tuple(type(result) for result in results[0]) == (dict, tuple)
    assert tuple(type(result) for result in results[0][1]) == (dict,) * 3
    assert tuple(type(result) for result in results[1]) == (dict, tuple)
    assert len(results[1][1]) == 4
    assert tuple(len(result) for result in results[1][1][:3]) == (8, 4, 3)
    broadcast_results = results[1][1][3]
    assert len(broadcast_results) == 2
    assert (broadcast_results[0] is None) is (not include_layout_transpose)
    assert isinstance(broadcast_results[1], dict)


def test_late_dequant_swish_layout_tail_has_one_shared_context_owner() -> None:
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
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_late_dequant_swish_layout_tail_runtime_order_option_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
) -> None:
    model_ir = ModelIR("late_dequant_swish_layout_tail_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = (
        ({"dequant": 1}, tuple({f"fanout_{index}": index} for index in range(3))),
        ({"swish": 1}, tuple({f"tail_{index}": index} for index in range(4))),
    )
    observed: list[tuple[str, object, dict[str, object]]] = []

    def dequant(active_context: ModelIRPassContext) -> tuple[object, ...]:
        observed.append((CHILD_OWNERS[0], active_context, {}))
        return expected_results[0]

    def swish(
        active_context: ModelIRPassContext,
        **options: object,
    ) -> tuple[object, ...]:
        observed.append((CHILD_OWNERS[1], active_context, options))
        return expected_results[1]

    monkeypatch.setattr(
        late_dequant_swish_layout_tail_orchestration,
        CHILD_OWNERS[0],
        dequant,
    )
    monkeypatch.setattr(
        late_dequant_swish_layout_tail_orchestration,
        CHILD_OWNERS[1],
        swish,
    )

    actual = (
        late_dequant_swish_layout_tail_orchestration.run_late_dequant_swish_layout_tail_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        )
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], context, {}),
        (
            CHILD_OWNERS[1],
            context,
            {"include_layout_transpose": include_layout_transpose},
        ),
    ]
