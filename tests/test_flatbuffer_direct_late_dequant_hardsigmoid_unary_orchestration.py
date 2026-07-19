from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_dequant_hardsigmoid_unary_orchestration,
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
    / "late_dequant_hardsigmoid_unary_orchestration.py"
)
OWNER = "run_late_dequant_hardsigmoid_unary_cleanup"
CHILD_OWNERS = (
    "optimize_transpose_dequant_hardsigmoid_quantize_bridges",
    "run_late_dequant_unary_fanout",
)
CURRENT_CHILD_OWNERS = (
    "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
    "_run_late_dequant_unary_fanout_pass_cluster",
)
RESULT_TARGETS = (
    "_late_dequant_hardsigmoid_bridge_stats",
    "_late_dequant_unary_fanout_results",
)
COMPOSITE_TARGET = "_late_dequant_hardsigmoid_unary_results"
PREDECESSOR_GUARD = "optimize_layout_transpose_chains"
SUCCESSOR_TARGET = "_late_swish_transpose_passthrough_stats"
SUCCESSOR_OWNER = "_optimize_swish_transpose_passthrough_chains"
EXPECTED_SCHEMAS = (
    {"removed_transpose_dequant_hardsigmoid_quantize_bridges": 0},
    (
        {
            "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains": 0
        },
        {"rewritten_transpose_unary_passthrough_chains": 0},
        {"rewritten_transpose_unary_fanout_inverse_post_bridges": 0},
    ),
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr, ast.Return)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def test_late_dequant_hardsigmoid_unary_current_boundary_and_schema() -> None:
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
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert _call_name(lowerer.body[index + 1]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("late_dequant_hardsigmoid_unary_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        late_dequant_hardsigmoid_unary_orchestration.run_late_dequant_hardsigmoid_unary_cleanup(
            context
        )
        == EXPECTED_SCHEMAS
    )


def test_late_dequant_hardsigmoid_unary_has_one_context_owner() -> None:
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
        "context.model_ir"
    ]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
    assert calls[1].keywords == []

    lowerer = _lowerer()
    assert CURRENT_CHILD_OWNERS[0] in _functions(LOWERER_PATH)
    assert any(
        isinstance(statement, ast.FunctionDef)
        and statement.name == CURRENT_CHILD_OWNERS[1]
        for statement in lowerer.body
    )
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
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert _call_name(lowerer.body[index + 1]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_late_dequant_hardsigmoid_unary_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_dequant_hardsigmoid_unary_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = (
        {"hard_sigmoid": 1},
        tuple({f"fanout_{index}": index} for index in range(3)),
    )
    observed: list[tuple[str, object]] = []

    def _hard_sigmoid(active_model_ir: ModelIR) -> dict[str, int]:
        observed.append((CHILD_OWNERS[0], active_model_ir))
        return expected_results[0]

    def _fanout(
        active_context: ModelIRPassContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[1], active_context))
        return expected_results[1]

    monkeypatch.setattr(
        late_dequant_hardsigmoid_unary_orchestration,
        CHILD_OWNERS[0],
        _hard_sigmoid,
    )
    monkeypatch.setattr(
        late_dequant_hardsigmoid_unary_orchestration,
        CHILD_OWNERS[1],
        _fanout,
    )

    actual = (
        late_dequant_hardsigmoid_unary_orchestration.run_late_dequant_hardsigmoid_unary_cleanup(
            context
        )
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], context.model_ir),
        (CHILD_OWNERS[1], context),
    ]
