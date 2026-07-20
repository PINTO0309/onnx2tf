from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_fanout_singleton_orchestration,
)
from onnx2tf.tflite_builder.passes.elementwise_fanout_layout import (
    optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
)
from onnx2tf.tflite_builder.passes.terminal_singleton_maxpool_reshape_orchestration import (
    run_terminal_singleton_maxpool_reshape,
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
    / "terminal_fanout_singleton_orchestration.py"
)
OWNER = "run_terminal_fanout_singleton_cleanup"
LOWERER_OWNER = "run_late_affine_final_shape_terminal_convpool_cleanup"
CHILD_OWNERS = (
    "optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
    "run_terminal_singleton_maxpool_reshape",
)
CURRENT_CHILD_OWNERS = (
    "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
    "_run_terminal_singleton_maxpool_reshape_pass_pair",
)
RESULT_TARGETS = (
    "_terminal_elementwise_fanout_stats",
    "_terminal_singleton_maxpool_reshape_results",
)
COMPOSITE_TARGET = "_late_affine_final_shape_terminal_convpool_results"
PREDECESSOR_PHASE_ID = "cleanup.late.ndhwc_cost_volume"
SUCCESSOR_TARGET = "_no_layout_fallback_affine_prepost_stats"
SUCCESSOR_OWNER = "_optimize_transpose_mul_add_const_prepost_nhwc_chains"
GUARD = "optimize_layout_transpose_chains"
SUCCESSOR_GUARD = (
    "not optimize_layout_transpose_chains and "
    "apply_safe_transpose_reduction_lite_on_no_layout_opt"
)

FANOUT_SCHEMA = (
    "optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
)
SINGLETON_SCHEMA = (
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


def _context() -> ModelIRPassContext:
    model_ir = ModelIR("terminal_fanout_singleton_schema")
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_terminal_fanout_singleton_current_contract() -> None:
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
        "late_final_shape_boundary_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"optimize_layout_transpose_chains": GUARD}

    predecessor = lowerer.body[index - 1]
    assert _phase_id(predecessor) == PREDECESSOR_PHASE_ID

    successor_guard = lowerer.body[index + 1]
    assert isinstance(successor_guard, ast.If)
    assert ast.unparse(successor_guard.test) == SUCCESSOR_GUARD
    assert _phase_id(successor_guard.body[0]) == (
        "layout.no_layout.safe_transpose_reduction"
    )
    assert _single_target(successor_guard.body[1]) == SUCCESSOR_TARGET
    assert _call_name(successor_guard.body[1]) == SUCCESSOR_OWNER
    assert successor_guard.orelse == []

    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_terminal_fanout_singleton_child_schemas_and_wrappers() -> None:
    context = _context()
    fanout_results = (
        optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
            context.model_ir
        )
    )
    singleton_results = run_terminal_singleton_maxpool_reshape(context)
    assert tuple(fanout_results) == FANOUT_SCHEMA
    assert _dict_schema(singleton_results) == SINGLETON_SCHEMA

    lowerer = _lowerer()
    fanout_wrapper = _functions(LOWERER_PATH)[CURRENT_CHILD_OWNERS[0]]
    singleton_wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == CURRENT_CHILD_OWNERS[1]
    )
    assert len(fanout_wrapper.body) == 1
    assert isinstance(fanout_wrapper.body[0], ast.Return)
    assert ast.unparse(fanout_wrapper.body[0].value.func).endswith(
        "optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains_pass"
    )
    assert len(singleton_wrapper.body) == 1
    assert _call_name(singleton_wrapper.body[0]) == CHILD_OWNERS[1]
    singleton_call = _call(singleton_wrapper.body[0])
    assert singleton_call is not None
    assert [ast.unparse(argument) for argument in singleton_call.args] == [
        "terminal_singleton_maxpool_reshape_context"
    ]
    assert singleton_call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement)
        == "terminal_singleton_maxpool_reshape_context"
    )
    assert ast.unparse(context_assignment.value) == (
        "shared_model_ir_pass_context"
    )


def test_terminal_fanout_singleton_has_one_optional_context_owner() -> None:
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
    fanout_guard = next(
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "include_elementwise_fanout"
    )
    assert calls[0] in list(ast.walk(fanout_guard))
    assert calls[1] not in list(ast.walk(fanout_guard))

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
        "late_final_shape_boundary_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"optimize_layout_transpose_chains": GUARD}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    successor_guard = lowerer.body[index + 1]
    assert isinstance(successor_guard, ast.If)
    assert ast.unparse(successor_guard.test) == SUCCESSOR_GUARD
    assert _phase_id(successor_guard.body[0]) == (
        "layout.no_layout.safe_transpose_reduction"
    )
    assert _single_target(successor_guard.body[1]) == SUCCESSOR_TARGET
    assert _call_name(successor_guard.body[1]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    module_functions = {
        node.name
        for node in ast.parse(LOWERER_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    nested_lowerer_functions = {
        node.name
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
    }
    assert CURRENT_CHILD_OWNERS[0] in module_functions
    assert CURRENT_CHILD_OWNERS[1] in nested_lowerer_functions
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


@pytest.mark.parametrize("include_elementwise_fanout", [False, True])
def test_terminal_fanout_singleton_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    include_elementwise_fanout: bool,
) -> None:
    context = _context()
    fanout_results = {"fanout": 1}
    singleton_results = ({"singleton": 2}, {"reshape": 3})
    observed: list[tuple[str, object]] = []

    def _fanout(active_model_ir: ModelIR) -> dict[str, int]:
        observed.append((CHILD_OWNERS[0], active_model_ir))
        return fanout_results

    def _singleton(
        active_context: ModelIRPassContext,
    ) -> tuple[dict[str, int], ...]:
        observed.append((CHILD_OWNERS[1], active_context))
        return singleton_results

    monkeypatch.setattr(
        terminal_fanout_singleton_orchestration,
        CHILD_OWNERS[0],
        _fanout,
    )
    monkeypatch.setattr(
        terminal_fanout_singleton_orchestration,
        CHILD_OWNERS[1],
        _singleton,
    )

    actual = (
        terminal_fanout_singleton_orchestration.run_terminal_fanout_singleton_cleanup(
            context,
            include_elementwise_fanout=include_elementwise_fanout,
        )
    )
    assert actual[1] is singleton_results
    if include_elementwise_fanout:
        assert actual[0] is fanout_results
        assert observed == [
            (CHILD_OWNERS[0], context.model_ir),
            (CHILD_OWNERS[1], context),
        ]
    else:
        assert actual[0] is None
        assert observed == [(CHILD_OWNERS[1], context)]
