from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_affine_optional_fanout_orchestration,
)
from onnx2tf.tflite_builder.passes.elementwise_fanout_layout import (
    optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
)
from onnx2tf.tflite_builder.passes.late_affine_concat_orchestration import (
    run_late_affine_concat_cleanup,
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
    / "late_affine_optional_fanout_orchestration.py"
)
OWNER = "run_late_affine_optional_fanout_cleanup"
LOWERER_OWNER = "run_late_affine_final_shape_terminal_cleanup"
CHILD_OWNERS = (
    "run_late_affine_concat_cleanup",
    "optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
)
CURRENT_CHILD_OWNERS = (
    "run_late_affine_concat_cleanup",
    "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
)
RESULT_TARGETS = (
    "_late_affine_concat_results",
    "_late_concat_elementwise_fanout_stats",
)
COMPOSITE_TARGET = "_late_affine_final_shape_terminal_results"
PREDECESSOR_PHASE_ID = "cleanup.late.ndhwc_cost_volume"
SUCCESSOR_TARGET = "_terminal_convpool_output_passthrough_stats"
SUCCESSOR_OWNER = (
    "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
)
GUARD = "optimize_layout_transpose_chains"

AFFINE_SCHEMA = (
    (
        "folded_conv_mul_add_affine_chains",
        "folded_conv_add_only_affine_chains",
        "folded_conv_mul_only_affine_chains",
        "folded_conv_mul_add_only_affine_chains",
    ),
    (
        ("optimized_transpose_axis3_const_concat_bridge_nhwc_chains",),
        ("optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains",),
        (
            "optimized_transpose_layernorm_stats_nhwc_propagation_chains",
            "optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains",
        ),
        (
            "iterations",
            "removed_identity_transpose",
            "removed_inverse_transpose_pairs",
            "removed_inverse_transpose_fanout_branches",
            "composed_consecutive_transpose_pairs",
        ),
    ),
)
FANOUT_SCHEMA = (
    "optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
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


def _optional_guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == GUARD
        and any(
            _single_target(candidate) == RESULT_TARGETS[1]
            for candidate in statement.body
        )
    )


def _context() -> ModelIRPassContext:
    model_ir = ModelIR("late_affine_optional_fanout_schema")
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_late_affine_optional_fanout_current_contract() -> None:
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
    } == {"include_elementwise_fanout": GUARD}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID

    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert _call_name(successor.body[0]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    wrapper = _functions(LOWERER_PATH)[CURRENT_CHILD_OWNERS[1]]
    assert len(wrapper.body) == 1
    delegate = wrapper.body[0]
    assert isinstance(delegate, ast.Return)
    assert isinstance(delegate.value, ast.Call)
    assert ast.unparse(delegate.value.func).endswith(
        "optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains_pass"
    )
    assert [ast.unparse(argument) for argument in delegate.value.args] == [
        "model_ir"
    ]


def test_late_affine_optional_fanout_child_schemas() -> None:
    context = _context()
    affine_results = run_late_affine_concat_cleanup(context)
    fanout_results = (
        optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
            context.model_ir
        )
    )

    assert tuple(affine_results[0]) == AFFINE_SCHEMA[0]
    assert _dict_schema(affine_results[1]) == AFFINE_SCHEMA[1]
    assert tuple(fanout_results) == FANOUT_SCHEMA


def test_late_affine_optional_fanout_has_one_optional_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.model_ir"
    ]
    assert calls[1].keywords == []
    fanout_guard = next(
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "include_elementwise_fanout"
    )
    assert calls[1] in list(ast.walk(fanout_guard))

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
    } == {"include_elementwise_fanout": GUARD}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert _call_name(successor.body[0]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    lowerer_functions = {
        node.name
        for node in ast.parse(LOWERER_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    assert CURRENT_CHILD_OWNERS[1] in lowerer_functions
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


@pytest.mark.parametrize("include_elementwise_fanout", [False, True])
def test_late_affine_optional_fanout_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    include_elementwise_fanout: bool,
) -> None:
    context = _context()
    affine_results = ({"affine": 1}, ({"concat": 2},))
    fanout_results = {"fanout": 3}
    observed: list[tuple[str, object]] = []

    def _affine(
        active_context: ModelIRPassContext,
    ) -> tuple[dict[str, int], tuple[dict[str, int], ...]]:
        observed.append((CHILD_OWNERS[0], active_context))
        return affine_results

    def _fanout(active_model_ir: ModelIR) -> dict[str, int]:
        observed.append((CHILD_OWNERS[1], active_model_ir))
        return fanout_results

    monkeypatch.setattr(
        late_affine_optional_fanout_orchestration,
        CHILD_OWNERS[0],
        _affine,
    )
    monkeypatch.setattr(
        late_affine_optional_fanout_orchestration,
        CHILD_OWNERS[1],
        _fanout,
    )

    actual = (
        late_affine_optional_fanout_orchestration.run_late_affine_optional_fanout_cleanup(
            context,
            include_elementwise_fanout=include_elementwise_fanout,
        )
    )
    assert actual[0] is affine_results
    if include_elementwise_fanout:
        assert actual[1] is fanout_results
        assert observed == [
            (CHILD_OWNERS[0], context),
            (CHILD_OWNERS[1], context.model_ir),
        ]
    else:
        assert actual[1] is None
        assert observed == [(CHILD_OWNERS[0], context)]
