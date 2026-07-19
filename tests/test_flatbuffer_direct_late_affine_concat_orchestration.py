from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.conv_mul_affine_fold_compat import (
    optimize_fold_conv_mul_add_affine_chains,
)
from onnx2tf.tflite_builder.passes.late_concat_layout_orchestration import (
    run_late_concat_layout_cleanup,
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
    / "late_affine_concat_orchestration.py"
)
OWNER = "run_late_affine_concat_cleanup"
CHILD_OWNERS = (
    "optimize_fold_conv_mul_add_affine_chains",
    "run_late_concat_layout_cleanup",
)
CURRENT_CHILD_OWNERS = (
    "_optimize_fold_conv_mul_add_affine_chains",
    CHILD_OWNERS[1],
)
RESULT_TARGETS = (
    "_late_cost_volume_conv_affine_stats",
    "_late_concat_layout_results",
)
COMPOSITE_TARGET = "_late_affine_concat_results"
PREDECESSOR_PHASE_ID = "cleanup.late.ndhwc_cost_volume"
SUCCESSOR_TARGET = "_late_concat_elementwise_fanout_stats"
SUCCESSOR_OWNER = (
    "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains"
)
EXPECTED_SCHEMAS = (
    {
        "folded_conv_mul_add_affine_chains": 0,
        "folded_conv_add_only_affine_chains": 0,
        "folded_conv_mul_only_affine_chains": 0,
        "folded_conv_mul_add_only_affine_chains": 0,
    },
    (
        {"optimized_transpose_axis3_const_concat_bridge_nhwc_chains": 0},
        {"optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains": 0},
        {
            "optimized_transpose_layernorm_stats_nhwc_propagation_chains": 0,
            "optimized_layernorm_stats_via_existing_post_transpose_nhwc_chains": 0,
        },
        {
            "iterations": 0,
            "removed_identity_transpose": 0,
            "removed_inverse_transpose_pairs": 0,
            "removed_inverse_transpose_fanout_branches": 0,
            "composed_consecutive_transpose_pairs": 0,
        },
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
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _phase_id(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_late_affine_concat_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    statements = tuple(
        next(
            statement
            for statement in lowerer.body
            if _single_target(statement) == target
        )
        for target in RESULT_TARGETS
    )
    indices = tuple(lowerer.body.index(statement) for statement in statements)
    assert indices == (indices[0], indices[0] + 1)
    assert _phase_id(lowerer.body[indices[0] - 1]) == PREDECESSOR_PHASE_ID
    successor = lowerer.body[indices[-1] + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == "optimize_layout_transpose_chains"
    assert len(successor.body) == 1
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert _call_name(successor.body[0]) == SUCCESSOR_OWNER
    assert tuple(_call_name(statement) for statement in statements) == (
        CURRENT_CHILD_OWNERS
    )

    affine_call = _call(statements[0])
    concat_call = _call(statements[1])
    assert affine_call is not None
    assert concat_call is not None
    assert [ast.unparse(argument) for argument in affine_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in affine_call.keywords
    } == {
        "enable_conv_add_only_fold": "True",
        "layout_state": "session.layout_state",
    }
    assert [ast.unparse(argument) for argument in concat_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert concat_call.keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("late_affine_concat_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        optimize_fold_conv_mul_add_affine_chains(
            context.model_ir,
            enable_conv_add_only_fold=True,
            layout_state=context.layout_state,
        ),
        run_late_concat_layout_cleanup(context),
    ) == EXPECTED_SCHEMAS


@pytest.mark.xfail(
    strict=True,
    reason="late affine/Concat tail has no shared-context owner",
)
def test_late_affine_concat_has_one_context_owner() -> None:
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
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
    } == {
        "enable_conv_add_only_fold": "True",
        "layout_state": "context.layout_state",
    }
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
    assert calls[1].keywords == []

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
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == "optimize_layout_transpose_chains"
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert _call_name(successor.body[0]) == SUCCESSOR_OWNER
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
