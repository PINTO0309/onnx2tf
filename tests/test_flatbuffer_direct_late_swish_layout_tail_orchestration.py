from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    optimize_swish_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.very_late_layout_tail_orchestration import (
    run_very_late_layout_tail_cleanup,
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
    / "late_swish_layout_tail_orchestration.py"
)
OWNER = "run_late_swish_layout_tail_cleanup"
CHILD_OWNERS = (
    "optimize_swish_transpose_passthrough_chains",
    "run_very_late_layout_tail_cleanup",
)
CURRENT_CHILD_OWNERS = (
    "_optimize_swish_transpose_passthrough_chains",
    "run_very_late_layout_tail_cleanup",
)
RESULT_TARGETS = (
    "_late_swish_transpose_passthrough_stats",
    "_very_late_layout_tail_results",
)
COMPOSITE_TARGET = "_late_swish_layout_tail_results"
PREDECESSOR_TARGET = "_late_dequant_hardsigmoid_unary_results"
SUCCESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_broadcast"
SWISH_SCHEMA = {"rewritten_swish_transpose_passthrough_chains": 0}
TAIL_PREFIX_SCHEMA = (
    (
        {"optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_chains": 0},
        {
            "optimized_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains": 0
        },
        {
            "optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains": 0
        },
        {
            "optimized_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains": 0
        },
        {"optimized_tencoder_add_expand_transpose_conv_nhwc_chains": 0},
        {"optimized_transpose_squeeze_unary_batchmatmul_nhwc_chains": 0},
        {
            "optimized_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input": 0
        },
        {"optimized_transpose_squeeze_mean_squeeze_terminal_nhwc_chains": 0},
    ),
    (
        {
            "optimized_transpose_pad_prepost_nhwc_chains": 0,
            "optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains": 0,
            "optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains": 0,
        },
        {
            "optimized_transpose_instancenorm_posttranspose_bias_add_nhwc_chains": 0
        },
        {
            "optimized_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains": 0
        },
        {
            "optimized_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains": 0
        },
    ),
    (
        {"rewritten_singleton_channel_layout_transpose_to_reshape": 0},
        {"removed_duplicate_reshape_fanout": 0},
        {
            "removed_noop_reshape_chains": 0,
            "rewritten_consecutive_reshape_passthrough_chains": 0,
            "rewritten_fanout_bypass_reshape_passthrough_chains": 0,
        },
    ),
)
BROADCAST_REPAIR_SCHEMA = {
    False: (None, {"repaired_rank4_channelwise_broadcast_constants": 0}),
    True: (
        {
            "iterations": 0,
            "removed_identity_transpose": 0,
            "removed_inverse_transpose_pairs": 0,
            "removed_inverse_transpose_fanout_branches": 0,
            "composed_consecutive_transpose_pairs": 0,
        },
        {"repaired_rank4_channelwise_broadcast_constants": 0},
    ),
}


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
def test_late_swish_layout_tail_current_boundary_and_schema(
    include_layout_transpose: bool,
) -> None:
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
    assert tuple(_call_name(statement) for statement in statements) == (
        CURRENT_CHILD_OWNERS
    )
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _phase_id(lowerer.body[indices[-1] + 1]) == SUCCESSOR_PHASE_ID

    swish_call = _call(statements[0])
    assert swish_call is not None
    assert [ast.unparse(argument) for argument in swish_call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in swish_call.keywords
    } == {"layout_state": "session.layout_state"}
    tail_call = _call(statements[1])
    assert tail_call is not None
    assert [ast.unparse(argument) for argument in tail_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in tail_call.keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("late_swish_layout_tail_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        optimize_swish_transpose_passthrough_chains(
            model_ir,
            layout_state=context.layout_state,
        ),
        run_very_late_layout_tail_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    ) == (
        SWISH_SCHEMA,
        (*TAIL_PREFIX_SCHEMA, BROADCAST_REPAIR_SCHEMA[include_layout_transpose]),
    )


@pytest.mark.xfail(
    strict=True,
    reason="late swish/very-late layout tail has no shared-context owner",
)
def test_late_swish_layout_tail_has_one_context_owner() -> None:
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
    } == {"layout_state": "context.layout_state"}
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {"include_layout_transpose": "include_layout_transpose"}

    lowerer = _lowerer()
    assert CURRENT_CHILD_OWNERS[0] in _functions(LOWERER_PATH)
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
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
