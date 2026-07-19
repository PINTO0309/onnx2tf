from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_qkv_activation_bridge_orchestration,
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
    / "terminal_qkv_activation_bridge_orchestration.py"
)
OWNER = "run_terminal_qkv_activation_bridge_cleanup"
CHILD_OWNERS = (
    "run_terminal_qkv_shape_attention_cleanup",
    "run_terminal_activation_bridge_cleanup",
)
RESULT_TARGETS = (
    "_terminal_qkv_shape_attention_results",
    "_terminal_activation_bridge_results",
)
COMPOSITE_TARGET = "_terminal_qkv_activation_bridge_results"
PREDECESSOR_TARGET = "_pre_terminal_affine_slice_spp_results"
SUCCESSOR_PHASE_ID = "shape_reconciliation.terminal.expand_squeeze"
OUTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_activation_layout_shape_orchestration.py"
)
OUTER_OWNER = "run_terminal_qkv_activation_layout_shape_cleanup"
OUTER_TARGET = "_terminal_qkv_activation_layout_shape_results"
LOWERER_OWNER = "run_terminal_affine_qkv_layout_shape_cleanup"
LOWERER_TARGET = "_terminal_affine_qkv_layout_shape_results"
LOWERER_PREDECESSOR_GUARD = (
    "_late_binary_layout_recovery_requires_reconciliation"
)
EXPECTED_SCHEMAS = (
    (
        {"optimized_transpose_shape_extract_nhwc_to_nchw_chains": 0},
        {
            "removed_identity_transpose": 0,
            "removed_inverse_transpose_pairs": 0,
            "removed_inverse_transpose_fanout_branches": 0,
            "composed_consecutive_transpose_pairs": 0,
            "optimized_attention_qkv_gather_reshape_transpose_hoist_chains": 0,
            "optimized_attention_qkv_slice_replace_gather_reshape_chains": 0,
            "optimized_attention_qkv_slice_to_split_chains": 0,
            "optimized_attention_split_post_reshape_collapse_chains": 0,
            "optimized_attention_qkv_shared_pretranspose_slice_nchw_chains": 0,
            "optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains": 0,
            "pruned_unused_tensors": 0,
        },
    ),
    (
        {"optimized_split_conv_concat_transpose_bridge_to_single_post_nchw": 0},
        {
            "optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains": 0,
            "pruned_unused_tensors": 0,
        },
        {
            "rewritten_hardswish_transpose_passthrough_chains": 0,
            "rewritten_hardsigmoid_transpose_passthrough_chains": 0,
            "rewritten_hardsigmoid_mul_transpose_passthrough_chains": 0,
            "removed_identity_transpose": 0,
            "removed_inverse_transpose_pairs": 0,
            "removed_inverse_transpose_fanout_branches": 0,
            "composed_consecutive_transpose_pairs": 0,
            "pruned_unused_tensors": 0,
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


def _outer_calls() -> list[ast.Call]:
    owner = _functions(OUTER_OWNER_PATH)[OUTER_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_terminal_qkv_activation_bridge_current_boundary_and_schema(
    include_layout_transpose: bool,
) -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == LOWERER_TARGET
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
    assert ast.unparse(predecessor.test) == LOWERER_PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert _call_name(lowerer.body[index + 2]) == "_advance_post_progress"
    assert len(_outer_calls()) == 1
    assert [ast.unparse(argument) for argument in _outer_calls()[0].args] == [
        "context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in _outer_calls()[0].keywords
    } == {"include_layout_transpose": "include_layout_transpose"}
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("terminal_qkv_activation_bridge_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        terminal_qkv_activation_bridge_orchestration.run_terminal_qkv_activation_bridge_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        )
        == EXPECTED_SCHEMAS
    )


def test_terminal_qkv_activation_bridge_has_one_context_owner() -> None:
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
    for call in calls:
        assert [ast.unparse(argument) for argument in call.args] == [
            "context"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"include_layout_transpose": "include_layout_transpose"}

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == LOWERER_TARGET
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
    assert ast.unparse(predecessor.test) == LOWERER_PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert _call_name(lowerer.body[index + 2]) == "_advance_post_progress"
    assert len(_outer_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_terminal_qkv_activation_bridge_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
) -> None:
    model_ir = ModelIR("terminal_qkv_activation_bridge_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = (
        tuple({f"qkv_{index}": index} for index in range(2)),
        tuple({f"activation_{index}": index} for index in range(3)),
    )
    observed: list[
        tuple[str, ModelIRPassContext, dict[str, bool]]
    ] = []

    def _callback(index: int):
        def _run(
            active_context: ModelIRPassContext,
            **kwargs: bool,
        ) -> tuple[dict[str, int], ...]:
            observed.append((CHILD_OWNERS[index], active_context, kwargs))
            return expected_results[index]

        return _run

    for index, child_owner in enumerate(CHILD_OWNERS):
        monkeypatch.setattr(
            terminal_qkv_activation_bridge_orchestration,
            child_owner,
            _callback(index),
        )

    actual = (
        terminal_qkv_activation_bridge_orchestration.run_terminal_qkv_activation_bridge_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        )
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    expected_kwargs = {
        "include_layout_transpose": include_layout_transpose
    }
    assert observed == [
        (CHILD_OWNERS[0], context, expected_kwargs),
        (CHILD_OWNERS[1], context, expected_kwargs),
    ]
