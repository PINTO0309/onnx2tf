from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_activation_bridge_orchestration,
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
    / "terminal_activation_bridge_orchestration.py"
)
OWNER = "run_terminal_activation_bridge_cleanup"
CHILD_OWNERS = (
    "optimize_split_conv_concat_transpose_bridge_to_single_post_nchw",
    "run_hardswish_se_layout_summary",
    "run_late_hard_activation_layout_summary",
)
CURRENT_CHILD_OWNERS = (
    "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw",
    CHILD_OWNERS[1],
    CHILD_OWNERS[2],
)
RESULT_TARGETS = (
    "_terminal_split_conv_concat_bridge_stats",
    "_terminal_hardswish_se_stats",
    "_late_hard_activation_stats",
)
COMPOSITE_TARGET = "_terminal_activation_bridge_results"
PREDECESSOR_TARGET = "_terminal_qkv_shape_attention_results"
SUCCESSOR_TARGET = "_absolute_final_pre_concat_stats"
EXPECTED_SCHEMAS = (
    {"optimized_split_conv_concat_transpose_bridge_to_single_post_nchw": 0},
    {
        "optimized_transpose_hardswish_se_conv_hardsigmoid_mul_"
        "prepost_nhwc_chains": 0,
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


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_terminal_activation_bridge_current_boundary_and_schema(
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
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("terminal_activation_bridge_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        terminal_activation_bridge_orchestration.run_terminal_activation_bridge_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        )
        == EXPECTED_SCHEMAS
    )


def test_terminal_activation_shared_context_alias_is_fixed() -> None:
    lowerer = _lowerer()
    alias = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == "late_hard_activation_layout_context"
    )
    assert ast.unparse(alias.value) == "shared_model_ir_pass_context"


def test_terminal_activation_bridge_has_one_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.model_ir"
    ]
    assert calls[1].keywords == []
    assert [ast.unparse(argument) for argument in calls[2].args] == ["context"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[2].keywords
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
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_terminal_activation_bridge_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("terminal_activation_bridge_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = tuple(
        {f"stage_{index}": index} for index in range(len(CHILD_OWNERS))
    )
    observed: list[tuple[str, object, dict[str, object]]] = []

    def _split(
        active_model_ir: ModelIR,
        **kwargs: object,
    ) -> dict[str, int]:
        observed.append((CHILD_OWNERS[0], active_model_ir, kwargs))
        return expected_results[0]

    def _hardswish(active_model_ir: ModelIR) -> dict[str, int]:
        observed.append((CHILD_OWNERS[1], active_model_ir, {}))
        return expected_results[1]

    def _hard_activation(
        active_context: ModelIRPassContext,
        **kwargs: object,
    ) -> dict[str, int]:
        observed.append((CHILD_OWNERS[2], active_context, kwargs))
        return expected_results[2]

    monkeypatch.setattr(
        terminal_activation_bridge_orchestration,
        CHILD_OWNERS[0],
        _split,
    )
    monkeypatch.setattr(
        terminal_activation_bridge_orchestration,
        CHILD_OWNERS[1],
        _hardswish,
    )
    monkeypatch.setattr(
        terminal_activation_bridge_orchestration,
        CHILD_OWNERS[2],
        _hard_activation,
    )

    actual = (
        terminal_activation_bridge_orchestration.run_terminal_activation_bridge_cleanup(
            context,
            include_layout_transpose=True,
        )
    )
    assert actual == expected_results
    assert all(
        actual[index] is expected_results[index]
        for index in range(len(expected_results))
    )
    assert observed == [
        (
            CHILD_OWNERS[0],
            context.model_ir,
            {"layout_state": context.layout_state},
        ),
        (CHILD_OWNERS[1], context.model_ir, {}),
        (
            CHILD_OWNERS[2],
            context,
            {"include_layout_transpose": True},
        ),
    ]
