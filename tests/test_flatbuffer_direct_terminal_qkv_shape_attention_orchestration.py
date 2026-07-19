from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_qkv_shape_attention_orchestration,
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
    / "terminal_qkv_shape_attention_orchestration.py"
)
OWNER = "run_terminal_qkv_shape_attention_cleanup"
CHILD_OWNERS = (
    "optimize_transpose_shape_extract_nhwc_to_nchw_chains",
    "run_qkv_attention_summary",
)
CURRENT_CHILD_OWNERS = (
    "_optimize_transpose_shape_extract_nhwc_to_nchw_chains",
    CHILD_OWNERS[1],
)
RESULT_TARGETS = (
    "_late_pre_qkv_shape_extract_stats",
    "_late_qkv_stats",
)
COMPOSITE_TARGET = "_terminal_qkv_shape_attention_results"
PREDECESSOR_TARGET = "_terminal_affine_slice_spp_results"
SUCCESSOR_TARGET = "_terminal_activation_bridge_results"
TERMINAL_LAYOUT_SHAPE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_layout_shape_orchestration.py"
)
TERMINAL_LAYOUT_SHAPE_OWNER = "run_terminal_layout_shape_cleanup"
SHAPE_SCHEMA = {"optimized_transpose_shape_extract_nhwc_to_nchw_chains": 0}
QKV_SCHEMA = {
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
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_terminal_qkv_shape_attention_current_boundary_and_schema(
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

    model_ir = ModelIR("terminal_qkv_shape_attention_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        terminal_qkv_shape_attention_orchestration.run_terminal_qkv_shape_attention_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        )
        == (SHAPE_SCHEMA, QKV_SCHEMA)
    )


def test_terminal_qkv_shared_context_and_independent_shape_route_are_fixed() -> None:
    lowerer = _lowerer()
    alias = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == "qkv_attention_context"
    )
    assert ast.unparse(alias.value) == "shared_model_ir_pass_context"
    shape_calls = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == CURRENT_CHILD_OWNERS[0]
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert shape_calls == []
    owner = _functions(OWNER_PATH)[OWNER]
    owner_shape_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHILD_OWNERS[0]
    ]
    assert len(owner_shape_calls) == 1
    assert [
        ast.unparse(argument) for argument in owner_shape_calls[0].args
    ] == ["context.model_ir"]
    terminal_layout_owner = _functions(TERMINAL_LAYOUT_SHAPE_OWNER_PATH)[
        TERMINAL_LAYOUT_SHAPE_OWNER
    ]
    terminal_layout_shape_calls = [
        node
        for node in ast.walk(terminal_layout_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHILD_OWNERS[0]
    ]
    assert len(terminal_layout_shape_calls) == 1
    assert [
        ast.unparse(argument)
        for argument in terminal_layout_shape_calls[0].args
    ] == ["context.model_ir"]
    assert terminal_layout_shape_calls[0].keywords == []


def test_terminal_qkv_shape_attention_has_one_context_owner() -> None:
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
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {
        "include_layout_transpose": "include_layout_transpose",
        "include_prefix": "False",
    }

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


def test_terminal_qkv_shape_attention_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("terminal_qkv_shape_attention_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = ({"shape": 1}, {"qkv": 2})
    observed: list[tuple[str, object, dict[str, bool]]] = []

    def _shape(active_model_ir: ModelIR) -> dict[str, int]:
        observed.append((CHILD_OWNERS[0], active_model_ir, {}))
        return expected_results[0]

    def _qkv(
        active_context: ModelIRPassContext,
        **kwargs: bool,
    ) -> dict[str, int]:
        observed.append((CHILD_OWNERS[1], active_context, kwargs))
        return expected_results[1]

    monkeypatch.setattr(
        terminal_qkv_shape_attention_orchestration,
        CHILD_OWNERS[0],
        _shape,
    )
    monkeypatch.setattr(
        terminal_qkv_shape_attention_orchestration,
        CHILD_OWNERS[1],
        _qkv,
    )

    actual = (
        terminal_qkv_shape_attention_orchestration.run_terminal_qkv_shape_attention_cleanup(
            context,
            include_layout_transpose=True,
        )
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], context.model_ir, {}),
        (
            CHILD_OWNERS[1],
            context,
            {"include_layout_transpose": True, "include_prefix": False},
        ),
    ]
