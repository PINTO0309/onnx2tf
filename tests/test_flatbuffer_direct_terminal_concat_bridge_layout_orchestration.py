from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_concat_bridge_layout_orchestration,
)
from onnx2tf.tflite_builder.passes.terminal_concat_bridge_layout_orchestration import (
    TERMINAL_CONCAT_BRIDGE_LAYOUT_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_concat_bridge_layout_orchestration.py"
)
OWNER = "run_terminal_concat_bridge_layout_cleanup"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_boundary_slice_concat_orchestration.py"
)
COMPOSITE_OWNER = "run_final_boundary_slice_concat_cleanup"
COMPOSITE_TARGET = "_late_final_shape_terminal_fanout_results"
RESULT_TARGET = "_terminal_concat_bridge_layout_results"
PREDECESSOR_TARGET = "_late_affine_optional_fanout_results"
SUCCESSOR_TARGET = "_terminal_convpool_output_passthrough_stats"
OLD_RESULT_TARGETS = (
    "_terminal_relu_split_all_outputs_stats",
    "_terminal_relu_split_conv_concat_stats",
    "_terminal_split_mixed_pre_concat_stats",
    "_terminal_concat_input_adapter_stats",
    "_terminal_concat_unary_conv_stats",
    "_terminal_shape_extract_stats",
)
PASS_IDS = (
    "optimize_transpose_relu_split_all_outputs_to_nhwc_chains",
    "optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains",
    "optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
    "optimize_transpose_input_chains_pre_concat_to_single_post_adapter",
    "run_concat_unary_conv_layout_cleanup",
    "optimize_transpose_shape_extract_nhwc_to_nchw_chains",
)
LOWERER_PASS_IDS = (
    "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains",
    "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains",
    "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
    "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter",
    "run_concat_unary_conv_layout_cleanup",
    "_optimize_transpose_shape_extract_nhwc_to_nchw_chains",
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


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _composite_calls() -> list[ast.Call]:
    owner = _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]


def test_terminal_concat_bridge_cluster_uses_composite_result_outside_store() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_final_shape_terminal_fanout_cleanup("
        "late_final_shape_boundary_context, "
        "include_elementwise_fanout=optimize_layout_transpose_chains)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert _single_target(predecessor) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == "optimize_layout_transpose_chains"
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_phase_result"
        and any(
            isinstance(child, ast.Name) and child.id == OWNER
            for child in ast.walk(node)
        )
        for node in ast.walk(lowerer)
    )


def test_terminal_concat_bridge_cluster_uses_one_composite_owner() -> None:
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PASS_IDS
    ]
    assert owner_calls == list(PASS_IDS)

    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_final_shape_terminal_fanout_cleanup("
        "late_final_shape_boundary_context, "
        "include_elementwise_fanout=optimize_layout_transpose_chains)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert _single_target(predecessor) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == "optimize_layout_transpose_chains"
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_terminal_concat_bridge_owner_preserves_argument_and_result_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("terminal_concat_bridge_layout_owner")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[str, object, object | None, object | None]] = []
    expected_results = tuple(
        {f"result_{index}": index}
        for index in range(1, len(PASS_IDS) + 1)
    )

    def _layout_callback(name: str, result: dict[str, int]):
        def _run(
            candidate: ModelIR,
            *,
            layout_state: object,
        ) -> dict[str, int]:
            observed.append((name, candidate, layout_state, None))
            return dict(result)

        return _run

    def _diagnostic_callback(
        candidate: ModelIR,
        *,
        layout_state: object,
        diagnostics: object,
    ) -> dict[str, int]:
        observed.append((PASS_IDS[4], candidate, layout_state, diagnostics))
        return dict(expected_results[4])

    def _model_callback(candidate: ModelIR) -> dict[str, int]:
        observed.append((PASS_IDS[5], candidate, None, None))
        return dict(expected_results[5])

    for pass_id, result in zip(PASS_IDS[:4], expected_results[:4], strict=True):
        monkeypatch.setattr(
            terminal_concat_bridge_layout_orchestration,
            pass_id,
            _layout_callback(pass_id, result),
        )
    monkeypatch.setattr(
        terminal_concat_bridge_layout_orchestration,
        PASS_IDS[4],
        _diagnostic_callback,
    )
    monkeypatch.setattr(
        terminal_concat_bridge_layout_orchestration,
        PASS_IDS[5],
        _model_callback,
    )

    assert (
        terminal_concat_bridge_layout_orchestration.run_terminal_concat_bridge_layout_cleanup(
            context
        )
        == expected_results
    )
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert all(entry[2] is context.layout_state for entry in observed[:5])
    assert all(entry[3] is None for entry in observed[:4])
    assert observed[4][3] is context.diagnostics
    assert observed[5][2:] == (None, None)
    assert TERMINAL_CONCAT_BRIDGE_LAYOUT_PASS_IDS == LOWERER_PASS_IDS
