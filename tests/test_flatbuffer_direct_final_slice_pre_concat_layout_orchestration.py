from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    final_slice_pre_concat_layout_orchestration,
)
from onnx2tf.tflite_builder.passes.final_slice_pre_concat_layout_orchestration import (
    FINAL_SLICE_PRE_CONCAT_LAYOUT_PASS_IDS,
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
    / "final_slice_pre_concat_layout_orchestration.py"
)
OWNER = "run_final_slice_pre_concat_layout_cleanup"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_boundary_slice_concat_orchestration.py"
)
COMPOSITE_OWNER = "run_final_boundary_slice_concat_cleanup"
COMPOSITE_TARGET = "_final_boundary_slice_concat_results"
RESULT_TARGET = "_final_slice_pre_concat_layout_results"
PREDECESSOR_TARGET = "_late_final_shape_activation_convergence_stats"
SUCCESSOR_TARGET = "_terminal_elementwise_fanout_stats"
OLD_RESULT_TARGETS = (
    "_final_slice_prepost_passthrough_stats",
    "_final_pre_concat_stats",
)
PASS_IDS = (
    "optimize_transpose_slice_prepost_nhwc_passthrough_chains",
    "optimize_transpose_pre_concat_nhwc_chains",
)
LOWERER_PASS_IDS = (
    "_optimize_transpose_slice_prepost_nhwc_passthrough_chains",
    "_optimize_transpose_pre_concat_nhwc_chains",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
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


def test_final_slice_pre_concat_pair_uses_composite_result_outside_store() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_final_boundary_slice_concat_cleanup("
        "terminal_slice_concat_recovery_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
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


def test_final_slice_pre_concat_pair_uses_one_composite_owner() -> None:
    assert OWNER_PATH.exists()
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
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_final_boundary_slice_concat_cleanup("
        "terminal_slice_concat_recovery_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_final_slice_pre_concat_owner_preserves_argument_and_result_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("final_slice_pre_concat_layout_owner")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[str, object, object | None, object | None]] = []
    expected_results = (
        {"optimized_slice_prepost": 1},
        {"optimized_pre_concat": 2},
    )

    def _model_callback(candidate: ModelIR) -> dict[str, int]:
        observed.append((PASS_IDS[0], candidate, None, None))
        return dict(expected_results[0])

    def _layout_callback(
        candidate: ModelIR,
        *,
        layout_state: object,
        diagnostics: object,
    ) -> dict[str, int]:
        observed.append((PASS_IDS[1], candidate, layout_state, diagnostics))
        return dict(expected_results[1])

    monkeypatch.setattr(
        final_slice_pre_concat_layout_orchestration,
        PASS_IDS[0],
        _model_callback,
    )
    monkeypatch.setattr(
        final_slice_pre_concat_layout_orchestration,
        PASS_IDS[1],
        _layout_callback,
    )

    assert (
        final_slice_pre_concat_layout_orchestration.run_final_slice_pre_concat_layout_cleanup(
            context
        )
        == expected_results
    )
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert observed[0][2:] == (None, None)
    assert observed[1][2] is context.layout_state
    assert observed[1][3] is context.diagnostics
    assert FINAL_SLICE_PRE_CONCAT_LAYOUT_PASS_IDS == LOWERER_PASS_IDS
