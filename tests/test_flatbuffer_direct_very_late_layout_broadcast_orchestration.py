from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    very_late_layout_broadcast_orchestration,
)
from onnx2tf.tflite_builder.passes.very_late_layout_broadcast_orchestration import (
    VERY_LATE_LAYOUT_BROADCAST_PASS_IDS,
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
    / "very_late_layout_broadcast_orchestration.py"
)
OWNER = "run_very_late_layout_broadcast_cleanup"
COMPOSITE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_layout_tail_orchestration.py"
)
COMPOSITE_OWNER = "run_very_late_layout_tail_cleanup"
LOWERER_OWNER = "run_late_dequant_swish_layout_tail_cleanup"
RESULT_TARGET = "_late_dequant_swish_layout_tail_results"
PREDECESSOR_GUARD = "optimize_layout_transpose_chains"
SUCCESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_broadcast"
OLD_RESULT_TARGETS = (
    "_very_late_layout_transpose_cleanup_stats",
    "_very_late_broadcast_repair_stats",
)
PASS_IDS = (
    "run_layout_transpose_cleanup",
    "repair_rank4_channelwise_broadcast_constants_to_runtime_layout",
)
LOWERER_PASS_IDS = (
    "run_layout_transpose_cleanup",
    "_repair_rank4_channelwise_broadcast_constants_to_runtime_layout",
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


def _phase_id(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
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


def test_very_late_layout_broadcast_boundary_uses_composite_outside_store() -> (
    None
):
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == LOWERER_OWNER
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_very_late_layout_broadcast_boundary_uses_one_composite_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in sorted(
            (
                node
                for node in ast.walk(owner)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in PASS_IDS
            ),
            key=lambda node: node.lineno,
        )
    ]
    assert owner_calls == list(PASS_IDS)

    composite_owner = _functions(COMPOSITE_OWNER_PATH)[COMPOSITE_OWNER]
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(composite_owner)
    ) == 1

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == LOWERER_OWNER
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_very_late_layout_broadcast_owner_preserves_guard_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
) -> None:
    model_ir = ModelIR("very_late_layout_broadcast_owner")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    layout_result = {"layout_result": 1}
    broadcast_result = {"broadcast_result": 2}
    observed: list[tuple[str, object, object | None, object | None]] = []

    def _layout_callback(
        candidate: ModelIR,
        *,
        layout_state: object,
        diagnostics: object,
    ) -> dict[str, int]:
        observed.append((PASS_IDS[0], candidate, layout_state, diagnostics))
        return dict(layout_result)

    def _broadcast_callback(candidate: ModelIR) -> dict[str, int]:
        observed.append((PASS_IDS[1], candidate, None, None))
        return dict(broadcast_result)

    monkeypatch.setattr(
        very_late_layout_broadcast_orchestration,
        PASS_IDS[0],
        _layout_callback,
    )
    monkeypatch.setattr(
        very_late_layout_broadcast_orchestration,
        PASS_IDS[1],
        _broadcast_callback,
    )

    expected_layout_result = (
        layout_result if include_layout_transpose else None
    )
    assert (
        very_late_layout_broadcast_orchestration.run_very_late_layout_broadcast_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        )
        == (expected_layout_result, broadcast_result)
    )
    expected_pass_ids = (
        list(PASS_IDS) if include_layout_transpose else [PASS_IDS[1]]
    )
    assert [entry[0] for entry in observed] == expected_pass_ids
    assert all(entry[1] is context.model_ir for entry in observed)
    if include_layout_transpose:
        assert observed[0][2] is context.layout_state
        assert observed[0][3] is context.diagnostics
    assert VERY_LATE_LAYOUT_BROADCAST_PASS_IDS == LOWERER_PASS_IDS
