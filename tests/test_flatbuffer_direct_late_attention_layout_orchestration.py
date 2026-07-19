from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import late_attention_layout_orchestration
from onnx2tf.tflite_builder.passes.late_attention_layout_orchestration import (
    LATE_ATTENTION_LAYOUT_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_attention_layout_orchestration.py"
)
OWNER = "run_late_attention_layout_cleanup"
RESULT_TARGET = "_late_attention_layout_results"
PREDECESSOR_TARGET = "_late_channel_shuffle_gather_results"
SUCCESSOR_TARGET = "_late_window_layout_results"
OLD_RESULT_TARGETS = (
    "_late_attention_qkv_reshape_stats",
    "_late_attention_gather_cleanup_stats",
    "_late_gather_axis0_reshape_stats",
    "_late_attention_preproj_ranklift_stats",
)
PASS_IDS = (
    "optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat",
    "_optimize_attention_gather_transpose_reshape_cleanup_chains",
    "_optimize_gather_axis0_singleton_to_reshape_input_chains",
    "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains",
)
LOWERER_PASS_IDS = (
    "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains",
    "_optimize_attention_gather_transpose_reshape_cleanup_chains",
    "_optimize_gather_axis0_singleton_to_reshape_input_chains",
    "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains",
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


def test_late_attention_layout_cluster_uses_one_composite_result_outside_store() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_attention_layout_cleanup(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
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


def test_late_attention_layout_cluster_uses_one_composite_owner() -> None:
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
        if _single_target(statement) == RESULT_TARGET
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_attention_layout_cleanup(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
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


def test_late_attention_layout_owner_preserves_arguments_and_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_attention_layout_owner")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[str, object, object | None]] = []
    expected_results = (
        {"qkv_reshape": 1},
        {"attention_gather": 2},
        {"axis0_gather": 3},
        {"preproj_ranklift": 4},
    )

    def _layout_callback(name: str, result: dict[str, int]):
        def _run(
            candidate: ModelIR,
            *,
            layout_state: object,
        ) -> dict[str, int]:
            observed.append((name, candidate, layout_state))
            return dict(result)

        return _run

    def _model_callback(name: str, result: dict[str, int]):
        def _run(candidate: ModelIR) -> dict[str, int]:
            observed.append((name, candidate, None))
            return dict(result)

        return _run

    for index, (pass_id, result) in enumerate(
        zip(PASS_IDS, expected_results, strict=True)
    ):
        callback = (
            _layout_callback(pass_id, result)
            if index in {0, 2}
            else _model_callback(pass_id, result)
        )
        monkeypatch.setattr(
            late_attention_layout_orchestration,
            pass_id,
            callback,
        )

    assert late_attention_layout_orchestration.run_late_attention_layout_cleanup(
        context
    ) == expected_results
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert observed[0][2] is context.layout_state
    assert observed[1][2] is None
    assert observed[2][2] is context.layout_state
    assert observed[3][2] is None
    assert LATE_ATTENTION_LAYOUT_PASS_IDS == LOWERER_PASS_IDS
