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
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_reshape_shuffle_attention_window_orchestration.py"
)
COMPOSITE_OWNER = "run_late_reshape_shuffle_attention_window_cleanup"
COMPOSITE_TARGET = "_late_affine_final_shape_terminal_results"
RESULT_TARGET = "_late_attention_layout_results"
PREDECESSOR_PHASE_ID = "cleanup.late.ndhwc_cost_volume"
SUCCESSOR_TARGET = "_terminal_convpool_output_passthrough_stats"
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


def _composite_calls() -> list[ast.Call]:
    owner = _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]


def test_late_attention_layout_cluster_uses_one_composite_result_outside_store() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_affine_final_shape_terminal_cleanup("
        "late_final_shape_boundary_context, "
        "include_elementwise_fanout=optimize_layout_transpose_chains)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.Expr)
    assert ast.literal_eval(predecessor.value.args[0]) == PREDECESSOR_PHASE_ID
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
        if _single_target(statement) == COMPOSITE_TARGET
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_affine_final_shape_terminal_cleanup("
        "late_final_shape_boundary_context, "
        "include_elementwise_fanout=optimize_layout_transpose_chains)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.Expr)
    assert ast.literal_eval(predecessor.value.args[0]) == PREDECESSOR_PHASE_ID
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
