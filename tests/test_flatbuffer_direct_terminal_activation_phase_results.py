from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_terminal_boundary_stridedslice_qdq_concat_stats",
    "_terminal_swish_residual_concat_closure_stats",
    "_terminal_dequant_logistic_mul_quantize_bridge_stats",
    "_terminal_swish_qdq_island_stats",
)
EXPECTED_PHASE_IDS = (
    "cleanup.terminal.boundary_stridedslice_qdq_concat",
    "cleanup.terminal.swish_residual_concat_closure",
    "cleanup.terminal.dequant_logistic_mul_quantize_bridge",
    "cleanup.terminal.swish_qdq_island",
)
EXPECTED_OWNER_EXPRESSIONS = (
    (
        "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks("
        "model_ir, layout_state=session.layout_state)"
    ),
    "_optimize_transpose_swish_residual_concat_closure_nhwc_chains(model_ir)",
    "_optimize_transpose_dequant_logistic_mul_quantize_bridges(model_ir)",
    "_optimize_transpose_swish_qdq_nhwc_islands(model_ir)",
)
PREDECESSOR_TARGET = "_terminal_slice_concat_recovery_results"
SUCCESSOR_TARGET = "_terminal_instancenorm_post_bias_stats"


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


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


def test_terminal_activation_results_are_consecutive_and_unconsumed() -> None:
    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) in EXPECTED_RESULT_TARGETS
    ]
    indices = [lowerer.body.index(statement) for statement in assignments]

    assert tuple(_single_target(statement) for statement in assignments) == (
        EXPECTED_RESULT_TARGETS
    )
    assert tuple(ast.unparse(statement.value) for statement in assignments) == (
        EXPECTED_OWNER_EXPRESSIONS
    )
    assert indices == list(range(indices[0], indices[0] + 4))
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


@pytest.mark.xfail(
    strict=True,
    reason="terminal activation results have not moved to phase records",
)
def test_terminal_activation_results_use_phase_result_store() -> None:
    lowerer = _lowerer()
    records = [
        statement
        for statement in lowerer.body
        if _phase_id(statement) in EXPECTED_PHASE_IDS
    ]
    indices = [lowerer.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == EXPECTED_PHASE_IDS
    assert tuple(
        ast.unparse(_statement_call(statement).args[1]) for statement in records
    ) == EXPECTED_OWNER_EXPRESSIONS
    assert indices == list(range(indices[0], indices[0] + 4))
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
