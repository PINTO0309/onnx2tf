from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_terminal_pre_argmax_stats",
    "_terminal_transpose_gather_channel_fanout_stats",
    "_terminal_softmax_transpose_stats",
    "_terminal_boundary_input_normalization_stats",
    "_terminal_boundary_input_channel_slice_stats",
    "_terminal_internal_channel_slice_stats",
    "_terminal_channel_slice_muladd_bridge_stats",
)
EXPECTED_PHASE_IDS = (
    "cleanup.terminal.pre_argmax",
    "cleanup.terminal.transpose_gather_channel_fanout",
    "cleanup.terminal.softmax_transpose",
    "cleanup.terminal.boundary_input_normalization",
    "cleanup.terminal.boundary_input_channel_slice",
    "cleanup.terminal.internal_channel_slice",
    "cleanup.terminal.channel_slice_muladd_bridge",
)
EXPECTED_OWNER_EXPRESSIONS = (
    (
        "_optimize_transpose_pre_argmax_nhwc_terminal_chains(model_ir, "
        "layout_state=session.layout_state)"
    ),
    (
        "run_transpose_gather_channel_fanout_cleanup(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    (
        "_optimize_terminal_softmax_transpose_after_nhwc_propagation(model_ir, "
        "layout_state=session.layout_state)"
    ),
    (
        "run_boundary_input_normalization_cleanup(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    (
        "_optimize_boundary_input_transpose_channel_slice_blocks(model_ir, "
        "layout_state=session.layout_state)"
    ),
    (
        "_optimize_internal_transpose_channel_slice_nhwc_propagation_chains("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains(model_ir, "
        "layout_state=session.layout_state)"
    ),
)
PREDECESSOR_PHASE_ID = "cleanup.terminal.conv_activation"
SUCCESSOR_PHASE_ID = "cleanup.terminal.boundary_stridedslice_qdq_concat"


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


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


def test_terminal_boundary_results_use_consecutive_phase_records() -> None:
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
    assert indices == list(range(indices[0], indices[0] + 7))
    assert _phase_id(lowerer.body[indices[0] - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(lowerer.body[indices[-1] + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_terminal_boundary_result_locals_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
