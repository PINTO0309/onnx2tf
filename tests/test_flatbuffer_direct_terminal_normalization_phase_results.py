from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_terminal_instancenorm_post_bias_stats",
    "_terminal_normalization_pad_stats",
    "_terminal_instancenorm_residual_add_stats",
    "_terminal_instancenorm_residual_mul_concat_stats",
    "_terminal_instancenorm_dualstats_stats",
)
EXPECTED_PHASE_IDS = (
    "cleanup.terminal.instancenorm_post_bias",
    "cleanup.terminal.normalization_pad",
    "cleanup.terminal.instancenorm_residual_add",
    "cleanup.terminal.instancenorm_residual_mul_concat",
    "cleanup.terminal.instancenorm_dualstats",
)
EXPECTED_OWNER_EXPRESSIONS = (
    (
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "run_normalization_pad_layout_cleanup(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    (
        "_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_"
        "nhwc_chains(model_ir, layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_"
        "nhwc_chains(model_ir, layout_state=session.layout_state)"
    ),
)
PREDECESSOR_PHASE_ID = "cleanup.terminal.swish_qdq_island"
SUCCESSOR_TARGET = "_terminal_boundary_mean_attention_results"


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


def test_terminal_normalization_results_use_phase_result_store() -> None:
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
    assert indices == list(range(indices[0], indices[0] + 5))
    assert _phase_id(lowerer.body[indices[0] - 1]) == PREDECESSOR_PHASE_ID
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
