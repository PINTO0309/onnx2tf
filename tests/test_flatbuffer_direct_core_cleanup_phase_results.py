from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_core_cleanup_pseudo_leakyrelu_stats",
    "_core_cleanup_yolo_decode_stats",
    "_core_cleanup_consecutive_mul_stats",
    "_core_cleanup_terminal_dequant_stats",
    "_core_cleanup_terminal_qdq_stats",
    "_core_cleanup_conv_affine_stats",
    "_core_cleanup_conv_activation_stats",
    "_core_cleanup_squeeze_reshape_identity_stats",
    "_core_cleanup_prune_reconcile_stats",
)
EXPECTED_PHASE_IDS = (
    "cleanup.core.pseudo_leakyrelu",
    "cleanup.core.yolo_decode",
    "cleanup.core.consecutive_mul",
    "cleanup.core.terminal_dequant",
    "cleanup.core.terminal_qdq",
    "cleanup.core.conv_affine",
    "cleanup.core.conv_activation",
    "cleanup.core.squeeze_reshape_identity",
    "cleanup.core.prune_reconcile",
)
EXPECTED_OWNER_EXPRESSIONS = (
    "_optimize_fuse_pseudo_leakyrelu_chains(model_ir)",
    "_optimize_yolo_decode_mul_square_anchor_chains(model_ir)",
    (
        "run_consecutive_mul_constants_cleanup(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    "_sanitize_terminal_transpose_before_dequantize(model_ir)",
    (
        "run_terminal_quantize_dequantize_cleanup(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    (
        "_optimize_fold_conv_mul_add_affine_chains(model_ir, "
        "enable_conv_add_only_fold=True, layout_state=session.layout_state)"
    ),
    (
        "_optimize_fuse_conv_activation_chains(model_ir, "
        "layout_state=session.layout_state)"
    ),
    (
        "run_squeeze_reshape_identity_cleanup(model_ir, "
        "include_unary_passthrough=True, layout_state=session.layout_state, "
        "diagnostics=session.diagnostics)"
    ),
    (
        "run_indexed_prune_reconcile_cleanup(model_ir, "
        "layout_state=session.layout_state)"
    ),
)


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


def _core_records(lowerer: ast.FunctionDef) -> list[ast.Expr]:
    return [
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Expr)
        and _phase_id(statement) in EXPECTED_PHASE_IDS
    ]


def test_core_cleanup_results_are_unconditional_and_unconsumed() -> None:
    lowerer = _lowerer()
    records = _core_records(lowerer)
    indices = [lowerer.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == EXPECTED_PHASE_IDS
    assert tuple(
        ast.unparse(_statement_call(statement).args[1]) for statement in records
    ) == EXPECTED_OWNER_EXPRESSIONS
    assert ast.unparse(lowerer.body[indices[0] - 1]) == (
        "_set_post_progress_desc('core cleanup passes')"
    )
    assert ast.unparse(lowerer.body[indices[-1] + 1]) == "_advance_post_progress()"

    dynamic_record_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _phase_id(statement) == "shape_resolution.core.dynamic_reshape"
    )
    assert indices[6] < dynamic_record_index < indices[7]
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


def test_core_cleanup_result_locals_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
