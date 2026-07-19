from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
LAYOUT_RESULT_TARGET = "_layout_pass_set_1_layout_transpose_cleanup_stats"
BINARY_RESULT_TARGET = "_layout_pass_set_1_transpose_binary_bridge_stats"
DUPLICATE_RESULT_TARGET = "_layout_pass_set_1_duplicate_fanout_stats"
DEQUANT_MEAN_RESULT_TARGET = "_layout_pass_set_1_dequant_mean_quantize_stats"
EXPECTED_RESULT_TARGETS = (
    LAYOUT_RESULT_TARGET,
    BINARY_RESULT_TARGET,
    DUPLICATE_RESULT_TARGET,
    DEQUANT_MEAN_RESULT_TARGET,
)
LAYOUT_PHASE_ID = "cleanup.layout_pass_set_1.layout_transpose"
BINARY_PHASE_ID = "cleanup.layout_pass_set_1.transpose_binary_bridge"
DUPLICATE_PHASE_ID = "cleanup.layout_pass_set_1.duplicate_fanout"
DEQUANT_MEAN_PHASE_ID = "cleanup.layout_pass_set_1.dequant_mean_quantize"
EXPECTED_PHASE_IDS = (
    LAYOUT_PHASE_ID,
    BINARY_PHASE_ID,
    DUPLICATE_PHASE_ID,
    DEQUANT_MEAN_PHASE_ID,
)
EXPECTED_OWNER_EXPRESSIONS = (
    (
        "run_layout_transpose_cleanup(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    (
        "_optimize_transpose_binary_bridges(model_ir, "
        "layout_state=session.layout_state)"
    ),
    (
        "run_duplicate_fanout_cleanup(model_ir, "
        "include_transpose=enable_duplicate_transpose_fanout_optimizations, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    "_optimize_transpose_dequantize_mean_quantize_bridges(model_ir)",
)


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


def _layout_guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(
            _single_target(child) == LAYOUT_RESULT_TARGET
            or _phase_id(child) == LAYOUT_PHASE_ID
            for child in statement.body
        )
    )


def _binary_guard(layout_guard: ast.If) -> tuple[int, ast.If]:
    index, guard = next(
        (index, statement)
        for index, statement in enumerate(layout_guard.body)
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test)
        == "enable_transpose_binary_bridge_optimizations"
    )
    return index, guard


def _assert_outer_boundaries(
    layout_guard: ast.If,
    layout_index: int,
    binary_guard_index: int,
    duplicate_index: int,
    dequant_mean_index: int,
) -> None:
    assert _single_target(layout_guard.body[layout_index - 1]) == (
        "enable_duplicate_transpose_fanout_optimizations"
    )
    assert _single_target(layout_guard.body[layout_index + 1]) == (
        "_layout_pass_set_1_initial_attention_recovery_results"
    )
    assert duplicate_index == binary_guard_index + 1
    assert _single_target(layout_guard.body[binary_guard_index - 1]) == (
        "_layout_pass_set_1_quantized_activation_binary_results"
    )
    assert _single_target(layout_guard.body[duplicate_index + 1]) == (
        "_layout_pass_set_1_post_binary_attention_recovery_results"
    )
    assert _single_target(layout_guard.body[dequant_mean_index - 1]) == (
        "_layout_pass_set_1_attention_quantized_safe_binary_results"
    )
    assert _single_target(layout_guard.body[dequant_mean_index + 1]) == (
        "_layout_pass_set_1_qlinear_mean_concat_results"
    )


def test_layout_pass_set_1_residual_results_use_guarded_phase_records() -> None:
    lowerer = _lowerer()
    layout_guard = _layout_guard(lowerer)
    binary_guard_index, binary_guard = _binary_guard(layout_guard)
    outer_phase_ids = (
        LAYOUT_PHASE_ID,
        DUPLICATE_PHASE_ID,
        DEQUANT_MEAN_PHASE_ID,
    )
    records = [
        statement
        for statement in layout_guard.body
        if _phase_id(statement) in outer_phase_ids
    ]
    indices = [layout_guard.body.index(statement) for statement in records]

    assert layout_guard.orelse == []
    assert tuple(_phase_id(statement) for statement in records) == outer_phase_ids
    assert tuple(
        ast.unparse(_statement_call(statement).args[1]) for statement in records
    ) == (
        EXPECTED_OWNER_EXPRESSIONS[0],
        EXPECTED_OWNER_EXPRESSIONS[2],
        EXPECTED_OWNER_EXPRESSIONS[3],
    )
    assert binary_guard.orelse == []
    assert len(binary_guard.body) == 1
    assert _phase_id(binary_guard.body[0]) == BINARY_PHASE_ID
    assert (
        ast.unparse(_statement_call(binary_guard.body[0]).args[1])
        == EXPECTED_OWNER_EXPRESSIONS[1]
    )
    _assert_outer_boundaries(
        layout_guard,
        indices[0],
        binary_guard_index,
        indices[1],
        indices[2],
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_layout_pass_set_1_residual_result_locals_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
