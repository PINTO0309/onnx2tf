from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_layout_pass_set_1_initial_affine_chain_fold_stats",
    "_layout_pass_set_1_affine_prepost_stats",
    "_layout_pass_set_1_pre_unary_affine_fanout_stats",
    "_layout_pass_set_1_mean_affine_prepost_stats",
    "_layout_pass_set_1_post_binary_affine_chain_fold_stats",
)
EXPECTED_PHASE_IDS = (
    "cleanup.layout_pass_set_1.initial_affine_chain_fold",
    "cleanup.layout_pass_set_1.affine_prepost",
    "cleanup.layout_pass_set_1.pre_unary_affine_fanout",
    "cleanup.layout_pass_set_1.mean_affine_prepost",
    "cleanup.layout_pass_set_1.post_binary_affine_chain_fold",
)
EXPECTED_OWNER_EXPRESSIONS = (
    (
        "_optimize_fold_mul_add_mul_affine_chains(model_ir, "
        "layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir, "
        "layout_state=session.layout_state)"
    ),
    "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(model_ir)",
    "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains(model_ir)",
    (
        "_optimize_fold_mul_add_mul_affine_chains(model_ir, "
        "layout_state=session.layout_state)"
    ),
)
PREFIX_PREDECESSOR_TARGET = "_layout_pass_set_1_initial_attention_recovery_results"
PREFIX_SUCCESSOR_TARGET = "_layout_pass_set_1_mean_attention_gate_results"
POST_BINARY_PREDECESSOR_TARGET = (
    "_layout_pass_set_1_post_binary_attention_recovery_results"
)
POST_BINARY_SUCCESSOR_TARGET = (
    "_layout_pass_set_1_attention_quantized_safe_binary_results"
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


def _guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        node
        for node in lowerer.body
        if isinstance(node, ast.If)
        and any(
            _single_target(statement) in EXPECTED_RESULT_TARGETS
            or _phase_id(statement) in EXPECTED_PHASE_IDS
            for statement in node.body
        )
    )


def _assert_boundaries(guard: ast.If, indices: list[int]) -> None:
    assert indices[:4] == list(range(indices[0], indices[0] + 4))
    assert _single_target(guard.body[indices[0] - 1]) == PREFIX_PREDECESSOR_TARGET
    assert _single_target(guard.body[indices[3] + 1]) == PREFIX_SUCCESSOR_TARGET
    assert _single_target(guard.body[indices[4] - 1]) == (
        POST_BINARY_PREDECESSOR_TARGET
    )
    assert _single_target(guard.body[indices[4] + 1]) == POST_BINARY_SUCCESSOR_TARGET


def test_layout_pass_set_1_affine_results_use_guarded_phase_records() -> None:
    lowerer = _lowerer()
    guard = _guard(lowerer)
    records = [
        statement
        for statement in guard.body
        if _phase_id(statement) in EXPECTED_PHASE_IDS
    ]
    indices = [guard.body.index(statement) for statement in records]

    assert ast.unparse(guard.test) == "optimize_layout_transpose_chains"
    assert guard.orelse == []
    assert tuple(_phase_id(statement) for statement in records) == EXPECTED_PHASE_IDS
    assert tuple(
        ast.unparse(_statement_call(statement).args[1]) for statement in records
    ) == EXPECTED_OWNER_EXPRESSIONS
    _assert_boundaries(guard, indices)
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_layout_pass_set_1_affine_result_locals_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
