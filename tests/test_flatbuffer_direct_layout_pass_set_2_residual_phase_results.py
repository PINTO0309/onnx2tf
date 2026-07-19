from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_layout_opt_elementwise_concat_conv_stats",
    "_layout_opt_spp_stats",
    "_layout_opt_pre_concat_stats",
    "_layout_opt_ndhwc_concat_stats",
    "_layout_opt_stridedslice_pre_concat_stats",
    "_layout_opt_split_mixed_pre_concat_stats",
    "_layout_opt_concat_input_adapter_stats",
    "_layout_opt_slice_logistic_concat_tail_stats",
    "_layout_opt_sa_pa_mirrorpad_stats",
)
EXPECTED_PHASE_IDS = (
    "cleanup.layout_pass_set_2.elementwise_concat_conv",
    "cleanup.layout_pass_set_2.spp",
    "cleanup.layout_pass_set_2.pre_concat",
    "cleanup.layout_pass_set_2.ndhwc_concat",
    "cleanup.layout_pass_set_2.stridedslice_pre_concat",
    "cleanup.layout_pass_set_2.split_mixed_pre_concat",
    "cleanup.layout_pass_set_2.concat_input_adapter",
    "cleanup.layout_pass_set_2.slice_logistic_concat_tail",
    "cleanup.layout_pass_set_2.sa_pa_mirrorpad",
)
EXPECTED_OWNER_EXPRESSIONS = (
    (
        "_optimize_transpose_elementwise_concat_conv_nhwc_groups(model_ir, "
        "layout_state=session.layout_state)"
    ),
    (
        "run_spp_layout_cleanup(model_ir, layout_state=session.layout_state, "
        "diagnostics=session.diagnostics)"
    ),
    (
        "_optimize_transpose_pre_concat_nhwc_chains(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    (
        "run_ndhwc_concat_layout_cleanup(model_ir, "
        "layout_state=session.layout_state, diagnostics=session.diagnostics)"
    ),
    (
        "_optimize_transpose_stridedslice_pre_concat_nhwc_chains(model_ir, "
        "layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(model_ir, "
        "layout_state=session.layout_state)"
    ),
)
PREFIX_PREDECESSOR_TARGET = (
    "_layout_pass_set_2_quantized_activation_binary_results"
)
PREFIX_SUCCESSOR_TARGET = "_layout_opt_channel_shuffle_gather_results"
SA_PA_PREDECESSOR_TARGET = "_layout_opt_preadd_mean_attention_results"
SA_PA_SUCCESSOR_TARGET = "_layout_opt_gate_layout_results"


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
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(
            _single_target(child) in EXPECTED_RESULT_TARGETS
            or _phase_id(child) in EXPECTED_PHASE_IDS
            for child in statement.body
        )
    )


def _assert_boundaries(guard: ast.If, indices: list[int]) -> None:
    assert indices[:8] == list(range(indices[0], indices[0] + 8))
    assert _single_target(guard.body[indices[0] - 1]) == PREFIX_PREDECESSOR_TARGET
    assert _single_target(guard.body[indices[7] + 1]) == PREFIX_SUCCESSOR_TARGET
    assert _single_target(guard.body[indices[8] - 1]) == SA_PA_PREDECESSOR_TARGET
    assert _single_target(guard.body[indices[8] + 1]) == SA_PA_SUCCESSOR_TARGET


def test_layout_pass_set_2_residual_results_are_guarded_and_unconsumed() -> None:
    lowerer = _lowerer()
    guard = _guard(lowerer)
    assignments = [
        statement
        for statement in guard.body
        if _single_target(statement) in EXPECTED_RESULT_TARGETS
    ]
    indices = [guard.body.index(statement) for statement in assignments]

    assert guard.orelse == []
    assert tuple(_single_target(statement) for statement in assignments) == (
        EXPECTED_RESULT_TARGETS
    )
    assert tuple(ast.unparse(statement.value) for statement in assignments) == (
        EXPECTED_OWNER_EXPRESSIONS
    )
    _assert_boundaries(guard, indices)
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


@pytest.mark.xfail(
    strict=True,
    reason="layout pass-set 2 residual results have not moved to phase records",
)
def test_layout_pass_set_2_residual_results_use_phase_result_store() -> None:
    lowerer = _lowerer()
    guard = _guard(lowerer)
    records = [
        statement
        for statement in guard.body
        if _phase_id(statement) in EXPECTED_PHASE_IDS
    ]
    indices = [guard.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == EXPECTED_PHASE_IDS
    assert tuple(
        ast.unparse(_statement_call(statement).args[1]) for statement in records
    ) == EXPECTED_OWNER_EXPRESSIONS
    _assert_boundaries(guard, indices)
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
