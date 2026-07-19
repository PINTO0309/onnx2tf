from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_conv1d_decoder_layout_orchestration.py"
)
OWNER = "run_late_conv1d_decoder_layout_cleanup"
RESULT_TARGET = "_late_conv1d_decoder_layout_results"
PREDECESSOR_TARGET = "_late_swish_transpose_passthrough_stats"
SUCCESSOR_TARGET = "_very_late_pad_layout_stats"
OLD_RESULT_TARGETS = (
    "_late_conv1d_squeeze_unary_stats",
    "_late_conv1d_rank4_unary_stats",
    "_late_conv1d_unary_fanout_stats",
    "_late_conv1d_instancenorm_unary_stats",
    "_late_conv1d_tencoder_stats",
    "_late_conv1d_batchmatmul_stats",
    "_late_decoder_deconv_stats",
    "_late_terminal_squeeze_mean_stats",
)
PASS_IDS = (
    "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains",
    "_optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains",
    "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains",
    "_optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains",
    "_optimize_tencoder_add_expand_transpose_conv_nhwc_chains",
    "_optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains",
    "_optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input",
    "_optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains",
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


def test_late_conv1d_decoder_cluster_is_ordered_and_unconsumed() -> None:
    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) in OLD_RESULT_TARGETS
    ]
    assert tuple(_single_target(statement) for statement in assignments) == (
        OLD_RESULT_TARGETS
    )
    indices = [lowerer.body.index(statement) for statement in assignments]
    assert indices == list(range(indices[0], indices[0] + len(indices)))
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET
    assert tuple(_call_name(statement) for statement in assignments) == PASS_IDS

    for statement in assignments:
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == [
            "model_ir"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}
    for target in OLD_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


@pytest.mark.xfail(
    strict=True,
    reason="late Conv1D/decoder cluster has not moved to one composite owner",
)
def test_late_conv1d_decoder_cluster_uses_one_composite_owner() -> None:
    assert OWNER_PATH.exists()
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
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_conv1d_decoder_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
