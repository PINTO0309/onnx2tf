from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "quantized_activation.py"
)
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "attention_recovery_orchestration.py"
)
DEQUANT_HARDSIGMOID = (
    "_optimize_transpose_dequant_hardsigmoid_quantize_bridges"
)
OWNER_NAME = "optimize_transpose_dequant_hardsigmoid_quantize_bridges"
HARDSWISH_SE = (
    "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
)
MIXED_ATTENTION = "run_mixed_attention_layout_cleanup"
LATE_DEQUANT_CLUSTER = "_run_late_dequant_unary_fanout_pass_cluster"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        return call.args[1]
    return call


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def test_dequant_hardsigmoid_schema_cleanup_and_selection_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[DEQUANT_HARDSIGMOID]
    assert len(lowerer_wrapper.body) == 1
    wrapper_return = lowerer_wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == OWNER_NAME
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir",
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in wrapper_return.value.keywords
    } == {"graph_index": "graph_index"}

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    unconditional_cleanup = [
        statement
        for statement in owner.body
        if _call_name(statement) == "_prune_unused_tensors"
    ]
    assert len(unconditional_cleanup) == 1
    active_index_guard = next(
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "active_index is None"
    )
    no_transpose_guard = next(
        statement
        for statement in active_index_guard.body
        if isinstance(statement, ast.If)
    )
    assert ast.unparse(no_transpose_guard.test) == (
        "not any((str(op.op_type) == 'TRANSPOSE' for op in model_ir.operators))"
    )
    early_return = no_transpose_guard.body[0]
    assert isinstance(early_return, ast.Return)
    assert ast.unparse(early_return.value) == (
        "{'removed_transpose_dequant_hardsigmoid_quantize_bridges': 0}"
    )
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'removed_transpose_dequant_hardsigmoid_quantize_bridges': int(removed_bridges)}"
    )

    orchestration = _functions(ORCHESTRATION_PATH)[
        "build_attention_gate_qdq_invocations"
    ]
    assert sum(
        1
        for node in ast.walk(orchestration)
        if isinstance(node, ast.Name) and node.id == OWNER_NAME
    ) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id == DEQUANT_HARDSIGMOID
        for node in ast.walk(orchestration)
    )


def test_lowerer_retains_all_dequant_hardsigmoid_results() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_results = [
        statement
        for statement in lowerer.body
        if _call_name(statement) == DEQUANT_HARDSIGMOID
    ]
    assert len(direct_results) == 3
    expected_targets = [
        "_post_sinet_dequant_hardsigmoid_bridge_stats",
        "_late_dequant_hardsigmoid_bridge_stats",
    ]
    assert _phase_id(direct_results[0]) == (
        "cleanup.terminal.dequant_hardsigmoid_bridge"
    )
    assert [_single_target(statement) for statement in direct_results[1:]] == (
        expected_targets
    )
    for statement in direct_results:
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert call.keywords == []
    for target in ["_terminal_dequant_hardsigmoid_bridge_stats", *expected_targets]:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )

    terminal_index = lowerer.body.index(direct_results[0])
    assert _call_name(lowerer.body[terminal_index - 1]) == HARDSWISH_SE
    assert _single_target(lowerer.body[terminal_index + 1]) == (
        "_terminal_sinet_preadd_resize_results"
    )

    post_sinet_index = lowerer.body.index(direct_results[1])
    assert _call_name(lowerer.body[post_sinet_index - 1]) == MIXED_ATTENTION
    assert _single_target(lowerer.body[post_sinet_index + 1]) == (
        "late_ndhwc_cost_volume_state_scope"
    )

    late_index = lowerer.body.index(direct_results[2])
    branch = lowerer.body[late_index - 1]
    assert isinstance(branch, ast.If)
    assert ast.unparse(branch.test) == "optimize_layout_transpose_chains"
    assert _single_target(branch.body[0]) == (
        "_terminal_convpool_output_passthrough_stats"
    )
    assert _call_name(lowerer.body[late_index + 1]) == LATE_DEQUANT_CLUSTER
