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
    / "sinet_sa_pa_mirrorpad_layout.py"
)
ATTENTION_RECOVERY_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "attention_recovery_orchestration.py"
)
SA_PA_MIRRORPAD = (
    "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"
)
OWNER_NAME = "optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"


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


def test_sa_pa_mirrorpad_result_schema_and_positive_cleanup_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[SA_PA_MIRRORPAD]
    wrapper_return = next(
        statement
        for statement in lowerer_wrapper.body
        if isinstance(statement, ast.Return)
    )
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{SA_PA_MIRRORPAD}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir",
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in wrapper_return.value.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
        "max_rewrites": "max_rewrites",
        "candidate": "candidate",
    }

    owner_tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    stats_key = next(
        statement
        for statement in owner_tree.body
        if isinstance(statement, ast.Assign)
        and _single_target(statement) == "_STATS_KEY"
    )
    assert isinstance(stats_key.value, ast.Constant)
    assert stats_key.value.value == (
        "optimized_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"
    )

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    cleanup_guards = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(cleanup_guards) == 1
    cleanup_guard = cleanup_guards[0]
    assert ast.unparse(cleanup_guard.test) == "rewritten > 0"
    assert [
        node.func.id
        for node in ast.walk(cleanup_guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ] == ["_prune_unused_tensors"]
    assert [
        node.func.attr
        for node in ast.walk(cleanup_guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ] == ["sync_from_model_ir"]
    assert not any(
        isinstance(statement, ast.Expr)
        and _call_name(statement) == "_prune_unused_tensors"
        for statement in owner.body
    )

    orchestrator = _functions(ATTENTION_RECOVERY_PATH)[
        "build_attention_gate_qdq_invocations"
    ]
    selected_owner_names = [
        node.id
        for node in ast.walk(orchestrator)
        if isinstance(node, ast.Name) and node.id == OWNER_NAME
    ]
    assert selected_owner_names == [OWNER_NAME]


def test_lowerer_records_both_direct_sa_pa_mirrorpad_results() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_results = sorted(
        (
            statement
            for statement in ast.walk(lowerer)
            if isinstance(statement, (ast.Assign, ast.Expr))
            and _call_name(statement) == SA_PA_MIRRORPAD
        ),
        key=lambda statement: statement.lineno,
    )
    assert len(direct_results) == 2
    assert [_phase_id(statement) for statement in direct_results] == [
        "cleanup.layout_pass_set_2.sa_pa_mirrorpad",
        "cleanup.post_cleanup.sa_pa_mirrorpad",
    ]
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_layout_opt_sa_pa_mirrorpad_stats"
        for node in ast.walk(lowerer)
    )
    for statement in direct_results:
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}

    layout_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and direct_results[0] in statement.body
    )
    assert ast.unparse(layout_guard.test) == "optimize_layout_transpose_chains"
    first_index = layout_guard.body.index(direct_results[0])
    assert _call_name(layout_guard.body[first_index - 1]) == (
        "run_layout_pass_set_2_channel_preadd_recovery"
    )
    assert _call_name(layout_guard.body[first_index + 1]) == (
        "_run_gate_layout_pass_cluster"
    )

    second_index = lowerer.body.index(direct_results[1])
    previous = lowerer.body[second_index - 1]
    following = lowerer.body[second_index + 1]
    assert _phase_id(previous) == "cleanup.post_cleanup.csp_attention"
    assert ast.unparse(previous.value.args[1]) == (
        "run_post_cleanup_sinet_csp_attention_cleanup("
        "shared_model_ir_pass_context)[1]"
    )
    assert _phase_id(following) == (
        "cleanup.post_sinet.batchmatmul_affine_input"
    )
    assert _call_name(following) == (
        "_optimize_batchmatmul_affine_transpose_input_chains"
    )
