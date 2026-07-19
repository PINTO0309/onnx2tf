from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_optimize_transpose_instancenorm_prepost_nhwc_chains"
RESULT_TARGET = "_layout_pass_set_1_instancenorm_prepost_stats"
PREVIOUS_TARGET = "_layout_pass_set_1_final_attention_recovery_results"
FOLLOWING_OWNER = "run_squeeze_reshape_identity_cleanup"
STATS_KEY = "optimized_transpose_instancenorm_prepost_nhwc_chains"


def _functions() -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(LOWERER_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    call = statement.value if isinstance(statement.value, ast.Call) else None
    if (
        call is not None
        and isinstance(call.func, ast.Attribute)
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


def test_instancenorm_prepost_schema_dispatch_and_both_forms_are_explicit() -> None:
    functions = _functions()
    owner = functions[OWNER]
    owner_source = ast.get_source_segment(
        LOWERER_PATH.read_text(encoding="utf-8"), owner
    )
    assert owner_source is not None
    assert "max_total_rewrites = 32" in owner_source
    assert "while rewritten < max_total_rewrites:" in owner_source
    assert owner_source.count("max_rewrites=1") == 1
    owner_names = (
        "_optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains_pass",
        "_optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains_pass",
        "_optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains_pass",
        "_optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains_pass",
    )
    assert all(name in owner_source for name in owner_names)
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'optimized_transpose_instancenorm_prepost_nhwc_chains': "
        "int(rewritten)}"
    )

    lowerer = functions["lower_onnx_to_ir"]
    calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert len(calls) == 2
    assert [ast.unparse(argument) for argument in calls[0].args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
    } == {"layout_state": "session.layout_state"}
    assert [ast.unparse(argument) for argument in calls[1].args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {
        "graph_index": "normalization_graph_index",
        "layout_state": "session.layout_state",
    }

    convergence_loop = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "_"
        and ast.unparse(node.iter) == "range(2)"
        and any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == OWNER
            for candidate in ast.walk(node)
        )
    )
    consumed = next(
        statement
        for statement in convergence_loop.body
        if _single_target(statement) == "rewritten_instnorm"
    )
    assert ast.unparse(consumed.value) == (
        "int(_optimize_transpose_instancenorm_prepost_nhwc_chains(model_ir, "
        "graph_index=normalization_graph_index, "
        "layout_state=session.layout_state).get("
        "'optimized_transpose_instancenorm_prepost_nhwc_chains', 0))"
    )
    assert _single_target(convergence_loop.body[0]) == (
        "normalization_graph_index"
    )
    assert _single_target(convergence_loop.body[2]) == (
        "rewritten_instnorm_posttranspose_bias"
    )


def test_direct_instancenorm_prepost_result_is_retained_observation_only() -> None:
    lowerer = _functions()["lower_onnx_to_ir"]
    layout_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(_call_name(child) == OWNER for child in statement.body)
    )
    result_index = next(
        index
        for index, statement in enumerate(layout_guard.body)
        if _call_name(statement) == OWNER
    )
    result = layout_guard.body[result_index]
    assert ast.unparse(result) == (
        "session.record_phase_result("
        "'cleanup.layout_pass_set_1.instancenorm_prepost', "
        "_optimize_transpose_instancenorm_prepost_nhwc_chains(model_ir, "
        "layout_state=session.layout_state))"
    )
    call = _statement_call(result)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
    } == {"layout_state": "session.layout_state"}
    assert _single_target(layout_guard.body[result_index - 1]) == (
        PREVIOUS_TARGET
    )
    assert _call_name(layout_guard.body[result_index + 1]) == FOLLOWING_OWNER
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
