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
    / "attention_layout.py"
)
CSP_ATTENTION = "_optimize_transpose_csp_attention_nhwc_chains"
SA_PA_MIRRORPAD = (
    "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"
)


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
    return statement.value


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


def test_csp_attention_result_schema_and_cleanup_semantics_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[CSP_ATTENTION]
    assert len(lowerer_wrapper.body) == 1
    wrapper_return = lowerer_wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{CSP_ATTENTION}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir",
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in wrapper_return.value.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
    }

    owner = _functions(OWNER_PATH)[CSP_ATTENTION]
    owner_return = next(
        statement
        for statement in reversed(owner.body)
        if isinstance(statement, ast.Return)
    )
    assert isinstance(owner_return.value, ast.Dict)
    assert [
        key.value
        for key in owner_return.value.keys
        if isinstance(key, ast.Constant)
    ] == ["optimized_transpose_csp_attention_nhwc_chains"]

    prune_index = next(
        index
        for index, statement in enumerate(owner.body)
        if _call_name(statement) == "_prune_unused_tensors"
    )
    sync_guard_index = next(
        index
        for index, statement in enumerate(owner.body)
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sync_from_model_ir"
            for node in ast.walk(statement)
        )
    )
    return_index = owner.body.index(owner_return)
    assert prune_index < sync_guard_index < return_index
    sync_guard = owner.body[sync_guard_index]
    assert isinstance(sync_guard, ast.If)
    assert ast.unparse(sync_guard.test) == (
        "rewritten > 0 and layout_state is not None"
    )


def test_lowerer_retains_post_cleanup_csp_attention_result() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    invocations = [
        statement
        for statement in lowerer.body
        if _call_name(statement) == CSP_ATTENTION
    ]
    assert len(invocations) == 1
    invocation = invocations[0]
    assert _single_target(invocation) == "_post_cleanup_csp_attention_stats"

    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
    } == {"layout_state": "session.layout_state"}

    invocation_index = lowerer.body.index(invocation)
    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    assert _single_target(previous) == "_post_cleanup_sinet_preadd_resize_results"
    assert _call_name(previous) == "_run_sinet_preadd_resize_recovery_sequence"
    assert _call_name(following) == SA_PA_MIRRORPAD
    following_call = _statement_call(following)
    assert following_call is not None
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in following_call.keywords
    } == {"layout_state": "session.layout_state"}
