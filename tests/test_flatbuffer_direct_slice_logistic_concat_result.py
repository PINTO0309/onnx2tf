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
    / "slice_logistic_concat_reshape_tail_layout.py"
)
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_recovery_orchestration.py"
)
SLICE_LOGISTIC_CONCAT = (
    "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"
)
OWNER_NAME = (
    "optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"
)
CONCAT_INPUT_ADAPTER = (
    "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter"
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


def test_slice_logistic_concat_schema_cleanup_and_selection_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[SLICE_LOGISTIC_CONCAT]
    assert len(lowerer_wrapper.body) == 1
    wrapper_return = lowerer_wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{SLICE_LOGISTIC_CONCAT}_pass"
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
        "optimized_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"
    )

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    unconditional_cleanup = [
        statement
        for statement in owner.body
        if _call_name(statement) == "_prune_unused_tensors"
    ]
    assert len(unconditional_cleanup) == 1
    assert not any(
        isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
        for statement in owner.body
    )

    orchestration = _functions(ORCHESTRATION_PATH)[
        "build_layout_recovery_invocations"
    ]
    assert sum(
        1
        for node in ast.walk(orchestration)
        if isinstance(node, ast.Name) and node.id == OWNER_NAME
    ) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id == SLICE_LOGISTIC_CONCAT
        for node in ast.walk(orchestration)
    )


def test_lowerer_retains_slice_logistic_concat_result() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    layout_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            _call_name(child) == SLICE_LOGISTIC_CONCAT
            for child in statement.body
        )
    )
    result_index = next(
        index
        for index, statement in enumerate(layout_guard.body)
        if _call_name(statement) == SLICE_LOGISTIC_CONCAT
    )
    result = layout_guard.body[result_index]
    target = "_layout_opt_slice_logistic_concat_tail_stats"
    assert _single_target(result) is None
    assert sum(
        1
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SLICE_LOGISTIC_CONCAT
    ) == 1
    call = _statement_call(result)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
    } == {"layout_state": "session.layout_state"}
    assert not any(
        isinstance(node, ast.Name)
        and node.id == target
        for node in ast.walk(lowerer)
    )

    previous = layout_guard.body[result_index - 1]
    assert _single_target(previous) is None
    assert _call_name(previous) == CONCAT_INPUT_ADAPTER
    following = layout_guard.body[result_index + 1]
    assert _single_target(following) == (
        "_layout_pass_set_2_channel_preadd_results"
    )
    assert _call_name(following) == (
        "run_layout_pass_set_2_channel_preadd_recovery"
    )
    following_call = _statement_call(following)
    assert following_call is not None
    assert [ast.unparse(argument) for argument in following_call.args] == [
        "attention_recovery_context"
    ]
    assert following_call.keywords == []
