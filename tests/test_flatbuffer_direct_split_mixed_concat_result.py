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
    / "split_mixed_concat_layout.py"
)
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_recovery_orchestration.py"
)
SPLIT_MIXED_CONCAT = (
    "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains"
)
OWNER_NAME = (
    "optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains"
)
STRIDED_SLICE_PRE_CONCAT = (
    "_optimize_transpose_stridedslice_pre_concat_nhwc_chains"
)
INPUT_PRE_CONCAT = (
    "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter"
)
RELU_SPLIT_CONV_CONCAT = (
    "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
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


def test_split_mixed_concat_schema_cleanup_and_orchestration_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[SPLIT_MIXED_CONCAT]
    assert len(lowerer_wrapper.body) == 1
    wrapper_return = lowerer_wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{SPLIT_MIXED_CONCAT}_pass"
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
        "optimized_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains"
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
    assert not any(
        isinstance(statement, ast.Expr)
        and _call_name(statement) == "_prune_unused_tensors"
        for statement in owner.body
    )

    orchestration = _functions(ORCHESTRATION_PATH)[
        "build_layout_recovery_invocations"
    ]
    selected_owner_names = [
        node
        for node in ast.walk(orchestration)
        if isinstance(node, ast.Name) and node.id == OWNER_NAME
    ]
    assert len(selected_owner_names) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id == SPLIT_MIXED_CONCAT
        for node in ast.walk(orchestration)
    )


def test_lowerer_retains_both_split_mixed_concat_results() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    layout_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(_call_name(child) == SPLIT_MIXED_CONCAT for child in statement.body)
    )
    guarded_index = next(
        index
        for index, statement in enumerate(layout_guard.body)
        if _call_name(statement) == SPLIT_MIXED_CONCAT
    )
    guarded_result = layout_guard.body[guarded_index]

    terminal_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == SPLIT_MIXED_CONCAT
    )
    terminal_result = lowerer.body[terminal_index]
    direct_results = [guarded_result, terminal_result]
    expected_targets = [
        "_layout_opt_split_mixed_pre_concat_stats",
        "_terminal_split_mixed_pre_concat_stats",
    ]
    assert [_single_target(statement) for statement in direct_results] == (
        expected_targets
    )
    assert sum(
        1
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SPLIT_MIXED_CONCAT
    ) == 2
    for statement in direct_results:
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}
    for target in expected_targets:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )

    assert _call_name(layout_guard.body[guarded_index - 1]) == (
        STRIDED_SLICE_PRE_CONCAT
    )
    assert _call_name(layout_guard.body[guarded_index + 1]) == INPUT_PRE_CONCAT

    terminal_previous = lowerer.body[terminal_index - 1]
    assert _single_target(terminal_previous) == (
        "_terminal_relu_split_conv_concat_stats"
    )
    assert _call_name(terminal_previous) == RELU_SPLIT_CONV_CONCAT
    assert _call_name(lowerer.body[terminal_index + 1]) == INPUT_PRE_CONCAT
