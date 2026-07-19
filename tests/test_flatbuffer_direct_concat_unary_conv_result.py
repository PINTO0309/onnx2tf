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
    / "concat_unary_conv_layout.py"
)
FINAL_COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_boundary_slice_concat_orchestration.py"
)
FINAL_COMPOSITE_OWNER = "run_final_boundary_slice_concat_cleanup"
FINAL_COMPOSITE_TARGET = "_final_boundary_slice_concat_results"
RUNNER = "run_concat_unary_conv_layout_cleanup"
OWNER = "_optimize_transpose_concat_unary_fanout_conv_nhwc_chains"
CONCAT_INPUT_ADAPTER = (
    "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter"
)
SHAPE_EXTRACT = "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"


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


def test_concat_unary_conv_runner_schema_and_mutation_contract_are_explicit() -> None:
    functions = _functions(OWNER_PATH)
    owner = functions[OWNER]
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
    assert ast.unparse(cleanup_guard.test) == "optimized > 0"
    cleanup_calls = [
        ast.unparse(node.func)
        for node in ast.walk(cleanup_guard)
        if isinstance(node, ast.Call)
    ]
    assert cleanup_calls == [
        "_prune_unused_tensors",
        "layout_state.sync_from_model_ir",
    ]
    assert not any(
        isinstance(statement, ast.Expr)
        and _call_name(statement) == "_prune_unused_tensors"
        for statement in owner.body
    )

    runner = functions[RUNNER]
    stats_key = next(
        statement
        for statement in runner.body
        if isinstance(statement, ast.Assign)
        and _single_target(statement) == "stats_key"
    )
    assert isinstance(stats_key.value, ast.Constant)
    assert stats_key.value.value == (
        "optimized_transpose_concat_unary_fanout_conv_nhwc_chains"
    )
    pass_spec = next(
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PassSpec"
    )
    pass_spec_keywords = {
        keyword.arg: ast.unparse(keyword.value) for keyword in pass_spec.keywords
    }
    assert pass_spec_keywords == {
        "pass_id": "'layout.concat_unary_conv_nhwc'",
        "phase": "PassPhase.LAYOUT_PLAN",
        "callback": "_run",
        "precondition": "_has_concat_unary_conv_candidate",
        "priority": "10",
        "transactional": "True",
    }
    group_call = next(
        node
        for node in ast.walk(runner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_model_ir_pass_group"
    )
    group_keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in group_call.keywords
    }
    assert group_keywords["layout_state"] == "layout_state"
    assert group_keywords["default_details"] == "{stats_key: 0}"
    assert group_keywords["diagnostics"] == "diagnostics"
    assert group_keywords["state_scope"] == "state_scope"
    assert group_keywords["preflight"] == "_preflight"
    runner_return = runner.body[-1]
    assert isinstance(runner_return, ast.Return)
    assert ast.unparse(runner_return.value) == (
        "{str(key): int(value) for key, value in details.items()}"
    )


def test_lowerer_moves_terminal_concat_unary_conv_to_composite() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_results = [
        statement
        for statement in lowerer.body
        if _call_name(statement) == RUNNER
    ]
    assert direct_results == []
    target = "_terminal_concat_unary_conv_stats"
    assert not any(
        isinstance(node, ast.Name) and node.id == target
        for node in ast.walk(lowerer)
    )
    composite = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == FINAL_COMPOSITE_TARGET
    )
    assert _call_name(composite) == FINAL_COMPOSITE_OWNER
    owner = _functions(FINAL_COMPOSITE_PATH)[FINAL_COMPOSITE_OWNER]
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_terminal_concat_bridge_layout_cleanup"
        for node in ast.walk(owner)
    ) == 1
