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
    / "binary_layout_adapter.py"
)
RAW_OWNER = "run_indexed_binary_layout_adapter_cleanup"
SUMMARY_OWNER = "run_indexed_binary_layout_adapter_summary"
COUNT_TARGET = "final_placeholder_binary_tensor_count"
RAW_TARGETS = (
    "final_placeholder_exact_binary_stats",
    "final_placeholder_singleton_binary_stats",
)
SUMMARY_TARGET = "final_placeholder_binary_stats"
PREDECESSOR_TARGET = "final_placeholder_reconcile_stats"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _tuple_targets(statement: ast.stmt) -> tuple[str, ...]:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return ()
    target = statement.targets[0]
    if not isinstance(target, ast.Tuple):
        return ()
    return tuple(
        element.id for element in target.elts if isinstance(element, ast.Name)
    )


def _containing_body(root: ast.AST, target: ast.stmt) -> list[ast.stmt]:
    for node in ast.walk(root):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and target in value:
                return value
    raise AssertionError("statement is not contained by an AST body")


def _assert_topology_successor(statement: ast.stmt) -> None:
    assert isinstance(statement, ast.Expr)
    assert ast.unparse(statement) == (
        "session.record_phase_result('topology.primary.final_placeholder', "
        "_topologically_sort_operators(model_ir))"
    )


def test_final_placeholder_binary_prune_aware_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    count = next(
        statement
        for statement in ast.walk(lowerer)
        if _single_target(statement) == COUNT_TARGET
    )
    body = _containing_body(lowerer, count)
    index = body.index(count)
    pair = body[index + 1]
    guard = body[index + 2]
    assert isinstance(count, ast.Assign)
    assert ast.unparse(count.value) == "len(model_ir.tensors)"
    assert _tuple_targets(pair) == RAW_TARGETS
    assert isinstance(pair, ast.Assign)
    assert ast.unparse(pair.value) == f"{RAW_OWNER}(model_ir)"
    assert _single_target(body[index - 1]) == PREDECESSOR_TARGET
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "_stats_have_positive_count(final_placeholder_reconcile_stats, "
        "final_placeholder_exact_binary_stats, "
        "final_placeholder_singleton_binary_stats) or "
        "len(model_ir.tensors) < final_placeholder_binary_tensor_count"
    )
    _assert_topology_successor(body[index + 3])


@pytest.mark.xfail(
    strict=True,
    reason="final placeholder binary site lacks one prune-aware summary owner",
)
def test_final_placeholder_binary_uses_merged_prune_aware_summary_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    raw_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RAW_OWNER
    ]
    assert len(raw_calls) == 1
    initial_count = next(
        statement
        for statement in owner.body
        if _single_target(statement) == "initial_tensor_count"
    )
    assert isinstance(initial_count, ast.Assign)
    assert ast.unparse(initial_count.value) == "len(model_ir.tensors)"

    lowerer = _lowerer()
    summary = next(
        statement
        for statement in ast.walk(lowerer)
        if _single_target(statement) == SUMMARY_TARGET
    )
    body = _containing_body(lowerer, summary)
    index = body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == f"{SUMMARY_OWNER}(model_ir)"
    assert _single_target(body[index - 1]) == PREDECESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {COUNT_TARGET, *RAW_TARGETS}
        for node in ast.walk(lowerer)
    )
    guard = body[index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "_stats_have_positive_count(final_placeholder_reconcile_stats, "
        "final_placeholder_binary_stats)"
    )
    _assert_topology_successor(body[index + 2])
