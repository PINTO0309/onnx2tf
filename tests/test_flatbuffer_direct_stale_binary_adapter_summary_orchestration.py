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
    / "stale_binary_adapter_repair.py"
)
RAW_OWNER = "_repair_stale_nchw_to_nhwc_channelwise_binary_transposes"
SUMMARY_OWNER = "run_stale_binary_adapter_repair_summary"
SITE_CONTRACTS = (
    (
        "fallback_binary_layout_tensor_count",
        "fallback_binary_layout_stats",
        "fallback_ir",
        "fallback_concat_axis_stats",
        (
            "session.record_phase_result("
            "'topology.fallback.post_layout_repair', "
            "_topologically_sort_operators(fallback_ir))"
        ),
    ),
    (
        "final_binary_layout_tensor_count",
        "final_binary_layout_stats",
        "model_ir",
        "final_concat_axis_stats",
        "_advance_post_progress()",
    ),
)


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


def _containing_body(root: ast.AST, target: ast.stmt) -> list[ast.stmt]:
    for node in ast.walk(root):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and target in value:
                return value
    raise AssertionError("statement is not contained by an AST body")


def test_stale_binary_adapter_prune_aware_boundaries_are_fixed() -> None:
    lowerer = _lowerer()
    for count_target, stats_target, model_name, predecessor, successor in (
        SITE_CONTRACTS
    ):
        count = next(
            statement
            for statement in ast.walk(lowerer)
            if _single_target(statement) == count_target
        )
        body = _containing_body(lowerer, count)
        index = body.index(count)
        stats = body[index + 1]
        assert isinstance(count, ast.Assign)
        assert ast.unparse(count.value) == f"len({model_name}.tensors)"
        assert _single_target(stats) == stats_target
        assert isinstance(stats, ast.Assign)
        assert isinstance(stats.value, ast.Dict)
        assert len(stats.value.keys) == 2
        assert stats.value.keys[0] is None
        assert ast.unparse(stats.value.values[0]) == (
            f"{RAW_OWNER}({model_name})"
        )
        assert ast.unparse(stats.value.keys[1]) == "'pruned_unused_tensors'"
        assert ast.unparse(stats.value.values[1]) == (
            f"max(0, {count_target} - len({model_name}.tensors))"
        )
        preceding = body[index - 1]
        assert isinstance(preceding, ast.If)
        assert any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == predecessor
            for node in ast.walk(preceding.test)
        )
        guard = body[index + 2]
        assert isinstance(guard, ast.If)
        assert any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == stats_target
            for node in ast.walk(guard.test)
        )
        assert ast.unparse(body[index + 3]) == successor


@pytest.mark.xfail(
    strict=True,
    reason="stale binary-adapter sites lack one prune-aware summary owner",
)
def test_stale_binary_adapter_uses_one_shared_prune_aware_summary_owner() -> None:
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
    for count_target, stats_target, model_name, predecessor, successor in (
        SITE_CONTRACTS
    ):
        stats = next(
            statement
            for statement in ast.walk(lowerer)
            if _single_target(statement) == stats_target
        )
        body = _containing_body(lowerer, stats)
        index = body.index(stats)
        assert isinstance(stats, ast.Assign)
        assert ast.unparse(stats.value) == f"{SUMMARY_OWNER}({model_name})"
        preceding = body[index - 1]
        assert isinstance(preceding, ast.If)
        assert any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == predecessor
            for node in ast.walk(preceding.test)
        )
        assert not any(
            isinstance(node, ast.Name) and node.id == count_target
            for node in ast.walk(lowerer)
        )
        following = body[index + 1]
        assert isinstance(following, ast.If)
        assert any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == stats_target
            for node in ast.walk(following.test)
        )
        assert ast.unparse(body[index + 2]) == successor

    wrapper = _functions(LOWERER_PATH)[RAW_OWNER]
    assert len(wrapper.body) == 1
    assert ast.unparse(wrapper.body[0]) == (
        f"return {RAW_OWNER}_pass(model_ir, graph_index=graph_index)"
    )
