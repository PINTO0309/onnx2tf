from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_reshape_layout_orchestration.py"
)
OWNER = "run_late_reshape_layout_cleanup"
RESULT_TARGET = "_late_reshape_layout_results"
PREDECESSOR_TARGET = "_late_concat_elementwise_fanout_stats"
SUCCESSOR_TARGET = "_late_channel_shuffle_gather_results"
OLD_RESULT_TARGETS = (
    "_late_expanddims_reshape_layout_stats",
    "_late_flatten_hw_reshape_layout_stats",
    "_late_nhwc_reshape_collapse_stats",
)
PASS_IDS = (
    "optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat",
    "optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat",
    "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains",
)
LOWERER_PASS_IDS = (
    "_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains",
    "_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains",
    "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains",
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


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def test_late_reshape_layout_cluster_is_ordered_and_unconsumed() -> None:
    lowerer = _lowerer()
    indices: list[int] = []

    for offset, (target, owner) in enumerate(
        zip(OLD_RESULT_TARGETS, LOWERER_PASS_IDS, strict=True)
    ):
        matches = [
            index
            for index, statement in enumerate(lowerer.body)
            if _single_target(statement) == target
        ]
        assert len(matches) == 1
        index = matches[0]
        indices.append(index)
        statement = lowerer.body[index]
        assert _call_name(statement) == owner
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        expected_keywords = (
            {"layout_state": "session.layout_state"}
            if offset < 2
            else {}
        )
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == expected_keywords

    assert indices == list(range(indices[0], indices[0] + len(indices)))
    predecessor = lowerer.body[indices[0] - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == "optimize_layout_transpose_chains"
    assert len(predecessor.body) == 1
    assert _single_target(predecessor.body[0]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET

    for target in OLD_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


@pytest.mark.xfail(
    strict=True,
    reason="late reshape cluster has not moved to one composite owner",
)
def test_late_reshape_layout_cluster_uses_one_composite_owner() -> None:
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
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_reshape_layout_cleanup(shared_model_ir_pass_context)"
    )

    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == "optimize_layout_transpose_chains"
    assert len(predecessor.body) == 1
    assert _single_target(predecessor.body[0]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_phase_result"
        and any(
            isinstance(child, ast.Name) and child.id == OWNER
            for child in ast.walk(node)
        )
        for node in ast.walk(lowerer)
    )
