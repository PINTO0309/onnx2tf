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
    / "terminal_concat_bridge_layout_orchestration.py"
)
OWNER = "run_terminal_concat_bridge_layout_cleanup"
RESULT_TARGET = "_terminal_concat_bridge_layout_results"
PREDECESSOR_TARGET = "_final_pre_concat_stats"
SUCCESSOR_TARGET = "_terminal_elementwise_fanout_stats"
OLD_RESULT_TARGETS = (
    "_terminal_relu_split_all_outputs_stats",
    "_terminal_relu_split_conv_concat_stats",
    "_terminal_split_mixed_pre_concat_stats",
    "_terminal_concat_input_adapter_stats",
    "_terminal_concat_unary_conv_stats",
    "_terminal_shape_extract_stats",
)
PASS_IDS = (
    "optimize_transpose_relu_split_all_outputs_to_nhwc_chains",
    "optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains",
    "optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
    "optimize_transpose_input_chains_pre_concat_to_single_post_adapter",
    "run_concat_unary_conv_layout_cleanup",
    "optimize_transpose_shape_extract_nhwc_to_nchw_chains",
)
LOWERER_PASS_IDS = (
    "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains",
    "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains",
    "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
    "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter",
    "run_concat_unary_conv_layout_cleanup",
    "_optimize_transpose_shape_extract_nhwc_to_nchw_chains",
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


def test_terminal_concat_bridge_cluster_is_ordered_and_unconsumed() -> None:
    lowerer = _lowerer()
    indices: list[int] = []
    expected_keywords = (
        {"layout_state": "session.layout_state"},
        {"layout_state": "session.layout_state"},
        {"layout_state": "session.layout_state"},
        {"layout_state": "session.layout_state"},
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
        },
        {},
    )

    for target, owner, keywords in zip(
        OLD_RESULT_TARGETS,
        LOWERER_PASS_IDS,
        expected_keywords,
        strict=True,
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
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == keywords

    assert indices == list(range(indices[0], indices[0] + len(indices)))
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    successor = lowerer.body[indices[-1] + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == "optimize_layout_transpose_chains"
    assert len(successor.body) == 1
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
    for target in OLD_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


@pytest.mark.xfail(
    strict=True,
    reason="terminal concat bridge cluster has not moved to one composite owner",
)
def test_terminal_concat_bridge_cluster_uses_one_composite_owner() -> None:
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
        "run_terminal_concat_bridge_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert _single_target(successor.body[0]) == SUCCESSOR_TARGET
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
