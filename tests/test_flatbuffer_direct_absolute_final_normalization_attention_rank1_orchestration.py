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
    / "absolute_final_normalization_attention_orchestration.py"
)
SUMMARY_OWNER = "run_absolute_final_normalization_attention_rank1_cleanup"
SUMMARY_TARGET = "_absolute_final_normalization_attention_rank1_results"
OLD_TARGETS = (
    "_absolute_final_normalization_attention_results",
    "_absolute_final_dynamic_rank1_stats",
)
NORMALIZATION_WRAPPER = (
    "_run_absolute_final_normalization_attention_pass_pair"
)
NORMALIZATION_OWNER = "run_absolute_final_normalization_attention"
DYNAMIC_WRAPPER = "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
DYNAMIC_OWNER = "rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
CONTEXT_ALIAS = "absolute_final_normalization_attention_context"
PREDECESSOR_TARGET = "_absolute_final_affine_instancenorm_results"
SUCCESSOR_PHASE = "topology_layout.primary.absolute_final"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in _tree(path).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _target_index(lowerer: ast.FunctionDef, target_name: str) -> int:
    return next(
        index
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == target_name
    )


def _assert_topology_refresh(statement: ast.stmt) -> None:
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Attribute)
    assert ast.unparse(call.func) == "session.record_phase_result"
    assert ast.literal_eval(call.args[0]) == SUCCESSOR_PHASE
    assert ast.unparse(call.args[1]) == "run_topology_layout_refresh(model_ir)"


def test_absolute_final_normalization_attention_rank1_raw_boundaries_are_fixed() -> (
    None
):
    lowerer = _lowerer()
    normalization_index = _target_index(lowerer, OLD_TARGETS[0])
    assert _target_index(lowerer, OLD_TARGETS[1]) == normalization_index + 1
    assert (
        _single_target(lowerer.body[normalization_index - 1])
        == PREDECESSOR_TARGET
    )
    _assert_topology_refresh(lowerer.body[normalization_index + 2])

    normalization = lowerer.body[normalization_index]
    dynamic_rank1 = lowerer.body[normalization_index + 1]
    assert isinstance(normalization, ast.Assign)
    assert ast.unparse(normalization.value) == f"{NORMALIZATION_WRAPPER}()"
    assert isinstance(dynamic_rank1, ast.Assign)
    assert ast.unparse(dynamic_rank1.value) == (
        f"{DYNAMIC_WRAPPER}(model_ir, layout_state=session.layout_state)"
    )

    nested = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == NORMALIZATION_WRAPPER
    )
    assert len(nested.body) == 1
    assert isinstance(nested.body[0], ast.Return)
    assert ast.unparse(nested.body[0].value) == (
        f"{NORMALIZATION_OWNER}({CONTEXT_ALIAS})"
    )
    context_assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == CONTEXT_ALIAS
    )
    assert isinstance(context_assignment, ast.Assign)
    assert ast.unparse(context_assignment.value) == "shared_model_ir_pass_context"


@pytest.mark.xfail(
    strict=True,
    reason="absolute-final normalization/attention and rank-one repair lack one owner",
)
def test_absolute_final_normalization_attention_rank1_uses_one_context_owner() -> (
    None
):
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {NORMALIZATION_OWNER, DYNAMIC_OWNER}
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    assert [call.func.id for call in calls] == [
        NORMALIZATION_OWNER,
        DYNAMIC_OWNER,
    ]
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {"layout_state": "context.layout_state"}

    lowerer = _lowerer()
    summary_index = _target_index(lowerer, SUMMARY_TARGET)
    summary = lowerer.body[summary_index]
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[summary_index - 1]) == PREDECESSOR_TARGET
    _assert_topology_refresh(lowerer.body[summary_index + 1])
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {*OLD_TARGETS, CONTEXT_ALIAS, NORMALIZATION_WRAPPER}
        for node in ast.walk(lowerer)
    )
    assert DYNAMIC_WRAPPER in _functions(LOWERER_PATH)
