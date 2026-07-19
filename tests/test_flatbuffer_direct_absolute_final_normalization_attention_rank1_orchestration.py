from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    absolute_final_normalization_attention_orchestration,
)


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


def test_absolute_final_normalization_attention_rank1_summary_boundaries_are_fixed() -> (
    None
):
    lowerer = _lowerer()
    summary_index = _target_index(lowerer, SUMMARY_TARGET)
    assert _single_target(lowerer.body[summary_index - 1]) == PREDECESSOR_TARGET
    _assert_topology_refresh(lowerer.body[summary_index + 1])

    summary = lowerer.body[summary_index]
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(shared_model_ir_pass_context)"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {*OLD_TARGETS, CONTEXT_ALIAS, NORMALIZATION_WRAPPER}
        for node in ast.walk(lowerer)
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


def test_absolute_final_normalization_attention_rank1_preserves_nested_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("absolute_final_normalization_attention_rank1")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict[str, object]] = []
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    normalization_results = (
        {"normalization_pad": 2},
        {"mixed_attention": 3},
    )
    dynamic_result = {
        "rewritten_dynamic_rank1_unsqueeze_reshape_shape_inputs": 4
    }
    events: list[tuple[str, object, dict[str, object]]] = []

    def _run_normalization(
        candidate: ModelIRPassContext,
    ) -> tuple[dict[str, int], ...]:
        events.append((NORMALIZATION_OWNER, candidate, {}))
        return tuple(dict(result) for result in normalization_results)

    def _run_dynamic(
        candidate: ModelIR,
        **kwargs: object,
    ) -> dict[str, int]:
        events.append((DYNAMIC_OWNER, candidate, dict(kwargs)))
        return dict(dynamic_result)

    monkeypatch.setattr(
        absolute_final_normalization_attention_orchestration,
        NORMALIZATION_OWNER,
        _run_normalization,
    )
    monkeypatch.setattr(
        absolute_final_normalization_attention_orchestration,
        DYNAMIC_OWNER,
        _run_dynamic,
    )

    owner_module = absolute_final_normalization_attention_orchestration
    runner = owner_module.run_absolute_final_normalization_attention_rank1_cleanup
    assert runner(context) == (normalization_results, dynamic_result)
    assert events == [
        (NORMALIZATION_OWNER, context, {}),
        (
            DYNAMIC_OWNER,
            model_ir,
            {"layout_state": layout_state},
        ),
    ]
