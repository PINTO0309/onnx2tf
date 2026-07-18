from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import gate_layout_orchestration
from onnx2tf.tflite_builder.passes.gate_layout_orchestration import (
    GATE_LAYOUT_PASS_IDS,
    GATE_LAYOUT_REQUIRED_PASS_IDS,
    run_gate_layout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "gate_layout_orchestration.py"
)
OWNER = "_run_gate_layout_pass_cluster"
RESULT_TARGET = "_layout_opt_gate_layout_results"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer_and_helper() -> tuple[ast.FunctionDef, ast.FunctionDef]:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == OWNER
    )
    return lowerer, helper


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr, ast.Return)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


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


def _direct_location() -> tuple[ast.FunctionDef, list[ast.stmt], int]:
    lowerer, _ = _lowerer_and_helper()
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if _call_name(candidate) == OWNER:
                return lowerer, statement.body, index
    raise AssertionError("direct gate-layout call not found")


def test_gate_layout_result_policy_and_direct_boundary_are_explicit() -> None:
    assert GATE_LAYOUT_REQUIRED_PASS_IDS == GATE_LAYOUT_PASS_IDS[1:]
    assert len(GATE_LAYOUT_REQUIRED_PASS_IDS) == 7
    assert len(GATE_LAYOUT_PASS_IDS) == 8

    lowerer, helper = _lowerer_and_helper()
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_mixed_attention"
    ]
    assert len(helper.args.kw_defaults) == 1
    assert ast.unparse(helper.args.kw_defaults[0]) == "True"
    assert len(helper.body) == 1
    helper_call = _statement_call(helper.body[0])
    assert helper_call is not None
    assert isinstance(helper_call.func, ast.Name)
    assert helper_call.func.id == "run_gate_layout"
    assert [ast.unparse(argument) for argument in helper_call.args] == [
        "gate_layout_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in helper_call.keywords
    } == {"include_mixed_attention": "include_mixed_attention"}

    observed_lowerer, body, index = _direct_location()
    assert observed_lowerer.name == lowerer.name
    call = _statement_call(body[index])
    assert call is not None
    assert call.args == []
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_mixed_attention": "False"}
    assert _single_target(body[index - 1]) == (
        "_layout_opt_sa_pa_mirrorpad_stats"
    )
    following = body[index + 1]
    assert isinstance(following, ast.For)
    assert ast.unparse(following.target) == "_"
    assert ast.unparse(following.iter) == "range(2)"
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(lowerer)
    ) == 1


@pytest.mark.xfail(
    strict=True,
    reason="gate-layout runner, helper, and direct call discard child results",
)
def test_gate_layout_results_propagate_to_the_direct_observation_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_by_id = {
        pass_id: {"slot": index}
        for index, pass_id in enumerate(GATE_LAYOUT_PASS_IDS)
    }

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> dict[str, int]:
            return dict(expected_by_id[pass_id])

        return record

    for pass_id in GATE_LAYOUT_PASS_IDS:
        monkeypatch.setattr(
            gate_layout_orchestration,
            pass_id,
            recorder(pass_id),
        )

    def context(name: str) -> ModelIRPassContext:
        model_ir = ModelIR(name)
        return ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        )

    assert run_gate_layout(
        context("gate_layout_full_results"),
        include_mixed_attention=True,
    ) == tuple(expected_by_id[pass_id] for pass_id in GATE_LAYOUT_PASS_IDS)
    assert run_gate_layout(
        context("gate_layout_required_results"),
        include_mixed_attention=False,
    ) == tuple(
        expected_by_id[pass_id] for pass_id in GATE_LAYOUT_REQUIRED_PASS_IDS
    )

    runner = _functions(ORCHESTRATION_PATH)["run_gate_layout"]
    assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
    assert len(runner.body) == 2
    assert isinstance(runner.body[-1], ast.Return)
    assert _call_name(runner.body[-1]) == "run_recovery_invocations"

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert _call_name(helper.body[0]) == "run_gate_layout"

    observed_lowerer, body, index = _direct_location()
    assert observed_lowerer.name == lowerer.name
    assert _single_target(body[index]) == RESULT_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
