from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import late_dequant_unary_fanout_orchestration
from onnx2tf.tflite_builder.passes.late_dequant_unary_fanout_orchestration import (
    LATE_DEQUANT_UNARY_FANOUT_PASS_IDS,
    build_late_dequant_unary_fanout_invocations,
    run_late_dequant_unary_fanout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_dequant_unary_fanout_orchestration.py"
)
OWNER = "_run_late_dequant_unary_fanout_pass_cluster"
RESULT_TARGET = "_late_dequant_unary_fanout_results"
PREDECESSOR_TARGET = "_late_dequant_hardsigmoid_bridge_stats"
SUCCESSOR = "run_late_swish_layout_tail_cleanup"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_dequant_hardsigmoid_unary_orchestration.py"
)
COMPOSITE_OWNER = "run_late_dequant_hardsigmoid_unary_cleanup"
COMPOSITE_TARGET = "_late_dequant_hardsigmoid_unary_results"


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


def _context(name: str) -> ModelIRPassContext:
    model_ir = ModelIR(name)
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _direct_location() -> tuple[ast.FunctionDef, int]:
    lowerer, _ = _lowerer_and_helper()
    return lowerer, next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == COMPOSITE_OWNER
    )


def test_late_dequant_unary_fanout_result_contract_is_explicit() -> None:
    assert LATE_DEQUANT_UNARY_FANOUT_PASS_IDS == (
        "run_dequant_concat_quantize_layout_cleanup",
        "run_transpose_unary_passthrough_cleanup",
        "run_transpose_unary_fanout_bridge_cleanup",
    )
    context = _context("late_dequant_unary_fanout_schema")
    assert tuple(
        invocation.run()
        for invocation in build_late_dequant_unary_fanout_invocations(context)
    ) == (
        {
            "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains": 0,
        },
        {"rewritten_transpose_unary_passthrough_chains": 0},
        {"rewritten_transpose_unary_fanout_inverse_post_bridges": 0},
    )

    runner = _functions(ORCHESTRATION_PATH)["run_late_dequant_unary_fanout"]
    assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
    assert len(runner.body) == 1
    assert isinstance(runner.body[0], ast.Return)
    assert _call_name(runner.body[0]) == "run_recovery_invocations"

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert _call_name(helper.body[0]) == "run_late_dequant_unary_fanout"

    observed_lowerer, index = _direct_location()
    assert observed_lowerer.name == lowerer.name
    assert _single_target(observed_lowerer.body[index]) == COMPOSITE_TARGET
    assert isinstance(observed_lowerer.body[index - 1], ast.If)
    assert _call_name(observed_lowerer.body[index + 1]) == SUCCESSOR
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(lowerer)
    )
    composite = _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_late_dequant_unary_fanout"
        for node in ast.walk(composite)
    ) == 1


def test_late_dequant_unary_fanout_results_propagate_to_direct_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_results = (
        {
            "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains": 1,
        },
        {"rewritten_transpose_unary_passthrough_chains": 2},
        {"rewritten_transpose_unary_fanout_inverse_post_bridges": 3},
    )
    for pass_id, expected in zip(
        LATE_DEQUANT_UNARY_FANOUT_PASS_IDS,
        expected_results,
    ):

        def result(
            *args: Any,
            _expected: dict[str, int] = expected,
            **kwargs: Any,
        ) -> dict[str, int]:
            return dict(_expected)

        monkeypatch.setattr(
            late_dequant_unary_fanout_orchestration,
            pass_id,
            result,
        )

    assert run_late_dequant_unary_fanout(
        _context("late_dequant_unary_fanout_results")
    ) == expected_results

    runner = _functions(ORCHESTRATION_PATH)["run_late_dequant_unary_fanout"]
    assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
    assert len(runner.body) == 1
    assert isinstance(runner.body[0], ast.Return)
    assert _call_name(runner.body[0]) == "run_recovery_invocations"

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert _call_name(helper.body[0]) == "run_late_dequant_unary_fanout"

    observed_lowerer, index = _direct_location()
    assert observed_lowerer.name == lowerer.name
    assert _single_target(observed_lowerer.body[index]) == COMPOSITE_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
