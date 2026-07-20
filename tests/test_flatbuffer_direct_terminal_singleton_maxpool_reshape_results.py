from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_singleton_maxpool_reshape_orchestration,
)
from onnx2tf.tflite_builder.passes.terminal_singleton_maxpool_reshape_orchestration import (
    TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS,
    build_terminal_singleton_maxpool_reshape_invocations,
    run_terminal_singleton_maxpool_reshape,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_singleton_maxpool_reshape_orchestration.py"
)
LOWERER_HELPER = "_run_terminal_singleton_maxpool_reshape_pass_pair"
OUTER_OWNER = "run_late_affine_final_shape_terminal_convpool_cleanup"
RESULT_TARGET = "_late_affine_final_shape_terminal_convpool_results"


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
        if isinstance(node, ast.FunctionDef) and node.name == LOWERER_HELPER
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
        if _call_name(statement) == OUTER_OWNER
    )


def test_terminal_singleton_maxpool_reshape_result_contract_is_explicit() -> None:
    assert TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS == (
        "run_singleton_maxpool_layout_cleanup",
        "run_consecutive_reshape_cleanup",
    )
    context = _context("terminal_singleton_maxpool_reshape_schema")
    assert tuple(
        invocation.run()
        for invocation in build_terminal_singleton_maxpool_reshape_invocations(
            context
        )
    ) == (
        {
            "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains": 0,
            "rewritten_singleton_nms_maxpool_nhwc_chains": 0,
        },
        {
            "removed_noop_reshape_chains": 0,
            "rewritten_consecutive_reshape_passthrough_chains": 0,
            "rewritten_fanout_bypass_reshape_passthrough_chains": 0,
        },
    )

    runner = _functions(ORCHESTRATION_PATH)[
        "run_terminal_singleton_maxpool_reshape"
    ]
    assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
    assert len(runner.body) == 1
    assert isinstance(runner.body[0], ast.Return)
    assert _call_name(runner.body[0]) == "run_recovery_invocations"

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert _call_name(helper.body[0]) == "run_terminal_singleton_maxpool_reshape"

    observed_lowerer, index = _direct_location()
    assert observed_lowerer.name == lowerer.name
    invocation = observed_lowerer.body[index]
    assert isinstance(invocation, ast.Assign)
    assert _single_target(invocation) == RESULT_TARGET
    assert _statement_call(invocation) is not None
    assert observed_lowerer.body[index - 1].__class__ is ast.Expr
    assert ast.literal_eval(
        observed_lowerer.body[index - 1].value.args[0]
    ) == "cleanup.late.ndhwc_cost_volume"
    assert observed_lowerer.body[index + 1].__class__ is ast.If
    assert ast.unparse(observed_lowerer.body[index + 1].test) == (
        "not optimize_layout_transpose_chains and "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OUTER_OWNER
        for node in ast.walk(lowerer)
    ) == 1


def test_terminal_singleton_maxpool_reshape_results_propagate_to_direct_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_results = (
        {
            "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains": 1,
            "rewritten_singleton_nms_maxpool_nhwc_chains": 2,
        },
        {
            "removed_noop_reshape_chains": 3,
            "rewritten_consecutive_reshape_passthrough_chains": 4,
            "rewritten_fanout_bypass_reshape_passthrough_chains": 5,
        },
    )
    for pass_id, expected in zip(
        TERMINAL_SINGLETON_MAXPOOL_RESHAPE_PASS_IDS,
        expected_results,
    ):

        def result(
            *args: Any,
            _expected: dict[str, int] = expected,
            **kwargs: Any,
        ) -> dict[str, int]:
            return dict(_expected)

        monkeypatch.setattr(
            terminal_singleton_maxpool_reshape_orchestration,
            pass_id,
            result,
        )

    assert run_terminal_singleton_maxpool_reshape(
        _context("terminal_singleton_maxpool_reshape_results")
    ) == expected_results

    runner = _functions(ORCHESTRATION_PATH)[
        "run_terminal_singleton_maxpool_reshape"
    ]
    assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
    assert len(runner.body) == 1
    assert isinstance(runner.body[0], ast.Return)
    assert _call_name(runner.body[0]) == "run_recovery_invocations"

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert _call_name(helper.body[0]) == (
        "run_terminal_singleton_maxpool_reshape"
    )

    observed_lowerer, index = _direct_location()
    assert observed_lowerer.name == lowerer.name
    assert _single_target(observed_lowerer.body[index]) == RESULT_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
