from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.absolute_final_normalization_attention_orchestration import (
    ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS,
    build_absolute_final_normalization_attention_invocations,
)
from onnx2tf.tflite_builder.passes.attention_layout import (
    run_mixed_attention_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.gate_layout_orchestration import (
    GATE_LAYOUT_PASS_IDS,
    GATE_LAYOUT_REQUIRED_PASS_IDS,
    build_gate_layout_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "attention_layout.py"
)
OWNER = "run_mixed_attention_layout_cleanup"
INNER_OWNER = "_optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains"
RESULT_TARGET = "_post_sinet_mixed_attention_layout_stats"
PREDECESSOR_TARGET = "_post_sinet_mix_attention_stats"
SUCCESSOR_TARGET = "_post_sinet_dequant_hardsigmoid_bridge_stats"
RESULT_KEY = "optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
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


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _direct_location() -> tuple[ast.FunctionDef, int]:
    lowerer = _lowerer()
    return lowerer, next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == OWNER
    )


def _context(name: str) -> ModelIRPassContext:
    model_ir = ModelIR(name)
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def test_mixed_attention_layout_schema_cleanup_and_routes_are_explicit() -> None:
    functions = _functions(OWNER_PATH)
    owner = functions[OWNER]
    pass_spec = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PassSpec"
    )
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in pass_spec.keywords
    } == {
        "pass_id": "'layout.mixed_attention_mirrorpad'",
        "phase": "PassPhase.LAYOUT_PLAN",
        "callback": "_run",
        "precondition": "_has_candidate",
        "transactional": "True",
    }
    inner_owner = functions[INNER_OWNER]
    assert _call_name(inner_owner.body[-2]) == "_prune_unused_tensors"
    inner_return = inner_owner.body[-1]
    assert isinstance(inner_return, ast.Return)
    assert ast.unparse(inner_return.value) == (
        "{'optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains': int(optimized)}"
    )

    context = _context("mixed_attention_layout_schema")
    assert run_mixed_attention_layout_cleanup(
        context.model_ir,
        layout_state=context.layout_state,
        diagnostics=context.diagnostics,
    ) == {RESULT_KEY: 0}

    full = build_gate_layout_invocations(
        context,
        include_mixed_attention=True,
    )
    reduced = build_gate_layout_invocations(
        context,
        include_mixed_attention=False,
    )
    assert GATE_LAYOUT_PASS_IDS[0] == OWNER
    assert full[0].callback is run_mixed_attention_layout_cleanup
    assert full[0].args == (context.model_ir,)
    assert set(dict(full[0].keyword_args)) == {
        "layout_state",
        "diagnostics",
        "state_scope",
    }
    assert tuple(invocation.pass_id for invocation in reduced) == (
        GATE_LAYOUT_REQUIRED_PASS_IDS
    )
    assert all(invocation.pass_id != OWNER for invocation in reduced)

    absolute_context = _context("mixed_attention_layout_absolute_final")
    absolute = build_absolute_final_normalization_attention_invocations(
        absolute_context
    )
    assert ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS[1] == OWNER
    assert absolute[1].callback is run_mixed_attention_layout_cleanup
    assert absolute[1].args == (absolute_context.model_ir,)
    assert set(dict(absolute[1].keyword_args)) == {
        "layout_state",
        "diagnostics",
        "state_scope",
    }


def test_mixed_attention_layout_direct_boundary_is_explicit() -> None:
    lowerer, index = _direct_location()
    invocation = lowerer.body[index]
    assert isinstance(invocation, ast.Expr)
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(lowerer)
    ) == 1


@pytest.mark.xfail(
    strict=True,
    reason="post-SINet mixed-attention layout result is discarded",
)
def test_mixed_attention_layout_direct_result_is_retained_for_observation() -> None:
    lowerer, index = _direct_location()
    assert _single_target(lowerer.body[index]) == RESULT_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
