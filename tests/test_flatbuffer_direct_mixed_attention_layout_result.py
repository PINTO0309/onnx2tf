from __future__ import annotations

import ast
from pathlib import Path

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
RESULT_KEY = "optimized_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains"
POST_SINET_RESULT_TARGETS = (
    "_post_sinet_mix_attention_stats",
    "_post_sinet_mixed_attention_layout_stats",
    "_post_sinet_dequant_hardsigmoid_bridge_stats",
)
POST_SINET_PHASE_IDS = (
    "cleanup.post_sinet.mix_attention",
    "cleanup.post_sinet.mixed_attention_layout",
    "cleanup.post_sinet.dequant_hardsigmoid_bridge",
)
POST_SINET_OWNER_EXPRESSIONS = (
    (
        "_optimize_sinet_mix_attention_double_logistic_nhwc_chains("
        "model_ir, layout_state=session.layout_state)"
    ),
    (
        "run_mixed_attention_layout_cleanup("
        "model_ir, layout_state=session.layout_state, "
        "diagnostics=session.diagnostics)"
    ),
    "_optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir)",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        return call.args[1]
    return call


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


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


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


def test_mixed_attention_layout_direct_phase_boundary_is_explicit() -> None:
    lowerer, index = _direct_location()
    invocation = lowerer.body[index]
    assert _phase_id(invocation) == "cleanup.post_sinet.mixed_attention_layout"
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }
    assert _phase_id(lowerer.body[index - 1]) == "cleanup.post_sinet.mix_attention"
    assert _phase_id(lowerer.body[index + 1]) == (
        "cleanup.post_sinet.dequant_hardsigmoid_bridge"
    )
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(lowerer)
    ) == 1


def test_mixed_attention_layout_direct_result_uses_phase_store() -> None:
    lowerer, index = _direct_location()
    assert _phase_id(lowerer.body[index]) == (
        "cleanup.post_sinet.mixed_attention_layout"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )


def test_post_sinet_attention_activation_results_use_phase_store() -> None:
    lowerer = _lowerer()
    records = [
        statement
        for statement in lowerer.body
        if _phase_id(statement) in POST_SINET_PHASE_IDS
    ]
    indices = [lowerer.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == POST_SINET_PHASE_IDS
    assert tuple(ast.unparse(statement.value.args[1]) for statement in records) == (
        POST_SINET_OWNER_EXPRESSIONS
    )
    assert indices == list(range(indices[0], indices[0] + 3))
    assert _phase_id(lowerer.body[indices[0] - 1]) == (
        "cleanup.post_sinet.split_conv_concat_bridge"
    )
    assert _single_target(lowerer.body[indices[-1] + 1]) == (
        "late_ndhwc_cost_volume_state_scope"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in POST_SINET_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
