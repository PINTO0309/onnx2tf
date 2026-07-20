from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    layout_pass_set_2_preadd_attention_gate_orchestration,
)
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    ATTENTION_GATE_QDQ_PASS_IDS,
    PREADD_MEAN_ATTENTION_PASS_IDS,
    AttentionRecoveryContext,
    run_attention_gate_qdq_recovery,
    run_preadd_mean_attention_recovery,
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
    / "layout_pass_set_2_preadd_attention_gate_orchestration.py"
)
OWNER = "run_layout_pass_set_2_preadd_attention_gate_recovery"
CHILD_OWNERS = (
    "run_preadd_mean_attention_recovery",
    "run_attention_gate_qdq_recovery",
)
CURRENT_CHILD_OWNERS = (
    "_run_preadd_mean_attention_recovery_sequence",
    "_run_attention_gate_qdq_recovery_sequence",
)
RESULT_TARGETS = (
    "_layout_pass_set_2_preadd_mean_attention_results",
    "_layout_pass_set_2_attention_gate_qdq_results",
)
COMPOSITE_TARGET = "_layout_pass_set_2_preadd_attention_gate_results"
OUTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_pass_set_2_qlinear_preadd_orchestration.py"
)
OUTER_OWNER = "run_layout_pass_set_2_qlinear_preadd_cleanup"
OUTER_TARGET = "_layout_pass_set_2_qlinear_preadd_results"
SUCCESSOR_PHASE_ID = (
    "cleanup.layout_pass_set_2.dequant_transposeconv_quantize"
)
GUARD = "optimize_layout_transpose_chains"

PREADD_SCHEMA = (
    ("optimized_transpose_pre_add_nhwc_chains",),
    ("optimized_transpose_pre_add_mul_add_prelu_nhwc_chains",),
    (
        "optimized_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
    ),
    ("optimized_transpose_mul_add_const_prepost_nhwc_chains",),
    (
        "optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
    ),
    ("optimized_transpose_mean_mul_add_const_prepost_nhwc_chains",),
)
ATTENTION_GATE_QDQ_SCHEMA = (
    ("optimized_transpose_sa_pa_mirrorpad_nhwc_propagation_chains",),
    ("optimized_sinet_mix_attention_double_logistic_nhwc_chains",),
    ("rewritten_transposeconv_output_nhwc_passthrough_chains",),
    ("rewritten_transposeconv_output_channel1_terminal_transpose_chains",),
    ("removed_transpose_dequant_relu_quantize_bridges",),
    ("removed_transpose_dequant_hardsigmoid_quantize_bridges",),
    ("rewritten_trailing_output_transpose_passthrough_chains",),
    ("removed_transpose_dequant_mul_add_prelu_quantize_bridges",),
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _phase_id(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or ast.unparse(call.func) != "session.record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _guard_body() -> list[ast.stmt]:
    guard = next(
        statement
        for statement in _lowerer().body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == GUARD
        and any(
            _single_target(candidate)
                in (*RESULT_TARGETS, COMPOSITE_TARGET, OUTER_TARGET)
            for candidate in statement.body
        )
    )
    assert guard.orelse == []
    return guard.body


def _context() -> tuple[
    AttentionRecoveryContext,
    tuple[dict[str, int], ...],
    tuple[dict[str, int], ...],
    tuple[dict[str, int], ...],
]:
    model_ir = ModelIR("layout_pass_set_2_preadd_attention_gate_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    mean_results = ({"mean_attention": 1},)
    gate_results = ({"gate_layout": 1},)
    transpose_results = ({"transpose_unary_fanout": 1},)
    context = AttentionRecoveryContext(
        pass_context=pass_context,
        mean_attention_cluster=lambda: mean_results,
        gate_layout_cluster=lambda: gate_results,
        transpose_unary_fanout_cluster=lambda: transpose_results,
    )
    return context, mean_results, gate_results, transpose_results


def _dict_schema(values: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    assert all(isinstance(value, dict) for value in values)
    return tuple(tuple(value) for value in values)


def test_layout_pass_set_2_preadd_attention_gate_current_contract() -> None:
    body = _guard_body()
    assignment = next(
        statement
        for statement in body
        if _single_target(statement) == OUTER_TARGET
    )
    index = body.index(assignment)
    assert _call_name(assignment) == OUTER_OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "layout_recovery_context",
        "attention_recovery_context",
    ]
    assert call.keywords == []
    assert _call_name(body[index - 1]) == "_set_post_progress_desc"
    assert _phase_id(body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and node.id in RESULT_TARGETS
        for node in ast.walk(_lowerer())
    )

    lowerer_functions = {
        node.name: node
        for node in _lowerer().body
        if isinstance(node, ast.FunctionDef)
    }
    for helper_name, child_owner in zip(
        CURRENT_CHILD_OWNERS,
        CHILD_OWNERS,
        strict=True,
    ):
        helper = lowerer_functions[helper_name]
        assert len(helper.body) == 1
        statement = helper.body[0]
        assert isinstance(statement, ast.Return)
        assert isinstance(statement.value, ast.Call)
        assert isinstance(statement.value.func, ast.Name)
        assert statement.value.func.id == child_owner
        assert [ast.unparse(argument) for argument in statement.value.args] == [
            "attention_recovery_context"
        ]
        assert statement.value.keywords == []


def test_layout_pass_set_2_preadd_attention_gate_child_schemas() -> None:
    context, mean_results, gate_results, transpose_results = _context()
    preadd_results = run_preadd_mean_attention_recovery(context)
    attention_results = run_attention_gate_qdq_recovery(context)

    assert len(preadd_results) == len(PREADD_MEAN_ATTENTION_PASS_IDS) == 7
    assert _dict_schema(preadd_results[:6]) == PREADD_SCHEMA
    assert preadd_results[6] is mean_results

    assert len(attention_results) == len(ATTENTION_GATE_QDQ_PASS_IDS) == 10
    assert _dict_schema(
        (
            *attention_results[:2],
            *attention_results[3:5],
            *attention_results[6:],
        )
    ) == ATTENTION_GATE_QDQ_SCHEMA
    assert attention_results[2] is gate_results
    assert attention_results[5] is transpose_results


def test_layout_pass_set_2_preadd_attention_gate_has_one_context_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in CHILD_OWNERS
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [call.func.id for call in calls] == list(CHILD_OWNERS)
    assert all(
        [ast.unparse(argument) for argument in call.args] == ["context"]
        for call in calls
    )
    assert all(call.keywords == [] for call in calls)

    body = _guard_body()
    assignment = next(
        statement
        for statement in body
        if _single_target(statement) == OUTER_TARGET
    )
    index = body.index(assignment)
    assert _call_name(assignment) == OUTER_OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "layout_recovery_context",
        "attention_recovery_context",
    ]
    assert call.keywords == []
    assert _call_name(body[index - 1]) == "_set_post_progress_desc"
    assert _phase_id(body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(_lowerer())
    )

    lowerer_functions = _functions(LOWERER_PATH)
    lowerer_owner = lowerer_functions["lower_onnx_to_ir"]
    nested_functions = {
        node.name
        for node in lowerer_owner.body
        if isinstance(node, ast.FunctionDef)
    }
    assert all(name in nested_functions for name in CURRENT_CHILD_OWNERS)
    outer_owner = _functions(OUTER_OWNER_PATH)[OUTER_OWNER]
    outer_calls = [
        node
        for node in ast.walk(outer_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert len(outer_calls) == 1
    assert [ast.unparse(argument) for argument in outer_calls[0].args] == [
        "attention_context"
    ]
    assert outer_calls[0].keywords == []
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


def test_layout_pass_set_2_preadd_attention_gate_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _, _, _ = _context()
    expected_results = (
        tuple({f"preadd_{index}": index} for index in range(7)),
        tuple({f"attention_{index}": index} for index in range(10)),
    )
    observed: list[tuple[str, object]] = []

    def preadd(active_context: object) -> object:
        observed.append((CHILD_OWNERS[0], active_context))
        return expected_results[0]

    def attention(active_context: object) -> object:
        observed.append((CHILD_OWNERS[1], active_context))
        return expected_results[1]

    monkeypatch.setattr(
        layout_pass_set_2_preadd_attention_gate_orchestration,
        CHILD_OWNERS[0],
        preadd,
    )
    monkeypatch.setattr(
        layout_pass_set_2_preadd_attention_gate_orchestration,
        CHILD_OWNERS[1],
        attention,
    )

    actual = layout_pass_set_2_preadd_attention_gate_orchestration.run_layout_pass_set_2_preadd_attention_gate_recovery(
        context
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], context),
        (CHILD_OWNERS[1], context),
    ]
