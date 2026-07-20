from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    ATTENTION_GATE_QDQ_PASS_IDS,
    PREADD_MEAN_ATTENTION_PASS_IDS,
    AttentionRecoveryContext,
)
from onnx2tf.tflite_builder.passes.layout_pass_set_2_preadd_attention_gate_orchestration import (
    run_layout_pass_set_2_preadd_attention_gate_recovery,
)
from onnx2tf.tflite_builder.passes.layout_pass_set_2_qlinear_layout_recovery_orchestration import (
    run_layout_pass_set_2_qlinear_layout_recovery,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LAYOUT_RECOVERY_PASS_IDS,
    LayoutRecoveryContext,
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
    / "layout_pass_set_2_qlinear_preadd_orchestration.py"
)
OWNER = "run_layout_pass_set_2_qlinear_preadd_cleanup"
CHILD_OWNERS = (
    "run_layout_pass_set_2_qlinear_layout_recovery",
    "run_layout_pass_set_2_preadd_attention_gate_recovery",
)
RESULT_TARGETS = (
    "_layout_pass_set_2_qlinear_layout_recovery_results",
    "_layout_pass_set_2_preadd_attention_gate_results",
)
COMPOSITE_TARGET = "_layout_pass_set_2_qlinear_preadd_results"
PREDECESSOR = "_set_post_progress_desc"
PREDECESSOR_ARGUMENT = "layout recovery pass-set 2"
SUCCESSOR_PHASE_ID = "cleanup.layout_pass_set_2.dequant_transposeconv_quantize"
GUARD = "optimize_layout_transpose_chains"


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
            _single_target(candidate) in (*RESULT_TARGETS, COMPOSITE_TARGET)
            for candidate in statement.body
        )
    )
    assert guard.orelse == []
    return guard.body


def _contexts() -> tuple[
    LayoutRecoveryContext,
    AttentionRecoveryContext,
    dict[str, object],
]:
    model_ir = ModelIR("layout_pass_set_2_qlinear_preadd_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    callback_results: dict[str, object] = {
        "boundary": ({"boundary": 1},),
        "pre_concat": {"pre_concat": 2},
        "channel": ({"channel": 3},),
        "mean": ({"mean": 4},),
        "gate": ({"gate": 5},),
        "unary": ({"unary": 6},),
    }
    layout_context = LayoutRecoveryContext(
        pass_context=pass_context,
        boundary_batchmatmul_unary_cluster=lambda: callback_results[
            "boundary"
        ],
        pre_concat_cleanup=lambda *args, **kwargs: callback_results[
            "pre_concat"
        ],
        channel_shuffle_gather_cluster=lambda: callback_results["channel"],
    )
    attention_context = AttentionRecoveryContext(
        pass_context=pass_context,
        mean_attention_cluster=lambda: callback_results["mean"],
        gate_layout_cluster=lambda: callback_results["gate"],
        transpose_unary_fanout_cluster=lambda: callback_results["unary"],
    )
    return layout_context, attention_context, callback_results


def test_layout_pass_set_2_qlinear_preadd_current_contract() -> None:
    body = _guard_body()
    first = next(
        statement
        for statement in body
        if _single_target(statement) == RESULT_TARGETS[0]
    )
    index = body.index(first)
    assert _call_name(first) == CHILD_OWNERS[0]
    first_call = _call(first)
    assert first_call is not None
    assert [ast.unparse(argument) for argument in first_call.args] == [
        "layout_recovery_context"
    ]
    assert first_call.keywords == []

    predecessor = body[index - 1]
    assert _call_name(predecessor) == PREDECESSOR
    predecessor_call = _call(predecessor)
    assert predecessor_call is not None
    assert [ast.literal_eval(argument) for argument in predecessor_call.args] == [
        PREDECESSOR_ARGUMENT
    ]
    assert predecessor_call.keywords == []

    second = body[index + 1]
    assert _single_target(second) == RESULT_TARGETS[1]
    assert _call_name(second) == CHILD_OWNERS[1]
    second_call = _call(second)
    assert second_call is not None
    assert [ast.unparse(argument) for argument in second_call.args] == [
        "attention_recovery_context"
    ]
    assert second_call.keywords == []
    assert _phase_id(body[index + 2]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(_lowerer())
    )


def test_layout_pass_set_2_qlinear_preadd_child_schemas_and_contexts() -> None:
    layout_context, attention_context, callback_results = _contexts()
    assert layout_context.pass_context is attention_context.pass_context

    qlinear_results = run_layout_pass_set_2_qlinear_layout_recovery(
        layout_context
    )
    preadd_results = run_layout_pass_set_2_preadd_attention_gate_recovery(
        attention_context
    )
    assert len(qlinear_results) == 2
    assert len(qlinear_results[0]) == 5
    assert len(qlinear_results[1]) == len(LAYOUT_RECOVERY_PASS_IDS)
    assert len(preadd_results) == 2
    assert len(preadd_results[0]) == len(PREADD_MEAN_ATTENTION_PASS_IDS)
    assert len(preadd_results[1]) == len(ATTENTION_GATE_QDQ_PASS_IDS)
    assert qlinear_results[1][1] is callback_results["boundary"]
    assert qlinear_results[1][12] is callback_results["pre_concat"]
    assert qlinear_results[1][-1] is callback_results["channel"]
    assert preadd_results[0][-1] is callback_results["mean"]
    assert preadd_results[1][2] is callback_results["gate"]
    assert preadd_results[1][5] is callback_results["unary"]


@pytest.mark.xfail(
    strict=True,
    reason="layout-pass-set-2 QLinear/pre-add owner is not implemented",
)
def test_layout_pass_set_2_qlinear_preadd_has_one_two_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "layout_context"
    ]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "attention_context"
    ]
    assert calls[1].keywords == []

    body = _guard_body()
    assignment = next(
        statement
        for statement in body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "layout_recovery_context",
        "attention_recovery_context",
    ]
    assert call.keywords == []
    assert _call_name(body[index - 1]) == PREDECESSOR
    assert _phase_id(body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(_lowerer())
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )
