from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.residual_affine_prelu_layout import (
    optimize_transpose_pre_add_mul_add_prelu_nhwc_chains,
)
from onnx2tf.tflite_builder.passes import (
    very_late_sinet_residual_affine_prelu_orchestration,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes.very_late_sinet_recovery_tail_orchestration import (
    run_very_late_sinet_recovery_tail_cleanup,
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
    / "very_late_sinet_residual_affine_prelu_orchestration.py"
)
OWNER = "run_very_late_sinet_residual_affine_prelu_cleanup"
CHILD_OWNERS = (
    "run_very_late_sinet_recovery_tail_cleanup",
    "optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
)
CURRENT_TARGET = "_very_late_sinet_recovery_tail_results"
CURRENT_PRELU_WRAPPER = (
    "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains"
)
PHASE_ID = "cleanup.very_late.residual_affine_prelu"
PREDECESSOR_PHASE_ID = "shape_topology.terminal.indexed_convergence"
SUCCESSOR_PHASE_ID = "cleanup.very_late.residual_affine_fanout"
FUTURE_OWNER_EXPRESSION = (
    "run_very_late_sinet_residual_affine_prelu_cleanup("
    "sinet_terminal_layout_recovery_context)[1]"
)
PRELU_SCHEMA = (
    "optimized_transpose_pre_add_mul_add_prelu_nhwc_chains",
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
    if not isinstance(statement, (ast.Assign, ast.Expr, ast.Return)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


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


def _phase_record(lowerer: ast.FunctionDef) -> ast.Expr:
    records = [
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Expr) and _phase_id(statement) == PHASE_ID
    ]
    assert len(records) == 1
    return records[0]


def _context() -> SINetTerminalLayoutRecoveryContext:
    model_ir = ModelIR("very_late_sinet_residual_affine_prelu_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    return SINetTerminalLayoutRecoveryContext(
        pass_context=pass_context,
        preadd_resize_recovery=lambda: (),
    )


def test_very_late_sinet_residual_affine_prelu_current_contract() -> None:
    lowerer = _lowerer()
    record = _phase_record(lowerer)
    index = lowerer.body.index(record)
    predecessor = lowerer.body[index - 1]

    assert _phase_id(predecessor) == PREDECESSOR_PHASE_ID

    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == FUTURE_OWNER_EXPRESSION
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )


def test_very_late_sinet_residual_affine_prelu_schemas_are_fixed() -> None:
    context = _context()
    sinet_results = run_very_late_sinet_recovery_tail_cleanup(context)
    prelu_results = optimize_transpose_pre_add_mul_add_prelu_nhwc_chains(
        context.pass_context.model_ir
    )

    assert tuple(type(result) for result in sinet_results) == (tuple, tuple)
    assert tuple(len(result) for result in sinet_results) == (3, 0)
    assert tuple(type(result) for result in sinet_results[0]) == (
        dict,
        tuple,
        dict,
    )
    assert tuple(prelu_results) == PRELU_SCHEMA
    assert all(type(value) is int for value in prelu_results.values())


def test_very_late_sinet_residual_affine_prelu_wrapper_is_retained() -> None:
    wrapper = _functions(LOWERER_PATH)[CURRENT_PRELU_WRAPPER]
    assert len(wrapper.body) == 1
    statement = wrapper.body[0]
    assert isinstance(statement, ast.Return)
    call = _call(statement)
    assert call is not None
    assert isinstance(call.func, ast.Name)
    assert call.func.id == f"{CURRENT_PRELU_WRAPPER}_pass"
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []


def test_very_late_sinet_residual_affine_prelu_has_one_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.pass_context.model_ir"
    ]
    assert calls[1].keywords == []
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == "(sinet_results, prelu_results)"

    lowerer = _lowerer()
    record = _phase_record(lowerer)
    index = lowerer.body.index(record)
    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == FUTURE_OWNER_EXPRESSION
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


def test_very_late_sinet_residual_affine_prelu_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    sinet_results = (({"sinet": 1},), ({"preadd": 2},))
    prelu_results = {"prelu": 3}
    observed: list[tuple[str, object]] = []

    def sinet(active_context: object) -> object:
        observed.append((CHILD_OWNERS[0], active_context))
        return sinet_results

    def prelu(active_model_ir: object) -> object:
        observed.append((CHILD_OWNERS[1], active_model_ir))
        return prelu_results

    monkeypatch.setattr(
        very_late_sinet_residual_affine_prelu_orchestration,
        CHILD_OWNERS[0],
        sinet,
    )
    monkeypatch.setattr(
        very_late_sinet_residual_affine_prelu_orchestration,
        CHILD_OWNERS[1],
        prelu,
    )

    actual = very_late_sinet_residual_affine_prelu_orchestration.run_very_late_sinet_residual_affine_prelu_cleanup(
        context
    )
    assert actual[0] is sinet_results
    assert actual[1] is prelu_results
    assert observed == [
        (CHILD_OWNERS[0], context),
        (CHILD_OWNERS[1], context.pass_context.model_ir),
    ]
