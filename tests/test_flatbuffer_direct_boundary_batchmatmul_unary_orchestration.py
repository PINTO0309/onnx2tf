from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    boundary_batchmatmul_unary_orchestration,
)
from onnx2tf.tflite_builder.passes.boundary_batchmatmul_unary_orchestration import (
    BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS,
    BoundaryBatchMatMulUnaryContext,
    build_boundary_batchmatmul_unary_invocations,
    run_boundary_batchmatmul_unary,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LAYOUT_RECOVERY_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
BOUNDARY_BATCHMATMUL_UNARY = "_run_boundary_batchmatmul_unary_layout_pass_cluster"


def _lowerer_and_helper() -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == BOUNDARY_BATCHMATMUL_UNARY
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context() -> BoundaryBatchMatMulUnaryContext:
    model_ir = ModelIR("boundary_batchmatmul_unary_test")
    return BoundaryBatchMatMulUnaryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: boundary_batchmatmul_unary_orchestration.RecoveryInvocation,
    context: BoundaryBatchMatMulUnaryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        if value is context.diagnostics:
            return "session.diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_boundary_batchmatmul_unary_is_a_straight_line_delegate() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert len(helper.body) == 1
    assert not any(
        isinstance(
            node,
            (
                ast.AsyncFor,
                ast.AsyncWith,
                ast.For,
                ast.If,
                ast.Match,
                ast.Try,
                ast.While,
                ast.With,
            ),
        )
        for node in ast.walk(helper)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )


def test_boundary_batchmatmul_unary_preserves_both_cleanup_contracts() -> None:
    context = _context()
    invocations = build_boundary_batchmatmul_unary_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS
    )
    expected_contract = (
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {pass_id: expected_contract for pass_id in BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_boundary_batchmatmul_unary_invocations(context)[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_boundary_batchmatmul_unary_remains_a_context_callback_only() -> None:
    lowerer, _ = _lowerer_and_helper()
    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == BOUNDARY_BATCHMATMUL_UNARY
    ]
    assert direct_invocations == []

    layout_context = next(
        statement.value
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "layout_recovery_context"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Call)
    )
    callback_keyword = next(
        keyword
        for keyword in layout_context.keywords
        if keyword.arg == "boundary_batchmatmul_unary_cluster"
    )
    assert isinstance(callback_keyword.value, ast.Name)
    assert callback_keyword.value.id == BOUNDARY_BATCHMATMUL_UNARY


def test_boundary_batchmatmul_unary_preserves_stable_callback_boundaries() -> None:
    callback_index = LAYOUT_RECOVERY_PASS_IDS.index(BOUNDARY_BATCHMATMUL_UNARY)
    assert LAYOUT_RECOVERY_PASS_IDS[callback_index - 1] == (
        "_optimize_transpose_quant_dequant_bridges"
    )
    assert LAYOUT_RECOVERY_PASS_IDS[callback_index + 1] == (
        "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains"
    )


def test_boundary_batchmatmul_unary_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    statement = helper.body[0]
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_boundary_batchmatmul_unary"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "boundary_batchmatmul_unary_context",
    )
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "boundary_batchmatmul_unary_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_boundary_batchmatmul_unary_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for pass_id in BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS:
        monkeypatch.setattr(
            boundary_batchmatmul_unary_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_boundary_batchmatmul_unary(context)

    assert events == list(BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS)


def test_boundary_batchmatmul_unary_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "boundary_batchmatmul_unary_orchestration.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.lower_from_onnx2tf"
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "onnx2tf.tflite_builder.lower_from_onnx2tf"
            for alias in node.names
        )
        for node in tree.body
    )
