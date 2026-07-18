from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    sinet_preadd_resize_recovery_orchestration,
)
from onnx2tf.tflite_builder.passes.sinet_preadd_resize_recovery_orchestration import (
    SINET_PREADD_RESIZE_RECOVERY_PASS_IDS,
    SINetPreaddResizeRecoveryContext,
    build_sinet_preadd_resize_recovery_invocations,
    run_sinet_preadd_resize_recovery,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SINET_PREADD_RESIZE = "_run_sinet_preadd_resize_recovery_sequence"
SINET_TERMINAL = "_run_sinet_terminal_layout_recovery_sequence"


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
        if isinstance(node, ast.FunctionDef) and node.name == SINET_PREADD_RESIZE
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context() -> SINetPreaddResizeRecoveryContext:
    model_ir = ModelIR("sinet_preadd_resize_recovery_test")
    return SINetPreaddResizeRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: sinet_preadd_resize_recovery_orchestration.RecoveryInvocation,
    context: SINetPreaddResizeRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_sinet_preadd_resize_recovery_is_a_straight_line_closure() -> None:
    _, helper = _lowerer_and_helper()
    control_flow_nodes = (
        ast.AsyncFor,
        ast.AsyncWith,
        ast.For,
        ast.If,
        ast.Match,
        ast.Try,
        ast.While,
        ast.With,
    )

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 4
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert not any(isinstance(node, control_flow_nodes) for node in ast.walk(helper))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )

    called_names = {
        node.func.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    loaded_data_names = {
        node.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in called_names
    }
    assert loaded_data_names == {"sinet_preadd_resize_recovery_context"}


def test_sinet_preadd_resize_recovery_preserves_all_call_contracts() -> None:
    context = _context()
    invocations = build_sinet_preadd_resize_recovery_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == SINET_PREADD_RESIZE_RECOVERY_PASS_IDS
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_sinet_concat_resize_affine_transpose_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_sinet_dual_resize_affine_transpose_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_sinet_concat_resize_affine_tail_concat_transpose_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_sinet_softmax_mask_residual_nhwc_tail_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
    }


def test_sinet_preadd_resize_recovery_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SINET_PREADD_RESIZE
    ]

    assert len(invocations) == 3
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)

    terminal_context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "sinet_terminal_layout_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(terminal_context_assignment.value, ast.Call)
    callback_keyword = next(
        keyword
        for keyword in terminal_context_assignment.value.keywords
        if keyword.arg == "preadd_resize_recovery"
    )
    assert isinstance(callback_keyword.value, ast.Name)
    assert callback_keyword.value.id == SINET_PREADD_RESIZE


def test_sinet_preadd_resize_recovery_preserves_all_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    nested_index = SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS.index(SINET_PREADD_RESIZE)
    assert list(
        SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS[nested_index - 1 : nested_index + 2]
    ) == [
        "_optimize_sinet_shuffle_residual_transpose_chains",
        SINET_PREADD_RESIZE,
        "_optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains",
    ]

    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINET_PREADD_RESIZE
    ]
    assert len(invocation_indexes) == 3
    observed: list[tuple[str, str]] = []
    assigned_boundary_targets: list[str] = []
    for index in invocation_indexes:
        previous = lowerer.body[index - 1]
        following = lowerer.body[index + 1]
        for boundary in (previous, following):
            assert isinstance(boundary, (ast.Assign, ast.Expr))
            assert isinstance(boundary.value, ast.Call)
            assert isinstance(boundary.value.func, ast.Name)
            if isinstance(boundary, ast.Assign):
                assert len(boundary.targets) == 1
                assert isinstance(boundary.targets[0], ast.Name)
                assigned_boundary_targets.append(boundary.targets[0].id)
        observed.append((previous.value.func.id, following.value.func.id))

    assert observed == [
        (
            "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
            "_run_singleton_reshape_layout_pass_cluster",
        ),
        (
            SINET_TERMINAL,
            "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
        ),
        (
            "_reconcile_static_tensor_shapes",
            "_optimize_transpose_csp_attention_nhwc_chains",
        ),
    ]
    assert assigned_boundary_targets == [
        "_post_terminal_singleton_reshape_results",
    ]


def test_sinet_preadd_resize_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    statement = helper.body[0]
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_sinet_preadd_resize_recovery"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "sinet_preadd_resize_recovery_context"
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "sinet_preadd_resize_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_sinet_preadd_resize_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    probe_steps = build_sinet_preadd_resize_recovery_invocations(context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for step in probe_steps:
        module_name = next(
            name
            for name, value in vars(sinet_preadd_resize_recovery_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            sinet_preadd_resize_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_sinet_preadd_resize_recovery(context)

    assert events == list(SINET_PREADD_RESIZE_RECOVERY_PASS_IDS)


def test_sinet_preadd_resize_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "sinet_preadd_resize_recovery_orchestration.py"
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
