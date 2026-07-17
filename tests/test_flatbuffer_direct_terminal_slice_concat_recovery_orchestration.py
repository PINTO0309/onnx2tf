from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_slice_concat_recovery_orchestration,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS,
    TerminalSliceConcatRecoveryContext,
    build_terminal_slice_concat_recovery_invocations,
    run_terminal_slice_concat_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TERMINAL_SLICE_CONCAT = "_run_terminal_slice_concat_layout_recovery_sequence"


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
        if isinstance(node, ast.FunctionDef) and node.name == TERMINAL_SLICE_CONCAT
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


def _context() -> TerminalSliceConcatRecoveryContext:
    model_ir = ModelIR("terminal_slice_concat_recovery_test")

    def no_op() -> None:
        return None

    return TerminalSliceConcatRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
        channel_slice_pad_mul_cluster=no_op,
    )


def _normalize_new_contract(
    invocation: terminal_slice_concat_recovery_orchestration.RecoveryInvocation,
    context: TerminalSliceConcatRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        if value is context.diagnostics:
            return "session.diagnostics"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_terminal_slice_concat_recovery_is_a_straight_line_closure() -> None:
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
    assert loaded_data_names == {"terminal_slice_concat_recovery_context"}


def test_terminal_slice_concat_recovery_preserves_all_call_contracts() -> None:
    context = _context()
    invocations = build_terminal_slice_concat_recovery_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS
    )
    contracts = {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    }
    assert contracts == {
        "_run_channel_slice_pad_mul_layout_pass_cluster": ((), {}),
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_concat_mul_add_transpose_nhwc_bridge_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_unary_split_concat_single_post_nchw": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_split_channelwise_tail_to_single_post_nchw": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_sanitize_probable_nhwc_axis_sensitive_ops": (("model_ir",), {}),
        "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_transpose_pre_add_nhwc_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "run_layout_transpose_cleanup": (
            ("model_ir",),
            {
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            },
        ),
    }
    assert invocations[0].callback is context.channel_slice_pad_mul_cluster


def test_terminal_slice_concat_recovery_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_SLICE_CONCAT
    ]

    assert len(invocations) == 2
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)


def test_terminal_slice_concat_recovery_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TERMINAL_SLICE_CONCAT
    ]

    assert len(invocation_indexes) == 2
    observed: list[tuple[str, tuple[str | None, ...], str]] = []
    for index in invocation_indexes:
        previous = lowerer.body[index - 1]
        following = lowerer.body[index + 1]
        assert isinstance(previous, ast.Expr)
        assert isinstance(previous.value, ast.Call)
        assert isinstance(previous.value.func, ast.Name)
        assert isinstance(following, ast.Expr)
        assert isinstance(following.value, ast.Call)
        assert isinstance(following.value.func, ast.Name)
        observed.append(
            (
                previous.value.func.id,
                tuple(keyword.arg for keyword in previous.value.keywords),
                following.value.func.id,
            )
        )

    assert observed == [
        (
            "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains",
            ("layout_state",),
            "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks",
        ),
        (
            "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains",
            (),
            "_optimize_transpose_slice_prepost_nhwc_passthrough_chains",
        ),
    ]


def test_terminal_slice_concat_recovery_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    statement = helper.body[0]
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_terminal_slice_concat_recovery"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "terminal_slice_concat_recovery_context"
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "terminal_slice_concat_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    assert isinstance(context_assignment.value.func, ast.Name)
    assert context_assignment.value.func.id == "TerminalSliceConcatRecoveryContext"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in context_assignment.value.keywords
    } == {
        "model_ir": "model_ir",
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "channel_slice_pad_mul_cluster": (
            "_run_channel_slice_pad_mul_layout_pass_cluster"
        ),
    }


def test_terminal_slice_concat_recovery_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    model_ir = ModelIR("terminal_slice_concat_recovery_order_test")
    context = TerminalSliceConcatRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
        channel_slice_pad_mul_cluster=recorder(
            "_run_channel_slice_pad_mul_layout_pass_cluster"
        ),
    )
    probe_steps = build_terminal_slice_concat_recovery_invocations(context)
    for step in probe_steps:
        if step.callback is context.channel_slice_pad_mul_cluster:
            continue
        module_name = next(
            name
            for name, value in vars(
                terminal_slice_concat_recovery_orchestration
            ).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            terminal_slice_concat_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_terminal_slice_concat_recovery(context)

    assert events == list(TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS)


def test_terminal_slice_concat_recovery_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_slice_concat_recovery_orchestration.py"
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
