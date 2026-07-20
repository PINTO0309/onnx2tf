from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
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
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_slice_concat_recovery_orchestration.py"
)
PRE_ADD_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "pre_add_layout.py"
)
LAYOUT_TRANSPOSE_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "layout_transpose.py"
)
TERMINAL_SLICE_CONCAT = "_run_terminal_slice_concat_layout_recovery_sequence"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_boundary_slice_concat_orchestration.py"
)
COMPOSITE_OWNER = "run_final_boundary_slice_concat_cleanup"
COMPOSITE_TARGET = "_late_affine_final_shape_terminal_convpool_results"
RESULT_TARGETS = (
    "_terminal_slice_concat_recovery_results",
    "_final_slice_concat_recovery_results",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _direct_call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    function = statement.value.func
    return function.id if isinstance(function, ast.Name) else None


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


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


def _composite_recovery_calls() -> list[ast.Call]:
    owner = _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_terminal_slice_concat_recovery"
    ]


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context(*, callback_result: Any = None) -> TerminalSliceConcatRecoveryContext:
    model_ir = ModelIR("terminal_slice_concat_recovery_test")

    def no_op() -> Any:
        return callback_result

    return TerminalSliceConcatRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        channel_slice_pad_mul_cluster=no_op,
    )


def _normalize_new_contract(
    invocation: terminal_slice_concat_recovery_orchestration.RecoveryInvocation,
    context: TerminalSliceConcatRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.pass_context.model_ir:
            return "model_ir"
        if value is context.pass_context.layout_state:
            return "session.layout_state"
        if value is context.pass_context.diagnostics:
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
        for statement in helper.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    loaded_data_names = {
        node.id
        for statement in helper.body
        for node in ast.walk(statement)
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


def test_terminal_slice_concat_child_schemas_and_cleanup_are_explicit() -> None:
    channel_slice_results = (
        {"channel_slice_mutations": 0},
        {"pad_mul_mutations": 0},
    )
    invocations = build_terminal_slice_concat_recovery_invocations(
        _context(callback_result=channel_slice_results)
    )
    assert tuple(invocation.run() for invocation in invocations) == (
        channel_slice_results,
        {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 0},
        {"optimized_concat_mul_add_transpose_nhwc_bridge_chains": 0},
        {"optimized_concat_mul_add_transpose_add_nhwc_bridge_chains": 0},
        {
            "optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": 0
        },
        {"optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains": 0},
        {"optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 0},
        {"optimized_transpose_unary_split_concat_single_post_nchw": 0},
        {"optimized_transpose_split_channelwise_tail_to_single_post_nchw": 0},
        {
            "optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 0
        },
        {
            "sanitized_probable_nhwc_axis_sensitive_ops": 0,
            "inserted_probable_nhwc_terminal_transposes": 0,
        },
        {
            "optimized_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": 0
        },
        {"optimized_transpose_pre_add_nhwc_chains": 0},
        {
            "iterations": 0,
            "removed_identity_transpose": 0,
            "removed_inverse_transpose_pairs": 0,
            "removed_inverse_transpose_fanout_branches": 0,
            "composed_consecutive_transpose_pairs": 0,
        },
    )

    for path, owner_name in (
        (PRE_ADD_PATH, "optimize_transpose_pre_add_nhwc_chains"),
        (LAYOUT_TRANSPOSE_PATH, "_optimize_layout_transpose_chains"),
    ):
        owner = _functions(path)[owner_name]
        assert sum(
            1
            for statement in owner.body
            if _direct_call_name(statement) == "_prune_unused_tensors"
        ) == 1


def test_terminal_slice_concat_recovery_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_SLICE_CONCAT
    ]

    assert invocations == []
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)
    composite_calls = _composite_recovery_calls()
    assert len(composite_calls) == 1
    assert [ast.unparse(argument) for argument in composite_calls[0].args] == [
        "context"
    ]


def test_terminal_slice_concat_recovery_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    record_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and ast.unparse(statement.value.func) == "session.record_phase_result"
        and len(statement.value.args) == 2
        and ast.literal_eval(statement.value.args[0])
        == "cleanup.terminal.boundary_stridedslice_qdq_concat"
    ]
    assert len(record_indexes) == 1
    index = record_indexes[0]
    record = lowerer.body[index]
    assert isinstance(record, ast.Expr)
    assert ast.unparse(record.value.args[1]) == (
        "run_terminal_slice_concat_boundary_stridedslice_cleanup("
        "terminal_slice_concat_recovery_context)[1]"
    )
    assert ast.unparse(lowerer.body[index - 1]) == (
        "session.record_phase_result('cleanup.terminal.channel_slice_muladd_bridge', "
        "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains(model_ir, "
        "layout_state=session.layout_state))"
    )
    assert ast.unparse(lowerer.body[index + 1]) == (
        "session.record_phase_result('cleanup.terminal.swish_residual_concat_closure', "
        "_optimize_transpose_swish_residual_concat_closure_nhwc_chains(model_ir))"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_terminal_slice_concat_recovery_results"
        for node in ast.walk(lowerer)
    )
    composite = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(composite)
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.Expr)
    assert ast.literal_eval(predecessor.value.args[0]) == (
        "cleanup.late.ndhwc_cost_volume"
    )
    successor = lowerer.body[index + 1]
    assert isinstance(successor, ast.If)
    assert _single_target(successor.body[1]) == (
        "_no_layout_fallback_affine_prepost_stats"
    )
    assert len(_composite_recovery_calls()) == 1


def test_terminal_slice_concat_recovery_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
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
        "pass_context": "session.model_ir_pass_context",
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
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
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


def test_terminal_slice_concat_propagates_and_retains_both_ordered_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_results = tuple(
        {"slot": index}
        for index in range(len(TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS))
    )
    context = _context(callback_result=expected_results[0])
    probe_steps = build_terminal_slice_concat_recovery_invocations(context)
    for step, expected in zip(probe_steps[1:], expected_results[1:]):
        module_name = next(
            name
            for name, value in vars(
                terminal_slice_concat_recovery_orchestration
            ).items()
            if value is step.callback
        )

        def result(*args: Any, _expected: dict[str, int] = expected, **kwargs: Any):
            return dict(_expected)

        monkeypatch.setattr(
            terminal_slice_concat_recovery_orchestration,
            module_name,
            result,
        )

    assert run_terminal_slice_concat_recovery(context) == expected_results

    runner = _functions(ORCHESTRATION_PATH)[
        "run_terminal_slice_concat_recovery"
    ]
    assert ast.unparse(runner.returns) == "Tuple[Any, ...]"
    assert len(runner.body) == 1
    assert isinstance(runner.body[0], ast.Return)

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Any, ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    production_results = [
        statement
        for statement in lowerer.body
        if _direct_call_name(statement) == TERMINAL_SLICE_CONCAT
    ]
    assert production_results == []
    assert len(_composite_recovery_calls()) == 1
    for target in RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


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
