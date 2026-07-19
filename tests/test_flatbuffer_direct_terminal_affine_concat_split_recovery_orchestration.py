from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_affine_concat_split_recovery_orchestration,
)
from onnx2tf.tflite_builder.passes.terminal_affine_concat_split_recovery_orchestration import (
    TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS,
    TerminalAffineConcatSplitRecoveryContext,
    build_terminal_affine_concat_split_recovery_invocations,
    run_terminal_affine_concat_split_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TERMINAL_AFFINE_CONCAT_SPLIT = "_run_terminal_affine_concat_split_recovery_sequence"
TERMINAL_AFFINE_SUMMARY = (
    "run_terminal_affine_concat_split_recovery_summary"
)
VERY_LATE_PAD_INSTANCENORM = (
    "run_very_late_pad_instancenorm_layout_cleanup"
)
VERY_LATE_LAYOUT_TAIL_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_layout_tail_orchestration.py"
)
VERY_LATE_LAYOUT_TAIL = "run_very_late_layout_tail_cleanup"
PRE_TERMINAL_INSTANCENORM_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_instancenorm_layout_orchestration.py"
)
PRE_TERMINAL_INSTANCENORM = (
    "run_pre_terminal_instancenorm_layout_cleanup"
)
PRE_TERMINAL_INSTANCENORM_RESULT = (
    "_pre_terminal_instancenorm_layout_results"
)
PRE_TERMINAL_CLEANUP_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_cleanup_orchestration.py"
)
PRE_TERMINAL_CLEANUP = "run_pre_terminal_cleanup"
PRE_TERMINAL_CLEANUP_RESULT = "_pre_terminal_cleanup_results"
ABSOLUTE_FINAL_AFFINE_INSTANCENORM_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "absolute_final_affine_instancenorm_orchestration.py"
)
ABSOLUTE_FINAL_AFFINE_INSTANCENORM = (
    "run_absolute_final_affine_instancenorm_cleanup"
)


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
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_AFFINE_CONCAT_SPLIT
    )
    return lowerer, helper


def _very_late_pad_instancenorm_call_count(lowerer: ast.FunctionDef) -> int:
    del lowerer
    tree = ast.parse(VERY_LATE_LAYOUT_TAIL_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == VERY_LATE_LAYOUT_TAIL
    )
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == VERY_LATE_PAD_INSTANCENORM
        for node in ast.walk(owner)
    )


def _pre_terminal_instancenorm_calls(function_name: str) -> list[ast.Call]:
    tree = ast.parse(
        PRE_TERMINAL_INSTANCENORM_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == PRE_TERMINAL_INSTANCENORM
    )
    owner_name = function_name.removeprefix("_")
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
    ]


def _pre_terminal_cleanup_calls(function_name: str) -> list[ast.Call]:
    tree = ast.parse(PRE_TERMINAL_CLEANUP_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == PRE_TERMINAL_CLEANUP
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _absolute_final_affine_instancenorm_calls(
    function_name: str,
) -> list[ast.Call]:
    tree = ast.parse(
        ABSOLUTE_FINAL_AFFINE_INSTANCENORM_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == ABSOLUTE_FINAL_AFFINE_INSTANCENORM
    )
    owner_name = function_name.removeprefix("_")
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
    ]


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context() -> TerminalAffineConcatSplitRecoveryContext:
    model_ir = ModelIR("terminal_affine_concat_split_recovery_test")
    return TerminalAffineConcatSplitRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: terminal_affine_concat_split_recovery_orchestration.RecoveryInvocation,
    context: TerminalAffineConcatSplitRecoveryContext,
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


def test_terminal_affine_concat_split_recovery_is_straight_line() -> None:
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
    assert loaded_data_names == {"terminal_affine_concat_split_recovery_context"}


def test_terminal_affine_concat_split_preserves_all_call_contracts() -> None:
    context = _context()
    invocations = build_terminal_affine_concat_split_recovery_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        "_optimize_fold_mul_add_mul_affine_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains": (
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
    }


def test_terminal_affine_concat_split_invocations_remain_zero_argument() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
        for node in ast.walk(lowerer)
    )
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_AFFINE_SUMMARY
    ]
    assert len(invocations) == 1
    assert all(
        [ast.unparse(argument) for argument in call.args]
        == ["terminal_affine_concat_split_recovery_context"]
        for call in invocations
    )
    assert all(call.keywords == [] for call in invocations)
    composite_invocations = _pre_terminal_cleanup_calls(
        TERMINAL_AFFINE_SUMMARY
    )
    assert len(composite_invocations) == 1
    assert [ast.unparse(arg) for arg in composite_invocations[0].args] == [
        "context"
    ]
    assert composite_invocations[0].keywords == []


def test_terminal_affine_concat_split_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    composite = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == PRE_TERMINAL_CLEANUP_RESULT
    )
    composite_index = lowerer.body.index(composite)
    assert ast.unparse(composite.value) == (
        "run_pre_terminal_cleanup(shared_model_ir_pass_context)"
    )
    assert isinstance(lowerer.body[composite_index - 1], ast.If)
    assert len(_pre_terminal_cleanup_calls(TERMINAL_AFFINE_SUMMARY)) == 1

    terminal = lowerer.body[composite_index + 1]
    assert isinstance(terminal, ast.Assign)
    assert isinstance(terminal.targets[0], ast.Name)
    assert terminal.targets[0].id == "_terminal_affine_stats"
    assert ast.unparse(terminal.value) == (
        "run_terminal_affine_concat_split_recovery_summary("
        "terminal_affine_concat_split_recovery_context)"
    )
    assert isinstance(lowerer.body[composite_index + 2], ast.Assign)
    assert isinstance(lowerer.body[composite_index + 2].targets[0], ast.Name)
    assert (
        lowerer.body[composite_index + 2].targets[0].id
        == "_terminal_slice_pad_concat_stats"
    )


def test_terminal_affine_concat_split_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_terminal_affine_concat_split_recovery"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "terminal_affine_concat_split_recovery_context"
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "terminal_affine_concat_split_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_terminal_affine_concat_split_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    probe_steps = build_terminal_affine_concat_split_recovery_invocations(context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for step in probe_steps:
        module_name = next(
            name
            for name, value in vars(
                terminal_affine_concat_split_recovery_orchestration
            ).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            terminal_affine_concat_split_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_terminal_affine_concat_split_recovery(context)

    assert events == list(TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS)


def test_terminal_affine_returns_and_summarizes_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    result_keys = (
        ("optimized_fold_mul_add_mul_affine_chains",),
        ("optimized_transpose_mul_add_const_prepost_nhwc_chains",),
        ("optimized_concat_mul_add_transpose_nhwc_bridge_chains",),
        ("optimized_concat_mul_add_transpose_add_nhwc_bridge_chains",),
        ("optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",),
        ("optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains",),
        ("optimized_singleton_gate_conv_concat_nhwc_bridge_blocks",),
        ("optimized_transpose_unary_split_concat_single_post_nchw",),
        ("optimized_transpose_split_channelwise_tail_to_single_post_nchw",),
        ("optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw",),
        (
            "sanitized_probable_nhwc_axis_sensitive_ops",
            "inserted_probable_nhwc_terminal_transposes",
        ),
    )
    value = 1
    expected_results = []
    expected_summary: dict[str, int] = {}
    for keys in result_keys:
        result: dict[str, int] = {}
        for key in keys:
            result[key] = value
            expected_summary[key] = value
            value += 1
        expected_results.append(result)
    expected_tuple = tuple(expected_results)

    def return_results(invocations, *, expected_pass_ids, phase_name):
        assert tuple(
            invocation.pass_id for invocation in invocations
        ) == TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS
        assert (
            tuple(expected_pass_ids)
            == TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS
        )
        assert phase_name == "terminal affine/concat/split recovery"
        return expected_tuple

    monkeypatch.setattr(
        terminal_affine_concat_split_recovery_orchestration,
        "run_recovery_invocations",
        return_results,
    )

    results = run_terminal_affine_concat_split_recovery(context)
    summarize = getattr(
        terminal_affine_concat_split_recovery_orchestration,
        "summarize_terminal_affine_concat_split_mutations",
    )
    summary = summarize(results, pruned_unused_tensors=13)

    assert results == expected_tuple
    assert summary == {**expected_summary, "pruned_unused_tensors": 13}
    with pytest.raises(
        ValueError,
        match=r"terminal affine mutation summary expected 11 pass results",
    ):
        summarize((), pruned_unused_tensors=0)

    _, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert isinstance(helper.body[0].value, ast.Call)


def test_lowerer_captures_second_terminal_affine_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    summary = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_affine_stats"
    )
    index = lowerer.body.index(summary)
    assert ast.unparse(summary.value) == (
        "run_terminal_affine_concat_split_recovery_summary("
        "terminal_affine_concat_split_recovery_context)"
    )
    assert isinstance(lowerer.body[index - 1], ast.Assign)
    assert isinstance(lowerer.body[index - 1].targets[0], ast.Name)
    assert lowerer.body[index - 1].targets[0].id == (
        PRE_TERMINAL_CLEANUP_RESULT
    )
    assert isinstance(lowerer.body[index + 1], ast.Assign)
    assert isinstance(lowerer.body[index + 1].targets[0], ast.Name)
    assert lowerer.body[index + 1].targets[0].id == (
        "_terminal_slice_pad_concat_stats"
    )


def test_lowerer_captures_first_terminal_affine_mutation_evidence() -> None:
    invocations = _pre_terminal_cleanup_calls(TERMINAL_AFFINE_SUMMARY)
    assert len(invocations) == 1
    assert [ast.unparse(argument) for argument in invocations[0].args] == [
        "context"
    ]
    assert invocations[0].keywords == []


def _assert_pre_terminal_instancenorm_owner_call(
    function_name: str,
    *,
    expected_total_calls: int,
) -> None:
    lowerer, _ = _lowerer_and_helper()
    composite_calls = _pre_terminal_cleanup_calls(PRE_TERMINAL_INSTANCENORM)
    assert len(composite_calls) == 1
    assert [ast.unparse(argument) for argument in composite_calls[0].args] == [
        "context"
    ]
    assert composite_calls[0].keywords == []

    direct_statements = [
        statement
        for statement in lowerer.body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == function_name
            for node in ast.walk(statement)
        )
    ]
    owner_calls = _pre_terminal_instancenorm_calls(function_name)
    assert len(owner_calls) == 1
    owner_call = owner_calls[0]
    assert [ast.unparse(argument) for argument in owner_call.args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in owner_call.keywords
    } == {"layout_state": "context.layout_state"}
    assert (
        len(direct_statements)
        + _very_late_pad_instancenorm_call_count(lowerer)
        + len(owner_calls)
        + len(_absolute_final_affine_instancenorm_calls(function_name))
        == expected_total_calls
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id
        in {
            "_pre_terminal_affine_instancenorm_post_bias_stats",
            "_pre_terminal_affine_instancenorm_residual_mul_concat_stats",
            "_pre_terminal_affine_instancenorm_dualstats_stats",
        }
        for node in ast.walk(lowerer)
    )


def test_pre_terminal_affine_dualstats_captures_complete_mutation_evidence() -> None:
    _assert_pre_terminal_instancenorm_owner_call(
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
        expected_total_calls=3,
    )


def test_pre_terminal_affine_residual_mul_concat_captures_mutation_evidence() -> None:
    _assert_pre_terminal_instancenorm_owner_call(
        "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
        expected_total_calls=3,
    )


def test_pre_terminal_affine_post_bias_captures_mutation_evidence() -> None:
    _assert_pre_terminal_instancenorm_owner_call(
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
        expected_total_calls=4,
    )


def test_terminal_affine_concat_split_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_affine_concat_split_recovery_orchestration.py"
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
