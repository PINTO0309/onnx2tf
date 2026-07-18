from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import layout_recovery_orchestration
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    ATTENTION_RECOVERY_PASS_IDS,
    LAYOUT_RECOVERY_PASS_IDS,
    LayoutRecoveryContext,
    build_attention_recovery_invocations,
    build_layout_recovery_invocations,
    run_layout_reshape_attention_recovery_prefix,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_recovery_orchestration.py"
)
LAYOUT_PREFIX = "_run_layout_recovery_prefix_pass_sequence"
ATTENTION_PREFIX = "_run_layout_reshape_attention_recovery_prefix"
LAYOUT_RESULT_TARGET = "_layout_pass_set_2_layout_recovery_prefix_results"


def _lowerer_and_helper(helper_name: str) -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
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


def _call_contracts(
    helper: ast.FunctionDef,
) -> dict[str, tuple[tuple[Any, ...], dict[str, Any]]]:
    contracts: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    for statement in helper.body:
        assert isinstance(statement, ast.Expr)
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id not in contracts
        contracts[call.func.id] = (
            tuple(_expression_path(argument) for argument in call.args),
            {
                str(keyword.arg): _expression_path(keyword.value)
                for keyword in call.keywords
            },
        )
    return contracts


def _model_layout_contract(
    *, diagnostics: bool = False
) -> tuple[tuple[str, ...], dict[str, str]]:
    keywords = {"layout_state": "session.layout_state"}
    if diagnostics:
        keywords["diagnostics"] = "session.diagnostics"
    return ("model_ir",), keywords


def _context() -> LayoutRecoveryContext:
    model_ir = ModelIR("layout_recovery_orchestration_test")
    return LayoutRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        boundary_batchmatmul_unary_cluster=lambda: None,
        pre_concat_cleanup=lambda *args, **kwargs: None,
        channel_shuffle_gather_cluster=lambda: None,
    )


def _normalize_new_contract(
    invocation: layout_recovery_orchestration.RecoveryInvocation,
    context: LayoutRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context:
            return "context"
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


def test_layout_recovery_helpers_are_straight_line_closures() -> None:
    expected_lines = {
        LAYOUT_PREFIX: 2,
        ATTENTION_PREFIX: 4,
    }
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
    for helper_name, line_count in expected_lines.items():
        _, helper = _lowerer_and_helper(helper_name)

        assert helper.end_lineno is not None
        assert helper.end_lineno - helper.lineno + 1 == line_count
        assert helper.args.args == []
        assert helper.args.posonlyargs == []
        assert helper.args.kwonlyargs == []
        assert helper.args.vararg is None
        assert helper.args.kwarg is None
        assert not any(
            isinstance(node, control_flow_nodes) for node in ast.walk(helper)
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ModelIRPassStateScope"
            for node in ast.walk(helper)
        )


def test_layout_recovery_prefix_preserves_exact_argument_contracts() -> None:
    context = _context()
    contracts = {
        step.pass_id: _normalize_new_contract(step, context)
        for step in build_layout_recovery_invocations(context)
    }
    model_only = (("model_ir",), {})
    no_arguments = ((), {})
    model_layout = _model_layout_contract()
    model_layout_diagnostics = _model_layout_contract(diagnostics=True)

    assert contracts == {
        "_optimize_transpose_quant_dequant_bridges": model_only,
        "_run_boundary_batchmatmul_unary_layout_pass_cluster": no_arguments,
        "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains": model_only,
        "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains": model_only,
        "run_hard_activation_passthrough_cleanup": model_layout_diagnostics,
        "_optimize_swish_transpose_passthrough_chains": model_layout,
        "_optimize_gelu_tanh_transpose_passthrough_chains": model_layout,
        "_optimize_center_size_offset_terminal_transpose_chains": model_layout,
        "_optimize_leakyrelu_transpose_passthrough_chains": model_layout,
        "_optimize_prelu_transpose_passthrough_chains": model_layout,
        "_optimize_transpose_elementwise_concat_conv_nhwc_groups": model_layout,
        "run_spp_layout_cleanup": model_layout_diagnostics,
        "_optimize_transpose_pre_concat_nhwc_chains": model_layout_diagnostics,
        "run_ndhwc_concat_layout_cleanup": model_layout_diagnostics,
        "_optimize_transpose_stridedslice_pre_concat_nhwc_chains": model_layout,
        "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains": model_layout,
        "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter": model_layout,
        "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains": model_layout,
        "_run_channel_shuffle_gather_layout_pass_cluster": no_arguments,
    }


def test_attention_recovery_prefix_preserves_exact_argument_contracts() -> None:
    context = _context()
    contracts = {
        step.pass_id: _normalize_new_contract(step, context)
        for step in build_attention_recovery_invocations(context)
    }
    model_only = (("model_ir",), {})
    model_layout = _model_layout_contract()

    assert contracts == {
        LAYOUT_PREFIX: (("context",), {}),
        "_optimize_transpose_pre_add_nhwc_chains": model_layout,
        "_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains": model_layout,
        "_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains": model_layout,
        "_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains": model_layout,
        "_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains": model_layout,
        "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains": model_only,
        "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains": model_layout,
        "_optimize_attention_gather_transpose_reshape_cleanup_chains": model_only,
        "_optimize_gather_axis0_singleton_to_reshape_input_chains": model_layout,
        "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains": model_only,
        "_optimize_window_partition_reshape_transpose_to_space_to_depth_chains": model_layout,
        "_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains": model_layout,
        "_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains": model_layout,
        "run_squeeze_reshape_identity_cleanup": (
            ("model_ir",),
            {
                "include_unary_passthrough": True,
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            },
        ),
    }


def test_layout_recovery_wrappers_only_capture_explicit_context() -> None:
    for helper_name in (LAYOUT_PREFIX, ATTENTION_PREFIX):
        _, helper = _lowerer_and_helper(helper_name)
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

        assert loaded_data_names == {"layout_recovery_context"}


def test_new_phase_specs_match_characterized_order_and_arguments() -> None:
    context = _context()
    lowerer, layout_helper = _lowerer_and_helper(LAYOUT_PREFIX)
    _, attention_helper = _lowerer_and_helper(ATTENTION_PREFIX)
    layout_invocations = build_layout_recovery_invocations(context)
    attention_invocations = build_attention_recovery_invocations(context)

    assert (
        tuple(step.pass_id for step in layout_invocations) == LAYOUT_RECOVERY_PASS_IDS
    )
    assert (
        tuple(step.pass_id for step in attention_invocations)
        == ATTENTION_RECOVERY_PASS_IDS
    )
    assert _call_contracts(layout_helper) == {
        "run_layout_recovery_prefix": (("layout_recovery_context",), {}),
    }
    assert _call_contracts(attention_helper) == {
        "run_layout_reshape_attention_recovery_prefix": (
            ("layout_recovery_context",),
            {},
        ),
    }
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "layout_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    assert isinstance(context_assignment.value.func, ast.Name)
    assert context_assignment.value.func.id == "LayoutRecoveryContext"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in context_assignment.value.keywords
    } == {
        "pass_context": "session.model_ir_pass_context",
        "boundary_batchmatmul_unary_cluster": (
            "_run_boundary_batchmatmul_unary_layout_pass_cluster"
        ),
        "pre_concat_cleanup": "_optimize_transpose_pre_concat_nhwc_chains",
        "channel_shuffle_gather_cluster": (
            "_run_channel_shuffle_gather_layout_pass_cluster"
        ),
    }


def test_new_attention_runner_executes_the_same_flattened_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_context = _context()
    probe_steps = (
        *build_layout_recovery_invocations(probe_context),
        *build_attention_recovery_invocations(probe_context)[1:],
    )
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    context_callbacks = {
        probe_context.boundary_batchmatmul_unary_cluster,
        probe_context.pre_concat_cleanup,
        probe_context.channel_shuffle_gather_cluster,
    }
    for step in probe_steps:
        if step.callback in context_callbacks:
            continue
        module_name = next(
            name
            for name, value in vars(layout_recovery_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            layout_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    context = LayoutRecoveryContext(
        pass_context=probe_context.pass_context,
        boundary_batchmatmul_unary_cluster=recorder(LAYOUT_RECOVERY_PASS_IDS[1]),
        pre_concat_cleanup=recorder(LAYOUT_RECOVERY_PASS_IDS[12]),
        channel_shuffle_gather_cluster=recorder(LAYOUT_RECOVERY_PASS_IDS[18]),
    )

    run_layout_reshape_attention_recovery_prefix(context)

    assert events == [
        *LAYOUT_RECOVERY_PASS_IDS,
        *ATTENTION_RECOVERY_PASS_IDS[1:],
    ]


@pytest.mark.xfail(
    strict=True,
    reason="the ordered layout-recovery prefix result is discarded",
)
def test_layout_recovery_prefix_propagates_direct_and_nested_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary_results = (
        {"boundary_slot": 0},
        {"boundary_slot": 1},
    )
    pre_concat_result = {"pre_concat_slot": 12}
    channel_shuffle_results = (
        {"channel_shuffle_slot": 0},
        {"channel_shuffle_slot": 1},
        {"channel_shuffle_slot": 2},
    )
    expected_results: tuple[Any, ...] = tuple(
        boundary_results
        if index == 1
        else pre_concat_result
        if index == 12
        else channel_shuffle_results
        if index == 18
        else {"slot": index}
        for index in range(len(LAYOUT_RECOVERY_PASS_IDS))
    )

    def result_callback(index: int):
        def result(*args: Any, **kwargs: Any) -> Any:
            value = expected_results[index]
            return tuple(value) if isinstance(value, tuple) else dict(value)

        return result

    model_ir = ModelIR("layout_recovery_prefix_results_test")
    context = LayoutRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        boundary_batchmatmul_unary_cluster=result_callback(1),
        pre_concat_cleanup=result_callback(12),
        channel_shuffle_gather_cluster=result_callback(18),
    )
    probe_steps = build_layout_recovery_invocations(context)
    injected_callbacks = {
        context.boundary_batchmatmul_unary_cluster,
        context.pre_concat_cleanup,
        context.channel_shuffle_gather_cluster,
    }
    for index, step in enumerate(probe_steps):
        if step.callback in injected_callbacks:
            continue
        module_name = next(
            name
            for name, value in vars(layout_recovery_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            layout_recovery_orchestration,
            module_name,
            result_callback(index),
        )

    assert (
        layout_recovery_orchestration.run_layout_recovery_prefix(context)
        == expected_results
    )

    attention_steps = build_attention_recovery_invocations(context)
    assert attention_steps[0].pass_id == LAYOUT_PREFIX
    assert attention_steps[0].callback is (
        layout_recovery_orchestration.run_layout_recovery_prefix
    )
    assert attention_steps[0].args == (context,)
    assert attention_steps[0].keyword_args == ()
    assert attention_steps[0].run() == expected_results

    orchestration_functions = {
        node.name: node
        for node in ast.parse(
            ORCHESTRATION_PATH.read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef)
    }
    runner = orchestration_functions["run_layout_recovery_prefix"]
    assert ast.unparse(runner.returns) == "Tuple[Any, ...]"
    assert isinstance(runner.body[-1], ast.Return)

    lowerer, helper = _lowerer_and_helper(LAYOUT_PREFIX)
    assert ast.unparse(helper.returns) == "Tuple[Any, ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)

    direct_results: list[tuple[list[ast.stmt], int]] = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if _direct_call_name(candidate) == LAYOUT_PREFIX:
                direct_results.append((statement.body, index))
    assert len(direct_results) == 1
    body, index = direct_results[0]
    assert _single_target(body[index]) == LAYOUT_RESULT_TARGET
    assert isinstance(body[index].value, ast.Call)
    assert body[index].value.args == []
    assert body[index].value.keywords == []
    assert _direct_call_name(body[index - 1]) == (
        "_run_qlinear_mean_concat_recovery_sequence"
    )
    assert _single_target(body[index + 1]) == (
        "_layout_pass_set_2_preadd_mean_attention_results"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == LAYOUT_RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
