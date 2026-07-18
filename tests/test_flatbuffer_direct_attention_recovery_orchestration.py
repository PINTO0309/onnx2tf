from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import attention_recovery_orchestration
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    ATTENTION_GATE_QDQ_PASS_IDS,
    PREADD_MEAN_ATTENTION_PASS_IDS,
    AttentionRecoveryContext,
    build_attention_gate_qdq_invocations,
    build_preadd_mean_attention_invocations,
    run_attention_gate_qdq_recovery,
    run_preadd_mean_attention_recovery,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "attention_recovery_orchestration.py"
)
PREADD_MEAN_ATTENTION = "_run_preadd_mean_attention_recovery_sequence"
ATTENTION_GATE_QDQ = "_run_attention_gate_qdq_recovery_sequence"
ATTENTION_GATE_RESULT_TARGETS = (
    "_layout_pass_set_1_attention_gate_qdq_results",
    "_layout_pass_set_2_attention_gate_qdq_results",
)


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


def _context() -> AttentionRecoveryContext:
    model_ir = ModelIR("attention_recovery_orchestration_test")
    return AttentionRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        mean_attention_cluster=lambda: None,
        gate_layout_cluster=lambda: None,
        transpose_unary_fanout_cluster=lambda: None,
    )


def _normalize_new_contract(
    invocation: attention_recovery_orchestration.RecoveryInvocation,
    context: AttentionRecoveryContext,
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


def test_recovery_orchestration_modules_do_not_import_the_lowerer() -> None:
    pass_root = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
    for module_name in (
        "attention_recovery_orchestration.py",
        "layout_recovery_orchestration.py",
        "recovery_orchestration.py",
    ):
        tree = ast.parse((pass_root / module_name).read_text(encoding="utf-8"))
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


def test_attention_recovery_sequences_are_straight_line_closures() -> None:
    expected_lines = {
        PREADD_MEAN_ATTENTION: 2,
        ATTENTION_GATE_QDQ: 2,
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
        assert loaded_data_names == {"attention_recovery_context"}


def test_preadd_mean_attention_preserves_exact_order_and_arguments() -> None:
    context = _context()
    invocations = build_preadd_mean_attention_invocations(context)
    contracts = {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    }

    assert tuple(step.pass_id for step in invocations) == PREADD_MEAN_ATTENTION_PASS_IDS
    assert contracts == {
        "_optimize_transpose_pre_add_nhwc_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains": (
            ("model_ir",),
            {},
        ),
        "_run_mean_attention_layout_pass_cluster": ((), {}),
    }


def test_attention_gate_qdq_preserves_exact_order_and_arguments() -> None:
    context = _context()
    invocations = build_attention_gate_qdq_invocations(context)
    contracts = {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    }

    assert tuple(step.pass_id for step in invocations) == ATTENTION_GATE_QDQ_PASS_IDS
    assert contracts == {
        "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_sinet_mix_attention_double_logistic_nhwc_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_run_gate_layout_pass_cluster": ((), {}),
        "_optimize_transposeconv_output_nhwc_passthrough_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transposeconv_output_channel1_terminal_transpose_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_run_transpose_unary_fanout_layout_pass_cluster": ((), {}),
        "_optimize_transpose_dequant_relu_quantize_bridges": (
            ("model_ir",),
            {},
        ),
        "_optimize_transpose_dequant_hardsigmoid_quantize_bridges": (
            ("model_ir",),
            {},
        ),
        "run_trailing_output_transpose_cleanup": (
            ("model_ir",),
            {
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            },
        ),
        "_optimize_transpose_dequant_mul_add_prelu_quantize_bridges": (
            ("model_ir",),
            {},
        ),
    }


def test_attention_recovery_context_and_wrappers_are_explicit() -> None:
    lowerer, preadd_helper = _lowerer_and_helper(PREADD_MEAN_ATTENTION)
    _, attention_helper = _lowerer_and_helper(ATTENTION_GATE_QDQ)

    expected_wrappers = {
        PREADD_MEAN_ATTENTION: (
            preadd_helper,
            "run_preadd_mean_attention_recovery",
        ),
        ATTENTION_GATE_QDQ: (
            attention_helper,
            "run_attention_gate_qdq_recovery",
        ),
    }
    for helper_name, (helper, runner_name) in expected_wrappers.items():
        assert len(helper.body) == 1
        statement = helper.body[0]
        assert isinstance(statement, ast.Expr)
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id == runner_name
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "attention_recovery_context"
        assert call.keywords == []
        assert helper.name == helper_name

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "attention_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Call)
    assert isinstance(context_assignment.value.func, ast.Name)
    assert context_assignment.value.func.id == "AttentionRecoveryContext"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in context_assignment.value.keywords
    } == {
        "pass_context": "session.model_ir_pass_context",
        "mean_attention_cluster": "_run_mean_attention_layout_pass_cluster",
        "gate_layout_cluster": "_run_gate_layout_pass_cluster",
        "transpose_unary_fanout_cluster": (
            "_run_transpose_unary_fanout_layout_pass_cluster"
        ),
    }


def test_attention_recovery_invocation_boundaries_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper(PREADD_MEAN_ATTENTION)
    expected_counts = {
        PREADD_MEAN_ATTENTION: 2,
        ATTENTION_GATE_QDQ: 3,
    }

    for helper_name, expected_count in expected_counts.items():
        invocations = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
        ]
        orchestrated_count = LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS.count(
            helper_name
        )
        assert len(invocations) + orchestrated_count == expected_count
        assert all(call.args == [] for call in invocations)
        assert all(call.keywords == [] for call in invocations)

    attention_index = LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS.index(
        ATTENTION_GATE_QDQ
    )
    assert list(
        LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[
            attention_index - 1 : attention_index + 2
        ]
    ) == [
        "_run_mean_attention_layout_pass_cluster",
        ATTENTION_GATE_QDQ,
        "_run_duplicate_quantized_prelu_pass_cluster",
    ]


@pytest.mark.parametrize(
    ("build_invocations", "run_phase", "expected_ids"),
    [
        (
            build_preadd_mean_attention_invocations,
            run_preadd_mean_attention_recovery,
            PREADD_MEAN_ATTENTION_PASS_IDS,
        ),
        (
            build_attention_gate_qdq_invocations,
            run_attention_gate_qdq_recovery,
            ATTENTION_GATE_QDQ_PASS_IDS,
        ),
    ],
)
def test_attention_recovery_runners_preserve_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
    build_invocations: Any,
    run_phase: Any,
    expected_ids: tuple[str, ...],
) -> None:
    probe_context = _context()
    probe_steps = build_invocations(probe_context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    context_callbacks = {
        probe_context.mean_attention_cluster,
        probe_context.gate_layout_cluster,
        probe_context.transpose_unary_fanout_cluster,
    }
    callback_by_id = {}
    for step in probe_steps:
        if step.callback in context_callbacks:
            callback_by_id[step.pass_id] = recorder(step.pass_id)
            continue
        module_name = next(
            name
            for name, value in vars(attention_recovery_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            attention_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    context = AttentionRecoveryContext(
        pass_context=probe_context.pass_context,
        mean_attention_cluster=callback_by_id.get(
            "_run_mean_attention_layout_pass_cluster",
            lambda: None,
        ),
        gate_layout_cluster=callback_by_id.get(
            "_run_gate_layout_pass_cluster",
            lambda: None,
        ),
        transpose_unary_fanout_cluster=callback_by_id.get(
            "_run_transpose_unary_fanout_layout_pass_cluster",
            lambda: None,
        ),
    )

    run_phase(context)

    assert events == list(expected_ids)


@pytest.mark.xfail(
    strict=True,
    reason="both ordered attention-gate/QDQ results are discarded",
)
def test_attention_gate_qdq_propagates_nested_results_to_both_direct_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unary_fanout_results = (
        {"unary_slot": 0},
        {"unary_slot": 1},
        {"unary_slot": 2},
    )
    expected_results: tuple[Any, ...] = tuple(
        unary_fanout_results if index == 5 else {"slot": index}
        for index in range(len(ATTENTION_GATE_QDQ_PASS_IDS))
    )

    def result_callback(index: int):
        def result(*args: Any, **kwargs: Any) -> Any:
            value = expected_results[index]
            return tuple(value) if isinstance(value, tuple) else dict(value)

        return result

    model_ir = ModelIR("attention_gate_qdq_results_test")
    context = AttentionRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        mean_attention_cluster=lambda: None,
        gate_layout_cluster=result_callback(2),
        transpose_unary_fanout_cluster=result_callback(5),
    )
    probe_steps = build_attention_gate_qdq_invocations(context)
    injected_callbacks = {
        context.gate_layout_cluster,
        context.transpose_unary_fanout_cluster,
    }
    for index, step in enumerate(probe_steps):
        if step.callback in injected_callbacks:
            continue
        module_name = next(
            name
            for name, value in vars(attention_recovery_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            attention_recovery_orchestration,
            module_name,
            result_callback(index),
        )

    assert run_attention_gate_qdq_recovery(context) == expected_results

    orchestration_functions = {
        node.name: node
        for node in ast.parse(
            ORCHESTRATION_PATH.read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef)
    }
    runner = orchestration_functions["run_attention_gate_qdq_recovery"]
    assert ast.unparse(runner.returns) == "Tuple[Any, ...]"
    assert isinstance(runner.body[-1], ast.Return)

    lowerer, helper = _lowerer_and_helper(ATTENTION_GATE_QDQ)
    assert ast.unparse(helper.returns) == "Tuple[Any, ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)

    direct_results: list[tuple[list[ast.stmt], int]] = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if _direct_call_name(candidate) == ATTENTION_GATE_QDQ:
                direct_results.append((statement.body, index))
    assert len(direct_results) == 2
    assert tuple(
        _single_target(body[index]) for body, index in direct_results
    ) == ATTENTION_GATE_RESULT_TARGETS
    assert all(
        body[index].value.args == [] and body[index].value.keywords == []
        for body, index in direct_results
        if isinstance(body[index].value, ast.Call)
    )
    assert _single_target(direct_results[0][0][direct_results[0][1] - 1]) == (
        "_layout_pass_set_1_mean_attention_results"
    )
    assert _direct_call_name(
        direct_results[1][0][direct_results[1][1] - 1]
    ) == "_run_preadd_mean_attention_recovery_sequence"
    assert tuple(
        _direct_call_name(body[index + 1]) for body, index in direct_results
    ) == (
        "run_quantized_prelu_cleanup",
        "_optimize_dequant_transposeconv_quantize_chains",
    )
    for target in ATTENTION_GATE_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )

    nested_index = LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS.index(
        ATTENTION_GATE_QDQ
    )
    assert LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[
        nested_index - 1 : nested_index + 2
    ] == (
        "_run_mean_attention_layout_pass_cluster",
        ATTENTION_GATE_QDQ,
        "_run_duplicate_quantized_prelu_pass_cluster",
    )


def test_shared_recovery_runner_rejects_id_drift_before_execution() -> None:
    events: list[str] = []
    invocations = (
        RecoveryInvocation(
            pass_id="actual_pass",
            callback=lambda: events.append("executed"),
        ),
    )

    with pytest.raises(RuntimeError, match="test phase pass IDs diverged"):
        run_recovery_invocations(
            invocations,
            expected_pass_ids=("expected_pass",),
            phase_name="test phase",
        )

    assert events == []


def test_shared_recovery_runner_returns_ordered_callback_results() -> None:
    invocations = (
        RecoveryInvocation(
            pass_id="first",
            callback=lambda: {"first_changes": 1},
        ),
        RecoveryInvocation(
            pass_id="second",
            callback=lambda: {"second_changes": 2},
        ),
    )

    assert run_recovery_invocations(
        invocations,
        expected_pass_ids=("first", "second"),
        phase_name="result-preserving phase",
    ) == (
        {"first_changes": 1},
        {"second_changes": 2},
    )
