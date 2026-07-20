from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import quantized_recovery_orchestration
from onnx2tf.tflite_builder.passes.quantized_recovery_orchestration import (
    QUANTIZED_ACTIVATION_BINARY_PASS_IDS,
    SAFE_BINARY_RECOVERY_PASS_IDS,
    QuantizedRecoveryContext,
    build_quantized_activation_binary_invocations,
    build_safe_binary_recovery_invocations,
    run_quantized_activation_binary_recovery,
    run_safe_binary_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "quantized_recovery_orchestration.py"
)
LAYOUT_PASS_SET_1_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_pass_set_1_attention_quantized_safe_binary_orchestration.py"
)
LAYOUT_PASS_SET_1_FINAL_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_pass_set_1_final_quantized_unary_safe_orchestration.py"
)
SAFE_BINARY = "_run_safe_binary_bridge_recovery_sequence"
QUANTIZED_ACTIVATION_BINARY = (
    "_run_quantized_activation_binary_bridge_recovery_sequence"
)
RESULT_TARGETS = (
    "_layout_pass_set_1_quantized_activation_binary_results",
    "_layout_pass_set_2_quantized_activation_binary_results",
)
SAFE_RESULT_TARGETS = (
    "_layout_pass_set_1_final_safe_binary_results",
)
REMOVED_LAYOUT_PASS_SET_1_SAFE_RESULT_TARGET = (
    "_layout_pass_set_1_safe_binary_results"
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


def _layout_pass_set_1_owner_calls(child_owner: str) -> list[ast.Call]:
    tree = ast.parse(
        LAYOUT_PASS_SET_1_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "run_layout_pass_set_1_attention_quantized_safe_binary_cleanup"
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == child_owner
    ]


def _layout_pass_set_1_final_owner_calls(child_owner: str) -> list[ast.Call]:
    tree = ast.parse(
        LAYOUT_PASS_SET_1_FINAL_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "run_layout_pass_set_1_final_quantized_unary_safe_cleanup"
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == child_owner
    ]


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
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        call = call.args[1]
    function = call.func
    return function.id if isinstance(function, ast.Name) else None


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _context() -> QuantizedRecoveryContext:
    model_ir = ModelIR("quantized_recovery_orchestration_test")
    return QuantizedRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: quantized_recovery_orchestration.RecoveryInvocation,
    context: QuantizedRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context:
            return "context"
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_quantized_recovery_sequences_are_straight_line_closures() -> None:
    expected_lines = {
        SAFE_BINARY: 2,
        QUANTIZED_ACTIVATION_BINARY: 5,
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
        assert loaded_data_names == {"quantized_recovery_context"}


def test_safe_binary_recovery_preserves_exact_arguments() -> None:
    context = _context()
    invocations = build_safe_binary_recovery_invocations(context)

    assert tuple(step.pass_id for step in invocations) == SAFE_BINARY_RECOVERY_PASS_IDS
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        "_run_safe_binary_bridge_recovery_pass": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
    }


def test_quantized_activation_binary_preserves_exact_order_and_arguments() -> None:
    context = _context()
    invocations = build_quantized_activation_binary_invocations(context)
    contracts = {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    }

    assert (
        tuple(step.pass_id for step in invocations)
        == QUANTIZED_ACTIVATION_BINARY_PASS_IDS
    )
    assert contracts == {
        "_optimize_dequant_hardsigmoid_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_maxpool_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_softmax_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_logistic_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_canonicalize_softmax_transpose_chains": (("model_ir",), {}),
        SAFE_BINARY: (("context",), {}),
    }


def test_quantized_recovery_invocation_boundaries_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper(QUANTIZED_ACTIVATION_BINARY)
    expected_counts = {
        SAFE_BINARY: 3,
        QUANTIZED_ACTIVATION_BINARY: 2,
    }

    for helper_name, expected_count in expected_counts.items():
        invocations = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
        ]
        orchestrated_count = 0
        composite_count = 0
        if helper_name == SAFE_BINARY:
            orchestrated_count = QUANTIZED_ACTIVATION_BINARY_PASS_IDS.count(helper_name)
            composite_count = len(
                _layout_pass_set_1_owner_calls("run_safe_binary_recovery")
            ) + len(
                _layout_pass_set_1_final_owner_calls(
                    "run_safe_binary_recovery"
                )
            )
        assert (
            len(invocations) + orchestrated_count + composite_count
            == expected_count
        )
        assert all(call.args == [] for call in invocations)
        assert all(call.keywords == [] for call in invocations)


def test_quantized_recovery_context_and_wrappers_are_explicit() -> None:
    lowerer, safe_helper = _lowerer_and_helper(SAFE_BINARY)
    _, quantized_helper = _lowerer_and_helper(QUANTIZED_ACTIVATION_BINARY)
    expected_wrappers = {
        SAFE_BINARY: (safe_helper, "run_safe_binary_recovery", ast.Return),
        QUANTIZED_ACTIVATION_BINARY: (
            quantized_helper,
            "run_quantized_activation_binary_recovery",
            ast.Return,
        ),
    }
    for helper_name, (
        helper,
        runner_name,
        statement_type,
    ) in expected_wrappers.items():
        assert len(helper.body) == 1
        statement = helper.body[0]
        assert isinstance(statement, statement_type)
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id == runner_name
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "quantized_recovery_context"
        assert call.keywords == []
        assert helper.name == helper_name

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "quantized_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize(
    ("build_invocations", "run_phase", "expected_ids"),
    [
        (
            build_safe_binary_recovery_invocations,
            run_safe_binary_recovery,
            SAFE_BINARY_RECOVERY_PASS_IDS,
        ),
        (
            build_quantized_activation_binary_invocations,
            run_quantized_activation_binary_recovery,
            QUANTIZED_ACTIVATION_BINARY_PASS_IDS,
        ),
    ],
)
def test_quantized_recovery_runners_preserve_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
    build_invocations: Any,
    run_phase: Any,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    probe_steps = build_invocations(context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for step in probe_steps:
        module_name = next(
            name
            for name, value in vars(quantized_recovery_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            quantized_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_phase(context)

    assert events == list(expected_ids)


def test_quantized_activation_binary_child_schemas_are_explicit() -> None:
    context = _context()
    quantized_invocations = build_quantized_activation_binary_invocations(context)
    safe_invocations = build_safe_binary_recovery_invocations(context)

    assert tuple(invocation.run() for invocation in quantized_invocations[:5]) == (
        {"folded_dequant_hardsigmoid_quantize_chains": 0},
        {"folded_dequant_maxpool_quantize_chains": 0},
        {"folded_dequant_softmax_quantize_chains": 0},
        {"folded_dequant_logistic_quantize_chains": 0},
        {"canonicalized_softmax_transpose_chains": 0},
    )
    assert tuple(invocation.run() for invocation in safe_invocations) == (
        {
            "rewritten_transpose_binary_symmetric_legacy_only_bridges_safe": 0,
            "rewritten_transpose_binary_single_post_bridges_safe": 0,
            "rewritten_transpose_binary_mixed_fanout_bridges_safe": 0,
            "rewritten_transpose_binary_asymmetric_fanout_bridges": 0,
            "rewritten_transpose_binary_full_post_fanout_bridges": 0,
        },
    )


def test_quantized_activation_binary_zero_counter_can_hide_cleanup() -> None:
    model_ir = ModelIR("quantized_activation_binary_zero_prune")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["x"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["unused"] = TensorIR(
        name="unused",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    context = QuantizedRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )

    stats = build_quantized_activation_binary_invocations(context)[0].run()

    assert stats == {"folded_dequant_hardsigmoid_quantize_chains": 0}
    assert "unused" not in model_ir.tensors


def test_quantized_activation_binary_propagates_nested_results_to_both_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    safe_result = {
        "rewritten_transpose_binary_symmetric_legacy_only_bridges_safe": 5,
    }
    monkeypatch.setattr(
        quantized_recovery_orchestration,
        "run_safe_binary_bridge_recovery",
        lambda *args, **kwargs: dict(safe_result),
    )
    assert run_safe_binary_recovery(context) == (safe_result,)

    first_five_results = tuple(
        {"slot": index} for index in range(5)
    )
    for pass_id, expected_result in zip(
        QUANTIZED_ACTIVATION_BINARY_PASS_IDS[:5],
        first_five_results,
        strict=True,
    ):
        monkeypatch.setattr(
            quantized_recovery_orchestration,
            pass_id,
            lambda *args, _result=expected_result, **kwargs: dict(_result),
        )
    safe_results = (safe_result,)
    monkeypatch.setattr(
        quantized_recovery_orchestration,
        "run_safe_binary_recovery",
        lambda *args, **kwargs: safe_results,
    )
    expected_results = (*first_five_results, safe_results)
    assert run_quantized_activation_binary_recovery(context) == expected_results

    orchestration_functions = {
        node.name: node
        for node in ast.parse(
            ORCHESTRATION_PATH.read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef)
    }
    for runner_name in (
        "run_safe_binary_recovery",
        "run_quantized_activation_binary_recovery",
    ):
        runner = orchestration_functions[runner_name]
        assert ast.unparse(runner.returns) == "Tuple[Any, ...]"
        assert len(runner.body) == 1
        assert isinstance(runner.body[0], ast.Return)

    lowerer, helper = _lowerer_and_helper(QUANTIZED_ACTIVATION_BINARY)
    assert ast.unparse(helper.returns) == "Tuple[Any, ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)

    production_results: list[tuple[list[ast.stmt], int]] = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if _direct_call_name(candidate) == QUANTIZED_ACTIVATION_BINARY:
                production_results.append((statement.body, index))
    assert len(production_results) == 2
    assert tuple(
        _single_target(body[index]) for body, index in production_results
    ) == RESULT_TARGETS
    assert tuple(
        _direct_call_name(body[index - 1])
        for body, index in production_results
    ) == (
        "run_quantized_reshape_cleanup",
        "_optimize_dequant_transposeconv_quantize_chains",
    )
    first_following = production_results[0][0][production_results[0][1] + 1]
    assert isinstance(first_following, ast.If)
    assert ast.unparse(first_following.test) == (
        "enable_transpose_binary_bridge_optimizations"
    )
    second_following = production_results[1][0][production_results[1][1] + 1]
    assert ast.unparse(second_following) == (
        "session.record_phase_result("
        "'cleanup.layout_pass_set_2.elementwise_concat_conv', "
        "_optimize_transpose_elementwise_concat_conv_nhwc_groups(model_ir, "
        "layout_state=session.layout_state))"
    )
    for target in RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


def test_safe_binary_helper_propagates_and_retains_all_routes() -> None:
    orchestration_functions = {
        node.name: node
        for node in ast.parse(
            ORCHESTRATION_PATH.read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef)
    }
    runner = orchestration_functions["run_safe_binary_recovery"]
    assert ast.unparse(runner.returns) == "Tuple[Any, ...]"
    assert len(runner.body) == 1
    assert isinstance(runner.body[0], ast.Return)

    lowerer, helper = _lowerer_and_helper(SAFE_BINARY)
    assert ast.unparse(helper.returns) == "Tuple[Any, ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    helper_call = helper.body[0].value
    assert isinstance(helper_call, ast.Call)
    assert isinstance(helper_call.func, ast.Name)
    assert helper_call.func.id == "run_safe_binary_recovery"
    assert [ast.unparse(argument) for argument in helper_call.args] == [
        "quantized_recovery_context"
    ]
    assert helper_call.keywords == []

    production_results: list[tuple[list[ast.stmt], int]] = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if _direct_call_name(candidate) == SAFE_BINARY:
                production_results.append((statement.body, index))
    assert production_results == []
    for target in (
        *SAFE_RESULT_TARGETS,
        REMOVED_LAYOUT_PASS_SET_1_SAFE_RESULT_TARGET,
    ):
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )

    owner_calls = _layout_pass_set_1_owner_calls("run_safe_binary_recovery")
    assert len(owner_calls) == 1
    assert [ast.unparse(argument) for argument in owner_calls[0].args] == [
        "context.pass_context"
    ]
    assert owner_calls[0].keywords == []
    final_owner_calls = _layout_pass_set_1_final_owner_calls(
        "run_safe_binary_recovery"
    )
    assert len(final_owner_calls) == 1
    assert [
        ast.unparse(argument) for argument in final_owner_calls[0].args
    ] == ["context.pass_context"]
    assert final_owner_calls[0].keywords == []


def test_quantized_recovery_module_does_not_import_the_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_recovery_orchestration.py"
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
