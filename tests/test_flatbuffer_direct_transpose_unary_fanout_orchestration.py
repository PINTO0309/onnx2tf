from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import transpose_unary_fanout_orchestration
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    ATTENTION_GATE_QDQ_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.transpose_unary_fanout_orchestration import (
    TRANSPOSE_UNARY_FANOUT_PASS_IDS,
    TransposeUnaryFanoutContext,
    active_transpose_unary_fanout_pass_ids,
    build_transpose_unary_fanout_invocations,
    run_transpose_unary_fanout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "transpose_unary_fanout_orchestration.py"
)
TRANSPOSE_UNARY_FANOUT = "_run_transpose_unary_fanout_layout_pass_cluster"
RESULT_TARGET = "_layout_pass_set_1_transpose_unary_fanout_results"
DEFAULT_PASS_IDS = TRANSPOSE_UNARY_FANOUT_PASS_IDS[1:]
POST_QDQ_PASS_IDS = (
    TRANSPOSE_UNARY_FANOUT_PASS_IDS[0],
    *TRANSPOSE_UNARY_FANOUT_PASS_IDS[2:],
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
        if isinstance(node, ast.FunctionDef) and node.name == TRANSPOSE_UNARY_FANOUT
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


def _context() -> TransposeUnaryFanoutContext:
    model_ir = ModelIR("transpose_unary_fanout_test")
    return TransposeUnaryFanoutContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: transpose_unary_fanout_orchestration.RecoveryInvocation,
    context: TransposeUnaryFanoutContext,
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


def test_transpose_unary_fanout_signature_and_delegate_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == [
        "include_layout_transpose",
        "include_unary_passthrough",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
    ]
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

    statement = helper.body[0]
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_transpose_unary_fanout"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "transpose_unary_fanout_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {
        "include_layout_transpose": "include_layout_transpose",
        "include_unary_passthrough": "include_unary_passthrough",
    }


def test_transpose_unary_fanout_context_is_explicit() -> None:
    lowerer, _ = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "transpose_unary_fanout_context"
            for target in statement.targets
        )
    )

    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize(
    ("include_layout_transpose", "include_unary_passthrough", "expected_ids"),
    [
        (False, True, DEFAULT_PASS_IDS),
        (True, False, POST_QDQ_PASS_IDS),
    ],
)
def test_transpose_unary_fanout_preserves_both_cleanup_contracts(
    include_layout_transpose: bool,
    include_unary_passthrough: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    invocations = build_transpose_unary_fanout_invocations(
        context,
        include_layout_transpose=include_layout_transpose,
        include_unary_passthrough=include_unary_passthrough,
    )

    assert (
        active_transpose_unary_fanout_pass_ids(
            include_layout_transpose=include_layout_transpose,
            include_unary_passthrough=include_unary_passthrough,
        )
        == expected_ids
    )
    assert tuple(step.pass_id for step in invocations) == expected_ids
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
    } == {pass_id: expected_contract for pass_id in expected_ids}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_transpose_unary_fanout_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
            include_unary_passthrough=include_unary_passthrough,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize(
    ("include_layout_transpose", "include_unary_passthrough", "expected_ids"),
    [
        (False, True, DEFAULT_PASS_IDS),
        (True, False, POST_QDQ_PASS_IDS),
    ],
)
def test_transpose_unary_fanout_runner_preserves_both_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    include_unary_passthrough: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for pass_id in expected_ids:
        monkeypatch.setattr(
            transpose_unary_fanout_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_transpose_unary_fanout(
        context,
        include_layout_transpose=include_layout_transpose,
        include_unary_passthrough=include_unary_passthrough,
    )

    assert events == list(expected_ids)


def test_transpose_unary_fanout_preserves_both_invocation_variants() -> None:
    lowerer, _ = _lowerer_and_helper()
    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TRANSPOSE_UNARY_FANOUT
    ]

    assert len(direct_invocations) == 1
    assert direct_invocations[0].args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in direct_invocations[0].keywords
    } == {
        "include_layout_transpose": True,
        "include_unary_passthrough": False,
    }

    attention_context = next(
        statement.value
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "attention_recovery_context"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Call)
    )
    callback_keyword = next(
        keyword
        for keyword in attention_context.keywords
        if keyword.arg == "transpose_unary_fanout_cluster"
    )
    assert isinstance(callback_keyword.value, ast.Name)
    assert callback_keyword.value.id == TRANSPOSE_UNARY_FANOUT


def test_transpose_unary_fanout_preserves_direct_and_callback_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    parent = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == TRANSPOSE_UNARY_FANOUT
            for node in ast.walk(statement)
        )
    )
    direct_index = next(
        index
        for index, statement in enumerate(parent.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TRANSPOSE_UNARY_FANOUT
    )
    previous = parent.body[direct_index - 1]
    following = parent.body[direct_index + 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == (
        "_layout_pass_set_1_final_attention_quantized_suffix_results"
    )
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == "_run_layout_attention_quantized_recovery_suffix"
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_layout_pass_set_1_final_safe_binary_results"
    )
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == "_run_safe_binary_bridge_recovery_sequence"

    callback_index = ATTENTION_GATE_QDQ_PASS_IDS.index(TRANSPOSE_UNARY_FANOUT)
    assert ATTENTION_GATE_QDQ_PASS_IDS[callback_index - 1] == (
        "_optimize_transposeconv_output_channel1_terminal_transpose_chains"
    )
    assert ATTENTION_GATE_QDQ_PASS_IDS[callback_index + 1] == (
        "_optimize_transpose_dequant_relu_quantize_bridges"
    )


@pytest.mark.xfail(
    strict=True,
    reason="the direct Transpose/unary-fanout result is discarded",
)
def test_transpose_unary_fanout_returns_both_variants_and_retains_direct_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    variants = (
        (False, True, DEFAULT_PASS_IDS),
        (True, False, POST_QDQ_PASS_IDS),
    )
    for include_layout, include_unary, expected_ids in variants:
        expected_results = tuple(
            {"slot": index} for index in range(len(expected_ids))
        )
        for index, pass_id in enumerate(expected_ids):
            monkeypatch.setattr(
                transpose_unary_fanout_orchestration,
                pass_id,
                lambda *args, _index=index, **kwargs: {"slot": _index},
            )
        assert run_transpose_unary_fanout(
            context,
            include_layout_transpose=include_layout,
            include_unary_passthrough=include_unary,
        ) == expected_results

    orchestration_functions = {
        node.name: node
        for node in ast.parse(
            ORCHESTRATION_PATH.read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef)
    }
    runner = orchestration_functions["run_transpose_unary_fanout"]
    assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
    assert isinstance(runner.body[-1], ast.Return)

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)

    direct_results = [
        statement
        for statement in ast.walk(lowerer)
        if _direct_call_name(statement) == TRANSPOSE_UNARY_FANOUT
    ]
    assert len(direct_results) == 1
    result = direct_results[0]
    assert _single_target(result) == RESULT_TARGET
    call = result.value
    assert isinstance(call, ast.Call)
    assert call.args == []
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
    } == {
        "include_layout_transpose": "True",
        "include_unary_passthrough": "False",
    }
    parent = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If) and result in statement.body
    )
    result_index = parent.body.index(result)
    assert _single_target(parent.body[result_index - 1]) == (
        "_layout_pass_set_1_final_attention_quantized_suffix_results"
    )
    assert _single_target(parent.body[result_index + 1]) == (
        "_layout_pass_set_1_final_safe_binary_results"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )

    attention_context = next(
        statement.value
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "attention_recovery_context"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Call)
    )
    callback_keyword = next(
        keyword
        for keyword in attention_context.keywords
        if keyword.arg == "transpose_unary_fanout_cluster"
    )
    assert isinstance(callback_keyword.value, ast.Name)
    assert callback_keyword.value.id == TRANSPOSE_UNARY_FANOUT


def test_transpose_unary_fanout_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "transpose_unary_fanout_orchestration.py"
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
