from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    optimize_swish_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LAYOUT_RECOVERY_PASS_IDS,
    LayoutRecoveryContext,
    build_layout_recovery_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "activation_passthrough_layout.py"
)
WRAPPER = "_optimize_swish_transpose_passthrough_chains"
DISPATCH = "_optimize_swish_transpose_passthrough_chains_pass"
RESULT_TARGET = "_late_swish_transpose_passthrough_stats"
PREDECESSOR_TARGET = "_late_dequant_unary_fanout_results"
SUCCESSOR = "run_late_conv1d_decoder_layout_cleanup"
RESULT_SCHEMA = {"rewritten_swish_transpose_passthrough_chains": 0}


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _direct_location() -> tuple[ast.FunctionDef, int]:
    lowerer = _lowerer()
    return lowerer, next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == WRAPPER
    )


def test_late_swish_schema_wrapper_and_cleanup_are_explicit() -> None:
    lowerer_functions = _functions(LOWERER_PATH)
    wrapper = lowerer_functions[WRAPPER]
    assert [argument.arg for argument in wrapper.args.args] == ["model_ir"]
    assert [argument.arg for argument in wrapper.args.kwonlyargs] == [
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    ]
    assert [ast.unparse(value) for value in wrapper.args.kw_defaults] == [
        "None",
        "None",
        "None",
        "None",
    ]
    statement = wrapper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == DISPATCH
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
        "max_rewrites": "max_rewrites",
        "candidate": "candidate",
    }

    owner_functions = _functions(OWNER_PATH)
    apply_plan = owner_functions["_apply_plan"]
    assert any(
        _call_name(node) == "_prune_unused_tensors"
        for node in ast.walk(apply_plan)
        if isinstance(node, ast.Expr)
    )
    assert optimize_swish_transpose_passthrough_chains(
        ModelIR("late_swish_schema")
    ) == RESULT_SCHEMA


def test_late_swish_direct_and_layout_recovery_route_are_explicit() -> None:
    lowerer, index = _direct_location()
    invocation = lowerer.body[index]
    assert _single_target(invocation) == RESULT_TARGET
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"layout_state": "session.layout_state"}
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _call_name(lowerer.body[index + 1]) == SUCCESSOR
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == WRAPPER
        for node in ast.walk(lowerer)
    ) == 1

    model_ir = ModelIR("late_swish_layout_recovery")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    recovery_context = LayoutRecoveryContext(
        pass_context=pass_context,
        boundary_batchmatmul_unary_cluster=lambda: (),
        pre_concat_cleanup=lambda: {},
        channel_shuffle_gather_cluster=lambda: (),
    )
    nested = build_layout_recovery_invocations(recovery_context)[5]
    assert LAYOUT_RECOVERY_PASS_IDS[5] == WRAPPER
    assert nested.callback is optimize_swish_transpose_passthrough_chains
    assert nested.args == (model_ir,)
    assert dict(nested.keyword_args) == {
        "layout_state": pass_context.layout_state,
    }


def test_late_swish_direct_result_is_retained_for_observation() -> None:
    lowerer, index = _direct_location()
    assert _single_target(lowerer.body[index]) == RESULT_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
