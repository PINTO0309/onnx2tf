from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    ATTENTION_GATE_QDQ_PASS_IDS,
    AttentionRecoveryContext,
    build_attention_gate_qdq_invocations,
)
from onnx2tf.tflite_builder.passes.sinet_mix_attention_layout import (
    optimize_sinet_mix_attention_double_logistic_nhwc_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "sinet_mix_attention_layout.py"
)
DIRECT_OWNER = "_optimize_sinet_mix_attention_double_logistic_nhwc_chains"
PUBLIC_OWNER = "optimize_sinet_mix_attention_double_logistic_nhwc_chains"
RESULT_TARGET = "_post_sinet_mix_attention_stats"
PREDECESSOR_PHASE_ID = "cleanup.post_sinet.split_conv_concat_bridge"
SUCCESSOR = "run_mixed_attention_layout_cleanup"
RESULT_KEY = "optimized_sinet_mix_attention_double_logistic_nhwc_chains"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
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
        return call.args[1]
    return call


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


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _direct_location() -> tuple[ast.FunctionDef, int]:
    lowerer = _lowerer()
    return lowerer, next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == DIRECT_OWNER
    )


def _attention_context() -> AttentionRecoveryContext:
    model_ir = ModelIR("sinet_mix_attention_nested")
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


def test_sinet_mix_attention_schema_cleanup_and_route_are_explicit() -> None:
    owner = _functions(OWNER_PATH)[PUBLIC_OWNER]
    assert [argument.arg for argument in owner.args.kwonlyargs] == [
        "graph_index",
        "layout_state",
        "max_rewrites",
        "candidate",
    ]
    assert [ast.unparse(value) for value in owner.args.kw_defaults] == [
        "None",
        "None",
        "32",
        "None",
    ]
    guarded_cleanup = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "rewritten > 0"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(guarded_cleanup) == 1
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == "{_STATS_KEY: int(rewritten)}"
    assert optimize_sinet_mix_attention_double_logistic_nhwc_chains(
        ModelIR("sinet_mix_attention_schema")
    ) == {RESULT_KEY: 0}

    wrapper = _functions(LOWERER_PATH)[DIRECT_OWNER]
    assert len(wrapper.body) == 2
    wrapper_return = wrapper.body[-1]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{DIRECT_OWNER}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in wrapper_return.value.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
        "max_rewrites": "max_rewrites",
        "candidate": "candidate",
    }

    context = _attention_context()
    invocation = build_attention_gate_qdq_invocations(context)[1]
    assert ATTENTION_GATE_QDQ_PASS_IDS[1] == DIRECT_OWNER
    assert invocation.callback is (
        optimize_sinet_mix_attention_double_logistic_nhwc_chains
    )
    assert invocation.args == (context.pass_context.model_ir,)
    assert dict(invocation.keyword_args) == {
        "layout_state": context.pass_context.layout_state,
    }


def test_sinet_mix_attention_direct_phase_boundary_is_explicit() -> None:
    lowerer, index = _direct_location()
    invocation = lowerer.body[index]
    assert _phase_id(invocation) == "cleanup.post_sinet.mix_attention"
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"layout_state": "session.layout_state"}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _call_name(lowerer.body[index + 1]) == SUCCESSOR
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == DIRECT_OWNER
        for node in ast.walk(lowerer)
    ) == 1


def test_sinet_mix_attention_direct_result_uses_phase_store() -> None:
    lowerer, index = _direct_location()
    assert _phase_id(lowerer.body[index]) == "cleanup.post_sinet.mix_attention"
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
