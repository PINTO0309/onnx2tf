from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    PREADD_MEAN_ATTENTION_PASS_IDS,
    AttentionRecoveryContext,
    build_preadd_mean_attention_invocations,
)
from onnx2tf.tflite_builder.passes.residual_affine_prelu_layout import (
    optimize_transpose_pre_add_mul_add_prelu_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.sinet_preadd_resize_recovery_orchestration import (
    SINET_PREADD_RESIZE_RECOVERY_PASS_IDS,
    build_sinet_preadd_resize_recovery_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "residual_affine_prelu_layout.py"
)
ORCHESTRATION_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_sinet_residual_affine_prelu_orchestration.py"
)
DIRECT_OWNER = "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains"
PUBLIC_OWNER = "optimize_transpose_pre_add_mul_add_prelu_nhwc_chains"
ORCHESTRATION_OWNER = "run_very_late_sinet_residual_affine_prelu_cleanup"
RESULT_TARGET = "_very_late_residual_affine_prelu_stats"
PREDECESSOR_PHASE_ID = "shape_topology.terminal.indexed_convergence"
PHASE_ID = "cleanup.very_late.residual_affine_prelu"
OWNER_EXPRESSION = (
    "run_very_late_sinet_residual_affine_prelu_cleanup("
    "sinet_terminal_layout_recovery_context)[1]"
)
SUCCESSOR = "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains"
RESULT_KEY = "optimized_transpose_pre_add_mul_add_prelu_nhwc_chains"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    call = statement.value if isinstance(statement.value, ast.Call) else None
    if (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        return call.args[1]
    return call


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


def _phase_location() -> tuple[ast.FunctionDef, int]:
    lowerer = _lowerer()
    return lowerer, next(
        index
        for index, statement in enumerate(lowerer.body)
        if _phase_id(statement) == PHASE_ID
    )


def _pass_context(name: str) -> ModelIRPassContext:
    model_ir = ModelIR(name)
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def test_residual_affine_prelu_schema_cleanup_and_routes_are_explicit() -> None:
    owner = _functions(OWNER_PATH)[PUBLIC_OWNER]
    assert isinstance(owner.body[-2], ast.Expr)
    assert _call_name(owner.body[-2]) == "_prune_unused_tensors"
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'optimized_transpose_pre_add_mul_add_prelu_nhwc_chains': int(optimized)}"
    )
    assert optimize_transpose_pre_add_mul_add_prelu_nhwc_chains(
        ModelIR("residual_affine_prelu_schema")
    ) == {RESULT_KEY: 0}

    wrapper = _functions(LOWERER_PATH)[DIRECT_OWNER]
    assert len(wrapper.body) == 1
    wrapper_return = wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{DIRECT_OWNER}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir"
    ]
    assert wrapper_return.value.keywords == []

    attention_context = AttentionRecoveryContext(
        pass_context=_pass_context("residual_affine_prelu_attention"),
        mean_attention_cluster=lambda: None,
        gate_layout_cluster=lambda: None,
        transpose_unary_fanout_cluster=lambda: None,
    )
    attention_invocation = build_preadd_mean_attention_invocations(
        attention_context
    )[1]
    assert PREADD_MEAN_ATTENTION_PASS_IDS[1] == DIRECT_OWNER
    assert attention_invocation.callback is (
        optimize_transpose_pre_add_mul_add_prelu_nhwc_chains
    )
    assert attention_invocation.args == (attention_context.pass_context.model_ir,)
    assert attention_invocation.keyword_args == ()

    sinet_context = _pass_context("residual_affine_prelu_sinet")
    sinet_invocation = build_sinet_preadd_resize_recovery_invocations(
        sinet_context
    )[0]
    assert SINET_PREADD_RESIZE_RECOVERY_PASS_IDS[0] == DIRECT_OWNER
    assert sinet_invocation.callback is (
        optimize_transpose_pre_add_mul_add_prelu_nhwc_chains
    )
    assert sinet_invocation.args == (sinet_context.model_ir,)
    assert sinet_invocation.keyword_args == ()


def test_residual_affine_prelu_direct_boundary_is_explicit() -> None:
    lowerer, index = _phase_location()
    invocation = lowerer.body[index]
    assert _phase_id(invocation) == PHASE_ID
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert ast.unparse(invocation.value.args[1]) == OWNER_EXPRESSION
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _call_name(lowerer.body[index + 1]) == SUCCESSOR
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == DIRECT_OWNER
        for node in ast.walk(lowerer)
    ) == 0

    orchestration_owner = _functions(ORCHESTRATION_OWNER_PATH)[
        ORCHESTRATION_OWNER
    ]
    production_calls = [
        node
        for node in ast.walk(orchestration_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PUBLIC_OWNER
    ]
    assert len(production_calls) == 1
    assert [
        ast.unparse(argument) for argument in production_calls[0].args
    ] == ["context.pass_context.model_ir"]
    assert production_calls[0].keywords == []


def test_residual_affine_prelu_direct_result_is_retained_for_observation() -> None:
    lowerer, index = _phase_location()
    assert _phase_id(lowerer.body[index]) == PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
