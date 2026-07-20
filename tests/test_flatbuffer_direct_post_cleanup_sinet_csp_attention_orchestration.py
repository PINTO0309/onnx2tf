from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.attention_layout import (
    _optimize_transpose_csp_attention_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.sinet_preadd_resize_recovery_orchestration import (
    run_sinet_preadd_resize_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "post_cleanup_sinet_csp_attention_orchestration.py"
)
OWNER = "run_post_cleanup_sinet_csp_attention_cleanup"
CHILD_OWNERS = (
    "run_sinet_preadd_resize_recovery",
    "_optimize_transpose_csp_attention_nhwc_chains",
)
CURRENT_TARGET = "_post_cleanup_sinet_preadd_resize_results"
CURRENT_SINET_OWNER = "_run_sinet_preadd_resize_recovery_sequence"
CURRENT_CSP_WRAPPER = "_optimize_transpose_csp_attention_nhwc_chains"
PHASE_ID = "cleanup.post_cleanup.csp_attention"
PREDECESSOR_PHASE_ID = "cleanup.very_late.prune_reconcile"
SUCCESSOR_PHASE_ID = "cleanup.post_cleanup.sa_pa_mirrorpad"
FUTURE_OWNER_EXPRESSION = (
    "run_post_cleanup_sinet_csp_attention_cleanup("
    "shared_model_ir_pass_context)[1]"
)
SINET_SCHEMA = (
    "optimized_transpose_pre_add_mul_add_prelu_nhwc_chains",
    "optimized_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
    "optimized_sinet_concat_resize_affine_transpose_chains",
    "optimized_sinet_dual_resize_affine_transpose_chains",
    "optimized_sinet_concat_resize_affine_tail_concat_transpose_chains",
    "optimized_sinet_softmax_mask_residual_nhwc_tail_chains",
)
CSP_SCHEMA = ("optimized_transpose_csp_attention_nhwc_chains",)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr, ast.Return)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _phase_id(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or ast.unparse(call.func) != "session.record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _phase_record(lowerer: ast.FunctionDef) -> ast.Expr:
    records = [
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Expr) and _phase_id(statement) == PHASE_ID
    ]
    assert len(records) == 1
    return records[0]


def _context() -> ModelIRPassContext:
    model_ir = ModelIR("post_cleanup_sinet_csp_attention_schema")
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def test_post_cleanup_sinet_csp_attention_current_contract() -> None:
    lowerer = _lowerer()
    record = _phase_record(lowerer)
    index = lowerer.body.index(record)
    current = lowerer.body[index - 1]

    assert _single_target(current) == CURRENT_TARGET
    assert _call_name(current) == CURRENT_SINET_OWNER
    current_call = _call(current)
    assert current_call is not None
    assert current_call.args == []
    assert current_call.keywords == []
    assert _phase_id(lowerer.body[index - 2]) == PREDECESSOR_PHASE_ID

    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == (
        f"{CURRENT_CSP_WRAPPER}(model_ir, "
        "layout_state=session.layout_state)"
    )
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )


def test_post_cleanup_sinet_csp_attention_schemas_are_fixed() -> None:
    context = _context()
    sinet_results = run_sinet_preadd_resize_recovery(context)
    csp_results = _optimize_transpose_csp_attention_nhwc_chains(
        context.model_ir,
        layout_state=context.layout_state,
    )

    assert tuple(tuple(result) for result in sinet_results) == tuple(
        (key,) for key in SINET_SCHEMA
    )
    assert all(
        type(value) is int
        for result in sinet_results
        for value in result.values()
    )
    assert tuple(csp_results) == CSP_SCHEMA
    assert all(type(value) is int for value in csp_results.values())


def test_post_cleanup_sinet_csp_attention_wrappers_are_retained() -> None:
    functions = _functions(LOWERER_PATH)

    sinet_wrapper = functions["lower_onnx_to_ir"]
    nested = next(
        node
        for node in sinet_wrapper.body
        if isinstance(node, ast.FunctionDef)
        and node.name == CURRENT_SINET_OWNER
    )
    assert len(nested.body) == 1
    nested_return = nested.body[0]
    assert isinstance(nested_return, ast.Return)
    nested_call = _call(nested_return)
    assert nested_call is not None
    assert isinstance(nested_call.func, ast.Name)
    assert nested_call.func.id == CHILD_OWNERS[0]
    assert [ast.unparse(argument) for argument in nested_call.args] == [
        "sinet_preadd_resize_recovery_context"
    ]
    assert nested_call.keywords == []

    csp_wrapper = functions[CURRENT_CSP_WRAPPER]
    assert len(csp_wrapper.body) == 1
    csp_return = csp_wrapper.body[0]
    assert isinstance(csp_return, ast.Return)
    csp_call = _call(csp_return)
    assert csp_call is not None
    assert isinstance(csp_call.func, ast.Name)
    assert csp_call.func.id == f"{CURRENT_CSP_WRAPPER}_pass"
    assert [ast.unparse(argument) for argument in csp_call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in csp_call.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
    }


@pytest.mark.xfail(
    strict=True,
    reason="post-cleanup SiNet/CSP-attention owner is not implemented",
)
def test_post_cleanup_sinet_csp_attention_has_one_context_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in CHILD_OWNERS
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [call.func.id for call in calls] == list(CHILD_OWNERS)
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {"layout_state": "context.layout_state"}
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == "(sinet_results, csp_results)"

    lowerer = _lowerer()
    record = _phase_record(lowerer)
    index = lowerer.body.index(record)
    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == FUTURE_OWNER_EXPRESSION
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )
