from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.qkv_attention_orchestration import (
    run_qkv_attention,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    optimize_transpose_relu_split_all_outputs_to_nhwc_chains,
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
    / "post_sinet_qkv_relu_split_all_orchestration.py"
)
OWNER = "run_post_sinet_qkv_relu_split_all_cleanup"
CHILD_OWNERS = (
    "run_qkv_attention",
    "optimize_transpose_relu_split_all_outputs_to_nhwc_chains",
)
CURRENT_TARGET = "_post_sinet_qkv_attention_results"
CURRENT_QKV_OWNER = "_run_qkv_attention_layout_pass_cluster"
CURRENT_RELU_WRAPPER = (
    "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains"
)
PHASE_ID = "cleanup.post_sinet.relu_split_all_outputs"
PREDECESSOR_PHASE_ID = "cleanup.post_sinet.batchmatmul_adj_flags"
SUCCESSOR_PHASE_ID = "cleanup.post_sinet.relu_split_conv_concat"
FUTURE_OWNER_EXPRESSION = (
    "run_post_sinet_qkv_relu_split_all_cleanup("
    "shared_model_ir_pass_context)[1]"
)
QKV_SCHEMA = (
    (
        "optimized_attention_qkv_gather_reshape_transpose_hoist_chains",
        "optimized_attention_qkv_slice_replace_gather_reshape_chains",
        "optimized_attention_qkv_slice_to_split_chains",
        "optimized_attention_split_post_reshape_collapse_chains",
    ),
    (
        "optimized_attention_qkv_shared_pretranspose_slice_nchw_chains",
        "optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains",
    ),
)
RELU_SCHEMA = ("optimized_transpose_relu_split_all_outputs_to_nhwc_chains",)


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
    model_ir = ModelIR("post_sinet_qkv_relu_split_all_schema")
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def test_post_sinet_qkv_relu_split_all_current_contract() -> None:
    lowerer = _lowerer()
    record = _phase_record(lowerer)
    index = lowerer.body.index(record)
    current = lowerer.body[index - 1]

    assert _single_target(current) == CURRENT_TARGET
    assert _call_name(current) == CURRENT_QKV_OWNER
    current_call = _call(current)
    assert current_call is not None
    assert current_call.args == []
    assert current_call.keywords == []
    assert _phase_id(lowerer.body[index - 2]) == PREDECESSOR_PHASE_ID

    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == (
        f"{CURRENT_RELU_WRAPPER}(model_ir, "
        "layout_state=session.layout_state)"
    )
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )


def test_post_sinet_qkv_relu_split_all_schemas_are_fixed() -> None:
    context = _context()
    qkv_results = run_qkv_attention(context)
    relu_results = optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
        context.model_ir,
        layout_state=context.layout_state,
    )

    assert tuple(tuple(result) for result in qkv_results) == QKV_SCHEMA
    assert all(
        type(value) is int
        for result in qkv_results
        for value in result.values()
    )
    assert tuple(relu_results) == RELU_SCHEMA
    assert all(type(value) is int for value in relu_results.values())


def test_post_sinet_qkv_relu_split_all_wrappers_and_terminal_route_are_retained() -> None:
    functions = _functions(LOWERER_PATH)
    lowerer = functions["lower_onnx_to_ir"]

    qkv_wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == CURRENT_QKV_OWNER
    )
    qkv_return = qkv_wrapper.body[0]
    assert isinstance(qkv_return, ast.Return)
    qkv_call = _call(qkv_return)
    assert qkv_call is not None
    assert isinstance(qkv_call.func, ast.Name)
    assert qkv_call.func.id == CHILD_OWNERS[0]
    assert [ast.unparse(argument) for argument in qkv_call.args] == [
        "qkv_attention_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in qkv_call.keywords
    } == {
        "include_layout_transpose": "include_layout_transpose",
        "include_prefix": "include_prefix",
    }

    relu_wrapper = functions[CURRENT_RELU_WRAPPER]
    assert len(relu_wrapper.body) == 1
    relu_return = relu_wrapper.body[0]
    assert isinstance(relu_return, ast.Return)
    relu_call = _call(relu_return)
    assert relu_call is not None
    assert isinstance(relu_call.func, ast.Name)
    assert relu_call.func.id == f"{CURRENT_RELU_WRAPPER}_pass"

    direct_qkv_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CURRENT_QKV_OWNER
    ]
    assert len(direct_qkv_calls) == 2
    assert all(call.args == [] for call in direct_qkv_calls)
    assert all(call.keywords == [] for call in direct_qkv_calls)


@pytest.mark.xfail(
    strict=True,
    reason="post-SiNet QKV/ReLU-Split-all owner is not implemented",
)
def test_post_sinet_qkv_relu_split_all_has_one_context_owner() -> None:
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
    assert ast.unparse(owner_return.value) == "(qkv_results, relu_results)"

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
