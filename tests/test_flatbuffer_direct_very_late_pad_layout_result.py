from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.gate_layout_orchestration import (
    GATE_LAYOUT_PASS_IDS,
    GATE_LAYOUT_REQUIRED_PASS_IDS,
    build_gate_layout_invocations,
)
from onnx2tf.tflite_builder.passes.pad_layout import run_pad_layout_cleanup
from onnx2tf.tflite_builder.passes.terminal_boundary_layout_orchestration import (
    TERMINAL_BOUNDARY_LAYOUT_PASS_IDS,
    build_terminal_boundary_layout_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes" / "pad_layout.py"
OWNER = "run_pad_layout_cleanup"
RESULT_TARGET = "_very_late_pad_layout_stats"
COMPOSITE_TARGET = "_very_late_pad_instancenorm_layout_results"
COMPOSITE_OWNER = "run_very_late_pad_instancenorm_layout_cleanup"
PREDECESSOR_TARGET = "_late_conv1d_decoder_layout_results"
SUCCESSOR_TARGET = "_very_late_singleton_consecutive_reshape_results"
RESULT_SCHEMA = {
    "optimized_transpose_pad_prepost_nhwc_chains": 0,
    "optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains": 0,
    "optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains": 0,
}


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


def _context(name: str) -> ModelIRPassContext:
    model_ir = ModelIR(name)
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def test_very_late_pad_schema_cleanup_and_routes_are_explicit() -> None:
    functions = _functions(OWNER_PATH)
    owner = functions[OWNER]
    assert [argument.arg for argument in owner.args.kwonlyargs] == [
        "include_pad",
        "include_unary",
        "include_norm",
        "layout_state",
        "diagnostics",
        "state_scope",
    ]
    assert [ast.unparse(value) for value in owner.args.kw_defaults] == [
        "True",
        "True",
        "True",
        "None",
        "None",
        "None",
    ]
    pass_specs = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PassSpec"
    ]
    assert [
        ast.literal_eval(
            next(
                keyword.value
                for keyword in pass_spec.keywords
                if keyword.arg == "pass_id"
            )
        )
        for pass_spec in pass_specs
    ] == [
        "layout.pad_prepost_nhwc",
        "layout.unary_pad_prepost_nhwc",
        "layout.norm_subgraph_pad_prepost_nhwc",
    ]
    assert all(
        any(
            keyword.arg == "transactional"
            and ast.unparse(keyword.value) == "True"
            for keyword in pass_spec.keywords
        )
        for pass_spec in pass_specs
    )
    for owner_name in (
        "_optimize_transpose_pad_prepost_nhwc_chains",
        "_optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains",
        "_optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains",
    ):
        assert any(
            _call_name(statement) == "_prune_unused_tensors"
            for statement in functions[owner_name].body
        )
    assert run_pad_layout_cleanup(ModelIR("very_late_pad_schema")) == RESULT_SCHEMA

    context = _context("very_late_pad_routes")
    required = build_gate_layout_invocations(
        context,
        include_mixed_attention=False,
    )
    full = build_gate_layout_invocations(
        context,
        include_mixed_attention=True,
    )
    assert GATE_LAYOUT_REQUIRED_PASS_IDS[1] == OWNER
    assert GATE_LAYOUT_PASS_IDS[2] == OWNER
    assert required[1].callback is run_pad_layout_cleanup
    assert full[2].callback is run_pad_layout_cleanup
    terminal = build_terminal_boundary_layout_invocations(context)
    assert TERMINAL_BOUNDARY_LAYOUT_PASS_IDS[2] == OWNER
    assert terminal[2].callback is run_pad_layout_cleanup
    for invocation in (required[1], full[2], terminal[2]):
        assert invocation.args == (context.model_ir,)
        assert set(dict(invocation.keyword_args)) == {
            "layout_state",
            "diagnostics",
            "state_scope",
        }


def test_very_late_pad_moves_to_composite_and_keeps_consumed_fallback() -> None:
    lowerer = _lowerer()
    composite = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(composite)
    assert _call_name(composite) == COMPOSITE_OWNER
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET

    owner_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]
    assert owner_calls == []
    summary_owner = "run_norm_subgraph_pad_layout_summary"
    fallback = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == summary_owner
    )
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in fallback.keywords
    } == {
        "diagnostics": "session.diagnostics",
    }
    fallback_parent = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Assign)
        and _single_target(node) == "fallback_norm_stats"
    )
    assert isinstance(fallback_parent.value, ast.Call)
    assert fallback in list(ast.walk(fallback_parent.value))


def test_very_late_pad_old_result_local_is_removed() -> None:
    lowerer = _lowerer()
    assert not any(
        isinstance(node, ast.Name) and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
