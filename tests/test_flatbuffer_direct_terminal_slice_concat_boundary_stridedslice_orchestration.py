from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_slice_concat_boundary_stridedslice_orchestration,
)
from onnx2tf.tflite_builder.passes.channel_slice_layout import (
    _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
    run_terminal_slice_concat_recovery,
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
    / "terminal_slice_concat_boundary_stridedslice_orchestration.py"
)
FINAL_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_boundary_slice_concat_orchestration.py"
)
OWNER = "run_terminal_slice_concat_boundary_stridedslice_cleanup"
CHILD_OWNERS = (
    "run_terminal_slice_concat_recovery",
    "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks",
)
CURRENT_TARGET = "_terminal_slice_concat_recovery_results"
RECOVERY_WRAPPER = "_run_terminal_slice_concat_layout_recovery_sequence"
BOUNDARY_WRAPPER = CHILD_OWNERS[1]
PHASE_ID = "cleanup.terminal.boundary_stridedslice_qdq_concat"
PREDECESSOR_PHASE_ID = "cleanup.terminal.channel_slice_muladd_bridge"
SUCCESSOR_PHASE_ID = "cleanup.terminal.swish_residual_concat_closure"
FUTURE_OWNER_EXPRESSION = (
    "run_terminal_slice_concat_boundary_stridedslice_cleanup("
    "terminal_slice_concat_recovery_context)[1]"
)
RECOVERY_SCHEMA = (
    (),
    ("optimized_transpose_mul_posttranspose_add_nhwc_chains",),
    ("optimized_concat_mul_add_transpose_nhwc_bridge_chains",),
    ("optimized_concat_mul_add_transpose_add_nhwc_bridge_chains",),
    ("optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",),
    ("optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains",),
    ("optimized_singleton_gate_conv_concat_nhwc_bridge_blocks",),
    ("optimized_transpose_unary_split_concat_single_post_nchw",),
    ("optimized_transpose_split_channelwise_tail_to_single_post_nchw",),
    ("optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw",),
    (
        "sanitized_probable_nhwc_axis_sensitive_ops",
        "inserted_probable_nhwc_terminal_transposes",
    ),
    (
        "optimized_transpose_stridedslice_pad_concat_mul_add_"
        "posttranspose_nhwc_chains",
    ),
    ("optimized_transpose_pre_add_nhwc_chains",),
    (
        "iterations",
        "removed_identity_transpose",
        "removed_inverse_transpose_pairs",
        "removed_inverse_transpose_fanout_branches",
        "composed_consecutive_transpose_pairs",
    ),
)
BOUNDARY_SCHEMA = (
    "removed_boundary_input_transpose_stridedslice_blocks",
    "rewritten_boundary_stridedslices",
    "rewritten_boundary_qdq_concat_axis",
    "removed_boundary_post_transposes",
)


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


def _context() -> TerminalSliceConcatRecoveryContext:
    model_ir = ModelIR("terminal_slice_concat_boundary_stridedslice_schema")
    return TerminalSliceConcatRecoveryContext(
        pass_context=ModelIRPassContext(
            model_ir=model_ir,
            layout_state=LayoutState.from_model_ir(model_ir),
            diagnostics=[],
        ),
        channel_slice_pad_mul_cluster=lambda: (),
    )


def test_terminal_slice_concat_boundary_stridedslice_current_contract() -> None:
    lowerer = _lowerer()
    record = _phase_record(lowerer)
    index = lowerer.body.index(record)
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID

    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == FUTURE_OWNER_EXPRESSION
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )


def test_terminal_slice_concat_boundary_stridedslice_schemas_are_fixed() -> None:
    context = _context()
    recovery_results = run_terminal_slice_concat_recovery(context)
    boundary_results = (
        _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks(
            context.pass_context.model_ir,
            layout_state=context.pass_context.layout_state,
        )
    )

    assert tuple(tuple(result) for result in recovery_results) == RECOVERY_SCHEMA
    assert recovery_results[0] == ()
    assert all(
        type(value) is int
        for result in recovery_results[1:]
        for value in result.values()
    )
    assert tuple(boundary_results) == BOUNDARY_SCHEMA
    assert all(type(value) is int for value in boundary_results.values())


def test_terminal_slice_concat_boundary_wrappers_and_route_are_retained() -> None:
    functions = _functions(LOWERER_PATH)
    lowerer = functions["lower_onnx_to_ir"]
    recovery_wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RECOVERY_WRAPPER
    )
    recovery_return = recovery_wrapper.body[0]
    assert isinstance(recovery_return, ast.Return)
    recovery_call = _call(recovery_return)
    assert recovery_call is not None
    assert isinstance(recovery_call.func, ast.Name)
    assert recovery_call.func.id == CHILD_OWNERS[0]
    assert [ast.unparse(argument) for argument in recovery_call.args] == [
        "terminal_slice_concat_recovery_context"
    ]
    assert recovery_call.keywords == []

    boundary_wrapper = functions[BOUNDARY_WRAPPER]
    assert len(boundary_wrapper.body) == 1
    boundary_return = boundary_wrapper.body[0]
    assert isinstance(boundary_return, ast.Return)
    boundary_call = _call(boundary_return)
    assert boundary_call is not None
    assert isinstance(boundary_call.func, ast.Name)
    assert boundary_call.func.id == f"{BOUNDARY_WRAPPER}_pass"
    assert [ast.unparse(argument) for argument in boundary_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in boundary_call.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
    }

    final_owner = _functions(FINAL_OWNER_PATH)[
        "run_final_boundary_slice_concat_cleanup"
    ]
    independent_calls = [
        node
        for node in ast.walk(final_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHILD_OWNERS[0]
    ]
    assert len(independent_calls) == 1
    assert [
        ast.unparse(argument) for argument in independent_calls[0].args
    ] == ["context"]
    assert independent_calls[0].keywords == []


def test_terminal_slice_concat_boundary_has_one_context_owner() -> None:
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
        "context.pass_context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[1].keywords
    } == {"layout_state": "context.pass_context.layout_state"}
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "(recovery_results, boundary_results)"
    )

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


def test_terminal_slice_concat_boundary_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    recovery_results = ({"recovery": 1},)
    boundary_results = {"boundary": 2}
    observed: list[tuple[str, object, object | None]] = []

    def recovery(active_context: object) -> object:
        observed.append((CHILD_OWNERS[0], active_context, None))
        return recovery_results

    def boundary(active_model_ir: object, *, layout_state: object) -> object:
        observed.append((CHILD_OWNERS[1], active_model_ir, layout_state))
        return boundary_results

    monkeypatch.setattr(
        terminal_slice_concat_boundary_stridedslice_orchestration,
        CHILD_OWNERS[0],
        recovery,
    )
    monkeypatch.setattr(
        terminal_slice_concat_boundary_stridedslice_orchestration,
        CHILD_OWNERS[1],
        boundary,
    )

    actual = terminal_slice_concat_boundary_stridedslice_orchestration.run_terminal_slice_concat_boundary_stridedslice_cleanup(
        context
    )
    assert actual[0] is recovery_results
    assert actual[1] is boundary_results
    assert observed == [
        (CHILD_OWNERS[0], context, None),
        (
            CHILD_OWNERS[1],
            context.pass_context.model_ir,
            context.pass_context.layout_state,
        ),
    ]
