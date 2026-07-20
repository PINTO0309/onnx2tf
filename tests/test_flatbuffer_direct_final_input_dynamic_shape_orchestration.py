from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.final_input_dynamic_orchestration import (
    run_final_input_dynamic_cleanup,
)
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    reconcile_static_tensor_shapes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
FINAL_INPUT_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_input_dynamic_orchestration.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_input_dynamic_shape_orchestration.py"
)
OWNER = "run_final_input_dynamic_shape_cleanup"
CHILD_OWNERS = (
    "run_final_input_dynamic_cleanup",
    "reconcile_static_tensor_shapes",
)
CURRENT_TARGET = "_final_input_dynamic_results"
CURRENT_SHAPE_WRAPPER = "_reconcile_static_tensor_shapes"
PHASE_ID = "shape_reconciliation.primary.very_late_final"
INDEPENDENT_PHASE_ID = "shape_reconciliation.primary.post_split_fallback"
PREDECESSOR_OWNER = "_advance_post_progress"
SUCCESSOR_TARGET = "split_fallback_stats"
FUTURE_OWNER_EXPRESSION = (
    "run_final_input_dynamic_shape_cleanup("
    "shared_model_ir_pass_context)[1]"
)
FINAL_INPUT_SCHEMA = (
    (
        ("repaired_orphan_recurrent_step_tensors",),
        ("repaired_unbound_nonconstant_inputs_with_layout_transpose",),
        ("optimized_transpose_mul_posttranspose_add_nhwc_chains",),
        (
            "optimized_transpose_gather_transpose_axis_remap_nhwc_chains",
            "optimized_constant_input_pad_chains",
            "optimized_constant_input_pool_chains",
            "optimized_constant_input_cast_chains",
            "optimized_redundant_int32_to_int64_passthrough_cast_chains",
            "optimized_redundant_int64_to_int32_cast_chains",
            "optimized_transpose_instancenorm_pad_prepost_nhwc_chains",
            "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains",
            "pruned_unused_tensors",
        ),
    ),
    (
        ("resolved_dynamic_reshape_shapes",),
        (
            "repaired_singleton_nhwc_conv_input_reshapes",
            "repaired_stale_nchw_to_nhwc_conv_input_transposes",
            "pruned_unused_tensors",
        ),
        ("repaired_nchw_channel_shuffle_concat_gathers",),
        ("repaired_nchw_concat_transpose_conv_axes",),
        ("repaired_nchw_concat_global_pool_conv_axes",),
        ("rewritten_dynamic_rank1_unsqueeze_reshape_shape_inputs",),
    ),
)
SHAPE_SCHEMA = (
    "reconciled_static_tensor_shapes",
    "reconciled_static_shape_mutations",
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


def _phase_record(lowerer: ast.FunctionDef, phase_id: str) -> ast.Expr:
    records = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Expr) and _phase_id(node) == phase_id
    ]
    assert len(records) == 1
    return records[0]


def _context() -> ModelIRPassContext:
    model_ir = ModelIR("final_input_dynamic_shape_schema")
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def test_final_input_dynamic_shape_current_contract() -> None:
    lowerer = _lowerer()
    record = _phase_record(lowerer, PHASE_ID)
    index = lowerer.body.index(record)
    current = lowerer.body[index - 1]

    assert _single_target(current) == CURRENT_TARGET
    assert _call_name(current) == CHILD_OWNERS[0]
    current_call = _call(current)
    assert current_call is not None
    assert [ast.unparse(argument) for argument in current_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert current_call.keywords == []
    assert _call_name(lowerer.body[index - 2]) == PREDECESSOR_OWNER

    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == (
        f"{CURRENT_SHAPE_WRAPPER}(model_ir, include_mutation_count=True)"
    )
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )


def test_final_input_dynamic_shape_schemas_are_fixed() -> None:
    context = _context()
    dynamic_results = run_final_input_dynamic_cleanup(context)
    shape_results = reconcile_static_tensor_shapes(
        context.model_ir,
        include_mutation_count=True,
    )

    assert tuple(
        tuple(tuple(result) for result in group)
        for group in dynamic_results
    ) == FINAL_INPUT_SCHEMA
    assert all(
        type(value) is int
        for group in dynamic_results
        for result in group
        for value in result.values()
    )
    assert tuple(shape_results) == SHAPE_SCHEMA
    assert all(type(value) is int for value in shape_results.values())


def test_final_input_dynamic_shape_children_and_independent_route_are_retained() -> None:
    final_input_owner = _functions(FINAL_INPUT_OWNER_PATH)[CHILD_OWNERS[0]]
    final_input_calls = sorted(
        (
            node
            for node in ast.walk(final_input_owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [call.func.id for call in final_input_calls] == [
        "run_late_input_affine_normalization_cleanup",
        "run_very_late_dynamic_adapter_cleanup",
    ]
    assert all(
        [ast.unparse(argument) for argument in call.args] == ["context"]
        and call.keywords == []
        for call in final_input_calls
    )

    shape_wrapper = _functions(LOWERER_PATH)[CURRENT_SHAPE_WRAPPER]
    assert len(shape_wrapper.body) == 1
    shape_return = shape_wrapper.body[0]
    assert isinstance(shape_return, ast.Return)
    shape_call = _call(shape_return)
    assert shape_call is not None
    assert ast.unparse(shape_call.func) == (
        "_static_shape_reconciliation_pass.reconcile_static_tensor_shapes"
    )
    assert [ast.unparse(argument) for argument in shape_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in shape_call.keywords
    } == {
        "graph_index": "graph_index",
        "include_mutation_count": "include_mutation_count",
    }

    independent_record = _phase_record(_lowerer(), INDEPENDENT_PHASE_ID)
    independent_call = _call(independent_record)
    assert independent_call is not None
    assert ast.unparse(independent_call.args[1]) == (
        f"{CURRENT_SHAPE_WRAPPER}(model_ir, include_mutation_count=True)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="final input/dynamic static-shape owner is not implemented",
)
def test_final_input_dynamic_shape_has_one_context_owner() -> None:
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
    } == {"include_mutation_count": "True"}
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "(dynamic_results, shape_results)"
    )

    lowerer = _lowerer()
    record = _phase_record(lowerer, PHASE_ID)
    index = lowerer.body.index(record)
    record_call = _call(record)
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == FUTURE_OWNER_EXPRESSION
    assert _call_name(lowerer.body[index - 1]) == PREDECESSOR_OWNER
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_TARGET
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )
