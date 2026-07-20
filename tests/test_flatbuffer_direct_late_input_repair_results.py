from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_orphan_recurrent_step_tensors,
    _repair_unbound_nonconstant_operator_inputs_with_layout_transpose,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_input_affine_normalization_orchestration.py"
)
COMPOSITE_OWNER = "run_late_input_affine_normalization_cleanup"
COMPOSITE_TARGET = "_late_input_affine_normalization_results"
FINAL_COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_input_dynamic_orchestration.py"
)
FINAL_COMPOSITE_OWNER = "run_final_input_dynamic_cleanup"
FINAL_COMPOSITE_TARGET = "_final_input_dynamic_results"
FINAL_SHAPE_OWNER_EXPRESSION = (
    "run_final_input_dynamic_shape_cleanup("
    "shared_model_ir_pass_context, "
    "shape_reconciler=_reconcile_static_tensor_shapes)[1]"
)
OWNERS = (
    "_repair_orphan_recurrent_step_tensors",
    "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose",
)
RESULT_TARGETS = (
    "_late_orphan_recurrent_repair_stats",
    "_late_unbound_input_repair_stats",
)
RESULT_SCHEMAS = (
    {"repaired_orphan_recurrent_step_tensors": 0},
    {"repaired_unbound_nonconstant_inputs_with_layout_transpose": 0},
)
PREDECESSOR = "_advance_post_progress"
SUCCESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_final"


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


def _composite_owner() -> ast.FunctionDef:
    return _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]


def _phase_id(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_late_input_repair_result_schemas_are_explicit() -> None:
    functions = _functions(LOWERER_PATH)
    callbacks = (
        _repair_orphan_recurrent_step_tensors,
        _repair_unbound_nonconstant_operator_inputs_with_layout_transpose,
    )
    for owner, callback, expected in zip(OWNERS, callbacks, RESULT_SCHEMAS):
        assert ast.unparse(functions[owner].returns) == "Dict[str, int]"
        assert callback(ModelIR(f"{owner}_schema")) == expected


def test_late_input_repair_direct_boundary_is_explicit() -> None:
    owner = _composite_owner()
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in {
                "repair_orphan_recurrent_step_tensors_summary",
                "repair_unbound_nonconstant_operator_inputs_with_layout_transpose",
            }
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert len(calls) == 2
    assert all(
        [ast.unparse(argument) for argument in call.args]
        == ["context.model_ir"]
        and call.keywords == []
        for call in calls
    )

    lowerer = _lowerer()
    index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _phase_id(statement) == SUCCESSOR_PHASE_ID
    )
    record_call = _statement_call(lowerer.body[index])
    assert record_call is not None
    assert ast.unparse(record_call.args[1]) == FINAL_SHAPE_OWNER_EXPRESSION
    assert _call_name(lowerer.body[index - 1]) == PREDECESSOR
    assert _single_target(lowerer.body[index + 1]) == "split_fallback_stats"
    assert not any(
        isinstance(node, ast.Name) and node.id == FINAL_COMPOSITE_TARGET
        for node in ast.walk(lowerer)
    )
    final_owner = _functions(FINAL_COMPOSITE_PATH)[FINAL_COMPOSITE_OWNER]
    child_calls = [
        node
        for node in ast.walk(final_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == COMPOSITE_OWNER
    ]
    assert len(child_calls) == 1
    assert [ast.unparse(argument) for argument in child_calls[0].args] == [
        "context"
    ]


def test_late_input_repair_results_are_retained_for_observation() -> None:
    lowerer = _lowerer()
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == COMPOSITE_TARGET
        for node in ast.walk(lowerer)
    )
