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
SUCCESSOR_TARGET = "_very_late_affine_post_add_stats"


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


def _direct_locations() -> tuple[ast.FunctionDef, tuple[int, ...]]:
    lowerer = _lowerer()
    return lowerer, tuple(
        next(
            index
            for index, statement in enumerate(lowerer.body)
            if _call_name(statement) == owner
        )
        for owner in OWNERS
    )


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
    lowerer, indices = _direct_locations()
    assert indices == tuple(range(indices[0], indices[0] + len(OWNERS)))
    for index, owner, target in zip(indices, OWNERS, RESULT_TARGETS):
        invocation = lowerer.body[index]
        assert _single_target(invocation) == target
        call = _statement_call(invocation)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert call.keywords == []
        assert sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == owner
            for node in ast.walk(lowerer)
        ) >= 1
    assert _call_name(lowerer.body[indices[0] - 1]) == PREDECESSOR
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET


def test_late_input_repair_results_are_retained_for_observation() -> None:
    lowerer, indices = _direct_locations()
    assert tuple(_single_target(lowerer.body[index]) for index in indices) == (
        RESULT_TARGETS
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
