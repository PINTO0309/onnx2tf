from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _reconcile_static_tensor_shapes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
RESULT_TARGET = "_fallback_norm_static_shape_stats"
PHASE_ID = "shape_topology.fallback.norm"
RECONCILE_OWNER = "_reconcile_static_tensor_shapes"
RECONCILE_TOPOLOGY_OWNER = "run_static_shape_topology_reconciliation"


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


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


def _phase_result_owner(statement: ast.stmt) -> ast.Call | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
        or not isinstance(call.args[1], ast.Call)
    ):
        return None
    return call.args[1]


def _fallback_norm_guard() -> ast.If:
    lowerer = _lowerer()
    guards = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.If)
        and "fallback_norm_stats.get" in ast.unparse(node.test)
    ]
    assert len(guards) == 1
    return guards[0]


def _stale_reshape_model_ir() -> ModelIR:
    model_ir = ModelIR("fallback_norm_reconcile")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "input": TensorIR(
            name="input",
            dtype="FLOAT32",
            shape=[1, 4],
            shape_signature=[1, 4],
        ),
        "shape": TensorIR(
            name="shape",
            dtype="INT32",
            shape=[2],
            shape_signature=[2],
            data=np.asarray([2, 2], dtype=np.int32),
        ),
        "output": TensorIR(
            name="output",
            dtype="FLOAT32",
            shape=[1, 4],
            shape_signature=[1, 4],
        ),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["input", "shape"],
            outputs=["output"],
            options={"newShape": [2, 2]},
        )
    ]
    return model_ir


def test_fallback_norm_reconciliation_schema_and_mutation_are_explicit() -> None:
    assert _reconcile_static_tensor_shapes(
        ModelIR("complete_schema"),
        include_mutation_count=True,
    ) == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    model_ir = _stale_reshape_model_ir()
    assert _reconcile_static_tensor_shapes(
        model_ir,
        include_mutation_count=True,
    ) == {
        "reconciled_static_tensor_shapes": 1,
        "reconciled_static_shape_mutations": 1,
    }
    assert model_ir.tensors["output"].shape == [2, 2]
    assert model_ir.tensors["output"].shape_signature == [1, 4]


def test_fallback_norm_reconciliation_boundary_is_explicit() -> None:
    guard = _fallback_norm_guard()
    assert len(guard.body) == 3
    assert _call_name(guard.body[0]) == (
        "run_indexed_binary_layout_adapter_cleanup"
    )
    assert _call_name(guard.body[1]) == (
        "_run_singleton_consecutive_reshape_pass_cluster"
    )
    reconciliation = guard.body[2]
    record = _statement_call(reconciliation)
    assert record is not None
    assert ast.literal_eval(record.args[0]) == PHASE_ID
    owner = _phase_result_owner(reconciliation)
    assert owner is not None
    assert isinstance(owner.func, ast.Name)
    assert owner.func.id == RECONCILE_TOPOLOGY_OWNER
    assert [ast.unparse(argument) for argument in owner.args] == ["fallback_ir"]
    assert owner.keywords == []


def test_fallback_norm_reconciliation_retains_complete_observation() -> None:
    guard = _fallback_norm_guard()
    reconciliation = guard.body[2]
    record = _statement_call(reconciliation)
    assert record is not None
    assert ast.literal_eval(record.args[0]) == PHASE_ID
    owner = _phase_result_owner(reconciliation)
    assert owner is not None
    assert isinstance(owner.func, ast.Name)
    assert owner.func.id == RECONCILE_TOPOLOGY_OWNER
    assert [ast.unparse(argument) for argument in owner.args] == ["fallback_ir"]
    assert owner.keywords == []

    lowerer = _lowerer()
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
