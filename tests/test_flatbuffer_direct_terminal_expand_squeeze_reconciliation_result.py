from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _reconcile_static_tensor_shapes,
    _replace_expand_dims_and_squeeze_with_reshape,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPAND_RESULT_TARGET = "_terminal_expand_squeeze_stats"
RECONCILE_RESULT_TARGET = "_terminal_expand_squeeze_static_shape_stats"


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _stale_reshape_model_ir() -> ModelIR:
    model_ir = ModelIR("terminal_expand_squeeze_reconcile")
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


def test_terminal_expand_squeeze_reconciliation_contract_is_explicit() -> None:
    assert _reconcile_static_tensor_shapes(ModelIR("default_schema")) == {
        "reconciled_static_tensor_shapes": 0,
    }
    assert _reconcile_static_tensor_shapes(
        ModelIR("complete_schema"),
        include_mutation_count=True,
    ) == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    lowerer = _lowerer()
    expand_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == EXPAND_RESULT_TARGET
    )
    expand_call = _statement_call(lowerer.body[expand_index])
    assert expand_call is not None
    assert isinstance(expand_call.func, ast.Name)
    assert expand_call.func.id == "_replace_expand_dims_and_squeeze_with_reshape"
    assert [ast.unparse(argument) for argument in expand_call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in expand_call.keywords
    } == {"layout_state": "session.layout_state"}

    reconciliation = lowerer.body[expand_index + 1]
    assert _single_target(reconciliation) == RECONCILE_RESULT_TARGET
    assert _call_name(reconciliation) == "_reconcile_static_tensor_shapes"
    reconcile_call = _statement_call(reconciliation)
    assert reconcile_call is not None
    assert [ast.unparse(argument) for argument in reconcile_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconcile_call.keywords
    } == {"include_mutation_count": "True"}
    assert _call_name(lowerer.body[expand_index + 2]) == "_advance_post_progress"


def test_terminal_reconciliation_is_required_after_zero_expand_rewrites() -> None:
    model_ir = _stale_reshape_model_ir()

    assert _replace_expand_dims_and_squeeze_with_reshape(model_ir) == {
        "replaced_expand_dims_and_squeeze_with_reshape": 0,
        "expand_dims_squeeze_rewrite_shape_tensors": 0,
    }
    assert _reconcile_static_tensor_shapes(
        model_ir,
        include_mutation_count=True,
    ) == {
        "reconciled_static_tensor_shapes": 1,
        "reconciled_static_shape_mutations": 1,
    }
    assert model_ir.tensors["output"].shape == [2, 2]
    assert model_ir.tensors["output"].shape_signature == [1, 4]


def test_terminal_reconciliation_retains_complete_observation_result() -> None:
    lowerer = _lowerer()
    expand_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == EXPAND_RESULT_TARGET
    )
    reconciliation = lowerer.body[expand_index + 1]
    assert _single_target(reconciliation) == RECONCILE_RESULT_TARGET
    reconcile_call = _statement_call(reconciliation)
    assert reconcile_call is not None
    assert isinstance(reconcile_call.func, ast.Name)
    assert reconcile_call.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconcile_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconcile_call.keywords
    } == {"include_mutation_count": "True"}
    assert _call_name(lowerer.body[expand_index + 2]) == "_advance_post_progress"
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RECONCILE_RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
