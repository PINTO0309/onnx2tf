from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _resolve_dynamic_reshape_shapes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_resolve_dynamic_reshape_shapes"
RESULT_TARGET = "_core_cleanup_dynamic_reshape_stats"
PREDECESSOR_TARGET = "_core_cleanup_conv_activation_stats"
SUCCESSOR_TARGET = "_core_cleanup_squeeze_reshape_identity_stats"


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


def _core_invocation() -> tuple[ast.FunctionDef, int]:
    lowerer = _lowerer()
    indices = [
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == OWNER
    ]
    assert len(indices) == 2
    core_indices = [
        index
        for index in indices
        if _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    ]
    assert len(core_indices) == 1
    return lowerer, core_indices[0]


def _dynamic_reshape_model_ir() -> ModelIR:
    model_ir = ModelIR("core_dynamic_reshape_result")
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
            data=np.asarray([-1, 2], dtype=np.int32),
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
            options={"newShape": [-1, 2]},
        )
    ]
    return model_ir


def test_core_dynamic_reshape_schema_and_mutation_are_explicit() -> None:
    assert _resolve_dynamic_reshape_shapes(ModelIR("stable_schema")) == {
        "resolved_dynamic_reshape_shapes": 0,
    }

    model_ir = _dynamic_reshape_model_ir()
    assert _resolve_dynamic_reshape_shapes(model_ir) == {
        "resolved_dynamic_reshape_shapes": 1,
    }
    reshape = model_ir.operators[0]
    assert reshape.options == {"newShape": [2, 2]}
    np.testing.assert_array_equal(
        model_ir.tensors["shape"].data,
        np.asarray([2, 2], dtype=np.int32),
    )
    assert model_ir.tensors["output"].shape == [2, 2]
    assert model_ir.tensors["output"].shape_signature == [2, 2]


def test_core_dynamic_reshape_boundary_is_explicit() -> None:
    lowerer, index = _core_invocation()
    invocation = lowerer.body[index]
    assert _single_target(invocation) == RESULT_TARGET
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET


def test_core_dynamic_reshape_result_is_retained_for_observation() -> None:
    lowerer, index = _core_invocation()
    invocation = lowerer.body[index]
    assert _single_target(invocation) == RESULT_TARGET
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
