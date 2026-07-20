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
PHASE_ID = "shape_resolution.core.dynamic_reshape"
PREDECESSOR_PHASE_ID = "cleanup.core.conv_activation"
SUCCESSOR_PHASE_ID = "cleanup.core.squeeze_reshape_identity"


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


def _phase_result_id(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if _phase_result_owner(statement) is None or call is None:
        return None
    return ast.literal_eval(call.args[0])


def _core_invocation() -> tuple[ast.FunctionDef, int]:
    lowerer = _lowerer()
    indices = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(_phase_result_owner(statement), ast.Call)
        and isinstance(_phase_result_owner(statement).func, ast.Name)
        and _phase_result_owner(statement).func.id == OWNER
    ]
    assert len(indices) == 1
    assert _phase_result_id(lowerer.body[indices[0] - 1]) == (
        PREDECESSOR_PHASE_ID
    )
    return lowerer, indices[0]


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
    record_call = _statement_call(invocation)
    assert record_call is not None
    assert ast.literal_eval(record_call.args[0]) == PHASE_ID
    call = _phase_result_owner(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []
    assert _phase_result_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_result_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID


def test_core_dynamic_reshape_result_is_retained_for_observation() -> None:
    lowerer, index = _core_invocation()
    invocation = lowerer.body[index]
    record_call = _statement_call(invocation)
    assert record_call is not None
    assert ast.literal_eval(record_call.args[0]) == PHASE_ID
    call = _phase_result_owner(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []
    assert _phase_result_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_result_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
