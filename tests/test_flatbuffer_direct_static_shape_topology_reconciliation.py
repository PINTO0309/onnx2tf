from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _topologically_sort_operators,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    reconcile_static_tensor_shapes,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
RECONCILE_OWNER = "_reconcile_static_tensor_shapes"
SORT_OWNER = "_topologically_sort_operators"
EXPECTED_RESULT_TARGETS = (
    "_fallback_norm_static_shape_stats",
    "_fallback_high_rank_bmm_static_shape_stats",
    "_final_high_rank_bmm_static_shape_stats",
    "_final_pad_layout_static_shape_stats",
    "_final_conv_input_static_shape_stats",
    "_final_mixed_concat_static_shape_stats",
    "_final_concat_axis_static_shape_stats",
    "_final_binary_layout_static_shape_stats",
)
EXPECTED_MODEL_ARGUMENTS = (
    "fallback_ir",
    "fallback_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
)


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _pipeline_blocks(statements: list[ast.stmt]) -> list[list[ast.stmt]]:
    blocks = [statements]
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for attribute in ("body", "orelse", "finalbody"):
            nested = getattr(statement, attribute, None)
            if isinstance(nested, list) and nested:
                blocks.extend(_pipeline_blocks(nested))
        if isinstance(statement, (ast.Try, ast.TryStar)):
            for handler in statement.handlers:
                blocks.extend(_pipeline_blocks(handler.body))
    return blocks


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


def _raw_reconcile_sort_pairs(
    lowerer: ast.FunctionDef,
) -> list[tuple[list[ast.stmt], int]]:
    return sorted(
        [
            (block, index)
            for block in _pipeline_blocks(lowerer.body)
            for index in range(len(block) - 1)
            if _call_name(block[index]) == RECONCILE_OWNER
            and _call_name(block[index + 1]) == SORT_OWNER
        ],
        key=lambda item: item[0][item[1]].lineno,
    )


def _stale_reversed_model_ir() -> ModelIR:
    model_ir = ModelIR("static_shape_topology_reconciliation")
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
        "reshaped": TensorIR(
            name="reshaped",
            dtype="FLOAT32",
            shape=[1, 4],
            shape_signature=[1, 4],
        ),
        "output": TensorIR(
            name="output",
            dtype="FLOAT32",
            shape=[2, 2],
            shape_signature=[2, 2],
        ),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["reshaped"], ["output"]),
        OperatorIR(
            "RESHAPE",
            ["input", "shape"],
            ["reshaped"],
            options={"newShape": [2, 2]},
        ),
    ]
    return model_ir


def test_eight_static_shape_topology_boundaries_are_explicit() -> None:
    pairs = _raw_reconcile_sort_pairs(_lowerer())

    assert len(pairs) == 8
    assert tuple(_single_target(block[index]) for block, index in pairs) == (
        EXPECTED_RESULT_TARGETS
    )
    assert tuple(
        ast.unparse(_statement_call(block[index]).args[0])
        for block, index in pairs
    ) == EXPECTED_MODEL_ARGUMENTS
    for block, index in pairs:
        reconcile_call = _statement_call(block[index])
        sort_call = _statement_call(block[index + 1])
        assert reconcile_call is not None
        assert sort_call is not None
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in reconcile_call.keywords
        } == {"include_mutation_count": "True"}
        assert [ast.unparse(argument) for argument in sort_call.args] == [
            ast.unparse(reconcile_call.args[0])
        ]
        assert sort_call.keywords == []


def test_static_shape_then_topology_contract_is_explicit() -> None:
    model_ir = _stale_reversed_model_ir()

    shape_stats = reconcile_static_tensor_shapes(
        model_ir,
        include_mutation_count=True,
    )
    sort_stats = _topologically_sort_operators(model_ir)

    assert shape_stats == {
        "reconciled_static_tensor_shapes": 3,
        "reconciled_static_shape_mutations": 3,
    }
    assert sort_stats == {
        "reordered_operators": 2,
        "cycle_detected": 0,
    }
    assert model_ir.tensors["reshaped"].shape == [2, 2]
    assert [operator.op_type for operator in model_ir.operators] == [
        "RESHAPE",
        "RELU",
    ]
