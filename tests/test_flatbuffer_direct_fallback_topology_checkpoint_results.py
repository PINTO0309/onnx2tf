from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _topologically_sort_operators,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SORT_OWNER = "_topologically_sort_operators"
EXPECTED_PREDECESSOR_GUARDS = (
    "int(fallback_placeholder_matmul_stats.get("
    "'restored_placeholder_matmul_flattened_inputs', 0)) > 0",
    "int(fallback_binary_layout_stats.get("
    "'repaired_stale_nchw_to_nhwc_channelwise_binary_transposes', 0)) > 0",
)
EXPECTED_RESULT_TARGETS = (
    "_fallback_post_placeholder_topology_stats",
    "_fallback_post_layout_repair_topology_stats",
)


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _safety_fallback_body(lowerer: ast.FunctionDef) -> list[ast.stmt]:
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test)
        == "optimize_layout_transpose_chains and len(unbound_inputs) > 0"
    )
    return guard.body


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


def _reversed_model_ir() -> ModelIR:
    model_ir = ModelIR("fallback_topology_checkpoint")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        name: TensorIR(name=name, dtype="FLOAT32", shape=[1])
        for name in ("input", "middle", "output")
    }
    model_ir.operators = [
        OperatorIR("RELU", ["middle"], ["output"]),
        OperatorIR("IDENTITY", ["input"], ["middle"]),
    ]
    return model_ir


def test_two_fallback_topology_checkpoints_are_explicit() -> None:
    body = _safety_fallback_body(_lowerer())
    locations = [
        index
        for index, statement in enumerate(body)
        if _call_name(statement) == SORT_OWNER
    ]

    assert len(locations) == 2
    assert tuple(_single_target(body[index]) for index in locations) == (
        EXPECTED_RESULT_TARGETS
    )
    assert tuple(
        ast.unparse(body[index - 1].test)
        for index in locations
        if isinstance(body[index - 1], ast.If)
    ) == EXPECTED_PREDECESSOR_GUARDS
    assert ast.unparse(body[locations[0] + 1]) == (
        "_fallback_precision_div_rewrite_stats = "
        "_rewrite_constant_divisors_to_multiplicative_reciprocals(fallback_ir)"
    )
    assert ast.unparse(body[locations[1] + 1].targets[0]) == (
        "fallback_ir.metadata['layout_optimize_fallback']"
    )
    for index in locations:
        call = _statement_call(body[index])
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == [
            "fallback_ir"
        ]
        assert call.keywords == []


def test_fallback_topology_checkpoint_result_schema_is_explicit() -> None:
    assert _topologically_sort_operators(ModelIR("stable_schema")) == {
        "reordered_operators": 0,
        "cycle_detected": 0,
    }

    model_ir = _reversed_model_ir()
    assert _topologically_sort_operators(model_ir) == {
        "reordered_operators": 2,
        "cycle_detected": 0,
    }
    assert [operator.op_type for operator in model_ir.operators] == [
        "IDENTITY",
        "RELU",
    ]
