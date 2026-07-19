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
EXPECTED_INNER_GUARDS = (
    None,
    "apply_safe_transpose_reduction_lite_on_no_layout_opt",
    "int(final_placeholder_matmul_stats.get("
    "'restored_placeholder_matmul_flattened_inputs', 0)) > 0",
)
EXPECTED_PREDECESSOR_TARGETS = (
    None,
    "_no_layout_final_affine_prepost_stats",
    None,
)
EXPECTED_RESULT_TARGETS = (
    "_primary_post_lowering_topology_stats",
    "_no_layout_post_reduction_topology_stats",
    "_final_placeholder_topology_stats",
)


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _pipeline_blocks(
    statements: list[ast.stmt],
    guard_stack: tuple[str, ...] = (),
) -> list[tuple[list[ast.stmt], tuple[str, ...]]]:
    blocks = [(statements, guard_stack)]
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(statement, ast.If):
            test = ast.unparse(statement.test)
            if statement.body:
                blocks.extend(
                    _pipeline_blocks(statement.body, (*guard_stack, test))
                )
            if statement.orelse:
                blocks.extend(
                    _pipeline_blocks(statement.orelse, (*guard_stack, test))
                )
            continue
        for attribute in ("body", "orelse", "finalbody"):
            nested = getattr(statement, attribute, None)
            if isinstance(nested, list) and nested:
                blocks.extend(_pipeline_blocks(nested, guard_stack))
        if isinstance(statement, (ast.Try, ast.TryStar)):
            for handler in statement.handlers:
                blocks.extend(_pipeline_blocks(handler.body, guard_stack))
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


def _reversed_model_ir() -> ModelIR:
    model_ir = ModelIR("primary_topology_checkpoint")
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


def test_three_primary_topology_checkpoints_are_explicit() -> None:
    locations = sorted(
        [
            (block, index, guards)
            for block, guards in _pipeline_blocks(_lowerer().body)
            for index, statement in enumerate(block)
            if _call_name(statement) == SORT_OWNER
            and ast.unparse(_statement_call(statement).args[0]) == "model_ir"
        ],
        key=lambda item: item[0][item[1]].lineno,
    )

    assert len(locations) == 3
    assert tuple(
        _single_target(block[index]) for block, index, _ in locations
    ) == EXPECTED_RESULT_TARGETS
    assert tuple(guards[-1] if guards else None for _, _, guards in locations) == (
        EXPECTED_INNER_GUARDS
    )
    assert tuple(
        _single_target(block[index - 1]) for block, index, _ in locations
    ) == EXPECTED_PREDECESSOR_TARGETS
    assert ast.unparse(locations[0][0][locations[0][1] - 1]) == (
        "_set_post_progress_desc('topological sort')"
    )
    assert isinstance(locations[2][0][locations[2][1] - 1], ast.If)
    for block, index, _ in locations:
        call = _statement_call(block[index])
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert call.keywords == []


def test_primary_topology_checkpoint_result_schema_is_explicit() -> None:
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
