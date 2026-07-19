from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _topologically_sort_operators,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    validate_model_ir_layout_annotations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SORT_OWNER = "_topologically_sort_operators"
VALIDATION_OWNER = "validate_model_ir_layout_annotations"
VALIDATION_METADATA_KEY = "logical_layout_validation_errors"
EXPECTED_PROBLEM_TARGETS = (
    "fallback_layout_problems",
    "layout_problems",
)
EXPECTED_MODEL_ARGUMENTS = (
    "fallback_ir",
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


def _raw_validation_boundaries(
    lowerer: ast.FunctionDef,
) -> list[tuple[list[ast.stmt], int]]:
    return sorted(
        [
            (block, index)
            for block in _pipeline_blocks(lowerer.body)
            for index in range(len(block) - 3)
            if _call_name(block[index]) == SORT_OWNER
            and _call_name(block[index + 1]) == VALIDATION_OWNER
            and isinstance(block[index + 2], ast.If)
            and isinstance(block[index + 3], ast.Return)
        ],
        key=lambda item: item[0][item[1]].lineno,
    )


def _invalid_reversed_model_ir() -> ModelIR:
    model_ir = ModelIR("topology_layout_validation")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "input": TensorIR(
            name="input",
            dtype="FLOAT32",
            shape=[1, 2, 3],
            shape_signature=[1, 2, 3],
            logical_layout="NCHW",
        ),
        "middle": TensorIR(
            name="middle",
            dtype="FLOAT32",
            shape=[1, 2, 3],
            shape_signature=[1, 2, 3],
        ),
        "output": TensorIR(
            name="output",
            dtype="FLOAT32",
            shape=[1, 2, 3],
            shape_signature=[1, 2, 3],
        ),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["middle"], ["output"]),
        OperatorIR("IDENTITY", ["input"], ["middle"]),
    ]
    return model_ir


def test_two_terminal_topology_layout_validation_boundaries_are_explicit() -> None:
    boundaries = _raw_validation_boundaries(_lowerer())

    assert len(boundaries) == 2
    assert tuple(
        _single_target(block[index + 1]) for block, index in boundaries
    ) == EXPECTED_PROBLEM_TARGETS
    assert tuple(
        ast.unparse(_statement_call(block[index]).args[0])
        for block, index in boundaries
    ) == EXPECTED_MODEL_ARGUMENTS
    for boundary_index, (block, index) in enumerate(boundaries):
        model_argument = EXPECTED_MODEL_ARGUMENTS[boundary_index]
        validation_call = _statement_call(block[index + 1])
        assert validation_call is not None
        assert [ast.unparse(argument) for argument in validation_call.args] == [
            model_argument
        ]
        guard = block[index + 2]
        assert isinstance(guard, ast.If)
        problem_target = _single_target(block[index + 1])
        assert ast.unparse(guard.test) == f"len({problem_target}) > 0"
        assert ast.unparse(guard.body[0]) == (
            f"{model_argument}.metadata['{VALIDATION_METADATA_KEY}'] = "
            f"list({problem_target})"
        )
        assert ast.unparse(guard.orelse[0]) == (
            f"{model_argument}.metadata.pop('{VALIDATION_METADATA_KEY}', None)"
        )
        assert ast.unparse(block[index + 3].value) == (
            f"_finalize_model_ir({model_argument})"
        )


def test_terminal_topology_then_layout_validation_effects_are_explicit() -> None:
    model_ir = _invalid_reversed_model_ir()

    sort_stats = _topologically_sort_operators(model_ir)
    problems = validate_model_ir_layout_annotations(model_ir)
    if problems:
        model_ir.metadata[VALIDATION_METADATA_KEY] = list(problems)
    else:
        model_ir.metadata.pop(VALIDATION_METADATA_KEY, None)

    assert sort_stats == {
        "reordered_operators": 2,
        "cycle_detected": 0,
    }
    assert problems == [
        "tensor=input shape=[1, 2, 3] logical_layout=NCHW",
    ]
    assert model_ir.metadata[VALIDATION_METADATA_KEY] == problems
    assert [operator.op_type for operator in model_ir.operators] == [
        "IDENTITY",
        "RELU",
    ]
