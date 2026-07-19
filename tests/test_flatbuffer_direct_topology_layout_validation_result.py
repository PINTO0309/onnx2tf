from __future__ import annotations

import ast
import copy
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
from onnx2tf.tflite_builder.passes.topology_layout_validation import (
    run_topology_layout_validation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SORT_OWNER = "_topologically_sort_operators"
VALIDATION_OWNER = "validate_model_ir_layout_annotations"
RUNNER = "run_topology_layout_validation"
VALIDATION_METADATA_KEY = "logical_layout_validation_errors"
EXPECTED_PHASE_IDS = (
    "layout_validation.fallback.terminal",
    "layout_validation.primary.terminal",
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


def _runner_locations(
    lowerer: ast.FunctionDef,
) -> list[tuple[list[ast.stmt], int]]:
    return sorted(
        [
            (block, index)
            for block in _pipeline_blocks(lowerer.body)
            for index, statement in enumerate(block)
            if isinstance(_phase_result_owner(statement), ast.Call)
            and isinstance(_phase_result_owner(statement).func, ast.Name)
            and _phase_result_owner(statement).func.id == RUNNER
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


def _cyclic_model_ir() -> ModelIR:
    model_ir = ModelIR("topology_layout_validation_cycle")
    model_ir.metadata[VALIDATION_METADATA_KEY] = ["stale"]
    model_ir.tensors = {
        name: TensorIR(name=name, dtype="FLOAT32", shape=[1])
        for name in ("a", "b")
    }
    model_ir.operators = [
        OperatorIR("CUSTOM_A", ["b"], ["a"]),
        OperatorIR("CUSTOM_B", ["a"], ["b"]),
    ]
    return model_ir


def test_two_terminal_topology_layout_validation_boundaries_are_explicit() -> None:
    lowerer = _lowerer()
    assert _raw_validation_boundaries(lowerer) == []
    locations = _runner_locations(lowerer)

    assert len(locations) == 2
    assert tuple(
        ast.literal_eval(_statement_call(block[index]).args[0])
        for block, index in locations
    ) == EXPECTED_PHASE_IDS
    for boundary_index, (block, index) in enumerate(locations):
        model_argument = EXPECTED_MODEL_ARGUMENTS[boundary_index]
        runner_call = _phase_result_owner(block[index])
        assert runner_call is not None
        assert [ast.unparse(argument) for argument in runner_call.args] == [
            model_argument
        ]
        assert runner_call.keywords == []
        assert isinstance(block[index + 1], ast.Return)
        assert ast.unparse(block[index + 1].value) == (
            f"_finalize_model_ir({model_argument})"
        )


def test_terminal_topology_then_layout_validation_effects_are_explicit() -> None:
    expected_ir = _invalid_reversed_model_ir()
    actual_ir = copy.deepcopy(expected_ir)

    sort_stats = _topologically_sort_operators(expected_ir)
    problems = validate_model_ir_layout_annotations(expected_ir)
    if problems:
        expected_ir.metadata[VALIDATION_METADATA_KEY] = list(problems)
    else:
        expected_ir.metadata.pop(VALIDATION_METADATA_KEY, None)
    actual_stats = run_topology_layout_validation(actual_ir)

    assert sort_stats == {
        "reordered_operators": 2,
        "cycle_detected": 0,
    }
    assert problems == [
        "tensor=input shape=[1, 2, 3] logical_layout=NCHW",
    ]
    assert actual_stats == {
        **sort_stats,
        "layout_validation_errors": 1,
    }
    assert actual_ir.metadata[VALIDATION_METADATA_KEY] == problems
    assert [operator.op_type for operator in actual_ir.operators] == [
        "IDENTITY",
        "RELU",
    ]
    assert [operator.op_type for operator in actual_ir.operators] == [
        operator.op_type for operator in expected_ir.operators
    ]


def test_terminal_topology_validation_preserves_cycle_and_clears_stale_errors() -> None:
    model_ir = _cyclic_model_ir()

    assert run_topology_layout_validation(model_ir) == {
        "reordered_operators": 0,
        "cycle_detected": 1,
        "layout_validation_errors": 0,
    }
    assert VALIDATION_METADATA_KEY not in model_ir.metadata
    assert [operator.op_type for operator in model_ir.operators] == [
        "CUSTOM_A",
        "CUSTOM_B",
    ]
