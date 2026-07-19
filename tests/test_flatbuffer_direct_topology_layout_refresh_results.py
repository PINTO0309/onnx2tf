from __future__ import annotations

import ast
import copy
import importlib
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _topologically_sort_operators,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    infer_model_ir_logical_layouts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_MODULE = "onnx2tf.tflite_builder.passes.topology_layout_refresh"
RUNNER = "run_topology_layout_refresh"
SORT_OWNER = "_topologically_sort_operators"
LAYOUT_OWNER = "infer_model_ir_logical_layouts"
EXPECTED_TARGETS = (
    "_fallback_dynamic_rank1_topology_layout_stats",
    "_fallback_broadcast_topology_layout_stats",
    "_absolute_final_topology_layout_stats",
    "_final_convinteger_topology_layout_stats",
    "_final_instancenorm_topology_layout_stats",
    "_final_broadcast_topology_layout_stats",
)
EXPECTED_MODEL_ARGUMENTS = (
    "fallback_ir",
    "fallback_ir",
    "model_ir",
    "model_ir",
    "model_ir",
    "model_ir",
)
EXPECTED_PREDECESSOR_TARGETS = (
    "_fallback_dynamic_rank1_stats",
    "_fallback_broadcast_static_shape_stats",
    "_absolute_final_dynamic_rank1_stats",
    "_final_convinteger_static_shape_stats",
    "_final_instancenorm_static_shape_stats",
    "_final_broadcast_static_shape_stats",
)
RESULT_SCHEMA = {
    "reordered_operators": 0,
    "cycle_detected": 0,
}


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _pipeline_blocks(
    statements: list[ast.stmt],
) -> list[list[ast.stmt]]:
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


def _raw_pair_locations(
    lowerer: ast.FunctionDef,
) -> list[tuple[list[ast.stmt], int]]:
    return sorted(
        [
            (block, index)
            for block in _pipeline_blocks(lowerer.body)
            for index in range(len(block) - 1)
            if _call_name(block[index]) == SORT_OWNER
            and _call_name(block[index + 1]) == LAYOUT_OWNER
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
            if _call_name(statement) == RUNNER
        ],
        key=lambda item: item[0][item[1]].lineno,
    )


def _model_ir() -> ModelIR:
    model_ir = ModelIR("topology_layout_refresh")
    model_ir.inputs = ["input_nhwc"]
    model_ir.outputs = ["output_nhwc"]
    model_ir.tensors = {
        name: TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 4, 5, 3],
            shape_signature=[1, 4, 5, 3],
        )
        for name in ("input_nhwc", "middle_nhwc", "output_nhwc")
    }
    model_ir.operators = [
        OperatorIR("RELU", ["middle_nhwc"], ["output_nhwc"]),
        OperatorIR("IDENTITY", ["input_nhwc"], ["middle_nhwc"]),
    ]
    return model_ir


def _snapshot(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(
            (op.op_type, tuple(op.inputs), tuple(op.outputs))
            for op in model_ir.operators
        ),
        tuple(
            (name, tensor.logical_layout)
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        tuple(
            sorted(
                (str(name), str(layout))
                for name, layout in dict(
                    model_ir.metadata.get("onnx_public_layout_map", {})
                ).items()
            )
        ),
    )


def test_topology_sort_and_layout_inference_contracts_are_explicit() -> None:
    assert _topologically_sort_operators(ModelIR("sort_schema")) == RESULT_SCHEMA

    model_ir = _model_ir()
    sort_stats = _topologically_sort_operators(model_ir)
    layout_map = infer_model_ir_logical_layouts(model_ir)

    assert sort_stats["reordered_operators"] == 2
    assert sort_stats["cycle_detected"] == 0
    assert [op.op_type for op in model_ir.operators] == ["IDENTITY", "RELU"]
    assert set(layout_map) == set(model_ir.tensors)
    assert all(
        tensor.logical_layout == layout_map[name]
        for name, tensor in model_ir.tensors.items()
    )


def test_six_raw_topology_layout_pairs_are_explicit() -> None:
    locations = _raw_pair_locations(_lowerer())
    assert len(locations) == 6
    assert tuple(
        ast.unparse(_statement_call(block[index]).args[0])
        for block, index in locations
    ) == EXPECTED_MODEL_ARGUMENTS
    assert tuple(
        _single_target(block[index - 1]) for block, index in locations
    ) == EXPECTED_PREDECESSOR_TARGETS
    for block, index in locations:
        sort_call = _statement_call(block[index])
        layout_call = _statement_call(block[index + 1])
        assert sort_call is not None
        assert layout_call is not None
        assert sort_call.keywords == []
        assert layout_call.keywords == []
        assert ast.unparse(sort_call.args[0]) == ast.unparse(layout_call.args[0])


@pytest.mark.xfail(
    strict=True,
    reason="the six topology/layout pairs do not yet share an explicit owner",
)
def test_topology_layout_runner_preserves_effects_and_small_results() -> None:
    owner_module = importlib.import_module(OWNER_MODULE)
    runner = getattr(owner_module, RUNNER)
    expected_ir = _model_ir()
    actual_ir = copy.deepcopy(expected_ir)

    expected_result = _topologically_sort_operators(expected_ir)
    infer_model_ir_logical_layouts(expected_ir)
    actual_result = runner(actual_ir)

    assert actual_result == expected_result
    assert actual_result == {
        "reordered_operators": 2,
        "cycle_detected": 0,
    }
    assert _snapshot(actual_ir) == _snapshot(expected_ir)

    lowerer = _lowerer()
    assert _raw_pair_locations(lowerer) == []
    locations = _runner_locations(lowerer)
    assert len(locations) == 6
    assert tuple(
        _single_target(block[index]) for block, index in locations
    ) == EXPECTED_TARGETS
    assert tuple(
        ast.unparse(_statement_call(block[index]).args[0])
        for block, index in locations
    ) == EXPECTED_MODEL_ARGUMENTS
    assert tuple(
        _single_target(block[index - 1]) for block, index in locations
    ) == EXPECTED_PREDECESSOR_TARGETS
    for block, index in locations:
        call = _statement_call(block[index])
        assert call is not None
        assert call.keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
