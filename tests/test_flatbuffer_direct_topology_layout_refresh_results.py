from __future__ import annotations

import ast
import copy
from pathlib import Path

import onnx2tf.tflite_builder.passes.topology_layout_refresh as topology_layout_refresh_module

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _topologically_sort_operators,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    infer_model_ir_logical_layouts,
)
from onnx2tf.tflite_builder.passes.topology_layout_refresh import (
    run_topology_layout_refresh,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
RUNNER = "run_topology_layout_refresh"
SORT_OWNER = "_topologically_sort_operators"
LAYOUT_OWNER = "infer_model_ir_logical_layouts"
EXPECTED_PHASE_IDS = (
    "topology_layout.fallback.post_dynamic_rank1",
    "topology_layout.fallback.broadcast",
    "topology_layout.primary.absolute_final",
    "topology_layout.primary.final_convinteger",
    "topology_layout.primary.final_instancenorm",
    "topology_layout.primary.final_broadcast",
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
            if isinstance(_phase_result_owner(statement), ast.Call)
            and isinstance(_phase_result_owner(statement).func, ast.Name)
            and _phase_result_owner(statement).func.id == RUNNER
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


def test_six_topology_layout_refresh_boundaries_are_explicit() -> None:
    lowerer = _lowerer()
    assert _raw_pair_locations(lowerer) == []
    locations = _runner_locations(lowerer)
    assert len(locations) == 6
    assert tuple(
        ast.literal_eval(_statement_call(block[index]).args[0])
        for block, index in locations
    ) == EXPECTED_PHASE_IDS
    assert tuple(
        _single_target(block[index - 1]) for block, index in locations
    ) == EXPECTED_PREDECESSOR_TARGETS
    assert tuple(
        ast.unparse(_phase_result_owner(block[index]).args[0])
        for block, index in locations
    ) == EXPECTED_MODEL_ARGUMENTS
    for block, index in locations:
        call = _phase_result_owner(block[index])
        assert call is not None
        assert call.keywords == []


def test_topology_layout_runner_preserves_effects_and_small_results() -> None:
    expected_ir = _model_ir()
    actual_ir = copy.deepcopy(expected_ir)

    expected_result = _topologically_sort_operators(expected_ir)
    infer_model_ir_logical_layouts(expected_ir)
    actual_result = run_topology_layout_refresh(actual_ir)

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
        ast.literal_eval(_statement_call(block[index]).args[0])
        for block, index in locations
    ) == EXPECTED_PHASE_IDS
    assert tuple(
        _single_target(block[index - 1]) for block, index in locations
    ) == EXPECTED_PREDECESSOR_TARGETS
    assert tuple(
        ast.unparse(_phase_result_owner(block[index]).args[0])
        for block, index in locations
    ) == EXPECTED_MODEL_ARGUMENTS
    for block, index in locations:
        call = _phase_result_owner(block[index])
        assert call is not None
        assert call.keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and node.id.endswith("_topology_layout_stats")
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


def test_topology_layout_runner_keeps_layout_refresh_after_cycle(
    monkeypatch,
) -> None:
    model_ir = ModelIR("topology_cycle")
    model_ir.operators = [
        OperatorIR("IDENTITY", ["second"], ["first"]),
        OperatorIR("IDENTITY", ["first"], ["second"]),
    ]
    layout_refreshes = 0

    def record_layout_refresh(target_model_ir: ModelIR) -> dict[str, str]:
        nonlocal layout_refreshes
        assert target_model_ir is model_ir
        layout_refreshes += 1
        return {}

    monkeypatch.setattr(
        topology_layout_refresh_module,
        "infer_model_ir_logical_layouts",
        record_layout_refresh,
    )

    assert run_topology_layout_refresh(model_ir) == {
        "reordered_operators": 0,
        "cycle_detected": 1,
    }
    assert layout_refreshes == 1
