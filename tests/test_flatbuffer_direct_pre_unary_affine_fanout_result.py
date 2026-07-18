from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.pre_unary_affine_fanout_layout import (
    optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PASSES_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
ORCHESTRATION_SELECTIONS = {
    "layout_attention_quantized_suffix_orchestration.py": (
        "LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[1]",
        "context",
    ),
    "attention_recovery_orchestration.py": (
        "PREADD_MEAN_ATTENTION_PASS_IDS[4]",
        "context",
    ),
}
OWNER = "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains"
NESTED_OWNER = (
    "optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains"
)
RESULT_TARGET = "_layout_pass_set_1_pre_unary_affine_fanout_stats"


def _lowerer() -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(LOWERER_PATH.read_text(encoding="utf-8")).body
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


def _direct_locations(
    statements: list[ast.stmt],
) -> list[tuple[list[ast.stmt], int]]:
    locations: list[tuple[list[ast.stmt], int]] = []
    for index, statement in enumerate(statements):
        if _call_name(statement) == OWNER:
            locations.append((statements, index))
        if isinstance(statement, (ast.For, ast.If, ast.While)):
            locations.extend(_direct_locations(statement.body))
            locations.extend(_direct_locations(statement.orelse))
        elif isinstance(statement, ast.With):
            locations.extend(_direct_locations(statement.body))
        elif isinstance(statement, ast.Try):
            locations.extend(_direct_locations(statement.body))
            locations.extend(_direct_locations(statement.orelse))
            locations.extend(_direct_locations(statement.finalbody))
            for handler in statement.handlers:
                locations.extend(_direct_locations(handler.body))
    return locations


def test_pre_unary_affine_fanout_schema_and_selections_are_explicit() -> None:
    assert optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(
        ModelIR("pre_unary_affine_fanout_result_schema")
    ) == {
        "optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains": 0
    }

    locations = _direct_locations(_lowerer().body)
    assert len(locations) == 1
    body, index = locations[0]
    call = _statement_call(body[index])
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []

    for filename, (expected_pass_id, expected_context) in (
        ORCHESTRATION_SELECTIONS.items()
    ):
        tree = ast.parse((PASSES_PATH / filename).read_text(encoding="utf-8"))
        nested = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_model_invocation"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == NESTED_OWNER
        ]
        assert len(nested) == 1
        assert ast.unparse(nested[0].args[0]) == expected_pass_id
        assert ast.unparse(nested[0].args[2]) == expected_context
        assert nested[0].keywords == []


def test_direct_pre_unary_affine_fanout_result_is_retained_observation_only() -> (
    None
):
    lowerer = _lowerer()
    locations = _direct_locations(lowerer.body)
    assert len(locations) == 1
    body, index = locations[0]
    assert _single_target(body[index]) == RESULT_TARGET
    assert _single_target(body[index - 1]) == (
        "_layout_pass_set_1_affine_prepost_stats"
    )
    assert _call_name(body[index + 1]) == (
        "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
