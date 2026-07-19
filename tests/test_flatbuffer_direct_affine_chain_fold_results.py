from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.affine_chain_fold import (
    optimize_fold_mul_add_mul_affine_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_affine_concat_split_recovery_orchestration.py"
)
OWNER = "_optimize_fold_mul_add_mul_affine_chains"
NESTED_OWNER = "optimize_fold_mul_add_mul_affine_chains"
RESULT_TARGETS = (
    "_layout_pass_set_1_initial_affine_chain_fold_stats",
    "_layout_pass_set_1_post_binary_affine_chain_fold_stats",
)


def _lowerer() -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(LOWERER_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    call = statement.value if isinstance(statement.value, ast.Call) else None
    if (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        return call.args[1]
    return call


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


def test_affine_chain_fold_schema_and_all_selections_are_explicit() -> None:
    assert optimize_fold_mul_add_mul_affine_chains(
        ModelIR("affine_chain_fold_result_schema")
    ) == {"optimized_fold_mul_add_mul_affine_chains": 0}

    locations = _direct_locations(_lowerer().body)
    assert len(locations) == 2
    for body, index in locations:
        call = _statement_call(body[index])
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}

    orchestration = ast.parse(ORCHESTRATION_PATH.read_text(encoding="utf-8"))
    nested = [
        node
        for node in ast.walk(orchestration)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_model_invocation"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == NESTED_OWNER
    ]
    assert len(nested) == 1
    assert ast.unparse(nested[0].args[0]) == (
        "TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[0]"
    )
    assert ast.unparse(nested[0].args[2]) == "context"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in nested[0].keywords
    } == {"include_layout": "True"}


def test_direct_affine_chain_fold_results_are_retained_observation_only() -> None:
    lowerer = _lowerer()
    locations = _direct_locations(lowerer.body)
    assert len(locations) == 2
    assert tuple(
        _single_target(body[index]) for body, index in locations
    ) == (None, None)

    first_body, first_index = locations[0]
    assert _single_target(first_body[first_index - 1]) == (
        "_layout_pass_set_1_initial_attention_recovery_results"
    )
    assert _call_name(first_body[first_index + 1]) == (
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains"
    )

    second_body, second_index = locations[1]
    assert _single_target(second_body[second_index - 1]) == (
        "_layout_pass_set_1_post_binary_attention_recovery_results"
    )
    assert _single_target(second_body[second_index + 1]) == (
        "_layout_pass_set_1_attention_quantized_suffix_results"
    )

    for target in RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            for node in ast.walk(lowerer)
        )
