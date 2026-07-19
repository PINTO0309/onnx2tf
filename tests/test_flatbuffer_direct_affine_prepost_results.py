from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    optimize_transpose_mul_add_const_prepost_nhwc_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PASSES_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
ORCHESTRATION_SELECTIONS = {
    "terminal_affine_concat_split_recovery_orchestration.py": (
        "TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[1]",
        "context",
    ),
    "layout_attention_quantized_suffix_orchestration.py": (
        "LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[0]",
        "context",
    ),
    "attention_recovery_orchestration.py": (
        "PREADD_MEAN_ATTENTION_PASS_IDS[3]",
        "context",
    ),
}
LATE_BINARY_PATH = PASSES_PATH / "late_binary_layout_recovery.py"
OWNER = "_optimize_transpose_mul_add_const_prepost_nhwc_chains"
NESTED_OWNER = "optimize_transpose_mul_add_const_prepost_nhwc_chains"
RESULT_TARGETS = (
    "_layout_pass_set_1_affine_prepost_stats",
    "_no_layout_fallback_affine_prepost_stats",
    "_no_layout_final_affine_prepost_stats",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


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


def test_affine_prepost_schema_and_all_selections_are_explicit() -> None:
    assert optimize_transpose_mul_add_const_prepost_nhwc_chains(
        ModelIR("affine_prepost_result_schema")
    ) == {"optimized_transpose_mul_add_const_prepost_nhwc_chains": 0}

    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    locations = _direct_locations(lowerer.body)
    assert len(locations) == 3
    for body, index in locations:
        call = _statement_call(body[index])
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}

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
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in nested[0].keywords
        } == {"include_layout": "True"}

    late_binary = _functions(LATE_BINARY_PATH)["run_late_binary_layout_recovery"]
    assignments = [
        statement
        for statement in late_binary.body
        if _single_target(statement) == "affine_prepost_stats"
    ]
    assert len(assignments) == 1
    late_call = _statement_call(assignments[0])
    assert late_call is not None
    assert isinstance(late_call.func, ast.Name)
    assert late_call.func.id == NESTED_OWNER
    assert [ast.unparse(argument) for argument in late_call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in late_call.keywords
    } == {"layout_state": "layout_state"}
    assert any(
        isinstance(node, ast.Name)
        and node.id == "affine_prepost_stats"
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(late_binary)
    )


def test_all_direct_affine_prepost_results_are_retained_observation_only() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    locations = _direct_locations(lowerer.body)
    assert len(locations) == 3
    assert tuple(
        _single_target(body[index]) for body, index in locations
    ) == RESULT_TARGETS

    initial_body, initial_index = locations[0]
    assert _single_target(initial_body[initial_index - 1]) == (
        "_layout_pass_set_1_initial_affine_chain_fold_stats"
    )
    assert _call_name(initial_body[initial_index + 1]) == (
        "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains"
    )

    fallback_body, fallback_index = locations[1]
    assert fallback_index == len(fallback_body) - 1
    assert _call_name(fallback_body[fallback_index - 1]) == (
        "_apply_safe_transpose_reduction_lite"
    )

    final_body, final_index = locations[2]
    assert _single_target(final_body[final_index - 1]) == (
        "_no_layout_final_se_fc_stats"
    )
    assert _call_name(final_body[final_index + 1]) == (
        "_topologically_sort_operators"
    )

    for target in RESULT_TARGETS[:2]:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )
