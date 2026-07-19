from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_squeeze_reshape_identity_cleanup,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATHS = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_recovery_orchestration.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "singleton_reshape_orchestration.py",
)
OWNER = "run_squeeze_reshape_identity_cleanup"
RESULT_TARGETS = (
    "_layout_pass_set_1_squeeze_reshape_identity_stats",
    "_core_cleanup_squeeze_reshape_identity_stats",
    "_layout_pass_set_2_squeeze_reshape_identity_stats",
)


def _functions() -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(LOWERER_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


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


def test_squeeze_reshape_policy_schemas_and_selections_are_explicit() -> None:
    identity_only = run_squeeze_reshape_identity_cleanup(
        ModelIR("squeeze_reshape_identity_only_schema")
    )
    with_unary = run_squeeze_reshape_identity_cleanup(
        ModelIR("squeeze_reshape_with_unary_schema"),
        include_unary_passthrough=True,
    )
    assert identity_only == {"optimized_squeeze_reshape_identity_chains": 0}
    assert with_unary == {
        "optimized_squeeze_reshape_identity_chains": 0,
        "optimized_squeeze_unary_reshape_passthrough_chains": 0,
    }

    lowerer = _functions()["lower_onnx_to_ir"]
    locations = _direct_locations(lowerer.body)
    assert len(locations) == 3
    for body, index in locations:
        call = _statement_call(body[index])
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {
            "include_unary_passthrough": "True",
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
        }

    for path in ORCHESTRATION_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selections = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == OWNER
        ]
        assert len(selections) == 1


def test_all_direct_squeeze_reshape_results_are_retained_observation_only() -> None:
    lowerer = _functions()["lower_onnx_to_ir"]
    locations = _direct_locations(lowerer.body)
    assert len(locations) == 3
    assert tuple(
        _single_target(body[index]) for body, index in locations
    ) == (RESULT_TARGETS[0], None, None)

    first_body, first_index = locations[0]
    assert _single_target(first_body[first_index - 1]) == (
        "_layout_pass_set_1_instancenorm_prepost_stats"
    )
    assert _single_target(first_body[first_index + 1]) == (
        "_layout_pass_set_1_final_attention_quantized_suffix_results"
    )

    core_body, core_index = locations[1]
    assert ast.unparse(core_body[core_index]) == (
        "session.record_phase_result("
        "'cleanup.core.squeeze_reshape_identity', "
        "run_squeeze_reshape_identity_cleanup(model_ir, "
        "include_unary_passthrough=True, "
        "layout_state=session.layout_state, "
        "diagnostics=session.diagnostics))"
    )
    assert ast.unparse(core_body[core_index - 1]) == (
        "session.record_phase_result("
        "'shape_resolution.core.dynamic_reshape', "
        "_resolve_dynamic_reshape_shapes(model_ir))"
    )
    assert _call_name(core_body[core_index + 1]) == (
        "run_indexed_prune_reconcile_cleanup"
    )

    final_body, final_index = locations[2]
    assert ast.unparse(final_body[final_index]) == (
        "session.record_phase_result("
        "'cleanup.layout_pass_set_2.squeeze_reshape_identity', "
        "run_squeeze_reshape_identity_cleanup(model_ir, "
        "include_unary_passthrough=True, "
        "layout_state=session.layout_state, "
        "diagnostics=session.diagnostics))"
    )
    convergence_loop = final_body[final_index - 1]
    assert isinstance(convergence_loop, ast.For)
    assert ast.unparse(convergence_loop.target) == "_"
    assert ast.unparse(convergence_loop.iter) == "range(2)"
    assert any(
        isinstance(node, ast.Name)
        and node.id == "rewritten_instnorm"
        for node in ast.walk(convergence_loop)
    )
    assert _call_name(final_body[final_index + 1]) == (
        "run_indexed_prune_reconcile_cleanup"
    )

    for target in RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )
