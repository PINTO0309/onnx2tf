from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.quantized_transpose_conv import (
    _optimize_dequant_transposeconv_quantize_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_attention_quantized_suffix_orchestration.py"
)
OWNER = "_optimize_dequant_transposeconv_quantize_chains"
RESULT_TARGETS = (
    "_layout_pass_set_1_dequant_transposeconv_quantize_stats",
    "_layout_pass_set_2_dequant_transposeconv_quantize_stats",
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


def test_quantized_transpose_conv_schema_and_all_selections_are_explicit() -> None:
    assert _optimize_dequant_transposeconv_quantize_chains(
        ModelIR("quantized_transpose_conv_result_schema")
    ) == {"folded_dequant_transposeconv_quantize_chains": 0}

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
        and node.args[1].id == OWNER
    ]
    assert len(nested) == 1
    assert ast.unparse(nested[0].args[0]) == (
        "LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[6]"
    )
    assert ast.unparse(nested[0].args[2]) == "context"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in nested[0].keywords
    } == {"include_layout": "True"}


def test_direct_quantized_transpose_conv_results_are_retained_observation_only() -> None:
    lowerer = _lowerer()
    locations = _direct_locations(lowerer.body)
    assert len(locations) == 2
    assert tuple(
        _single_target(body[index]) for body, index in locations
    ) == (None, None)

    first_body, first_index = locations[0]
    assert ast.unparse(first_body[first_index - 1]) == (
        "session.record_phase_result("
        "'cleanup.layout_pass_set_1.quantized_prelu', "
        "run_quantized_prelu_cleanup(model_ir, "
        "layout_state=session.layout_state, "
        "diagnostics=session.diagnostics))"
    )
    assert ast.unparse(first_body[first_index]) == (
        "session.record_phase_result("
        "'cleanup.layout_pass_set_1.dequant_transposeconv_quantize', "
        "_optimize_dequant_transposeconv_quantize_chains(model_ir, "
        "layout_state=session.layout_state))"
    )
    assert ast.unparse(first_body[first_index + 1]) == (
        "session.record_phase_result("
        "'cleanup.layout_pass_set_1.quantized_reshape', "
        "run_quantized_reshape_cleanup(model_ir, "
        "layout_state=session.layout_state, "
        "diagnostics=session.diagnostics))"
    )

    second_body, second_index = locations[1]
    assert ast.unparse(second_body[second_index]) == (
        "session.record_phase_result("
        "'cleanup.layout_pass_set_2.dequant_transposeconv_quantize', "
        "_optimize_dequant_transposeconv_quantize_chains(model_ir, "
        "layout_state=session.layout_state))"
    )
    assert _single_target(second_body[second_index - 1]) == (
        "_layout_pass_set_2_attention_gate_qdq_results"
    )
    assert _single_target(second_body[second_index + 1]) == (
        "_layout_pass_set_2_quantized_activation_binary_results"
    )

    for target in RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )
