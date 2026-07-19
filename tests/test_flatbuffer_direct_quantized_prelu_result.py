from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.quantized_prelu import (
    run_quantized_prelu_cleanup,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "duplicate_quantized_prelu_orchestration.py"
)
OWNER = "run_quantized_prelu_cleanup"
RESULT_TARGET = "_layout_pass_set_1_quantized_prelu_stats"


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


def test_quantized_prelu_result_schema_and_all_selections_are_explicit() -> None:
    assert run_quantized_prelu_cleanup(ModelIR("quantized_prelu_result_schema")) == {
        "removed_transpose_dequant_prelu_quantize_bridges": 0,
        "removed_transpose_dequant_prelu_transpose_bridges": 0,
        "folded_dequant_prelu_quantize_chains": 0,
        "folded_dequant_prelu_depthwise_quantize_chains": 0,
    }

    locations = _direct_locations(_lowerer().body)
    assert len(locations) == 1
    body, index = locations[0]
    call = _statement_call(body[index])
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }

    orchestration = ast.parse(ORCHESTRATION_PATH.read_text(encoding="utf-8"))
    nested = []
    for node in ast.walk(orchestration):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RecoveryInvocation"
        ):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords}
        callback = keywords.get("callback")
        if isinstance(callback, ast.Name) and callback.id == OWNER:
            nested.append(keywords)
    assert len(nested) == 1
    assert ast.unparse(nested[0]["args"]) == "(context.model_ir,)"
    assert ast.unparse(nested[0]["keyword_args"]) == "shared_keyword_args"


def test_direct_quantized_prelu_result_is_retained_observation_only() -> None:
    lowerer = _lowerer()
    locations = _direct_locations(lowerer.body)
    assert len(locations) == 1
    body, index = locations[0]
    assert _single_target(body[index]) == RESULT_TARGET
    assert _single_target(body[index - 1]) == (
        "_layout_pass_set_1_attention_gate_qdq_results"
    )
    assert _call_name(body[index + 1]) == (
        "_optimize_dequant_transposeconv_quantize_chains"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
