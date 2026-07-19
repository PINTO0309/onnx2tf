from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
RESULT_TARGET = "_layout_pass_set_2_dequant_transposeconv_quantize_stats"
PHASE_ID = "cleanup.layout_pass_set_2.dequant_transposeconv_quantize"
OWNER_EXPRESSION = (
    "_optimize_dequant_transposeconv_quantize_chains(model_ir, "
    "layout_state=session.layout_state)"
)
PREDECESSOR_TARGET = "_layout_pass_set_2_attention_gate_qdq_results"
SUCCESSOR_TARGET = "_layout_pass_set_2_quantized_activation_binary_results"


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _phase_id(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        node
        for node in lowerer.body
        if isinstance(node, ast.If)
        and any(
            _single_target(statement) == RESULT_TARGET
            or _phase_id(statement) == PHASE_ID
            for statement in node.body
        )
    )


def test_layout_pass_set_2_quantized_result_is_guarded_and_unconsumed() -> None:
    lowerer = _lowerer()
    guard = _guard(lowerer)
    index = next(
        index
        for index, statement in enumerate(guard.body)
        if _phase_id(statement) == PHASE_ID
    )
    record = guard.body[index]

    assert ast.unparse(guard.test) == "optimize_layout_transpose_chains"
    assert guard.orelse == []
    assert isinstance(record, ast.Expr)
    assert ast.unparse(record) == (
        f"session.record_phase_result('{PHASE_ID}', {OWNER_EXPRESSION})"
    )
    assert _single_target(guard.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(guard.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )


def test_layout_pass_set_2_quantized_result_local_is_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
