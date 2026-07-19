from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "quantization_cleanup.py"
)
WRAPPER = "_optimize_transpose_dequantize_mean_quantize_bridges"
INNER_OWNER = WRAPPER
RESULT_TARGET = "_layout_pass_set_1_dequant_mean_quantize_stats"
PREVIOUS_TARGET = "_layout_pass_set_1_attention_quantized_safe_binary_results"
FOLLOWING_TARGET = "_layout_pass_set_1_qlinear_mean_concat_results"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
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


def test_dequant_mean_quantize_schema_cleanup_and_wrapper_are_explicit() -> None:
    owner = _functions(OWNER_PATH)[INNER_OWNER]
    owner_source = ast.get_source_segment(
        OWNER_PATH.read_text(encoding="utf-8"), owner
    )
    assert owner_source is not None
    assert owner_source.count("_prune_unused_tensors(model_ir)") == 3
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'moved_transpose_dequantize_mean_quantize_bridges': "
        "int(moved_bridges)}"
    )
    early_returns = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Dict)
        and ast.unparse(node.value)
        == "{'moved_transpose_dequantize_mean_quantize_bridges': 0}"
    ]
    assert len(early_returns) == 2

    wrapper = _functions(LOWERER_PATH)[WRAPPER]
    assert len(wrapper.body) == 1
    wrapper_return = wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    wrapper_call = wrapper_return.value
    assert isinstance(wrapper_call, ast.Call)
    assert isinstance(wrapper_call.func, ast.Name)
    assert wrapper_call.func.id == f"{WRAPPER}_pass"
    assert [ast.unparse(argument) for argument in wrapper_call.args] == [
        "model_ir"
    ]
    assert wrapper_call.keywords == []


def test_dequant_mean_quantize_result_is_retained_observation_only() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    layout_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(_call_name(child) == WRAPPER for child in statement.body)
    )
    result_index = next(
        index
        for index, statement in enumerate(layout_guard.body)
        if _call_name(statement) == WRAPPER
    )
    result = layout_guard.body[result_index]
    assert _single_target(result) is None
    call = _statement_call(result)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []
    assert _single_target(layout_guard.body[result_index - 1]) == (
        PREVIOUS_TARGET
    )
    assert _single_target(layout_guard.body[result_index + 1]) == (
        FOLLOWING_TARGET
    )
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == WRAPPER
        for node in ast.walk(lowerer)
    ) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
