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
    / "hardswish_se_layout.py"
)
TERMINAL_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_activation_bridge_orchestration.py"
)
HARDSWISH_SE = (
    "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
)
OWNER_NAME = (
    "optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
)
SINET_TERMINAL_TARGET = "_terminal_clamp_sinet_layout_results"
RESULT_TARGET = "_terminal_sinet_hardswish_se_stats"
DEQUANT_TARGET = "_terminal_dequant_hardsigmoid_bridge_stats"
LATE_RESULT_TARGET = "_terminal_hardswish_se_stats"
EXPECTED_PHASE_IDS = (
    "cleanup.terminal.sinet_hardswish_se",
    "cleanup.terminal.dequant_hardsigmoid_bridge",
)
EXPECTED_OWNER_EXPRESSIONS = (
    (
        "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_"
        "nhwc_chains(model_ir)"
    ),
    "_optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir)",
)
SINET_RECOVERY_TARGET = "_terminal_sinet_singleton_reshape_results"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _direct_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    value = statement.value
    if not isinstance(value, ast.Call):
        return None
    if (
        isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "session"
        and value.func.attr == "record_phase_result"
        and len(value.args) == 2
        and isinstance(value.args[1], ast.Call)
    ):
        value = value.args[1]
    if not isinstance(value.func, ast.Name):
        return None
    return value if value.func.id == HARDSWISH_SE else None


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_hardswish_se_wrapper_schema_and_cleanup_are_explicit() -> None:
    wrapper = _functions(LOWERER_PATH)[HARDSWISH_SE]
    assert len(wrapper.body) == 1
    wrapper_return = wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{HARDSWISH_SE}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir"
    ]
    assert wrapper_return.value.keywords == []

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    top_level_cleanup = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_prune_unused_tensors"
    ]
    assert len(top_level_cleanup) == 1
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains': "
        "int(rewritten)}"
    )


def test_hardswish_se_has_exactly_two_production_forms() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    owner_statements = [
        statement
        for statement in lowerer.body
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == HARDSWISH_SE
            for node in ast.walk(statement)
        )
    ]
    assert len(owner_statements) == 1

    first_call = next(
        node
        for node in ast.walk(owner_statements[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == HARDSWISH_SE
    )
    assert [ast.unparse(argument) for argument in first_call.args] == ["model_ir"]
    assert first_call.keywords == []

    terminal_owner = _functions(TERMINAL_OWNER_PATH)[
        "run_terminal_activation_bridge_cleanup"
    ]
    late_call = next(
        node
        for node in ast.walk(terminal_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_hardswish_se_layout_summary"
    )
    assert [ast.unparse(argument) for argument in late_call.args] == [
        "context.model_ir"
    ]
    assert late_call.keywords == []
    assert not any(
        isinstance(node, ast.Name) and node.id == LATE_RESULT_TARGET
        for node in ast.walk(lowerer)
    )


def test_terminal_hardswish_hardsigmoid_results_use_phase_result_store() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    records = [
        statement
        for statement in lowerer.body
        if _phase_id(statement) in EXPECTED_PHASE_IDS
    ]
    indices = [lowerer.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == EXPECTED_PHASE_IDS
    assert tuple(ast.unparse(statement.value.args[1]) for statement in records) == (
        EXPECTED_OWNER_EXPRESSIONS
    )
    assert indices == [indices[0], indices[0] + 1]
    assert _single_target(lowerer.body[indices[0] - 1]) == SINET_TERMINAL_TARGET
    assert _single_target(lowerer.body[indices[-1] + 1]) == SINET_RECOVERY_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {RESULT_TARGET, DEQUANT_TARGET}
        for node in ast.walk(lowerer)
    )
