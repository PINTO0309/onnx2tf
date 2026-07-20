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
    / "convpool_output_passthrough_compat.py"
)
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_affine_final_shape_terminal_convpool_orchestration.py"
)
CONVPOOL_OUTPUT = (
    "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
)
OWNER_NAME = "optimize_convpool_output_transpose_nhwc_passthrough_chains"
TERMINAL_FANOUT_SINGLETON = "run_late_affine_final_shape_terminal_convpool_cleanup"
SAFE_REDUCTION = "_apply_safe_transpose_reduction_lite"
MUL_ADD_CONST = "_optimize_transpose_mul_add_const_prepost_nhwc_chains"
DEQUANT_HARDSIGMOID = (
    "_optimize_transpose_dequant_hardsigmoid_quantize_bridges"
)
LATE_DEQUANT_COMPOSITE = "run_late_dequant_swish_layout_tail_cleanup"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
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


def test_convpool_output_schema_and_unconditional_cleanup_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[CONVPOOL_OUTPUT]
    assert len(lowerer_wrapper.body) == 1
    wrapper_return = lowerer_wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{CONVPOOL_OUTPUT}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir",
    ]
    assert wrapper_return.value.keywords == []

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    unconditional_cleanup = [
        statement
        for statement in owner.body
        if _call_name(statement) == "_prune_unused_tensors"
    ]
    assert len(unconditional_cleanup) == 1
    assert not any(
        isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
        for statement in owner.body
    )
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'optimized_convpool_output_transpose_nhwc_passthrough_chains': int(rewritten)}"
    )


def test_lowerer_retains_guarded_convpool_output_result() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    composite = next(
        statement
        for statement in lowerer.body
        if _single_target(statement)
        == "_late_affine_final_shape_terminal_convpool_results"
    )
    assert _call_name(composite) == TERMINAL_FANOUT_SINGLETON
    composite_call = _statement_call(composite)
    assert composite_call is not None
    assert [ast.unparse(argument) for argument in composite_call.args] == [
        "late_final_shape_boundary_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in composite_call.keywords
    } == {
        "optimize_layout_transpose_chains": "optimize_layout_transpose_chains"
    }

    orchestration_owner = _functions(ORCHESTRATION_PATH)[
        TERMINAL_FANOUT_SINGLETON
    ]
    guard = next(
        statement
        for statement in orchestration_owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
    )
    assert len(guard.body) == 1
    result = guard.body[0]
    target = "_terminal_convpool_output_passthrough_stats"
    assert _single_target(result) == "convpool_results"
    call = _statement_call(result)
    assert call is not None
    assert isinstance(call.func, ast.Name)
    assert call.func.id == OWNER_NAME
    assert [ast.unparse(argument) for argument in call.args] == [
        "context.pass_context.model_ir"
    ]
    assert call.keywords == []
    assert sum(
        1
        for node in ast.walk(orchestration_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER_NAME
    ) == 1
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CONVPOOL_OUTPUT
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == target
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )

    composite_index = lowerer.body.index(composite)
    fallback = lowerer.body[composite_index + 1]
    assert isinstance(fallback, ast.If)
    assert ast.unparse(fallback.test) == (
        "not optimize_layout_transpose_chains and "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )
    assert _call_name(fallback.body[0]) == SAFE_REDUCTION
    assert _call_name(fallback.body[1]) == MUL_ADD_CONST
    assert _call_name(lowerer.body[composite_index + 2]) == (
        LATE_DEQUANT_COMPOSITE
    )
