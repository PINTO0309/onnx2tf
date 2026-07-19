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
CONVPOOL_OUTPUT = (
    "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
)
OWNER_NAME = "optimize_convpool_output_transpose_nhwc_passthrough_chains"
TERMINAL_SINGLETON_MAXPOOL = (
    "_run_terminal_singleton_maxpool_reshape_pass_pair"
)
SAFE_REDUCTION = "_apply_safe_transpose_reduction_lite"
MUL_ADD_CONST = "_optimize_transpose_mul_add_const_prepost_nhwc_chains"
DEQUANT_HARDSIGMOID = (
    "_optimize_transpose_dequant_hardsigmoid_quantize_bridges"
)


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
    return statement.value


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
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(_call_name(child) == CONVPOOL_OUTPUT for child in statement.body)
    )
    assert len(guard.body) == 1
    result = guard.body[0]
    target = "_terminal_convpool_output_passthrough_stats"
    assert _single_target(result) == target
    call = _statement_call(result)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []
    assert sum(
        1
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CONVPOOL_OUTPUT
    ) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == target
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )

    guard_index = lowerer.body.index(guard)
    assert _call_name(lowerer.body[guard_index - 1]) == (
        TERMINAL_SINGLETON_MAXPOOL
    )
    assert _call_name(lowerer.body[guard_index + 1]) == DEQUANT_HARDSIGMOID

    assert len(guard.orelse) == 1
    fallback = guard.orelse[0]
    assert isinstance(fallback, ast.If)
    assert ast.unparse(fallback.test) == (
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )
    assert [_call_name(statement) for statement in fallback.body[:2]] == [
        SAFE_REDUCTION,
        MUL_ADD_CONST,
    ]
