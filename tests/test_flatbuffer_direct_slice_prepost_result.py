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
    / "slice_prepost_layout.py"
)
SLICE_PREPOST = "_optimize_transpose_slice_prepost_nhwc_passthrough_chains"
OWNER_NAME = "optimize_transpose_slice_prepost_nhwc_passthrough_chains"
RESULT_TARGET = "_final_slice_prepost_passthrough_stats"
PREVIOUS_TARGET = "_final_slice_concat_recovery_results"
COMPOSITE_TARGET = "_final_slice_pre_concat_layout_results"
COMPOSITE_OWNER = "run_final_slice_pre_concat_layout_cleanup"
NEXT_TARGET = "_terminal_concat_bridge_layout_results"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _direct_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    function = statement.value.func
    if not isinstance(function, ast.Name) or function.id != SLICE_PREPOST:
        return None
    return statement.value


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def test_slice_prepost_wrapper_schema_and_guarded_cleanup_are_explicit() -> None:
    wrapper = _functions(LOWERER_PATH)[SLICE_PREPOST]
    assert len(wrapper.body) == 1
    wrapper_return = wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{SLICE_PREPOST}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir"
    ]
    assert wrapper_return.value.keywords == []

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    cleanup_guards = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimized > 0"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(cleanup_guards) == 1
    assert not any(
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_prune_unused_tensors"
        for statement in owner.body
    )
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'optimized_transpose_slice_prepost_nhwc_passthrough_chains': "
        "int(optimized)}"
    )


def test_final_slice_prepost_result_moves_to_final_pair_composite() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    production_results = [
        statement
        for statement in lowerer.body
        if _direct_call(statement) is not None
    ]
    assert production_results == []
    assert not any(
        isinstance(node, ast.Name) and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )

    composite = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    composite_index = lowerer.body.index(composite)
    assert _direct_call(composite) is None
    assert isinstance(composite.value, ast.Call)
    assert isinstance(composite.value.func, ast.Name)
    assert composite.value.func.id == COMPOSITE_OWNER
    assert _single_target(lowerer.body[composite_index - 1]) == PREVIOUS_TARGET
    assert _single_target(lowerer.body[composite_index + 1]) == NEXT_TARGET
