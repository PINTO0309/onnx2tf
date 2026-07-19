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
    / "shape_extract_layout.py"
)
SHAPE_EXTRACT = "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
OWNER_NAME = "optimize_transpose_shape_extract_nhwc_to_nchw_chains"
PRE_CONCAT = "_optimize_transpose_pre_concat_nhwc_chains"


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


def test_shape_extract_schema_and_positive_cleanup_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[SHAPE_EXTRACT]
    assert len(lowerer_wrapper.body) == 1
    wrapper_return = lowerer_wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{SHAPE_EXTRACT}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir",
    ]
    assert wrapper_return.value.keywords == []

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    cleanup_guards = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(cleanup_guards) == 1
    cleanup_guard = cleanup_guards[0]
    assert ast.unparse(cleanup_guard.test) == "optimized > 0"
    assert [
        node.func.id
        for node in ast.walk(cleanup_guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ] == ["_prune_unused_tensors"]
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == (
        "{'optimized_transpose_shape_extract_nhwc_to_nchw_chains': int(optimized)}"
    )


def test_lowerer_retains_all_shape_extract_results() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_results = [
        statement
        for statement in lowerer.body
        if _call_name(statement) == SHAPE_EXTRACT
    ]
    assert len(direct_results) == 2
    expected_targets = [
        "_late_pre_qkv_shape_extract_stats",
        "_late_pre_layout_cluster_shape_extract_stats",
    ]
    assert [_single_target(statement) for statement in direct_results] == (
        expected_targets
    )
    for statement in direct_results:
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert call.keywords == []
    for target in expected_targets:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )

    pre_qkv_index = lowerer.body.index(direct_results[0])
    assert _single_target(lowerer.body[pre_qkv_index - 1]) == (
        "_terminal_affine_slice_spp_results"
    )
    assert _single_target(lowerer.body[pre_qkv_index + 1]) == (
        "_late_qkv_stats"
    )

    late_layout_index = lowerer.body.index(direct_results[1])
    assert _call_name(lowerer.body[late_layout_index - 1]) == PRE_CONCAT
    assert _single_target(lowerer.body[late_layout_index - 1]) == (
        "_absolute_final_pre_concat_stats"
    )
    assert _single_target(lowerer.body[late_layout_index + 1]) == (
        "_late_layout_cluster_stats"
    )
