from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_affine_tail_orchestration.py"
)
OWNER = "run_pre_terminal_affine_tail_cleanup"
RESULT_TARGET = "_pre_terminal_affine_tail_results"
OLD_RESULT_TARGETS = (
    "_pre_terminal_affine_post_add_stats",
    "_pre_terminal_affine_slice_pad_concat_stats",
)
PASS_IDS = (
    "_optimize_transpose_mul_posttranspose_add_nhwc_chains",
    "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
)
OWNER_CALLS = (
    "optimize_transpose_mul_posttranspose_add_nhwc_chains",
    "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
)
PREDECESSOR_TARGET = "_pre_terminal_channel_slice_pad_mul_stats"
SUCCESSOR_TARGET = "_terminal_affine_stats"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def test_pre_terminal_affine_tail_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    first = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == OLD_RESULT_TARGETS[0]
    )
    index = lowerer.body.index(first)
    second = lowerer.body[index + 1]
    assert ast.unparse(first.value) == (
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains("
        "model_ir, layout_state=session.layout_state)"
    )
    assert _single_target(second) == OLD_RESULT_TARGETS[1]
    assert isinstance(second, ast.Assign)
    assert ast.unparse(second.value) == f"{PASS_IDS[1]}(model_ir)"
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


@pytest.mark.xfail(
    strict=True,
    reason="pre-terminal affine tail lacks one ordered owner",
)
def test_pre_terminal_affine_tail_uses_one_ordered_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in sorted(
            (
                node
                for node in ast.walk(owner)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in OWNER_CALLS
            ),
            key=lambda node: node.lineno,
        )
    ]
    assert tuple(owner_calls) == OWNER_CALLS

    lowerer = _lowerer()
    result = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(result)
    assert isinstance(result, ast.Assign)
    assert ast.unparse(result.value) == (
        f"{OWNER}(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
