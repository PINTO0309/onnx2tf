from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import pre_terminal_affine_tail_orchestration
from onnx2tf.tflite_builder.passes.pre_terminal_affine_tail_orchestration import (
    PRE_TERMINAL_AFFINE_TAIL_PASS_IDS,
)

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
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_cleanup_orchestration.py"
)
COMPOSITE_OWNER = "run_pre_terminal_cleanup"
COMPOSITE_TARGET = "_pre_terminal_cleanup_results"
OUTER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_affine_slice_spp_orchestration.py"
)
OUTER_OWNER = "run_pre_terminal_affine_slice_spp_cleanup"
OUTER_TARGET = "_pre_terminal_affine_slice_spp_results"
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
SUCCESSOR_TARGET = "_terminal_affine_slice_spp_results"


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


def _composite_calls() -> list[ast.Call]:
    owner = _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    ]


def _outer_calls() -> list[ast.Call]:
    owner = _functions(OUTER_PATH)[OUTER_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == COMPOSITE_OWNER
    ]


def test_pre_terminal_affine_tail_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    result = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == OUTER_TARGET
    )
    index = lowerer.body.index(result)
    assert isinstance(result, ast.Assign)
    assert ast.unparse(result.value) == (
        f"{OUTER_OWNER}(shared_model_ir_pass_context)"
    )
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert _single_target(lowerer.body[index + 1]) == (
        "_terminal_qkv_activation_layout_shape_results"
    )
    assert len(_outer_calls()) == 1
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
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
        if _single_target(statement) == OUTER_TARGET
    )
    index = lowerer.body.index(result)
    assert isinstance(result, ast.Assign)
    assert ast.unparse(result.value) == (
        f"{OUTER_OWNER}(shared_model_ir_pass_context)"
    )
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert _single_target(lowerer.body[index + 1]) == (
        "_terminal_qkv_activation_layout_shape_results"
    )
    assert len(_outer_calls()) == 1
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_pre_terminal_affine_tail_owner_preserves_order_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("pre_terminal_affine_tail")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = ({"post_add": 1}, {"slice_pad_concat": 2})
    observed: list[tuple[str, object, object | None]] = []

    def _post_add(
        candidate: ModelIR,
        *,
        layout_state: object,
    ) -> dict[str, int]:
        observed.append((OWNER_CALLS[0], candidate, layout_state))
        return expected_results[0]

    def _slice_pad_concat(candidate: ModelIR) -> dict[str, int]:
        observed.append((OWNER_CALLS[1], candidate, None))
        return expected_results[1]

    monkeypatch.setattr(
        pre_terminal_affine_tail_orchestration,
        OWNER_CALLS[0],
        _post_add,
    )
    monkeypatch.setattr(
        pre_terminal_affine_tail_orchestration,
        OWNER_CALLS[1],
        _slice_pad_concat,
    )

    assert (
        pre_terminal_affine_tail_orchestration.run_pre_terminal_affine_tail_cleanup(
            context
        )
        == expected_results
    )
    assert [entry[0] for entry in observed] == list(OWNER_CALLS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert observed[0][2] is context.layout_state
    assert observed[1][2] is None
    assert PRE_TERMINAL_AFFINE_TAIL_PASS_IDS == PASS_IDS
