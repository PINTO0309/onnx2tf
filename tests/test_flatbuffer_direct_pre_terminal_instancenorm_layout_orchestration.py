from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    pre_terminal_instancenorm_layout_orchestration,
)
from onnx2tf.tflite_builder.passes.pre_terminal_instancenorm_layout_orchestration import (
    PRE_TERMINAL_INSTANCENORM_LAYOUT_PASS_IDS,
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
    / "pre_terminal_instancenorm_layout_orchestration.py"
)
OWNER = "run_pre_terminal_instancenorm_layout_cleanup"
RESULT_TARGET = "_pre_terminal_instancenorm_layout_results"
PREDECESSOR_GUARD = (
    "_late_binary_layout_recovery_requires_reconciliation"
)
SUCCESSOR_TARGET = "_pre_terminal_affine_stats"
OLD_RESULT_TARGETS = (
    "_pre_terminal_affine_instancenorm_post_bias_stats",
    "_pre_terminal_affine_instancenorm_residual_mul_concat_stats",
    "_pre_terminal_affine_instancenorm_dualstats_stats",
)
PASS_IDS = (
    "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
)


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


def test_pre_terminal_instancenorm_layout_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert isinstance(assignment, ast.Assign)
    assert ast.unparse(assignment.value) == (
        "run_pre_terminal_instancenorm_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_pre_terminal_instancenorm_layout_uses_one_composite_owner() -> None:
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
                and f"_{node.func.id}" in PASS_IDS
            ),
            key=lambda node: node.lineno,
        )
    ]
    assert tuple(f"_{name}" for name in owner_calls) == PASS_IDS

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_pre_terminal_instancenorm_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_pre_terminal_instancenorm_layout_owner_preserves_results_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("pre_terminal_instancenorm_layout")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = tuple(
        {f"result_{index}": index}
        for index in range(len(PASS_IDS))
    )
    observed: list[tuple[str, object, object]] = []

    def _callback(pass_id: str, result: dict[str, int]):
        def _owner(
            candidate: ModelIR,
            *,
            layout_state: object,
        ) -> dict[str, int]:
            observed.append((pass_id, candidate, layout_state))
            return result

        return _owner

    for pass_id, result in zip(PASS_IDS, expected_results):
        monkeypatch.setattr(
            pre_terminal_instancenorm_layout_orchestration,
            pass_id.removeprefix("_"),
            _callback(pass_id, result),
        )

    assert (
        pre_terminal_instancenorm_layout_orchestration.run_pre_terminal_instancenorm_layout_cleanup(
            context
        )
        == expected_results
    )
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert all(entry[2] is context.layout_state for entry in observed)
    assert PRE_TERMINAL_INSTANCENORM_LAYOUT_PASS_IDS == PASS_IDS
