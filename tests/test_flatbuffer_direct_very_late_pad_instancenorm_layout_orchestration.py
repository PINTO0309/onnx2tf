from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    very_late_pad_instancenorm_layout_orchestration,
)
from onnx2tf.tflite_builder.passes.very_late_pad_instancenorm_layout_orchestration import (
    VERY_LATE_PAD_INSTANCENORM_LAYOUT_PASS_IDS,
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
    / "very_late_pad_instancenorm_layout_orchestration.py"
)
OWNER = "run_very_late_pad_instancenorm_layout_cleanup"
RESULT_TARGET = "_very_late_pad_instancenorm_layout_results"
PREDECESSOR_TARGET = "_late_conv1d_decoder_layout_results"
SUCCESSOR_TARGET = "_very_late_singleton_consecutive_reshape_results"
OLD_RESULT_TARGETS = (
    "_very_late_pad_layout_stats",
    "_very_late_instancenorm_post_bias_stats",
    "_very_late_instancenorm_residual_mul_concat_stats",
    "_very_late_instancenorm_dualstats_stats",
)
PASS_IDS = (
    "run_pad_layout_cleanup",
    "optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
)
LOWERER_PASS_IDS = (
    "run_pad_layout_cleanup",
    "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def test_very_late_pad_instancenorm_cluster_uses_composite_outside_store() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_very_late_pad_instancenorm_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_phase_result"
        and any(
            isinstance(child, ast.Name) and child.id == OWNER
            for child in ast.walk(node)
        )
        for node in ast.walk(lowerer)
    )


def test_very_late_pad_instancenorm_cluster_uses_one_composite_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PASS_IDS
    ]
    assert owner_calls == list(PASS_IDS)

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_very_late_pad_instancenorm_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_very_late_pad_instancenorm_owner_preserves_argument_and_result_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("very_late_pad_instancenorm_layout_owner")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[str, object, object, object | None]] = []
    expected_results = tuple(
        {f"result_{index}": index}
        for index in range(1, len(PASS_IDS) + 1)
    )

    def _pad_callback(
        candidate: ModelIR,
        *,
        layout_state: object,
        diagnostics: object,
    ) -> dict[str, int]:
        observed.append((PASS_IDS[0], candidate, layout_state, diagnostics))
        return dict(expected_results[0])

    def _layout_callback(name: str, result: dict[str, int]):
        def _run(
            candidate: ModelIR,
            *,
            layout_state: object,
        ) -> dict[str, int]:
            observed.append((name, candidate, layout_state, None))
            return dict(result)

        return _run

    monkeypatch.setattr(
        very_late_pad_instancenorm_layout_orchestration,
        PASS_IDS[0],
        _pad_callback,
    )
    for pass_id, result in zip(PASS_IDS[1:], expected_results[1:], strict=True):
        monkeypatch.setattr(
            very_late_pad_instancenorm_layout_orchestration,
            pass_id,
            _layout_callback(pass_id, result),
        )

    assert (
        very_late_pad_instancenorm_layout_orchestration.run_very_late_pad_instancenorm_layout_cleanup(
            context
        )
        == expected_results
    )
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert all(entry[2] is context.layout_state for entry in observed)
    assert observed[0][3] is context.diagnostics
    assert all(entry[3] is None for entry in observed[1:])
    assert VERY_LATE_PAD_INSTANCENORM_LAYOUT_PASS_IDS == LOWERER_PASS_IDS
