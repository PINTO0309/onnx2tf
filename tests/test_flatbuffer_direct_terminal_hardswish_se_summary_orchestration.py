from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import hardswish_se_layout


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "hardswish_se_layout.py"
)
RAW_WRAPPER = (
    "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_"
    "nhwc_chains"
)
RAW_OWNER = (
    "optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_"
    "nhwc_chains"
)
SUMMARY_OWNER = "run_hardswish_se_layout_summary"
COUNT_TARGET = "terminal_hardswish_se_tensor_count"
SUMMARY_TARGET = "_terminal_hardswish_se_stats"
PREDECESSOR_TARGET = "_terminal_split_conv_concat_bridge_stats"
SUCCESSOR_TARGET = "_late_hard_activation_stats"


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


def test_terminal_hardswish_se_prune_aware_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == SUMMARY_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == f"{SUMMARY_OWNER}(model_ir)"
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == SUMMARY_TARGET
        for node in ast.walk(lowerer)
    )


def test_terminal_hardswish_se_uses_one_prune_aware_summary_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    raw_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RAW_OWNER
    ]
    assert len(raw_calls) == 1
    initial_count = next(
        statement
        for statement in owner.body
        if _single_target(statement) == "initial_tensor_count"
    )
    assert isinstance(initial_count, ast.Assign)
    assert ast.unparse(initial_count.value) == "len(model_ir.tensors)"

    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == SUMMARY_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == f"{SUMMARY_OWNER}(model_ir)"
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )

    wrapper = _functions(LOWERER_PATH)[RAW_WRAPPER]
    assert ast.unparse(wrapper.body[0]) == f"return {RAW_WRAPPER}_pass(model_ir)"


@pytest.mark.parametrize("prune", (False, True))
def test_hardswish_se_summary_owner_preserves_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    prune: bool,
) -> None:
    model_ir = ModelIR("terminal_hardswish_se_summary")
    model_ir.tensors["probe"] = TensorIR(
        name="probe",
        dtype="float32",
        shape=[1],
    )
    raw_result = {
        "optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains": 7,
    }
    observed: list[ModelIR] = []

    def _run(candidate: ModelIR) -> dict[str, int]:
        observed.append(candidate)
        if prune:
            del candidate.tensors["probe"]
        return raw_result

    monkeypatch.setattr(
        hardswish_se_layout,
        RAW_OWNER,
        _run,
    )

    assert hardswish_se_layout.run_hardswish_se_layout_summary(model_ir) == {
        **raw_result,
        "pruned_unused_tensors": int(prune),
    }
    assert observed == [model_ir]
