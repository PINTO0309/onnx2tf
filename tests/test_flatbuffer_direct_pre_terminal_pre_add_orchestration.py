from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import pre_terminal_pre_add_orchestration
from onnx2tf.tflite_builder.passes.pre_terminal_pre_add_orchestration import (
    PRE_TERMINAL_PRE_ADD_PASS_IDS,
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
    / "pre_terminal_pre_add_orchestration.py"
)
OWNER = "run_pre_terminal_pre_add_cleanup"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_cleanup_orchestration.py"
)
COMPOSITE_OWNER = "run_pre_terminal_cleanup"
COMPOSITE_TARGET = "_pre_terminal_cleanup_results"
RESULT_TARGET = "_pre_terminal_pre_add_stats"
COUNT_TARGET = "pre_terminal_pre_add_tensor_count"
PASS_ID = "_optimize_transpose_pre_add_nhwc_chains"
PREDECESSOR_TARGET = "_pre_terminal_affine_stats"
SUCCESSOR_TARGET = "_pre_terminal_channel_slice_pad_mul_stats"


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


def test_pre_terminal_pre_add_prune_evidence_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    result = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(result)
    assert isinstance(result, ast.Assign)
    assert ast.unparse(result.value) == (
        "run_pre_terminal_cleanup(shared_model_ir_pass_context)"
    )
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert _single_target(lowerer.body[index + 1]) == "_terminal_affine_stats"
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )


def test_pre_terminal_pre_add_uses_one_prune_aware_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    pass_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PASS_ID.removeprefix("_")
    ]
    assert len(pass_calls) == 1
    assert ast.unparse(pass_calls[0].args[0]) == "context.model_ir"

    lowerer = _lowerer()
    result = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(result)
    assert isinstance(result, ast.Assign)
    assert ast.unparse(result.value) == (
        "run_pre_terminal_cleanup(shared_model_ir_pass_context)"
    )
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert _single_target(lowerer.body[index + 1]) == "_terminal_affine_stats"
    assert len(_composite_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id == COUNT_TARGET
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("prune", [False, True])
def test_pre_terminal_pre_add_owner_preserves_result_and_prune_evidence(
    monkeypatch: pytest.MonkeyPatch,
    prune: bool,
) -> None:
    model_ir = ModelIR("pre_terminal_pre_add")
    model_ir.tensors["probe"] = TensorIR(
        name="probe",
        dtype="float32",
        shape=[1],
    )
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[object, object]] = []

    def _owner(
        candidate: ModelIR,
        *,
        layout_state: object,
    ) -> dict[str, int]:
        observed.append((candidate, layout_state))
        if prune:
            del candidate.tensors["probe"]
        return {"rewritten": 1}

    monkeypatch.setattr(
        pre_terminal_pre_add_orchestration,
        PASS_ID.removeprefix("_"),
        _owner,
    )

    assert pre_terminal_pre_add_orchestration.run_pre_terminal_pre_add_cleanup(
        context
    ) == {
        "rewritten": 1,
        "pruned_unused_tensors": int(prune),
    }
    assert observed == [(context.model_ir, context.layout_state)]
    assert PRE_TERMINAL_PRE_ADD_PASS_IDS == (PASS_ID,)
