from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import conv_input_adapter_repair


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "conv_input_adapter_repair.py"
)
RAW_OWNER = "_run_indexed_conv_input_adapter_repairs"
SUMMARY_OWNER = "run_indexed_conv_input_adapter_repairs_summary"
COMPOSITE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_dynamic_adapter_orchestration.py"
)
COMPOSITE_OWNER = "run_very_late_dynamic_adapter_cleanup"
SITE_CONTRACTS = (
    (
        "fallback_conv_input_tensor_count",
        "fallback_conv_input_stats",
        "fallback_ir",
        "_fallback_unbound_repair_stats",
        None,
    ),
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _composite_summary_calls() -> list[ast.Call]:
    owner = _functions(COMPOSITE_OWNER_PATH)[COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SUMMARY_OWNER
    ]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _containing_body(root: ast.AST, target: ast.stmt) -> list[ast.stmt]:
    for node in ast.walk(root):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and target in value:
                return value
    raise AssertionError("statement is not contained by an AST body")


def test_indexed_conv_input_prune_aware_boundaries_are_fixed() -> None:
    lowerer = _lowerer()
    composite_calls = _composite_summary_calls()
    assert len(composite_calls) == 1
    assert ast.unparse(composite_calls[0]) == (
        f"{SUMMARY_OWNER}(context.model_ir)"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "very_late_conv_input_tensor_count"
        for node in ast.walk(lowerer)
    )
    for count_target, stats_target, model_name, predecessor, successor in (
        SITE_CONTRACTS
    ):
        stats = next(
            statement
            for statement in ast.walk(lowerer)
            if _single_target(statement) == stats_target
        )
        body = _containing_body(lowerer, stats)
        index = body.index(stats)
        assert isinstance(stats, ast.Assign)
        assert ast.unparse(stats.value) == f"{SUMMARY_OWNER}({model_name})"
        assert _single_target(body[index - 1]) == predecessor
        assert not any(
            isinstance(node, ast.Name) and node.id == count_target
            for node in ast.walk(lowerer)
        )
        if successor is None:
            following = body[index + 1]
            assert isinstance(following, ast.If)
            assert any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id == stats_target
                for node in ast.walk(following.test)
            )
        else:
            assert _single_target(body[index + 1]) == successor


def test_indexed_conv_input_uses_one_shared_prune_aware_summary_owner() -> None:
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
    composite_calls = _composite_summary_calls()
    assert len(composite_calls) == 1
    assert ast.unparse(composite_calls[0]) == (
        f"{SUMMARY_OWNER}(context.model_ir)"
    )
    for count_target, stats_target, model_name, predecessor, successor in (
        SITE_CONTRACTS
    ):
        stats = next(
            statement
            for statement in ast.walk(lowerer)
            if _single_target(statement) == stats_target
        )
        body = _containing_body(lowerer, stats)
        index = body.index(stats)
        assert isinstance(stats, ast.Assign)
        assert ast.unparse(stats.value) == f"{SUMMARY_OWNER}({model_name})"
        assert _single_target(body[index - 1]) == predecessor
        assert not any(
            isinstance(node, ast.Name) and node.id == count_target
            for node in ast.walk(lowerer)
        )
        if successor is None:
            assert isinstance(body[index + 1], ast.If)
        else:
            assert _single_target(body[index + 1]) == successor

    wrapper = _functions(LOWERER_PATH)[RAW_OWNER]
    assert len(wrapper.body) == 1
    assert ast.unparse(wrapper.body[0]) == (
        f"return {RAW_OWNER}_pass(model_ir)"
    )


@pytest.mark.parametrize("prune", (False, True))
def test_indexed_conv_input_summary_preserves_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    prune: bool,
) -> None:
    model_ir = ModelIR("indexed_conv_input_summary")
    model_ir.tensors["probe"] = TensorIR(
        name="probe",
        dtype="float32",
        shape=[1],
    )
    raw_result = {
        "repaired_singleton_nhwc_conv_input_reshapes": 3,
        "repaired_stale_nchw_to_nhwc_conv_input_transposes": 4,
    }
    observed: list[ModelIR] = []

    def _run(candidate: ModelIR) -> dict[str, int]:
        observed.append(candidate)
        if prune:
            del candidate.tensors["probe"]
        return raw_result

    monkeypatch.setattr(conv_input_adapter_repair, RAW_OWNER, _run)

    assert conv_input_adapter_repair.run_indexed_conv_input_adapter_repairs_summary(
        model_ir
    ) == {
        **raw_result,
        "pruned_unused_tensors": int(prune),
    }
    assert observed == [model_ir]
