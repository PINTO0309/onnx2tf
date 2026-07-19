from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import (
    se_fc_gather_channel_fanout_orchestration,
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
    / "se_fc_gather_channel_fanout_orchestration.py"
)
RAW_PAIR_HELPER = "_run_se_fc_gather_channel_fanout_pass_cluster"
RAW_SINET_OWNER = (
    "optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains"
)
RAW_PAIR_OWNER = "run_se_fc_gather_channel_fanout"
SUMMARY_OWNER = "run_sinet_se_fc_gather_summary"
SUMMARY_HELPER = "_run_sinet_se_fc_gather_summary"
SITE_CONTRACTS = (
    (
        "fallback_se_fc_gather_tensor_count",
        "fallback_sinet_shuffle_stats",
        ("fallback_se_fc_stats", "fallback_gather_stats"),
        "fallback_se_fc_gather_stats",
        "fallback_ir",
        "None",
        "fallback_broadcast_repair_stats",
        "fallback_placeholder_matmul_stats",
    ),
    (
        "final_se_fc_gather_tensor_count",
        "final_sinet_shuffle_stats",
        ("final_se_fc_stats", "final_gather_stats"),
        "final_se_fc_gather_stats",
        "model_ir",
        "session.layout_state",
        "final_placeholder_matmul_stats",
        "final_prelu_stats",
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


def _assert_predecessor(
    statement: ast.stmt,
    target: str | None,
) -> None:
    assert target is not None
    assert isinstance(statement, ast.If)
    assert any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == target
        for node in ast.walk(statement.test)
    )


def test_sinet_se_fc_gather_rewrite_or_prune_boundaries_are_fixed() -> None:
    lowerer = _lowerer()
    old_targets = {
        target
        for count, sinet, pair, _, _, _, _, _ in SITE_CONTRACTS
        for target in (count, sinet, *pair)
    }
    for (
        _,
        _,
        _,
        summary_target,
        model_name,
        layout_expression,
        predecessor,
        successor,
    ) in SITE_CONTRACTS:
        summary = next(
            statement
            for statement in ast.walk(lowerer)
            if _single_target(statement) == summary_target
        )
        body = _containing_body(lowerer, summary)
        index = body.index(summary)
        assert isinstance(summary, ast.Assign)
        assert ast.unparse(summary.value) == (
            f"{SUMMARY_HELPER}({model_name}, {layout_expression})"
        )
        _assert_predecessor(body[index - 1], predecessor)
        guard = body[index + 1]
        assert isinstance(guard, ast.If)
        assert ast.unparse(guard.test) == (
            f"_stats_have_positive_count({summary_target})"
        )
        assert _single_target(body[index + 2]) == successor
    assert not any(
        isinstance(node, ast.Name) and node.id in old_targets
        for node in ast.walk(lowerer)
    )


def test_sinet_se_fc_gather_uses_one_shared_prune_aware_summary_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {RAW_SINET_OWNER, RAW_PAIR_OWNER}
    ]
    assert owner_calls.count(RAW_SINET_OWNER) == 1
    assert owner_calls.count(RAW_PAIR_OWNER) == 1
    initial_count = next(
        statement
        for statement in owner.body
        if _single_target(statement) == "initial_tensor_count"
    )
    assert isinstance(initial_count, ast.Assign)
    assert ast.unparse(initial_count.value) == "len(context.model_ir.tensors)"

    lowerer = _lowerer()
    helper = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.FunctionDef) and node.name == SUMMARY_HELPER
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SUMMARY_OWNER
        for node in ast.walk(helper)
    )
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == RAW_PAIR_HELPER
        for node in ast.walk(lowerer)
    )

    old_targets = {
        target
        for count, sinet, pair, _, _, _, _, _ in SITE_CONTRACTS
        for target in (count, sinet, *pair)
    }
    for (
        _,
        _,
        _,
        summary_target,
        model_name,
        layout_expression,
        predecessor,
        successor,
    ) in SITE_CONTRACTS:
        summary = next(
            statement
            for statement in ast.walk(lowerer)
            if _single_target(statement) == summary_target
        )
        body = _containing_body(lowerer, summary)
        index = body.index(summary)
        assert isinstance(summary, ast.Assign)
        assert ast.unparse(summary.value) == (
            f"{SUMMARY_HELPER}({model_name}, {layout_expression})"
        )
        _assert_predecessor(body[index - 1], predecessor)
        guard = body[index + 1]
        assert isinstance(guard, ast.If)
        assert ast.unparse(guard.test) == (
            f"_stats_have_positive_count({summary_target})"
        )
        assert _single_target(body[index + 2]) == successor

    assert not any(
        isinstance(node, ast.Name) and node.id in old_targets
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("prune", (False, True))
def test_sinet_se_fc_gather_summary_preserves_order_context_schema_and_pruning(
    monkeypatch: pytest.MonkeyPatch,
    prune: bool,
) -> None:
    model_ir = ModelIR("sinet_se_fc_gather_summary")
    model_ir.tensors["probe"] = TensorIR(
        name="probe",
        dtype="float32",
        shape=[1],
    )
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict[str, object]] = []
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    events: list[tuple[str, object, object]] = []

    def _run_sinet(
        candidate: ModelIR,
        *,
        layout_state: LayoutState | None = None,
    ) -> dict[str, int]:
        events.append(("sinet", candidate, layout_state))
        if prune:
            del candidate.tensors["probe"]
        return {
            "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 3
        }

    def _run_pair(
        candidate_context: ModelIRPassContext,
    ) -> tuple[dict[str, int], dict[str, int]]:
        events.append(("pair", candidate_context, candidate_context.diagnostics))
        return (
            {"optimized_transpose_se_fc_mul_prepost_nhwc_chains": 4},
            {
                "optimized_transpose_gather_transpose_nhwc_channel_chains": 5
            },
        )

    monkeypatch.setattr(
        se_fc_gather_channel_fanout_orchestration,
        RAW_SINET_OWNER,
        _run_sinet,
    )
    monkeypatch.setattr(
        se_fc_gather_channel_fanout_orchestration,
        RAW_PAIR_OWNER,
        _run_pair,
    )

    assert se_fc_gather_channel_fanout_orchestration.run_sinet_se_fc_gather_summary(
        context
    ) == {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": 3,
        "optimized_transpose_se_fc_mul_prepost_nhwc_chains": 4,
        "optimized_transpose_gather_transpose_nhwc_channel_chains": 5,
        "pruned_unused_tensors": int(prune),
    }
    assert events == [
        ("sinet", model_ir, layout_state),
        ("pair", context, diagnostics),
    ]
