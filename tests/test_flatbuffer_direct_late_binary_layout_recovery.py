from __future__ import annotations

import ast
from pathlib import Path

import onnx2tf.tflite_builder.passes.late_binary_layout_recovery as recovery

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


REPO_ROOT = Path(__file__).resolve().parents[1]


def _zero_result(key: str) -> dict[str, int]:
    return {key: 0}


def test_late_binary_layout_recovery_preserves_order_and_mutation_counts(
    monkeypatch,
) -> None:
    model_ir = ModelIR("late_binary_layout_recovery")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict[str, object]] = []
    events: list[str] = []

    def result(event: str, key: str, value: int):
        def owner(target_model_ir, *args, **kwargs):
            assert target_model_ir is model_ir
            events.append(event)
            if event in {"prelu", "affine", "layout"}:
                assert kwargs.get("layout_state") is layout_state
            if event == "layout":
                assert kwargs.get("diagnostics") is diagnostics
            return {key: value}

        return owner

    monkeypatch.setattr(
        recovery,
        "optimize_prelu_transpose_passthrough_chains",
        result("prelu", "rewritten_prelu_transpose_passthrough_chains", 1),
    )
    monkeypatch.setattr(
        recovery,
        "optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains",
        result(
            "dual",
            "optimized_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains",
            2,
        ),
    )
    monkeypatch.setattr(
        recovery,
        "optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains",
        result(
            "fc",
            "optimized_terminal_transpose_mul_add_reshape_fc_nhwc_chains",
            3,
        ),
    )
    monkeypatch.setattr(
        recovery,
        "optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains",
        result(
            "bmm",
            "optimized_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains",
            4,
        ),
    )
    monkeypatch.setattr(
        recovery,
        "optimize_transpose_mul_add_const_prepost_nhwc_chains",
        result(
            "affine",
            "optimized_transpose_mul_add_const_prepost_nhwc_chains",
            5,
        ),
    )

    def layout_owner(target_model_ir, *args, **kwargs):
        assert target_model_ir is model_ir
        assert kwargs.get("layout_state") is layout_state
        assert kwargs.get("diagnostics") is diagnostics
        events.append("layout")
        return {
            "iterations": 99,
            "removed_identity_transpose": 6,
            "removed_inverse_transpose_pairs": 7,
            "removed_inverse_transpose_fanout_branches": 8,
            "composed_consecutive_transpose_pairs": 9,
        }

    monkeypatch.setattr(recovery, "run_layout_transpose_cleanup", layout_owner)

    stats = recovery.run_late_binary_layout_recovery(
        model_ir,
        include_layout_transpose=True,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )

    assert events == ["prelu", "dual", "fc", "bmm", "affine", "layout"]
    assert stats == {
        "rewritten_prelu_transpose_passthrough_chains": 1,
        "optimized_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains": 2,
        "optimized_terminal_transpose_mul_add_reshape_fc_nhwc_chains": 3,
        "optimized_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains": 4,
        "optimized_transpose_mul_add_const_prepost_nhwc_chains": 5,
        "removed_identity_transpose": 6,
        "removed_inverse_transpose_pairs": 7,
        "removed_inverse_transpose_fanout_branches": 8,
        "composed_consecutive_transpose_pairs": 9,
        "pruned_unused_tensors": 0,
    }
    assert "iterations" not in stats


def test_late_binary_layout_recovery_skips_optional_owners_and_reports_prune(
    monkeypatch,
) -> None:
    model_ir = ModelIR("late_binary_layout_recovery_fallback")
    model_ir.tensors["unused"] = TensorIR(
        name="unused",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    events: list[str] = []

    def prelu_owner(target_model_ir, *args, **kwargs):
        events.append("prelu")
        del target_model_ir.tensors["unused"]
        return _zero_result("rewritten_prelu_transpose_passthrough_chains")

    def zero_owner(event: str, key: str):
        def owner(*args, **kwargs):
            events.append(event)
            return _zero_result(key)

        return owner

    monkeypatch.setattr(
        recovery,
        "optimize_prelu_transpose_passthrough_chains",
        prelu_owner,
    )
    monkeypatch.setattr(
        recovery,
        "optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains",
        zero_owner(
            "dual",
            "optimized_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains",
        ),
    )
    monkeypatch.setattr(
        recovery,
        "optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains",
        zero_owner(
            "fc",
            "optimized_terminal_transpose_mul_add_reshape_fc_nhwc_chains",
        ),
    )
    monkeypatch.setattr(
        recovery,
        "optimize_transpose_mul_add_const_prepost_nhwc_chains",
        zero_owner(
            "affine",
            "optimized_transpose_mul_add_const_prepost_nhwc_chains",
        ),
    )

    stats = recovery.run_late_binary_layout_recovery(
        model_ir,
        include_layout_transpose=False,
    )

    assert events == ["prelu", "dual", "fc", "affine"]
    assert stats[
        "optimized_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains"
    ] == 0
    assert all(stats[key] == 0 for key in recovery._LAYOUT_MUTATION_KEYS)
    assert stats["pruned_unused_tensors"] == 1


def test_late_binary_layout_recovery_empty_model_is_stable() -> None:
    model_ir = ModelIR("empty_late_binary_layout_recovery")
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = recovery.run_late_binary_layout_recovery(
        model_ir,
        include_layout_transpose=True,
        layout_state=layout_state,
        diagnostics=[],
    )

    assert all(value == 0 for value in stats.values())
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_late_binary_recovery_connects_to_terminal_evidence_boundary() -> None:
    lowerer_tree = ast.parse(
        (
            REPO_ROOT
            / "onnx2tf"
            / "tflite_builder"
            / "lower_from_onnx2tf.py"
        ).read_text(encoding="utf-8")
    )
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    decision_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_late_binary_layout_recovery_requires_reconciliation"
    )
    decision = lowerer.body[decision_index]
    assert isinstance(decision, ast.Assign)
    assert ast.unparse(decision.value) == (
        "run_optional_late_binary_layout_recovery_cleanup("
        "shared_model_ir_pass_context, "
        "enabled=optimize_layout_transpose_chains or "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )
    reconcile_guard = lowerer.body[decision_index + 1]
    assert isinstance(reconcile_guard, ast.If)
    assert ast.unparse(reconcile_guard.test) == (
        "_late_binary_layout_recovery_requires_reconciliation"
    )

    following = lowerer.body[decision_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_pre_terminal_instancenorm_layout_results"
    )
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == (
        "run_pre_terminal_instancenorm_layout_cleanup"
    )
    assert [ast.unparse(argument) for argument in following.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert following.value.keywords == []


def test_late_binary_recovery_retains_complete_shape_result() -> None:
    lowerer_tree = ast.parse(
        (
            REPO_ROOT
            / "onnx2tf"
            / "tflite_builder"
            / "lower_from_onnx2tf.py"
        ).read_text(encoding="utf-8")
    )
    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    decision = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_late_binary_layout_recovery_requires_reconciliation"
    )
    decision_index = lowerer.body.index(decision)
    guard = lowerer.body[decision_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "_late_binary_layout_recovery_requires_reconciliation"
    )
    assert len(guard.body) == 1
    statement = guard.body[0]
    assert isinstance(statement, ast.Expr)
    assert ast.unparse(statement) == (
        "session.record_phase_result("
        "'shape_reconciliation.primary.late_binary_layout_recovery', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )
