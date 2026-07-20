from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    TensorIR,
    validate_model_ir_layout_annotations,
)
from onnx2tf.tflite_builder.passes import pad_layout
from onnx2tf.tflite_builder.passes import high_rank_matmul
from onnx2tf.tflite_builder.passes import stale_binary_adapter_repair


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _safety_fallback_body(lowerer: ast.FunctionDef) -> list[ast.stmt]:
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test)
        == "optimize_layout_transpose_chains and len(unbound_inputs) > 0"
    )
    return guard.body


def _assert_phase_result_record(
    statement: ast.stmt,
    *,
    phase_id: str,
    owner_expression: str,
) -> None:
    assert isinstance(statement, ast.Expr)
    assert ast.unparse(statement) == (
        f"session.record_phase_result('{phase_id}', {owner_expression})"
    )


def test_fallback_norm_owner_can_prune_without_a_rewrite() -> None:
    model_ir = ModelIR("fallback_norm_zero_rewrite_prune")
    model_ir.tensors["unused"] = TensorIR(
        name="unused",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )

    stats = pad_layout._optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains(
        model_ir
    )

    assert stats == {
        "optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains": 0
    }
    assert "unused" not in model_ir.tensors


def test_safety_fallback_stages_complete_norm_mutation_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_norm_stats"
    )

    stats = body[stats_index]
    assert isinstance(stats, ast.Assign)
    assert ast.unparse(stats.value) == (
        "run_norm_subgraph_pad_layout_summary("
        "fallback_ir, diagnostics=session.diagnostics)"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_norm_stats.get("
        "'optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains', 0)) > 0"
    )


def test_safety_fallback_stages_dynamic_rank1_mutation_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    invocation_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, (ast.Assign, ast.Expr))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id
        == "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    )

    invocation = body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == "_fallback_dynamic_rank1_stats"
    assert isinstance(invocation.value, ast.Call)
    assert [ast.unparse(argument) for argument in invocation.value.args] == [
        "fallback_ir"
    ]
    assert invocation.value.keywords == []

    topology_layout_refresh = body[invocation_index + 1]
    _assert_phase_result_record(
        topology_layout_refresh,
        phase_id="topology_layout.fallback.post_dynamic_rank1",
        owner_expression="run_topology_layout_refresh(fallback_ir)",
    )


def test_safety_fallback_retains_precision_unbound_composite_results() -> None:
    body = _safety_fallback_body(_lowerer())
    sequence_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_fallback_precision_unbound_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id
        == "run_fallback_precision_unbound_cleanup"
    )
    sequence = body[sequence_index]
    assert isinstance(sequence, ast.Assign)
    assert ast.unparse(sequence.value) == (
        "run_fallback_precision_unbound_cleanup("
        "fallback_precision_unbound_context)"
    )

    previous = body[sequence_index - 1]
    _assert_phase_result_record(
        previous,
        phase_id="topology.fallback.post_placeholder",
        owner_expression="_topologically_sort_operators(fallback_ir)",
    )
    following = body[sequence_index + 1]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "fallback_conv_input_stats"


def test_safety_fallback_stages_broadcast_reconciliation_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    owner_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_broadcast_repair_stats"
    )

    guard = body[owner_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_broadcast_repair_stats.get("
        "'repaired_rank4_channelwise_broadcast_constants', 0)) > 0"
    )
    assert len(guard.body) == 2
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.fallback.broadcast",
        owner_expression=(
            "_reconcile_static_tensor_shapes(fallback_ir, "
            "include_mutation_count=True)"
        ),
    )

    topology_layout_refresh = guard.body[1]
    _assert_phase_result_record(
        topology_layout_refresh,
        phase_id="topology_layout.fallback.broadcast",
        owner_expression="run_topology_layout_refresh(fallback_ir)",
    )


def test_safety_fallback_stages_se_fc_gather_reconciliation_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    summary_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_se_fc_gather_stats"
    )

    summary = body[summary_index]
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        "_run_sinet_se_fc_gather_summary(fallback_ir, None)"
    )
    guard = body[summary_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "_stats_have_positive_count(fallback_se_fc_gather_stats)"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.fallback.se_fc_gather",
        owner_expression=(
            "_reconcile_static_tensor_shapes(fallback_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[summary_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "fallback_placeholder_matmul_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == (
        "_restore_placeholder_matmul_flattened_inputs"
    )


def test_safety_fallback_stages_placeholder_matmul_reconciliation_evidence() -> (
    None
):
    body = _safety_fallback_body(_lowerer())
    owner_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_placeholder_matmul_stats"
    )

    owner = body[owner_index]
    assert isinstance(owner, ast.Assign)
    assert isinstance(owner.value, ast.Call)
    assert isinstance(owner.value.func, ast.Name)
    assert owner.value.func.id == "_restore_placeholder_matmul_flattened_inputs"
    assert [ast.unparse(argument) for argument in owner.value.args] == [
        "fallback_ir"
    ]
    assert owner.value.keywords == []

    guard = body[owner_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_placeholder_matmul_stats.get("
        "'restored_placeholder_matmul_flattened_inputs', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.fallback.placeholder_matmul",
        owner_expression=(
            "_reconcile_static_tensor_shapes(fallback_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[owner_index + 2]
    _assert_phase_result_record(
        following,
        phase_id="topology.fallback.post_placeholder",
        owner_expression="_topologically_sort_operators(fallback_ir)",
    )


def test_safety_fallback_does_not_repeat_unbound_input_reconciliation() -> None:
    body = _safety_fallback_body(_lowerer())
    owner_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_fallback_precision_unbound_results"
    )

    owner = body[owner_index]
    assert isinstance(owner, ast.Assign)
    assert ast.unparse(owner.value) == (
        "run_fallback_precision_unbound_cleanup("
        "fallback_precision_unbound_context)"
    )

    following = body[owner_index + 1]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "fallback_conv_input_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert (
        following.value.func.id
        == "run_indexed_conv_input_adapter_repairs_summary"
    )


def test_safety_fallback_stages_complete_conv_input_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_conv_input_stats"
    )

    stats = body[stats_index]
    assert isinstance(stats, ast.Assign)
    assert isinstance(stats.value, ast.Call)
    assert isinstance(stats.value.func, ast.Name)
    assert (
        stats.value.func.id
        == "run_indexed_conv_input_adapter_repairs_summary"
    )
    assert [ast.unparse(argument) for argument in stats.value.args] == [
        "fallback_ir"
    ]
    assert stats.value.keywords == []
    predecessor = body[stats_index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert isinstance(predecessor.targets[0], ast.Name)
    assert predecessor.targets[0].id == "_fallback_precision_unbound_results"

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_conv_input_stats.get("
        "'repaired_stale_nchw_to_nhwc_conv_input_transposes', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.fallback.conv_input",
        owner_expression=(
            "_reconcile_static_tensor_shapes(fallback_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "fallback_concat_layout_stats"


def test_safety_fallback_stages_mixed_concat_reconciliation_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    owner_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_concat_layout_stats"
    )

    guard = body[owner_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_concat_layout_stats.get("
        "'repaired_mixed_nhwc_inputs_for_nchw_concat', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.fallback.mixed_concat",
        owner_expression=(
            "_reconcile_static_tensor_shapes(fallback_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[owner_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "fallback_concat_axis_stats"


def test_safety_fallback_stages_concat_axis_reconciliation_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    owner_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_concat_axis_stats"
    )

    guard = body[owner_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_concat_axis_stats.get("
        "'repaired_nchw_concat_transpose_conv_axes', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.fallback.concat_axis",
        owner_expression=(
            "_reconcile_static_tensor_shapes(fallback_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[owner_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "fallback_binary_layout_stats"


def test_fallback_binary_layout_owner_can_prune_without_a_rewrite() -> None:
    model_ir = ModelIR("fallback_binary_layout_zero_rewrite_prune")
    model_ir.tensors["unused"] = TensorIR(
        name="unused",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )

    stats = (
        stale_binary_adapter_repair
        ._repair_stale_nchw_to_nhwc_channelwise_binary_transposes(model_ir)
    )

    assert stats == {
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": 0,
    }
    assert "unused" not in model_ir.tensors


def test_safety_fallback_stages_complete_binary_layout_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_binary_layout_stats"
    )

    stats = body[stats_index]
    assert isinstance(stats, ast.Assign)
    assert ast.unparse(stats.value) == (
        "run_stale_binary_adapter_repair_summary(fallback_ir)"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_binary_layout_stats.get("
        "'repaired_stale_nchw_to_nhwc_channelwise_binary_transposes', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.fallback.binary_layout",
        owner_expression=(
            "_reconcile_static_tensor_shapes(fallback_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[stats_index + 2]
    _assert_phase_result_record(
        following,
        phase_id="topology.fallback.post_layout_repair",
        owner_expression="_topologically_sort_operators(fallback_ir)",
    )


def test_layout_annotation_validator_is_pure() -> None:
    model_ir = ModelIR("layout_validation_purity")
    model_ir.metadata["sentinel"] = {"unchanged": True}
    model_ir.tensors["value"] = TensorIR(
        name="value",
        dtype="FLOAT32",
        shape=[1, 2, 3],
        shape_signature=[1, 2, 3],
        logical_layout="NCHW",
        physical_layout="NCHW",
    )
    metadata_before = dict(model_ir.metadata)
    tensor_before = repr(model_ir.tensors["value"])

    problems = validate_model_ir_layout_annotations(model_ir)

    assert problems == [
        "tensor=value shape=[1, 2, 3] logical_layout=NCHW",
    ]
    assert model_ir.metadata == metadata_before
    assert repr(model_ir.tensors["value"]) == tensor_before


def test_safety_fallback_validates_terminal_layout_and_clears_stale_errors() -> None:
    body = _safety_fallback_body(_lowerer())
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_high_rank_bmm_stats"
    )

    metadata = body[stats_index - 1]
    assert isinstance(metadata, ast.Assign)
    assert ast.unparse(metadata.targets[0]) == (
        "fallback_ir.metadata['layout_optimize_fallback']"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_high_rank_bmm_stats.get("
        "'compressed_static_high_rank_batch_matmul', 0)) > 0"
    )

    convergence = body[stats_index + 2]
    assert isinstance(convergence, ast.Assign)
    assert isinstance(convergence.targets[0], ast.Name)
    assert convergence.targets[0].id == (
        "_fallback_binary_layout_convergence_stats"
    )
    assert ast.unparse(convergence.value) == (
        "_run_indexed_binary_layout_convergence(fallback_ir)"
    )

    validation = body[stats_index + 3]
    _assert_phase_result_record(
        validation,
        phase_id="layout_validation.fallback.terminal",
        owner_expression="run_topology_layout_validation(fallback_ir)",
    )

    terminal = body[stats_index + 4]
    assert isinstance(terminal, ast.Return)
    assert ast.unparse(terminal.value) == "_finalize_model_ir(fallback_ir)"


def test_fallback_high_rank_bmm_owner_does_not_prune_on_noop() -> None:
    model_ir = ModelIR("fallback_high_rank_bmm_noop")
    model_ir.tensors["unused"] = TensorIR(
        name="unused",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )

    stats = high_rank_matmul._compress_static_high_rank_batch_matmul(model_ir)

    assert stats == {"compressed_static_high_rank_batch_matmul": 0}
    assert "unused" in model_ir.tensors


def test_safety_fallback_stages_high_rank_bmm_reconciliation_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "fallback_high_rank_bmm_stats"
    )

    stats = body[stats_index]
    assert isinstance(stats, ast.Assign)
    assert isinstance(stats.value, ast.Call)
    assert isinstance(stats.value.func, ast.Name)
    assert stats.value.func.id == "_compress_static_high_rank_batch_matmul"
    assert [ast.unparse(argument) for argument in stats.value.args] == [
        "fallback_ir"
    ]
    assert stats.value.keywords == []

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_high_rank_bmm_stats.get("
        "'compressed_static_high_rank_batch_matmul', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_topology.fallback.high_rank_batch_matmul",
        owner_expression=(
            "run_static_shape_topology_reconciliation(fallback_ir)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_fallback_binary_layout_convergence_stats"
    )
    assert ast.unparse(following.value) == (
        "_run_indexed_binary_layout_convergence(fallback_ir)"
    )


def test_safety_fallback_retains_indexed_binary_convergence_result() -> None:
    body = _safety_fallback_body(_lowerer())
    owner_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, (ast.Assign, ast.Expr))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_run_indexed_binary_layout_convergence"
    )

    owner = body[owner_index]
    assert isinstance(owner, ast.Assign)
    assert isinstance(owner.targets[0], ast.Name)
    assert owner.targets[0].id == "_fallback_binary_layout_convergence_stats"
    assert isinstance(owner.value, ast.Call)
    assert isinstance(owner.value.func, ast.Name)
    assert owner.value.func.id == "_run_indexed_binary_layout_convergence"
    assert [ast.unparse(argument) for argument in owner.value.args] == [
        "fallback_ir"
    ]
    assert owner.value.keywords == []

    following = body[owner_index + 1]
    _assert_phase_result_record(
        following,
        phase_id="layout_validation.fallback.terminal",
        owner_expression="run_topology_layout_validation(fallback_ir)",
    )
