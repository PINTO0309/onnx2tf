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

    tensor_count = body[stats_index - 1]
    assert isinstance(tensor_count, ast.Assign)
    assert len(tensor_count.targets) == 1
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "fallback_norm_tensor_count"
    assert ast.unparse(tensor_count.value) == "len(fallback_ir.tensors)"

    stats = body[stats_index]
    assert isinstance(stats, ast.Assign)
    assert isinstance(stats.value, ast.Dict)
    assert stats.value.keys[0] is None
    owner = stats.value.values[0]
    assert isinstance(owner, ast.Call)
    assert isinstance(owner.func, ast.Name)
    assert owner.func.id == "run_pad_layout_cleanup"
    assert [ast.unparse(argument) for argument in owner.args] == ["fallback_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in owner.keywords
    } == {
        "include_pad": "False",
        "include_unary": "False",
        "include_norm": "True",
        "diagnostics": "session.diagnostics",
    }
    prune_key = stats.value.keys[1]
    assert isinstance(prune_key, ast.Constant)
    assert prune_key.value == "pruned_unused_tensors"
    assert ast.unparse(stats.value.values[1]) == (
        "max(0, fallback_norm_tensor_count - len(fallback_ir.tensors))"
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
    assert isinstance(topology_layout_refresh, ast.Assign)
    assert len(topology_layout_refresh.targets) == 1
    assert isinstance(topology_layout_refresh.targets[0], ast.Name)
    assert topology_layout_refresh.targets[0].id == (
        "_fallback_dynamic_rank1_topology_layout_stats"
    )
    assert ast.unparse(topology_layout_refresh.value) == (
        "run_topology_layout_refresh(fallback_ir)"
    )


def test_safety_fallback_retains_precision_cleanup_results() -> None:
    body = _safety_fallback_body(_lowerer())
    rewrite_name = "_rewrite_constant_divisors_to_multiplicative_reciprocals"
    consecutive_name = "run_consecutive_mul_constants_cleanup"
    restore_name = "_restore_precision_sensitive_reciprocal_divisions"
    rewrite_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, (ast.Assign, ast.Expr))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == rewrite_name
    )
    assert rewrite_index + 2 < len(body)

    expected = (
        (
            "_fallback_precision_div_rewrite_stats",
            rewrite_name,
            {},
        ),
        (
            "_fallback_precision_consecutive_mul_stats",
            consecutive_name,
            {"diagnostics": "session.diagnostics"},
        ),
        (
            "_fallback_precision_div_restore_stats",
            restore_name,
            {},
        ),
    )
    for statement, (target_name, function_name, keywords) in zip(
        body[rewrite_index : rewrite_index + 3],
        expected,
    ):
        assert isinstance(statement, ast.Assign)
        assert len(statement.targets) == 1
        assert isinstance(statement.targets[0], ast.Name)
        assert statement.targets[0].id == target_name
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id == function_name
        assert [ast.unparse(argument) for argument in call.args] == [
            "fallback_ir"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == keywords

    previous = body[rewrite_index - 1]
    assert ast.unparse(previous) == "_topologically_sort_operators(fallback_ir)"
    following = body[rewrite_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_fallback_unbound_repair_stats"


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

    default_stats = body[owner_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert len(default_stats.targets) == 1
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_fallback_broadcast_static_shape_stats"
    )
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = body[owner_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_broadcast_repair_stats.get("
        "'repaired_rank4_channelwise_broadcast_constants', 0)) > 0"
    )
    assert len(guard.body) == 2
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert len(reconciliation.targets) == 1
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_fallback_broadcast_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "fallback_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    topology_layout_refresh = guard.body[1]
    assert isinstance(topology_layout_refresh, ast.Assign)
    assert len(topology_layout_refresh.targets) == 1
    assert isinstance(topology_layout_refresh.targets[0], ast.Name)
    assert topology_layout_refresh.targets[0].id == (
        "_fallback_broadcast_topology_layout_stats"
    )
    assert ast.unparse(topology_layout_refresh.value) == (
        "run_topology_layout_refresh(fallback_ir)"
    )


def test_safety_fallback_stages_se_fc_gather_reconciliation_evidence() -> None:
    body = _safety_fallback_body(_lowerer())
    cluster_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Tuple)
        and {
            element.id
            for element in statement.targets[0].elts
            if isinstance(element, ast.Name)
        }
        == {"fallback_se_fc_stats", "fallback_gather_stats"}
    )

    default_stats = body[cluster_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert len(default_stats.targets) == 1
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_fallback_se_fc_gather_static_shape_stats"
    )
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = body[cluster_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_sinet_shuffle_stats.get("
        "'optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains', 0)) "
        "+ int(fallback_se_fc_stats.get("
        "'optimized_transpose_se_fc_mul_prepost_nhwc_chains', 0)) + "
        "int(fallback_gather_stats.get("
        "'optimized_transpose_gather_transpose_nhwc_channel_chains', 0)) > 0 "
        "or len(fallback_ir.tensors) < fallback_se_fc_gather_tensor_count"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert len(reconciliation.targets) == 1
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_fallback_se_fc_gather_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "fallback_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    following = body[cluster_index + 3]
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

    default_stats = body[owner_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert len(default_stats.targets) == 1
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_fallback_placeholder_matmul_static_shape_stats"
    )
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = body[owner_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_placeholder_matmul_stats.get("
        "'restored_placeholder_matmul_flattened_inputs', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert len(reconciliation.targets) == 1
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_fallback_placeholder_matmul_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "fallback_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    following = body[owner_index + 3]
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following.value) == (
        "_topologically_sort_operators(fallback_ir)"
    )


def test_safety_fallback_does_not_repeat_unbound_input_reconciliation() -> None:
    body = _safety_fallback_body(_lowerer())
    owner_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_fallback_unbound_repair_stats"
    )

    owner = body[owner_index]
    assert isinstance(owner, ast.Assign)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
        for node in ast.walk(owner.value)
    )

    following = body[owner_index + 1]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "fallback_conv_input_tensor_count"
    stats = body[owner_index + 2]
    assert isinstance(stats, ast.Assign)
    assert isinstance(stats.targets[0], ast.Name)
    assert stats.targets[0].id == "fallback_conv_input_stats"


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

    tensor_count = body[stats_index - 1]
    assert isinstance(tensor_count, ast.Assign)
    assert len(tensor_count.targets) == 1
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "fallback_conv_input_tensor_count"
    assert ast.unparse(tensor_count.value) == "len(fallback_ir.tensors)"

    stats = body[stats_index]
    assert isinstance(stats, ast.Assign)
    assert isinstance(stats.value, ast.Dict)
    assert stats.value.keys[0] is None
    owner = stats.value.values[0]
    assert isinstance(owner, ast.Call)
    assert isinstance(owner.func, ast.Name)
    assert owner.func.id == "_run_indexed_conv_input_adapter_repairs"
    assert [ast.unparse(argument) for argument in owner.args] == ["fallback_ir"]
    assert owner.keywords == []
    prune_key = stats.value.keys[1]
    assert isinstance(prune_key, ast.Constant)
    assert prune_key.value == "pruned_unused_tensors"
    assert ast.unparse(stats.value.values[1]) == (
        "max(0, fallback_conv_input_tensor_count - len(fallback_ir.tensors))"
    )

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert len(default_stats.targets) == 1
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_fallback_conv_input_static_shape_stats"
    )
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = body[stats_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_conv_input_stats.get("
        "'repaired_stale_nchw_to_nhwc_conv_input_transposes', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert len(reconciliation.targets) == 1
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_fallback_conv_input_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "fallback_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    following = body[stats_index + 3]
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

    default_stats = body[owner_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert len(default_stats.targets) == 1
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_fallback_mixed_concat_static_shape_stats"
    )
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = body[owner_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_concat_layout_stats.get("
        "'repaired_mixed_nhwc_inputs_for_nchw_concat', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert len(reconciliation.targets) == 1
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_fallback_mixed_concat_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "fallback_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    following = body[owner_index + 3]
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

    default_stats = body[owner_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_fallback_concat_axis_static_shape_stats"
    )
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = body[owner_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_concat_axis_stats.get("
        "'repaired_nchw_concat_transpose_conv_axes', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_fallback_concat_axis_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "fallback_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    following = body[owner_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "fallback_binary_layout_tensor_count"


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

    tensor_count = body[stats_index - 1]
    assert isinstance(tensor_count, ast.Assign)
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "fallback_binary_layout_tensor_count"
    assert ast.unparse(tensor_count.value) == "len(fallback_ir.tensors)"

    stats = body[stats_index]
    assert isinstance(stats, ast.Assign)
    assert isinstance(stats.value, ast.Dict)
    assert stats.value.keys[0] is None
    owner = stats.value.values[0]
    assert isinstance(owner, ast.Call)
    assert isinstance(owner.func, ast.Name)
    assert owner.func.id == (
        "_repair_stale_nchw_to_nhwc_channelwise_binary_transposes"
    )
    assert [ast.unparse(argument) for argument in owner.args] == ["fallback_ir"]
    assert owner.keywords == []
    prune_key = stats.value.keys[1]
    assert isinstance(prune_key, ast.Constant)
    assert prune_key.value == "pruned_unused_tensors"
    assert ast.unparse(stats.value.values[1]) == (
        "max(0, fallback_binary_layout_tensor_count - len(fallback_ir.tensors))"
    )

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_fallback_binary_layout_static_shape_stats"
    )
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = body[stats_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_binary_layout_stats.get("
        "'repaired_stale_nchw_to_nhwc_channelwise_binary_transposes', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_fallback_binary_layout_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "fallback_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    following = body[stats_index + 3]
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following.value) == (
        "_topologically_sort_operators(fallback_ir)"
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

    guard = body[stats_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_high_rank_bmm_stats.get("
        "'compressed_static_high_rank_batch_matmul', 0)) > 0"
    )

    convergence = body[stats_index + 3]
    assert isinstance(convergence, ast.Assign)
    assert isinstance(convergence.targets[0], ast.Name)
    assert convergence.targets[0].id == (
        "_fallback_binary_layout_convergence_stats"
    )
    assert ast.unparse(convergence.value) == (
        "_run_indexed_binary_layout_convergence(fallback_ir)"
    )

    final_sort = body[stats_index + 4]
    assert isinstance(final_sort, ast.Expr)
    assert ast.unparse(final_sort.value) == (
        "_topologically_sort_operators(fallback_ir)"
    )

    validation = body[stats_index + 5]
    assert isinstance(validation, ast.Assign)
    assert isinstance(validation.targets[0], ast.Name)
    assert validation.targets[0].id == "fallback_layout_problems"
    assert ast.unparse(validation.value) == (
        "validate_model_ir_layout_annotations(fallback_ir)"
    )

    validation_guard = body[stats_index + 6]
    assert isinstance(validation_guard, ast.If)
    assert ast.unparse(validation_guard.test) == (
        "len(fallback_layout_problems) > 0"
    )
    assert len(validation_guard.body) == 1
    assert ast.unparse(validation_guard.body[0]) == (
        "fallback_ir.metadata['logical_layout_validation_errors'] = "
        "list(fallback_layout_problems)"
    )
    assert len(validation_guard.orelse) == 1
    assert ast.unparse(validation_guard.orelse[0]) == (
        "fallback_ir.metadata.pop('logical_layout_validation_errors', None)"
    )

    terminal = body[stats_index + 7]
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

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_fallback_high_rank_bmm_static_shape_stats"
    )
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = body[stats_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(fallback_high_rank_bmm_stats.get("
        "'compressed_static_high_rank_batch_matmul', 0)) > 0"
    )
    assert len(guard.body) == 2
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_fallback_high_rank_bmm_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "fallback_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert isinstance(guard.body[1], ast.Expr)
    assert ast.unparse(guard.body[1].value) == (
        "_topologically_sort_operators(fallback_ir)"
    )

    following = body[stats_index + 3]
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
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following.value) == (
        "_topologically_sort_operators(fallback_ir)"
    )
