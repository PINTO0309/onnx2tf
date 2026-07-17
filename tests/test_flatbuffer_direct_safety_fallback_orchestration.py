from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import pad_layout


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

    topological_sort = body[invocation_index + 1]
    assert isinstance(topological_sort, ast.Expr)
    assert ast.unparse(topological_sort.value) == (
        "_topologically_sort_operators(fallback_ir)"
    )
    layout_inference = body[invocation_index + 2]
    assert isinstance(layout_inference, ast.Expr)
    assert ast.unparse(layout_inference.value) == (
        "infer_model_ir_logical_layouts(fallback_ir)"
    )


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
    assert len(guard.body) == 3
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

    assert ast.unparse(guard.body[1]) == (
        "_topologically_sort_operators(fallback_ir)"
    )
    assert ast.unparse(guard.body[2]) == (
        "infer_model_ir_logical_layouts(fallback_ir)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="the fallback SE/FC/Gather reconciliation result is discarded",
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
    assert isinstance(following, ast.If)
    assert "_restore_placeholder_matmul_flattened_inputs(fallback_ir)" in (
        ast.unparse(following.test)
    )
