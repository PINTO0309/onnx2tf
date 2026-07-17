from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

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
