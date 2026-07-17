from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"


def _lowerer_body() -> list[ast.stmt]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    return lowerer.body


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if isinstance(statement, (ast.Assign, ast.Expr)) and isinstance(
        statement.value,
        ast.Call,
    ):
        return statement.value
    return None


def _call_name(call: ast.Call | None) -> str | None:
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _call_index(
    body: list[ast.stmt],
    function_name: str,
    *,
    start: int = 0,
) -> int:
    return next(
        index
        for index, statement in enumerate(body[start:], start=start)
        if _call_name(_statement_call(statement)) == function_name
    )


def test_primary_path_validates_terminal_layout_and_clears_stale_errors() -> None:
    body = _lowerer_body()
    convergence_index = _call_index(
        body,
        "_run_indexed_binary_layout_convergence",
    )
    coalesce_index = _call_index(
        body,
        "coalesce_static_high_rank_binary_operators",
        start=convergence_index + 1,
    )
    realign_index = _call_index(
        body,
        "_realign_dynamic_boundary_shape_signature_map",
        start=coalesce_index + 1,
    )
    terminal_sort_index = next(
        index
        for index in range(realign_index + 1, len(body))
        if _call_name(_statement_call(body[index]))
        == "_topologically_sort_operators"
    )
    validation_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "layout_problems"
    )

    assert convergence_index < coalesce_index < realign_index < terminal_sort_index
    assert validation_index == terminal_sort_index + 1
    validation = body[validation_index]
    assert isinstance(validation, ast.Assign)
    assert ast.unparse(validation.value) == (
        "validate_model_ir_layout_annotations(model_ir)"
    )

    guard = body[validation_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "len(layout_problems) > 0"
    assert len(guard.body) == 1
    assert ast.unparse(guard.body[0]) == (
        "model_ir.metadata['logical_layout_validation_errors'] = "
        "list(layout_problems)"
    )
    assert len(guard.orelse) == 1
    assert ast.unparse(guard.orelse[0]) == (
        "model_ir.metadata.pop('logical_layout_validation_errors', None)"
    )

    terminal = body[validation_index + 2]
    assert isinstance(terminal, ast.Return)
    assert ast.unparse(terminal.value) == "_finalize_model_ir(model_ir)"


def test_primary_path_retains_terminal_mutation_results() -> None:
    body = _lowerer_body()
    convergence_index = _call_index(
        body,
        "_run_indexed_binary_layout_convergence",
    )
    coalesce_index = _call_index(
        body,
        "coalesce_static_high_rank_binary_operators",
        start=convergence_index + 1,
    )
    realign_index = _call_index(
        body,
        "_realign_dynamic_boundary_shape_signature_map",
        start=coalesce_index + 1,
    )

    expected = (
        (
            convergence_index,
            "_final_binary_layout_convergence_stats",
            "_run_indexed_binary_layout_convergence",
            ["model_ir"],
            {},
        ),
        (
            coalesce_index,
            "_final_high_rank_binary_stats",
            "coalesce_static_high_rank_binary_operators",
            ["model_ir"],
            {"layout_state": "session.layout_state"},
        ),
        (
            realign_index,
            "_final_dynamic_boundary_signature_stats",
            "_realign_dynamic_boundary_shape_signature_map",
            ["model_ir"],
            {},
        ),
    )
    for index, result_name, function_name, arguments, keywords in expected:
        statement = body[index]
        assert isinstance(statement, ast.Assign)
        assert len(statement.targets) == 1
        assert isinstance(statement.targets[0], ast.Name)
        assert statement.targets[0].id == result_name
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id == function_name
        assert [ast.unparse(argument) for argument in call.args] == arguments
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == keywords


def test_primary_path_stages_final_high_rank_bmm_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_high_rank_bmm_stats"
    )

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_final_high_rank_bmm_static_shape_stats"
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
        "int(final_high_rank_bmm_stats.get("
        "'compressed_static_high_rank_batch_matmul', 0)) > 0"
    )
    assert len(guard.body) == 2
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_final_high_rank_bmm_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert isinstance(guard.body[1], ast.Expr)
    assert ast.unparse(guard.body[1].value) == (
        "_topologically_sort_operators(model_ir)"
    )

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_pad_layout_stats"


def test_primary_path_stages_final_pad_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_pad_layout_stats"
    )

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == "_final_pad_layout_static_shape_stats"
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
        "int(final_pad_layout_stats.get("
        "'repaired_channel_last_inputs_for_channel_first_pad', 0)) > 0"
    )
    assert len(guard.body) == 2
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_final_pad_layout_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert isinstance(guard.body[1], ast.Expr)
    assert ast.unparse(guard.body[1].value) == (
        "_topologically_sort_operators(model_ir)"
    )

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_conv_input_stats"


@pytest.mark.xfail(
    strict=True,
    reason="final Conv-input stats omit cleanup and reconciliation evidence",
)
def test_primary_path_stages_complete_final_conv_input_evidence() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_conv_input_stats"
    )

    tensor_count = body[stats_index - 1]
    assert isinstance(tensor_count, ast.Assign)
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "final_conv_input_tensor_count"
    assert ast.unparse(tensor_count.value) == "len(model_ir.tensors)"

    stats = body[stats_index]
    assert isinstance(stats.value, ast.Dict)
    assert stats.value.keys[0] is None
    owner = stats.value.values[0]
    assert isinstance(owner, ast.Call)
    assert isinstance(owner.func, ast.Name)
    assert owner.func.id == "_repair_stale_nchw_to_nhwc_conv_input_transposes"
    assert [ast.unparse(argument) for argument in owner.args] == ["model_ir"]
    assert owner.keywords == []
    prune_key = stats.value.keys[1]
    assert isinstance(prune_key, ast.Constant)
    assert prune_key.value == "pruned_unused_tensors"
    assert ast.unparse(stats.value.values[1]) == (
        "max(0, final_conv_input_tensor_count - len(model_ir.tensors))"
    )

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_final_conv_input_static_shape_stats"
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
        "int(final_conv_input_stats.get("
        "'repaired_stale_nchw_to_nhwc_conv_input_transposes', 0)) > 0"
    )
    assert len(guard.body) == 2
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_final_conv_input_static_shape_stats"
    )
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in reconciliation.value.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert isinstance(guard.body[1], ast.Expr)
    assert ast.unparse(guard.body[1].value) == (
        "_topologically_sort_operators(model_ir)"
    )

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_concat_layout_stats"
