from __future__ import annotations

import ast
from pathlib import Path

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
    assert following.targets[0].id == "final_conv_input_tensor_count"


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


def test_primary_path_stages_final_mixed_concat_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_concat_layout_stats"
    )

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_final_mixed_concat_static_shape_stats"
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
        "int(final_concat_layout_stats.get("
        "'repaired_mixed_nhwc_inputs_for_nchw_concat', 0)) > 0"
    )
    assert len(guard.body) == 2
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_final_mixed_concat_static_shape_stats"
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
    assert following.targets[0].id == "final_concat_axis_stats"


def test_primary_path_stages_complete_final_concat_axis_binary_evidence() -> None:
    body = _lowerer_body()
    axis_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_concat_axis_stats"
    )

    axis_default = body[axis_index + 1]
    assert isinstance(axis_default, ast.Assign)
    assert isinstance(axis_default.targets[0], ast.Name)
    assert axis_default.targets[0].id == "_final_concat_axis_static_shape_stats"
    assert isinstance(axis_default.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(axis_default.value.keys, axis_default.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    axis_guard = body[axis_index + 2]
    assert isinstance(axis_guard, ast.If)
    assert ast.unparse(axis_guard.test) == (
        "int(final_concat_axis_stats.get("
        "'repaired_nchw_concat_transpose_conv_axes', 0)) > 0"
    )
    assert len(axis_guard.body) == 2
    axis_reconciliation = axis_guard.body[0]
    assert isinstance(axis_reconciliation, ast.Assign)
    assert isinstance(axis_reconciliation.targets[0], ast.Name)
    assert axis_reconciliation.targets[0].id == (
        "_final_concat_axis_static_shape_stats"
    )
    assert isinstance(axis_reconciliation.value, ast.Call)
    assert isinstance(axis_reconciliation.value.func, ast.Name)
    assert axis_reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [
        ast.unparse(argument) for argument in axis_reconciliation.value.args
    ] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in axis_reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert isinstance(axis_guard.body[1], ast.Expr)
    assert ast.unparse(axis_guard.body[1].value) == (
        "_topologically_sort_operators(model_ir)"
    )

    tensor_count = body[axis_index + 3]
    assert isinstance(tensor_count, ast.Assign)
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "final_binary_layout_tensor_count"
    assert ast.unparse(tensor_count.value) == "len(model_ir.tensors)"

    binary_index = axis_index + 4
    binary_stats = body[binary_index]
    assert isinstance(binary_stats, ast.Assign)
    assert isinstance(binary_stats.targets[0], ast.Name)
    assert binary_stats.targets[0].id == "final_binary_layout_stats"
    assert isinstance(binary_stats.value, ast.Dict)
    assert binary_stats.value.keys[0] is None
    binary_owner = binary_stats.value.values[0]
    assert isinstance(binary_owner, ast.Call)
    assert isinstance(binary_owner.func, ast.Name)
    assert binary_owner.func.id == (
        "_repair_stale_nchw_to_nhwc_channelwise_binary_transposes"
    )
    assert [ast.unparse(argument) for argument in binary_owner.args] == [
        "model_ir"
    ]
    assert binary_owner.keywords == []
    prune_key = binary_stats.value.keys[1]
    assert isinstance(prune_key, ast.Constant)
    assert prune_key.value == "pruned_unused_tensors"
    assert ast.unparse(binary_stats.value.values[1]) == (
        "max(0, final_binary_layout_tensor_count - len(model_ir.tensors))"
    )

    binary_default = body[binary_index + 1]
    assert isinstance(binary_default, ast.Assign)
    assert isinstance(binary_default.targets[0], ast.Name)
    assert binary_default.targets[0].id == (
        "_final_binary_layout_static_shape_stats"
    )
    assert isinstance(binary_default.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(
            binary_default.value.keys,
            binary_default.value.values,
        )
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    binary_guard = body[binary_index + 2]
    assert isinstance(binary_guard, ast.If)
    assert ast.unparse(binary_guard.test) == (
        "int(final_binary_layout_stats.get("
        "'repaired_stale_nchw_to_nhwc_channelwise_binary_transposes', 0)) > 0"
    )
    assert len(binary_guard.body) == 2
    binary_reconciliation = binary_guard.body[0]
    assert isinstance(binary_reconciliation, ast.Assign)
    assert isinstance(binary_reconciliation.targets[0], ast.Name)
    assert binary_reconciliation.targets[0].id == (
        "_final_binary_layout_static_shape_stats"
    )
    assert isinstance(binary_reconciliation.value, ast.Call)
    assert isinstance(binary_reconciliation.value.func, ast.Name)
    assert binary_reconciliation.value.func.id == (
        "_reconcile_static_tensor_shapes"
    )
    assert [
        ast.unparse(argument) for argument in binary_reconciliation.value.args
    ] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in binary_reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert isinstance(binary_guard.body[1], ast.Expr)
    assert ast.unparse(binary_guard.body[1].value) == (
        "_topologically_sort_operators(model_ir)"
    )

    following = body[binary_index + 3]
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following.value) == "_advance_post_progress()"


def test_primary_path_stages_final_prelu_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_prelu_stats"
    )

    tensor_count = body[stats_index - 1]
    assert isinstance(tensor_count, ast.Assign)
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "final_prelu_tensor_count"
    assert ast.unparse(tensor_count.value) == "len(model_ir.tensors)"

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == "_final_prelu_static_shape_stats"
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
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "rewritten_prelu_transpose_passthrough_chains"
    )
    assert "len(model_ir.tensors) < final_prelu_tensor_count" in ast.unparse(
        guard.test
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == "_final_prelu_static_shape_stats"
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

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_consecutive_reshape_stats"


def test_primary_path_stages_complete_final_placeholder_reconciliations() -> None:
    body = _lowerer_body()
    restore_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_placeholder_matmul_stats"
    )

    result_names = (
        "_final_placeholder_matmul_static_shape_stats",
        "_final_placeholder_binary_static_shape_stats",
    )
    for offset, result_name in enumerate(result_names, start=1):
        default_stats = body[restore_index + offset]
        assert isinstance(default_stats, ast.Assign)
        assert isinstance(default_stats.targets[0], ast.Name)
        assert default_stats.targets[0].id == result_name
        assert isinstance(default_stats.value, ast.Dict)
        assert {
            key.value: value.value
            for key, value in zip(
                default_stats.value.keys,
                default_stats.value.values,
            )
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
        } == {
            "reconciled_static_tensor_shapes": 0,
            "reconciled_static_shape_mutations": 0,
        }

    outer_guard = body[restore_index + 3]
    assert isinstance(outer_guard, ast.If)
    assert len(outer_guard.body) == 7

    first_reconciliation = outer_guard.body[0]
    assert isinstance(first_reconciliation, ast.Assign)
    assert isinstance(first_reconciliation.targets[0], ast.Name)
    assert first_reconciliation.targets[0].id == result_names[0]
    assert isinstance(first_reconciliation.value, ast.Call)
    assert isinstance(first_reconciliation.value.func, ast.Name)
    assert first_reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in first_reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    legacy_projection = outer_guard.body[1]
    assert isinstance(legacy_projection, ast.Assign)
    assert isinstance(legacy_projection.targets[0], ast.Name)
    assert legacy_projection.targets[0].id == "final_placeholder_reconcile_stats"
    assert isinstance(legacy_projection.value, ast.Dict)
    assert ast.unparse(legacy_projection.value) == (
        "{'reconciled_static_tensor_shapes': "
        "int(_final_placeholder_matmul_static_shape_stats.get("
        "'reconciled_static_tensor_shapes', 0))}"
    )

    assert isinstance(outer_guard.body[2], ast.Assign)
    assert isinstance(outer_guard.body[2].targets[0], ast.Name)
    assert outer_guard.body[2].targets[0].id == (
        "final_placeholder_binary_tensor_count"
    )
    assert isinstance(outer_guard.body[3], ast.Assign)
    assert isinstance(outer_guard.body[3].targets[0], ast.Name)
    assert outer_guard.body[3].targets[0].id == (
        "final_placeholder_exact_binary_stats"
    )
    assert isinstance(outer_guard.body[4], ast.Assign)
    assert isinstance(outer_guard.body[4].targets[0], ast.Name)
    assert outer_guard.body[4].targets[0].id == (
        "final_placeholder_singleton_binary_stats"
    )

    binary_guard = outer_guard.body[5]
    assert isinstance(binary_guard, ast.If)
    assert ast.unparse(binary_guard.test) == (
        "_stats_have_positive_count(final_placeholder_reconcile_stats, "
        "final_placeholder_exact_binary_stats, "
        "final_placeholder_singleton_binary_stats) or "
        "len(model_ir.tensors) < final_placeholder_binary_tensor_count"
    )
    assert len(binary_guard.body) == 1
    second_reconciliation = binary_guard.body[0]
    assert isinstance(second_reconciliation, ast.Assign)
    assert isinstance(second_reconciliation.targets[0], ast.Name)
    assert second_reconciliation.targets[0].id == result_names[1]
    assert isinstance(second_reconciliation.value, ast.Call)
    assert isinstance(second_reconciliation.value.func, ast.Name)
    assert second_reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in second_reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    following = body[restore_index + 4]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_se_fc_gather_tensor_count"


def test_primary_path_stages_final_mixed_singleton_concat_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_mixed_singleton_concat_stats"
    )

    result_name = "_final_mixed_singleton_concat_static_shape_stats"
    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == result_name
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
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "repaired_mixed_singleton_nchw_inputs_for_nhwc_concat"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == result_name
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

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_placeholder_matmul_stats"


def test_primary_path_stages_final_broadcast_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_broadcast_repair_stats"
    )

    result_name = "_final_broadcast_static_shape_stats"
    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == result_name
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
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "repaired_rank4_channelwise_broadcast_constants"
    )
    assert len(guard.body) == 3

    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == result_name
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert _call_name(_statement_call(guard.body[1])) == (
        "_topologically_sort_operators"
    )
    assert _call_name(_statement_call(guard.body[2])) == (
        "infer_model_ir_logical_layouts"
    )

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_mixed_singleton_concat_stats"


def test_primary_path_stages_final_instancenorm_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_instancenorm_repair_stats"
    )

    result_name = "_final_instancenorm_static_shape_stats"
    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == result_name
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
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "repaired_decomposed_instance_normalization_layouts"
    )
    assert len(guard.body) == 3

    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == result_name
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert _call_name(_statement_call(guard.body[1])) == (
        "_topologically_sort_operators"
    )
    assert _call_name(_statement_call(guard.body[2])) == (
        "infer_model_ir_logical_layouts"
    )

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_broadcast_repair_stats"


def test_primary_path_stages_final_convinteger_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_convinteger_layout_stats"
    )

    result_name = "_final_convinteger_static_shape_stats"
    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == result_name
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
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "repaired_channel_last_convinteger_input_transposes"
    )
    assert "propagated_channel_last_layout_hints" not in ast.unparse(guard.test)
    assert len(guard.body) == 3

    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == result_name
    assert isinstance(reconciliation.value, ast.Call)
    assert isinstance(reconciliation.value.func, ast.Name)
    assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconciliation.value.keywords
    } == {"include_mutation_count": "True"}
    assert _call_name(_statement_call(guard.body[1])) == (
        "_topologically_sort_operators"
    )
    assert _call_name(_statement_call(guard.body[2])) == (
        "infer_model_ir_logical_layouts"
    )

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_instancenorm_repair_stats"


def test_primary_path_stages_absolute_final_dynamic_rank1_result() -> None:
    body = _lowerer_body()
    owner_name = "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    direct_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == owner_name
    ]
    assert len(direct_indices) == 2

    very_late = body[direct_indices[0]]
    assert isinstance(very_late, ast.Assign)
    assert isinstance(very_late.targets[0], ast.Name)
    assert very_late.targets[0].id == "_very_late_dynamic_rank1_reshape_stats"

    final_index = direct_indices[1]
    final = body[final_index]
    assert isinstance(final, ast.Assign)
    assert isinstance(final.targets[0], ast.Name)
    assert final.targets[0].id == "_absolute_final_dynamic_rank1_stats"
    assert _call_name(_statement_call(final)) == owner_name

    assert _call_name(_statement_call(body[final_index + 1])) == (
        "_topologically_sort_operators"
    )
    assert _call_name(_statement_call(body[final_index + 2])) == (
        "infer_model_ir_logical_layouts"
    )
    following = body[final_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_convinteger_layout_stats"

    all_calls = sorted(
        (
            node
            for node in ast.walk(next(
                function
                for function in ast.parse(
                    LOWERER_PATH.read_text(encoding="utf-8")
                ).body
                if isinstance(function, ast.FunctionDef)
                and function.name == "lower_onnx_to_ir"
            ))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == owner_name
        ),
        key=lambda call: call.lineno,
    )
    assert len(all_calls) == 3


def test_primary_path_retains_absolute_final_boundary_signature_results() -> None:
    body = _lowerer_body()
    realign_name = "_realign_dynamic_boundary_shape_signature_map"
    sanitize_name = "_sanitize_static_shape_signature_consistency"
    realign_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == realign_name
    ]
    sanitize_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == sanitize_name
    ]
    assert len(realign_indices) == 3
    assert len(sanitize_indices) == 2

    expected_realign_targets = [
        "shared_boundary_signature_stats",
        "_absolute_final_boundary_signature_stats",
        "_final_dynamic_boundary_signature_stats",
    ]
    for index, target_name in zip(
        realign_indices,
        expected_realign_targets,
    ):
        statement = body[index]
        assert isinstance(statement, ast.Assign)
        assert len(statement.targets) == 1
        assert isinstance(statement.targets[0], ast.Name)
        assert statement.targets[0].id == target_name
        assert ast.unparse(statement.value) == f"{realign_name}(model_ir)"

    expected_sanitize_targets = [
        "late_signature_stats",
        "_absolute_final_static_signature_stats",
    ]
    for index, target_name in zip(
        sanitize_indices,
        expected_sanitize_targets,
    ):
        statement = body[index]
        assert isinstance(statement, ast.Assign)
        assert len(statement.targets) == 1
        assert isinstance(statement.targets[0], ast.Name)
        assert statement.targets[0].id == target_name
        assert ast.unparse(statement.value) == f"{sanitize_name}(model_ir)"

    absolute_realign_index = realign_indices[1]
    absolute_sanitize_index = sanitize_indices[1]
    assert absolute_sanitize_index == absolute_realign_index + 1
    following = body[absolute_sanitize_index + 1]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_absolute_final_affine_post_add_stats"


def test_primary_path_retains_guarded_no_layout_final_cleanup_results() -> None:
    body = _lowerer_body()
    guard_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test)
        == "apply_safe_transpose_reduction_lite_on_no_layout_opt"
        and any(
            _call_name(_statement_call(child)) == "run_se_fc_layout_cleanup"
            for child in statement.body
        )
    )
    guard = body[guard_index]
    assert isinstance(guard, ast.If)
    assert len(guard.body) == 3

    expected = (
        (
            "_no_layout_final_se_fc_stats",
            "run_se_fc_layout_cleanup",
            {
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            },
        ),
        (
            "_no_layout_final_affine_prepost_stats",
            "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
            {"layout_state": "session.layout_state"},
        ),
    )
    for statement, (target_name, function_name, keywords) in zip(
        guard.body[:2],
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
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == keywords

    assert _call_name(_statement_call(guard.body[2])) == (
        "_topologically_sort_operators"
    )
    previous = body[guard_index - 1]
    assert _call_name(_statement_call(previous)) == "_topologically_sort_operators"
    following = body[guard_index + 1]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_absolute_final_boundary_signature_stats"


def test_primary_path_stages_final_consecutive_reshape_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_consecutive_reshape_stats"
    )

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_final_consecutive_reshape_static_shape_stats"
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
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert {
        str(call.args[0].value)
        for call in get_calls
        if len(call.args) >= 1 and isinstance(call.args[0], ast.Constant)
    } == {
        "removed_noop_reshape_chains",
        "rewritten_consecutive_reshape_passthrough_chains",
        "rewritten_fanout_bypass_reshape_passthrough_chains",
    }
    assert len(get_calls) == 3
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_final_consecutive_reshape_static_shape_stats"
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

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_sinet_late_residual_stats"


def test_primary_path_stages_final_sinet_concat_resize_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_sinet_concat_resize_stats"
    )

    default_stats = body[stats_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == (
        "_final_sinet_concat_resize_static_shape_stats"
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
        "int(final_sinet_concat_resize_stats.get("
        "'optimized_sinet_concat_resize_affine_transpose_chains', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    assert isinstance(reconciliation, ast.Assign)
    assert isinstance(reconciliation.targets[0], ast.Name)
    assert reconciliation.targets[0].id == (
        "_final_sinet_concat_resize_static_shape_stats"
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

    following = body[stats_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_high_rank_bmm_stats"


def test_primary_path_stages_remaining_final_sinet_reconciliations() -> None:
    body = _lowerer_body()
    owners = (
        (
            "final_sinet_late_residual_stats",
            "_final_sinet_late_residual_static_shape_stats",
            "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains",
            "final_sinet_preadd_fanout_stats",
        ),
        (
            "final_sinet_preadd_fanout_stats",
            "_final_sinet_preadd_fanout_static_shape_stats",
            "optimized_sinet_deep_skip_pre_add_concat_prelu_fanout_chains",
            "final_sinet_dual_resize_stats",
        ),
        (
            "final_sinet_dual_resize_stats",
            "_final_sinet_dual_resize_static_shape_stats",
            "optimized_sinet_deep_skip_dual_resize_affine_transpose_chains",
            "final_sinet_shared_post_stats",
        ),
        (
            "final_sinet_shared_post_stats",
            "_final_sinet_shared_post_static_shape_stats",
            "optimized_sinet_shared_post_prelu_transpose_fanout_chains",
            "final_sinet_deep_skip_stats",
        ),
        (
            "final_sinet_deep_skip_stats",
            "_final_sinet_deep_skip_static_shape_stats",
            "optimized_sinet_deep_skip_concat_resize_affine_tail_chains",
            "final_sinet_concat_resize_stats",
        ),
    )

    for stats_name, result_name, stats_key, following_name in owners:
        stats_index = next(
            index
            for index, statement in enumerate(body)
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == stats_name
        )

        default_stats = body[stats_index + 1]
        assert isinstance(default_stats, ast.Assign)
        assert isinstance(default_stats.targets[0], ast.Name)
        assert default_stats.targets[0].id == result_name
        assert isinstance(default_stats.value, ast.Dict)
        assert {
            key.value: value.value
            for key, value in zip(
                default_stats.value.keys,
                default_stats.value.values,
            )
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
        } == {
            "reconciled_static_tensor_shapes": 0,
            "reconciled_static_shape_mutations": 0,
        }

        guard = body[stats_index + 2]
        assert isinstance(guard, ast.If)
        get_calls = [
            node
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        ]
        assert len(get_calls) == 1
        assert isinstance(get_calls[0].args[0], ast.Constant)
        assert get_calls[0].args[0].value == stats_key
        assert len(guard.body) == 1
        reconciliation = guard.body[0]
        assert isinstance(reconciliation, ast.Assign)
        assert isinstance(reconciliation.targets[0], ast.Name)
        assert reconciliation.targets[0].id == result_name
        assert isinstance(reconciliation.value, ast.Call)
        assert isinstance(reconciliation.value.func, ast.Name)
        assert reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
        assert [
            ast.unparse(argument) for argument in reconciliation.value.args
        ] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in reconciliation.value.keywords
        } == {"include_mutation_count": "True"}

        following = body[stats_index + 3]
        assert isinstance(following, ast.Assign)
        assert isinstance(following.targets[0], ast.Name)
        assert following.targets[0].id == following_name
