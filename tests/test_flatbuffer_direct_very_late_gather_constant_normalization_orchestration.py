from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import constant_fold_cast_orchestration
from onnx2tf.tflite_builder.passes import pad_layout
from onnx2tf.tflite_builder.passes import (
    very_late_gather_constant_normalization_orchestration,
)
from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.very_late_gather_constant_normalization_orchestration import (
    VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS,
    VeryLateGatherConstantNormalizationContext,
    build_very_late_gather_constant_normalization_invocations,
    run_very_late_gather_constant_normalization,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
VERY_LATE = "_run_very_late_gather_constant_normalization_pass_cluster"


def _lowerer_and_helper() -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == VERY_LATE
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected expression: {ast.dump(node)}")


def _context() -> VeryLateGatherConstantNormalizationContext:
    model_ir = ModelIR("very_late_gather_constant_normalization_test")
    return VeryLateGatherConstantNormalizationContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: very_late_gather_constant_normalization_orchestration.RecoveryInvocation,
    context: VeryLateGatherConstantNormalizationContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        if value is context.diagnostics:
            return "session.diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_very_late_is_a_straight_line_delegate() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert len(helper.body) == 1
    assert not any(
        isinstance(
            node,
            (
                ast.AsyncFor,
                ast.AsyncWith,
                ast.For,
                ast.If,
                ast.Match,
                ast.Try,
                ast.While,
                ast.With,
            ),
        )
        for node in ast.walk(helper)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )

    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_very_late_gather_constant_normalization"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "very_late_gather_constant_normalization_context",
    )
    assert call.keywords == []


def test_very_late_preserves_all_four_effective_owner_contracts() -> None:
    context = _context()
    invocations = build_very_late_gather_constant_normalization_invocations(context)

    assert VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS == (
        "run_transpose_gather_axis_cleanup",
        *CONSTANT_FOLD_CAST_PASS_IDS,
        "run_normalization_pad_layout_cleanup",
    )
    assert (
        tuple(step.pass_id for step in invocations)
        == VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS
    )
    shared_contract = (
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[0]: shared_contract,
        CONSTANT_FOLD_CAST_PASS_IDS[0]: shared_contract,
        CONSTANT_FOLD_CAST_PASS_IDS[1]: shared_contract,
        VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[-1]: (
            ("model_ir",),
            {
                "include_instance": False,
                "include_flatten": True,
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
                "state_scope": "state_scope",
            },
        ),
    }


def test_very_late_composes_one_fresh_scope_across_both_builders() -> None:
    context = _context()
    invocations = build_very_late_gather_constant_normalization_invocations(context)
    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]

    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_very_late_gather_constant_normalization_invocations(context)[
            0
        ].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_very_late_runner_preserves_instrumented_effective_order(
    monkeypatch,
) -> None:
    context = _context()
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    monkeypatch.setattr(
        very_late_gather_constant_normalization_orchestration,
        "run_transpose_gather_axis_cleanup",
        recorder(VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[0]),
    )
    for pass_id in CONSTANT_FOLD_CAST_PASS_IDS:
        monkeypatch.setattr(
            constant_fold_cast_orchestration,
            pass_id,
            recorder(pass_id),
        )
    monkeypatch.setattr(
        very_late_gather_constant_normalization_orchestration,
        "run_normalization_pad_layout_cleanup",
        recorder(VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[-1]),
    )

    run_very_late_gather_constant_normalization(context)

    assert [pass_id for pass_id, _ in events] == list(
        VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS
    )
    assert all(scope is events[0][1] for _, scope in events)


def test_very_late_runner_returns_ordered_mutation_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    expected = (
        {"optimized_transpose_gather_transpose_axis_remap_nhwc_chains": 1},
        {
            "optimized_constant_input_pad_chains": 2,
            "optimized_constant_input_pool_chains": 3,
            "optimized_constant_input_cast_chains": 4,
        },
        {
            "optimized_redundant_int32_to_int64_passthrough_cast_chains": 5,
            "optimized_redundant_int64_to_int32_cast_chains": 6,
        },
        {
            "optimized_transpose_instancenorm_pad_prepost_nhwc_chains": 0,
            "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 7,
        },
    )

    monkeypatch.setattr(
        very_late_gather_constant_normalization_orchestration,
        "run_transpose_gather_axis_cleanup",
        lambda *args, **kwargs: expected[0],
    )
    monkeypatch.setattr(
        constant_fold_cast_orchestration,
        "run_constant_input_fold_cleanup",
        lambda *args, **kwargs: expected[1],
    )
    monkeypatch.setattr(
        constant_fold_cast_orchestration,
        "run_redundant_cast_cleanup",
        lambda *args, **kwargs: expected[2],
    )
    monkeypatch.setattr(
        very_late_gather_constant_normalization_orchestration,
        "run_normalization_pad_layout_cleanup",
        lambda *args, **kwargs: expected[3],
    )

    assert run_very_late_gather_constant_normalization(context) == expected


def test_very_late_mutation_summary_has_fixed_schema_and_net_pruning() -> None:
    summarize = getattr(
        very_late_gather_constant_normalization_orchestration,
        "summarize_very_late_gather_constant_normalization_mutations",
    )
    results = (
        {"optimized_transpose_gather_transpose_axis_remap_nhwc_chains": 1},
        {
            "optimized_constant_input_pad_chains": 2,
            "optimized_constant_input_pool_chains": 3,
            "optimized_constant_input_cast_chains": 4,
        },
        {
            "optimized_redundant_int32_to_int64_passthrough_cast_chains": 5,
            "optimized_redundant_int64_to_int32_cast_chains": 6,
        },
        {
            "optimized_transpose_instancenorm_pad_prepost_nhwc_chains": 0,
            "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 7,
        },
    )

    assert summarize(results, pruned_unused_tensors=8) == {
        "optimized_transpose_gather_transpose_axis_remap_nhwc_chains": 1,
        "optimized_constant_input_pad_chains": 2,
        "optimized_constant_input_pool_chains": 3,
        "optimized_constant_input_cast_chains": 4,
        "optimized_redundant_int32_to_int64_passthrough_cast_chains": 5,
        "optimized_redundant_int64_to_int32_cast_chains": 6,
        "optimized_transpose_instancenorm_pad_prepost_nhwc_chains": 0,
        "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 7,
        "pruned_unused_tensors": 8,
    }
    assert summarize(results, pruned_unused_tensors=-8)["pruned_unused_tensors"] == 0


@pytest.mark.parametrize("result_count", [0, 3, 5])
def test_very_late_mutation_summary_rejects_wrong_result_count(
    result_count: int,
) -> None:
    summarize = getattr(
        very_late_gather_constant_normalization_orchestration,
        "summarize_very_late_gather_constant_normalization_mutations",
    )

    with pytest.raises(
        ValueError,
        match=f"expected 4 pass results, got {result_count}",
    ):
        summarize(
            tuple({} for _ in range(result_count)),
            pruned_unused_tensors=0,
        )


def test_very_late_flatten_owner_can_prune_without_a_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("very_late_flatten_zero_rewrite_prune")
    prune_calls: list[tuple[ModelIR, LayoutState | None]] = []

    def record_prune(
        active_model_ir: ModelIR,
        *,
        layout_state: LayoutState | None = None,
    ) -> None:
        prune_calls.append((active_model_ir, layout_state))

    monkeypatch.setattr(pad_layout, "_prune_unused_tensors", record_prune)

    stats = pad_layout._optimize_transpose_flatten_globalnorm_pad_prepost_nhwc_chains(
        model_ir
    )

    assert stats == {
        "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 0
    }
    assert prune_calls == [(model_ir, None)]


def test_very_late_lowerer_stages_complete_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    resolve_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, (ast.Expr, ast.Assign))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_resolve_dynamic_reshape_shapes"
        and any(
            keyword.arg == "prefer_runtime_inferable_from_onnx_raw"
            for keyword in statement.value.keywords
        )
    )

    tensor_count = lowerer.body[resolve_index - 3]
    results = lowerer.body[resolve_index - 2]
    summary = lowerer.body[resolve_index - 1]
    assert isinstance(tensor_count, ast.Assign)
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "very_late_normalization_tensor_count"
    assert isinstance(results, ast.Assign)
    assert isinstance(results.targets[0], ast.Name)
    assert results.targets[0].id == "very_late_normalization_results"
    assert isinstance(results.value, ast.Call)
    assert isinstance(results.value.func, ast.Name)
    assert (
        results.value.func.id
        == "_run_very_late_gather_constant_normalization_pass_cluster"
    )
    assert isinstance(summary, ast.Assign)
    assert isinstance(summary.targets[0], ast.Name)
    assert summary.targets[0].id == "_very_late_normalization_stats"
    assert isinstance(summary.value, ast.Call)
    assert isinstance(summary.value.func, ast.Name)
    assert summary.value.func.id == (
        "summarize_very_late_gather_constant_normalization_mutations"
    )


def test_very_late_dynamic_reshape_captures_complete_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_resolve_dynamic_reshape_shapes"
            and any(
                keyword.arg == "prefer_runtime_inferable_from_onnx_raw"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            for node in ast.walk(statement)
        )
    )
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == "_very_late_dynamic_reshape_stats"
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == "_resolve_dynamic_reshape_shapes"
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    prefer_runtime = invocation.value.keywords[0]
    assert prefer_runtime.arg == "prefer_runtime_inferable_from_onnx_raw"
    assert isinstance(prefer_runtime.value, ast.Constant)
    assert prefer_runtime.value.value is True

    previous = lowerer.body[invocation_index - 1]
    assert isinstance(previous, ast.Assign)
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_very_late_normalization_stats"
    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "very_late_conv_input_tensor_count"

    direct_statements = [
        statement
        for statement in lowerer.body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_resolve_dynamic_reshape_shapes"
            for node in ast.walk(statement)
        )
    ]
    assert len(direct_statements) == 2
    assert isinstance(direct_statements[0], ast.Expr)
    assert direct_statements[1] is invocation


def test_very_late_conv_input_repairs_capture_complete_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    dynamic_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_very_late_dynamic_reshape_stats"
    )
    tensor_count = lowerer.body[dynamic_index + 1]
    stats = lowerer.body[dynamic_index + 2]
    following = lowerer.body[dynamic_index + 3]

    assert isinstance(tensor_count, ast.Assign)
    assert len(tensor_count.targets) == 1
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "very_late_conv_input_tensor_count"
    assert isinstance(stats, ast.Assign)
    assert len(stats.targets) == 1
    assert isinstance(stats.targets[0], ast.Name)
    assert stats.targets[0].id == "_very_late_conv_input_stats"
    assert isinstance(stats.value, ast.Dict)
    owner = stats.value.values[0]
    assert isinstance(owner, ast.Call)
    assert isinstance(owner.func, ast.Name)
    assert owner.func.id == "_run_indexed_conv_input_adapter_repairs"
    assert len(owner.args) == 1
    assert isinstance(owner.args[0], ast.Name)
    assert owner.args[0].id == "model_ir"
    assert owner.keywords == []
    assert len(stats.value.keys) == 2
    prune_key = stats.value.keys[1]
    assert isinstance(prune_key, ast.Constant)
    assert prune_key.value == "pruned_unused_tensors"

    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_very_late_stale_channel_shuffle_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == "run_stale_nchw_channel_shuffle_repair"

    fallback_assignments = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "fallback_conv_input_stats"
    ]
    assert len(fallback_assignments) == 1
    fallback = fallback_assignments[0]
    assert isinstance(fallback.value, ast.Dict)
    fallback_owner = fallback.value.values[0]
    assert isinstance(fallback_owner, ast.Call)
    assert isinstance(fallback_owner.func, ast.Name)
    assert fallback_owner.func.id == "_run_indexed_conv_input_adapter_repairs"


def test_very_late_stale_channel_shuffle_captures_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    conv_stats_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_very_late_conv_input_stats"
    )
    invocation = lowerer.body[conv_stats_index + 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        "_very_late_stale_channel_shuffle_stats"
    )
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == "run_stale_nchw_channel_shuffle_repair"
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 2
    keyword_paths = {
        keyword.arg: _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    }
    assert keyword_paths == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }

    following = lowerer.body[conv_stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_very_late_concat_transpose_conv_axis_stats"
    )
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == "_repair_nchw_concat_transpose_conv_axes"

    occurrences = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_stale_nchw_channel_shuffle_repair"
    ]
    assert len(occurrences) == 1
    assert occurrences[0] is invocation.value


def test_very_late_concat_transpose_conv_axis_captures_mutation_evidence() -> (
    None
):
    lowerer, _ = _lowerer_and_helper()
    shuffle_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_very_late_stale_channel_shuffle_stats"
    )
    invocation = lowerer.body[shuffle_index + 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        "_very_late_concat_transpose_conv_axis_stats"
    )
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == "_repair_nchw_concat_transpose_conv_axes"
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert _expression_path(layout_keyword.value) == "session.layout_state"

    following = lowerer.body[shuffle_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_very_late_concat_global_pool_conv_axis_stats"
    )
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == "_repair_nchw_concat_global_pool_conv_axes"

    existing_targets = {
        target.id
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        for target in node.targets
        if any(
            isinstance(candidate, ast.Call)
            and isinstance(candidate.func, ast.Name)
            and candidate.func.id == "_repair_nchw_concat_transpose_conv_axes"
            for candidate in ast.walk(node.value)
        )
    }
    assert existing_targets == {
        "_very_late_concat_transpose_conv_axis_stats",
        "fallback_concat_axis_stats",
        "final_concat_axis_stats",
    }


def test_very_late_concat_global_pool_conv_axis_captures_mutation_evidence() -> (
    None
):
    lowerer, _ = _lowerer_and_helper()
    transpose_axis_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_very_late_concat_transpose_conv_axis_stats"
    )
    invocation = lowerer.body[transpose_axis_index + 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        "_very_late_concat_global_pool_conv_axis_stats"
    )
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == "_repair_nchw_concat_global_pool_conv_axes"
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert _expression_path(layout_keyword.value) == "session.layout_state"

    following = lowerer.body[transpose_axis_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_very_late_dynamic_rank1_reshape_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == (
        "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    )

    occurrences = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_repair_nchw_concat_global_pool_conv_axes"
    ]
    assert len(occurrences) == 1
    assert occurrences[0] is invocation.value


def test_very_late_dynamic_rank1_reshape_captures_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    concat_pool_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_very_late_concat_global_pool_conv_axis_stats"
    )
    invocation = lowerer.body[concat_pool_index + 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == "_very_late_dynamic_rank1_reshape_stats"
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == (
        "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    )
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert _expression_path(layout_keyword.value) == "session.layout_state"

    following = lowerer.body[concat_pool_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_very_late_static_shape_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == "_reconcile_static_tensor_shapes"

    remaining_expressions = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id
        == "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    ]
    assert len(remaining_expressions) == 1
    remaining_inputs = []
    for expression in remaining_expressions:
        assert len(expression.value.args) == 1
        argument = expression.value.args[0]
        assert isinstance(argument, ast.Name)
        remaining_inputs.append(argument.id)
    assert remaining_inputs == ["model_ir"]


def test_very_late_static_reconciliation_captures_complete_mutation_evidence() -> (
    None
):
    lowerer, _ = _lowerer_and_helper()
    dynamic_rank1_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_very_late_dynamic_rank1_reshape_stats"
    )
    invocation = lowerer.body[dynamic_rank1_index + 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == "_very_late_static_shape_stats"
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == "_reconcile_static_tensor_shapes"
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    mutation_keyword = invocation.value.keywords[0]
    assert mutation_keyword.arg == "include_mutation_count"
    assert isinstance(mutation_keyword.value, ast.Constant)
    assert mutation_keyword.value.value is True

    following = lowerer.body[dynamic_rank1_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "split_fallback_stats"


def test_post_split_fallback_reconciliation_captures_complete_mutation_evidence() -> (
    None
):
    lowerer, _ = _lowerer_and_helper()
    split_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "split_fallback_stats"
    )

    default_stats = lowerer.body[split_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert len(default_stats.targets) == 1
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == "_post_split_fallback_static_shape_stats"
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(default_stats.value.keys, default_stats.value.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = lowerer.body[split_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(split_fallback_stats.get('replaced_unsupported_split_with_slice', 0)) > 0"
    )
    assert len(guard.body) == 1
    invocation = guard.body[0]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == "_post_split_fallback_static_shape_stats"
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == "_reconcile_static_tensor_shapes"
    assert [ast.unparse(argument) for argument in invocation.value.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_mutation_count": "True"}


def test_very_late_preserves_sole_terminal_invocation_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "very_late_normalization_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == VERY_LATE
    ]

    assert len(invocation_indexes) == 1
    invocation_index = invocation_indexes[0]
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert invocation.value.keywords == []

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "very_late_normalization_tensor_count"
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_very_late_normalization_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == (
        "summarize_very_late_gather_constant_normalization_mutations"
    )
    affine = lowerer.body[invocation_index - 2]
    assert isinstance(affine, ast.Assign)
    assert isinstance(affine.targets[0], ast.Name)
    assert affine.targets[0].id == "_very_late_affine_post_add_stats"
    resolve = lowerer.body[invocation_index + 2]
    assert isinstance(resolve, ast.Assign)
    assert len(resolve.targets) == 1
    assert isinstance(resolve.targets[0], ast.Name)
    assert resolve.targets[0].id == "_very_late_dynamic_reshape_stats"
    assert isinstance(resolve.value, ast.Call)
    assert isinstance(resolve.value.func, ast.Name)
    assert resolve.value.func.id == "_resolve_dynamic_reshape_shapes"


def test_very_late_affine_post_add_captures_complete_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "very_late_normalization_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == VERY_LATE
    )
    invocation = lowerer.body[invocation_index - 2]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == "_very_late_affine_post_add_stats"
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == (
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"

    previous = lowerer.body[invocation_index - 3]
    assert isinstance(previous, ast.Expr)
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
    )
    tensor_count = lowerer.body[invocation_index - 1]
    assert isinstance(tensor_count, ast.Assign)
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "very_late_normalization_tensor_count"
    following = lowerer.body[invocation_index]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == VERY_LATE

    direct_statements = [
        statement
        for statement in lowerer.body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
            for node in ast.walk(statement)
        )
    ]
    assert len(direct_statements) == 3
    assert isinstance(direct_statements[0], ast.Assign)
    first_target = direct_statements[0].targets[0]
    assert isinstance(first_target, ast.Name)
    assert first_target.id == "_pre_terminal_affine_post_add_stats"
    assert direct_statements[1] is invocation
    assert isinstance(direct_statements[2], ast.Assign)
    third_target = direct_statements[2].targets[0]
    assert isinstance(third_target, ast.Name)
    assert third_target.id == "_absolute_final_affine_post_add_stats"


def test_very_late_context_is_explicit() -> None:
    lowerer, _ = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "very_late_gather_constant_normalization_context"
            for target in statement.targets
        )
    )

    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_very_late_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "very_late_gather_constant_normalization_orchestration.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.lower_from_onnx2tf"
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "onnx2tf.tflite_builder.lower_from_onnx2tf"
            for alias in node.names
        )
        for node in tree.body
    )
