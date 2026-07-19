from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import constant_fold_cast_orchestration
from onnx2tf.tflite_builder.passes import (
    late_layout_mean_spp_gather_constant_cast_orchestration,
)
from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.late_layout_mean_spp_gather_constant_cast_orchestration import (
    LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_PASS_IDS,
    LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS,
    LateLayoutMeanSPPGatherConstantCastContext,
    build_late_layout_mean_spp_gather_constant_cast_invocations,
    run_late_layout_mean_spp_gather_constant_cast,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PHASE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_layout_mean_spp_gather_constant_cast_orchestration.py"
)
LATE_LAYOUT = "_run_late_layout_mean_spp_gather_constant_cast_pass_cluster"
CONSTANT_FOLD_CAST = "_run_constant_fold_cast_cleanup_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == LATE_LAYOUT
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


def _context(*, use_layout_state: bool) -> LateLayoutMeanSPPGatherConstantCastContext:
    model_ir = ModelIR("late_layout_mean_spp_gather_constant_cast_test")
    return LateLayoutMeanSPPGatherConstantCastContext(
        model_ir=model_ir,
        layout_state=(
            LayoutState.from_model_ir(model_ir) if use_layout_state else None
        ),
        diagnostics=[],
    )


def _normalize_contract(
    invocation: late_layout_mean_spp_gather_constant_cast_orchestration.RecoveryInvocation,
    context: LateLayoutMeanSPPGatherConstantCastContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "layout_state"
        if value is context.diagnostics:
            return "diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_late_layout_context_and_delegate_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_layout_transpose"
    ]
    assert helper.args.kw_defaults == [None]
    assert helper.args.defaults == []
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
    assert call.func.id == "run_late_layout_mean_spp_gather_constant_cast"
    assert tuple(_expression_path(argument) for argument in call.args) == (
        "late_layout_mean_spp_gather_constant_cast_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {"include_layout_transpose": "include_layout_transpose"}

    context_assignment = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "late_layout_mean_spp_gather_constant_cast_context"
            for target in node.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize("include_layout_transpose", [False, True])
@pytest.mark.parametrize("use_layout_state", [False, True])
def test_late_layout_preserves_both_policy_contracts(
    include_layout_transpose: bool,
    use_layout_state: bool,
) -> None:
    context = _context(use_layout_state=use_layout_state)
    invocations = build_late_layout_mean_spp_gather_constant_cast_invocations(
        context,
        include_layout_transpose=include_layout_transpose,
    )
    expected_ids = (
        LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_PASS_IDS
        if include_layout_transpose
        else LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS
    )

    assert tuple(invocation.pass_id for invocation in invocations) == expected_ids
    shared_contract = (
        ("model_ir",),
        {
            "layout_state": "layout_state",
            "diagnostics": "diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        invocation.pass_id: _normalize_contract(invocation, context)
        for invocation in invocations
    } == {pass_id: shared_contract for pass_id in expected_ids}

    scopes = [
        dict(invocation.keyword_args)["state_scope"] for invocation in invocations
    ]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_late_layout_mean_spp_gather_constant_cast_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_late_layout_runner_preserves_both_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
) -> None:
    context = _context(use_layout_state=True)
    expected_ids = (
        LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_PASS_IDS
        if include_layout_transpose
        else LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS
    )
    events: list[tuple[str, ModelIRPassStateScope]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs["state_scope"]))

        return record

    for pass_id in LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_PASS_IDS[
        : -len(CONSTANT_FOLD_CAST_PASS_IDS)
    ]:
        monkeypatch.setattr(
            late_layout_mean_spp_gather_constant_cast_orchestration,
            pass_id,
            recorder(pass_id),
        )
    for pass_id in CONSTANT_FOLD_CAST_PASS_IDS:
        monkeypatch.setattr(
            constant_fold_cast_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_late_layout_mean_spp_gather_constant_cast(
        context,
        include_layout_transpose=include_layout_transpose,
    )

    assert [pass_id for pass_id, _ in events] == list(expected_ids)
    assert all(scope is events[0][1] for _, scope in events)


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_late_layout_returns_ordered_results_through_lowerer_helper(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
) -> None:
    context = _context(use_layout_state=True)
    expected_ids = (
        LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_PASS_IDS
        if include_layout_transpose
        else LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS
    )
    expected_results = tuple(
        {f"mutation_{index}": index + 1}
        for index in range(len(expected_ids))
    )

    def return_results(invocations, *, expected_pass_ids, phase_name):
        assert tuple(invocation.pass_id for invocation in invocations) == tuple(
            expected_ids
        )
        assert tuple(expected_pass_ids) == tuple(expected_ids)
        assert phase_name == (
            "late layout/mean/SPP/gather/constant-fold/cast"
        )
        return expected_results

    monkeypatch.setattr(
        late_layout_mean_spp_gather_constant_cast_orchestration,
        "run_recovery_invocations",
        return_results,
    )

    result = run_late_layout_mean_spp_gather_constant_cast(
        context,
        include_layout_transpose=include_layout_transpose,
    )

    assert result == expected_results
    _, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert isinstance(helper.body[0].value, ast.Call)
    assert isinstance(helper.body[0].value.func, ast.Name)
    assert (
        helper.body[0].value.func.id
        == "run_late_layout_mean_spp_gather_constant_cast"
    )


@pytest.mark.parametrize("include_layout_transpose", [False, True])
def test_late_layout_mutation_summary_filters_iterations_and_reports_pruning(
    include_layout_transpose: bool,
) -> None:
    summarize = getattr(
        late_layout_mean_spp_gather_constant_cast_orchestration,
        "summarize_late_layout_mean_spp_gather_constant_cast_mutations",
    )
    layout_result = {
        "iterations": 9,
        "removed_identity_transpose": 1,
        "removed_inverse_transpose_pairs": 2,
        "removed_inverse_transpose_fanout_branches": 3,
        "composed_consecutive_transpose_pairs": 4,
    }
    required_results = (
        {"mean_mutations": 5},
        {"spp_mutations": 6},
        {"gather_mutations": 7},
        {"constant_fold_mutations": 8},
        {"cast_mutations": 10},
    )
    pass_results = (
        (layout_result, *required_results)
        if include_layout_transpose
        else required_results
    )

    result = summarize(
        pass_results,
        include_layout_transpose=include_layout_transpose,
        pruned_unused_tensors=11,
    )

    expected_layout = {
        "removed_identity_transpose": int(include_layout_transpose),
        "removed_inverse_transpose_pairs": int(include_layout_transpose) * 2,
        "removed_inverse_transpose_fanout_branches": (
            int(include_layout_transpose) * 3
        ),
        "composed_consecutive_transpose_pairs": (
            int(include_layout_transpose) * 4
        ),
    }
    assert result == {
        **expected_layout,
        "mean_mutations": 5,
        "spp_mutations": 6,
        "gather_mutations": 7,
        "constant_fold_mutations": 8,
        "cast_mutations": 10,
        "pruned_unused_tensors": 11,
    }
    assert "iterations" not in result
    with pytest.raises(
        ValueError,
        match=r"late layout mutation summary expected [56] pass results",
    ):
        summarize(
            pass_results[:-1],
            include_layout_transpose=include_layout_transpose,
            pruned_unused_tensors=0,
        )


def test_lowerer_captures_late_layout_cluster_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    summary = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_late_layout_cluster_stats"
    )
    summary_call = summary.value
    assert isinstance(summary_call, ast.Call)
    assert isinstance(summary_call.func, ast.Name)
    assert summary_call.func.id == (
        "run_late_layout_mean_spp_gather_constant_cast_summary"
    )
    assert len(summary_call.args) == 1
    assert isinstance(summary_call.args[0], ast.Name)
    assert summary_call.args[0].id == (
        "late_layout_mean_spp_gather_constant_cast_context"
    )
    assert {
        keyword.arg: _expression_path(keyword.value)
        for keyword in summary_call.keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}
    assert not any(
        isinstance(node, ast.Name)
        and node.id
        in {"late_layout_cluster_tensor_count", "late_layout_cluster_results"}
        for node in ast.walk(lowerer)
    )


def test_late_layout_has_one_required_policy_production_call() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "run_late_layout_mean_spp_gather_constant_cast_summary"
    ]

    assert len(invocations) == 1
    assert [ast.unparse(argument) for argument in invocations[0].args] == [
        "late_layout_mean_spp_gather_constant_cast_context"
    ]
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocations[0].keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}


def test_late_layout_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_late_layout_cluster_stats"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id
        == "run_late_layout_mean_spp_gather_constant_cast_summary"
    )

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == (
        "_late_pre_layout_cluster_shape_extract_stats"
    )
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )
    assert isinstance(following, ast.Assign)
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert (
        following.value.func.id
        == "_replace_expand_dims_and_squeeze_with_reshape"
    )


def test_late_layout_composes_child_builder_without_lowerer_import() -> None:
    tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_late_layout_mean_spp_gather_constant_cast_invocations"
    )
    child_calls = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_constant_fold_cast_invocations"
    ]

    assert len(child_calls) == 1
    child_call = child_calls[0]
    assert len(child_call.args) == 1
    assert isinstance(child_call.args[0], ast.Name)
    assert child_call.args[0].id == "context"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in child_call.keywords
    } == {"state_scope": "state_scope"}

    imported_modules = {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules


def test_late_layout_removes_dead_constant_fold_cast_lowerer_state() -> None:
    lowerer, _ = _lowerer_and_helper()
    nested_function_names = {
        node.name for node in lowerer.body if isinstance(node, ast.FunctionDef)
    }
    lowerer_names = {
        node.id for node in ast.walk(lowerer) if isinstance(node, ast.Name)
    }

    assert CONSTANT_FOLD_CAST not in nested_function_names
    assert "constant_fold_cast_context" not in lowerer_names
    assert "run_constant_fold_cast" not in lowerer_names
    assert "ConstantFoldCastContext" not in lowerer_names
