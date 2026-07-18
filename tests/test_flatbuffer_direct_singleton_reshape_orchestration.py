from __future__ import annotations

import ast
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import singleton_reshape_orchestration
from onnx2tf.tflite_builder.passes.singleton_reshape_orchestration import (
    SINGLETON_RESHAPE_BASE_PASS_IDS,
    SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
    SINGLETON_RESHAPE_DUPLICATE_BASE_PASS_IDS,
    SINGLETON_RESHAPE_DUPLICATE_PASS_IDS,
    SINGLETON_RESHAPE_LAYOUT_MULTI_PASS_IDS,
    SINGLETON_RESHAPE_LAYOUT_PASS_IDS,
    SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS,
    SINGLETON_RESHAPE_PASS_IDS,
    SINGLETON_RESHAPE_PREFIX_PASS_IDS,
    SingletonReshapeContext,
    build_singleton_reshape_invocations,
    run_singleton_reshape,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PHASE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "singleton_reshape_orchestration.py"
)
SINGLETON_RESHAPE = "_run_singleton_reshape_layout_pass_cluster"
POLICIES = tuple(product((False, True), repeat=4))


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
        if isinstance(node, ast.FunctionDef) and node.name == SINGLETON_RESHAPE
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


def _direct_call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    if not isinstance(statement.value.func, ast.Name):
        return None
    return statement.value.func.id


def _context(*, use_layout_state: bool = False) -> SingletonReshapeContext:
    model_ir = ModelIR("singleton_reshape_test")
    return SingletonReshapeContext(
        model_ir=model_ir,
        layout_state=(
            LayoutState.from_model_ir(model_ir) if use_layout_state else None
        ),
        diagnostics=[],
    )


def _expected_ids(
    include_layout_transpose: bool,
    include_duplicate_fanout: bool,
    include_multi_branch_gate: bool,
) -> tuple[str, ...]:
    return (
        *(SINGLETON_RESHAPE_LAYOUT_PASS_IDS if include_layout_transpose else ()),
        *SINGLETON_RESHAPE_PREFIX_PASS_IDS,
        *(SINGLETON_RESHAPE_DUPLICATE_PASS_IDS if include_duplicate_fanout else ()),
        *SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
        *(SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS if include_multi_branch_gate else ()),
    )


def _normalize_contract(
    invocation: singleton_reshape_orchestration.RecoveryInvocation,
    context: SingletonReshapeContext,
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


def test_singleton_reshape_context_and_delegate_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_layout_transpose",
        "include_duplicate_fanout",
        "include_multi_branch_gate",
        "include_spatial_concat_post_transpose",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        False,
        False,
        True,
    ]
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
    assert call.func.id == "run_singleton_reshape"
    assert tuple(_expression_path(argument) for argument in call.args) == (
        "singleton_reshape_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {
        "include_layout_transpose": "include_layout_transpose",
        "include_duplicate_fanout": "include_duplicate_fanout",
        "include_multi_branch_gate": "include_multi_branch_gate",
        "include_spatial_concat_post_transpose": (
            "include_spatial_concat_post_transpose"
        ),
    }

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "singleton_reshape_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize(
    (
        "include_layout_transpose",
        "include_duplicate_fanout",
        "include_multi_branch_gate",
        "include_spatial_concat_post_transpose",
    ),
    POLICIES,
)
def test_singleton_reshape_preserves_all_policy_contracts(
    include_layout_transpose: bool,
    include_duplicate_fanout: bool,
    include_multi_branch_gate: bool,
    include_spatial_concat_post_transpose: bool,
) -> None:
    context = _context()
    invocations = build_singleton_reshape_invocations(
        context,
        include_layout_transpose=include_layout_transpose,
        include_duplicate_fanout=include_duplicate_fanout,
        include_multi_branch_gate=include_multi_branch_gate,
        include_spatial_concat_post_transpose=(include_spatial_concat_post_transpose),
    )
    expected_ids = _expected_ids(
        include_layout_transpose,
        include_duplicate_fanout,
        include_multi_branch_gate,
    )

    assert tuple(invocation.pass_id for invocation in invocations) == expected_ids
    shared_contract = {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    for invocation in invocations:
        args, keyword_args = _normalize_contract(invocation, context)
        if invocation.pass_id == SINGLETON_RESHAPE_DUPLICATE_PASS_IDS[0]:
            assert keyword_args.pop("include_transpose") is False
        if invocation.pass_id == SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[5]:
            assert keyword_args.pop("include_concat_post_transpose") is (
                include_spatial_concat_post_transpose
            )
        assert args == ("model_ir",)
        assert keyword_args == shared_contract

    scopes = [
        dict(invocation.keyword_args)["state_scope"] for invocation in invocations
    ]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_singleton_reshape_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
            include_duplicate_fanout=include_duplicate_fanout,
            include_multi_branch_gate=include_multi_branch_gate,
            include_spatial_concat_post_transpose=(
                include_spatial_concat_post_transpose
            ),
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_singleton_reshape_preserves_layout_state_and_named_sequences() -> None:
    context = _context(use_layout_state=True)
    invocations = build_singleton_reshape_invocations(context)
    scopes = [
        dict(invocation.keyword_args)["state_scope"] for invocation in invocations
    ]

    assert SINGLETON_RESHAPE_BASE_PASS_IDS == (
        *SINGLETON_RESHAPE_PREFIX_PASS_IDS,
        *SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
    )
    assert SINGLETON_RESHAPE_LAYOUT_MULTI_PASS_IDS == (
        *SINGLETON_RESHAPE_LAYOUT_PASS_IDS,
        *SINGLETON_RESHAPE_BASE_PASS_IDS,
        *SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS,
    )
    assert SINGLETON_RESHAPE_DUPLICATE_BASE_PASS_IDS == (
        *SINGLETON_RESHAPE_PREFIX_PASS_IDS,
        *SINGLETON_RESHAPE_DUPLICATE_PASS_IDS,
        *SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
    )
    assert SINGLETON_RESHAPE_PASS_IDS == (
        *SINGLETON_RESHAPE_LAYOUT_PASS_IDS,
        *SINGLETON_RESHAPE_PREFIX_PASS_IDS,
        *SINGLETON_RESHAPE_DUPLICATE_PASS_IDS,
        *SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
        *SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS,
    )
    assert all(scope.layout_state is context.layout_state for scope in scopes)


@pytest.mark.parametrize(
    (
        "include_layout_transpose",
        "include_duplicate_fanout",
        "include_multi_branch_gate",
        "include_spatial_concat_post_transpose",
    ),
    POLICIES,
)
def test_singleton_reshape_runner_preserves_all_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    include_duplicate_fanout: bool,
    include_multi_branch_gate: bool,
    include_spatial_concat_post_transpose: bool,
) -> None:
    context = _context(use_layout_state=True)
    events: list[tuple[str, dict[str, Any]]] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append((pass_id, kwargs))

        return record

    for pass_id in SINGLETON_RESHAPE_PASS_IDS:
        monkeypatch.setattr(
            singleton_reshape_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_singleton_reshape(
        context,
        include_layout_transpose=include_layout_transpose,
        include_duplicate_fanout=include_duplicate_fanout,
        include_multi_branch_gate=include_multi_branch_gate,
        include_spatial_concat_post_transpose=(include_spatial_concat_post_transpose),
    )
    expected_ids = _expected_ids(
        include_layout_transpose,
        include_duplicate_fanout,
        include_multi_branch_gate,
    )

    assert [pass_id for pass_id, _ in events] == list(expected_ids)
    assert all(
        keyword_args["state_scope"] is events[0][1]["state_scope"]
        for _, keyword_args in events
    )
    spatial_kwargs = next(
        keyword_args
        for pass_id, keyword_args in events
        if pass_id == SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[5]
    )
    assert spatial_kwargs["include_concat_post_transpose"] is (
        include_spatial_concat_post_transpose
    )
    if include_duplicate_fanout:
        duplicate_kwargs = next(
            keyword_args
            for pass_id, keyword_args in events
            if pass_id == SINGLETON_RESHAPE_DUPLICATE_PASS_IDS[0]
        )
        assert duplicate_kwargs["include_transpose"] is False


def test_singleton_reshape_propagates_policy_results_to_direct_primary_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(use_layout_state=True)
    result_by_pass_id = {
        pass_id: {f"{pass_id}_mutations": index}
        for index, pass_id in enumerate(SINGLETON_RESHAPE_PASS_IDS, start=1)
    }
    for pass_id, result in result_by_pass_id.items():
        monkeypatch.setattr(
            singleton_reshape_orchestration,
            pass_id,
            lambda *args, result=result, **kwargs: dict(result),
        )

    for (
        include_layout_transpose,
        include_duplicate_fanout,
        include_multi_branch_gate,
        include_spatial_concat_post_transpose,
    ) in POLICIES:
        expected_ids = _expected_ids(
            include_layout_transpose,
            include_duplicate_fanout,
            include_multi_branch_gate,
        )
        assert run_singleton_reshape(
            context,
            include_layout_transpose=include_layout_transpose,
            include_duplicate_fanout=include_duplicate_fanout,
            include_multi_branch_gate=include_multi_branch_gate,
            include_spatial_concat_post_transpose=(
                include_spatial_concat_post_transpose
            ),
        ) == tuple(result_by_pass_id[pass_id] for pass_id in expected_ids)

    lowerer, helper = _lowerer_and_helper()
    assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
    assert len(helper.body) == 1
    delegate = helper.body[0]
    assert isinstance(delegate, ast.Return)
    assert isinstance(delegate.value, ast.Call)
    assert isinstance(delegate.value.func, ast.Name)
    assert delegate.value.func.id == "run_singleton_reshape"

    direct_results = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, (ast.Assign, ast.Expr))
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == SINGLETON_RESHAPE
        ),
        key=lambda node: node.lineno,
    )
    assert len(direct_results) == 2
    assert all(isinstance(result, ast.Assign) for result in direct_results)
    assert [
        result.targets[0].id
        for result in direct_results
        if isinstance(result, ast.Assign)
        and len(result.targets) == 1
        and isinstance(result.targets[0], ast.Name)
    ] == [
        "_terminal_singleton_reshape_results",
        "_post_terminal_singleton_reshape_results",
    ]
    assert [
        {
            keyword.arg: _expression_path(keyword.value)
            for keyword in result.value.keywords
        }
        for result in direct_results
    ] == [
        {
            "include_layout_transpose": True,
            "include_multi_branch_gate": True,
        },
        {
            "include_duplicate_fanout": True,
            "include_spatial_concat_post_transpose": False,
        },
    ]


def test_singleton_reshape_preserves_layout_multi_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    outer_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == SINGLETON_RESHAPE
            for node in ast.walk(statement)
        )
    )
    guard = lowerer.body[outer_index]
    assert isinstance(guard, ast.If)
    invocation_index = next(
        index
        for index, statement in enumerate(guard.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_singleton_reshape_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINGLETON_RESHAPE
    )
    invocation = guard.body[invocation_index]

    assert invocation_index == len(guard.body) - 1
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {
        "include_layout_transpose": True,
        "include_multi_branch_gate": True,
    }
    assert _direct_call_name(guard.body[invocation_index - 1]) == (
        "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )
    assert _direct_call_name(lowerer.body[outer_index + 1]) == (
        "_run_terminal_clamp_unary_relu_pass_cluster"
    )


def test_singleton_reshape_preserves_duplicate_spatial_policy_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_post_terminal_singleton_reshape_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SINGLETON_RESHAPE
    )
    invocation = lowerer.body[invocation_index]

    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocation.value.keywords
    } == {
        "include_duplicate_fanout": True,
        "include_spatial_concat_post_transpose": False,
    }
    assert _direct_call_name(lowerer.body[invocation_index - 1]) == (
        "_run_sinet_preadd_resize_recovery_sequence"
    )
    assert _direct_call_name(lowerer.body[invocation_index + 1]) == (
        "_run_indexed_shape_convergence_cleanup"
    )


def test_singleton_reshape_phase_imports_owners_without_lowerer() -> None:
    tree = ast.parse(PHASE_PATH.read_text(encoding="utf-8"))
    imported_modules = {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
    assert {
        "onnx2tf.tflite_builder.passes.graph_cleanup",
        "onnx2tf.tflite_builder.passes.layout_transpose",
        "onnx2tf.tflite_builder.passes.multi_branch_gate_layout",
        "onnx2tf.tflite_builder.passes.singleton_maxpool_layout",
        "onnx2tf.tflite_builder.passes.singleton_reshape_layout",
    } <= imported_modules
