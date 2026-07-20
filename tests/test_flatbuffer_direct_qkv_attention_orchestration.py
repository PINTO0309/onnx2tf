from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import qkv_attention_orchestration
from onnx2tf.tflite_builder.passes.qkv_attention_orchestration import (
    QKV_ATTENTION_PASS_IDS,
    QKVAttentionContext,
    active_qkv_attention_pass_ids,
    build_qkv_attention_invocations,
    run_qkv_attention,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
QKV_ATTENTION = "_run_qkv_attention_layout_pass_cluster"
DEFAULT_PASS_IDS = QKV_ATTENTION_PASS_IDS[1:]
BRIDGE_ONLY_PASS_IDS = QKV_ATTENTION_PASS_IDS[2:]
LAYOUT_BRIDGE_PASS_IDS = (
    QKV_ATTENTION_PASS_IDS[0],
    QKV_ATTENTION_PASS_IDS[2],
)
PRODUCTION_FORMS = [
    (False, True, DEFAULT_PASS_IDS),
    (False, False, BRIDGE_ONLY_PASS_IDS),
    (True, False, LAYOUT_BRIDGE_PASS_IDS),
]
TERMINAL_QKV_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_shape_attention_orchestration.py"
)
TERMINAL_QKV_OWNER = "run_terminal_qkv_shape_attention_cleanup"
TERMINAL_QKV_RESULT = "_terminal_qkv_shape_attention_results"
POST_SINET_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "post_sinet_qkv_relu_split_all_orchestration.py"
)
POST_SINET_OWNER = "run_post_sinet_qkv_relu_split_all_cleanup"
POST_SINET_PHASE_ID = "cleanup.post_sinet.relu_split_all_outputs"
POST_SINET_OWNER_EXPRESSION = (
    "run_post_sinet_qkv_relu_split_all_cleanup("
    "shared_model_ir_pass_context)[1]"
)
OUTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_activation_bridge_orchestration.py"
)
OUTER_OWNER = "run_terminal_qkv_activation_bridge_cleanup"
OUTER_RESULT = "_terminal_qkv_activation_bridge_results"
TOP_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_activation_layout_shape_orchestration.py"
)
TOP_OWNER = "run_terminal_qkv_activation_layout_shape_cleanup"
TOP_RESULT = "_terminal_qkv_activation_layout_shape_results"
LOWERER_OWNER = "run_terminal_affine_qkv_layout_shape_cleanup"
LOWERER_RESULT = "_terminal_affine_qkv_layout_shape_results"


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
        if isinstance(node, ast.FunctionDef) and node.name == QKV_ATTENTION
    )
    return lowerer, helper


def _terminal_qkv_owner_calls(function_name: str) -> list[ast.Call]:
    tree = ast.parse(TERMINAL_QKV_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_QKV_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _post_sinet_owner_calls(function_name: str) -> list[ast.Call]:
    tree = ast.parse(POST_SINET_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == POST_SINET_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _outer_calls() -> list[ast.Call]:
    tree = ast.parse(OUTER_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == OUTER_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_QKV_OWNER
    ]


def _top_calls() -> list[ast.Call]:
    tree = ast.parse(TOP_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == TOP_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OUTER_OWNER
    ]


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        call = call.args[1]
    if not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _assert_phase_result_record(statement: ast.stmt, phase_id: str) -> None:
    assert isinstance(statement, ast.Expr)
    record = statement.value
    assert isinstance(record, ast.Call)
    assert isinstance(record.func, ast.Attribute)
    assert isinstance(record.func.value, ast.Name)
    assert record.func.value.id == "session"
    assert record.func.attr == "record_phase_result"
    assert len(record.args) == 2
    assert ast.literal_eval(record.args[0]) == phase_id


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _context() -> QKVAttentionContext:
    model_ir = ModelIR("qkv_attention_test")
    return QKVAttentionContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: qkv_attention_orchestration.RecoveryInvocation,
    context: QKVAttentionContext,
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


def test_qkv_attention_signature_and_delegate_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == [
        "include_layout_transpose",
        "include_prefix",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
    ]
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
    assert call.func.id == "run_qkv_attention"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "qkv_attention_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {
        "include_layout_transpose": "include_layout_transpose",
        "include_prefix": "include_prefix",
    }


@pytest.mark.parametrize(
    ("include_layout_transpose", "include_prefix", "expected_ids"),
    PRODUCTION_FORMS,
)
def test_qkv_attention_preserves_all_production_cleanup_forms(
    include_layout_transpose: bool,
    include_prefix: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    invocations = build_qkv_attention_invocations(
        context,
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
    )

    assert (
        active_qkv_attention_pass_ids(
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
        )
        == expected_ids
    )
    assert tuple(step.pass_id for step in invocations) == expected_ids
    expected_contract = (
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {pass_id: expected_contract for pass_id in expected_ids}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_qkv_attention_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize(
    ("include_layout_transpose", "include_prefix", "expected_ids"),
    PRODUCTION_FORMS,
)
def test_qkv_attention_runner_preserves_all_production_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    include_prefix: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for pass_id in expected_ids:
        monkeypatch.setattr(
            qkv_attention_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_qkv_attention(
        context,
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
    )

    assert events == list(expected_ids)


@pytest.mark.parametrize(
    ("include_layout_transpose", "include_prefix", "expected_ids"),
    PRODUCTION_FORMS,
)
def test_qkv_attention_returns_and_summarizes_all_production_forms(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    include_prefix: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    layout_result = {
        "iterations": 9,
        "removed_identity_transpose": 1,
        "removed_inverse_transpose_pairs": 2,
        "removed_inverse_transpose_fanout_branches": 3,
        "composed_consecutive_transpose_pairs": 4,
    }
    prefix_result = {
        "optimized_attention_qkv_gather_reshape_transpose_hoist_chains": 5,
        "optimized_attention_qkv_slice_replace_gather_reshape_chains": 6,
        "optimized_attention_qkv_slice_to_split_chains": 7,
        "optimized_attention_split_post_reshape_collapse_chains": 8,
    }
    bridge_result = {
        "optimized_attention_qkv_shared_pretranspose_slice_nchw_chains": 10,
        "optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains": 11,
    }
    expected_results = (
        *((layout_result,) if include_layout_transpose else ()),
        *((prefix_result,) if include_prefix else ()),
        bridge_result,
    )

    def return_results(invocations, *, expected_pass_ids, phase_name):
        assert tuple(invocation.pass_id for invocation in invocations) == expected_ids
        assert tuple(expected_pass_ids) == expected_ids
        assert phase_name == "QKV attention"
        return expected_results

    monkeypatch.setattr(
        qkv_attention_orchestration,
        "run_recovery_invocations",
        return_results,
    )

    results = run_qkv_attention(
        context,
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
    )
    summarize = getattr(
        qkv_attention_orchestration,
        "summarize_qkv_attention_mutations",
    )
    summary = summarize(
        results,
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
        pruned_unused_tensors=12,
    )

    assert results == expected_results
    assert summary == {
        "removed_identity_transpose": 1 if include_layout_transpose else 0,
        "removed_inverse_transpose_pairs": 2 if include_layout_transpose else 0,
        "removed_inverse_transpose_fanout_branches": (
            3 if include_layout_transpose else 0
        ),
        "composed_consecutive_transpose_pairs": (
            4 if include_layout_transpose else 0
        ),
        "optimized_attention_qkv_gather_reshape_transpose_hoist_chains": (
            5 if include_prefix else 0
        ),
        "optimized_attention_qkv_slice_replace_gather_reshape_chains": (
            6 if include_prefix else 0
        ),
        "optimized_attention_qkv_slice_to_split_chains": (
            7 if include_prefix else 0
        ),
        "optimized_attention_split_post_reshape_collapse_chains": (
            8 if include_prefix else 0
        ),
        "optimized_attention_qkv_shared_pretranspose_slice_nchw_chains": 10,
        "optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains": 11,
        "pruned_unused_tensors": 12,
    }
    assert "iterations" not in summary
    with pytest.raises(
        ValueError,
        match=r"QKV attention mutation summary expected [12] pass results",
    ):
        summarize(
            (),
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
            pruned_unused_tensors=0,
        )

    _, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert isinstance(helper.body[0].value, ast.Call)


def test_lowerer_captures_terminal_qkv_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    summary_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LOWERER_RESULT
    )
    summary = lowerer.body[summary_index]
    assert isinstance(summary, ast.Assign)
    summary_call = summary.value
    assert isinstance(summary_call, ast.Call)
    assert isinstance(summary_call.func, ast.Name)
    assert summary_call.func.id == LOWERER_OWNER
    assert [_expression_path(argument) for argument in summary_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: _expression_path(keyword.value)
        for keyword in summary_call.keywords
    } == {
        "include_layout_transpose": "optimize_layout_transpose_chains",
    }

    previous = lowerer.body[summary_index - 1]
    assert isinstance(previous, ast.If)
    assert ast.unparse(previous.test) == (
        "_late_binary_layout_recovery_requires_reconciliation"
    )
    _assert_phase_result_record(
        lowerer.body[summary_index + 1],
        "shape_reconciliation.terminal.expand_squeeze",
    )
    assert _call_name(lowerer.body[summary_index + 2]) == (
        "_advance_post_progress"
    )
    assert len(_top_calls()) == 1
    assert len(_outer_calls()) == 1
    owner_calls = _terminal_qkv_owner_calls("run_qkv_attention_summary")
    assert len(owner_calls) == 1
    assert [_expression_path(argument) for argument in owner_calls[0].args] == [
        "context"
    ]
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in owner_calls[0].keywords
    } == {
        "include_layout_transpose": "include_layout_transpose",
        "include_prefix": False,
    }


def test_qkv_attention_preserves_both_invocation_forms() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == QKV_ATTENTION
    ]

    assert len(invocations) == 1
    default_invocations = [call for call in invocations if call.keywords == []]
    assert len(default_invocations) == 1
    assert all(call.args == [] for call in default_invocations)

    (post_sinet_invocation,) = _post_sinet_owner_calls("run_qkv_attention")
    assert [
        _expression_path(argument) for argument in post_sinet_invocation.args
    ] == ["context"]
    assert post_sinet_invocation.keywords == []

    (late_invocation,) = _terminal_qkv_owner_calls(
        "run_qkv_attention_summary"
    )
    assert [_expression_path(argument) for argument in late_invocation.args] == [
        "context"
    ]
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in late_invocation.keywords
    } == {
        "include_layout_transpose": "include_layout_transpose",
        "include_prefix": False,
    }


def test_qkv_attention_preserves_both_default_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    layout_block = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == QKV_ATTENTION
            for node in ast.walk(statement)
        )
    )
    nested_index = next(
        index
        for index, statement in enumerate(layout_block.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_qkv_attention_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == QKV_ATTENTION
    )
    assert (
        _call_name(layout_block.body[nested_index - 1])
        == "_optimize_batchmatmul_transpose_input_to_adj_flags"
    )
    assert (
        _call_name(layout_block.body[nested_index + 1])
        == "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )

    post_sinet_record = next(
        statement
        for statement in lowerer.body
        if _phase_id(statement) == POST_SINET_PHASE_ID
    )
    top_level_index = lowerer.body.index(post_sinet_record)
    assert isinstance(post_sinet_record, ast.Expr)
    assert ast.unparse(post_sinet_record.value.args[1]) == (
        POST_SINET_OWNER_EXPRESSION
    )
    _assert_phase_result_record(
        lowerer.body[top_level_index - 1],
        "cleanup.post_sinet.batchmatmul_adj_flags",
    )
    _assert_phase_result_record(
        lowerer.body[top_level_index + 1],
        "cleanup.post_sinet.relu_split_conv_concat",
    )


def test_qkv_attention_retains_both_default_policy_results() -> None:
    lowerer, _ = _lowerer_and_helper()
    terminal_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == QKV_ATTENTION
            for node in ast.walk(statement)
        )
    )
    terminal_index = next(
        index
        for index, statement in enumerate(terminal_guard.body)
        if _call_name(statement) == QKV_ATTENTION
    )
    terminal = terminal_guard.body[terminal_index]
    assert isinstance(terminal, ast.Assign)
    assert len(terminal.targets) == 1
    assert isinstance(terminal.targets[0], ast.Name)
    assert terminal.targets[0].id == "_terminal_qkv_attention_results"
    assert isinstance(terminal.value, ast.Call)
    assert terminal.value.args == []
    assert terminal.value.keywords == []
    predecessor = terminal_guard.body[terminal_index - 1]
    _assert_phase_result_record(
        predecessor,
        "cleanup.terminal.batchmatmul_adj_flags",
    )
    assert _call_name(terminal_guard.body[terminal_index + 1]) == (
        "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )

    post_sinet = next(
        statement
        for statement in lowerer.body
        if _phase_id(statement) == POST_SINET_PHASE_ID
    )
    post_sinet_index = lowerer.body.index(post_sinet)
    assert isinstance(post_sinet, ast.Expr)
    assert ast.unparse(post_sinet.value.args[1]) == (
        POST_SINET_OWNER_EXPRESSION
    )
    predecessor = lowerer.body[post_sinet_index - 1]
    _assert_phase_result_record(
        predecessor,
        "cleanup.post_sinet.batchmatmul_adj_flags",
    )
    _assert_phase_result_record(
        lowerer.body[post_sinet_index + 1],
        "cleanup.post_sinet.relu_split_conv_concat",
    )

    all_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == QKV_ATTENTION
    ]
    assert len(all_calls) == 1
    post_sinet_calls = _post_sinet_owner_calls("run_qkv_attention")
    assert len(post_sinet_calls) == 1
    assert [
        _expression_path(argument) for argument in post_sinet_calls[0].args
    ] == ["context"]
    assert post_sinet_calls[0].keywords == []
    late_result = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LOWERER_RESULT
    )
    assert isinstance(late_result.value, ast.Call)
    assert isinstance(late_result.value.func, ast.Name)
    assert late_result.value.func.id == LOWERER_OWNER
    assert [ast.unparse(argument) for argument in late_result.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in late_result.value.keywords
    } == {
        "include_layout_transpose": "optimize_layout_transpose_chains",
    }
    assert len(_terminal_qkv_owner_calls("run_qkv_attention_summary")) == 1
    assert len(_top_calls()) == 1
    assert len(_outer_calls()) == 1


def test_qkv_attention_preserves_late_bridge_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    late_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LOWERER_RESULT
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == LOWERER_OWNER
    )

    previous_boundary = lowerer.body[late_index - 1]
    assert isinstance(previous_boundary, ast.If)
    assert ast.unparse(previous_boundary.test) == (
        "_late_binary_layout_recovery_requires_reconciliation"
    )
    _assert_phase_result_record(
        lowerer.body[late_index + 1],
        "shape_reconciliation.terminal.expand_squeeze",
    )
    assert _call_name(lowerer.body[late_index + 2]) == "_advance_post_progress"
    assert len(_terminal_qkv_owner_calls("run_qkv_attention_summary")) == 1
    assert len(_top_calls()) == 1
    assert len(_outer_calls()) == 1


def test_qkv_attention_context_is_explicit() -> None:
    lowerer, _ = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "qkv_attention_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_qkv_attention_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "qkv_attention_orchestration.py"
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
