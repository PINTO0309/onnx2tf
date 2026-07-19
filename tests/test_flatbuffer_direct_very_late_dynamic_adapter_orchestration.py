from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_dynamic_adapter_orchestration.py"
)
OWNER = "run_very_late_dynamic_adapter_cleanup"
RESULT_NAME = "_very_late_dynamic_adapter_results"
RAW_CONTRACTS = (
    (
        "_very_late_dynamic_reshape_stats",
        "_resolve_dynamic_reshape_shapes",
        ("model_ir",),
        {"prefer_runtime_inferable_from_onnx_raw": "True"},
    ),
    (
        "_very_late_conv_input_stats",
        "run_indexed_conv_input_adapter_repairs_summary",
        ("model_ir",),
        {},
    ),
    (
        "_very_late_stale_channel_shuffle_stats",
        "run_stale_nchw_channel_shuffle_repair",
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
        },
    ),
    (
        "_very_late_concat_transpose_conv_axis_stats",
        "_repair_nchw_concat_transpose_conv_axes",
        ("model_ir",),
        {"layout_state": "session.layout_state"},
    ),
    (
        "_very_late_concat_global_pool_conv_axis_stats",
        "_repair_nchw_concat_global_pool_conv_axes",
        ("model_ir",),
        {"layout_state": "session.layout_state"},
    ),
    (
        "_very_late_dynamic_rank1_reshape_stats",
        "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs",
        ("model_ir",),
        {"layout_state": "session.layout_state"},
    ),
)
OWNER_CONTRACTS = (
    (
        "resolve_dynamic_reshape_shapes",
        ("context.model_ir",),
        {"prefer_runtime_inferable_from_onnx_raw": "True"},
    ),
    (
        "run_indexed_conv_input_adapter_repairs_summary",
        ("context.model_ir",),
        {},
    ),
    (
        "run_stale_nchw_channel_shuffle_repair",
        ("context.model_ir",),
        {
            "layout_state": "context.layout_state",
            "diagnostics": "context.diagnostics",
        },
    ),
    (
        "repair_nchw_concat_transpose_conv_axes",
        ("context.model_ir",),
        {"layout_state": "context.layout_state"},
    ),
    (
        "repair_nchw_concat_global_pool_conv_axes",
        ("context.model_ir",),
        {"layout_state": "context.layout_state"},
    ),
    (
        "rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs",
        ("context.model_ir",),
        {"layout_state": "context.layout_state"},
    ),
)


def _function(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _lowerer() -> ast.FunctionDef:
    return _function(LOWERER_PATH, "lower_onnx_to_ir")


def _assignment_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _assert_call(
    expression: ast.expr,
    *,
    name: str,
    arguments: tuple[str, ...],
    keywords: dict[str, str],
) -> None:
    assert isinstance(expression, ast.Call)
    assert isinstance(expression.func, ast.Name)
    assert expression.func.id == name
    assert tuple(
        ast.unparse(argument) for argument in expression.args
    ) == arguments
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in expression.keywords
    } == keywords


def _assert_reconciliation_successor(
    body: list[ast.stmt],
    index: int,
) -> None:
    reconciliation = body[index]
    assert isinstance(reconciliation, ast.Expr)
    assert ast.unparse(reconciliation) == (
        "session.record_phase_result("
        "'shape_reconciliation.primary.very_late_final', "
        "_reconcile_static_tensor_shapes("
        "model_ir, include_mutation_count=True))"
    )
    split = body[index + 1]
    assert isinstance(split, ast.Assign)
    assert _assignment_name(split) == "split_fallback_stats"
    assert isinstance(split.value, ast.Call)
    assert isinstance(split.value.func, ast.Name)
    assert split.value.func.id == "_replace_unsupported_split_with_slice"


def test_very_late_dynamic_adapter_raw_lowerer_contract_is_fixed() -> None:
    body = _lowerer().body
    indices = []
    for result_name, owner_name, arguments, keywords in RAW_CONTRACTS:
        matches = [
            (index, statement)
            for index, statement in enumerate(body)
            if _assignment_name(statement) == result_name
        ]
        assert len(matches) == 1
        index, statement = matches[0]
        assert isinstance(statement, ast.Assign)
        _assert_call(
            statement.value,
            name=owner_name,
            arguments=arguments,
            keywords=keywords,
        )
        indices.append(index)

    assert indices == list(range(indices[0], indices[0] + len(indices)))
    _assert_reconciliation_successor(body, indices[-1] + 1)


@pytest.mark.xfail(
    strict=True,
    reason="very-late dynamic/adapter cleanup still has six lowerer results",
)
def test_very_late_dynamic_adapter_has_one_context_owner() -> None:
    owner = _function(OWNER_PATH, OWNER)
    assert [argument.arg for argument in owner.args.args] == ["context"]
    assert len(owner.body) == 1
    terminal = owner.body[0]
    assert isinstance(terminal, ast.Return)
    assert isinstance(terminal.value, ast.Tuple)
    assert len(terminal.value.elts) == len(OWNER_CONTRACTS)
    for expression, (name, arguments, keywords) in zip(
        terminal.value.elts,
        OWNER_CONTRACTS,
        strict=True,
    ):
        _assert_call(
            expression,
            name=name,
            arguments=arguments,
            keywords=keywords,
        )

    body = _lowerer().body
    matches = [
        (index, statement)
        for index, statement in enumerate(body)
        if _assignment_name(statement) == RESULT_NAME
    ]
    assert len(matches) == 1
    index, statement = matches[0]
    assert isinstance(statement, ast.Assign)
    _assert_call(
        statement.value,
        name=OWNER,
        arguments=("shared_model_ir_pass_context",),
        keywords={},
    )
    _assert_reconciliation_successor(body, index + 1)
