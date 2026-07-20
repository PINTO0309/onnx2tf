from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    very_late_dynamic_adapter_orchestration as owner_module,
)

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
FINAL_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_input_dynamic_orchestration.py"
)
FINAL_OWNER = "run_final_input_dynamic_cleanup"
FINAL_RESULT_NAME = "_final_input_dynamic_results"
FINAL_SHAPE_PHASE_ID = "shape_reconciliation.primary.very_late_final"
FINAL_SHAPE_OWNER_EXPRESSION = (
    "run_final_input_dynamic_shape_cleanup("
    "shared_model_ir_pass_context, "
    "shape_reconciler=_reconcile_static_tensor_shapes)[1]"
)
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


def _final_shape_record(
    body: list[ast.stmt],
) -> tuple[int, ast.Expr]:
    return next(
        (index, statement)
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and ast.unparse(statement.value.func) == "session.record_phase_result"
        and len(statement.value.args) == 2
        and ast.literal_eval(statement.value.args[0]) == FINAL_SHAPE_PHASE_ID
    )


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
        "run_final_input_dynamic_shape_cleanup("
        "shared_model_ir_pass_context, "
        "shape_reconciler=_reconcile_static_tensor_shapes)[1])"
    )
    split = body[index + 1]
    assert isinstance(split, ast.Assign)
    assert _assignment_name(split) == "split_fallback_stats"
    assert isinstance(split.value, ast.Call)
    assert isinstance(split.value.func, ast.Name)
    assert split.value.func.id == "_replace_unsupported_split_with_slice"


def _final_dynamic_child_call() -> ast.Call:
    owner = _function(FINAL_OWNER_PATH, FINAL_OWNER)
    return next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
    )


def test_very_late_dynamic_adapter_context_owner_preserves_raw_contracts() -> None:
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
    index, _ = _final_shape_record(body)
    assert ast.unparse(body[index - 1]) == "_advance_post_progress()"
    child_call = _final_dynamic_child_call()
    _assert_call(
        child_call,
        name=OWNER,
        arguments=("context",),
        keywords={},
    )
    _assert_reconciliation_successor(body, index)
    assert not any(
        _assignment_name(statement) == FINAL_RESULT_NAME for statement in body
    )

    assert not any(
        _assignment_name(statement) in {
            result_name for result_name, *_ in RAW_CONTRACTS
        }
        for statement in body
    )


def test_very_late_dynamic_adapter_lowerer_uses_one_composite_phase_owner() -> None:
    body = _lowerer().body
    index, record = _final_shape_record(body)
    assert ast.unparse(record.value.args[1]) == FINAL_SHAPE_OWNER_EXPRESSION
    assert ast.unparse(body[index - 1]) == "_advance_post_progress()"
    child_call = _final_dynamic_child_call()
    _assert_call(
        child_call,
        name=OWNER,
        arguments=("context",),
        keywords={},
    )
    _assert_reconciliation_successor(body, index)
    assert not any(
        _assignment_name(statement) == FINAL_RESULT_NAME for statement in body
    )


def test_very_late_dynamic_adapter_runtime_preserves_identity_order_and_tuple(
    monkeypatch,
) -> None:
    model_ir = ModelIR("very_late_dynamic_adapter")
    diagnostics: list[dict[str, object]] = []
    context = owner_module.VeryLateDynamicAdapterContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=diagnostics,
    )
    expected = tuple({f"result_{index}": index} for index in range(6))
    calls: list[str] = []

    def dynamic_reshape(
        received_model_ir,
        *,
        prefer_runtime_inferable_from_onnx_raw,
    ):
        assert received_model_ir is context.model_ir
        assert prefer_runtime_inferable_from_onnx_raw is True
        calls.append("dynamic_reshape")
        return expected[0]

    def conv_input(received_model_ir):
        assert received_model_ir is context.model_ir
        calls.append("conv_input")
        return expected[1]

    def channel_shuffle(
        received_model_ir,
        *,
        layout_state,
        diagnostics,
    ):
        assert received_model_ir is context.model_ir
        assert layout_state is context.layout_state
        assert diagnostics is context.diagnostics
        calls.append("channel_shuffle")
        return expected[2]

    def concat_transpose_conv(received_model_ir, *, layout_state):
        assert received_model_ir is context.model_ir
        assert layout_state is context.layout_state
        calls.append("concat_transpose_conv")
        return expected[3]

    def concat_global_pool(received_model_ir, *, layout_state):
        assert received_model_ir is context.model_ir
        assert layout_state is context.layout_state
        calls.append("concat_global_pool")
        return expected[4]

    def dynamic_rank1(received_model_ir, *, layout_state):
        assert received_model_ir is context.model_ir
        assert layout_state is context.layout_state
        calls.append("dynamic_rank1")
        return expected[5]

    monkeypatch.setattr(
        owner_module,
        "resolve_dynamic_reshape_shapes",
        dynamic_reshape,
    )
    monkeypatch.setattr(
        owner_module,
        "run_indexed_conv_input_adapter_repairs_summary",
        conv_input,
    )
    monkeypatch.setattr(
        owner_module,
        "run_stale_nchw_channel_shuffle_repair",
        channel_shuffle,
    )
    monkeypatch.setattr(
        owner_module,
        "repair_nchw_concat_transpose_conv_axes",
        concat_transpose_conv,
    )
    monkeypatch.setattr(
        owner_module,
        "repair_nchw_concat_global_pool_conv_axes",
        concat_global_pool,
    )
    monkeypatch.setattr(
        owner_module,
        "rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs",
        dynamic_rank1,
    )

    result = owner_module.run_very_late_dynamic_adapter_cleanup(context)

    assert calls == [
        "dynamic_reshape",
        "conv_input",
        "channel_shuffle",
        "concat_transpose_conv",
        "concat_global_pool",
        "dynamic_rank1",
    ]
    assert result == expected
    assert all(actual is wanted for actual, wanted in zip(result, expected))
