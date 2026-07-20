from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.hardswish_se_layout import (
    optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes import (
    terminal_singleton_clamp_sinet_hardswish_orchestration,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes.terminal_singleton_clamp_sinet_orchestration import (
    run_terminal_singleton_clamp_sinet_cleanup,
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
    / "terminal_singleton_clamp_sinet_hardswish_orchestration.py"
)
OWNER = "run_terminal_singleton_clamp_sinet_hardswish_cleanup"
CHILD_OWNERS = (
    "run_terminal_singleton_clamp_sinet_cleanup",
    "optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains",
)
CURRENT_HARDSWISH_WRAPPER = (
    "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
)
CURRENT_RESULT_TARGET = "_terminal_singleton_clamp_sinet_results"
COMPOSITE_TARGET = "_terminal_singleton_clamp_sinet_hardswish_results"
CONTEXT_TARGET = "sinet_terminal_layout_recovery_context"
LAYOUT_GUARD = "optimize_layout_transpose_chains"
PREDECESSOR_PHASE_ID = "cleanup.terminal.qkv_split_conv_concat_bridge"
HARDSWISH_PHASE_ID = "cleanup.terminal.sinet_hardswish_se"
SUCCESSOR_PHASE_ID = "cleanup.terminal.dequant_hardsigmoid_bridge"
SUCCESSOR_RESULT_TARGET = "_terminal_sinet_singleton_reshape_results"

SINGLETON_SCHEMA = (
    (
        "iterations",
        "removed_identity_transpose",
        "removed_inverse_transpose_pairs",
        "removed_inverse_transpose_fanout_branches",
        "composed_consecutive_transpose_pairs",
    ),
    ("rewritten_singleton_channel_layout_transpose_to_reshape",),
    (
        "rewritten_singleton_layout_reshape_unary_passthrough_chains",
        "rewritten_consecutive_inverse_singleton_layout_reshapes",
    ),
    (
        "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains",
        "rewritten_singleton_nms_maxpool_nhwc_chains",
    ),
    ("rewritten_flatten_concat_expanddims_to_nhwc_concat",),
    (
        "removed_noop_reshape_chains",
        "rewritten_consecutive_reshape_passthrough_chains",
        "rewritten_fanout_bypass_reshape_passthrough_chains",
    ),
    ("optimized_squeeze_reshape_identity_chains",),
    (
        "optimized_singleton_spatial_nhwc_transpose_reshape_flatten",
        "rewritten_singleton_reshape_concat_post_transpose_nhwc_chains",
    ),
    ("optimized_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains",),
)
CLAMP_SINET_SCHEMA = (
    (
        ("rewritten_maximum_minimum_relu0to1_chains",),
        ("rewritten_transpose_unary_passthrough_chains",),
        ("rewritten_maximum_with_zero_input2_to_relu",),
    ),
    (
        ("optimized_sinet_shuffle_residual_transpose_chains",),
        (("preadd_resize",),),
        (
            "optimized_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains",
        ),
    ),
)
HARDSWISH_SCHEMA = (
    "optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr, ast.Return)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _phase_call(statement: ast.stmt) -> ast.Call | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or ast.unparse(call.func) != "session.record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return call


def _phase_id(statement: ast.stmt) -> str | None:
    call = _phase_call(statement)
    return None if call is None else ast.literal_eval(call.args[0])


def _context() -> tuple[
    SINetTerminalLayoutRecoveryContext,
    tuple[dict[str, int], ...],
]:
    model_ir = ModelIR("terminal_singleton_clamp_sinet_hardswish_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    preadd_results = ({"preadd_resize": 1},)
    return (
        SINetTerminalLayoutRecoveryContext(
            pass_context=pass_context,
            preadd_resize_recovery=lambda: preadd_results,
        ),
        preadd_results,
    )


def _schema(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(value)
    if isinstance(value, tuple):
        return tuple(_schema(child) for child in value)
    return value


def _predecessor_guard(lowerer: ast.FunctionDef) -> ast.If:
    return next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == LAYOUT_GUARD
        and any(
            _phase_id(child) == PREDECESSOR_PHASE_ID
            for child in statement.body
        )
    )


def test_terminal_singleton_clamp_sinet_hardswish_current_contract() -> None:
    lowerer = _lowerer()
    predecessor = _predecessor_guard(lowerer)
    predecessor_index = lowerer.body.index(predecessor)
    assignment = lowerer.body[predecessor_index + 1]
    assert _single_target(assignment) == COMPOSITE_TARGET
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        CONTEXT_TARGET
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_terminal_singleton": LAYOUT_GUARD}

    hardswish_record = lowerer.body[predecessor_index + 2]
    assert _phase_id(hardswish_record) == HARDSWISH_PHASE_ID
    hardswish_call = _phase_call(hardswish_record)
    assert hardswish_call is not None
    assert ast.unparse(hardswish_call.args[1]) == f"{COMPOSITE_TARGET}[1]"
    assert _phase_id(lowerer.body[predecessor_index + 3]) == (
        SUCCESSOR_PHASE_ID
    )
    assert _single_target(lowerer.body[predecessor_index + 4]) == (
        SUCCESSOR_RESULT_TARGET
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == CURRENT_RESULT_TARGET
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("include_terminal_singleton", [False, True])
def test_terminal_singleton_clamp_sinet_hardswish_child_schemas(
    include_terminal_singleton: bool,
) -> None:
    context, preadd_results = _context()
    terminal_results = run_terminal_singleton_clamp_sinet_cleanup(
        context,
        include_terminal_singleton=include_terminal_singleton,
    )
    hardswish_results = (
        optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            context.pass_context.model_ir
        )
    )
    assert terminal_results[0] is None or (
        _schema(terminal_results[0]) == SINGLETON_SCHEMA
    )
    assert (terminal_results[0] is not None) is include_terminal_singleton
    assert _schema(terminal_results[1]) == CLAMP_SINET_SCHEMA
    assert terminal_results[1][1][1] is preadd_results
    assert tuple(hardswish_results) == HARDSWISH_SCHEMA


def test_terminal_singleton_clamp_sinet_hardswish_wrapper_is_retained() -> None:
    wrapper = _functions(LOWERER_PATH)[CURRENT_HARDSWISH_WRAPPER]
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    call = _call(wrapper.body[0])
    assert call is not None
    assert ast.unparse(call.func).endswith(
        "optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains_pass"
    )
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []


def test_terminal_singleton_clamp_sinet_hardswish_has_one_context_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in CHILD_OWNERS
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [call.func.id for call in calls] == list(CHILD_OWNERS)
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
    } == {"include_terminal_singleton": "include_terminal_singleton"}
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.pass_context.model_ir"
    ]
    assert calls[1].keywords == []

    lowerer = _lowerer()
    predecessor = _predecessor_guard(lowerer)
    predecessor_index = lowerer.body.index(predecessor)
    assignment = lowerer.body[predecessor_index + 1]
    assert _single_target(assignment) == COMPOSITE_TARGET
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        CONTEXT_TARGET
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"include_terminal_singleton": LAYOUT_GUARD}

    hardswish_record = lowerer.body[predecessor_index + 2]
    assert _phase_id(hardswish_record) == HARDSWISH_PHASE_ID
    hardswish_call = _phase_call(hardswish_record)
    assert hardswish_call is not None
    assert ast.unparse(hardswish_call.args[1]) == f"{COMPOSITE_TARGET}[1]"
    assert _phase_id(lowerer.body[predecessor_index + 3]) == (
        SUCCESSOR_PHASE_ID
    )
    assert _single_target(lowerer.body[predecessor_index + 4]) == (
        SUCCESSOR_RESULT_TARGET
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == CURRENT_RESULT_TARGET
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CURRENT_HARDSWISH_WRAPPER
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


@pytest.mark.parametrize("include_terminal_singleton", [False, True])
def test_terminal_singleton_clamp_sinet_hardswish_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    include_terminal_singleton: bool,
) -> None:
    context, _ = _context()
    terminal_results = ({"terminal": 1}, {"sinet": 2})
    hardswish_results = {"hardswish": 3}
    observed: list[tuple[str, object, dict[str, object]]] = []

    def terminal(active_context: object, **options: object) -> object:
        observed.append((CHILD_OWNERS[0], active_context, options))
        return terminal_results

    def hardswish(active_model_ir: object) -> object:
        observed.append((CHILD_OWNERS[1], active_model_ir, {}))
        return hardswish_results

    monkeypatch.setattr(
        terminal_singleton_clamp_sinet_hardswish_orchestration,
        CHILD_OWNERS[0],
        terminal,
    )
    monkeypatch.setattr(
        terminal_singleton_clamp_sinet_hardswish_orchestration,
        CHILD_OWNERS[1],
        hardswish,
    )

    actual = terminal_singleton_clamp_sinet_hardswish_orchestration.run_terminal_singleton_clamp_sinet_hardswish_cleanup(
        context,
        include_terminal_singleton=include_terminal_singleton,
    )
    assert actual[0] is terminal_results
    assert actual[1] is hardswish_results
    assert observed == [
        (
            CHILD_OWNERS[0],
            context,
            {"include_terminal_singleton": include_terminal_singleton},
        ),
        (CHILD_OWNERS[1], context.pass_context.model_ir, {}),
    ]
