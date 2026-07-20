from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_affine_final_shape_terminal_convpool_orchestration,
)
from onnx2tf.tflite_builder.passes.convpool_output_passthrough_compat import (
    optimize_convpool_output_transpose_nhwc_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.late_affine_final_shape_terminal_orchestration import (
    run_late_affine_final_shape_terminal_cleanup,
)
from onnx2tf.tflite_builder.passes.late_final_shape_boundary_orchestration import (
    LateFinalShapeBoundaryContext,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
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
    / "late_affine_final_shape_terminal_convpool_orchestration.py"
)
OWNER = "run_late_affine_final_shape_terminal_convpool_cleanup"
CHILD_OWNERS = (
    "run_late_affine_final_shape_terminal_cleanup",
    "optimize_convpool_output_transpose_nhwc_passthrough_chains",
)
CURRENT_CONVPOOL_WRAPPER = (
    "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
)
RESULT_TARGETS = (
    "_late_affine_final_shape_terminal_results",
    "_terminal_convpool_output_passthrough_stats",
)
COMPOSITE_TARGET = "_late_affine_final_shape_terminal_convpool_results"
PREDECESSOR_PHASE_ID = "cleanup.late.ndhwc_cost_volume"
LAYOUT_GUARD = "optimize_layout_transpose_chains"
NO_LAYOUT_GUARD = "apply_safe_transpose_reduction_lite_on_no_layout_opt"
FUTURE_NO_LAYOUT_GUARD = (
    "not optimize_layout_transpose_chains and "
    "apply_safe_transpose_reduction_lite_on_no_layout_opt"
)
NO_LAYOUT_PHASE_ID = "layout.no_layout.safe_transpose_reduction"
NO_LAYOUT_TARGET = "_no_layout_fallback_affine_prepost_stats"
NO_LAYOUT_OWNER = "_optimize_transpose_mul_add_const_prepost_nhwc_chains"
CONTEXT_TARGET = "late_final_shape_boundary_context"


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


def _phase_id(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or ast.unparse(call.func) != "session.record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _context(name: str) -> LateFinalShapeBoundaryContext:
    model_ir = ModelIR(name)
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    terminal_context = TerminalSliceConcatRecoveryContext(
        pass_context=pass_context,
        channel_slice_pad_mul_cluster=lambda: (),
    )
    return LateFinalShapeBoundaryContext(
        pass_context=pass_context,
        terminal_slice_concat_context=terminal_context,
    )


def test_late_affine_final_shape_terminal_convpool_current_contract() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        CONTEXT_TARGET
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"optimize_layout_transpose_chains": LAYOUT_GUARD}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID

    no_layout_guard = lowerer.body[index + 1]
    assert isinstance(no_layout_guard, ast.If)
    assert ast.unparse(no_layout_guard.test) == FUTURE_NO_LAYOUT_GUARD
    assert _phase_id(no_layout_guard.body[0]) == NO_LAYOUT_PHASE_ID
    assert _single_target(no_layout_guard.body[1]) == NO_LAYOUT_TARGET
    assert _call_name(no_layout_guard.body[1]) == NO_LAYOUT_OWNER
    assert no_layout_guard.orelse == []
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("optimize_layout_transpose_chains", [False, True])
def test_late_affine_final_shape_terminal_convpool_child_schemas(
    optimize_layout_transpose_chains: bool,
) -> None:
    context = _context(
        "late_affine_final_shape_terminal_convpool_"
        f"{optimize_layout_transpose_chains}"
    )
    late_results = run_late_affine_final_shape_terminal_cleanup(
        context,
        include_elementwise_fanout=optimize_layout_transpose_chains,
    )
    assert tuple(type(result) for result in late_results) == (tuple, tuple)
    optional_type = (
        dict if optimize_layout_transpose_chains else type(None)
    )
    assert tuple(type(result) for result in late_results[0]) == (
        tuple,
        optional_type,
    )
    assert tuple(type(result) for result in late_results[1]) == (tuple, tuple)

    if optimize_layout_transpose_chains:
        convpool_results: dict[str, int] | None = (
            optimize_convpool_output_transpose_nhwc_passthrough_chains(
                context.pass_context.model_ir
            )
        )
        assert tuple(convpool_results) == (
            "optimized_convpool_output_transpose_nhwc_passthrough_chains",
        )
    else:
        convpool_results = None
    assert convpool_results is None or isinstance(convpool_results, dict)


def test_late_affine_final_shape_terminal_convpool_wrapper_is_retained() -> None:
    wrapper = _functions(LOWERER_PATH)[CURRENT_CONVPOOL_WRAPPER]
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    call = _call(wrapper.body[0])
    assert call is not None
    assert ast.unparse(call.func).endswith(
        "optimize_convpool_output_transpose_nhwc_passthrough_chains_pass"
    )
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []


def test_late_affine_final_shape_terminal_convpool_has_one_context_owner() -> None:
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
    } == {
        "include_elementwise_fanout": "optimize_layout_transpose_chains"
    }
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.pass_context.model_ir"
    ]
    assert calls[1].keywords == []
    convpool_guard = next(
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
    )
    assert calls[1] in list(ast.walk(convpool_guard))

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        CONTEXT_TARGET
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"optimize_layout_transpose_chains": LAYOUT_GUARD}
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID

    no_layout_guard = lowerer.body[index + 1]
    assert isinstance(no_layout_guard, ast.If)
    assert ast.unparse(no_layout_guard.test) == FUTURE_NO_LAYOUT_GUARD
    assert _phase_id(no_layout_guard.body[0]) == NO_LAYOUT_PHASE_ID
    assert _single_target(no_layout_guard.body[1]) == NO_LAYOUT_TARGET
    assert _call_name(no_layout_guard.body[1]) == NO_LAYOUT_OWNER
    assert no_layout_guard.orelse == []
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "lower_from_onnx2tf" in ast.unparse(node)
        for node in ast.parse(OWNER_PATH.read_text(encoding="utf-8")).body
    )


@pytest.mark.parametrize("optimize_layout_transpose_chains", [False, True])
def test_late_affine_final_shape_terminal_convpool_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
    optimize_layout_transpose_chains: bool,
) -> None:
    context = _context(
        "late_affine_final_shape_terminal_convpool_runtime_"
        f"{optimize_layout_transpose_chains}"
    )
    late_results = ({"late": 1}, {"terminal": 2})
    convpool_results = {"convpool": 3}
    observed: list[tuple[str, object, dict[str, object]]] = []

    def late(active_context: object, **options: object) -> object:
        observed.append((CHILD_OWNERS[0], active_context, options))
        return late_results

    def convpool(active_model_ir: object) -> object:
        observed.append((CHILD_OWNERS[1], active_model_ir, {}))
        return convpool_results

    monkeypatch.setattr(
        late_affine_final_shape_terminal_convpool_orchestration,
        CHILD_OWNERS[0],
        late,
    )
    monkeypatch.setattr(
        late_affine_final_shape_terminal_convpool_orchestration,
        CHILD_OWNERS[1],
        convpool,
    )

    actual = late_affine_final_shape_terminal_convpool_orchestration.run_late_affine_final_shape_terminal_convpool_cleanup(
        context,
        optimize_layout_transpose_chains=optimize_layout_transpose_chains,
    )
    assert actual[0] is late_results
    assert observed[0] == (
        CHILD_OWNERS[0],
        context,
        {
            "include_elementwise_fanout": (
                optimize_layout_transpose_chains
            )
        },
    )
    if optimize_layout_transpose_chains:
        assert actual[1] is convpool_results
        assert observed[1] == (
            CHILD_OWNERS[1],
            context.pass_context.model_ir,
            {},
        )
    else:
        assert actual[1] is None
        assert len(observed) == 1
