from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.cost_volume_scatter_layout import (
    run_cost_volume_scatter_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.gate_layout_orchestration import (
    GATE_LAYOUT_PASS_IDS,
    GATE_LAYOUT_REQUIRED_PASS_IDS,
    build_gate_layout_invocations,
)
from onnx2tf.tflite_builder.passes.ndhwc_gate_layout import (
    run_ndhwc_gate_layout_cleanup,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
NDHWC_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "ndhwc_gate_layout.py"
)
COST_VOLUME_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "cost_volume_scatter_layout.py"
)
NDHWC_OWNER = "run_ndhwc_gate_layout_cleanup"
COST_VOLUME_OWNER = "run_cost_volume_scatter_layout_cleanup"
NDHWC_TARGET = "_late_ndhwc_gate_layout_stats"
COST_VOLUME_TARGET = "_late_cost_volume_scatter_layout_stats"
SCOPE_TARGET = "late_ndhwc_cost_volume_state_scope"
PREDECESSOR_TARGET = "_post_sinet_dequant_hardsigmoid_bridge_stats"
SUCCESSOR_TARGET = "_late_cost_volume_conv_affine_stats"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _scope_location() -> tuple[ast.FunctionDef, int]:
    lowerer = _lowerer()
    return lowerer, next(
        index
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == SCOPE_TARGET
    )


def _context(name: str) -> ModelIRPassContext:
    model_ir = ModelIR(name)
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _transactional_pass_ids(owner: ast.FunctionDef) -> tuple[str, ...]:
    pass_specs = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PassSpec"
    ]
    assert all(
        any(
            keyword.arg == "transactional"
            and ast.unparse(keyword.value) == "True"
            for keyword in pass_spec.keywords
        )
        for pass_spec in pass_specs
    )
    pass_id_values = [
        next(
            keyword.value
            for keyword in pass_spec.keywords
            if keyword.arg == "pass_id"
        )
        for pass_spec in pass_specs
    ]
    if all(isinstance(value, ast.Constant) for value in pass_id_values):
        return tuple(str(ast.literal_eval(value)) for value in pass_id_values)
    assert len(pass_id_values) == 1
    assert isinstance(pass_id_values[0], ast.Name)
    assert pass_id_values[0].id == "pass_id"
    callbacks = next(
        statement.value
        for statement in owner.body
        if isinstance(statement, ast.Assign)
        and _single_target(statement) == "callbacks"
    )
    assert isinstance(callbacks, ast.List)
    assert all(isinstance(element, ast.Tuple) for element in callbacks.elts)
    return tuple(
        str(ast.literal_eval(element.elts[0]))
        for element in callbacks.elts
        if isinstance(element, ast.Tuple)
    )


def test_late_ndhwc_cost_volume_schemas_and_routes_are_explicit() -> None:
    ndhwc_functions = _functions(NDHWC_OWNER_PATH)
    cost_functions = _functions(COST_VOLUME_OWNER_PATH)
    assert _transactional_pass_ids(ndhwc_functions[NDHWC_OWNER]) == (
        "layout.ndhwc_leaky_logistic_gate",
        "layout.ndhwc_conv3d_leaky_unsqueeze_gate",
    )
    assert _transactional_pass_ids(cost_functions[COST_VOLUME_OWNER]) == (
        "layout.cost_volume_scatter_ndhwc",
    )
    for owner_name in (
        "_optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains",
        "_optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains",
    ):
        assert any(
            _call_name(statement) == "_prune_unused_tensors"
            for statement in ndhwc_functions[owner_name].body
        )
    assert any(
        _call_name(statement) == "_prune_unused_tensors"
        for statement in cost_functions[
            "_optimize_transpose_cost_volume_scatter_ndhwc_chains"
        ].body
    )

    context = _context("late_ndhwc_cost_volume_schema")
    scope = ModelIRPassStateScope(
        context.model_ir,
        layout_state=context.layout_state,
    )
    assert run_ndhwc_gate_layout_cleanup(
        context.model_ir,
        layout_state=context.layout_state,
        diagnostics=context.diagnostics,
        state_scope=scope,
    ) == {
        "optimized_transpose_3d_leaky_logistic_muladd_ndhwc_chains": 0,
        "optimized_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains": 0,
    }
    assert run_cost_volume_scatter_layout_cleanup(
        context.model_ir,
        layout_state=context.layout_state,
        diagnostics=context.diagnostics,
        state_scope=scope,
    ) == {"optimized_transpose_cost_volume_scatter_ndhwc_chains": 0}

    required = build_gate_layout_invocations(
        context,
        include_mixed_attention=False,
    )
    full = build_gate_layout_invocations(
        context,
        include_mixed_attention=True,
    )
    assert GATE_LAYOUT_REQUIRED_PASS_IDS[3:5] == (
        NDHWC_OWNER,
        COST_VOLUME_OWNER,
    )
    assert GATE_LAYOUT_PASS_IDS[4:6] == (NDHWC_OWNER, COST_VOLUME_OWNER)
    assert [invocation.callback for invocation in required[3:5]] == [
        run_ndhwc_gate_layout_cleanup,
        run_cost_volume_scatter_layout_cleanup,
    ]
    assert [invocation.callback for invocation in full[4:6]] == [
        run_ndhwc_gate_layout_cleanup,
        run_cost_volume_scatter_layout_cleanup,
    ]
    required_scopes = [
        dict(invocation.keyword_args)["state_scope"]
        for invocation in required[3:5]
    ]
    assert required_scopes[0] is required_scopes[1]


def test_late_ndhwc_cost_volume_direct_pair_boundary_is_explicit() -> None:
    lowerer, scope_index = _scope_location()
    ndhwc = lowerer.body[scope_index + 1]
    cost_volume = lowerer.body[scope_index + 2]
    assert isinstance(ndhwc, ast.Expr)
    assert isinstance(cost_volume, ast.Expr)
    assert _call_name(ndhwc) == NDHWC_OWNER
    assert _call_name(cost_volume) == COST_VOLUME_OWNER
    for statement in (ndhwc, cost_volume):
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": SCOPE_TARGET,
        }
    assert _single_target(lowerer.body[scope_index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[scope_index + 3]) == SUCCESSOR_TARGET
    for owner in (NDHWC_OWNER, COST_VOLUME_OWNER):
        assert sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == owner
            for node in ast.walk(lowerer)
        ) == 1


@pytest.mark.xfail(
    strict=True,
    reason="late NDHWC/cost-volume direct results are discarded",
)
def test_late_ndhwc_cost_volume_direct_results_are_retained_for_observation() -> None:
    lowerer, scope_index = _scope_location()
    assert _single_target(lowerer.body[scope_index + 1]) == NDHWC_TARGET
    assert _single_target(lowerer.body[scope_index + 2]) == COST_VOLUME_TARGET
    for target in (NDHWC_TARGET, COST_VOLUME_TARGET):
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )
