from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
LEGACY_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "nhwc_concat_legacy_layout.py"
)
PRE_CONCAT_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_concat_nhwc_layout.py"
)
PRE_CONCAT = "_optimize_transpose_pre_concat_nhwc_chains"
PRE_CONCAT_OWNER = "optimize_transpose_pre_concat_nhwc_chains"
LEGACY_OWNER = "optimize_transpose_pre_concat_nhwc_chains_legacy"
RESULT_TARGETS = (
    "_layout_opt_pre_concat_stats",
    "_final_pre_concat_stats",
    "_absolute_final_pre_concat_stats",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _direct_call(statement: ast.stmt) -> ast.Call | None:
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
    function = call.func
    if not isinstance(function, ast.Name) or function.id != PRE_CONCAT:
        return None
    return call


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _containing_body(root: ast.AST, target: ast.stmt) -> list[ast.stmt]:
    for node in ast.walk(root):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and target in value:
                return value
    raise AssertionError("statement is not contained by an AST body")


def _call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    value = statement.value
    if not isinstance(value, ast.Call):
        return None
    if (
        isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "session"
        and value.func.attr == "record_phase_result"
        and len(value.args) == 2
        and isinstance(value.args[1], ast.Call)
    ):
        value = value.args[1]
    if not isinstance(value.func, ast.Name):
        return None
    return value.func.id


def test_pre_concat_composite_schema_order_and_cleanup_are_explicit() -> None:
    composite_tree = ast.parse(
        PRE_CONCAT_OWNER_PATH.read_text(encoding="utf-8")
    )
    composite = _functions(PRE_CONCAT_OWNER_PATH)[PRE_CONCAT_OWNER]
    expected_dispatches = (
        "run_nhwc_concat_layout_cleanup",
        "run_nhwc_concat_quantized_layout_cleanup",
        LEGACY_OWNER,
    )
    dispatches = sorted(
        (
            node
            for node in ast.walk(composite)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in expected_dispatches
        ),
        key=lambda node: node.lineno,
    )
    assert tuple(node.func.id for node in dispatches) == expected_dispatches
    stats_key = next(
        node
        for node in composite_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_STATS_KEY"
    )
    assert ast.literal_eval(stats_key.value) == (
        "optimized_transpose_pre_concat_nhwc_chains"
    )
    composite_return = composite.body[-1]
    assert isinstance(composite_return, ast.Return)
    assert ast.unparse(composite_return.value) == (
        "{_STATS_KEY: int(optimized)}"
    )

    compatibility_wrapper = _functions(LOWERER_PATH)[PRE_CONCAT]
    assert len(compatibility_wrapper.body) == 1
    assert ast.unparse(compatibility_wrapper.body[0]) == (
        "return _optimize_transpose_pre_concat_nhwc_chains_pass(model_ir, "
        "layout_state=layout_state, diagnostics=diagnostics)"
    )

    legacy_owner = _functions(LEGACY_OWNER_PATH)[LEGACY_OWNER]
    assert sum(
        1
        for statement in legacy_owner.body
        if _call_name(statement) == "_prune_unused_tensors"
    ) == 1
    legacy_return = legacy_owner.body[-1]
    assert isinstance(legacy_return, ast.Return)
    assert ast.unparse(legacy_return.value) == (
        "{'optimized_transpose_pre_concat_nhwc_chains': int(optimized)}"
    )


def test_all_direct_pre_concat_results_are_retained_observation_only() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_results = sorted(
        (
            statement
            for statement in ast.walk(lowerer)
            if isinstance(statement, (ast.Assign, ast.Expr))
            and _direct_call(statement) is not None
        ),
        key=lambda statement: statement.lineno,
    )
    assert len(direct_results) == 2
    assert tuple(_single_target(statement) for statement in direct_results) == (
        None,
        RESULT_TARGETS[2],
    )
    for statement in direct_results:
        call = _direct_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
        }
    for target in RESULT_TARGETS[:2]:
        assert not any(
            isinstance(node, ast.Name) and node.id == target
            for node in ast.walk(lowerer)
        )
    for target in RESULT_TARGETS[2:]:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )

    expected_boundaries = (
        (
            "run_spp_layout_cleanup",
            None,
            "run_ndhwc_concat_layout_cleanup",
        ),
        (
            "run_late_hard_activation_layout_summary",
            "_late_hard_activation_stats",
            "_optimize_transpose_shape_extract_nhwc_to_nchw_chains",
        ),
    )
    observed_boundaries = []
    for statement in direct_results:
        body = _containing_body(lowerer, statement)
        index = body.index(statement)
        previous = body[index - 1]
        following = body[index + 1]
        observed_boundaries.append(
            (
                _call_name(previous),
                _single_target(previous),
                _call_name(following),
            )
        )
    assert tuple(observed_boundaries) == expected_boundaries


def test_layout_recovery_keeps_independent_pre_concat_callback_selection() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and _single_target(statement) == "layout_recovery_context"
    )
    assert isinstance(context_assignment.value, ast.Call)
    callback = next(
        keyword.value
        for keyword in context_assignment.value.keywords
        if keyword.arg == "pre_concat_cleanup"
    )
    assert isinstance(callback, ast.Name)
    assert callback.id == PRE_CONCAT
