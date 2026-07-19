from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
INDEXED_SHAPE_CONVERGENCE = "_run_indexed_shape_convergence_cleanup"
FINAL_CONVERGENCE = "_run_indexed_final_shape_activation_convergence"
PHASE_ID = "shape_topology.terminal.indexed_convergence"


def _module_functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
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
        return call.args[1]
    return call


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _direct_invocations(function: ast.FunctionDef) -> list[ast.stmt]:
    return [
        statement
        for statement in function.body
        if _call_name(statement) == INDEXED_SHAPE_CONVERGENCE
    ]


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


def test_indexed_shape_convergence_result_schema_and_forms_are_explicit() -> None:
    functions = _module_functions()
    owner = functions[INDEXED_SHAPE_CONVERGENCE]
    result_return = next(
        statement
        for statement in owner.body
        if isinstance(statement, ast.Return)
    )
    assert isinstance(result_return.value, ast.Dict)
    assert [
        key.value
        for key in result_return.value.keys
        if isinstance(key, ast.Constant)
    ] == [
        "removed_dead_operators",
        "resolved_dynamic_reshape_shapes",
        "reconciled_static_tensor_shapes",
    ]

    lowerer_invocations = _direct_invocations(functions["lower_onnx_to_ir"])
    nested_invocations = _direct_invocations(functions[FINAL_CONVERGENCE])
    assert len(lowerer_invocations) == 1
    assert len(nested_invocations) == 1

    top_level_call = _statement_call(lowerer_invocations[0])
    assert top_level_call is not None
    assert [ast.unparse(argument) for argument in top_level_call.args] == [
        "model_ir",
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in top_level_call.keywords
    } == {"layout_state": "session.layout_state"}

    nested_statement = nested_invocations[0]
    assert _single_target(nested_statement) == "convergence_stats"
    nested_call = _statement_call(nested_statement)
    assert nested_call is not None
    assert [ast.unparse(argument) for argument in nested_call.args] == [
        "model_ir",
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in nested_call.keywords
    } == {
        "layout_state": "layout_state",
        "graph_index": "graph_index",
    }


def test_top_level_indexed_shape_convergence_uses_phase_result_store() -> None:
    functions = _module_functions()
    lowerer = functions["lower_onnx_to_ir"]
    records = [
        statement for statement in lowerer.body if _phase_id(statement) == PHASE_ID
    ]
    assert len(records) == 1
    record = records[0]
    index = lowerer.body.index(record)

    assert ast.unparse(record.value.args[1]) == (
        "_run_indexed_shape_convergence_cleanup("
        "model_ir, layout_state=session.layout_state)"
    )
    assert _single_target(lowerer.body[index - 1]) == (
        "_terminal_sinet_singleton_reshape_results"
    )
    assert _single_target(lowerer.body[index + 1]) == (
        "_very_late_sinet_recovery_tail_results"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_post_terminal_indexed_shape_convergence_stats"
        for node in ast.walk(lowerer)
    )

    nested_invocations = _direct_invocations(functions[FINAL_CONVERGENCE])
    assert len(nested_invocations) == 1
    assert _single_target(nested_invocations[0]) == "convergence_stats"
