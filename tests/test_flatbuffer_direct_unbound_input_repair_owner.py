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
    / "unbound_input_repair_orchestration.py"
)
WRAPPER = "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
OWNER = "repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
RAW_OWNER = "repair_unbound_nonconstant_inputs_with_layout_transpose"
LOWERER_RECONCILE_OWNER = "_reconcile_static_tensor_shapes"
PASS_RECONCILE_OWNER = "reconcile_static_tensor_shapes"
RESULT_KEY = "repaired_unbound_nonconstant_inputs_with_layout_transpose"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _ordered_calls(
    function: ast.FunctionDef,
    names: tuple[str, ...],
) -> list[ast.Call]:
    return sorted(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )


def _assert_owner_contract(
    owner: ast.FunctionDef,
    *,
    reconcile_owner: str,
) -> None:
    raw_calls = _ordered_calls(owner, (RAW_OWNER,))
    assert len(raw_calls) == 1
    assert [ast.unparse(argument) for argument in raw_calls[0].args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in raw_calls[0].keywords
    } == {"graph_index": "graph_index"}

    guard = next(statement for statement in owner.body if isinstance(statement, ast.If))
    assert ast.unparse(guard.test) == "result.repaired > 0"
    reconcile_calls = _ordered_calls(guard, (reconcile_owner,))
    assert len(reconcile_calls) == 1
    assert [ast.unparse(argument) for argument in reconcile_calls[0].args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in reconcile_calls[0].keywords
    } == {"graph_index": "result.graph_index"}

    terminal = owner.body[-1]
    assert isinstance(terminal, ast.Return)
    assert isinstance(terminal.value, ast.Dict)
    assert [ast.literal_eval(key) for key in terminal.value.keys] == [RESULT_KEY]
    assert ast.unparse(terminal.value.values[0]) == "int(result.repaired)"


def test_unbound_input_repair_lowerer_contract_is_fixed() -> None:
    functions = _functions(LOWERER_PATH)
    _assert_owner_contract(
        functions[WRAPPER],
        reconcile_owner=LOWERER_RECONCILE_OWNER,
    )

    calls = _ordered_calls(functions["lower_onnx_to_ir"], (WRAPPER,))
    assert len(calls) == 2
    assert [ast.unparse(call.args[0]) for call in calls] == [
        "model_ir",
        "fallback_ir",
    ]
    assert all(call.keywords == [] for call in calls)


@pytest.mark.xfail(
    strict=True,
    reason="unbound-input mapping/reconciliation still lives in the lowerer",
)
def test_unbound_input_repair_has_one_pass_module_owner() -> None:
    _assert_owner_contract(
        _functions(OWNER_PATH)[OWNER],
        reconcile_owner=PASS_RECONCILE_OWNER,
    )

    wrapper = _functions(LOWERER_PATH)[WRAPPER]
    assert len(wrapper.body) == 2
    assert isinstance(wrapper.body[0], ast.Expr)
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    assert ast.unparse(dispatch.value) == (
        f"{OWNER}(model_ir, graph_index=graph_index)"
    )

    calls = _ordered_calls(_functions(LOWERER_PATH)["lower_onnx_to_ir"], (WRAPPER,))
    assert len(calls) == 2
    assert [ast.unparse(call.args[0]) for call in calls] == [
        "model_ir",
        "fallback_ir",
    ]
