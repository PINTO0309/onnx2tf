from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import unbound_input_repair_orchestration


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
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_input_affine_normalization_orchestration.py"
)
COMPOSITE_OWNER = "run_late_input_affine_normalization_cleanup"
WRAPPER = "_repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
OWNER = "repair_unbound_nonconstant_operator_inputs_with_layout_transpose"
RAW_OWNER = "repair_unbound_nonconstant_inputs_with_layout_transpose"
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


def _composite_call_count() -> int:
    return len(
        _ordered_calls(
            _functions(COMPOSITE_PATH)[COMPOSITE_OWNER],
            (OWNER,),
        )
    )


def test_unbound_input_repair_lowerer_contract_is_fixed() -> None:
    functions = _functions(LOWERER_PATH)
    _assert_owner_contract(
        _functions(OWNER_PATH)[OWNER],
        reconcile_owner=PASS_RECONCILE_OWNER,
    )

    calls = _ordered_calls(functions["lower_onnx_to_ir"], (WRAPPER,))
    assert len(calls) == 1
    assert [ast.unparse(call.args[0]) for call in calls] == [
        "fallback_ir",
    ]
    assert all(call.keywords == [] for call in calls)
    assert _composite_call_count() == 1


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
    assert len(calls) == 1
    assert [ast.unparse(call.args[0]) for call in calls] == [
        "fallback_ir",
    ]
    assert _composite_call_count() == 1


def test_unbound_input_repair_forwards_returned_index_to_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("unbound_input_repair_owner")
    initial_index = ModelIRGraphIndex(model_ir)
    repaired_index = ModelIRGraphIndex(model_ir)
    calls: list[tuple[str, object, object]] = []

    def repair(active_model_ir: object, *, graph_index: object) -> object:
        calls.append(("repair", active_model_ir, graph_index))
        return SimpleNamespace(repaired=2, graph_index=repaired_index)

    def reconcile(active_model_ir: object, *, graph_index: object) -> None:
        calls.append(("reconcile", active_model_ir, graph_index))

    monkeypatch.setattr(
        unbound_input_repair_orchestration,
        RAW_OWNER,
        repair,
    )
    monkeypatch.setattr(
        unbound_input_repair_orchestration,
        PASS_RECONCILE_OWNER,
        reconcile,
    )

    result = unbound_input_repair_orchestration.repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
        model_ir,
        graph_index=initial_index,
    )

    assert result == {RESULT_KEY: 2}
    assert calls == [
        ("repair", model_ir, initial_index),
        ("reconcile", model_ir, repaired_index),
    ]
