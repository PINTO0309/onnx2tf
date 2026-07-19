from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import recurrent_alias_repair_orchestration


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "recurrent_alias_repair_orchestration.py"
)
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_input_affine_normalization_orchestration.py"
)
COMPOSITE_OWNER = "run_late_input_affine_normalization_cleanup"
WRAPPER = "_repair_orphan_recurrent_step_tensors"
OWNER = "repair_orphan_recurrent_step_tensors_summary"
RAW_OWNER = "repair_orphan_recurrent_step_tensors"
RESULT_KEY = "repaired_orphan_recurrent_step_tensors"


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


def _assert_mapping_contract(function: ast.FunctionDef) -> None:
    raw_calls = _ordered_calls(function, (RAW_OWNER,))
    assert len(raw_calls) == 1
    assert [ast.unparse(argument) for argument in raw_calls[0].args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in raw_calls[0].keywords
    } == {"graph_index": "graph_index"}

    terminal = function.body[-1]
    assert isinstance(terminal, ast.Return)
    assert isinstance(terminal.value, ast.Dict)
    assert [ast.literal_eval(key) for key in terminal.value.keys] == [RESULT_KEY]
    assert ast.unparse(terminal.value.values[0]) == "int(repaired)"


def _composite_call_count() -> int:
    return len(
        _ordered_calls(
            _functions(COMPOSITE_PATH)[COMPOSITE_OWNER],
            (OWNER,),
        )
    )


def test_recurrent_alias_repair_lowerer_mapping_contract_is_fixed() -> None:
    functions = _functions(LOWERER_PATH)
    _assert_mapping_contract(_functions(OWNER_PATH)[OWNER])

    calls = _ordered_calls(functions["lower_onnx_to_ir"], (WRAPPER,))
    assert calls == []
    assert _composite_call_count() == 1


def test_recurrent_alias_repair_has_one_pass_module_mapping_owner() -> None:
    _assert_mapping_contract(_functions(OWNER_PATH)[OWNER])

    wrapper = _functions(LOWERER_PATH)[WRAPPER]
    assert len(wrapper.body) == 2
    assert isinstance(wrapper.body[0], ast.Expr)
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    assert ast.unparse(dispatch.value) == (
        f"{OWNER}(model_ir, graph_index=graph_index)"
    )

    calls = _ordered_calls(_functions(LOWERER_PATH)["lower_onnx_to_ir"], (WRAPPER,))
    assert calls == []
    assert _composite_call_count() == 1


def test_recurrent_alias_repair_owner_preserves_index_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("recurrent_alias_repair_owner")
    graph_index = ModelIRGraphIndex(model_ir)
    calls: list[tuple[ModelIR, ModelIRGraphIndex | None]] = []

    def repair(
        active_model_ir: ModelIR,
        *,
        graph_index: ModelIRGraphIndex | None = None,
    ) -> int:
        calls.append((active_model_ir, graph_index))
        return 3

    monkeypatch.setattr(
        recurrent_alias_repair_orchestration,
        RAW_OWNER,
        repair,
    )

    result = recurrent_alias_repair_orchestration.repair_orphan_recurrent_step_tensors_summary(
        model_ir,
        graph_index=graph_index,
    )

    assert result == {RESULT_KEY: 3}
    assert calls == [(model_ir, graph_index)]
