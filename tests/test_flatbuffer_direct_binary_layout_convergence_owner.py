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
    / "binary_layout_convergence.py"
)
WRAPPER = "_run_indexed_binary_layout_convergence"
OWNER = "run_indexed_binary_layout_convergence"
ROUND_OWNERS = (
    "_repair_rank4_channelwise_broadcast_constants_to_runtime_layout",
    "_repair_stale_nchw_to_nhwc_channelwise_binary_transposes",
    "_reconcile_static_tensor_shapes",
)
OWNER_ROUND_OWNERS = tuple(name.removeprefix("_") for name in ROUND_OWNERS)
RESULT_KEYS = (
    "repaired_rank4_channelwise_broadcast_constants",
    "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes",
    "reconciled_static_tensor_shapes",
)


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
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
    ]
    return sorted(calls, key=lambda node: (node.lineno, node.col_offset))


def _assert_owner_contract(
    owner: ast.FunctionDef,
    round_owners: tuple[str, ...],
) -> None:
    index_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRGraphIndex"
    ]
    assert len(index_calls) == 1
    assert [ast.unparse(argument) for argument in index_calls[0].args] == [
        "model_ir"
    ]

    loop = next(node for node in owner.body if isinstance(node, ast.For))
    assert ast.unparse(loop.iter) == "range(3)"
    calls = _ordered_calls(loop, round_owners)
    assert [call.func.id for call in calls] == list(round_owners)
    for call in calls:
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"graph_index": "graph_index"}

    terminal = owner.body[-1]
    assert isinstance(terminal, ast.Return)
    assert isinstance(terminal.value, ast.Dict)
    assert tuple(ast.literal_eval(key) for key in terminal.value.keys) == RESULT_KEYS


def test_binary_layout_convergence_lowerer_contract_is_fixed() -> None:
    functions = _functions(LOWERER_PATH)
    _assert_owner_contract(functions[WRAPPER], ROUND_OWNERS)

    lowerer = functions["lower_onnx_to_ir"]
    calls = _ordered_calls(lowerer, (WRAPPER,))
    assert len(calls) == 2
    assert [ast.unparse(call.args[0]) for call in calls] == [
        "fallback_ir",
        "model_ir",
    ]
    assert all(call.keywords == [] for call in calls)


@pytest.mark.xfail(
    strict=True,
    reason="indexed binary-layout convergence is still implemented in the lowerer",
)
def test_binary_layout_convergence_has_one_pass_module_owner() -> None:
    owner = _functions(OWNER_PATH)[OWNER]
    _assert_owner_contract(owner, OWNER_ROUND_OWNERS)

    wrapper = _functions(LOWERER_PATH)[WRAPPER]
    assert len(wrapper.body) == 2
    assert isinstance(wrapper.body[0], ast.Expr)
    dispatch = wrapper.body[1]
    assert isinstance(dispatch, ast.Return)
    assert ast.unparse(dispatch.value) == f"{OWNER}(model_ir)"

    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    calls = _ordered_calls(lowerer, (WRAPPER,))
    assert len(calls) == 2
    assert [ast.unparse(call.args[0]) for call in calls] == [
        "fallback_ir",
        "model_ir",
    ]
