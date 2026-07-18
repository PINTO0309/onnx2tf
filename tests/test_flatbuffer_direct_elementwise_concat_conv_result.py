from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes.elementwise_concat_layout import (
    optimize_transpose_elementwise_concat_conv_nhwc_groups,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "elementwise_concat_layout.py"
)
ORCHESTRATION_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "layout_recovery_orchestration.py"
)
ELEMENTWISE_CONCAT_CONV = (
    "_optimize_transpose_elementwise_concat_conv_nhwc_groups"
)
OWNER_NAME = "optimize_transpose_elementwise_concat_conv_nhwc_groups"
RESULT_TARGET = "_layout_opt_elementwise_concat_conv_stats"
PREVIOUS_TARGET = "_layout_pass_set_2_quantized_activation_binary_results"
FOLLOWING_TARGET = "_layout_opt_spp_stats"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
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


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
    )


def test_elementwise_concat_conv_schema_cleanup_and_selection_are_explicit() -> None:
    wrapper = _functions(LOWERER_PATH)[ELEMENTWISE_CONCAT_CONV]
    assert len(wrapper.body) == 1
    wrapper_return = wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{ELEMENTWISE_CONCAT_CONV}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in wrapper_return.value.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
        "max_rewrites": "max_rewrites",
        "candidate": "candidate",
    }

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    assert sum(
        1
        for statement in owner.body
        if _call_name(statement) == "_prune_unused_tensors"
    ) == 1
    owner_return = owner.body[-1]
    assert isinstance(owner_return, ast.Return)
    assert ast.unparse(owner_return.value) == "{_STATS_KEY: int(rewritten)}"

    orchestration = _functions(ORCHESTRATION_PATH)[
        "build_layout_recovery_invocations"
    ]
    assert sum(
        1
        for node in ast.walk(orchestration)
        if isinstance(node, ast.Name) and node.id == OWNER_NAME
    ) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id == ELEMENTWISE_CONCAT_CONV
        for node in ast.walk(orchestration)
    )


def test_elementwise_concat_conv_zero_counter_can_prune_unused_tensor() -> None:
    model_ir = ModelIR("elementwise_concat_conv_zero_prune")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["x"]
    model_ir.tensors["x"] = _tensor("x", [1, 2, 3, 4])
    model_ir.tensors["unused"] = _tensor("unused", [1])

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(
        model_ir,
        max_rewrites=0,
    )

    assert stats == {
        "optimized_transpose_elementwise_concat_conv_nhwc_groups": 0,
    }
    assert "unused" not in model_ir.tensors


def test_direct_elementwise_concat_conv_result_is_retained_observation_only() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    layout_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(
            _call_name(child) == ELEMENTWISE_CONCAT_CONV
            for child in statement.body
        )
    )
    result_index = next(
        index
        for index, statement in enumerate(layout_guard.body)
        if _call_name(statement) == ELEMENTWISE_CONCAT_CONV
    )
    result = layout_guard.body[result_index]
    assert _single_target(result) == RESULT_TARGET
    call = _statement_call(result)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
    } == {"layout_state": "session.layout_state"}
    assert sum(
        1
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == ELEMENTWISE_CONCAT_CONV
    ) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )

    assert _single_target(layout_guard.body[result_index - 1]) == PREVIOUS_TARGET
    assert _single_target(layout_guard.body[result_index + 1]) == FOLLOWING_TARGET
