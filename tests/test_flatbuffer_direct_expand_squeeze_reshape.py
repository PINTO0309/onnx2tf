from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _replace_expand_dims_and_squeeze_with_reshape,
)
from onnx2tf.tflite_builder.passes.expand_squeeze_reshape import (
    replace_expand_dims_and_squeeze_with_reshape,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


REPO_ROOT = Path(__file__).resolve().parents[1]
TERMINAL_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_layout_shape_orchestration.py"
)
TERMINAL_OWNER = "run_terminal_layout_shape_cleanup"


def _dynamic_squeeze_model() -> ModelIR:
    model_ir = ModelIR("dynamic_squeeze_to_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR(
            name="x",
            dtype="FLOAT32",
            shape=[1, 8, 1],
            shape_signature=[-1, 8, 1],
        ),
        "y": TensorIR(
            name="y",
            dtype="FLOAT32",
            shape=[1, 8],
            shape_signature=[-1, 8],
        ),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="SQUEEZE",
            inputs=["x"],
            outputs=["y"],
            options={"squeezeDims": [2]},
        )
    ]
    return model_ir


def _snapshot(model_ir: ModelIR) -> tuple:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                (
                    None
                    if tensor.shape_signature is None
                    else tuple(tensor.shape_signature)
                ),
                _freeze(tensor.data),
                tensor.logical_layout,
                tensor.physical_layout,
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                _freeze(operator.options),
            )
            for operator in model_ir.operators
        ),
    )


def test_dynamic_squeeze_inserts_runtime_shape_ops_in_order() -> None:
    model_ir = _dynamic_squeeze_model()
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = replace_expand_dims_and_squeeze_with_reshape(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {
        "replaced_expand_dims_and_squeeze_with_reshape": 1,
        "expand_dims_squeeze_rewrite_shape_tensors": 1,
    }
    assert [operator.op_type for operator in model_ir.operators] == [
        "SHAPE",
        "GATHER",
        "RESHAPE",
    ]
    reshape = model_ir.operators[-1]
    assert reshape.inputs[0] == "x"
    assert reshape.outputs == ["y"]
    assert reshape.options == {
        "newShape": [],
        "onnxSqueezeDims": [2],
        "preserveSemanticRank": True,
    }
    gather = model_ir.operators[1]
    kept_axes = model_ir.tensors[gather.inputs[1]]
    np.testing.assert_array_equal(
        kept_axes.data,
        np.asarray([0, 1], dtype=np.int32),
    )
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_compatibility_wrapper_matches_owner_and_result_is_idempotent() -> None:
    direct_model = _dynamic_squeeze_model()
    wrapper_model = copy.deepcopy(direct_model)
    direct_layout = LayoutState.from_model_ir(direct_model)
    wrapper_layout = LayoutState.from_model_ir(wrapper_model)

    direct_stats = replace_expand_dims_and_squeeze_with_reshape(
        direct_model,
        layout_state=direct_layout,
    )
    wrapper_stats = _replace_expand_dims_and_squeeze_with_reshape(
        wrapper_model,
        layout_state=wrapper_layout,
    )

    assert wrapper_stats == direct_stats
    assert _snapshot(wrapper_model) == _snapshot(direct_model)
    before = _snapshot(direct_model)
    assert replace_expand_dims_and_squeeze_with_reshape(direct_model) == {
        "replaced_expand_dims_and_squeeze_with_reshape": 0,
        "expand_dims_squeeze_rewrite_shape_tensors": 0,
    }
    assert _snapshot(direct_model) == before


def test_terminal_expand_squeeze_result_is_captured_before_reconciliation() -> None:
    lowerer_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    assignment_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_layout_shape_results"
    )
    assignment = lowerer.body[assignment_index]
    assert isinstance(assignment, ast.Assign)
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == TERMINAL_OWNER
    following = lowerer.body[assignment_index + 1]
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following) == (
        "session.record_phase_result("
        "'shape_reconciliation.terminal.expand_squeeze', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )

    owner_tree = ast.parse(TERMINAL_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == TERMINAL_OWNER
    )
    calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "replace_expand_dims_and_squeeze_with_reshape"
    ]
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
    } == {"layout_state": "context.layout_state"}
