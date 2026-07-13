from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.precision import (
    _restore_precision_sensitive_reciprocal_divisions,
    _rewrite_constant_divisors_to_multiplicative_reciprocals,
)


def _tensor(
    name: str,
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=[1],
        shape_signature=[1],
        data=data,
    )


def test_constant_div_rewrite_updates_one_index_and_layout_state(monkeypatch) -> None:
    model_ir = ModelIR("constant_div_rewrite")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "rhs": _tensor("rhs", data=np.asarray([2.0], dtype=np.float32)),
        "y": _tensor("y"),
        "z": _tensor("z"),
    }
    model_ir.operators = [
        OperatorIR("DIV", ["x", "rhs"], ["y"]),
        OperatorIR("RELU", ["y"], ["z"]),
    ]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _rewrite_constant_divisors_to_multiplicative_reciprocals(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"rewritten_constant_div_to_mul": 1}
    assert refresh_count == 1
    assert [operator.op_type for operator in model_ir.operators] == ["MUL", "RELU"]
    assert graph_index.operator_indices("DIV") == []
    assert graph_index.operator_indices("MUL") == [0]
    assert graph_index.producer("y") is model_ir.operators[0]
    assert graph_index.consumer_indices("y") == [1]
    reciprocal_name = str(model_ir.operators[0].inputs[1])
    np.testing.assert_array_equal(
        model_ir.tensors[reciprocal_name].data,
        np.asarray([0.5], dtype=np.float32),
    )
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_constant_div_rewrite_preserves_integer_cast_path() -> None:
    model_ir = ModelIR("constant_div_integer_cast")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "rhs": _tensor("rhs", data=np.asarray([2.0], dtype=np.float32)),
        "y": _tensor("y"),
        "z": _tensor("z", dtype="INT32"),
    }
    model_ir.operators = [
        OperatorIR("DIV", ["x", "rhs"], ["y"]),
        OperatorIR(
            "CAST",
            ["y"],
            ["z"],
            options={"inDataType": "FLOAT32", "outDataType": "INT32"},
        ),
    ]

    stats = _rewrite_constant_divisors_to_multiplicative_reciprocals(model_ir)

    assert stats == {"rewritten_constant_div_to_mul": 0}
    assert [operator.op_type for operator in model_ir.operators] == ["DIV", "CAST"]
    assert not any("div_reciprocal" in name for name in model_ir.tensors)


def test_precision_restore_updates_inputs_and_type_index(monkeypatch) -> None:
    model_ir = ModelIR("precision_restore")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "y_div_reciprocal": _tensor(
            "y_div_reciprocal",
            data=np.asarray([0.5], dtype=np.float32),
        ),
        "y": _tensor("y"),
        "scale": _tensor("scale", data=np.asarray([3.0], dtype=np.float32)),
        "scaled": _tensor("scaled"),
        "out": _tensor("out", dtype="INT32"),
    }
    model_ir.operators = [
        OperatorIR("MUL", ["x", "y_div_reciprocal"], ["y"]),
        OperatorIR("MUL", ["y", "scale"], ["scaled"]),
        OperatorIR(
            "CAST",
            ["scaled"],
            ["out"],
            options={"inDataType": "FLOAT32", "outDataType": "INT32"},
        ),
    ]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _restore_precision_sensitive_reciprocal_divisions(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"restored_precision_sensitive_reciprocal_divisions": 1}
    assert refresh_count == 1
    assert graph_index.operator_indices("DIV") == [0]
    assert graph_index.operator_indices("MUL") == [1]
    divisor_name = str(model_ir.operators[0].inputs[1])
    np.testing.assert_array_equal(
        model_ir.tensors[divisor_name].data,
        np.asarray([2.0], dtype=np.float32),
    )
    assert "y_div_reciprocal" not in model_ir.tensors
    assert graph_index.consumer_indices(divisor_name) == [0]
    assert layout_state.validate_against_model_ir(model_ir) == []
