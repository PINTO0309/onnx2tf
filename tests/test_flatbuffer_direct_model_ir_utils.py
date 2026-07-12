from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _tensor(name: str) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )


def test_prune_unused_tensors_records_deterministic_lineage() -> None:
    model_ir = ModelIR("prune_unused_tensor_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "y": _tensor("y"),
        "unused": _tensor("unused"),
    }

    _prune_unused_tensors(model_ir)

    assert set(model_ir.tensors) == {"x", "y"}
    assert model_ir.metadata["tensor_lineage_events"] == [
        {
            "kind": "prune_unused_tensors",
            "removed_names": ["unused"],
            "event_index": 0,
        }
    ]


def test_fully_known_positive_shape_rejects_dynamic_or_empty_shapes() -> None:
    assert _is_fully_known_positive_shape([1, 2, 3]) is True
    assert _is_fully_known_positive_shape([1, -1, 3]) is False
    assert _is_fully_known_positive_shape([1, 0, 3]) is False
    assert _is_fully_known_positive_shape([]) is False
    assert _is_fully_known_positive_shape(None) is False


def test_graph_helpers_read_transpose_and_record_input_replacement() -> None:
    model_ir = ModelIR("graph_helper_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "x_internal": _tensor("x_internal"),
        "perm": TensorIR(
            name="perm",
            dtype="INT32",
            shape=[3],
            shape_signature=[3],
            data=np.asarray([0, 2, 1], dtype=np.int32),
            is_variable=False,
        ),
        "y": _tensor("y"),
    }
    transpose = OperatorIR(
        op_type="TRANSPOSE",
        inputs=["x", "perm"],
        outputs=["x_internal"],
    )
    consumer = OperatorIR(
        op_type="ABS",
        inputs=["x_internal"],
        outputs=["y"],
    )
    model_ir.operators = [transpose, consumer]

    assert _read_transpose_perm(model_ir, transpose) == [0, 2, 1]
    assert _build_tensor_consumer_map(model_ir) == {
        "x": [0],
        "perm": [0],
        "x_internal": [1],
    }

    _replace_tensor_inputs(model_ir, "x_internal", "x")

    assert consumer.inputs == ["x"]
    assert model_ir.metadata["tensor_lineage_events"] == [
        {
            "kind": "replace_input",
            "src_name": "x_internal",
            "dst_name": "x",
            "event_index": 0,
        }
    ]
