from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _read_const_ints_from_tensor,
    _replace_tensor_inputs,
    _write_const_ints_to_tensor,
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


def test_static_shape_and_constant_vector_helpers_are_deterministic() -> None:
    tensor = TensorIR(
        name="axes",
        dtype="INT64",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 3], dtype=np.int64),
        is_variable=False,
    )

    assert _broadcast_static_shapes([2, 1, 4], [1, 3, 4]) == [2, 3, 4]
    assert _broadcast_static_shapes([2, 3], [4, 3]) is None
    assert _read_const_ints_from_tensor(tensor) == [1, 3]
    assert _write_const_ints_to_tensor(tensor, [0, 2, 3]) is True
    assert _read_const_ints_from_tensor(tensor) == [0, 2, 3]
    assert tensor.data is not None and tensor.data.dtype == np.int64
    assert tensor.shape == [3]
    assert tensor.shape_signature == [3]
    assert _write_const_ints_to_tensor(tensor, [0, 2, 3]) is False
