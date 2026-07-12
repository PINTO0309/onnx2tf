from __future__ import annotations

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


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
