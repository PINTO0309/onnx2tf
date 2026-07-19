from __future__ import annotations

from copy import deepcopy

import numpy as np
import onnx2tf.tflite_builder.passes.binary_layout_adapter as binary_layout_owner

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_rank4_channelwise_broadcast_constants_to_runtime_layout,
)
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    repair_rank4_channelwise_broadcast_constants_to_runtime_layout,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
    layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        logical_layout=layout,
    )


def test_repair_stale_nhwc_constant_back_to_broadcastable_spatial_shape() -> None:
    model_ir = ModelIR("stale_nhwc_spatial_constant")
    model_ir.tensors = {
        "coordinate": _tensor(
            "coordinate",
            [1, 64, 1, 1],
            data=np.arange(64, dtype=np.float32).reshape(1, 64, 1, 1),
            layout="NHWC",
        ),
        "x": _tensor("x", [1, 32, 64, 3], layout="NHWC"),
        "y": _tensor("y", [1, 32, 64, 3], layout="NHWC"),
    }
    model_ir.operators = [
        OperatorIR(op_type="SUB", inputs=["coordinate", "x"], outputs=["y"])
    ]

    owner_model_ir = deepcopy(model_ir)
    owner_stats = (
        repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
            owner_model_ir
        )
    )
    stats = _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
        model_ir
    )

    assert owner_stats == stats
    assert (
        ModelIRPassState(owner_model_ir).fingerprint()
        == ModelIRPassState(model_ir).fingerprint()
    )
    assert stats == {"repaired_rank4_channelwise_broadcast_constants": 1}
    assert model_ir.tensors["coordinate"].shape == [1, 1, 64, 1]
    np.testing.assert_array_equal(
        model_ir.tensors["coordinate"].data.reshape(-1),
        np.arange(64, dtype=np.float32),
    )


def test_broadcast_constant_repair_keeps_already_compatible_spatial_shape() -> None:
    coordinate = np.arange(64, dtype=np.float32).reshape(1, 1, 64, 1)
    model_ir = ModelIR("compatible_spatial_constant")
    model_ir.tensors = {
        "coordinate": _tensor(
            "coordinate", [1, 1, 64, 1], data=coordinate, layout="NHWC"
        ),
        "x": _tensor("x", [1, 32, 64, 3], layout="NHWC"),
        "y": _tensor("y", [1, 32, 64, 3], layout="NHWC"),
    }
    model_ir.operators = [
        OperatorIR(op_type="SUB", inputs=["coordinate", "x"], outputs=["y"])
    ]

    stats = _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
        model_ir
    )

    assert stats == {"repaired_rank4_channelwise_broadcast_constants": 0}
    np.testing.assert_array_equal(model_ir.tensors["coordinate"].data, coordinate)


def test_shared_broadcast_constant_repair_preserves_snapshot_clone_policy(
    monkeypatch,
) -> None:
    model_ir = ModelIR("shared_channelwise_constant")
    model_ir.inputs = ["x0", "x1"]
    model_ir.outputs = ["y0", "y1"]
    model_ir.tensors = {
        "x0": _tensor("x0", [1, 2, 2, 3], layout="NHWC"),
        "x1": _tensor("x1", [1, 2, 2, 3], layout="NHWC"),
        "scale": _tensor(
            "scale",
            [1, 3, 1, 1],
            data=np.arange(3, dtype=np.float32).reshape(1, 3, 1, 1),
        ),
        "y0": _tensor("y0", [1, 2, 2, 3], layout="NHWC"),
        "y1": _tensor("y1", [1, 2, 2, 3], layout="NHWC"),
    }
    model_ir.operators = [
        OperatorIR("MUL", ["x0", "scale"], ["y0"]),
        OperatorIR("ADD", ["x1", "scale"], ["y1"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    def unexpected_graph_rescan(*args, **kwargs):
        raise AssertionError("unexpected producer/consumer map rebuild")

    monkeypatch.setattr(
        binary_layout_owner,
        "_build_tensor_consumer_map",
        unexpected_graph_rescan,
        raising=False,
    )
    monkeypatch.setattr(
        binary_layout_owner,
        "_build_tensor_producer_map",
        unexpected_graph_rescan,
        raising=False,
    )
    stats = _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {"repaired_rank4_channelwise_broadcast_constants": 2}
    assert model_ir.operators[0].inputs == ["x0", "scale__nhwc"]
    assert model_ir.operators[1].inputs == ["x1", "scale__nhwc_1"]
    assert model_ir.tensors["scale"].shape == [1, 3, 1, 1]
    for clone_name in ["scale__nhwc", "scale__nhwc_1"]:
        assert model_ir.tensors[clone_name].shape == [1, 1, 1, 3]
        np.testing.assert_array_equal(
            model_ir.tensors[clone_name].data.reshape(-1),
            np.arange(3, dtype=np.float32),
        )
    assert graph_index.consumer_indices("scale") == []
    assert graph_index.consumer_indices("scale__nhwc") == [0]
    assert graph_index.consumer_indices("scale__nhwc_1") == [1]
    assert refresh_count == 1
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type
