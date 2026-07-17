from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import onnx2tf.tflite_builder.lower_from_onnx2tf as lowering_module
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _reconcile_static_tensor_shapes,
    _repair_rank4_channelwise_broadcast_constants_to_runtime_layout,
    _repair_stale_nchw_to_nhwc_channelwise_binary_transposes,
    _run_indexed_binary_layout_convergence,
)
from onnx2tf.tflite_builder.passes.stale_binary_adapter_repair import (
    _repair_stale_nchw_to_nhwc_channelwise_binary_transposes as _repair_stale_nchw_to_nhwc_channelwise_binary_transposes_owner,
)


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if is_dataclass(value):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
    layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32" if data is None else str(data.dtype).upper(),
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        logical_layout=layout,
    )


def _make_binary_layout_convergence_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_binary_layout_convergence")
    model_ir.inputs = ["source0", "source1", "peer_input", "broadcast_x"]
    model_ir.outputs = ["add0", "add1", "broadcast_y"]
    model_ir.tensors = {
        "source0": _tensor("source0", [1, 4, 2, 3], layout="NHWC"),
        "source1": _tensor("source1", [1, 4, 2, 3], layout="NHWC"),
        "adapter0": _tensor("adapter0", [1, 2, 3, 4]),
        "adapter1": _tensor("adapter1", [1, 2, 3, 4]),
        "perm": _tensor(
            "perm",
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "peer_input": _tensor("peer_input", [1, 4, 2, 3], layout="NHWC"),
        "peer_filter": _tensor(
            "peer_filter",
            [3, 1, 1, 3],
            data=np.ones((3, 1, 1, 3), dtype=np.float32),
        ),
        "peer_bias": _tensor(
            "peer_bias",
            [3],
            data=np.zeros((3,), dtype=np.float32),
        ),
        "peer": _tensor("peer", [1, 4, 2, 3], layout="NHWC"),
        "add0": _tensor("add0", [1, 2, 3, 4]),
        "add1": _tensor("add1", [1, 2, 3, 4]),
        "broadcast_x": _tensor(
            "broadcast_x",
            [1, 2, 2, 3],
            layout="NHWC",
        ),
        "scale": _tensor(
            "scale",
            [1, 3, 1, 1],
            data=np.arange(3, dtype=np.float32).reshape(1, 3, 1, 1),
        ),
        "broadcast_y": _tensor(
            "broadcast_y",
            [1, 2, 2, 3],
            layout="NHWC",
        ),
    }
    model_ir.operators = [
        OperatorIR(
            "CONV_2D",
            ["peer_input", "peer_filter", "peer_bias"],
            ["peer"],
            {
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR("TRANSPOSE", ["source0", "perm"], ["adapter0"]),
        OperatorIR("ADD", ["adapter0", "peer"], ["add0"]),
        OperatorIR("TRANSPOSE", ["source1", "perm"], ["adapter1"]),
        OperatorIR("MUL", ["peer", "adapter1"], ["add1"]),
        OperatorIR("MUL", ["broadcast_x", "scale"], ["broadcast_y"]),
    ]
    return model_ir


def _run_legacy_binary_layout_convergence(
    model_ir: ModelIR,
) -> dict[str, int]:
    repaired_constants = 0
    removed_transposes = 0
    reconciled_shapes = 0
    for _ in range(3):
        broadcast_stats = (
            _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
                model_ir
            )
        )
        transpose_stats = (
            _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(model_ir)
        )
        reconcile_stats = _reconcile_static_tensor_shapes(model_ir)
        repaired_constants += int(
            broadcast_stats["repaired_rank4_channelwise_broadcast_constants"]
        )
        removed_transposes += int(
            transpose_stats[
                "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes"
            ]
        )
        reconciled_shapes += int(
            reconcile_stats["reconciled_static_tensor_shapes"]
        )
    return {
        "repaired_rank4_channelwise_broadcast_constants": repaired_constants,
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": (
            removed_transposes
        ),
        "reconciled_static_tensor_shapes": reconciled_shapes,
    }


def test_stale_binary_transpose_repair_owner_matches_lowerer_wrapper() -> None:
    owner_model_ir = _make_binary_layout_convergence_model_ir()
    wrapper_model_ir = copy.deepcopy(owner_model_ir)

    owner_stats = _repair_stale_nchw_to_nhwc_channelwise_binary_transposes_owner(
        owner_model_ir
    )
    wrapper_stats = _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(
        wrapper_model_ir
    )

    assert owner_stats == wrapper_stats
    assert _normalize(owner_model_ir) == _normalize(wrapper_model_ir)


def test_indexed_stale_binary_transpose_repair_handles_multiple_matches(
    monkeypatch,
) -> None:
    model_ir = _make_binary_layout_convergence_model_ir()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    def unexpected_graph_rescan(*args, **kwargs):
        raise AssertionError("unexpected producer/consumer map rebuild")

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_consumer_map",
        unexpected_graph_rescan,
    )
    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_producer_map",
        unexpected_graph_rescan,
    )
    graph_index = ModelIRGraphIndex(model_ir)
    stats = _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": 2,
    }
    assert refresh_count == 1
    assert [str(op.op_type) for op in model_ir.operators] == [
        "CONV_2D",
        "ADD",
        "MUL",
        "MUL",
    ]
    assert model_ir.operators[1].inputs == ["source0", "peer"]
    assert model_ir.operators[2].inputs == ["peer", "source1"]
    assert model_ir.tensors["add0"].shape == [1, 4, 2, 3]
    assert model_ir.tensors["add1"].shape == [1, 4, 2, 3]
    assert "adapter0" not in model_ir.tensors
    assert "adapter1" not in model_ir.tensors
    assert "perm" not in model_ir.tensors
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type


def test_indexed_stale_binary_transpose_repair_preserves_fanout_adapter() -> None:
    model_ir = _make_binary_layout_convergence_model_ir()
    model_ir.outputs.append("adapter0_side")
    model_ir.tensors["adapter0_side"] = _tensor(
        "adapter0_side",
        [1, 2, 3, 4],
    )
    model_ir.operators.append(
        OperatorIR("RELU", ["adapter0"], ["adapter0_side"])
    )
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": 1,
    }
    assert any(
        str(op.op_type) == "TRANSPOSE" and op.outputs == ["adapter0"]
        for op in model_ir.operators
    )
    add_op = next(op for op in model_ir.operators if op.outputs == ["add0"])
    mul_op = next(op for op in model_ir.operators if op.outputs == ["add1"])
    assert add_op.inputs == ["adapter0", "peer"]
    assert mul_op.inputs == ["peer", "source1"]
    assert graph_index.consumer_indices("adapter0") == [2, 5]
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type


@pytest.mark.parametrize(
    "invalid_metadata",
    ["short_shape", "short_signature"],
)
def test_stale_binary_transpose_repair_rejects_invalid_metadata_atomically(
    invalid_metadata: str,
) -> None:
    model_ir = _make_binary_layout_convergence_model_ir()
    model_ir.outputs.append("adapter1")
    source_tensor = model_ir.tensors["source0"]
    if invalid_metadata == "short_shape":
        source_tensor.shape = [1, 4, 2]
        source_tensor.shape_signature = [1, 4, 2]
    else:
        source_tensor.shape_signature = [1, 4]
    before = _normalize(copy.deepcopy(model_ir))

    stats = _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(model_ir)

    assert stats == {
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": 0,
    }
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("tensor_name", ["source0", "adapter0"])
def test_stale_binary_transpose_repair_prevalidates_channelwise_ranks(
    tensor_name: str,
) -> None:
    model_ir = _make_binary_layout_convergence_model_ir()
    model_ir.outputs.append("adapter1")
    model_ir.tensors["channelwise_bias"] = _tensor(
        "channelwise_bias",
        [1, 1, 1, 3],
        data=np.zeros((1, 1, 1, 3), dtype=np.float32),
    )
    add_op = next(op for op in model_ir.operators if op.outputs == ["add0"])
    add_op.inputs[1] = "channelwise_bias"
    model_ir.tensors[tensor_name].shape = [1, 4, 2]
    model_ir.tensors[tensor_name].shape_signature = [1, 4, 2]
    before = _normalize(copy.deepcopy(model_ir))

    stats = _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(model_ir)

    assert stats == {
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": 0,
    }
    assert _normalize(model_ir) == before


def test_indexed_binary_layout_convergence_matches_legacy_sequence(
    monkeypatch,
) -> None:
    model_ir = _make_binary_layout_convergence_model_ir()
    legacy_model_ir = copy.deepcopy(model_ir)
    legacy_stats = _run_legacy_binary_layout_convergence(legacy_model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    indexed_stats = _run_indexed_binary_layout_convergence(model_ir)

    assert refresh_count == 1
    assert indexed_stats == legacy_stats
    assert indexed_stats[
        "repaired_rank4_channelwise_broadcast_constants"
    ] == 1
    assert indexed_stats[
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes"
    ] == 2
    assert _normalize(model_ir) == _normalize(legacy_model_ir)
