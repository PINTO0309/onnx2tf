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
    _repair_singleton_nhwc_conv_input_reshapes,
    _repair_stale_nchw_to_nhwc_conv_input_transposes,
    _run_indexed_conv_input_adapter_repairs,
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
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32" if data is None else str(data.dtype).upper(),
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _make_conv_input_adapter_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_conv_input_adapter_repairs")
    model_ir.inputs = ["singleton_source", "transpose_source0", "transpose_source1"]
    model_ir.outputs = ["singleton_y", "transpose_y0", "transpose_y1"]
    model_ir.tensors = {
        "singleton_source": _tensor("singleton_source", [1, 1, 1, 4]),
        "singleton_adapter": _tensor("singleton_adapter", [1, 1, 4, 1]),
        "singleton_shape": _tensor(
            "singleton_shape",
            [4],
            data=np.asarray([1, 1, 4, 1], dtype=np.int32),
        ),
        "singleton_filter": _tensor(
            "singleton_filter",
            [3, 1, 1, 4],
            data=np.ones((3, 1, 1, 4), dtype=np.float32),
        ),
        "singleton_bias": _tensor(
            "singleton_bias",
            [3],
            data=np.zeros((3,), dtype=np.float32),
        ),
        "singleton_y": _tensor("singleton_y", [1, 1, 4, 3]),
        "transpose_source0": _tensor("transpose_source0", [1, 5, 6, 4]),
        "transpose_adapter0": _tensor("transpose_adapter0", [1, 6, 4, 5]),
        "transpose_filter0": _tensor(
            "transpose_filter0",
            [2, 1, 1, 4],
            data=np.ones((2, 1, 1, 4), dtype=np.float32),
        ),
        "transpose_bias0": _tensor(
            "transpose_bias0",
            [2],
            data=np.zeros((2,), dtype=np.float32),
        ),
        "transpose_y0": _tensor("transpose_y0", [1, 6, 4, 2]),
        "transpose_source1": _tensor("transpose_source1", [1, 7, 8, 4]),
        "transpose_adapter1": _tensor("transpose_adapter1", [1, 8, 4, 7]),
        "transpose_filter1": _tensor(
            "transpose_filter1",
            [5, 1, 1, 4],
            data=np.ones((5, 1, 1, 4), dtype=np.float32),
        ),
        "transpose_bias1": _tensor(
            "transpose_bias1",
            [5],
            data=np.zeros((5,), dtype=np.float32),
        ),
        "transpose_y1": _tensor("transpose_y1", [1, 8, 4, 5]),
        "perm": _tensor(
            "perm",
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
    }
    conv_options = {
        "padding": "SAME",
        "strideH": 1,
        "strideW": 1,
        "dilationHFactor": 1,
        "dilationWFactor": 1,
        "fusedActivationFunction": "NONE",
    }
    model_ir.operators = [
        OperatorIR(
            "RESHAPE",
            ["singleton_source", "singleton_shape"],
            ["singleton_adapter"],
        ),
        OperatorIR(
            "CONV_2D",
            ["singleton_adapter", "singleton_filter", "singleton_bias"],
            ["singleton_y"],
            dict(conv_options),
        ),
        OperatorIR(
            "TRANSPOSE",
            ["transpose_source0", "perm"],
            ["transpose_adapter0"],
        ),
        OperatorIR(
            "CONV_2D",
            ["transpose_adapter0", "transpose_filter0", "transpose_bias0"],
            ["transpose_y0"],
            dict(conv_options),
        ),
        OperatorIR(
            "TRANSPOSE",
            ["transpose_source1", "perm"],
            ["transpose_adapter1"],
        ),
        OperatorIR(
            "CONV_2D",
            ["transpose_adapter1", "transpose_filter1", "transpose_bias1"],
            ["transpose_y1"],
            dict(conv_options),
        ),
    ]
    return model_ir


def test_indexed_conv_input_adapter_repairs_match_legacy_pair(
    monkeypatch,
) -> None:
    model_ir = _make_conv_input_adapter_model_ir()
    legacy_model_ir = copy.deepcopy(model_ir)
    legacy_stats = {
        **_repair_singleton_nhwc_conv_input_reshapes(legacy_model_ir),
        **_repair_stale_nchw_to_nhwc_conv_input_transposes(legacy_model_ir),
    }
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
    indexed_stats = _run_indexed_conv_input_adapter_repairs(model_ir)

    assert refresh_count == 1
    assert indexed_stats == legacy_stats
    assert indexed_stats == {
        "repaired_singleton_nhwc_conv_input_reshapes": 1,
        "repaired_stale_nchw_to_nhwc_conv_input_transposes": 2,
    }
    assert [str(op.op_type) for op in model_ir.operators] == [
        "CONV_2D",
        "CONV_2D",
        "CONV_2D",
    ]
    assert [op.inputs[0] for op in model_ir.operators] == [
        "singleton_source",
        "transpose_source0",
        "transpose_source1",
    ]
    assert _normalize(model_ir) == _normalize(legacy_model_ir)


@pytest.mark.parametrize("protection", ["fanout", "graph_output"])
def test_indexed_conv_transpose_repair_preserves_protected_adapter(
    protection: str,
) -> None:
    model_ir = _make_conv_input_adapter_model_ir()
    if protection == "fanout":
        model_ir.outputs.append("transpose_side")
        model_ir.tensors["transpose_side"] = _tensor(
            "transpose_side",
            [1, 6, 4, 5],
        )
        model_ir.operators.append(
            OperatorIR("RELU", ["transpose_adapter0"], ["transpose_side"])
        )
    else:
        model_ir.outputs.append("transpose_adapter0")
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _repair_stale_nchw_to_nhwc_conv_input_transposes(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {
        "repaired_stale_nchw_to_nhwc_conv_input_transposes": 1,
    }
    kept_adapter = graph_index.producer("transpose_adapter0")
    assert kept_adapter is not None
    assert str(kept_adapter.op_type) == "TRANSPOSE"
    first_conv = next(op for op in model_ir.operators if op.outputs == ["transpose_y0"])
    second_conv = next(op for op in model_ir.operators if op.outputs == ["transpose_y1"])
    assert first_conv.inputs[0] == "transpose_adapter0"
    assert second_conv.inputs[0] == "transpose_source1"
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type


@pytest.mark.parametrize(
    ("repair", "source_name", "expected_stats"),
    [
        (
            _repair_singleton_nhwc_conv_input_reshapes,
            "singleton_source",
            {"repaired_singleton_nhwc_conv_input_reshapes": 0},
        ),
        (
            _repair_stale_nchw_to_nhwc_conv_input_transposes,
            "transpose_source0",
            {"repaired_stale_nchw_to_nhwc_conv_input_transposes": 0},
        ),
    ],
    ids=["singleton_reshape", "stale_transpose"],
)
@pytest.mark.xfail(
    strict=True,
    reason="source shape_signature is read after the first Conv input mutation",
)
def test_conv_input_adapter_repair_rejects_invalid_source_signature_atomically(
    repair,
    source_name: str,
    expected_stats: dict[str, int],
) -> None:
    model_ir = _make_conv_input_adapter_model_ir()
    model_ir.tensors[source_name].shape_signature = [1, 1]
    before = _normalize(copy.deepcopy(model_ir))

    stats = repair(model_ir)

    assert stats == expected_stats
    assert _normalize(model_ir) == before
