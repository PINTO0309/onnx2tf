from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes import split_fallback
from onnx2tf.tflite_builder.passes.split_fallback import (
    replace_unsupported_split_with_slice,
)


def _split_model(*, output_dtype: str, axis: int) -> ModelIR:
    model_ir = ModelIR("unsupported_split")
    model_ir.inputs = ["source"]
    model_ir.outputs = ["left", "right"]
    model_ir.tensors = {
        "source": TensorIR(
            name="source",
            dtype="FLOAT16",
            shape=[2, 1, 4],
            shape_signature=[2, 1, 4],
        ),
        "axis": TensorIR(
            name="axis",
            dtype="INT32",
            shape=[1],
            shape_signature=[1],
            data=np.asarray([axis], dtype=np.int32),
        ),
        "left": TensorIR(
            name="left",
            dtype=output_dtype,
            shape=[1, 1, 4] if axis == 0 else [2, 1, 2],
            shape_signature=[1, 1, 4] if axis == 0 else [2, 1, 2],
        ),
        "right": TensorIR(
            name="right",
            dtype=output_dtype,
            shape=[1, 1, 4] if axis == 0 else [2, 1, 2],
            shape_signature=[1, 1, 4] if axis == 0 else [2, 1, 2],
        ),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="SPLIT",
            inputs=["axis", "source"],
            outputs=["left", "right"],
            options={"numSplits": 2},
        )
    ]
    return model_ir


def test_split_fallback_inserts_cast_and_indexed_slices(monkeypatch) -> None:
    model_ir = _split_model(output_dtype="FLOAT32", axis=0)
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = replace_unsupported_split_with_slice(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"replaced_unsupported_split_with_slice": 1}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["CAST", "SLICE", "SLICE"]
    assert graph_index.operator_indices("SPLIT") == []
    assert graph_index.operator_indices("CAST") == [0]
    assert graph_index.operator_indices("SLICE") == [1, 2]
    cast_output = str(model_ir.operators[0].outputs[0])
    assert model_ir.tensors[cast_output].dtype == "FLOAT32"
    assert all(op.inputs[0] == cast_output for op in model_ir.operators[1:])
    begin_values = [
        np.asarray(model_ir.tensors[op.inputs[1]].data).tolist()
        for op in model_ir.operators[1:]
    ]
    assert begin_values == [[0, 0, 0], [1, 0, 0]]
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_split_fallback_uses_source_directly_when_output_dtype_matches() -> None:
    model_ir = _split_model(output_dtype="FLOAT16", axis=-1)

    stats = replace_unsupported_split_with_slice(model_ir)

    assert stats == {"replaced_unsupported_split_with_slice": 1}
    assert [op.op_type for op in model_ir.operators] == ["SLICE", "SLICE"]
    assert all(op.inputs[0] == "source" for op in model_ir.operators)
    begin_values = [
        np.asarray(model_ir.tensors[op.inputs[1]].data).tolist()
        for op in model_ir.operators
    ]
    size_values = [
        np.asarray(model_ir.tensors[op.inputs[2]].data).tolist()
        for op in model_ir.operators
    ]
    assert begin_values == [[0, 0, 0], [0, 0, 2]]
    assert size_values == [[-1, -1, 2], [-1, -1, 2]]


def test_split_fallback_zero_counter_performs_no_cleanup_or_layout_sync(
    monkeypatch,
) -> None:
    model_ir = _split_model(output_dtype="FLOAT32", axis=0)
    model_ir.tensors["source"].dtype = "FLOAT32"
    layout_state = LayoutState.from_model_ir(model_ir)

    def unexpected_prune(*args, **kwargs) -> None:
        raise AssertionError("zero-rewrite split fallback must not prune")

    def unexpected_sync(*args, **kwargs) -> None:
        raise AssertionError("zero-rewrite split fallback must not sync layout")

    monkeypatch.setattr(split_fallback, "_prune_unused_tensors", unexpected_prune)
    monkeypatch.setattr(LayoutState, "sync_from_model_ir", unexpected_sync)

    stats = replace_unsupported_split_with_slice(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"replaced_unsupported_split_with_slice": 0}
    assert [op.op_type for op in model_ir.operators] == ["SPLIT"]


def test_split_fallback_positive_counter_covers_cleanup_and_layout_sync(
    monkeypatch,
) -> None:
    model_ir = _split_model(output_dtype="FLOAT32", axis=0)
    layout_state = LayoutState.from_model_ir(model_ir)
    prune_calls = 0
    sync_calls = 0
    original_prune = split_fallback._prune_unused_tensors
    original_sync = LayoutState.sync_from_model_ir

    def counted_prune(*args, **kwargs) -> None:
        nonlocal prune_calls
        prune_calls += 1
        original_prune(*args, **kwargs)

    def counted_sync(self, synced_model_ir) -> None:
        nonlocal sync_calls
        sync_calls += 1
        original_sync(self, synced_model_ir)

    monkeypatch.setattr(split_fallback, "_prune_unused_tensors", counted_prune)
    monkeypatch.setattr(LayoutState, "sync_from_model_ir", counted_sync)

    stats = replace_unsupported_split_with_slice(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"replaced_unsupported_split_with_slice": 1}
    assert prune_calls == 1
    assert sync_calls == 1
