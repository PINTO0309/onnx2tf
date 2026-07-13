from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.pytorch_layout_validation import (
    _apply_feature_last_sequence_layouts,
    _collect_feature_last_sequence_tensor_names,
    _ensure_public_boundary_layout_bridges,
    _is_attention_like_softmax_op,
    _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island,
    _is_transpose_sandwiched_last_axis_softmax_op,
    _propagate_channel_last_layouts,
    _propagate_feature_last_tensor_names,
    _restore_non_preserved_channel_first_layouts,
    _shrink_preserved_channel_last_regions_for_pytorch,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    layout: str = "UNKNOWN",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT32" if data is not None else "FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        logical_layout=layout,
    )


def test_attention_softmax_reuses_supplied_graph_index(monkeypatch) -> None:
    model_ir = ModelIR(name="attention_softmax")
    model_ir.tensors = {
        "scores": _tensor("scores", [1, 2, 4, 5]),
        "probabilities": _tensor("probabilities", [1, 2, 4, 5]),
        "values": _tensor("values", [1, 2, 5, 3]),
        "context": _tensor("context", [1, 2, 4, 3]),
    }
    softmax = OperatorIR("SOFTMAX", ["scores"], ["probabilities"], {"axis": -1})
    model_ir.operators = [
        softmax,
        OperatorIR("BATCH_MATMUL", ["probabilities", "values"], ["context"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    assert _is_attention_like_softmax_op(
        model_ir,
        softmax,
        graph_index=graph_index,
    ) is True
    assert refresh_count == 1


def _transpose_sandwich_model() -> tuple[ModelIR, OperatorIR]:
    model_ir = ModelIR(name="transpose_softmax_transpose")
    model_ir.tensors = {
        "x": _tensor("x", [1, 3, 4], layout="NCW"),
        "pre_perm": _tensor(
            "pre_perm",
            [3],
            data=np.asarray([0, 2, 1], dtype=np.int32),
        ),
        "scores": _tensor("scores", [1, 4, 3]),
        "probabilities": _tensor("probabilities", [1, 4, 3]),
        "post_perm": _tensor(
            "post_perm",
            [3],
            data=np.asarray([0, 2, 1], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 3, 4], layout="NCW"),
    }
    softmax = OperatorIR(
        "SOFTMAX",
        ["scores"],
        ["probabilities"],
        {"axis": -1},
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "pre_perm"], ["scores"]),
        softmax,
        OperatorIR("TRANSPOSE", ["probabilities", "post_perm"], ["y"]),
    ]
    return model_ir, softmax


def test_transpose_sandwiched_softmax_uses_indexed_edges() -> None:
    model_ir, softmax = _transpose_sandwich_model()
    graph_index = ModelIRGraphIndex(model_ir)

    assert _is_transpose_sandwiched_last_axis_softmax_op(
        model_ir,
        softmax,
        graph_index=graph_index,
    ) is True


def test_transpose_sandwiched_softmax_rejects_duplicate_producer() -> None:
    model_ir, softmax = _transpose_sandwich_model()
    model_ir.operators.insert(0, OperatorIR("IDENTITY", ["x"], ["scores"]))
    graph_index = ModelIRGraphIndex(model_ir)

    assert _is_transpose_sandwiched_last_axis_softmax_op(
        model_ir,
        softmax,
        graph_index=graph_index,
    ) is False


def test_feature_last_worklist_reaches_bidirectional_fixed_point(monkeypatch) -> None:
    model_ir = ModelIR(name="feature_last_worklist")
    model_ir.tensors = {
        name: _tensor(name, [1, 2, 3, 4])
        for name in ["x", "relu", "residual", "sum"]
    }
    model_ir.operators = [
        OperatorIR("RELU", ["x"], ["relu"]),
        OperatorIR("ADD", ["relu", "residual"], ["sum"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    preserved = _propagate_feature_last_tensor_names(
        model_ir,
        {"relu"},
        graph_index=graph_index,
    )

    assert preserved == {"x", "relu", "residual", "sum"}
    assert refresh_count == 1


def test_feature_last_worklist_stops_at_standard_layout_transpose() -> None:
    model_ir = ModelIR(name="feature_last_transpose_barrier")
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4], layout="NHWC"),
        "perm": _tensor(
            "perm",
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 4, 2, 3], layout="NCHW"),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["y"]),
    ]
    graph_index = ModelIRGraphIndex(model_ir)

    assert _propagate_feature_last_tensor_names(
        model_ir,
        {"x"},
        graph_index=graph_index,
    ) == {"x"}
    assert _propagate_feature_last_tensor_names(
        model_ir,
        {"y"},
        graph_index=graph_index,
    ) == {"y"}


def test_channel_last_layout_worklist_handles_reverse_operator_order() -> None:
    model_ir = ModelIR(name="reverse_order_channel_last")
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4], layout="NHWC"),
        "middle": _tensor("middle", [1, 2, 3, 4]),
        "y": _tensor("y", [1, 2, 3, 4]),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["middle"], ["y"]),
        OperatorIR("RELU", ["x"], ["middle"]),
    ]
    graph_index = ModelIRGraphIndex(model_ir)

    changed = _propagate_channel_last_layouts(
        model_ir,
        consumers=graph_index.consumers,
    )

    assert changed is True
    assert model_ir.tensors["middle"].logical_layout == "NHWC"
    assert model_ir.tensors["y"].logical_layout == "NHWC"


def test_channel_last_layout_worklist_ignores_unsafe_ops() -> None:
    model_ir = ModelIR(name="unsafe_channel_last_boundary")
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4], layout="NHWC"),
        "y": _tensor("y", [1, 2, 3, 4]),
    }
    model_ir.operators = [OperatorIR("CUSTOM", ["x"], ["y"])]
    graph_index = ModelIRGraphIndex(model_ir)

    changed = _propagate_channel_last_layouts(
        model_ir,
        consumers=graph_index.consumers,
    )

    assert changed is False
    assert model_ir.tensors["y"].logical_layout == "UNKNOWN"


def test_feature_last_application_restores_preserved_reshape_contract() -> None:
    model_ir = ModelIR(name="preserved_feature_last_reshape")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3]),
        "shape": _tensor(
            "shape",
            [3],
            data=np.asarray([1, 3, 2], dtype=np.int32),
        ),
        "y": _tensor("y", [1, 3, 2]),
    }
    model_ir.operators = [
        OperatorIR(
            "RESHAPE",
            ["x", "shape"],
            ["y"],
            {
                "newShape": [1, 3, 2],
                "onnxRawNewShape": [1, 2, 3],
            },
        )
    ]

    changed = _apply_feature_last_sequence_layouts(model_ir, {"y"})

    assert changed is True
    assert model_ir.tensors["y"].logical_layout == "NWC"
    assert model_ir.tensors["y"].shape == [1, 2, 3]
    assert model_ir.tensors["y"].shape_signature == [1, 2, 3]
    assert model_ir.operators[0].options["newShape"] == [1, 2, 3]
    np.testing.assert_array_equal(
        model_ir.tensors["shape"].data,
        np.asarray([1, 2, 3], dtype=np.int32),
    )


def test_feature_last_application_skips_index_for_empty_preserve_set(
    monkeypatch,
) -> None:
    model_ir = ModelIR(name="no_preserved_feature_last")
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    changed = _apply_feature_last_sequence_layouts(model_ir, set())

    assert changed is False
    assert refresh_count == 0


def test_feature_last_collector_keeps_complete_channel_last_island() -> None:
    model_ir = ModelIR(name="complete_channel_last_island")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4], layout="NHWC"),
        "y": _tensor("y", [1, 2, 3, 4], layout="NHWC"),
    }
    model_ir.operators = [OperatorIR("RELU", ["x"], ["y"])]

    preserved = _collect_feature_last_sequence_tensor_names(model_ir)

    assert preserved == {"x", "y"}
    assert _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(
        model_ir
    ) is True


def test_feature_last_shrink_removes_safe_internal_region_only() -> None:
    model_ir = ModelIR(name="partially_safe_channel_last_region")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        name: _tensor(name, [1, 2, 3, 4], layout="NHWC")
        for name in ["x", "middle", "y", "custom_output"]
    }
    model_ir.operators = [
        OperatorIR("RELU", ["x"], ["middle"]),
        OperatorIR("RELU", ["middle"], ["y"]),
        OperatorIR("CUSTOM", ["y"], ["custom_output"]),
    ]
    graph_index = ModelIRGraphIndex(model_ir)

    shrunken = _shrink_preserved_channel_last_regions_for_pytorch(
        model_ir,
        {"x", "middle", "y"},
        producer_index=graph_index.producers,
        consumers=graph_index.consumers,
    )

    assert shrunken == {"x", "y"}


def test_restore_non_preserved_channel_first_layouts_respects_boundaries() -> None:
    model_ir = ModelIR(name="restore_channel_first")
    model_ir.tensors = {
        "restore": _tensor("restore", [1, 2, 3, 4], layout="NHWC"),
        "preserve": _tensor("preserve", [1, 2, 3, 4], layout="NHWC"),
        "bridge": _tensor("bridge", [1, 2, 3, 4], layout="NHWC"),
    }
    model_ir.metadata["public_layout_bridge_tensor_names"] = ["bridge"]

    _restore_non_preserved_channel_first_layouts(model_ir, {"preserve"})

    assert model_ir.tensors["restore"].logical_layout == "NCHW"
    assert model_ir.tensors["preserve"].logical_layout == "NHWC"
    assert model_ir.tensors["bridge"].logical_layout == "NHWC"


def test_public_boundary_bridges_use_one_differential_index(monkeypatch) -> None:
    model_ir = ModelIR(name="public_boundary_bridges")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3, 2, 4], layout="NCHW"),
        "y": _tensor("y", [1, 3, 2, 4], layout="NCHW"),
    }
    model_ir.operators = [
        OperatorIR(
            "RELU",
            ["x"],
            ["y"],
            version=2,
            onnx_node_name="relu_boundary",
            onnx_op_type="Relu",
        )
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    _ensure_public_boundary_layout_bridges(
        model_ir=model_ir,
        desired_public_shape_map={
            "x": [1, 2, 4, 3],
            "y": [1, 2, 4, 3],
        },
        desired_public_layout_map={"x": "NHWC", "y": "NHWC"},
    )

    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == [
        "TRANSPOSE",
        "RELU",
        "TRANSPOSE",
    ]
    input_bridge_name = model_ir.operators[0].outputs[0]
    output_bridge_name = model_ir.operators[1].outputs[0]
    assert model_ir.operators[1].inputs == [input_bridge_name]
    assert model_ir.operators[2].inputs[0] == output_bridge_name
    assert model_ir.operators[2].outputs == ["y"]
    assert model_ir.operators[1].version == 2
    assert model_ir.operators[1].onnx_node_name == "relu_boundary"
    assert model_ir.metadata["public_layout_bridge_tensor_names"] == [
        input_bridge_name,
        output_bridge_name,
    ]


def test_public_boundary_bridge_no_op_avoids_graph_index(monkeypatch) -> None:
    model_ir = ModelIR(name="public_boundary_no_op")
    model_ir.inputs = ["x"]
    model_ir.tensors["x"] = _tensor("x", [1, 2, 4, 3], layout="NHWC")
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    _ensure_public_boundary_layout_bridges(
        model_ir=model_ir,
        desired_public_shape_map={"x": [1, 2, 4, 3]},
        desired_public_layout_map={"x": "NHWC"},
    )

    assert refresh_count == 0
    assert model_ir.operators == []
