from __future__ import annotations

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.pytorch_normalization import (
    _collect_model_op_types,
    normalize_model_ir_for_pytorch_channel_first,
    prepare_model_ir_for_native_pytorch,
)


def test_channel_first_normalizer_is_torch_free_and_owns_one_graph_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR(name="torch_free_normalization")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR(
            "x",
            "FLOAT32",
            [1, 3, 2, 4],
            [1, 3, 2, 4],
            logical_layout="NCHW",
            onnx_tensor_name="onnx_x",
        ),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 3, 2, 4],
            [1, 3, 2, 4],
            logical_layout="NCHW",
            onnx_tensor_name="onnx_y",
        ),
    }
    model_ir.operators = [
        OperatorIR(
            "RELU",
            ["x"],
            ["y"],
            version=2,
            onnx_node_name="relu_0",
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

    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)

    assert refresh_count == 1
    assert normalized is not model_ir
    assert normalized.tensors["x"] is not model_ir.tensors["x"]
    assert normalized.operators[0] is not model_ir.operators[0]
    assert normalized.operators[0].op_type == "RELU"
    assert normalized.operators[0].version == 2
    assert normalized.operators[0].onnx_node_name == "relu_0"
    assert normalized.tensors["x"].logical_layout == "NCHW"
    assert normalized.tensors["y"].logical_layout == "NCHW"
    assert normalized.metadata["assume_channel_last_layout_tensor_names"] == []
    assert model_ir.metadata == {}


def test_native_preparation_is_torch_free_and_reuses_boundary_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR(name="torch_free_preparation")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR(
            "x",
            "FLOAT32",
            [1, 3, 2, 4],
            [-1, 3, 2, 4],
            logical_layout="NCHW",
        ),
        "y": TensorIR(
            "y",
            "FLOAT32",
            [1, 3, 2, 4],
            [-1, 3, 2, 4],
            logical_layout="NCHW",
        ),
    }
    model_ir.operators = [OperatorIR("RELU", ["x"], ["y"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    prepared = prepare_model_ir_for_native_pytorch(model_ir)

    assert refresh_count == 2
    assert prepared is not model_ir
    assert prepared.metadata["onnx_public_layout_map"] == {
        "x": "NCHW",
        "y": "NCHW",
    }
    assert prepared.metadata["onnx_boundary_shape_signature_map"] == {
        "x": [-1, 3, 2, 4],
        "y": [-1, 3, 2, 4],
    }
    assert prepared.tensors["x"].shape_signature == [-1, 3, 2, 4]
    assert model_ir.metadata == {}


def test_model_op_type_collection_includes_subgraphs() -> None:
    model_ir = ModelIR(name="root")
    model_ir.operators = [OperatorIR("RELU", ["x"], ["y"])]
    subgraph = ModelIR(name="body")
    subgraph.operators = [OperatorIR("ADD", ["x", "x"], ["y"])]
    model_ir.subgraphs = [subgraph]

    assert _collect_model_op_types(model_ir) == {"ADD", "RELU"}
