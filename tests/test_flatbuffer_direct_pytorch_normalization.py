from __future__ import annotations

import onnx2tf.tflite_builder.passes.pytorch_normalization as normalization_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.pytorch_normalization import (
    _collect_model_op_types,
    _rewrite_native_pytorch_compatibility_ops,
    normalize_model_ir_for_pytorch_channel_first,
    prepare_model_ir_for_native_pytorch,
)
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
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

    assert refresh_count == 1
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


def test_native_preparation_fallback_indexes_its_distinct_graph_once(
    monkeypatch,
) -> None:
    model_ir = ModelIR(name="layout_agnostic_fallback")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 4], [1, 4], logical_layout="NC"),
        "y": TensorIR("y", "FLOAT32", [1, 4], [1, 4], logical_layout="NC"),
    }
    model_ir.operators = [OperatorIR("RELU", ["x"], ["y"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def reject_channel_first_normalization(_model_ir):
        raise ModelIRPyTorchExportError("forced layout-agnostic fallback")

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(
        normalization_module,
        "_normalize_model_ir_for_pytorch_channel_first_with_index",
        reject_channel_first_normalization,
    )
    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    prepared = prepare_model_ir_for_native_pytorch(model_ir)

    assert refresh_count == 1
    assert prepared is not model_ir
    assert prepared.operators[0].op_type == "RELU"


def test_native_compatibility_preflight_scans_irrelevant_root_once() -> None:
    class _CountingOperatorList(list):
        def __init__(self, values):
            super().__init__(values)
            self.iteration_count = 0

        def __iter__(self):
            self.iteration_count += 1
            return super().__iter__()

    model_ir = ModelIR(name="native_compatibility_no_op")
    operators = _CountingOperatorList(
        OperatorIR("RELU", [f"t{index}"], [f"t{index + 1}"])
        for index in range(64)
    )
    model_ir.operators = operators

    rewritten = _rewrite_native_pytorch_compatibility_ops(model_ir)

    assert rewritten is model_ir
    assert operators.iteration_count == 1


def test_native_compatibility_preflight_dispatches_only_present_families(
    monkeypatch,
) -> None:
    model_ir = ModelIR(name="native_compatibility_dispatch")
    model_ir.operators = [
        OperatorIR("WHILE", [], []),
        OperatorIR("UNIDIRECTIONAL_SEQUENCE_RNN", [], []),
    ]
    calls = []

    def record(name):
        def rewrite(candidate):
            calls.append(name)
            return candidate

        return rewrite

    monkeypatch.setattr(
        normalization_module,
        "_rewrite_static_while_ops_for_native_export",
        record("static_while"),
    )
    monkeypatch.setattr(
        normalization_module,
        "_rewrite_counter_bounded_while_ops_for_native_export",
        record("counter_while"),
    )
    monkeypatch.setattr(
        normalization_module,
        "_rewrite_recurrent_ops_for_native_export",
        record("recurrent"),
    )

    rewritten = _rewrite_native_pytorch_compatibility_ops(model_ir)

    assert rewritten is model_ir
    assert calls == ["static_while", "counter_while", "recurrent"]


def test_model_op_type_collection_includes_subgraphs() -> None:
    model_ir = ModelIR(name="root")
    model_ir.operators = [OperatorIR("RELU", ["x"], ["y"])]
    subgraph = ModelIR(name="body")
    subgraph.operators = [OperatorIR("ADD", ["x", "x"], ["y"])]
    model_ir.subgraphs = [subgraph]

    assert _collect_model_op_types(model_ir) == {"ADD", "RELU"}
