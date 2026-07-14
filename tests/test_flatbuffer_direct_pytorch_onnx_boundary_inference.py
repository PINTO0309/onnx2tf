from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_onnx_artifact_support import (
    _infer_batchless_rank3_image_boundaries_from_onnx_graph,
    _infer_public_layouts_from_onnx_graph,
    _is_onnx_boundary_layout_passthrough_node,
    _merge_reference_public_boundary_metadata,
    _read_onnx_transpose_perm,
)


def test_onnx_transpose_perm_and_passthrough_classification() -> None:
    transpose = helper.make_node("Transpose", ["x"], ["y"], perm=[0, 3, 1, 2])
    relu = helper.make_node("Relu", ["x"], ["y"])
    conv = helper.make_node("Conv", ["x", "w"], ["y"])

    assert _read_onnx_transpose_perm(transpose) == [0, 3, 1, 2]
    assert _read_onnx_transpose_perm(relu) is None
    assert _is_onnx_boundary_layout_passthrough_node(
        node=relu,
        source_tensor_name="x",
    )
    assert not _is_onnx_boundary_layout_passthrough_node(
        node=conv,
        source_tensor_name="x",
    )


def _layout_boundary_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["x_id"]),
            helper.make_node("Transpose", ["x_id"], ["core"], perm=[0, 3, 1, 2]),
            helper.make_node("Transpose", ["core"], ["y_id"], perm=[0, 2, 3, 1]),
            helper.make_node("Identity", ["y_id"], ["y"]),
        ],
        "layout_boundaries",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 8, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 8, 3])],
    )
    return helper.make_model(graph)


def test_public_layout_inference_walks_short_passthrough_boundary_chains() -> None:
    assert _infer_public_layouts_from_onnx_graph(_layout_boundary_model()) == {
        "x": "NHWC",
        "y": "NHWC",
    }


def test_public_layout_inference_rejects_input_fanout() -> None:
    model = _layout_boundary_model()
    model.graph.node.append(helper.make_node("Relu", ["x"], ["side"]))

    inferred = _infer_public_layouts_from_onnx_graph(model)

    assert "x" not in inferred
    assert inferred["y"] == "NHWC"


def _batchless_boundary_model() -> onnx.ModelProto:
    graph = helper.make_graph(
        [
            helper.make_node("Relu", ["image"], ["image_relu"]),
            helper.make_node("Unsqueeze", ["image_relu"], ["batched"], axes=[0]),
            helper.make_node("Squeeze", ["core"], ["squeezed"], axes=[0]),
            helper.make_node("Identity", ["squeezed"], ["image_out"]),
        ],
        "batchless_boundaries",
        [
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [3, 8, 8]),
            helper.make_tensor_value_info("core", TensorProto.FLOAT, [1, 3, 8, 8]),
        ],
        [helper.make_tensor_value_info("image_out", TensorProto.FLOAT, [3, 8, 8])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])


def test_batchless_rank3_boundary_inference_handles_input_and_output_chains() -> None:
    assert _infer_batchless_rank3_image_boundaries_from_onnx_graph(
        _batchless_boundary_model()
    ) == {"image", "image_out"}


def test_boundary_inference_returns_empty_without_graph() -> None:
    assert _infer_public_layouts_from_onnx_graph(None) == {}
    assert _infer_batchless_rank3_image_boundaries_from_onnx_graph(None) == set()


def _rank2_boundary_model_ir(name: str) -> ModelIR:
    model_ir = ModelIR(name=name, inputs=["x"], outputs=["y"])
    model_ir.tensors = {
        "x": TensorIR(
            name="x",
            dtype="FLOAT32",
            shape=[1, 4],
            shape_signature=[-1, 4],
        ),
        "y": TensorIR(
            name="y",
            dtype="FLOAT32",
            shape=[1, 4],
            shape_signature=[-1, 4],
        ),
    }
    return model_ir


def test_reference_boundary_merge_restores_public_shape_contract() -> None:
    reference = _rank2_boundary_model_ir("reference")
    reference.metadata = {
        "onnx_boundary_shape_signature_map": {"x": [-1, 4], "y": [-1, 4]},
        "onnx_public_layout_map": {"x": "UNKNOWN", "y": "UNKNOWN"},
    }
    imported = _rank2_boundary_model_ir("imported")
    imported.inputs = ["old_input"]
    imported.outputs = ["old_output"]

    _merge_reference_public_boundary_metadata(
        imported_model_ir=imported,
        reference_model_ir=reference,
    )

    assert imported.inputs == ["x"]
    assert imported.outputs == ["y"]
    assert imported.tensors["x"].shape_signature == [-1, 4]
    assert imported.metadata["onnx_boundary_shape_signature_map"] == {
        "x": [-1, 4],
        "y": [-1, 4],
    }
    assert imported.metadata["batchless_rank3_public_boundary_names"] == []


def test_reference_boundary_merge_forces_recurrent_rank3_feature_last() -> None:
    reference = ModelIR(name="recurrent", inputs=["x"], outputs=["y"])
    imported = ModelIR(name="imported", inputs=["x"], outputs=["y"])
    for model_ir in (reference, imported):
        model_ir.tensors = {
            "x": TensorIR(
                name="x", dtype="FLOAT32", shape=[1, 2, 4], logical_layout="NWC"
            ),
            "y": TensorIR(
                name="y", dtype="FLOAT32", shape=[1, 2, 4], logical_layout="NWC"
            ),
        }
    reference.operators.append(
        OperatorIR(
            op_type="UNIDIRECTIONAL_SEQUENCE_LSTM",
            inputs=["x"],
            outputs=["y"],
        )
    )

    _merge_reference_public_boundary_metadata(
        imported_model_ir=imported,
        reference_model_ir=reference,
    )

    assert imported.metadata["onnx_public_layout_map"] == {"x": "NWC", "y": "NWC"}
    assert imported.tensors["x"].logical_layout == "NWC"
    assert imported.tensors["y"].logical_layout == "NWC"
