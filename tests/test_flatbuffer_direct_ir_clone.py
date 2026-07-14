from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
    clone_model_ir_with_float16,
    clone_model_ir_with_float32,
)
from onnx2tf.tflite_builder.quantization import _clone_model_ir


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_model_ir() -> ModelIR:
    model_ir = ModelIR(
        name="clone_contract",
        description="clone contract fixture",
        metadata={"nested": {"value": 1}},
    )
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
        data=np.asarray([1.0], dtype=np.float32),
        is_variable=True,
        quantization=QuantParamIR(
            scale=[0.25],
            zero_point=[1],
            quantized_dimension=0,
            min=[-1.0],
            max=[1.0],
        ),
        logical_layout="invalid-logical-layout",
        physical_layout="invalid-physical-layout",
        onnx_tensor_name="onnx_x",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT16",
        shape=[1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="CAST",
            inputs=["x"],
            outputs=["y"],
            options={"outDataType": "FLOAT16", "nested": {"value": "FLOAT16"}},
            axis_semantics={"axis": "logical"},
            version=3,
            onnx_node_name="CastNode",
            onnx_op_type="Cast",
        )
    ]
    subgraph = ModelIR(name="body", metadata={"body": True})
    subgraph.tensors["body_value"] = TensorIR(
        name="body_value",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray([2.0], dtype=np.float32),
    )
    model_ir.subgraphs = [subgraph]
    return model_ir


def test_float16_clone_preserves_existing_precision_contract() -> None:
    source = _make_model_ir()
    clone = clone_model_ir_with_float16(source)

    assert clone.metadata == source.metadata
    assert clone.metadata is not source.metadata
    assert clone.metadata["nested"] is source.metadata["nested"]
    assert clone.subgraphs[0] is not source.subgraphs[0]
    assert clone.subgraphs[0].tensors["body_value"].dtype == "FLOAT16"
    assert clone.tensors["x"].dtype == "FLOAT16"
    assert clone.tensors["y"].dtype == "FLOAT16"
    assert clone.tensors["x"].logical_layout == LOGICAL_LAYOUT_UNKNOWN
    assert clone.tensors["x"].physical_layout == LOGICAL_LAYOUT_UNKNOWN
    assert clone.tensors["x"].data is not source.tensors["x"].data
    assert clone.tensors["x"].quantization is not source.tensors["x"].quantization
    assert clone.operators[0].options is not source.operators[0].options
    assert clone.operators[0].options["nested"] is source.operators[0].options["nested"]


def test_float32_clone_preserves_promotion_and_recursive_option_rewrite() -> None:
    source = _make_model_ir()
    clone = clone_model_ir_with_float32(source)

    assert clone.tensors["x"].dtype == "FLOAT32"
    assert clone.tensors["y"].dtype == "FLOAT32"
    assert clone.subgraphs[0].tensors["body_value"].dtype == "FLOAT32"
    assert clone.operators[0].options == {
        "outDataType": "FLOAT32",
        "nested": {"value": "FLOAT32"},
    }
    assert (
        clone.operators[0].options["nested"]
        is not source.operators[0].options["nested"]
    )
    assert source.operators[0].options["nested"]["value"] == "FLOAT16"


def test_quantization_clone_preserves_legacy_root_only_raw_layout_contract() -> None:
    source = _make_model_ir()
    clone = _clone_model_ir(source)

    assert clone.metadata == {}
    assert clone.subgraphs == []
    assert clone.tensors["x"].dtype == "FLOAT32"
    assert clone.tensors["x"].logical_layout == "invalid-logical-layout"
    assert clone.tensors["x"].physical_layout == "invalid-physical-layout"
    assert clone.tensors["x"].data is not source.tensors["x"].data
    assert clone.tensors["x"].quantization is not source.tensors["x"].quantization
    assert clone.operators[0].options is not source.operators[0].options
    assert clone.operators[0].options["nested"] is source.operators[0].options["nested"]


def test_quantization_clone_delegates_element_copy_contracts_to_ir_module() -> None:
    quantization_tree = ast.parse(
        (REPO_ROOT / "onnx2tf" / "tflite_builder" / "quantization.py").read_text(
            encoding="utf-8"
        )
    )
    clone_function = next(
        node
        for node in quantization_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_clone_model_ir"
    )
    called_names = {
        node.func.id
        for node in ast.walk(clone_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert {"clone_operator_ir", "clone_tensor_ir"} <= called_names
    assert "OperatorIR" not in called_names
    assert "TensorIR" not in called_names
