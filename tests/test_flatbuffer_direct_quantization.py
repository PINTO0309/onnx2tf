from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.quantization import (
    _elide_identity_operators,
    TensorCalibrationRange,
    build_dynamic_range_quantized_model_ir,
    build_integer_quantized_model_ir,
)


def test_dynamic_range_quantization_inserts_one_shared_dequantize(monkeypatch) -> None:
    model_ir = ModelIR("dynamic_range_shared_constant")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 4], [1, 4]),
        "constant": TensorIR(
            "constant",
            "FLOAT32",
            [4],
            [4],
            data=np.asarray([0.25, -0.5, 1.0, -2.0], dtype=np.float32),
        ),
        "mid": TensorIR("mid", "FLOAT32", [1, 4], [1, 4]),
        "out": TensorIR("out", "FLOAT32", [1, 4], [1, 4]),
    }
    first_add = OperatorIR("ADD", ["x", "constant"], ["mid"])
    second_add = OperatorIR("ADD", ["mid", "constant"], ["out"])
    model_ir.operators = [first_add, second_add]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    quantized = build_dynamic_range_quantized_model_ir(
        model_ir,
        quant_type="per-tensor",
    )

    assert refresh_count == 1
    assert [op.op_type for op in quantized.operators] == [
        "DEQUANTIZE",
        "ADD",
        "ADD",
    ]
    dequantized_name = str(quantized.operators[0].outputs[0])
    assert quantized.operators[1].inputs == ["x", dequantized_name]
    assert quantized.operators[2].inputs == ["mid", dequantized_name]
    assert quantized.tensors["constant"].dtype == "INT8"
    assert quantized.tensors["constant"].quantization is not None
    assert model_ir.tensors["constant"].dtype == "FLOAT32"
    assert model_ir.operators == [first_add, second_add]


def test_quantization_identity_elision_promotes_graph_output_producer(
    monkeypatch,
) -> None:
    model_ir = ModelIR("identity_output")
    model_ir.inputs = ["x", "bias"]
    model_ir.outputs = ["y"]
    add_op = OperatorIR("ADD", ["x", "bias"], ["mid"])
    identity_op = OperatorIR("IDENTITY", ["mid"], ["y"])
    model_ir.operators = [add_op, identity_op]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    _elide_identity_operators(model_ir)

    assert refresh_count == 1
    assert model_ir.operators == [add_op]
    assert add_op.outputs == ["y"]
    assert model_ir.outputs == ["y"]


def test_quantization_identity_elision_resolves_boundary_chain() -> None:
    model_ir = ModelIR("identity_input_chain")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.operators = [
        OperatorIR("IDENTITY", ["x"], ["mid"]),
        OperatorIR("IDENTITY", ["mid"], ["y"]),
    ]

    _elide_identity_operators(model_ir)

    assert model_ir.operators == []
    assert model_ir.outputs == ["x"]


def test_strict_integer_float_io_inserts_indexed_boundary_ops(monkeypatch) -> None:
    model_ir = ModelIR("strict_integer_boundary")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 4], [1, 4]),
        "constant": TensorIR(
            "constant",
            "FLOAT32",
            [4],
            [4],
            data=np.asarray([0.25, -0.5, 1.0, -2.0], dtype=np.float32),
        ),
        "y": TensorIR("y", "FLOAT32", [1, 4], [1, 4]),
    }
    model_ir.operators = [OperatorIR("ADD", ["x", "constant"], ["y"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    quantized = build_integer_quantized_model_ir(
        model_ir,
        quant_type="per-tensor",
        calibration_ranges={
            "x": TensorCalibrationRange(-1.0, 1.0, 1),
            "y": TensorCalibrationRange(-2.0, 2.0, 1),
        },
    )

    assert refresh_count == 1
    assert [op.op_type for op in quantized.operators] == [
        "QUANTIZE",
        "ADD",
        "DEQUANTIZE",
    ]
    assert quantized.operators[1].inputs[0] == quantized.operators[0].outputs[0]
    assert quantized.operators[1].outputs[0] == quantized.operators[2].inputs[0]
    assert quantized.outputs == quantized.operators[2].outputs
    assert model_ir.operators[0].inputs == ["x", "constant"]
    assert model_ir.operators[0].outputs == ["y"]
