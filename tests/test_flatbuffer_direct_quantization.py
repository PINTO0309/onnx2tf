from __future__ import annotations

import inspect

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder import quantization
from onnx2tf.tflite_builder.quantization import (
    _elide_identity_operators,
    TensorCalibrationRange,
    build_dynamic_range_quantized_model_ir,
    build_full_integer_quantized_model_ir,
    build_integer_quantized_model_ir,
    build_full_integer_quantized_with_int16_act_model_ir,
    build_integer_quantized_with_int16_act_model_ir,
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


def test_dynamic_range_kernel_only_quantization_skips_graph_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("dynamic_range_kernel_only")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 4], [1, 4]),
        "weight": TensorIR(
            "weight",
            "FLOAT32",
            [3, 4],
            [3, 4],
            data=np.arange(12, dtype=np.float32).reshape(3, 4),
        ),
        "y": TensorIR("y", "FLOAT32", [1, 3], [1, 3]),
    }
    model_ir.operators = [
        OperatorIR("FULLY_CONNECTED", ["x", "weight"], ["y"]),
    ]
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

    assert refresh_count == 0
    assert quantized.tensors["weight"].dtype == "INT8"


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


def test_full_integer_matching_boundary_dtype_skips_graph_index(monkeypatch) -> None:
    model_ir, calibration_ranges = _strict_integer_add_fixture()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        quant_type="per-tensor",
        input_quant_dtype="int8",
        output_quant_dtype="int8",
        calibration_ranges=calibration_ranges,
    )

    assert refresh_count == 0
    assert quantized.tensors["x"].dtype == "INT8"
    assert quantized.tensors["y"].dtype == "INT8"


def test_full_integer_mixed_output_dtype_builds_one_lazy_graph_index(
    monkeypatch,
) -> None:
    model_ir, calibration_ranges = _strict_integer_add_fixture()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    quantized = build_full_integer_quantized_model_ir(
        model_ir,
        quant_type="per-tensor",
        input_quant_dtype="int8",
        output_quant_dtype="uint8",
        calibration_ranges=calibration_ranges,
    )

    assert refresh_count == 1
    assert quantized.tensors["x"].dtype == "INT8"
    assert quantized.tensors["y"].dtype == "UINT8"
    assert quantized.operators[-1].op_type == "QUANTIZE"


def _strict_integer_add_fixture() -> tuple[
    ModelIR,
    dict[str, TensorCalibrationRange],
]:
    model_ir = ModelIR("strict_integer_report")
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
    return model_ir, {
        "x": TensorCalibrationRange(-1.0, 1.0, 1),
        "y": TensorCalibrationRange(-2.0, 2.0, 1),
    }


def test_strict_integer_report_payload_is_built_only_when_requested(
    monkeypatch,
) -> None:
    model_ir, calibration_ranges = _strict_integer_add_fixture()
    calls = {"qparams": 0, "ranges": 0}
    original_qparams = quantization._quant_param_to_report
    original_range = quantization._tensor_range_to_report

    def counted_qparams(value):
        calls["qparams"] += 1
        return original_qparams(value)

    def counted_range(value):
        calls["ranges"] += 1
        return original_range(value)

    monkeypatch.setattr(quantization, "_quant_param_to_report", counted_qparams)
    monkeypatch.setattr(quantization, "_tensor_range_to_report", counted_range)

    without_report = build_integer_quantized_model_ir(
        model_ir,
        quant_type="per-tensor",
        calibration_ranges=calibration_ranges,
    )
    assert calls == {"qparams": 0, "ranges": 0}

    with_report = build_integer_quantized_model_ir(
        model_ir,
        quant_type="per-tensor",
        calibration_ranges=calibration_ranges,
        return_report=True,
    )
    assert calls["qparams"] > 0
    assert calls["ranges"] == len(calibration_ranges)
    assert with_report.report["mode"] == "integer_float_io"
    assert set(with_report.report) == {
        "mode",
        "strict",
        "supported_ops",
        "tensor_ranges",
        "quantized_tensors",
        "quantized_ops",
        "failures",
    }
    assert (
        ModelIRPassState(without_report).fingerprint()
        == ModelIRPassState(with_report.model_ir).fingerprint()
    )


def test_int16_variants_skip_discarded_report_serialization(monkeypatch) -> None:
    model_ir, calibration_ranges = _strict_integer_add_fixture()
    calls = {"qparams": 0, "ranges": 0}

    def count_qparams(value):
        calls["qparams"] += 1
        return value

    def count_range(value):
        calls["ranges"] += 1
        return value

    monkeypatch.setattr(quantization, "_quant_param_to_report", count_qparams)
    monkeypatch.setattr(quantization, "_tensor_range_to_report", count_range)

    build_integer_quantized_with_int16_act_model_ir(
        model_ir,
        quant_type="per-tensor",
        calibration_ranges=calibration_ranges,
    )
    build_full_integer_quantized_with_int16_act_model_ir(
        model_ir,
        quant_type="per-tensor",
        calibration_ranges=calibration_ranges,
    )

    assert calls == {"qparams": 0, "ranges": 0}


def test_strict_activation_dtype_reuses_boundary_name_sets() -> None:
    source = inspect.getsource(quantization._build_strict_full_integer_model_ir)

    assert source.count("set(clone.inputs)") == 1
    assert source.count("set(clone.outputs)") == 1
    assert "set(model_ir.inputs)" not in source
    assert "set(model_ir.outputs)" not in source
    assert not hasattr(quantization, "_activation_dtype_for_tensor")


def test_strict_validation_collects_used_tensors_in_its_operator_loop() -> None:
    source = inspect.getsource(quantization._validate_strict_full_integer_model_ir)

    assert "for op_idx, op in enumerate(model_ir.operators):" in source
    assert source.count("used_tensors.update(") == 2
    assert not hasattr(quantization, "_used_tensor_names")
