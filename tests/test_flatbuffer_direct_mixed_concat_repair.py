from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_mixed_nhwc_inputs_for_nchw_concat,
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
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT8" if quantization is not None else "FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        is_variable=True,
        quantization=quantization,
    )


def _make_mixed_concat_model_ir(
    *,
    include_second_nhwc: bool = False,
    quantized: bool = False,
) -> ModelIR:
    model_ir = ModelIR("mixed_nhwc_inputs_for_nchw_concat")
    nhwc_quantization = (
        QuantParamIR(
            scale=[0.125, 0.25, 0.5, 1.0],
            zero_point=[0, 0, 0, 0],
            quantized_dimension=3,
        )
        if quantized
        else None
    )
    model_ir.tensors = {
        "nhwc0": _tensor(
            "nhwc0",
            [1, 4, 5, 4],
            quantization=nhwc_quantization,
        ),
        "nchw0": _tensor("nchw0", [1, 3, 4, 5]),
        "nchw1": _tensor("nchw1", [1, 2, 4, 5]),
        "concat_out": _tensor("concat_out", [1, 13, 4, 5]),
    }
    inputs = ["nhwc0", "nchw0", "nchw1"]
    if include_second_nhwc:
        model_ir.tensors["nhwc1"] = _tensor("nhwc1", [1, 4, 5, 6])
        inputs.insert(1, "nhwc1")
    model_ir.inputs = list(inputs)
    model_ir.outputs = ["concat_out"]
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            list(inputs),
            ["concat_out"],
            {"axis": 1, "fusedActivationFunction": "NONE"},
        )
    ]
    return model_ir


def test_mixed_concat_repair_inserts_local_adapter_and_is_idempotent() -> None:
    model_ir = _make_mixed_concat_model_ir()

    stats = _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir)

    assert stats == {"repaired_mixed_nhwc_inputs_for_nchw_concat": 1}
    assert [str(op.op_type) for op in model_ir.operators] == [
        "TRANSPOSE",
        "CONCATENATION",
    ]
    concat_op = model_ir.operators[1]
    adapter_name = str(concat_op.inputs[0])
    assert adapter_name == "nhwc0_nchw_concat_adapter"
    assert model_ir.tensors[adapter_name].shape == [1, 4, 4, 5]
    assert model_ir.tensors["concat_out"].shape == [1, 9, 4, 5]

    after_first = _normalize(copy.deepcopy(model_ir))
    second_stats = _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir)
    assert second_stats == {"repaired_mixed_nhwc_inputs_for_nchw_concat": 0}
    assert _normalize(model_ir) == after_first


def test_mixed_concat_repair_uses_two_input_output_contract() -> None:
    model_ir = ModelIR("two_input_mixed_concat")
    model_ir.inputs = ["nchw", "nhwc"]
    model_ir.outputs = ["concat_out"]
    model_ir.tensors = {
        "nchw": _tensor("nchw", [1, 3, 4, 5]),
        "nhwc": _tensor("nhwc", [1, 4, 5, 2]),
        "concat_out": _tensor("concat_out", [1, 5, 4, 5]),
    }
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["nchw", "nhwc"],
            ["concat_out"],
            {"axis": 1},
        )
    ]

    stats = _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir)

    assert stats == {"repaired_mixed_nhwc_inputs_for_nchw_concat": 1}
    concat_op = model_ir.operators[-1]
    assert concat_op.inputs[0] == "nchw"
    assert concat_op.inputs[1] != "nhwc"
    assert model_ir.tensors[concat_op.inputs[1]].shape == [1, 2, 4, 5]
    assert model_ir.tensors["concat_out"].shape == [1, 5, 4, 5]


@pytest.mark.parametrize(
    "mutation",
    ["wrong_axis", "missing_input", "invalid_rank", "no_nhwc_input"],
)
def test_mixed_concat_repair_rejection_is_complete_noop(mutation: str) -> None:
    model_ir = _make_mixed_concat_model_ir()
    concat_op = model_ir.operators[0]
    if mutation == "wrong_axis":
        concat_op.options["axis"] = 3
    elif mutation == "missing_input":
        model_ir.tensors.pop("nhwc0")
    elif mutation == "invalid_rank":
        model_ir.tensors["nhwc0"].shape = [1, 4, 5]
        model_ir.tensors["nhwc0"].shape_signature = [1, 4, 5]
    elif mutation == "no_nhwc_input":
        model_ir.tensors["nhwc0"].shape = [1, 4, 4, 5]
        model_ir.tensors["nhwc0"].shape_signature = [1, 4, 4, 5]
    before = _normalize(copy.deepcopy(model_ir))

    stats = _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir)

    assert stats == {"repaired_mixed_nhwc_inputs_for_nchw_concat": 0}
    assert _normalize(model_ir) == before


@pytest.mark.xfail(
    strict=True,
    reason="a later malformed NHWC signature is detected after the first adapter insertion",
)
def test_mixed_concat_repair_prevalidates_all_source_signatures_atomically() -> None:
    model_ir = _make_mixed_concat_model_ir(include_second_nhwc=True)
    model_ir.tensors["nhwc1"].shape_signature = [1, 4]
    before = _normalize(copy.deepcopy(model_ir))

    stats = _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir)

    assert stats == {"repaired_mixed_nhwc_inputs_for_nchw_concat": 0}
    assert _normalize(model_ir) == before


@pytest.mark.xfail(
    strict=True,
    reason="the required Concat output tensor is resolved after adapter insertion",
)
def test_mixed_concat_repair_requires_output_tensor_before_mutation() -> None:
    model_ir = _make_mixed_concat_model_ir()
    model_ir.tensors.pop("concat_out")
    before = _normalize(copy.deepcopy(model_ir))

    stats = _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir)

    assert stats == {"repaired_mixed_nhwc_inputs_for_nchw_concat": 0}
    assert _normalize(model_ir) == before


@pytest.mark.xfail(
    strict=True,
    reason="the cloned per-axis quantization dimension is not remapped for NHWC-to-NCHW",
)
def test_mixed_concat_repair_remaps_per_axis_quantized_dimension() -> None:
    model_ir = _make_mixed_concat_model_ir(quantized=True)

    stats = _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir)

    assert stats == {"repaired_mixed_nhwc_inputs_for_nchw_concat": 1}
    concat_op = model_ir.operators[-1]
    adapter_tensor = model_ir.tensors[str(concat_op.inputs[0])]
    assert adapter_tensor.quantization is not None
    assert adapter_tensor.quantization.quantized_dimension == 1
