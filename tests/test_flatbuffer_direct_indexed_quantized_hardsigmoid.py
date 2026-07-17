from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import onnx2tf.tflite_builder.core.model_ir_utils as lowering_module
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_dequant_hardsigmoid_quantize_bridges,
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


def _quant(scales: list[float] | None = None) -> QuantParamIR:
    values = [0.125] if scales is None else list(scales)
    return QuantParamIR(
        scale=values,
        zero_point=[0 for _ in values],
        quantized_dimension=3,
    )


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    quantization: QuantParamIR | None = None,
    data: np.ndarray | None = None,
    signature: list[int] | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        quantization=quantization,
        data=data,
        is_variable=data is None,
    )


def _rank_four_constant(seed: float) -> np.ndarray:
    return np.arange(seed, seed + 24, dtype=np.float32).reshape(1, 3, 4, 2)


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str,
    form: str,
    protection: str | None = None,
) -> None:
    names = {
        key: f"{prefix}_{key}"
        for key in [
            "source",
            "q_in",
            "float_in",
            "mul_out",
            "add_out",
            "activation_out",
            "q_out",
            "destination",
            "pre_perm",
            "post_perm",
            "mul_const",
            "add_const",
        ]
    }
    per_channel = protection == "per_channel"
    model_ir.inputs.append(names["source"])
    model_ir.outputs.append(names["destination"])
    model_ir.tensors.update(
        {
            names["source"]: _tensor(
                names["source"],
                [1, 2, 3, 4],
                dtype="INT8",
                quantization=_quant([0.125, 0.25]) if per_channel else _quant(),
                signature=[-1, 2, 3, 4],
            ),
            names["q_in"]: _tensor(
                names["q_in"],
                [1, 3, 4, 2],
                dtype="INT8",
                quantization=_quant(),
            ),
            names["float_in"]: _tensor(names["float_in"], [1, 3, 4, 2]),
            names["mul_out"]: _tensor(names["mul_out"], [1, 3, 4, 2]),
            names["add_out"]: _tensor(names["add_out"], [1, 3, 4, 2]),
            names["activation_out"]: _tensor(names["activation_out"], [1, 3, 4, 2]),
            names["q_out"]: _tensor(
                names["q_out"],
                [1, 3, 4, 2],
                dtype="INT8",
                quantization=_quant([0.25]),
            ),
            names["destination"]: _tensor(
                names["destination"],
                [1, 2, 3, 4],
                dtype="UINT8",
                quantization=_quant([0.5]),
            ),
            names["pre_perm"]: _tensor(
                names["pre_perm"],
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            names["post_perm"]: _tensor(
                names["post_perm"],
                [4],
                dtype="INT32",
                data=np.asarray(
                    [0, 2, 1, 3] if protection == "wrong_perm" else [0, 3, 1, 2],
                    dtype=np.int32,
                ),
            ),
            names["mul_const"]: _tensor(
                names["mul_const"],
                [1, 3, 4, 2],
                data=_rank_four_constant(0.0),
            ),
            names["add_const"]: _tensor(
                names["add_const"],
                [] if form == "relu" else [1, 3, 4, 2],
                data=(
                    np.asarray(0.5, dtype=np.float32)
                    if form == "relu"
                    else _rank_four_constant(100.0)
                ),
            ),
        }
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [names["source"], names["pre_perm"]],
                [names["q_in"]],
            ),
            OperatorIR("DEQUANTIZE", [names["q_in"]], [names["float_in"]]),
            OperatorIR(
                "MUL",
                [names["mul_const"], names["float_in"]],
                [names["mul_out"]],
            ),
            OperatorIR(
                "ADD",
                [names["mul_out"], names["add_const"]],
                [names["add_out"]],
            ),
        ]
    )
    if form == "relu":
        model_ir.operators.append(
            OperatorIR(
                "RELU_0_TO_1",
                [names["add_out"]],
                [names["activation_out"]],
            )
        )
    else:
        maximum_output = f"{prefix}_maximum_out"
        maximum_constant = f"{prefix}_maximum_const"
        minimum_constant = f"{prefix}_minimum_const"
        names.update(
            {
                "maximum_out": maximum_output,
                "maximum_const": maximum_constant,
                "minimum_const": minimum_constant,
            }
        )
        model_ir.tensors.update(
            {
                maximum_output: _tensor(maximum_output, [1, 3, 4, 2]),
                maximum_constant: _tensor(
                    maximum_constant,
                    [],
                    data=np.asarray(0.0, dtype=np.float32),
                ),
                minimum_constant: _tensor(
                    minimum_constant,
                    [1, 3, 4, 2],
                    data=_rank_four_constant(200.0),
                ),
            }
        )
        if protection == "missing_late_constant":
            del model_ir.tensors[minimum_constant]
        model_ir.operators.extend(
            [
                OperatorIR(
                    "MAXIMUM",
                    [maximum_constant, names["add_out"]],
                    [maximum_output],
                ),
                OperatorIR(
                    "MINIMUM",
                    [maximum_output, minimum_constant],
                    [names["activation_out"]],
                ),
            ]
        )
    model_ir.operators.extend(
        [
            OperatorIR("QUANTIZE", [names["activation_out"]], [names["q_out"]]),
            OperatorIR(
                "TRANSPOSE",
                [names["q_out"], names["post_perm"]],
                [names["destination"]],
            ),
        ]
    )

    if form == "clamp" and protection is None:
        side_output = f"{prefix}_constant_side_output"
        model_ir.tensors[side_output] = _tensor(side_output, [1, 3, 4, 2])
        model_ir.operators.append(
            OperatorIR("NEG", [names["mul_const"]], [side_output])
        )
        model_ir.outputs.append(side_output)
    if protection == "public_intermediate":
        model_ir.outputs.append(names["add_out"])
    elif protection == "public_clamp_intermediate":
        model_ir.outputs.append(names["maximum_out"])
    elif protection == "source_output":
        model_ir.outputs.append(names["source"])
    elif protection == "fanout":
        side_output = f"{prefix}_side_output"
        model_ir.tensors[side_output] = _tensor(side_output, [1, 3, 4, 2])
        model_ir.operators.append(OperatorIR("NEG", [names["float_in"]], [side_output]))
        model_ir.outputs.append(side_output)


def _make_valid_chains() -> ModelIR:
    model_ir = ModelIR("indexed_quantized_hardsigmoid")
    _add_chain(model_ir, prefix="relu_form", form="relu")
    _add_chain(model_ir, prefix="clamp_form", form="clamp")
    return model_ir


def _apply_expected_rewrite(model_ir: ModelIR, prefix: str, form: str) -> None:
    source = f"{prefix}_source"
    destination = f"{prefix}_destination"
    q_in = f"{prefix}_q_in"
    float_in = f"{prefix}_float_in"
    q_out = f"{prefix}_q_out"
    dequantize_op = next(op for op in model_ir.operators if op.outputs == [float_in])
    quantize_op = next(op for op in model_ir.operators if op.outputs == [q_out])
    pre_op = next(op for op in model_ir.operators if op.outputs == [q_in])
    post_op = next(op for op in model_ir.operators if op.outputs == [destination])

    constant_names = [f"{prefix}_mul_const", f"{prefix}_add_const"]
    if form == "clamp":
        constant_names.extend([f"{prefix}_maximum_const", f"{prefix}_minimum_const"])
    for constant_name in constant_names:
        constant_tensor = model_ir.tensors[constant_name]
        constant_data = np.asarray(constant_tensor.data)
        if constant_data.ndim != 4:
            continue
        remapped = np.transpose(constant_data, axes=[0, 3, 1, 2])
        consumer_ops = [op for op in model_ir.operators if constant_name in op.inputs]
        target_op = next(
            op
            for op in consumer_ops
            if op.op_type in {"MUL", "ADD", "MAXIMUM", "MINIMUM"}
        )
        if len(consumer_ops) == 1:
            constant_tensor.data = np.asarray(remapped)
            constant_tensor.shape = list(remapped.shape)
            constant_tensor.shape_signature = list(remapped.shape)
            continue
        new_name = f"{constant_name}_nhwc"
        model_ir.tensors[new_name] = TensorIR(
            name=new_name,
            dtype=str(constant_tensor.dtype),
            shape=list(remapped.shape),
            shape_signature=list(remapped.shape),
            data=np.asarray(remapped),
            is_variable=False,
            quantization=copy.deepcopy(constant_tensor.quantization),
        )
        lowering_module._replace_operator_input_at(
            model_ir=model_ir,
            op=target_op,
            input_index=target_op.inputs.index(constant_name),
            new_input_name=new_name,
        )

    lowering_module._set_operator_inputs(
        model_ir=model_ir,
        op=dequantize_op,
        new_inputs=[source],
    )
    lowering_module._set_operator_outputs(
        model_ir=model_ir,
        op=quantize_op,
        new_outputs=[destination],
    )
    source_tensor = model_ir.tensors[source]
    bridge_names = [
        float_in,
        f"{prefix}_mul_out",
        f"{prefix}_add_out",
        f"{prefix}_activation_out",
    ]
    if form == "clamp":
        bridge_names.append(f"{prefix}_maximum_out")
    for bridge_name in bridge_names:
        bridge_tensor = model_ir.tensors[bridge_name]
        bridge_tensor.shape = list(source_tensor.shape)
        bridge_tensor.shape_signature = list(source_tensor.shape_signature)
    destination_tensor = model_ir.tensors[destination]
    quantized_output_tensor = model_ir.tensors[q_out]
    destination_tensor.dtype = str(quantized_output_tensor.dtype)
    destination_tensor.quantization = copy.deepcopy(
        quantized_output_tensor.quantization
    )
    model_ir.operators.remove(post_op)
    model_ir.operators.remove(pre_op)


def test_indexed_hardsigmoid_matches_both_former_valid_results(monkeypatch) -> None:
    model_ir = _make_valid_chains()
    expected = copy.deepcopy(model_ir)
    _apply_expected_rewrite(expected, "relu_form", "relu")
    _apply_expected_rewrite(expected, "clamp_form", "clamp")
    lowering_module._prune_unused_tensors(expected)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    def unexpected_consumer_rescan(*args, **kwargs):
        raise AssertionError("unexpected consumer map rebuild")

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_consumer_map",
        unexpected_consumer_rescan,
    )

    stats = _optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir)

    assert refresh_count == 1
    assert stats == {"removed_transpose_dequant_hardsigmoid_quantize_bridges": 2}
    assert _normalize(model_ir) == _normalize(expected)


def test_indexed_hardsigmoid_keeps_supplied_index_current() -> None:
    model_ir = _make_valid_chains()
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _optimize_transpose_dequant_hardsigmoid_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {"removed_transpose_dequant_hardsigmoid_quantize_bridges": 2}
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type


@pytest.mark.parametrize(
    "protection",
    [
        "public_intermediate",
        "public_clamp_intermediate",
        "source_output",
        "fanout",
        "per_channel",
        "wrong_perm",
    ],
)
def test_indexed_hardsigmoid_preserves_protected_chain(protection: str) -> None:
    model_ir = ModelIR(f"protected_hardsigmoid_{protection}")
    _add_chain(
        model_ir,
        prefix="protected",
        form="clamp",
        protection=protection,
    )
    before = copy.deepcopy(model_ir)

    stats = _optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_hardsigmoid_quantize_bridges": 0}
    assert _normalize(model_ir) == _normalize(before)


def test_indexed_hardsigmoid_rejects_missing_late_constant_transactionally() -> None:
    model_ir = ModelIR("transactional_hardsigmoid_rejection")
    _add_chain(
        model_ir,
        prefix="transactional",
        form="clamp",
        protection="missing_late_constant",
    )
    before = copy.deepcopy(model_ir)

    stats = _optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_hardsigmoid_quantize_bridges": 0}
    assert _normalize(model_ir) == _normalize(before)


def test_indexed_hardsigmoid_skips_index_without_transpose(monkeypatch) -> None:
    model_ir = ModelIR("hardsigmoid_without_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1]),
        "y": _tensor("y", [1]),
    }
    model_ir.operators = [OperatorIR("RELU_0_TO_1", ["x"], ["y"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_hardsigmoid_quantize_bridges": 0}
    assert refresh_count == 0
