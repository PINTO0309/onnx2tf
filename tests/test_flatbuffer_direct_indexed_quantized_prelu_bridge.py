from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import onnx2tf.tflite_builder.lower_from_onnx2tf as lowering_module
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_dequant_mul_add_prelu_quantize_bridges,
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
    data: Any = None,
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
    reverse_binary_inputs: bool = False,
    shared_add_constant: bool = False,
    protection: str | None = None,
) -> None:
    def name(suffix: str) -> str:
        return f"{prefix}_{suffix}"

    source = name("source")
    q_input = name("q_input")
    float_input = name("float_input")
    multiply_output = name("multiply_output")
    add_output = name("add_output")
    prelu_output = name("prelu_output")
    q_output = name("q_output")
    destination = name("destination")
    multiply_constant = name("multiply_constant")
    add_constant = name("add_constant")
    alpha = name("alpha")
    pre_perm = name("pre_perm")
    post_perm = name("post_perm")
    per_channel = protection == "per_channel"

    model_ir.inputs.append(source)
    model_ir.outputs.append(destination)
    model_ir.tensors.update(
        {
            source: _tensor(
                source,
                [1, 2, 3, 4],
                dtype="INT8",
                quantization=_quant([0.125, 0.25]) if per_channel else _quant(),
                signature=[-1, 2, 3, 4],
            ),
            q_input: _tensor(
                q_input,
                [1, 3, 4, 2],
                dtype="INT8",
                quantization=_quant(),
            ),
            float_input: _tensor(float_input, [1, 3, 4, 2]),
            multiply_output: _tensor(multiply_output, [1, 3, 4, 2]),
            add_output: _tensor(add_output, [1, 3, 4, 2]),
            prelu_output: _tensor(prelu_output, [1, 3, 4, 2]),
            q_output: _tensor(
                q_output,
                [1, 3, 4, 2],
                dtype="INT8",
                quantization=_quant([0.25]),
            ),
            destination: _tensor(
                destination,
                [1, 2, 3, 4],
                dtype="UINT8",
                quantization=_quant([0.5]),
            ),
            multiply_constant: _tensor(
                multiply_constant,
                [] if reverse_binary_inputs else [1, 3, 4, 2],
                data=(
                    np.asarray(0.25, dtype=np.float32)
                    if reverse_binary_inputs
                    else _rank_four_constant(0.0)
                ),
            ),
            add_constant: _tensor(
                add_constant,
                [1, 3, 4, 2] if reverse_binary_inputs else [],
                data=(
                    _rank_four_constant(100.0)
                    if reverse_binary_inputs
                    else np.asarray(0.5, dtype=np.float32)
                ),
                quantization=_quant([0.75]) if shared_add_constant else None,
            ),
            alpha: _tensor(
                alpha,
                [] if reverse_binary_inputs else [1, 3, 4, 2],
                data=(
                    np.asarray(0.1, dtype=np.float32)
                    if reverse_binary_inputs
                    else _rank_four_constant(200.0)
                ),
            ),
            pre_perm: _tensor(
                pre_perm,
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            post_perm: _tensor(
                post_perm,
                [4],
                dtype="INT32",
                data=np.asarray(
                    [0, 2, 1, 3] if protection == "wrong_perm" else [0, 3, 1, 2],
                    dtype=np.int32,
                ),
            ),
        }
    )
    if protection == "non_array_alpha":
        model_ir.tensors[alpha].data = [0.1]
    elif protection == "missing_alpha":
        del model_ir.tensors[alpha]

    multiply_inputs = (
        [float_input, multiply_constant]
        if reverse_binary_inputs
        else [multiply_constant, float_input]
    )
    add_inputs = (
        [add_constant, multiply_output]
        if reverse_binary_inputs
        else [multiply_output, add_constant]
    )
    model_ir.operators.extend(
        [
            OperatorIR("TRANSPOSE", [source, pre_perm], [q_input]),
            OperatorIR("DEQUANTIZE", [q_input], [float_input]),
            OperatorIR("MUL", multiply_inputs, [multiply_output]),
            OperatorIR("ADD", add_inputs, [add_output]),
            OperatorIR("PRELU", [add_output, alpha], [prelu_output]),
            OperatorIR("QUANTIZE", [prelu_output], [q_output]),
            OperatorIR("TRANSPOSE", [q_output, post_perm], [destination]),
        ]
    )

    if shared_add_constant:
        side_output = name("constant_side_output")
        model_ir.tensors[side_output] = _tensor(side_output, [1, 3, 4, 2])
        model_ir.operators.append(OperatorIR("NEG", [add_constant], [side_output]))
        model_ir.outputs.append(side_output)
    if protection == "public_intermediate":
        model_ir.outputs.append(add_output)
    elif protection == "public_prelu_output":
        model_ir.outputs.append(prelu_output)
    elif protection == "source_output":
        model_ir.outputs.append(source)
    elif protection == "fanout":
        side_output = name("side_output")
        model_ir.tensors[side_output] = _tensor(side_output, [1, 3, 4, 2])
        model_ir.operators.append(OperatorIR("NEG", [float_input], [side_output]))
        model_ir.outputs.append(side_output)


def _make_valid_chains() -> ModelIR:
    model_ir = ModelIR("indexed_quantized_prelu_bridge")
    _add_chain(model_ir, prefix="private")
    _add_chain(
        model_ir,
        prefix="shared",
        reverse_binary_inputs=True,
        shared_add_constant=True,
    )
    return model_ir


def _apply_expected_rewrite(model_ir: ModelIR, prefix: str) -> None:
    def name(suffix: str) -> str:
        return f"{prefix}_{suffix}"

    source = name("source")
    destination = name("destination")
    q_input = name("q_input")
    float_input = name("float_input")
    q_output = name("q_output")
    dequantize_op = next(op for op in model_ir.operators if op.outputs == [float_input])
    quantize_op = next(op for op in model_ir.operators if op.outputs == [q_output])
    pre_op = next(op for op in model_ir.operators if op.outputs == [q_input])
    post_op = next(op for op in model_ir.operators if op.outputs == [destination])

    for constant_name in [
        name("multiply_constant"),
        name("add_constant"),
        name("alpha"),
    ]:
        constant_tensor = model_ir.tensors[constant_name]
        constant_data = np.asarray(constant_tensor.data)
        if constant_data.ndim != 4:
            continue
        remapped = np.transpose(constant_data, axes=[0, 3, 1, 2])
        consumers = [op for op in model_ir.operators if constant_name in op.inputs]
        target_op = next(
            op for op in consumers if op.op_type in {"MUL", "ADD", "PRELU"}
        )
        if len(consumers) == 1:
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
    for bridge_name in [
        float_input,
        name("multiply_output"),
        name("add_output"),
        name("prelu_output"),
    ]:
        bridge_tensor = model_ir.tensors[bridge_name]
        bridge_tensor.shape = list(source_tensor.shape)
        bridge_tensor.shape_signature = list(source_tensor.shape_signature)
    destination_tensor = model_ir.tensors[destination]
    quantized_output_tensor = model_ir.tensors[q_output]
    destination_tensor.dtype = str(quantized_output_tensor.dtype)
    destination_tensor.quantization = copy.deepcopy(
        quantized_output_tensor.quantization
    )
    model_ir.operators.remove(post_op)
    model_ir.operators.remove(pre_op)


def test_indexed_quantized_prelu_matches_both_former_valid_results(
    monkeypatch,
) -> None:
    model_ir = _make_valid_chains()
    expected = copy.deepcopy(model_ir)
    _apply_expected_rewrite(expected, "private")
    _apply_expected_rewrite(expected, "shared")
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

    stats = _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(model_ir)

    assert refresh_count == 1
    assert stats == {"removed_transpose_dequant_mul_add_prelu_quantize_bridges": 2}
    assert _normalize(model_ir) == _normalize(expected)


def test_indexed_quantized_prelu_keeps_supplied_index_current() -> None:
    model_ir = _make_valid_chains()
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {"removed_transpose_dequant_mul_add_prelu_quantize_bridges": 2}
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
        "public_prelu_output",
        "source_output",
        "fanout",
        "per_channel",
        "wrong_perm",
        "non_array_alpha",
        "missing_alpha",
    ],
)
def test_indexed_quantized_prelu_preserves_protected_chain(
    protection: str,
) -> None:
    model_ir = ModelIR(f"protected_quantized_prelu_{protection}")
    _add_chain(
        model_ir,
        prefix="protected",
        protection=protection,
    )
    before = copy.deepcopy(model_ir)

    stats = _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_mul_add_prelu_quantize_bridges": 0}
    assert _normalize(model_ir) == _normalize(before)


def test_indexed_quantized_prelu_skips_index_without_transpose(
    monkeypatch,
) -> None:
    model_ir = ModelIR("quantized_prelu_without_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1]),
        "alpha": _tensor("alpha", [], data=np.asarray(0.1, dtype=np.float32)),
        "y": _tensor("y", [1]),
    }
    model_ir.operators = [OperatorIR("PRELU", ["x", "alpha"], ["y"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_mul_add_prelu_quantize_bridges": 0}
    assert refresh_count == 0
