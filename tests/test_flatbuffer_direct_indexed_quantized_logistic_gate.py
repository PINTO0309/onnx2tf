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
    _optimize_transpose_dequant_logistic_mul_quantize_bridges,
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


def _add_gate(
    model_ir: ModelIR,
    *,
    prefix: str,
    post_count: int,
    reverse_multiply_inputs: bool = False,
    protection: str | None = None,
) -> None:
    def name(suffix: str) -> str:
        return f"{prefix}_{suffix}"

    source = name("source")
    transposed_input = name("transposed_input")
    dequantized_logistic = name("dequantized_logistic")
    logistic_output = name("logistic_output")
    quantized_gate = name("quantized_gate")
    dequantized_gate = name("dequantized_gate")
    dequantized_data = name("dequantized_data")
    multiply_output = name("multiply_output")
    quantized_multiply = name("quantized_multiply")
    pre_perm = name("pre_perm")
    post_perm = name("post_perm")
    per_channel = protection == "per_channel"

    model_ir.inputs.append(source)
    model_ir.tensors.update(
        {
            source: _tensor(
                source,
                [1, 2, 2, 3],
                dtype="INT8",
                quantization=_quant([0.125, 0.25]) if per_channel else _quant(),
                signature=[-1, 2, 2, 3],
            ),
            transposed_input: _tensor(
                transposed_input,
                [1, 3, 2, 2],
                dtype="INT8",
                quantization=_quant(),
            ),
            dequantized_logistic: _tensor(dequantized_logistic, [1, 3, 2, 2]),
            logistic_output: _tensor(logistic_output, [1, 3, 2, 2]),
            quantized_gate: _tensor(
                quantized_gate,
                [1, 3, 2, 2],
                dtype="INT8",
                quantization=_quant([0.25]),
            ),
            dequantized_gate: _tensor(dequantized_gate, [1, 3, 2, 2]),
            dequantized_data: _tensor(dequantized_data, [1, 3, 2, 2]),
            multiply_output: _tensor(multiply_output, [1, 3, 2, 2]),
            quantized_multiply: _tensor(
                quantized_multiply,
                [1, 3, 2, 2],
                dtype="INT8",
                quantization=_quant([0.375]),
            ),
            pre_perm: _tensor(
                pre_perm,
                [4],
                dtype="INT32",
                data=np.asarray([0, 3, 1, 2], dtype=np.int32),
            ),
            post_perm: _tensor(
                post_perm,
                [4],
                dtype="INT32",
                data=np.asarray(
                    [0, 3, 1, 2] if protection == "wrong_post_perm" else [0, 2, 3, 1],
                    dtype=np.int32,
                ),
            ),
        }
    )
    multiply_inputs = (
        [dequantized_gate, dequantized_data]
        if reverse_multiply_inputs
        else [dequantized_data, dequantized_gate]
    )
    model_ir.operators.extend(
        [
            OperatorIR("TRANSPOSE", [source, pre_perm], [transposed_input]),
            OperatorIR("DEQUANTIZE", [transposed_input], [dequantized_logistic]),
            OperatorIR("LOGISTIC", [dequantized_logistic], [logistic_output]),
            OperatorIR("QUANTIZE", [logistic_output], [quantized_gate]),
            OperatorIR("DEQUANTIZE", [quantized_gate], [dequantized_gate]),
            OperatorIR("DEQUANTIZE", [transposed_input], [dequantized_data]),
            OperatorIR("MUL", multiply_inputs, [multiply_output]),
            OperatorIR("QUANTIZE", [multiply_output], [quantized_multiply]),
        ]
    )

    post_outputs: list[str] = []
    for index in range(post_count):
        post_output = name(f"post_{index}")
        graph_output = name(f"output_{index}")
        post_outputs.append(post_output)
        model_ir.tensors[post_output] = _tensor(
            post_output,
            [1, 2, 2, 3],
            dtype="UINT8",
            quantization=_quant([0.75]),
        )
        model_ir.tensors[graph_output] = _tensor(graph_output, [1, 2, 2, 3])
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    [quantized_multiply, post_perm],
                    [post_output],
                ),
                OperatorIR("RELU", [post_output], [graph_output]),
            ]
        )
        model_ir.outputs.append(graph_output)

    public_intermediate = {
        "public_dequantized_data": dequantized_data,
        "public_dequantized_logistic": dequantized_logistic,
        "public_logistic": logistic_output,
        "public_quantized_gate": quantized_gate,
        "public_dequantized_gate": dequantized_gate,
    }.get(protection)
    if public_intermediate is not None:
        model_ir.outputs.append(public_intermediate)
    elif protection == "public_source":
        model_ir.outputs.append(source)
    elif protection == "public_post_alias":
        model_ir.outputs.append(post_outputs[0])
    elif protection in {"pre_fanout", "data_fanout", "gate_fanout"}:
        side_input = {
            "pre_fanout": transposed_input,
            "data_fanout": dequantized_data,
            "gate_fanout": quantized_gate,
        }[protection]
        side_output = name("side_output")
        model_ir.tensors[side_output] = _tensor(side_output, [1, 3, 2, 2])
        model_ir.operators.append(OperatorIR("NEG", [side_input], [side_output]))
        model_ir.outputs.append(side_output)
    elif protection == "non_transpose_post_user":
        side_output = name("side_output")
        model_ir.tensors[side_output] = _tensor(
            side_output,
            [1, 3, 2, 2],
            dtype="INT8",
            quantization=_quant(),
        )
        model_ir.operators.append(
            OperatorIR("RELU", [quantized_multiply], [side_output])
        )
        model_ir.outputs.append(side_output)


def _make_valid_gates() -> ModelIR:
    model_ir = ModelIR("indexed_quantized_logistic_gate")
    _add_gate(model_ir, prefix="single", post_count=1)
    _add_gate(
        model_ir,
        prefix="multi",
        post_count=2,
        reverse_multiply_inputs=True,
    )
    return model_ir


def _apply_expected_rewrite(model_ir: ModelIR, prefix: str) -> None:
    def name(suffix: str) -> str:
        return f"{prefix}_{suffix}"

    source = name("source")
    transposed_input = name("transposed_input")
    dequantized_logistic = name("dequantized_logistic")
    logistic_output = name("logistic_output")
    quantized_gate = name("quantized_gate")
    dequantized_gate = name("dequantized_gate")
    dequantized_data = name("dequantized_data")
    multiply_output = name("multiply_output")
    quantized_multiply = name("quantized_multiply")
    pre_op = next(op for op in model_ir.operators if op.outputs == [transposed_input])
    dequantize_logistic_op = next(
        op for op in model_ir.operators if op.outputs == [dequantized_logistic]
    )
    dequantize_data_op = next(
        op for op in model_ir.operators if op.outputs == [dequantized_data]
    )
    quantize_multiply_op = next(
        op for op in model_ir.operators if op.outputs == [quantized_multiply]
    )
    post_ops = [
        op
        for op in model_ir.operators
        if op.op_type == "TRANSPOSE" and op.inputs[0] == quantized_multiply
    ]
    post_outputs = [str(op.outputs[0]) for op in post_ops]

    lowering_module._set_operator_inputs(
        model_ir=model_ir,
        op=dequantize_data_op,
        new_inputs=[source],
    )
    lowering_module._set_operator_inputs(
        model_ir=model_ir,
        op=dequantize_logistic_op,
        new_inputs=[source],
    )
    for tensor_name in [
        dequantized_data,
        dequantized_logistic,
        logistic_output,
        quantized_gate,
        dequantized_gate,
        multiply_output,
        quantized_multiply,
    ]:
        lowering_module._permute_tensor_metadata_if_rank_matches(
            model_ir.tensors[tensor_name], [0, 2, 3, 1]
        )

    canonical_output = post_outputs[0]
    lowering_module._set_operator_outputs(
        model_ir=model_ir,
        op=quantize_multiply_op,
        new_outputs=[canonical_output],
    )
    for alias_output in post_outputs[1:]:
        lowering_module._replace_tensor_inputs(model_ir, alias_output, canonical_output)
    canonical_tensor = model_ir.tensors[canonical_output]
    quantized_multiply_tensor = model_ir.tensors[quantized_multiply]
    canonical_tensor.dtype = str(quantized_multiply_tensor.dtype)
    canonical_tensor.quantization = copy.deepcopy(
        quantized_multiply_tensor.quantization
    )
    canonical_tensor.shape = list(quantized_multiply_tensor.shape)
    canonical_tensor.shape_signature = list(quantized_multiply_tensor.shape_signature)
    for op in [pre_op, *post_ops]:
        model_ir.operators.remove(op)


def test_indexed_logistic_gate_matches_single_and_multi_alias_results(
    monkeypatch,
) -> None:
    model_ir = _make_valid_gates()
    expected = copy.deepcopy(model_ir)
    _apply_expected_rewrite(expected, "single")
    _apply_expected_rewrite(expected, "multi")
    lowering_module._prune_unused_tensors(expected)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    def unexpected_graph_rescan(*args, **kwargs):
        raise AssertionError("unexpected producer/consumer map rebuild")

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_consumer_map",
        unexpected_graph_rescan,
    )
    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_producer_map",
        unexpected_graph_rescan,
    )

    stats = _optimize_transpose_dequant_logistic_mul_quantize_bridges(model_ir)

    assert refresh_count == 1
    assert stats == {"removed_transpose_dequant_logistic_mul_quantize_bridges": 2}
    assert _normalize(model_ir) == _normalize(expected)


def test_indexed_logistic_gate_keeps_supplied_index_current() -> None:
    model_ir = _make_valid_gates()
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _optimize_transpose_dequant_logistic_mul_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {"removed_transpose_dequant_logistic_mul_quantize_bridges": 2}
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type


@pytest.mark.parametrize(
    "protection",
    [
        "public_dequantized_data",
        "public_dequantized_logistic",
        "public_logistic",
        "public_quantized_gate",
        "public_dequantized_gate",
        "public_source",
        "public_post_alias",
        "pre_fanout",
        "data_fanout",
        "gate_fanout",
        "non_transpose_post_user",
        "per_channel",
        "wrong_post_perm",
    ],
)
def test_indexed_logistic_gate_preserves_protected_graph(
    protection: str,
) -> None:
    model_ir = ModelIR(f"protected_logistic_gate_{protection}")
    _add_gate(
        model_ir,
        prefix="protected",
        post_count=2,
        protection=protection,
    )
    before = copy.deepcopy(model_ir)

    stats = _optimize_transpose_dequant_logistic_mul_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_logistic_mul_quantize_bridges": 0}
    assert _normalize(model_ir) == _normalize(before)


def test_indexed_logistic_gate_skips_index_without_transpose(monkeypatch) -> None:
    model_ir = ModelIR("logistic_gate_without_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1]),
        "y": _tensor("y", [1]),
    }
    model_ir.operators = [OperatorIR("LOGISTIC", ["x"], ["y"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_dequant_logistic_mul_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_logistic_mul_quantize_bridges": 0}
    assert refresh_count == 0
