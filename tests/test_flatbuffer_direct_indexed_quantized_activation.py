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
    _optimize_transpose_dequant_relu_quantize_bridges,
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
    dtype: str = "FLOAT32",
    quantization: QuantParamIR | None = None,
    data: np.ndarray | None = None,
    shape_signature: list[int] | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=(
            list(shape)
            if shape_signature is None
            else list(shape_signature)
        ),
        quantization=quantization,
        data=data,
        is_variable=data is None,
    )


def _quant(scale: list[float] | None = None) -> QuantParamIR:
    return QuantParamIR(
        scale=[0.125] if scale is None else list(scale),
        zero_point=[0] if scale is None else [0 for _ in scale],
        quantized_dimension=3,
    )


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str,
    activation: str,
    protection: str | None = None,
) -> None:
    source = f"{prefix}_source"
    q_input_bridge = f"{prefix}_q_input_bridge"
    float_input_bridge = f"{prefix}_float_input_bridge"
    float_output_bridge = f"{prefix}_float_output_bridge"
    q_output_bridge = f"{prefix}_q_output_bridge"
    destination = f"{prefix}_destination"
    pre_perm = f"{prefix}_pre_perm"
    post_perm = f"{prefix}_post_perm"
    per_channel = protection == "per_channel"
    source_quant = _quant([0.125, 0.25]) if per_channel else _quant()
    mid_quant = _quant()
    destination_quant = _quant([0.5])
    model_ir.inputs.append(source)
    model_ir.outputs.append(destination)
    model_ir.tensors.update(
        {
            source: _tensor(
                source,
                [1, 2, 3, 4],
                dtype="INT8",
                quantization=source_quant,
                shape_signature=[-1, 2, 3, 4],
            ),
            q_input_bridge: _tensor(
                q_input_bridge,
                [1, 3, 4, 2],
                dtype="INT8",
                quantization=_quant(),
            ),
            float_input_bridge: _tensor(
                float_input_bridge,
                [1, 3, 4, 2],
            ),
            float_output_bridge: _tensor(
                float_output_bridge,
                [1, 3, 4, 2],
            ),
            q_output_bridge: _tensor(
                q_output_bridge,
                [1, 3, 4, 2],
                dtype="INT8",
                quantization=mid_quant,
            ),
            destination: _tensor(
                destination,
                [1, 2, 3, 4],
                dtype="UINT8",
                quantization=destination_quant,
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
                    [0, 2, 1, 3]
                    if protection == "wrong_perm"
                    else [0, 3, 1, 2],
                    dtype=np.int32,
                ),
            ),
        }
    )
    model_ir.operators.extend(
        [
            OperatorIR("TRANSPOSE", [source, pre_perm], [q_input_bridge]),
            OperatorIR("DEQUANTIZE", [q_input_bridge], [float_input_bridge]),
            OperatorIR(activation, [float_input_bridge], [float_output_bridge]),
            OperatorIR("QUANTIZE", [float_output_bridge], [q_output_bridge]),
            OperatorIR(
                "TRANSPOSE",
                [q_output_bridge, post_perm],
                [destination],
            ),
        ]
    )
    if protection == "public_intermediate":
        model_ir.outputs.append(float_output_bridge)
    elif protection == "source_output":
        model_ir.outputs.append(source)
    elif protection == "fanout":
        side_output = f"{prefix}_side_output"
        model_ir.tensors[side_output] = _tensor(
            side_output,
            [1, 3, 4, 2],
        )
        model_ir.operators.append(
            OperatorIR("NEG", [float_input_bridge], [side_output])
        )
        model_ir.outputs.append(side_output)


def _make_two_valid_chains() -> ModelIR:
    model_ir = ModelIR("indexed_quantized_activation")
    _add_chain(model_ir, prefix="relu", activation="RELU")
    _add_chain(model_ir, prefix="relu6", activation="RELU6")
    return model_ir


def _apply_expected_legacy_rewrite(model_ir: ModelIR, prefix: str) -> None:
    source = f"{prefix}_source"
    destination = f"{prefix}_destination"
    q_input_bridge = f"{prefix}_q_input_bridge"
    float_input_bridge = f"{prefix}_float_input_bridge"
    float_output_bridge = f"{prefix}_float_output_bridge"
    q_output_bridge = f"{prefix}_q_output_bridge"
    pre_op = next(op for op in model_ir.operators if op.outputs == [q_input_bridge])
    dequantize_op = next(
        op for op in model_ir.operators if op.outputs == [float_input_bridge]
    )
    quantize_op = next(
        op for op in model_ir.operators if op.outputs == [q_output_bridge]
    )
    post_op = next(op for op in model_ir.operators if op.outputs == [destination])
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
    for bridge_name in [float_input_bridge, float_output_bridge]:
        model_ir.tensors[bridge_name].shape = list(source_tensor.shape)
        model_ir.tensors[bridge_name].shape_signature = list(
            source_tensor.shape_signature or source_tensor.shape
        )
    q_output_tensor = model_ir.tensors[q_output_bridge]
    destination_tensor = model_ir.tensors[destination]
    destination_tensor.dtype = str(q_output_tensor.dtype)
    destination_tensor.quantization = copy.deepcopy(q_output_tensor.quantization)
    model_ir.operators.remove(post_op)
    model_ir.operators.remove(pre_op)


def test_indexed_quantized_activation_matches_former_multi_chain_result(
    monkeypatch,
) -> None:
    model_ir = _make_two_valid_chains()
    legacy_model_ir = copy.deepcopy(model_ir)
    _apply_expected_legacy_rewrite(legacy_model_ir, "relu")
    _apply_expected_legacy_rewrite(legacy_model_ir, "relu6")
    lowering_module._prune_unused_tensors(legacy_model_ir)
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

    stats = _optimize_transpose_dequant_relu_quantize_bridges(model_ir)

    assert refresh_count == 1
    assert stats == {"removed_transpose_dequant_relu_quantize_bridges": 2}
    assert _normalize(model_ir) == _normalize(legacy_model_ir)


def test_indexed_quantized_activation_keeps_graph_index_current() -> None:
    model_ir = _make_two_valid_chains()
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _optimize_transpose_dequant_relu_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {"removed_transpose_dequant_relu_quantize_bridges": 2}
    assert graph_index.operator_indices("TRANSPOSE") == []
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type


@pytest.mark.parametrize(
    "protection",
    ["public_intermediate", "source_output", "fanout", "per_channel", "wrong_perm"],
)
def test_indexed_quantized_activation_preserves_protected_chain(
    protection: str,
) -> None:
    model_ir = ModelIR(f"protected_quantized_activation_{protection}")
    _add_chain(
        model_ir,
        prefix="protected",
        activation="RELU6",
        protection=protection,
    )
    before = copy.deepcopy(model_ir)

    stats = _optimize_transpose_dequant_relu_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_relu_quantize_bridges": 0}
    assert _normalize(model_ir) == _normalize(before)


def test_indexed_quantized_activation_skips_index_without_transpose(
    monkeypatch,
) -> None:
    model_ir = ModelIR("quantized_activation_without_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1], dtype="INT8", quantization=_quant()),
        "y": _tensor("y", [1]),
    }
    model_ir.operators = [OperatorIR("DEQUANTIZE", ["x"], ["y"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_dequant_relu_quantize_bridges(model_ir)

    assert stats == {"removed_transpose_dequant_relu_quantize_bridges": 0}
    assert refresh_count == 0
