from __future__ import annotations

import copy
import re
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import onnx2tf.tflite_builder.lower_from_onnx2tf as lowering_module

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_orphan_recurrent_step_tensors as repair_direct_orphan_aliases,
)
from onnx2tf.tflite_builder.passes.pytorch_recurrent import (
    _repair_orphan_recurrent_step_tensors as repair_pytorch_orphan_aliases,
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
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32" if data is None else str(data.dtype).upper(),
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _make_recurrent_alias_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_recurrent_alias")
    model_ir.inputs = ["source", "public_h_step_4"]
    model_ir.outputs = ["combined", "public_c_step_2"]
    model_ir.tensors = {
        "source": _tensor("source", [1, 4]),
        "state_h_step_shape_1": _tensor(
            "state_h_step_shape_1",
            [2],
            data=np.asarray([1, 4], dtype=np.int32),
        ),
        "state_h_step_1": _tensor("state_h_step_1", [1, 4]),
        "first_h_replacement": _tensor("first_h_replacement", [1, 4]),
        "second_h_replacement": _tensor("second_h_replacement", [1, 4]),
        "public_c_step_shape_2": _tensor(
            "public_c_step_shape_2",
            [2],
            data=np.asarray([1, 4], dtype=np.int32),
        ),
        "public_c_step_2": _tensor("public_c_step_2", [1, 4]),
        "c_replacement": _tensor("c_replacement", [1, 4]),
        "unused_h_step_shape_6": _tensor(
            "unused_h_step_shape_6",
            [2],
            data=np.asarray([1, 4], dtype=np.int32),
        ),
        "unused_h_step_6": _tensor("unused_h_step_6", [1, 4]),
        "unused_replacement": _tensor("unused_replacement", [1, 4]),
        "produced_h_step_3": _tensor("produced_h_step_3", [1, 4]),
        "public_h_step_4": _tensor("public_h_step_4", [1, 4]),
        "missing_shape_c_step_7": _tensor("missing_shape_c_step_7", [1, 4]),
        "invalid_h_step_tail": _tensor("invalid_h_step_tail", [1, 4]),
        "add_out": _tensor("add_out", [1, 4]),
        "combined": _tensor("combined", [2, 4]),
        "produced_use": _tensor("produced_use", [1, 4]),
        "public_use": _tensor("public_use", [1, 4]),
    }
    model_ir.operators = [
        OperatorIR(
            "RESHAPE",
            ["source", "state_h_step_shape_1"],
            ["first_h_replacement"],
        ),
        OperatorIR(
            "RESHAPE",
            ["source", "state_h_step_shape_1"],
            ["second_h_replacement"],
        ),
        OperatorIR(
            "RESHAPE",
            ["source", "public_c_step_shape_2"],
            ["c_replacement"],
        ),
        OperatorIR(
            "RESHAPE",
            ["source", "unused_h_step_shape_6"],
            ["unused_replacement"],
        ),
        OperatorIR("IDENTITY", ["source"], ["produced_h_step_3"]),
        OperatorIR(
            "ADD",
            ["state_h_step_1", "first_h_replacement"],
            ["add_out"],
        ),
        OperatorIR(
            "CONCATENATION",
            ["public_c_step_2", "state_h_step_1"],
            ["combined"],
        ),
        OperatorIR("IDENTITY", ["produced_h_step_3"], ["produced_use"]),
        OperatorIR("IDENTITY", ["public_h_step_4"], ["public_use"]),
    ]
    return model_ir


def _run_legacy_direct_recurrent_alias_repair(
    model_ir: ModelIR,
) -> dict[str, int]:
    repaired = 0
    producers = lowering_module._build_tensor_producer_map(model_ir)
    consumers = lowering_module._build_tensor_consumer_map(model_ir)
    model_inputs = {str(name) for name in model_ir.inputs}
    model_outputs = {str(name) for name in model_ir.outputs}
    for raw_tensor_name in list(model_ir.tensors.keys()):
        tensor_name = str(raw_tensor_name)
        if tensor_name in producers or tensor_name in model_inputs:
            continue
        match = re.match(r"^(.+_(?:h|c)_step_)(\d+)$", tensor_name)
        if match is None:
            continue
        shape_tensor_name = f"{match.group(1)}shape_{match.group(2)}"
        replacement_name = None
        for op in model_ir.operators:
            if (
                str(op.op_type) != "RESHAPE"
                or len(op.inputs) < 2
                or len(op.outputs) != 1
            ):
                continue
            if str(op.inputs[1]) != shape_tensor_name:
                continue
            candidate_name = str(op.outputs[0])
            if candidate_name == tensor_name:
                replacement_name = None
                break
            replacement_name = candidate_name
            break
        if replacement_name is None:
            continue
        for consumer_index in consumers.get(tensor_name, []):
            consumer = model_ir.operators[int(consumer_index)]
            consumer.inputs = [
                replacement_name
                if str(input_name) == tensor_name
                else str(input_name)
                for input_name in consumer.inputs
            ]
        if tensor_name not in model_outputs:
            model_ir.tensors.pop(tensor_name, None)
        repaired += 1
        producers = lowering_module._build_tensor_producer_map(model_ir)
        consumers = lowering_module._build_tensor_consumer_map(model_ir)
    return {"repaired_orphan_recurrent_step_tensors": int(repaired)}


def test_shared_recurrent_alias_owner_matches_legacy_direct_behavior(
    monkeypatch,
) -> None:
    model_ir = _make_recurrent_alias_model_ir()
    legacy_model_ir = copy.deepcopy(model_ir)
    legacy_stats = _run_legacy_direct_recurrent_alias_repair(legacy_model_ir)
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
        "_build_tensor_producer_map",
        unexpected_graph_rescan,
    )
    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_consumer_map",
        unexpected_graph_rescan,
    )

    stats = repair_direct_orphan_aliases(model_ir)

    assert refresh_count == 1
    assert stats == legacy_stats
    assert stats == {"repaired_orphan_recurrent_step_tensors": 3}
    assert _normalize(model_ir) == _normalize(legacy_model_ir)


def test_direct_and_pytorch_recurrent_alias_wrappers_share_exact_mutations(
) -> None:
    direct_model_ir = _make_recurrent_alias_model_ir()
    pytorch_model_ir = copy.deepcopy(direct_model_ir)
    direct_index = ModelIRGraphIndex(direct_model_ir)
    pytorch_index = ModelIRGraphIndex(pytorch_model_ir)

    direct_stats = repair_direct_orphan_aliases(
        direct_model_ir,
        graph_index=direct_index,
    )
    pytorch_result = repair_pytorch_orphan_aliases(
        pytorch_model_ir,
        graph_index=pytorch_index,
    )

    assert direct_stats == {"repaired_orphan_recurrent_step_tensors": 3}
    assert pytorch_result is None
    assert _normalize(direct_model_ir) == _normalize(pytorch_model_ir)
    assert direct_model_ir.operators[5].inputs == [
        "first_h_replacement",
        "first_h_replacement",
    ]
    assert direct_model_ir.operators[6].inputs == [
        "c_replacement",
        "first_h_replacement",
    ]
    assert "state_h_step_1" not in direct_model_ir.tensors
    assert "unused_h_step_6" not in direct_model_ir.tensors
    assert "public_c_step_2" in direct_model_ir.tensors
    refreshed = ModelIRGraphIndex(direct_model_ir)
    assert direct_index.producers == refreshed.producers
    assert direct_index.consumers == refreshed.consumers
    assert direct_index.duplicate_producers == refreshed.duplicate_producers
    assert direct_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert direct_index._operator_indices_by_type == refreshed._operator_indices_by_type


def test_direct_recurrent_alias_wrapper_skips_index_without_candidates(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_recurrent_alias_candidates")
    model_ir.inputs = ["source"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "source": _tensor("source", [1]),
        "output": _tensor("output", [1]),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["source"], ["output"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = repair_direct_orphan_aliases(model_ir)

    assert stats == {"repaired_orphan_recurrent_step_tensors": 0}
    assert refresh_count == 0
