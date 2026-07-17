from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import onnx2tf.tflite_builder.lower_from_onnx2tf as lowering_module
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.hardswish_shape_sanitization import (
    sanitize_hardswish_tensor_shapes,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_fuse_conv_activation_chains,
    _prune_dead_operators,
    _reconcile_static_tensor_shapes,
    _resolve_dynamic_reshape_shapes,
    _run_indexed_final_shape_activation_convergence,
    _sanitize_hardswish_tensor_shapes,
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


def _make_final_convergence_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_final_shape_activation_convergence")
    model_ir.inputs = ["x", "conv_x"]
    model_ir.outputs = ["reshape_y", "relu_y"]
    model_ir.tensors = {
        "x": TensorIR(
            name="x",
            dtype="FLOAT32",
            shape=[1, 2, 2, 3],
            shape_signature=[-1, 2, 2, 3],
        ),
        "dead": TensorIR(
            name="dead",
            dtype="FLOAT32",
            shape=[1, 2, 2, 3],
            shape_signature=[-1, 2, 2, 3],
        ),
        "hardswish_y": TensorIR(
            name="hardswish_y",
            dtype="FLOAT32",
            shape=[1, 3, 2, 2],
            shape_signature=[-1, 3, 2, 2],
        ),
        "reshape_shape": TensorIR(
            name="reshape_shape",
            dtype="INT32",
            shape=[2],
            shape_signature=[2],
            data=np.asarray([1, 12], dtype=np.int32),
        ),
        "reshape_y": TensorIR(
            name="reshape_y",
            dtype="FLOAT32",
            shape=[1, 12],
            shape_signature=[-1, 12],
        ),
        "conv_x": TensorIR(
            name="conv_x",
            dtype="FLOAT32",
            shape=[1, 4, 4, 3],
            shape_signature=[1, 4, 4, 3],
        ),
        "conv_w": TensorIR(
            name="conv_w",
            dtype="FLOAT32",
            shape=[2, 3, 3, 3],
            shape_signature=[2, 3, 3, 3],
            data=np.ones((2, 3, 3, 3), dtype=np.float32),
        ),
        "conv_b": TensorIR(
            name="conv_b",
            dtype="FLOAT32",
            shape=[2],
            shape_signature=[2],
            data=np.zeros((2,), dtype=np.float32),
        ),
        "conv_y": TensorIR(
            name="conv_y",
            dtype="FLOAT32",
            shape=[1, 4, 4, 2],
            shape_signature=[1, 4, 4, 2],
        ),
        "relu_y": TensorIR(
            name="relu_y",
            dtype="FLOAT32",
            shape=[1, 4, 4, 2],
            shape_signature=[1, 4, 4, 2],
        ),
    }
    model_ir.operators = [
        OperatorIR("RELU", ["x"], ["dead"]),
        OperatorIR("HARD_SWISH", ["x"], ["hardswish_y"]),
        OperatorIR(
            "RESHAPE",
            ["hardswish_y", "reshape_shape"],
            ["reshape_y"],
            {
                "newShape": [1, 12],
                "onnxRawNewShape": [1, -1],
                "allowZero": False,
            },
        ),
        OperatorIR(
            "CONV_2D",
            ["conv_x", "conv_w", "conv_b"],
            ["conv_y"],
            {
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR("RELU", ["conv_y"], ["relu_y"]),
    ]
    return model_ir


def _run_legacy_final_convergence(model_ir: ModelIR) -> dict[str, int]:
    prune_stats = _prune_dead_operators(model_ir)
    first_reconcile_stats = _reconcile_static_tensor_shapes(model_ir)
    first_reshape_stats = _resolve_dynamic_reshape_shapes(model_ir)
    second_reconcile_stats = _reconcile_static_tensor_shapes(model_ir)
    hardswish_stats = _sanitize_hardswish_tensor_shapes(model_ir)
    third_reconcile_stats = _reconcile_static_tensor_shapes(model_ir)
    second_reshape_stats = _resolve_dynamic_reshape_shapes(model_ir)
    fourth_reconcile_stats = _reconcile_static_tensor_shapes(model_ir)
    fusion_stats = _optimize_fuse_conv_activation_chains(model_ir)
    final_reconcile_stats = _reconcile_static_tensor_shapes(model_ir)
    return {
        "removed_dead_operators": int(prune_stats["removed_dead_operators"]),
        "sanitized_hardswish_tensor_shapes": int(
            hardswish_stats["sanitized_hardswish_tensor_shapes"]
        ),
        "resolved_dynamic_reshape_shapes": int(
            first_reshape_stats["resolved_dynamic_reshape_shapes"]
            + second_reshape_stats["resolved_dynamic_reshape_shapes"]
        ),
        "reconciled_static_tensor_shapes": int(
            first_reconcile_stats["reconciled_static_tensor_shapes"]
            + second_reconcile_stats["reconciled_static_tensor_shapes"]
            + third_reconcile_stats["reconciled_static_tensor_shapes"]
            + fourth_reconcile_stats["reconciled_static_tensor_shapes"]
            + final_reconcile_stats["reconciled_static_tensor_shapes"]
        ),
        **fusion_stats,
    }


def _run_instrumented_final_convergence(
    monkeypatch,
    *,
    changed_owner: str | None,
) -> tuple[list[str], list[ModelIRGraphIndex | None], dict[str, int]]:
    model_ir = ModelIR("instrumented_final_shape_activation_convergence")
    events: list[str] = []
    graph_indexes: list[ModelIRGraphIndex | None] = []

    def convergence_probe(
        target_model_ir,
        *,
        layout_state=None,
        graph_index=None,
    ):
        assert target_model_ir is model_ir
        assert layout_state is None
        events.append("convergence")
        graph_indexes.append(graph_index)
        return {
            "removed_dead_operators": int(changed_owner == "convergence"),
            "resolved_dynamic_reshape_shapes": 0,
            "reconciled_static_tensor_shapes": 0,
        }

    def hardswish_probe(target_model_ir, *, graph_index=None):
        assert target_model_ir is model_ir
        events.append("hardswish")
        graph_indexes.append(graph_index)
        return {
            "sanitized_hardswish_tensor_shapes": int(
                changed_owner == "hardswish"
            ),
        }

    def reconcile_probe(target_model_ir, *, graph_index=None):
        assert target_model_ir is model_ir
        events.append("reconcile")
        graph_indexes.append(graph_index)
        return {"reconciled_static_tensor_shapes": 0}

    def reshape_probe(target_model_ir, *, graph_index=None):
        assert target_model_ir is model_ir
        events.append("reshape")
        graph_indexes.append(graph_index)
        return {"resolved_dynamic_reshape_shapes": 0}

    def fusion_probe(
        target_model_ir,
        *,
        graph_index=None,
        layout_state=None,
    ):
        assert target_model_ir is model_ir
        assert layout_state is None
        events.append("fusion")
        graph_indexes.append(graph_index)
        return {"fused_conv_activation_chains": 0}

    monkeypatch.setattr(
        lowering_module,
        "_run_indexed_shape_convergence_cleanup",
        convergence_probe,
    )
    monkeypatch.setattr(
        lowering_module,
        "_sanitize_hardswish_tensor_shapes",
        hardswish_probe,
    )
    monkeypatch.setattr(
        lowering_module,
        "_reconcile_static_tensor_shapes",
        reconcile_probe,
    )
    monkeypatch.setattr(
        lowering_module,
        "_resolve_dynamic_reshape_shapes",
        reshape_probe,
    )
    monkeypatch.setattr(
        lowering_module,
        "_optimize_fuse_conv_activation_chains",
        fusion_probe,
    )

    stats = _run_indexed_final_shape_activation_convergence(model_ir)
    return events, graph_indexes, stats


def test_hardswish_shape_sanitizer_module_matches_compatibility_wrapper() -> None:
    direct_model_ir = _make_final_convergence_model_ir()
    wrapper_model_ir = copy.deepcopy(direct_model_ir)

    direct_stats = sanitize_hardswish_tensor_shapes(
        direct_model_ir,
        graph_index=ModelIRGraphIndex(direct_model_ir),
    )
    wrapper_stats = _sanitize_hardswish_tensor_shapes(
        wrapper_model_ir,
        graph_index=ModelIRGraphIndex(wrapper_model_ir),
    )

    assert direct_stats == {"sanitized_hardswish_tensor_shapes": 1}
    assert wrapper_stats == direct_stats
    assert _normalize(wrapper_model_ir) == _normalize(direct_model_ir)


def test_indexed_final_convergence_matches_legacy_sequence(monkeypatch) -> None:
    model_ir = _make_final_convergence_model_ir()
    legacy_model_ir = copy.deepcopy(model_ir)
    legacy_stats = _run_legacy_final_convergence(legacy_model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    indexed_stats = _run_indexed_final_shape_activation_convergence(model_ir)

    assert refresh_count == 1
    assert indexed_stats == legacy_stats
    assert _normalize(model_ir) == _normalize(legacy_model_ir)
    assert [str(op.op_type) for op in model_ir.operators] == [
        "HARD_SWISH",
        "RESHAPE",
        "CONV_2D",
    ]
    assert model_ir.operators[-1].options["fusedActivationFunction"] == "RELU"


def test_indexed_final_convergence_skips_first_reconcile_when_stable(
    monkeypatch,
) -> None:
    events, graph_indexes, stats = _run_instrumented_final_convergence(
        monkeypatch,
        changed_owner=None,
    )

    assert events == [
        "convergence",
        "hardswish",
        "reshape",
        "reconcile",
        "fusion",
        "reconcile",
    ]
    assert stats == {
        "removed_dead_operators": 0,
        "sanitized_hardswish_tensor_shapes": 0,
        "resolved_dynamic_reshape_shapes": 0,
        "reconciled_static_tensor_shapes": 0,
        "fused_conv_activation_chains": 0,
    }
    assert graph_indexes[0] is not None
    assert all(index is graph_indexes[0] for index in graph_indexes)


@pytest.mark.parametrize("changed_owner", ["convergence", "hardswish"])
def test_indexed_final_convergence_keeps_first_reconcile_after_mutation(
    monkeypatch,
    changed_owner: str,
) -> None:
    events, graph_indexes, stats = _run_instrumented_final_convergence(
        monkeypatch,
        changed_owner=changed_owner,
    )

    assert events == [
        "convergence",
        "hardswish",
        "reconcile",
        "reshape",
        "reconcile",
        "fusion",
        "reconcile",
    ]
    assert stats["removed_dead_operators"] == int(
        changed_owner == "convergence"
    )
    assert stats["sanitized_hardswish_tensor_shapes"] == int(
        changed_owner == "hardswish"
    )
    assert graph_indexes[0] is not None
    assert all(index is graph_indexes[0] for index in graph_indexes)


def test_indexed_activation_fusion_updates_index_without_consumer_rescan(
    monkeypatch,
) -> None:
    model_ir = ModelIR("indexed_activation_fusion")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["z"]
    for tensor_name in ["x", "y", "z"]:
        model_ir.tensors[tensor_name] = TensorIR(
            name=tensor_name,
            dtype="FLOAT32",
            shape=[1, 2],
            shape_signature=[1, 2],
        )
    producer = OperatorIR(
        "conv_2d",
        ["x"],
        ["y"],
        {"fusedActivationFunction": "NONE"},
    )
    model_ir.operators = [
        producer,
        OperatorIR("RELU", ["y"], ["z"]),
    ]
    graph_index = ModelIRGraphIndex(model_ir)

    def unexpected_consumer_rescan(*args, **kwargs):
        raise AssertionError("unexpected full consumer-map rebuild")

    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_consumer_map",
        unexpected_consumer_rescan,
        raising=False,
    )
    stats = _optimize_fuse_conv_activation_chains(
        model_ir,
        graph_index=graph_index,
    )

    assert stats["fused_conv_activation_chains"] == 1
    assert stats["fused_activation_chains_total"] == 1
    assert model_ir.operators == [producer]
    assert producer.outputs == ["z"]
    assert producer.options["fusedActivationFunction"] == "RELU"
    assert graph_index.producer("z") is producer
    assert graph_index.producer("y") is None
    assert graph_index.consumer_indices("y") == []
    assert graph_index.operator_indices_for_normalized_types(["CONV_2D"]) == [0]
    assert graph_index.operator_indices("RELU") == []
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type
