from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import onnx2tf.tflite_builder.lower_from_onnx2tf as lowering_module
from onnx2tf.tflite_builder.passes import unbound_input_repair_orchestration

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_unbound_nonconstant_operator_inputs_with_layout_transpose,
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
    data: np.ndarray | None = None,
    quantization: QuantParamIR | None = None,
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
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _add_source(
    model_ir: ModelIR,
    *,
    output_name: str,
    shape: list[int],
    dtype: str = "FLOAT32",
    quantization: QuantParamIR | None = None,
    shape_signature: list[int] | None = None,
) -> None:
    input_name = f"{output_name}_seed"
    const_name = f"{output_name}_zero"
    model_ir.inputs.append(input_name)
    model_ir.tensors[input_name] = _tensor(
        input_name,
        shape,
        dtype=dtype,
        quantization=quantization,
        shape_signature=shape_signature,
    )
    model_ir.tensors[const_name] = _tensor(
        const_name,
        [1],
        dtype=dtype,
        data=np.asarray([0], dtype=np.int8 if dtype == "INT8" else np.float32),
    )
    model_ir.tensors[output_name] = _tensor(
        output_name,
        shape,
        dtype=dtype,
        quantization=quantization,
        shape_signature=shape_signature,
    )
    model_ir.operators.append(
        OperatorIR("ADD", [input_name, const_name], [output_name])
    )


def _make_all_family_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_unbound_input_layout")
    quant = QuantParamIR(scale=[0.25], zero_point=[-3], quantized_dimension=3)
    _add_source(
        model_ir,
        output_name="dq_orphan_nhwc_bridge",
        shape=[1, 5, 6, 4],
        dtype="INT8",
        quantization=quant,
    )
    _add_source(
        model_ir,
        output_name="shape_far",
        shape=[1, 7, 9, 8],
    )
    _add_source(
        model_ir,
        output_name="shape_near",
        shape=[1, 7, 9, 8],
    )
    _add_source(
        model_ir,
        output_name="reshape_source",
        shape=[1, 4, 5, 3],
    )
    _add_source(
        model_ir,
        output_name="split_source",
        shape=[1, 6, 7, 2],
    )
    mul_quant = QuantParamIR(
        scale=[0.125],
        zero_point=[2],
        quantized_dimension=3,
    )
    _add_source(
        model_ir,
        output_name="branch_input_nhwc",
        shape=[1, 2, 3, 10],
        quantization=mul_quant,
        shape_signature=[1, -1, 3, 10],
    )
    model_ir.tensors.update(
        {
            "dq_orphan": _tensor("dq_orphan", [1, 4, 5, 6], dtype="INT8"),
            "dq_out": _tensor("dq_out", [1, 4, 5, 6]),
            "shape_orphan": _tensor("shape_orphan", [1, 8, 7, 9]),
            "shape_out": _tensor("shape_out", [4], dtype="INT64"),
            "reshape_orphan": _tensor("reshape_orphan", [1, 3, 4, 5]),
            "reshape_target": _tensor(
                "reshape_target",
                [2],
                dtype="INT32",
                data=np.asarray([1, 60], dtype=np.int32),
            ),
            "reshape_out": _tensor("reshape_out", [1, 60]),
            "split_orphan": _tensor("split_orphan", [1, 2, 6, 7]),
            "split_axis": _tensor(
                "split_axis",
                [1],
                dtype="INT32",
                data=np.asarray([1], dtype=np.int32),
            ),
            "split0": _tensor("split0", [1, 1, 6, 7]),
            "split1": _tensor("split1", [1, 1, 6, 7]),
            "input.alias": _tensor("input.alias", [1, 10, 2, 3]),
            "mul_const0": _tensor(
                "mul_const0",
                [1],
                data=np.asarray([2.0], dtype=np.float32),
            ),
            "mul_const1": _tensor(
                "mul_const1",
                [1],
                data=np.asarray([3.0], dtype=np.float32),
            ),
            "mul0": _tensor("mul0", [1, 10, 2, 3]),
            "mul1": _tensor("mul1", [1, 10, 2, 3]),
            "input.protected": _tensor("input.protected", [1, 10, 2, 3]),
            "protected_mul": _tensor("protected_mul", [1, 10, 2, 3]),
            "protected_relu": _tensor("protected_relu", [1, 10, 2, 3]),
        }
    )
    model_ir.operators.extend(
        [
            OperatorIR("DEQUANTIZE", ["dq_orphan"], ["dq_out"]),
            OperatorIR("SHAPE", ["shape_orphan"], ["shape_out"]),
            OperatorIR(
                "RESHAPE",
                ["reshape_orphan", "reshape_target"],
                ["reshape_out"],
            ),
            OperatorIR(
                "SPLIT",
                ["split_axis", "split_orphan"],
                ["split0", "split1"],
                {"numSplits": 2},
            ),
            OperatorIR("MUL", ["input.alias", "mul_const0"], ["mul0"]),
            OperatorIR("MUL", ["input.alias", "mul_const1"], ["mul1"]),
            OperatorIR(
                "MUL",
                ["input.protected", "mul_const0"],
                ["protected_mul"],
            ),
            OperatorIR("RELU", ["input.protected"], ["protected_relu"]),
        ]
    )
    model_ir.outputs = [
        "dq_out",
        "shape_out",
        "reshape_out",
        "split0",
        "split1",
        "mul0",
        "mul1",
        "protected_mul",
        "protected_relu",
    ]
    return model_ir


def _legacy_find(model_ir: ModelIR) -> list[dict[str, Any]]:
    producers = {
        str(output_name): int(op_index)
        for op_index, op in enumerate(model_ir.operators)
        for output_name in op.outputs
    }
    model_inputs = {str(name) for name in model_ir.inputs}
    issues: list[dict[str, Any]] = []
    for op_index, op in enumerate(model_ir.operators):
        for input_index, raw_name in enumerate(op.inputs):
            input_name = str(raw_name)
            tensor = model_ir.tensors.get(input_name)
            if (
                not input_name
                or input_name in model_inputs
                or input_name in producers
                or (tensor is not None and tensor.data is not None)
            ):
                continue
            issues.append(
                {
                    "op_index": op_index,
                    "op_type": str(op.op_type),
                    "input_index": input_index,
                    "tensor_name": input_name,
                }
            )
    return issues


def _legacy_insert(
    model_ir: ModelIR,
    *,
    op_index: int,
    source_name: str,
    tensor_name: str,
) -> None:
    base = f"{tensor_name}_repair_perm"
    perm_name = base
    serial = 1
    while perm_name in model_ir.tensors:
        perm_name = f"{base}_{serial}"
        serial += 1
    model_ir.tensors[perm_name] = _tensor(
        perm_name,
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.operators.insert(
        op_index,
        OperatorIR(
            "TRANSPOSE",
            [source_name, perm_name],
            [tensor_name],
        ),
    )


def _legacy_source(
    model_ir: ModelIR,
    producers: dict[str, int],
    *,
    expected_shape: list[int],
    op_index: int,
    suffix: str | None = None,
    dtype: str | None = None,
    require_add: bool = False,
) -> str | None:
    best_name = None
    best_index = -1
    for name, source_index in producers.items():
        if source_index >= op_index or (suffix and not name.endswith(suffix)):
            continue
        source_op = model_ir.operators[source_index]
        source_tensor = model_ir.tensors.get(name)
        if (
            source_tensor is None
            or source_tensor.data is not None
            or [int(value) for value in source_tensor.shape] != expected_shape
            or (dtype is not None and str(source_tensor.dtype) != dtype)
            or (require_add and str(source_op.op_type) != "ADD")
        ):
            continue
        if source_index > best_index:
            best_name = name
            best_index = source_index
    return best_name


def _run_legacy_repair(model_ir: ModelIR) -> dict[str, int]:
    repaired = 0
    while True:
        issues = _legacy_find(model_ir)
        producers = {
            str(output_name): int(op_index)
            for op_index, op in enumerate(model_ir.operators)
            for output_name in op.outputs
        }
        consumers: dict[str, list[int]] = {}
        for op_index, op in enumerate(model_ir.operators):
            for input_name in op.inputs:
                consumers.setdefault(str(input_name), []).append(op_index)
        changed = False
        for issue in issues:
            op_index = int(issue["op_index"])
            input_index = int(issue["input_index"])
            tensor_name = str(issue["tensor_name"])
            consumer = model_ir.operators[op_index]
            consumer_type = str(consumer.op_type)
            orphan = model_ir.tensors.get(tensor_name)
            if orphan is None or orphan.data is not None or len(orphan.shape) != 4:
                continue
            orphan_shape = [int(value) for value in orphan.shape]
            expected = [
                orphan_shape[0],
                orphan_shape[2],
                orphan_shape[3],
                orphan_shape[1],
            ]
            source_name = None
            if consumer_type == "DEQUANTIZE" and input_index == 0:
                exact_name = f"{tensor_name}_nhwc_bridge"
                if exact_name in producers:
                    source_name = exact_name
                else:
                    source_name = _legacy_source(
                        model_ir,
                        producers,
                        expected_shape=expected,
                        op_index=op_index,
                        dtype=str(orphan.dtype),
                        require_add=True,
                    )
                    if source_name is not None and not (
                        source_name.endswith("_nhwc")
                        or source_name.endswith("_nhwc_bridge")
                    ):
                        source_name = None
                source_tensor = model_ir.tensors.get(source_name or "")
                if (
                    source_name is None
                    or producers[source_name] >= op_index
                    or source_tensor is None
                    or source_tensor.data is not None
                    or str(source_tensor.dtype) != str(orphan.dtype)
                    or [int(value) for value in source_tensor.shape] != expected
                ):
                    source_name = None
                elif source_tensor is not None:
                    orphan.quantization = copy.deepcopy(source_tensor.quantization)
            elif consumer_type in {"RESHAPE", "SHAPE", "SPLIT"}:
                target_index = 1 if consumer_type == "SPLIT" else 0
                if input_index != target_index:
                    continue
                source_name = _legacy_source(
                    model_ir,
                    producers,
                    expected_shape=expected,
                    op_index=op_index,
                )
            elif consumer_type == "MUL" and input_index == 0:
                if not tensor_name.startswith("input."):
                    continue
                tensor_consumers = consumers.get(tensor_name, [])
                if not tensor_consumers or not all(
                    str(model_ir.operators[index].op_type) == "MUL"
                    and str(model_ir.operators[index].inputs[0]) == tensor_name
                    for index in tensor_consumers
                ):
                    continue
                source_name = _legacy_source(
                    model_ir,
                    producers,
                    expected_shape=expected,
                    op_index=op_index,
                    suffix="_input_nhwc",
                    dtype=str(orphan.dtype),
                    require_add=True,
                )
                source_tensor = model_ir.tensors.get(source_name or "")
                if source_tensor is not None:
                    orphan.quantization = copy.deepcopy(source_tensor.quantization)
                    orphan.shape_signature = (
                        list(source_tensor.shape_signature)
                        if source_tensor.shape_signature is not None
                        else list(orphan_shape)
                    )
            if source_name is None:
                continue
            _legacy_insert(
                model_ir,
                op_index=op_index,
                source_name=source_name,
                tensor_name=tensor_name,
            )
            repaired += 1
            changed = True
            break
        if not changed:
            break
    if repaired:
        lowering_module._reconcile_static_tensor_shapes(model_ir)
    return {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": repaired,
    }


def test_indexed_unbound_input_layout_matches_all_legacy_families(
    monkeypatch,
) -> None:
    model_ir = _make_all_family_model_ir()
    legacy_model_ir = copy.deepcopy(model_ir)
    assert lowering_module._find_unbound_nonconstant_operator_inputs(
        model_ir
    ) == _legacy_find(model_ir)
    legacy_stats = _run_legacy_repair(legacy_model_ir)
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
        raising=False,
    )
    monkeypatch.setattr(
        lowering_module,
        "_build_tensor_consumer_map",
        unexpected_graph_rescan,
        raising=False,
    )

    stats = _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
        model_ir
    )

    assert refresh_count == 1
    assert stats == legacy_stats
    assert stats == {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": 5,
    }
    assert _normalize(model_ir) == _normalize(legacy_model_ir)


def test_indexed_unbound_input_layout_keeps_index_current_and_guards_fanout(
) -> None:
    model_ir = _make_all_family_model_ir()
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": 5,
    }
    transposes = [
        op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"
    ]
    assert [str(op.outputs[0]) for op in transposes] == [
        "dq_orphan",
        "shape_orphan",
        "reshape_orphan",
        "split_orphan",
        "input.alias",
    ]
    assert transposes[1].inputs[0] == "shape_near"
    assert "input.protected" not in graph_index.producers
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type


def test_indexed_unbound_wrapper_reconciles_once_after_positive_repair(
    monkeypatch,
) -> None:
    model_ir = _make_all_family_model_ir()
    reconcile_graph_indexes: list[ModelIRGraphIndex | None] = []
    original_reconcile = (
        unbound_input_repair_orchestration.reconcile_static_tensor_shapes
    )

    def counted_reconcile(
        active_model_ir: ModelIR,
        *,
        graph_index: ModelIRGraphIndex | None = None,
        **kwargs,
    ) -> dict[str, int]:
        reconcile_graph_indexes.append(graph_index)
        return original_reconcile(
            active_model_ir,
            graph_index=graph_index,
            **kwargs,
        )

    monkeypatch.setattr(
        unbound_input_repair_orchestration,
        "reconcile_static_tensor_shapes",
        counted_reconcile,
    )

    stats = _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
        model_ir
    )

    assert stats == {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": 5,
    }
    assert len(reconcile_graph_indexes) == 1
    assert reconcile_graph_indexes[0] is not None
    assert reconcile_graph_indexes[0].model_ir is model_ir


def test_indexed_unbound_dequantize_fallback_is_nearest_and_exact_is_strict(
) -> None:
    model_ir = ModelIR("unbound_dequantize_fallback_policy")
    fallback_quant = QuantParamIR(
        scale=[0.25],
        zero_point=[-4],
        quantized_dimension=3,
    )
    blocked_quant = QuantParamIR(
        scale=[0.5],
        zero_point=[1],
        quantized_dimension=3,
    )
    _add_source(
        model_ir,
        output_name="fallback_far_nhwc",
        shape=[1, 5, 6, 4],
        dtype="INT8",
        quantization=fallback_quant,
    )
    _add_source(
        model_ir,
        output_name="fallback_near_nhwc_bridge",
        shape=[1, 5, 6, 4],
        dtype="INT8",
        quantization=fallback_quant,
    )
    _add_source(
        model_ir,
        output_name="blocked_fallback_nhwc",
        shape=[1, 7, 8, 3],
        dtype="INT8",
        quantization=blocked_quant,
    )
    model_ir.tensors.update(
        {
            "fallback_orphan": _tensor(
                "fallback_orphan",
                [1, 4, 5, 6],
                dtype="INT8",
            ),
            "fallback_out": _tensor("fallback_out", [1, 4, 5, 6]),
            "blocked_orphan": _tensor(
                "blocked_orphan",
                [1, 3, 7, 8],
                dtype="INT8",
            ),
            "blocked_out": _tensor("blocked_out", [1, 3, 7, 8]),
        }
    )
    model_ir.operators.extend(
        [
            OperatorIR("DEQUANTIZE", ["fallback_orphan"], ["fallback_out"]),
            OperatorIR("DEQUANTIZE", ["blocked_orphan"], ["blocked_out"]),
        ]
    )
    _add_source(
        model_ir,
        output_name="blocked_orphan_nhwc_bridge",
        shape=[1, 7, 8, 3],
        dtype="INT8",
        quantization=blocked_quant,
    )
    model_ir.outputs = ["fallback_out", "blocked_out"]

    stats = _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
        model_ir
    )

    assert stats == {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": 1,
    }
    fallback_bridge = next(
        op
        for op in model_ir.operators
        if str(op.op_type) == "TRANSPOSE"
        and op.outputs == ["fallback_orphan"]
    )
    assert fallback_bridge.inputs[0] == "fallback_near_nhwc_bridge"
    assert model_ir.tensors["fallback_orphan"].quantization == fallback_quant
    assert not any(
        str(op.op_type) == "TRANSPOSE"
        and op.outputs == ["blocked_orphan"]
        for op in model_ir.operators
    )


def test_indexed_unbound_input_layout_skips_index_without_issues(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_unbound_inputs")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1]),
        "y": _tensor("y", [1]),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
        model_ir
    )

    assert stats == {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": 0,
    }
    assert refresh_count == 0
