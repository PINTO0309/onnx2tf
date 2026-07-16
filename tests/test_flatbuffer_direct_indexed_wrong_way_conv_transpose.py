from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import onnx2tf.tflite_builder.lower_from_onnx2tf as lowering_module
import onnx2tf.tflite_builder.passes.conv_input_layout as owner_module

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv,
    _optimize_transpose_swish_qdq_nhwc_islands,
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


def _add_wrong_way_branch(
    model_ir: ModelIR,
    *,
    name: str,
    source_shape: list[int],
    filter_channels: int,
    conv_count: int = 1,
    non_conv_fanout: bool = False,
    public_adapter: bool = False,
) -> None:
    source_name = f"{name}_source"
    adapter_name = f"{name}_adapter"
    source_channels = int(source_shape[3])
    adapter_shape = [
        int(source_shape[0]),
        int(source_shape[2]),
        int(source_shape[3]),
        int(source_shape[1]),
    ]
    model_ir.inputs.append(source_name)
    model_ir.tensors[source_name] = _tensor(source_name, source_shape)
    model_ir.tensors[adapter_name] = _tensor(adapter_name, adapter_shape)
    model_ir.operators.append(
        OperatorIR(
            "TRANSPOSE",
            [source_name, "perm_nchw_to_nhwc"],
            [adapter_name],
        )
    )
    for conv_index in range(conv_count):
        filter_name = f"{name}_filter{conv_index}"
        output_name = f"{name}_conv{conv_index}_out"
        model_ir.tensors[filter_name] = _tensor(
            filter_name,
            [2, 1, 1, int(filter_channels)],
            data=np.ones(
                (2, 1, 1, int(filter_channels)),
                dtype=np.float32,
            ),
        )
        model_ir.tensors[output_name] = _tensor(
            output_name,
            [
                int(adapter_shape[0]),
                int(adapter_shape[1]),
                int(adapter_shape[2]),
                2,
            ],
        )
        model_ir.operators.append(
            OperatorIR(
                "CONV_2D",
                [adapter_name, filter_name],
                [output_name],
            )
        )
        model_ir.outputs.append(output_name)
    if non_conv_fanout:
        relu_output_name = f"{name}_relu_out"
        model_ir.tensors[relu_output_name] = _tensor(
            relu_output_name,
            adapter_shape,
        )
        model_ir.operators.append(
            OperatorIR("RELU", [adapter_name], [relu_output_name])
        )
        model_ir.outputs.append(relu_output_name)
    if public_adapter:
        model_ir.outputs.append(adapter_name)
    assert source_channels != int(adapter_shape[3])


def _make_wrong_way_conv_transpose_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_wrong_way_conv_transpose")
    model_ir.tensors["perm_nchw_to_nhwc"] = _tensor(
        "perm_nchw_to_nhwc",
        [4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    _add_wrong_way_branch(
        model_ir,
        name="removable_multi",
        source_shape=[1, 5, 6, 4],
        filter_channels=4,
        conv_count=2,
    )
    _add_wrong_way_branch(
        model_ir,
        name="protected_nonconv",
        source_shape=[1, 7, 8, 3],
        filter_channels=3,
        non_conv_fanout=True,
    )
    _add_wrong_way_branch(
        model_ir,
        name="removable_late",
        source_shape=[1, 9, 10, 2],
        filter_channels=2,
    )
    _add_wrong_way_branch(
        model_ir,
        name="protected_filter",
        source_shape=[1, 11, 12, 6],
        filter_channels=5,
    )
    _add_wrong_way_branch(
        model_ir,
        name="protected_output",
        source_shape=[1, 13, 14, 7],
        filter_channels=7,
        public_adapter=True,
    )
    return model_ir


def _run_legacy_wrong_way_sanitizer(model_ir: ModelIR) -> dict[str, int]:
    removed = 0
    expected_perm = [0, 2, 3, 1]
    model_outputs = {str(name) for name in model_ir.outputs}
    while True:
        changed = False
        consumers = lowering_module._build_tensor_consumer_map(model_ir)
        for op_index, op in enumerate(model_ir.operators):
            if (
                str(op.op_type) != "TRANSPOSE"
                or len(op.inputs) < 2
                or len(op.outputs) != 1
            ):
                continue
            if lowering_module._read_transpose_perm(model_ir, op) != expected_perm:
                continue
            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            if output_name in model_outputs:
                continue
            input_tensor = model_ir.tensors.get(input_name)
            output_tensor = model_ir.tensors.get(output_name)
            if (
                input_tensor is None
                or output_tensor is None
                or len(list(input_tensor.shape)) != 4
                or len(list(output_tensor.shape)) != 4
            ):
                continue
            user_indices = [int(value) for value in consumers.get(output_name, [])]
            if not user_indices:
                continue
            input_channels = int(input_tensor.shape[3])
            output_channels = int(output_tensor.shape[3])
            remove_this = True
            for user_index in user_indices:
                user_op = model_ir.operators[user_index]
                if (
                    str(user_op.op_type) != "CONV_2D"
                    or len(user_op.inputs) < 2
                    or str(user_op.inputs[0]) != output_name
                ):
                    remove_this = False
                    break
                filter_tensor = model_ir.tensors.get(str(user_op.inputs[1]))
                if filter_tensor is None or len(list(filter_tensor.shape)) != 4:
                    remove_this = False
                    break
                expected_channels = int(filter_tensor.shape[3])
                if not (
                    input_channels == expected_channels
                    and output_channels != expected_channels
                ):
                    remove_this = False
                    break
            if not remove_this:
                continue
            lowering_module._replace_tensor_inputs(
                model_ir,
                output_name,
                input_name,
            )
            del model_ir.operators[op_index]
            removed += 1
            changed = True
            break
        if not changed:
            break
    lowering_module._prune_unused_tensors(model_ir)
    return {
        "sanitized_wrong_way_nchw_to_nhwc_transpose_before_conv": removed,
    }


def test_indexed_wrong_way_conv_transpose_sanitizer_matches_legacy(
    monkeypatch,
) -> None:
    model_ir = _make_wrong_way_conv_transpose_model_ir()
    legacy_model_ir = copy.deepcopy(model_ir)
    legacy_stats = _run_legacy_wrong_way_sanitizer(legacy_model_ir)
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

    indexed_stats = _sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv(model_ir)

    assert refresh_count == 1
    assert indexed_stats == legacy_stats
    assert indexed_stats == {
        "sanitized_wrong_way_nchw_to_nhwc_transpose_before_conv": 2,
    }
    assert _normalize(model_ir) == _normalize(legacy_model_ir)


def test_indexed_wrong_way_conv_transpose_preserves_guards_and_index() -> None:
    model_ir = _make_wrong_way_conv_transpose_model_ir()
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {
        "sanitized_wrong_way_nchw_to_nhwc_transpose_before_conv": 2,
    }
    remaining_adapters = {
        str(op.outputs[0])
        for op in model_ir.operators
        if str(op.op_type) == "TRANSPOSE"
    }
    assert remaining_adapters == {
        "protected_nonconv_adapter",
        "protected_filter_adapter",
        "protected_output_adapter",
    }
    assert graph_index.consumer_indices("removable_multi_adapter") == []
    assert graph_index.consumer_indices("removable_late_adapter") == []
    removable_inputs = {
        str(op.outputs[0]): str(op.inputs[0])
        for op in model_ir.operators
        if str(op.op_type) == "CONV_2D" and str(op.outputs[0]).startswith("removable_")
    }
    assert removable_inputs == {
        "removable_multi_conv0_out": "removable_multi_source",
        "removable_multi_conv1_out": "removable_multi_source",
        "removable_late_conv0_out": "removable_late_source",
    }
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers
    assert graph_index.duplicate_producers == refreshed.duplicate_producers
    assert graph_index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert graph_index._operator_indices_by_type == refreshed._operator_indices_by_type


def test_wrong_way_conv_transpose_owner_skips_index_without_transpose(
    monkeypatch,
) -> None:
    model_ir = ModelIR("wrong_way_conv_transpose_no_candidate")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = _tensor("x", [1, 4])
    model_ir.tensors["y"] = _tensor("y", [1, 4])
    model_ir.tensors["dead"] = _tensor("dead", [1])
    model_ir.operators = [OperatorIR("RELU", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(owner_module, "ModelIRGraphIndex", unexpected_index)

    stats = owner_module.sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv(model_ir)

    assert stats == {
        "sanitized_wrong_way_nchw_to_nhwc_transpose_before_conv": 0,
    }
    assert "dead" not in model_ir.tensors


def test_swish_qdq_owner_delegates_independent_wrong_way_conv_safety_valve() -> None:
    model_ir = _make_wrong_way_conv_transpose_model_ir()
    legacy_model_ir = copy.deepcopy(model_ir)
    legacy_stats = _run_legacy_wrong_way_sanitizer(legacy_model_ir)

    stats = _optimize_transpose_swish_qdq_nhwc_islands(model_ir)

    assert legacy_stats == {
        "sanitized_wrong_way_nchw_to_nhwc_transpose_before_conv": 2,
    }
    assert stats == {
        "rewritten_transpose_swish_branches_to_nhwc": 0,
        "removed_transpose_swish_pre": 0,
        "propagated_nhwc_tensor_metadata": 0,
        "rewritten_concat_axis_to_nhwc": 0,
        "removed_transpose_swish_post": 2,
    }
    assert _normalize(model_ir) == _normalize(legacy_model_ir)
