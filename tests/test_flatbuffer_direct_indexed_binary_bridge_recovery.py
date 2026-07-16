from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.binary_bridge_layout import (
    optimize_transpose_binary_asymmetric_fanout_bridges,
    optimize_transpose_binary_full_post_fanout_bridges,
    optimize_transpose_binary_mixed_fanout_bridges_safe,
    optimize_transpose_binary_single_post_bridges_safe,
    optimize_transpose_binary_symmetric_legacy_only_bridges_safe,
    run_safe_binary_bridge_recovery,
)


_PERM = [0, 2, 3, 1]
_INVERSE_PERM = [0, 3, 1, 2]
_RAW_SHAPE = [1, 2, 3, 4]
_TRANSPOSED_SHAPE = [1, 3, 4, 2]


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    quantization=None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        quantization=quantization,
    )


def _add_permutations(model_ir: ModelIR) -> None:
    model_ir.tensors["perm"] = _tensor(
        "perm",
        [4],
        dtype="INT32",
        data=np.asarray(_PERM, dtype=np.int32),
    )
    model_ir.tensors["inv_perm"] = _tensor(
        "inv_perm",
        [4],
        dtype="INT64",
        data=np.asarray(_INVERSE_PERM, dtype=np.int64),
    )


def _make_multi_post(*, mode: str, op_type: str) -> ModelIR:
    model_ir = ModelIR(f"{mode}_{op_type.lower()}")
    _add_permutations(model_ir)
    for name in ("x", "y"):
        model_ir.tensors[name] = _tensor(name, _RAW_SHAPE)
        model_ir.tensors[f"{name}_t"] = _tensor(
            f"{name}_t",
            _TRANSPOSED_SHAPE,
        )
    model_ir.tensors["z_t"] = _tensor("z_t", _TRANSPOSED_SHAPE)
    for name in ("z0", "z1", "o0", "o1"):
        model_ir.tensors[name] = _tensor(name, _RAW_SHAPE)
    model_ir.inputs = ["x", "y"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["x_t"]),
        OperatorIR("TRANSPOSE", ["y", "perm"], ["y_t"]),
        OperatorIR(op_type, ["x_t", "y_t"], ["z_t"]),
        OperatorIR("TRANSPOSE", ["z_t", "inv_perm"], ["z0"]),
        OperatorIR("TRANSPOSE", ["z_t", "inv_perm"], ["z1"]),
    ]
    if mode == "mixed":
        model_ir.tensors["legacy"] = _tensor("legacy", _TRANSPOSED_SHAPE)
        model_ir.operators.append(OperatorIR("RELU", ["z_t"], ["legacy"]))
        model_ir.outputs.append("legacy")
    elif mode != "full":
        raise ValueError(mode)
    model_ir.operators.extend(
        [
            OperatorIR("RELU", ["z0"], ["o0"]),
            OperatorIR("RELU", ["z1"], ["o1"]),
        ]
    )
    model_ir.outputs.extend(["o0", "o1"])
    return model_ir


def _make_asymmetric_fanout(
    *,
    transpose_on_lhs: bool,
    op_type: str,
) -> ModelIR:
    model_ir = ModelIR(
        f"asymmetric_fanout_{'lhs' if transpose_on_lhs else 'rhs'}_{op_type.lower()}"
    )
    _add_permutations(model_ir)
    raw_name = "x" if transpose_on_lhs else "y"
    plain_name = "y" if transpose_on_lhs else "x"
    transposed_name = f"{raw_name}_t"
    plain_raw_name = f"{plain_name}_raw"
    model_ir.tensors[raw_name] = _tensor(raw_name, _RAW_SHAPE)
    model_ir.tensors[plain_name] = _tensor(plain_name, _TRANSPOSED_SHAPE)
    model_ir.tensors[transposed_name] = _tensor(
        transposed_name,
        _TRANSPOSED_SHAPE,
    )
    model_ir.tensors[plain_raw_name] = _tensor(plain_raw_name, _RAW_SHAPE)
    model_ir.tensors["z_t"] = _tensor("z_t", _TRANSPOSED_SHAPE)
    model_ir.tensors["z"] = _tensor("z", _RAW_SHAPE)
    model_ir.tensors["raw_result"] = _tensor("raw_result", _RAW_SHAPE)
    model_ir.tensors["legacy_result"] = _tensor(
        "legacy_result",
        _TRANSPOSED_SHAPE,
    )
    model_ir.inputs = [raw_name, plain_name]
    binary_inputs = (
        [transposed_name, plain_name]
        if transpose_on_lhs
        else [plain_name, transposed_name]
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", [raw_name, "perm"], [transposed_name]),
        OperatorIR(
            "TRANSPOSE",
            [plain_name, "inv_perm"],
            [plain_raw_name],
        ),
        OperatorIR(op_type, binary_inputs, ["z_t"]),
        OperatorIR("TRANSPOSE", ["z_t", "inv_perm"], ["z"]),
        OperatorIR("RELU", ["z_t"], ["legacy_result"]),
        OperatorIR("RELU", ["z"], ["raw_result"]),
    ]
    model_ir.outputs = ["raw_result", "legacy_result"]
    return model_ir


def _evaluate(
    model_ir: ModelIR,
    feeds: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    values = {str(name): np.asarray(value) for name, value in feeds.items()}
    for name, tensor in model_ir.tensors.items():
        if tensor.data is not None:
            values[str(name)] = np.asarray(tensor.data)
    for operator in model_ir.operators:
        inputs = [values[str(name)] for name in operator.inputs]
        if operator.op_type == "TRANSPOSE":
            result = np.transpose(inputs[0], tuple(int(v) for v in inputs[1]))
        elif operator.op_type == "ADD":
            result = inputs[0] + inputs[1]
        elif operator.op_type == "SUB":
            result = inputs[0] - inputs[1]
        elif operator.op_type == "MUL":
            result = inputs[0] * inputs[1]
        elif operator.op_type == "DIV":
            result = inputs[0] / inputs[1]
        elif operator.op_type == "RELU":
            result = np.maximum(inputs[0], 0.0)
        else:
            raise AssertionError(operator.op_type)
        values[str(operator.outputs[0])] = result
    return {name: values[name] for name in model_ir.outputs}


def _feeds(model_ir: ModelIR) -> Dict[str, np.ndarray]:
    random = np.random.default_rng(19)
    return {
        name: random.uniform(0.5, 2.0, model_ir.tensors[name].shape).astype(
            np.float32
        )
        for name in model_ir.inputs
    }


def _fingerprint(model_ir: ModelIR):
    return (
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                repr(operator.options),
            )
            for operator in model_ir.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or ()),
                repr(tensor.quantization),
                None
                if tensor.data is None
                else (
                    str(np.asarray(tensor.data).dtype),
                    tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                ),
            )
            for name, tensor in model_ir.tensors.items()
        ),
    )


def _assert_equivalent(before: ModelIR, after: ModelIR) -> None:
    feeds = _feeds(before)
    expected = _evaluate(before, feeds)
    actual = _evaluate(after, feeds)
    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=0.0, atol=0.0)


def _assert_index_current(index: ModelIRGraphIndex, model_ir: ModelIR) -> None:
    rebuilt = ModelIRGraphIndex(model_ir)
    assert index.producers == rebuilt.producers
    assert index.consumers == rebuilt.consumers
    assert index.duplicate_producers == rebuilt.duplicate_producers


@pytest.mark.parametrize("op_type", ["ADD", "SUB", "MUL", "DIV"])
@pytest.mark.parametrize("mode", ["mixed", "full"])
def test_indexed_multi_post_recovery_is_numerically_equivalent(
    op_type: str,
    mode: str,
) -> None:
    model_ir = _make_multi_post(mode=mode, op_type=op_type)
    before = copy.deepcopy(model_ir)
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    optimize = (
        optimize_transpose_binary_mixed_fanout_bridges_safe
        if mode == "mixed"
        else optimize_transpose_binary_full_post_fanout_bridges
    )

    stats = optimize(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    expected_key = (
        "rewritten_transpose_binary_mixed_fanout_bridges_safe"
        if mode == "mixed"
        else "rewritten_transpose_binary_full_post_fanout_bridges"
    )
    assert stats == {expected_key: 1}
    _assert_equivalent(before, model_ir)
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(index, model_ir)
    assert optimize(model_ir) == {expected_key: 0}


@pytest.mark.parametrize("op_type", ["SUB", "DIV"])
@pytest.mark.parametrize("transpose_on_lhs", [True, False])
def test_indexed_asymmetric_fanout_preserves_operand_order(
    op_type: str,
    transpose_on_lhs: bool,
) -> None:
    model_ir = _make_asymmetric_fanout(
        transpose_on_lhs=transpose_on_lhs,
        op_type=op_type,
    )
    before = copy.deepcopy(model_ir)
    index = ModelIRGraphIndex(model_ir)

    stats = optimize_transpose_binary_asymmetric_fanout_bridges(
        model_ir,
        graph_index=index,
    )

    assert stats == {
        "rewritten_transpose_binary_asymmetric_fanout_bridges": 1,
    }
    _assert_equivalent(before, model_ir)
    binary = next(operator for operator in model_ir.operators if operator.op_type == op_type)
    if transpose_on_lhs:
        assert binary.inputs == ["x", "y_raw"]
    else:
        assert binary.inputs == ["x_raw", "y"]
    assert binary.outputs == ["z"]
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    _assert_index_current(index, model_ir)


def test_indexed_safe_sequence_preserves_phase_stats_and_limit() -> None:
    model_ir = _make_multi_post(mode="mixed", op_type="ADD")
    binary = model_ir.operators[2]
    zero_stats = run_safe_binary_bridge_recovery(
        model_ir,
        max_rewrites=0,
        candidate=binary,
    )
    assert zero_stats == {
        "rewritten_transpose_binary_symmetric_legacy_only_bridges_safe": 0,
        "rewritten_transpose_binary_single_post_bridges_safe": 0,
        "rewritten_transpose_binary_mixed_fanout_bridges_safe": 0,
        "rewritten_transpose_binary_asymmetric_fanout_bridges": 0,
        "rewritten_transpose_binary_full_post_fanout_bridges": 0,
    }

    stats = run_safe_binary_bridge_recovery(model_ir, candidate=binary)

    assert stats == {
        "rewritten_transpose_binary_symmetric_legacy_only_bridges_safe": 0,
        "rewritten_transpose_binary_single_post_bridges_safe": 0,
        "rewritten_transpose_binary_mixed_fanout_bridges_safe": 1,
        "rewritten_transpose_binary_asymmetric_fanout_bridges": 0,
        "rewritten_transpose_binary_full_post_fanout_bridges": 0,
    }


def test_indexed_safe_legacy_phase_marks_inserted_boundary() -> None:
    model_ir = _make_multi_post(mode="mixed", op_type="ADD")
    model_ir.operators = [
        operator
        for operator in model_ir.operators
        if operator.outputs not in (["z0"], ["z1"], ["o0"], ["o1"])
    ]
    model_ir.outputs = ["legacy"]

    stats = optimize_transpose_binary_symmetric_legacy_only_bridges_safe(
        model_ir
    )

    assert stats == {
        "rewritten_transpose_binary_symmetric_legacy_only_bridges_safe": 1,
    }
    adapter = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE" and operator.outputs == ["z_t"]
    )
    assert adapter.options == {"__preserve_layout_boundary__": True}


def test_indexed_safe_single_post_phase_marks_retained_boundary() -> None:
    model_ir = _make_multi_post(mode="mixed", op_type="ADD")
    model_ir.operators = [
        operator
        for operator in model_ir.operators
        if operator.outputs not in (["z1"], ["o1"])
    ]
    model_ir.outputs = ["legacy", "o0"]

    stats = optimize_transpose_binary_single_post_bridges_safe(model_ir)

    assert stats == {
        "rewritten_transpose_binary_single_post_bridges_safe": 1,
    }
    adapter = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE" and operator.outputs == ["z_t"]
    )
    assert adapter.inputs == ["z0", "perm"]
    assert adapter.options == {"__preserve_layout_boundary__": True}


def test_indexed_safe_single_post_repairs_late_post_producer_order() -> None:
    model_ir = _make_multi_post(mode="mixed", op_type="ADD")
    model_ir.operators = [
        operator
        for operator in model_ir.operators
        if operator.outputs not in (["z1"], ["o1"])
    ]
    model_ir.outputs = ["legacy", "o0"]
    final_consumer = next(
        operator for operator in model_ir.operators if operator.outputs == ["o0"]
    )
    model_ir.operators.remove(final_consumer)
    post_index = next(
        index
        for index, operator in enumerate(model_ir.operators)
        if operator.outputs == ["z0"]
    )
    model_ir.operators.insert(post_index, final_consumer)

    stats = optimize_transpose_binary_single_post_bridges_safe(model_ir)

    assert stats == {
        "rewritten_transpose_binary_single_post_bridges_safe": 1,
    }
    index = ModelIRGraphIndex(model_ir)
    producer_index = index.producers["z0"]
    assert model_ir.operators[producer_index].op_type == "ADD"
    assert all(
        producer_index < consumer_index
        for consumer_index in index.consumer_indices("z0")
    )
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []


def test_indexed_mixed_fanout_does_not_mutate_shared_inverse_permutation() -> None:
    model_ir = _make_multi_post(mode="mixed", op_type="ADD")
    model_ir.tensors["external"] = _tensor("external", _TRANSPOSED_SHAPE)
    model_ir.tensors["external_raw"] = _tensor("external_raw", _RAW_SHAPE)
    model_ir.inputs.append("external")
    model_ir.outputs.append("external_raw")
    model_ir.operators.append(
        OperatorIR(
            "TRANSPOSE",
            ["external", "inv_perm"],
            ["external_raw"],
        )
    )
    expected_inverse = np.asarray(model_ir.tensors["inv_perm"].data).copy()

    stats = optimize_transpose_binary_mixed_fanout_bridges_safe(model_ir)

    assert stats["rewritten_transpose_binary_mixed_fanout_bridges_safe"] == 1
    np.testing.assert_array_equal(
        model_ir.tensors["inv_perm"].data,
        expected_inverse,
    )
    kept = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE" and operator.outputs == ["z_t"]
    )
    assert kept.inputs == ["z0", "perm"]


def test_indexed_mixed_fanout_rejects_adapter_after_legacy_consumer() -> None:
    model_ir = _make_multi_post(mode="mixed", op_type="ADD")
    legacy = next(
        operator for operator in model_ir.operators if operator.outputs == ["legacy"]
    )
    model_ir.operators.remove(legacy)
    model_ir.operators.insert(3, legacy)
    before = _fingerprint(model_ir)

    stats = optimize_transpose_binary_mixed_fanout_bridges_safe(model_ir)

    assert stats["rewritten_transpose_binary_mixed_fanout_bridges_safe"] == 0
    assert _fingerprint(model_ir) == before


def test_indexed_asymmetric_fanout_rejects_late_plain_transpose() -> None:
    model_ir = _make_asymmetric_fanout(transpose_on_lhs=True, op_type="ADD")
    plain_transpose = model_ir.operators.pop(1)
    model_ir.operators.insert(3, plain_transpose)
    before = _fingerprint(model_ir)

    stats = optimize_transpose_binary_asymmetric_fanout_bridges(model_ir)

    assert stats["rewritten_transpose_binary_asymmetric_fanout_bridges"] == 0
    assert _fingerprint(model_ir) == before


def test_indexed_multi_post_rejects_per_axis_quantization_transactionally() -> None:
    model_ir = _make_multi_post(mode="full", op_type="ADD")
    model_ir.tensors["z_t"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=1,
    )
    before = _fingerprint(model_ir)

    stats = optimize_transpose_binary_full_post_fanout_bridges(model_ir)

    assert stats["rewritten_transpose_binary_full_post_fanout_bridges"] == 0
    assert _fingerprint(model_ir) == before
