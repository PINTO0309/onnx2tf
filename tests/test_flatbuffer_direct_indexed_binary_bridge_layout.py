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
    optimize_transpose_binary_bridges,
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
        dtype="INT64",
        data=np.asarray(_PERM, dtype=np.int64),
    )
    model_ir.tensors["inv_perm"] = _tensor(
        "inv_perm",
        [4],
        dtype="INT32",
        data=np.asarray(_INVERSE_PERM, dtype=np.int32),
    )


def _make_symmetric(
    *,
    mode: str,
    op_type: str = "SUB",
) -> ModelIR:
    model_ir = ModelIR(f"symmetric_{mode}_{op_type.lower()}")
    _add_permutations(model_ir)
    for name in ("x", "y"):
        model_ir.tensors[name] = _tensor(name, _RAW_SHAPE)
        model_ir.tensors[f"{name}_t"] = _tensor(
            f"{name}_t",
            _TRANSPOSED_SHAPE,
        )
    model_ir.tensors["z_t"] = _tensor("z_t", _TRANSPOSED_SHAPE)
    model_ir.inputs = ["x", "y"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["x_t"]),
        OperatorIR("TRANSPOSE", ["y", "perm"], ["y_t"]),
        OperatorIR(op_type, ["x_t", "y_t"], ["z_t"]),
    ]
    if mode == "single_post":
        model_ir.tensors["z"] = _tensor("z", _RAW_SHAPE)
        model_ir.operators.append(
            OperatorIR("TRANSPOSE", ["z_t", "inv_perm"], ["z"])
        )
        model_ir.outputs = ["z"]
    elif mode == "single_post_fanout":
        model_ir.tensors["z"] = _tensor("z", _RAW_SHAPE)
        model_ir.tensors["raw_result"] = _tensor("raw_result", _RAW_SHAPE)
        model_ir.tensors["legacy_result"] = _tensor(
            "legacy_result",
            _TRANSPOSED_SHAPE,
        )
        model_ir.operators.extend(
            [
                OperatorIR("TRANSPOSE", ["z_t", "inv_perm"], ["z"]),
                OperatorIR("RELU", ["z_t"], ["legacy_result"]),
                OperatorIR("RELU", ["z"], ["raw_result"]),
            ]
        )
        model_ir.outputs = ["raw_result", "legacy_result"]
    elif mode == "legacy_only":
        del model_ir.tensors["inv_perm"]
        model_ir.tensors["legacy_result"] = _tensor(
            "legacy_result",
            _TRANSPOSED_SHAPE,
        )
        model_ir.operators.append(
            OperatorIR("RELU", ["z_t"], ["legacy_result"])
        )
        model_ir.outputs = ["legacy_result"]
    else:
        raise ValueError(mode)
    return model_ir


def _make_asymmetric(
    *,
    transpose_on_lhs: bool,
    op_type: str = "DIV",
) -> ModelIR:
    model_ir = ModelIR(
        f"asymmetric_{'lhs' if transpose_on_lhs else 'rhs'}_{op_type.lower()}"
    )
    _add_permutations(model_ir)
    raw_name = "x" if transpose_on_lhs else "y"
    plain_name = "y" if transpose_on_lhs else "x"
    transposed_name = f"{raw_name}_t"
    model_ir.tensors[raw_name] = _tensor(raw_name, _RAW_SHAPE)
    model_ir.tensors[plain_name] = _tensor(plain_name, _TRANSPOSED_SHAPE)
    model_ir.tensors[transposed_name] = _tensor(
        transposed_name,
        _TRANSPOSED_SHAPE,
    )
    model_ir.tensors["z_t"] = _tensor("z_t", _TRANSPOSED_SHAPE)
    model_ir.tensors["z"] = _tensor("z", _RAW_SHAPE)
    model_ir.inputs = [raw_name, plain_name]
    binary_inputs = (
        [transposed_name, plain_name]
        if transpose_on_lhs
        else [plain_name, transposed_name]
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", [raw_name, "perm"], [transposed_name]),
        OperatorIR(op_type, binary_inputs, ["z_t"]),
        OperatorIR("TRANSPOSE", ["z_t", "inv_perm"], ["z"]),
    ]
    model_ir.outputs = ["z"]
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
        elif operator.op_type == "IDENTITY":
            result = inputs[0]
        else:
            raise AssertionError(operator.op_type)
        values[str(operator.outputs[0])] = result
    return {name: values[name] for name in model_ir.outputs}


def _fingerprint(model_ir: ModelIR):
    return (
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                tuple(sorted(operator.options.items())),
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
            )
            for name, tensor in model_ir.tensors.items()
        ),
    )


def _assert_same_index(
    maintained: ModelIRGraphIndex,
    model_ir: ModelIR,
) -> None:
    rebuilt = ModelIRGraphIndex(model_ir)
    assert maintained.producers == rebuilt.producers
    assert maintained.consumers == rebuilt.consumers
    assert maintained.duplicate_producers == rebuilt.duplicate_producers


@pytest.mark.parametrize("op_type", ["ADD", "SUB", "MUL", "DIV"])
@pytest.mark.parametrize(
    "mode",
    ["single_post", "single_post_fanout", "legacy_only"],
)
def test_indexed_symmetric_binary_bridge_is_numerically_equivalent(
    op_type: str,
    mode: str,
) -> None:
    model_ir = _make_symmetric(mode=mode, op_type=op_type)
    before = copy.deepcopy(model_ir)
    random = np.random.default_rng(7)
    feeds = {
        "x": random.uniform(0.5, 2.0, _RAW_SHAPE).astype(np.float32),
        "y": random.uniform(0.5, 2.0, _RAW_SHAPE).astype(np.float32),
    }
    expected = _evaluate(before, feeds)
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_binary_bridges(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats == {
        "removed_transpose_binary_bridges": 1,
        "removed_transpose_binary_asymmetric_bridges": 0,
    }
    actual = _evaluate(model_ir, feeds)
    for name in model_ir.outputs:
        np.testing.assert_allclose(actual[name], expected[name], rtol=0.0, atol=0.0)
    expected_transposes = 0 if mode == "single_post" else 1
    assert [op.op_type for op in model_ir.operators].count("TRANSPOSE") == expected_transposes
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_same_index(index, model_ir)
    assert optimize_transpose_binary_bridges(model_ir) == {
        "removed_transpose_binary_bridges": 0,
        "removed_transpose_binary_asymmetric_bridges": 0,
    }


@pytest.mark.parametrize("op_type", ["SUB", "DIV"])
@pytest.mark.parametrize("transpose_on_lhs", [True, False])
def test_indexed_asymmetric_binary_bridge_preserves_operand_order(
    op_type: str,
    transpose_on_lhs: bool,
) -> None:
    model_ir = _make_asymmetric(
        transpose_on_lhs=transpose_on_lhs,
        op_type=op_type,
    )
    before = copy.deepcopy(model_ir)
    random = np.random.default_rng(11)
    feeds = {
        name: random.uniform(0.5, 2.0, model_ir.tensors[name].shape).astype(
            np.float32
        )
        for name in model_ir.inputs
    }
    expected = _evaluate(before, feeds)
    index = ModelIRGraphIndex(model_ir)

    stats = optimize_transpose_binary_bridges(model_ir, graph_index=index)

    assert stats == {
        "removed_transpose_binary_bridges": 0,
        "removed_transpose_binary_asymmetric_bridges": 1,
    }
    actual = _evaluate(model_ir, feeds)
    np.testing.assert_allclose(actual["z"], expected["z"], rtol=0.0, atol=0.0)
    binary = next(operator for operator in model_ir.operators if operator.op_type == op_type)
    if transpose_on_lhs:
        assert binary.inputs == ["x", "x_t"]
    else:
        assert binary.inputs == ["y_t", "y"]
    assert binary.outputs == ["z"]
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    _assert_same_index(index, model_ir)


def test_indexed_binary_bridge_noop_is_transactional_for_per_axis_quantization() -> None:
    model_ir = _make_symmetric(mode="legacy_only")
    model_ir.tensors["z_t"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=1,
    )
    before = _fingerprint(model_ir)

    stats = optimize_transpose_binary_bridges(model_ir)

    assert stats == {
        "removed_transpose_binary_bridges": 0,
        "removed_transpose_binary_asymmetric_bridges": 0,
    }
    assert _fingerprint(model_ir) == before


@pytest.mark.parametrize("unsafe", ["public", "fused", "duplicate_producer"])
def test_indexed_binary_bridge_rejects_unsafe_symmetric_contracts(
    unsafe: str,
) -> None:
    model_ir = _make_symmetric(mode="single_post")
    binary = model_ir.operators[2]
    if unsafe == "public":
        model_ir.outputs.append("z_t")
    elif unsafe == "fused":
        binary.options["fusedActivationFunction"] = "RELU"
    else:
        model_ir.operators.insert(
            2,
            OperatorIR("IDENTITY", ["x"], ["x_t"]),
        )
    before = _fingerprint(model_ir)

    stats = optimize_transpose_binary_bridges(model_ir, candidate=binary)

    assert stats["removed_transpose_binary_bridges"] == 0
    assert stats["removed_transpose_binary_asymmetric_bridges"] == 0
    assert _fingerprint(model_ir) == before


def test_indexed_asymmetric_binary_bridge_rejects_late_plain_source() -> None:
    model_ir = _make_asymmetric(transpose_on_lhs=True)
    model_ir.inputs.remove("y")
    model_ir.inputs.append("y_source")
    model_ir.tensors["y_source"] = _tensor("y_source", _TRANSPOSED_SHAPE)
    model_ir.operators.insert(
        1,
        OperatorIR("IDENTITY", ["y_source"], ["y"]),
    )
    before = _fingerprint(model_ir)

    stats = optimize_transpose_binary_bridges(model_ir)

    assert stats["removed_transpose_binary_asymmetric_bridges"] == 0
    assert _fingerprint(model_ir) == before


def test_indexed_binary_bridge_honors_candidate_and_rewrite_limit() -> None:
    model_ir = _make_symmetric(mode="single_post", op_type="ADD")
    binary = model_ir.operators[2]
    assert optimize_transpose_binary_bridges(
        model_ir,
        max_rewrites=0,
        candidate=binary,
    ) == {
        "removed_transpose_binary_bridges": 0,
        "removed_transpose_binary_asymmetric_bridges": 0,
    }
    assert optimize_transpose_binary_bridges(
        model_ir,
        candidate=binary,
    )["removed_transpose_binary_bridges"] == 1
