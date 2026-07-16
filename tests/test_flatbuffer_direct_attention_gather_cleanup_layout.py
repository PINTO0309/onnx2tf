from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.validation import (
    validate_model_ir_invariants,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_attention_gather_transpose_reshape_cleanup_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_ZERO_STATS = {
    "optimized_attention_gather_transpose_reshape_cleanup_pattern_a": 0,
    "optimized_attention_gather_transpose_reshape_cleanup_pattern_b": 0,
}
_PATTERN_A_STATS = {
    "optimized_attention_gather_transpose_reshape_cleanup_pattern_a": 1,
    "optimized_attention_gather_transpose_reshape_cleanup_pattern_b": 0,
}
_PATTERN_B_STATS = {
    "optimized_attention_gather_transpose_reshape_cleanup_pattern_a": 0,
    "optimized_attention_gather_transpose_reshape_cleanup_pattern_b": 1,
}


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
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
    is_variable: bool = False,
    quantization: QuantParamIR | None = None,
    onnx_tensor_name: str | None = None,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=(list(signature) if signature is not None else list(shape)),
        data=data,
        is_variable=bool(is_variable),
        quantization=quantization,
        onnx_tensor_name=onnx_tensor_name,
    )


def _const_ints(
    model_ir: ModelIR,
    name: str,
    values: list[int],
) -> None:
    _tensor(
        model_ir,
        name,
        [len(values)],
        dtype="INT32",
        data=np.asarray(values, dtype=np.int32),
        onnx_tensor_name=f"onnx::{name}",
    )


def _add_pattern_a(
    model_ir: ModelIR,
    *,
    prefix: str,
    negative_axis: bool = False,
) -> None:
    names = {
        "x": f"{prefix}x",
        "index": f"{prefix}index",
        "gather": f"{prefix}gather",
        "perm": f"{prefix}perm",
        "transpose": f"{prefix}transpose",
        "shape": f"{prefix}shape",
        "reshape": f"{prefix}reshape",
        "output": f"{prefix}output",
    }
    model_ir.inputs.append(names["x"])
    model_ir.outputs.append(names["output"])

    for name, shape in (
        (names["x"], [1, 2, 3, 4]),
        (names["gather"], [2, 3, 4]),
        (names["transpose"], [3, 2, 4]),
        (names["reshape"], [1, 1, 3, 8]),
        (names["output"], [1, 1, 3, 8]),
    ):
        _tensor(
            model_ir,
            name,
            shape,
            onnx_tensor_name=f"onnx::{name}",
        )
    _const_ints(model_ir, names["index"], [0])
    _const_ints(model_ir, names["perm"], [1, 0, 2])
    _const_ints(model_ir, names["shape"], [1, 1, 3, 8])

    model_ir.operators.extend(
        [
            OperatorIR(
                "GATHER",
                [names["x"], names["index"]],
                [names["gather"]],
                options={"axis": -4 if negative_axis else 0},
                axis_semantics={"axis": "logical"},
                version=2,
                onnx_node_name=f"{prefix}gather_node",
                onnx_op_type="Gather",
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["gather"], names["perm"]],
                [names["transpose"]],
                options={
                    "perm": [1, 0, 2],
                    "onnxPerm": [1, 0, 2],
                    "marker": prefix,
                },
                version=3,
                onnx_node_name=f"{prefix}transpose_node",
                onnx_op_type="Transpose",
            ),
            OperatorIR(
                "RESHAPE",
                [names["transpose"], names["shape"]],
                [names["reshape"]],
                options={"newShape": [1, 1, 3, 8]},
                version=2,
                onnx_node_name=f"{prefix}reshape_node",
                onnx_op_type="Reshape",
            ),
            OperatorIR(
                "IDENTITY",
                [names["reshape"]],
                [names["output"]],
                onnx_node_name=f"{prefix}identity_node",
                onnx_op_type="Identity",
            ),
        ]
    )


def _add_pattern_b(
    model_ir: ModelIR,
    *,
    prefix: str,
    negative_axes: bool = False,
) -> None:
    names = {
        "x": f"{prefix}x",
        "index0": f"{prefix}index0",
        "gather0": f"{prefix}gather0",
        "index1": f"{prefix}index1",
        "gather1": f"{prefix}gather1",
        "shape": f"{prefix}shape",
        "reshape": f"{prefix}reshape",
        "output": f"{prefix}output",
    }
    model_ir.inputs.append(names["x"])
    model_ir.outputs.append(names["output"])

    for name, shape in (
        (names["x"], [1, 1, 3, 8]),
        (names["gather0"], [1, 3, 8]),
        (names["gather1"], [3, 8]),
        (names["reshape"], [1, 1, 3, 8]),
        (names["output"], [1, 1, 3, 8]),
    ):
        _tensor(
            model_ir,
            name,
            shape,
            onnx_tensor_name=f"onnx::{name}",
        )
    _const_ints(model_ir, names["index0"], [0])
    _const_ints(model_ir, names["index1"], [0])
    _const_ints(model_ir, names["shape"], [1, 1, 3, 8])

    model_ir.operators.extend(
        [
            OperatorIR(
                "GATHER",
                [names["x"], names["index0"]],
                [names["gather0"]],
                options={"axis": -4 if negative_axes else 0},
                onnx_node_name=f"{prefix}gather0_node",
                onnx_op_type="Gather",
            ),
            OperatorIR(
                "GATHER",
                [names["gather0"], names["index1"]],
                [names["gather1"]],
                options={"axis": -3 if negative_axes else 0},
                onnx_node_name=f"{prefix}gather1_node",
                onnx_op_type="Gather",
            ),
            OperatorIR(
                "RESHAPE",
                [names["gather1"], names["shape"]],
                [names["reshape"]],
                onnx_node_name=f"{prefix}reshape_node",
                onnx_op_type="Reshape",
            ),
            OperatorIR(
                "IDENTITY",
                [names["reshape"]],
                [names["output"]],
                onnx_node_name=f"{prefix}identity_node",
                onnx_op_type="Identity",
            ),
        ]
    )


def _model(
    *,
    pattern_a: int = 0,
    pattern_b: int = 0,
    negative_axes: bool = False,
) -> ModelIR:
    model_ir = ModelIR("attention_gather_cleanup_characterization")
    for index in range(int(pattern_a)):
        _add_pattern_a(
            model_ir,
            prefix=f"a{index}_",
            negative_axis=negative_axes,
        )
    for index in range(int(pattern_b)):
        _add_pattern_b(
            model_ir,
            prefix=f"b{index}_",
            negative_axes=negative_axes,
        )
    return model_ir


def _evaluate(
    model_ir: ModelIR,
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    values = {
        name: np.asarray(tensor.data)
        for name, tensor in model_ir.tensors.items()
        if tensor.data is not None
    }
    values.update({name: np.asarray(value) for name, value in feeds.items()})
    for op in model_ir.operators:
        if op.op_type == "GATHER":
            indices = np.asarray(values[op.inputs[1]]).reshape(-1)
            assert indices.size == 1
            result = np.take(
                values[op.inputs[0]],
                int(indices[0]),
                axis=int(op.options.get("axis", 0)),
            )
        elif op.op_type == "TRANSPOSE":
            perm = np.asarray(values[op.inputs[1]]).reshape(-1).tolist()
            result = np.transpose(values[op.inputs[0]], perm)
        elif op.op_type == "RESHAPE":
            shape = np.asarray(values[op.inputs[1]]).reshape(-1).tolist()
            result = np.reshape(values[op.inputs[0]], shape)
        elif op.op_type == "IDENTITY":
            result = values[op.inputs[0]]
        else:
            raise AssertionError(f"unsupported test operator: {op.op_type}")
        assert len(op.outputs) == 1
        values[op.outputs[0]] = np.asarray(result)
    return {name: values[name] for name in model_ir.outputs}


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _ZERO_STATS
    assert _normalize(model_ir) == before


def _append_identity_consumer(
    model_ir: ModelIR,
    input_name: str,
    output_name: str,
) -> None:
    source = model_ir.tensors[input_name]
    _tensor(
        model_ir,
        output_name,
        list(source.shape),
        signature=list(source.shape_signature or source.shape),
    )
    model_ir.outputs.append(output_name)
    model_ir.operators.append(OperatorIR("IDENTITY", [input_name], [output_name]))


def test_attention_gather_cleanup_rewrites_pattern_a() -> None:
    model_ir = _model(pattern_a=1)

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _PATTERN_A_STATS
    assert [op.op_type for op in model_ir.operators] == [
        "TRANSPOSE",
        "RESHAPE",
        "IDENTITY",
    ]
    transpose = model_ir.operators[0]
    assert transpose.inputs == ["a0_x", "a0_perm"]
    assert transpose.outputs == ["a0_transpose"]
    assert transpose.options == {
        "perm": [0, 2, 1, 3],
        "onnxPerm": [0, 2, 1, 3],
        "marker": "a0_",
    }
    assert transpose.version == 3
    assert transpose.onnx_node_name == "a0_transpose_node"
    assert transpose.onnx_op_type == "Transpose"
    assert np.asarray(model_ir.tensors["a0_perm"].data).tolist() == [
        0,
        2,
        1,
        3,
    ]
    assert model_ir.tensors["a0_transpose"].shape == [1, 3, 2, 4]
    assert model_ir.tensors["a0_transpose"].shape_signature == [1, 3, 2, 4]
    assert "a0_gather" not in model_ir.tensors
    assert "a0_index" not in model_ir.tensors
    assert validate_model_ir_invariants(model_ir) == []


def test_attention_gather_cleanup_rewrites_pattern_b_to_identity() -> None:
    model_ir = _model(pattern_b=1)

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _PATTERN_B_STATS
    assert len(model_ir.operators) == 1
    identity = model_ir.operators[0]
    assert identity.op_type == "IDENTITY"
    assert identity.inputs == ["b0_x"]
    assert identity.outputs == ["b0_output"]
    assert set(model_ir.tensors) == {"b0_x", "b0_output"}
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("pattern", ["a", "b"])
def test_attention_gather_cleanup_is_numerically_exact(pattern: str) -> None:
    model_ir = _model(
        pattern_a=1 if pattern == "a" else 0,
        pattern_b=1 if pattern == "b" else 0,
    )
    before = copy.deepcopy(model_ir)
    input_name = f"{pattern}0_x"
    shape = model_ir.tensors[input_name].shape
    feed = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = _evaluate(before, {input_name: feed})

    _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)
    actual = _evaluate(model_ir, {input_name: feed})

    assert actual.keys() == expected.keys()
    for name in expected:
        max_abs_error = float(np.max(np.abs(actual[name] - expected[name])))
        assert max_abs_error == 0.0


def test_attention_gather_cleanup_rewrites_multiple_and_fixed_point() -> None:
    model_ir = _model(pattern_a=2, pattern_b=2)

    first = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)
    after_first = _normalize(copy.deepcopy(model_ir))
    second = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert first == {
        "optimized_attention_gather_transpose_reshape_cleanup_pattern_a": 2,
        "optimized_attention_gather_transpose_reshape_cleanup_pattern_b": 2,
    }
    assert second == _ZERO_STATS
    assert _normalize(model_ir) == after_first
    assert validate_model_ir_invariants(model_ir) == []


def test_attention_gather_cleanup_accepts_normalized_negative_axes() -> None:
    model_ir = _model(pattern_a=1, pattern_b=1, negative_axes=True)

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == {
        "optimized_attention_gather_transpose_reshape_cleanup_pattern_a": 1,
        "optimized_attention_gather_transpose_reshape_cleanup_pattern_b": 1,
    }
    assert validate_model_ir_invariants(model_ir) == []


def test_attention_gather_cleanup_clones_shared_permutation_collision() -> None:
    model_ir = _model(pattern_a=1)
    original = np.asarray(model_ir.tensors["a0_perm"].data).copy()
    _append_identity_consumer(model_ir, "a0_perm", "perm_copy")
    _const_ints(model_ir, "a0_perm_qkv_perm4", [99])
    model_ir.outputs.append("a0_perm_qkv_perm4")

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _PATTERN_A_STATS
    transpose = next(op for op in model_ir.operators if op.op_type == "TRANSPOSE")
    assert transpose.inputs[1] == "a0_perm_qkv_perm4_1"
    assert np.array_equal(model_ir.tensors["a0_perm"].data, original)
    assert np.asarray(model_ir.tensors["a0_perm_qkv_perm4"].data).tolist() == [99]
    assert np.asarray(model_ir.tensors["a0_perm_qkv_perm4_1"].data).tolist() == [
        0,
        2,
        1,
        3,
    ]


def test_attention_gather_cleanup_preserves_public_pattern_a_reshape() -> None:
    model_ir = _model(pattern_a=1)
    model_ir.outputs.append("a0_reshape")

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _PATTERN_A_STATS
    assert model_ir.outputs == ["a0_output", "a0_reshape"]
    assert "a0_reshape" in model_ir.tensors
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.operators[0].options.__setitem__("axis", 1),
            id="pattern-a-wrong-axis",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["a0_index"].__setattr__(
                "data", np.asarray([1], dtype=np.int32)
            ),
            id="pattern-a-nonzero-index",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["a0_x"].__setattr__(
                "shape", [2, 2, 3, 4]
            ),
            id="pattern-a-nonsingleton-leading-dimension",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("a0_gather"),
            id="pattern-a-public-gather",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir, "a0_gather", "gather_copy"
            ),
            id="pattern-a-gather-fanout",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["a0_perm"].__setattr__(
                "data", np.asarray([0, 2, 1], dtype=np.int32)
            ),
            id="pattern-a-wrong-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("a0_transpose"),
            id="pattern-a-public-transpose",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir, "a0_transpose", "transpose_copy"
            ),
            id="pattern-a-transpose-fanout",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["a0_transpose"].__setattr__(
                "shape", [3, 4, 2]
            ),
            id="pattern-a-wrong-transpose-shape",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["a0_shape"].__setattr__(
                "data", np.asarray([1, 1, 3, 7], dtype=np.int32)
            ),
            id="pattern-a-wrong-reshape-target",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["a0_x"].__setattr__(
                "shape", [1, 2, -1, 4]
            ),
            id="pattern-a-dynamic-shape",
        ),
    ],
)
def test_attention_gather_cleanup_pattern_a_preserves_rejections(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model(pattern_a=1)
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    [
        "wrong-axis0",
        "wrong-axis1",
        "nonzero-index0",
        "nonzero-index1",
        "nonsingleton-input",
        "nonsingleton-intermediate",
        "public-gather0",
        "public-gather1",
        "gather0-fanout",
        "gather1-fanout",
        "wrong-target",
        "public-reshape",
        "dynamic-input",
    ],
)
def test_attention_gather_cleanup_pattern_b_preserves_rejections(
    case: str,
) -> None:
    model_ir = _model(pattern_b=1)
    if case == "wrong-axis0":
        model_ir.operators[0].options["axis"] = 1
    elif case == "wrong-axis1":
        model_ir.operators[1].options["axis"] = 1
    elif case == "nonzero-index0":
        model_ir.tensors["b0_index0"].data = np.asarray([1], dtype=np.int32)
    elif case == "nonzero-index1":
        model_ir.tensors["b0_index1"].data = np.asarray([1], dtype=np.int32)
    elif case == "nonsingleton-input":
        model_ir.tensors["b0_x"].shape[0] = 2
    elif case == "nonsingleton-intermediate":
        model_ir.tensors["b0_gather0"].shape[0] = 2
    elif case == "public-gather0":
        model_ir.outputs.append("b0_gather0")
    elif case == "public-gather1":
        model_ir.outputs.append("b0_gather1")
    elif case == "gather0-fanout":
        _append_identity_consumer(model_ir, "b0_gather0", "gather0_copy")
    elif case == "gather1-fanout":
        _append_identity_consumer(model_ir, "b0_gather1", "gather1_copy")
    elif case == "wrong-target":
        model_ir.tensors["b0_shape"].data = np.asarray([1, 1, 3, 7], dtype=np.int32)
    elif case == "public-reshape":
        model_ir.outputs.append("b0_reshape")
    elif case == "dynamic-input":
        model_ir.tensors["b0_x"].shape[2] = -1

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="no-match execution prunes tensors instead of remaining a no-op",
)
def test_attention_gather_cleanup_does_not_prune_unmatched_graph() -> None:
    model_ir = _model(pattern_a=1)
    _tensor(model_ir, "unused", [1])
    model_ir.tensors["a0_perm"].data = np.asarray([0, 2, 1], dtype=np.int32)

    _assert_transactional_rejection(model_ir)


_UNSAFE_CONSTANT_ROLES = [
    "a-index",
    "a-perm",
    "a-shape",
    "b-index0",
    "b-index1",
    "b-shape",
]


@pytest.mark.xfail(
    strict=True,
    reason="matched constants lack an immutable ownership and type plan",
)
@pytest.mark.parametrize("role", _UNSAFE_CONSTANT_ROLES)
@pytest.mark.parametrize(
    "condition",
    ["public-input", "variable", "wrong-dtype", "wrong-buffer", "quantized"],
)
def test_attention_gather_cleanup_rejects_unsafe_match_constant(
    role: str,
    condition: str,
) -> None:
    pattern = role[0]
    model_ir = _model(
        pattern_a=1 if pattern == "a" else 0,
        pattern_b=1 if pattern == "b" else 0,
    )
    tensor_name = {
        "a-index": "a0_index",
        "a-perm": "a0_perm",
        "a-shape": "a0_shape",
        "b-index0": "b0_index0",
        "b-index1": "b0_index1",
        "b-shape": "b0_shape",
    }[role]
    tensor = model_ir.tensors[tensor_name]
    if condition == "public-input":
        model_ir.inputs.append(tensor_name)
    elif condition == "variable":
        tensor.is_variable = True
    elif condition == "wrong-dtype":
        tensor.dtype = "FLOAT32"
        tensor.data = np.asarray(tensor.data, dtype=np.float32)
    elif condition == "wrong-buffer":
        tensor.data = np.asarray(tensor.data, dtype=np.int64)
    elif condition == "quantized":
        tensor.quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="public permutation outputs are mutated instead of cloned",
)
def test_attention_gather_cleanup_clones_public_permutation_output() -> None:
    model_ir = _model(pattern_a=1)
    original = np.asarray(model_ir.tensors["a0_perm"].data).copy()
    model_ir.outputs.append("a0_perm")

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _PATTERN_A_STATS
    transpose = next(op for op in model_ir.operators if op.op_type == "TRANSPOSE")
    assert transpose.inputs[1] != "a0_perm"
    assert np.array_equal(model_ir.tensors["a0_perm"].data, original)


@pytest.mark.xfail(
    strict=True,
    reason="shared permutation clones drop tensor provenance metadata",
)
def test_attention_gather_cleanup_preserves_shared_permutation_metadata() -> None:
    model_ir = _model(pattern_a=1)
    _append_identity_consumer(model_ir, "a0_perm", "perm_copy")

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _PATTERN_A_STATS
    transpose = next(op for op in model_ir.operators if op.op_type == "TRANSPOSE")
    clone = model_ir.tensors[transpose.inputs[1]]
    assert clone.onnx_tensor_name == "onnx::a0_perm"


@pytest.mark.xfail(
    strict=True,
    reason="multi-element zero indices are treated as scalar Gather indices",
)
@pytest.mark.parametrize("pattern", ["a", "b"])
def test_attention_gather_cleanup_rejects_multi_element_index(
    pattern: str,
) -> None:
    model_ir = _model(
        pattern_a=1 if pattern == "a" else 0,
        pattern_b=1 if pattern == "b" else 0,
    )
    index_name = "a0_index" if pattern == "a" else "b0_index0"
    tensor = model_ir.tensors[index_name]
    tensor.shape = [2]
    tensor.shape_signature = [2]
    tensor.data = np.asarray([0, 0], dtype=np.int32)

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="Pattern A collapses dynamic signatures to concrete dimensions",
)
def test_attention_gather_cleanup_preserves_dynamic_signature() -> None:
    model_ir = _model(pattern_a=1)
    model_ir.tensors["a0_x"].shape_signature = [1, 2, -1, 4]
    model_ir.tensors["a0_gather"].shape_signature = [2, -1, 4]
    model_ir.tensors["a0_transpose"].shape_signature = [-1, 2, 4]

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _PATTERN_A_STATS
    assert model_ir.tensors["a0_transpose"].shape_signature == [1, -1, 2, 4]


@pytest.mark.xfail(
    strict=True,
    reason="Pattern A does not remap per-axis quantized dimensions",
)
def test_attention_gather_cleanup_remaps_per_axis_quantization() -> None:
    model_ir = _model(pattern_a=1)
    model_ir.tensors["a0_transpose"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=1,
    )

    stats = _optimize_attention_gather_transpose_reshape_cleanup_chains(model_ir)

    assert stats == _PATTERN_A_STATS
    quantization = model_ir.tensors["a0_transpose"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 2


@pytest.mark.xfail(
    strict=True,
    reason="Pattern B does not validate exact intermediate Gather shapes",
)
def test_attention_gather_cleanup_rejects_inconsistent_intermediate_shape() -> None:
    model_ir = _model(pattern_b=1)
    model_ir.tensors["b0_gather0"].shape = [1, 99, 8]
    model_ir.tensors["b0_gather0"].shape_signature = [1, 99, 8]

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="Pattern B bypasses incompatible output quantization metadata",
)
def test_attention_gather_cleanup_rejects_quantization_mismatch() -> None:
    model_ir = _model(pattern_b=1)
    model_ir.tensors["b0_reshape"].quantization = QuantParamIR(
        scale=[0.25],
        zero_point=[3],
        quantized_dimension=0,
    )

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="complete tensor metadata is not prevalidated transactionally",
)
@pytest.mark.parametrize(
    "case",
    [
        "short-input-signature",
        "short-transpose-signature",
        "missing-reshape-output",
    ],
)
def test_attention_gather_cleanup_rejects_incomplete_metadata(case: str) -> None:
    model_ir = _model(pattern_a=1)
    if case == "short-input-signature":
        model_ir.tensors["a0_x"].shape_signature = [1, 2]
    elif case == "short-transpose-signature":
        model_ir.tensors["a0_transpose"].shape_signature = [1, 2]
    elif case == "missing-reshape-output":
        del model_ir.tensors["a0_reshape"]

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="duplicate producers, reverse order, and public aliases are accepted",
)
@pytest.mark.parametrize(
    "case",
    [
        "reverse-gather-transpose",
        "duplicate-reshape-output",
        "public-internal-input",
        "reverse-source-producer",
    ],
)
def test_attention_gather_cleanup_rejects_invalid_topology(case: str) -> None:
    model_ir = _model(pattern_a=1)
    if case == "reverse-gather-transpose":
        model_ir.operators[0], model_ir.operators[1] = (
            model_ir.operators[1],
            model_ir.operators[0],
        )
    elif case == "duplicate-reshape-output":
        model_ir.operators.insert(
            2,
            OperatorIR(
                "IDENTITY",
                ["a0_x"],
                ["a0_reshape"],
            ),
        )
    elif case == "public-internal-input":
        model_ir.inputs.append("a0_gather")
    elif case == "reverse-source-producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["a0_output"],
                ["a0_x"],
            )
        )

    _assert_transactional_rejection(model_ir)


def test_attention_gather_cleanup_keeps_raw_owner_and_calls() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    owner = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_optimize_attention_gather_transpose_reshape_cleanup_chains"
    )
    assert owner.end_lineno - owner.lineno + 1 == 293
    assert sum(isinstance(node, ast.While) for node in ast.walk(owner)) == 2
    owner_source = ast.get_source_segment(lowering_source, owner)
    assert owner_source is not None
    assert "Pattern A:" in owner_source
    assert "Pattern B:" in owner_source
    assert "_build_tensor_consumer_map(model_ir)" in owner_source
    assert "_prune_unused_tensors(model_ir)" in owner_source

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_optimize_attention_gather_transpose_reshape_cleanup_chains"
    ]
    assert len(calls) == 2
    for call in calls:
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "model_ir"
        assert call.keywords == []
