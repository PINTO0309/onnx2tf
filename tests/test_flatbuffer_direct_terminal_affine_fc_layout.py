from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze
from onnx2tf.tflite_builder.passes.terminal_affine_fc_layout import (
    optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains,
)


_STATS_KEY = "optimized_terminal_transpose_mul_add_reshape_fc_nhwc_chains"
_NCHW_SHAPE = [1, 4, 2, 3]
_NHWC_SHAPE = [1, 2, 3, 4]
_INPUT_SIZE = 24


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=[-1, *shape[1:]],
    )


def _constant(name: str, values: np.ndarray, dtype: str = "FLOAT32") -> TensorIR:
    array = np.asarray(values)
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=[int(value) for value in array.shape],
        shape_signature=[int(value) for value in array.shape],
        data=array,
        is_variable=False,
    )


def _model(
    *,
    transposed_weight: bool = False,
    shared_constants: bool = False,
) -> ModelIR:
    model = ModelIR("terminal_affine_fc")
    model.inputs = ["x"]
    model.outputs = ["out"]
    weight = np.arange(5 * _INPUT_SIZE, dtype=np.float32).reshape(5, _INPUT_SIZE)
    if transposed_weight:
        weight = weight.T
    model.tensors = {
        "x": _tensor("x", _NHWC_SHAPE),
        "x_nchw": _tensor("x_nchw", _NCHW_SHAPE),
        "mul_out": _tensor("mul_out", _NCHW_SHAPE),
        "add_out": _tensor("add_out", _NCHW_SHAPE),
        "flat": _tensor("flat", [1, _INPUT_SIZE]),
        "out": _tensor("out", [1, 5]),
        "perm": _constant(
            "perm", np.asarray([0, 3, 1, 2], dtype=np.int32), "INT32"
        ),
        "gamma": _constant(
            "gamma", np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1)
        ),
        "beta": _constant(
            "beta", np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1)
        ),
        "shape": _constant(
            "shape", np.asarray([1, _INPUT_SIZE], dtype=np.int32), "INT32"
        ),
        "weight": _constant("weight", weight),
        "bias": _constant("bias", np.arange(5, dtype=np.float32)),
    }
    model.tensors["gamma"].quantization = QuantParamIR(
        scale=[0.125], zero_point=[0], quantized_dimension=0
    )
    model.tensors["weight"].quantization = QuantParamIR(
        scale=[0.25], zero_point=[0], quantized_dimension=0
    )
    model.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["x_nchw"]),
        OperatorIR("MUL", ["x_nchw", "gamma"], ["mul_out"]),
        OperatorIR("ADD", ["mul_out", "beta"], ["add_out"]),
        OperatorIR("RESHAPE", ["add_out", "shape"], ["flat"]),
        OperatorIR(
            "FULLY_CONNECTED",
            ["flat", "weight", "bias"],
            ["out"],
        ),
    ]
    if shared_constants:
        model.inputs.extend(["outside", "other_flat"])
        model.outputs.extend(["outside_out", "other_out"])
        model.tensors.update(
            {
                "outside": _tensor("outside", _NCHW_SHAPE),
                "outside_out": _tensor("outside_out", _NCHW_SHAPE),
                "other_flat": _tensor("other_flat", [1, _INPUT_SIZE]),
                "other_out": _tensor("other_out", [1, 5]),
            }
        )
        model.operators.extend(
            [
                OperatorIR("MUL", ["outside", "gamma"], ["outside_out"]),
                OperatorIR(
                    "FULLY_CONNECTED",
                    ["other_flat", "weight", "bias"],
                    ["other_out"],
                ),
            ]
        )
    return model


def _snapshot(model: ModelIR):
    return (
        tuple(model.inputs),
        tuple(model.outputs),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or tensor.shape),
                _freeze(tensor.data),
                _freeze(tensor.quantization),
            )
            for name, tensor in sorted(model.tensors.items())
        ),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                _freeze(operator.options),
            )
            for operator in model.operators
        ),
    )


@pytest.mark.parametrize("transposed_weight", [False, True])
def test_terminal_affine_fc_moves_flatten_order_to_nhwc(
    transposed_weight: bool,
) -> None:
    model = _model(transposed_weight=transposed_weight)
    old_weight = np.asarray(model.tensors["weight"].data).copy()
    permutation = np.transpose(
        np.arange(_INPUT_SIZE, dtype=np.int64).reshape(4, 2, 3),
        (1, 2, 0),
    ).reshape(-1)

    assert _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}

    assert not any(operator.op_type == "TRANSPOSE" for operator in model.operators)
    mul = next(operator for operator in model.operators if operator.op_type == "MUL")
    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    fc = next(
        operator
        for operator in model.operators
        if operator.op_type == "FULLY_CONNECTED"
    )
    assert mul.inputs == ["x", "gamma"]
    assert add.inputs == ["mul_out", "beta"]
    assert fc.inputs[1] == "weight"
    assert model.tensors["gamma"].shape == [1, 1, 1, 4]
    assert model.tensors["beta"].shape == [1, 1, 1, 4]
    assert model.tensors["mul_out"].shape == _NHWC_SHAPE
    assert model.tensors["add_out"].shape == _NHWC_SHAPE
    expected_weight = (
        old_weight[permutation, :]
        if transposed_weight
        else old_weight[:, permutation]
    )
    np.testing.assert_array_equal(model.tensors["weight"].data, expected_weight)


def test_terminal_affine_fc_clones_shared_constants() -> None:
    model = _model(shared_constants=True)
    original_gamma = copy.deepcopy(model.tensors["gamma"])
    original_weight = copy.deepcopy(model.tensors["weight"])

    assert _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}

    main_mul = next(
        operator
        for operator in model.operators
        if operator.op_type == "MUL" and operator.outputs == ["mul_out"]
    )
    main_fc = next(
        operator
        for operator in model.operators
        if operator.op_type == "FULLY_CONNECTED" and operator.outputs == ["out"]
    )
    assert main_mul.inputs == ["x", "gamma_nhwc"]
    assert main_fc.inputs[1] == "weight_nhwc"
    np.testing.assert_array_equal(model.tensors["gamma"].data, original_gamma.data)
    np.testing.assert_array_equal(model.tensors["weight"].data, original_weight.data)
    assert model.tensors["gamma_nhwc"].quantization == original_gamma.quantization
    assert model.tensors["gamma_nhwc"].quantization is not original_gamma.quantization
    assert model.tensors["weight_nhwc"].quantization == original_weight.quantization
    assert model.tensors["weight_nhwc"].quantization is not original_weight.quantization


def test_terminal_affine_fc_is_idempotent() -> None:
    model = _model()

    assert _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}
    after_first = _snapshot(model)
    assert _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
        model
    ) == {_STATS_KEY: 0}
    assert _snapshot(model) == after_first


def test_terminal_affine_fc_module_owner_matches_private_wrapper() -> None:
    direct = _model(shared_constants=True)
    wrapped = copy.deepcopy(direct)

    assert optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
        direct
    ) == {_STATS_KEY: 1}
    assert _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
        wrapped
    ) == {_STATS_KEY: 1}
    assert _snapshot(direct) == _snapshot(wrapped)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model.outputs.append("x_nchw"),
        lambda model: model.outputs.append("mul_out"),
        lambda model: model.outputs.append("add_out"),
        lambda model: model.outputs.append("flat"),
        lambda model: model.tensors["perm"].data.__setitem__(
            slice(None), np.asarray([0, 2, 3, 1], dtype=np.int32)
        ),
        lambda model: setattr(model.tensors["x_nchw"], "shape", [1, -1, 2, 3]),
        lambda model: setattr(
            model.tensors["weight"],
            "data",
            np.zeros((5, _INPUT_SIZE + 1), dtype=np.float32),
        ),
        lambda model: (
            model.tensors.__setitem__("side", _tensor("side", _NCHW_SHAPE)),
            model.outputs.append("side"),
            model.operators.append(OperatorIR("ABS", ["x_nchw"], ["side"])),
        ),
    ],
)
def test_terminal_affine_fc_guard_rejections_do_not_rewrite(mutate) -> None:
    model = _model()
    mutate(model)
    before_operators = copy.deepcopy(model.operators)

    assert _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
        model
    ) == {_STATS_KEY: 0}
    assert [
        (operator.op_type, operator.inputs, operator.outputs)
        for operator in model.operators
    ] == [
        (operator.op_type, operator.inputs, operator.outputs)
        for operator in before_operators
    ]
