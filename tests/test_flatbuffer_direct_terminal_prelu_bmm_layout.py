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
    _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze
from onnx2tf.tflite_builder.passes.terminal_prelu_bmm_layout import (
    optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains,
)


_STATS_KEY = (
    "optimized_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains"
)
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


def _alpha(kind: str) -> np.ndarray:
    values = np.arange(1, 5, dtype=np.float32) / 10.0
    if kind == "nchw4":
        return values.reshape(1, 4, 1, 1)
    if kind == "chw3":
        return values.reshape(4, 1, 1)
    if kind == "nhwc4":
        return values.reshape(1, 1, 1, 4)
    if kind == "scalar":
        return np.asarray(0.25, dtype=np.float32)
    if kind == "vector":
        return values
    raise AssertionError(kind)


def _model(*, alpha_kind: str = "nchw4", shared_constants: bool = False) -> ModelIR:
    model = ModelIR("terminal_prelu_bmm")
    model.inputs = ["x"]
    model.outputs = ["out"]
    rhs = np.arange(_INPUT_SIZE * 5, dtype=np.float32).reshape(_INPUT_SIZE, 5)
    model.tensors = {
        "x": _tensor("x", _NHWC_SHAPE),
        "x_nchw": _tensor("x_nchw", _NCHW_SHAPE),
        "prelu_out": _tensor("prelu_out", _NCHW_SHAPE),
        "flat": _tensor("flat", [1, _INPUT_SIZE]),
        "out": _tensor("out", [1, 5]),
        "perm": _constant(
            "perm", np.asarray([0, 3, 1, 2], dtype=np.int32), "INT32"
        ),
        "alpha": _constant("alpha", _alpha(alpha_kind)),
        "shape": _constant(
            "shape", np.asarray([1, _INPUT_SIZE], dtype=np.int32), "INT32"
        ),
        "rhs": _constant("rhs", rhs),
    }
    model.tensors["alpha"].quantization = QuantParamIR(
        scale=[0.125], zero_point=[0], quantized_dimension=0
    )
    model.tensors["rhs"].quantization = QuantParamIR(
        scale=[0.25], zero_point=[0], quantized_dimension=0
    )
    model.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["x_nchw"]),
        OperatorIR("PRELU", ["x_nchw", "alpha"], ["prelu_out"]),
        OperatorIR("RESHAPE", ["prelu_out", "shape"], ["flat"]),
        OperatorIR(
            "BATCH_MATMUL",
            ["flat", "rhs"],
            ["out"],
            options={"adjX": False, "adjY": False},
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
                OperatorIR("PRELU", ["outside", "alpha"], ["outside_out"]),
                OperatorIR(
                    "BATCH_MATMUL",
                    ["other_flat", "rhs"],
                    ["other_out"],
                    options={"adjX": False, "adjY": False},
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


@pytest.mark.parametrize(
    ("alpha_kind", "expected_shape"),
    [
        ("nchw4", [1, 1, 1, 4]),
        ("chw3", [1, 1, 4]),
        ("nhwc4", [1, 1, 1, 4]),
        ("scalar", []),
    ],
)
def test_terminal_prelu_bmm_moves_flatten_order_to_nhwc(
    alpha_kind: str,
    expected_shape: list[int],
) -> None:
    model = _model(alpha_kind=alpha_kind)
    old_rhs = np.asarray(model.tensors["rhs"].data).copy()
    permutation = np.transpose(
        np.arange(_INPUT_SIZE, dtype=np.int64).reshape(4, 2, 3),
        (1, 2, 0),
    ).reshape(-1)

    assert _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}

    assert not any(operator.op_type == "TRANSPOSE" for operator in model.operators)
    prelu = next(
        operator for operator in model.operators if operator.op_type == "PRELU"
    )
    assert prelu.inputs == ["x", "alpha"]
    assert model.tensors["alpha"].shape == expected_shape
    assert model.tensors["prelu_out"].shape == _NHWC_SHAPE
    np.testing.assert_array_equal(
        model.tensors["rhs"].data,
        old_rhs[permutation, :],
    )


def test_terminal_prelu_bmm_clones_shared_constants() -> None:
    model = _model(shared_constants=True)
    original_alpha = copy.deepcopy(model.tensors["alpha"])
    original_rhs = copy.deepcopy(model.tensors["rhs"])

    assert _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}

    main_prelu = next(
        operator
        for operator in model.operators
        if operator.op_type == "PRELU" and operator.outputs == ["prelu_out"]
    )
    main_bmm = next(
        operator
        for operator in model.operators
        if operator.op_type == "BATCH_MATMUL" and operator.outputs == ["out"]
    )
    assert main_prelu.inputs == ["x", "alpha_nhwc"]
    assert main_bmm.inputs[1] == "rhs_nhwc"
    np.testing.assert_array_equal(model.tensors["alpha"].data, original_alpha.data)
    np.testing.assert_array_equal(model.tensors["rhs"].data, original_rhs.data)
    assert model.tensors["alpha_nhwc"].quantization == original_alpha.quantization
    assert model.tensors["alpha_nhwc"].quantization is not original_alpha.quantization
    assert model.tensors["rhs_nhwc"].quantization == original_rhs.quantization
    assert model.tensors["rhs_nhwc"].quantization is not original_rhs.quantization


def test_terminal_prelu_bmm_is_idempotent() -> None:
    model = _model()

    assert _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}
    after_first = _snapshot(model)
    assert _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
        model
    ) == {_STATS_KEY: 0}
    assert _snapshot(model) == after_first


def test_terminal_prelu_bmm_module_owner_matches_private_wrapper() -> None:
    direct = _model(shared_constants=True)
    wrapped = copy.deepcopy(direct)

    assert optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
        direct
    ) == {_STATS_KEY: 1}
    assert _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
        wrapped
    ) == {_STATS_KEY: 1}
    assert _snapshot(direct) == _snapshot(wrapped)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model.outputs.append("x_nchw"),
        lambda model: model.outputs.append("prelu_out"),
        lambda model: model.outputs.append("flat"),
        lambda model: model.tensors["perm"].data.__setitem__(
            slice(None), np.asarray([0, 2, 3, 1], dtype=np.int32)
        ),
        lambda model: setattr(model.tensors["x_nchw"], "shape", [1, -1, 2, 3]),
        lambda model: setattr(
            model.tensors["rhs"],
            "data",
            np.zeros((_INPUT_SIZE + 1, 5), dtype=np.float32),
        ),
        lambda model: model.operators[-1].options.__setitem__("adjX", True),
        lambda model: model.operators[-1].options.__setitem__("adjY", True),
        lambda model: (
            model.tensors.__setitem__("side", _tensor("side", _NCHW_SHAPE)),
            model.outputs.append("side"),
            model.operators.append(OperatorIR("ABS", ["x_nchw"], ["side"])),
        ),
        lambda model: model.tensors.__setitem__(
            "alpha", _constant("alpha", _alpha("vector"))
        ),
    ],
)
def test_terminal_prelu_bmm_guard_rejections_do_not_rewrite(mutate) -> None:
    model = _model()
    mutate(model)
    before_operators = copy.deepcopy(model.operators)

    assert _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
        model
    ) == {_STATS_KEY: 0}
    assert [
        (operator.op_type, operator.inputs, operator.outputs)
        for operator in model.operators
    ] == [
        (operator.op_type, operator.inputs, operator.outputs)
        for operator in before_operators
    ]
