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
    _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.dual_pre_add_layout import (
    optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


_STATS_KEY = (
    "optimized_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains"
)


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=[-1, *shape[1:]],
    )


def _constant(name: str, values: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT32",
        shape=[len(values)],
        shape_signature=[len(values)],
        data=np.asarray(values, dtype=np.int32),
        is_variable=False,
    )


def _model() -> ModelIR:
    model = ModelIR("dual_pre_add_single_post")
    model.inputs = ["a", "b"]
    model.outputs = ["tail"]
    model.tensors = {
        "a": _tensor("a", [1, 4, 5, 3]),
        "b": _tensor("b", [1, 4, 5, 3]),
        "a_nchw": _tensor("a_nchw", [1, 3, 4, 5]),
        "b_nchw": _tensor("b_nchw", [1, 3, 4, 5]),
        "sum": _tensor("sum", [1, 3, 4, 5]),
        "tail": _tensor("tail", [1, 3, 4, 5]),
        "to_nchw": _constant("to_nchw", [0, 3, 1, 2]),
    }
    model.tensors["sum"].quantization = QuantParamIR(
        scale=[0.25],
        zero_point=[0],
        quantized_dimension=1,
    )
    model.operators = [
        OperatorIR("TRANSPOSE", ["a", "to_nchw"], ["a_nchw"]),
        OperatorIR("TRANSPOSE", ["b", "to_nchw"], ["b_nchw"]),
        OperatorIR("ADD", ["a_nchw", "b_nchw"], ["sum"]),
        OperatorIR("ABS", ["sum"], ["tail"]),
    ]
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


def test_dual_pre_add_moves_two_input_adapters_to_one_output_adapter() -> None:
    model = _model()

    assert _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}

    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    assert add.inputs == ["a", "b"]
    assert add.outputs == ["sum_nhwc"]
    assert model.tensors["sum_nhwc"].shape == [1, 4, 5, 3]
    assert model.tensors["sum_nhwc"].shape_signature == [-1, 4, 5, 3]
    assert model.tensors["sum_nhwc"].quantization == model.tensors["sum"].quantization
    assert model.tensors["sum_nhwc"].quantization is not model.tensors["sum"].quantization
    transposes = [
        operator for operator in model.operators if operator.op_type == "TRANSPOSE"
    ]
    assert len(transposes) == 1
    assert transposes[0].inputs == [
        "sum_nhwc",
        "__nhwc_to_nchw_perm_rank4__",
    ]
    assert transposes[0].outputs == ["sum"]
    np.testing.assert_array_equal(
        model.tensors["__nhwc_to_nchw_perm_rank4__"].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert next(
        operator for operator in model.operators if operator.outputs == ["tail"]
    ).inputs == ["sum"]


def test_dual_pre_add_is_idempotent() -> None:
    model = _model()

    assert _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}
    after_first = _snapshot(model)
    assert _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
        model
    ) == {_STATS_KEY: 0}
    assert _snapshot(model) == after_first


def test_dual_pre_add_module_owner_matches_private_wrapper() -> None:
    direct = _model()
    wrapped = copy.deepcopy(direct)

    assert optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
        direct
    ) == {_STATS_KEY: 1}
    assert _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
        wrapped
    ) == {_STATS_KEY: 1}
    assert _snapshot(direct) == _snapshot(wrapped)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model.outputs.append("sum"),
        lambda model: model.outputs.append("a_nchw"),
        lambda model: model.tensors["to_nchw"].data.__setitem__(
            slice(None), np.asarray([0, 2, 3, 1], dtype=np.int32)
        ),
        lambda model: setattr(model.tensors["sum"], "shape", [1, 3, 20]),
        lambda model: (
            model.tensors.__setitem__("a_side", _tensor("a_side", [1, 3, 4, 5])),
            model.outputs.append("a_side"),
            model.operators.append(OperatorIR("ABS", ["a_nchw"], ["a_side"])),
        ),
        lambda model: (
            model.tensors.__setitem__(
                "sum_nhwc", _tensor("sum_nhwc", [1, 4, 5, 3])
            ),
            model.tensors.__setitem__(
                "to_nhwc", _constant("to_nhwc", [0, 2, 3, 1])
            ),
            model.operators.append(
                OperatorIR("TRANSPOSE", ["sum", "to_nhwc"], ["sum_nhwc"])
            ),
        ),
    ],
)
def test_dual_pre_add_guard_rejections_do_not_rewrite(mutate) -> None:
    model = _model()
    mutate(model)
    before_operators = copy.deepcopy(model.operators)

    assert _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
        model
    ) == {_STATS_KEY: 0}
    assert [
        (operator.op_type, operator.inputs, operator.outputs)
        for operator in model.operators
    ] == [
        (operator.op_type, operator.inputs, operator.outputs)
        for operator in before_operators
    ]
    assert "__nhwc_to_nchw_perm_rank4__" not in model.tensors
