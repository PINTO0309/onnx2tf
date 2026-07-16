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
    _repair_rank4_binary_layout_mismatch_with_transpose_adapter,
)
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    repair_rank4_binary_layout_mismatch_with_transpose_adapter,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


def _model(
    op_type: str = "ADD",
    *,
    lhs_shape: tuple[int, ...] = (1, 4, 5, 3),
    rhs_shape: tuple[int, ...] = (1, 3, 4, 5),
) -> ModelIR:
    model_ir = ModelIR("rank4_binary_layout_adapter")
    model_ir.inputs = ["lhs", "rhs"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "lhs": TensorIR(
            name="lhs",
            dtype="FLOAT32",
            shape=list(lhs_shape),
            shape_signature=list(lhs_shape),
        ),
        "rhs": TensorIR(
            name="rhs",
            dtype="FLOAT32",
            shape=list(rhs_shape),
            shape_signature=list(rhs_shape),
            quantization=QuantParamIR(scale=[0.25], zero_point=[3]),
        ),
        "out": TensorIR(
            name="out",
            dtype="FLOAT32",
            shape=list(lhs_shape),
            shape_signature=list(lhs_shape),
        ),
    }
    model_ir.operators = [
        OperatorIR(op_type=op_type, inputs=["lhs", "rhs"], outputs=["out"])
    ]
    return model_ir


def _snapshot(model_ir: ModelIR) -> tuple:
    return (
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or tensor.shape),
                _freeze(tensor.data),
                _freeze(tensor.quantization),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                _freeze(operator.options),
            )
            for operator in model_ir.operators
        ),
    )


@pytest.mark.parametrize(
    "op_type",
    ["ADD", "MUL", "SUB", "DIV", "MAXIMUM", "MINIMUM"],
)
@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "expected_perm"),
    [
        ((1, 4, 5, 3), (1, 3, 4, 5), [0, 2, 3, 1]),
        ((1, 3, 4, 5), (1, 4, 5, 3), [0, 3, 1, 2]),
    ],
)
def test_exact_permutation_inserts_input1_adapter(
    op_type: str,
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
    expected_perm: list[int],
) -> None:
    model_ir = _model(
        op_type,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
    )

    stats = repair_rank4_binary_layout_mismatch_with_transpose_adapter(
        model_ir
    )

    assert stats == {"inserted_rank4_binary_layout_fix_transpose": 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        op_type,
    ]
    transpose, binary = model_ir.operators
    assert transpose.inputs[0] == "rhs"
    assert binary.inputs == ["lhs", transpose.outputs[0]]
    adapted = model_ir.tensors[transpose.outputs[0]]
    assert adapted.shape == list(lhs_shape)
    assert adapted.shape_signature == list(lhs_shape)
    assert _freeze(adapted.quantization) == _freeze(
        model_ir.tensors["rhs"].quantization
    )
    assert adapted.quantization is not model_ir.tensors["rhs"].quantization
    np.testing.assert_array_equal(
        model_ir.tensors[transpose.inputs[1]].data,
        np.asarray(expected_perm, dtype=np.int32),
    )
    before = _snapshot(model_ir)
    assert repair_rank4_binary_layout_mismatch_with_transpose_adapter(
        model_ir
    ) == {"inserted_rank4_binary_layout_fix_transpose": 0}
    assert _snapshot(model_ir) == before


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape"),
    [
        ((1, 4, 5, 3), (1, 4, 5, 3)),
        ((1, -1, 5, 3), (1, 3, -1, 5)),
        ((4, 5, 3), (3, 4, 5)),
        ((1, 4, 5, 3), (1, 3, 5, 4)),
    ],
)
def test_non_exact_or_dynamic_shapes_are_noop(
    lhs_shape: tuple[int, ...],
    rhs_shape: tuple[int, ...],
) -> None:
    model_ir = _model(lhs_shape=lhs_shape, rhs_shape=rhs_shape)
    before = _snapshot(model_ir)

    assert repair_rank4_binary_layout_mismatch_with_transpose_adapter(
        model_ir
    ) == {"inserted_rank4_binary_layout_fix_transpose": 0}
    assert _snapshot(model_ir) == before


def test_compatibility_wrapper_matches_module_owner() -> None:
    direct_model = _model()
    wrapper_model = copy.deepcopy(direct_model)

    direct_stats = repair_rank4_binary_layout_mismatch_with_transpose_adapter(
        direct_model
    )
    wrapper_stats = (
        _repair_rank4_binary_layout_mismatch_with_transpose_adapter(
            wrapper_model
        )
    )

    assert wrapper_stats == direct_stats
    assert _snapshot(wrapper_model) == _snapshot(direct_model)
