from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pre_unary_affine_fanout_layout import (
    optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains,
)


def _add_tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
    dtype: str = "FLOAT32",
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=None if data is None else np.asarray(data),
        is_variable=False,
    )


def _make_model(
    *,
    unary_op_type: str = "RELU",
    branch_count: int = 2,
    share_first_mul_constant: bool = True,
    expose_first_post_output: bool = False,
) -> ModelIR:
    model_ir = ModelIR("pre_unary_affine_transpose_fanout_layout_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = []
    _add_tensor(model_ir, "x_nhwc", [1, 6, 5, 8])
    _add_tensor(
        model_ir,
        "pre_perm",
        [4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        dtype="INT32",
    )
    _add_tensor(
        model_ir,
        "post_perm",
        [4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        dtype="INT32",
    )
    _add_tensor(model_ir, "x_nchw", [1, 8, 6, 5])
    _add_tensor(model_ir, "unary_out", [1, 8, 6, 5])
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x_nhwc", "pre_perm"],
            outputs=["x_nchw"],
        ),
        OperatorIR(
            op_type=unary_op_type,
            inputs=["x_nchw"],
            outputs=["unary_out"],
        ),
    ]

    for branch_idx in range(branch_count):
        mul_const_name = f"mul_const_{branch_idx}"
        mul_out_name = f"mul_out_{branch_idx}"
        add_const_name = f"add_const_{branch_idx}"
        add_out_name = f"add_out_{branch_idx}"
        post_out_name = f"post_out_{branch_idx}"
        result_name = f"result_{branch_idx}"
        mul_data = np.arange(1, 9, dtype=np.float32).reshape(1, 8, 1, 1)
        add_data = np.arange(8, dtype=np.float32).reshape(1, 8, 1, 1)
        _add_tensor(model_ir, mul_const_name, [1, 8, 1, 1], data=mul_data)
        _add_tensor(model_ir, mul_out_name, [1, 8, 6, 5])
        _add_tensor(model_ir, add_const_name, [1, 8, 1, 1], data=add_data)
        _add_tensor(model_ir, add_out_name, [1, 8, 6, 5])
        _add_tensor(model_ir, post_out_name, [1, 6, 5, 8])
        _add_tensor(model_ir, result_name, [1, 6, 5, 8])
        model_ir.operators.extend(
            [
                OperatorIR(
                    op_type="MUL",
                    inputs=["unary_out", mul_const_name],
                    outputs=[mul_out_name],
                    options={"fusedActivationFunction": "NONE"},
                ),
                OperatorIR(
                    op_type="ADD",
                    inputs=[mul_out_name, add_const_name],
                    outputs=[add_out_name],
                    options={"fusedActivationFunction": "NONE"},
                ),
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[add_out_name, "post_perm"],
                    outputs=[post_out_name],
                ),
                OperatorIR(
                    op_type="RELU",
                    inputs=[post_out_name],
                    outputs=[result_name],
                ),
            ]
        )
        model_ir.outputs.append(
            post_out_name
            if branch_idx == 0 and expose_first_post_output
            else result_name
        )

    if share_first_mul_constant:
        model_ir.outputs.append("shared_mul_result")
        _add_tensor(model_ir, "shared_mul_result", [1, 8, 1, 1])
        model_ir.operators.append(
            OperatorIR(
                op_type="RELU",
                inputs=["mul_const_0"],
                outputs=["shared_mul_result"],
            )
        )
    return model_ir


def _operator_snapshot(model_ir: ModelIR) -> list[tuple[object, ...]]:
    return [
        (
            operator.op_type,
            list(operator.inputs),
            list(operator.outputs),
            copy.deepcopy(operator.options),
        )
        for operator in model_ir.operators
    ]


def _assert_models_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert _operator_snapshot(actual) == _operator_snapshot(expected)
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert set(actual.tensors) == set(expected.tensors)
    for tensor_name, expected_tensor in expected.tensors.items():
        actual_tensor = actual.tensors[tensor_name]
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.shape_signature == expected_tensor.shape_signature
        assert actual_tensor.quantization == expected_tensor.quantization
        if expected_tensor.data is None:
            assert actual_tensor.data is None
        else:
            np.testing.assert_array_equal(actual_tensor.data, expected_tensor.data)


def test_pre_unary_affine_fanout_rewrites_multiple_branches_and_copies_shared_constant() -> (
    None
):
    model_ir = _make_model()
    wrapped_model_ir = copy.deepcopy(model_ir)
    original_shared_constant = np.asarray(model_ir.tensors["mul_const_0"].data).copy()

    stats = optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(model_ir)
    wrapped_stats = _optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(
        wrapped_model_ir
    )

    assert stats == {
        "optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains": 1
    }
    assert wrapped_stats == stats
    _assert_models_equal(wrapped_model_ir, model_ir)
    unary_op = next(
        operator
        for operator in model_ir.operators
        if list(operator.outputs) == ["unary_out"]
    )
    assert list(unary_op.inputs) == ["x_nhwc"]
    assert model_ir.tensors["unary_out"].shape == [1, 6, 5, 8]
    assert not any(
        operator.op_type == "TRANSPOSE" and list(operator.outputs) == ["x_nchw"]
        for operator in model_ir.operators
    )
    for branch_idx in range(2):
        add_op = next(
            operator
            for operator in model_ir.operators
            if list(operator.outputs) == [f"post_out_{branch_idx}"]
        )
        assert add_op.op_type == "ADD"
        assert model_ir.tensors[f"add_const_{branch_idx}"].shape == [1, 1, 1, 8]
        assert not any(
            operator.op_type == "TRANSPOSE"
            and list(operator.outputs) == [f"post_out_{branch_idx}"]
            for operator in model_ir.operators
        )
    first_mul = next(
        operator
        for operator in model_ir.operators
        if list(operator.outputs) == ["mul_out_0"]
    )
    assert list(first_mul.inputs)[1] == "mul_const_0_nhwc"
    np.testing.assert_array_equal(
        model_ir.tensors["mul_const_0"].data,
        original_shared_constant,
    )
    assert model_ir.tensors["mul_const_0_nhwc"].shape == [1, 1, 1, 8]
    assert model_ir.tensors["mul_const_1"].shape == [1, 1, 1, 8]
    assert optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(
        model_ir
    ) == {"optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains": 0}


@pytest.mark.parametrize(
    "unary_op_type",
    ["RELU", "RELU6", "LOGISTIC", "TANH", "HARD_SWISH", "LEAKY_RELU", "GELU"],
)
def test_pre_unary_affine_fanout_accepts_supported_unary_family(
    unary_op_type: str,
) -> None:
    model_ir = _make_model(
        unary_op_type=unary_op_type,
        branch_count=1,
        share_first_mul_constant=False,
    )

    stats = optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(model_ir)

    assert stats == {
        "optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains": 1
    }


@pytest.mark.parametrize(
    ("unary_op_type", "expose_first_post_output"),
    [("ABS", False), ("RELU", True)],
)
def test_pre_unary_affine_fanout_rejects_unsupported_or_public_boundary(
    unary_op_type: str,
    expose_first_post_output: bool,
) -> None:
    model_ir = _make_model(
        unary_op_type=unary_op_type,
        branch_count=1,
        share_first_mul_constant=False,
        expose_first_post_output=expose_first_post_output,
    )
    operators_before = _operator_snapshot(model_ir)
    tensor_names_before = set(model_ir.tensors)

    stats = optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(model_ir)

    assert stats == {
        "optimized_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains": 0
    }
    assert _operator_snapshot(model_ir) == operators_before
    assert set(model_ir.tensors) == tensor_names_before
