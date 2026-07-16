from __future__ import annotations

import copy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.residual_affine_fanout_layout import (
    optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains,
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
    branch_count: int = 2,
    keep_legacy_consumer: bool = True,
    share_first_mul_constant: bool = True,
    expose_first_post_output: bool = False,
) -> ModelIR:
    model_ir = ModelIR("residual_affine_transpose_fanout_layout_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc"]
    model_ir.outputs = []

    _add_tensor(model_ir, "a_nhwc", [1, 6, 5, 8])
    _add_tensor(model_ir, "b_nhwc", [1, 6, 5, 8])
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
    _add_tensor(model_ir, "a_nchw", [1, 8, 6, 5])
    _add_tensor(model_ir, "b_nchw", [1, 8, 6, 5])
    _add_tensor(model_ir, "residual_nchw", [1, 8, 6, 5])
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["a_nhwc", "pre_perm"],
            outputs=["a_nchw"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["b_nhwc", "pre_perm"],
            outputs=["b_nchw"],
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["a_nchw", "b_nchw"],
            outputs=["residual_nchw"],
            options={"fusedActivationFunction": "NONE"},
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
                    inputs=["residual_nchw", mul_const_name],
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

    if keep_legacy_consumer:
        model_ir.inputs.append("legacy_ref_nchw")
        model_ir.outputs.append("legacy_concat")
        _add_tensor(model_ir, "legacy_ref_nchw", [1, 8, 6, 5])
        _add_tensor(model_ir, "legacy_concat", [1, 16, 6, 5])
        model_ir.operators.append(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["residual_nchw", "legacy_ref_nchw"],
                outputs=["legacy_concat"],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            )
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


def test_residual_affine_fanout_rewrites_multiple_branches_and_preserves_legacy_adapter() -> (
    None
):
    model_ir = _make_model()
    wrapped_model_ir = copy.deepcopy(model_ir)
    original_shared_constant = np.asarray(model_ir.tensors["mul_const_0"].data).copy()

    stats = optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains(model_ir)
    wrapped_stats = _optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains(
        wrapped_model_ir
    )

    assert stats == {
        "optimized_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains": 1
    }
    assert wrapped_stats == stats
    _assert_models_equal(wrapped_model_ir, model_ir)
    root_add = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "ADD" and list(operator.inputs) == ["a_nhwc", "b_nhwc"]
    )
    assert list(root_add.outputs) == ["residual_nchw_nhwc"]
    assert not any(
        operator.op_type == "TRANSPOSE"
        and list(operator.outputs) in (["a_nchw"], ["b_nchw"])
        for operator in model_ir.operators
    )
    legacy_adapter = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE"
        and list(operator.outputs) == ["residual_nchw"]
    )
    assert list(legacy_adapter.inputs)[0] == "residual_nchw_nhwc"
    for branch_idx in range(2):
        mul_op = next(
            operator
            for operator in model_ir.operators
            if list(operator.outputs) == [f"mul_out_{branch_idx}"]
        )
        assert list(mul_op.inputs)[0] == "residual_nchw_nhwc"
        add_op = next(
            operator
            for operator in model_ir.operators
            if list(operator.outputs) == [f"post_out_{branch_idx}"]
        )
        assert add_op.op_type == "ADD"
        assert not any(
            operator.op_type == "TRANSPOSE"
            and list(operator.outputs) == [f"post_out_{branch_idx}"]
            for operator in model_ir.operators
        )
        assert model_ir.tensors[f"add_const_{branch_idx}"].shape == [1, 1, 1, 8]

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
    assert optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains(
        model_ir
    ) == {"optimized_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains": 0}


def test_residual_affine_fanout_public_post_output_is_a_noop() -> None:
    model_ir = _make_model(
        branch_count=1,
        keep_legacy_consumer=False,
        share_first_mul_constant=False,
        expose_first_post_output=True,
    )
    operators_before = _operator_snapshot(model_ir)
    tensor_names_before = set(model_ir.tensors)

    stats = optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains(model_ir)

    assert stats == {
        "optimized_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains": 0
    }
    assert _operator_snapshot(model_ir) == operators_before
    assert set(model_ir.tensors) == tensor_names_before
