from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_batchmatmul_transpose_input_to_adj_flags,
)
from onnx2tf.tflite_builder.passes.batchmatmul_adjoint_layout import (
    optimize_batchmatmul_transpose_input_to_adj_flags,
)


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
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
                (
                    None
                    if tensor.shape_signature is None
                    else tuple(tensor.shape_signature)
                ),
                tensor.logical_layout,
                tensor.physical_layout,
                repr(tensor.quantization),
                (
                    None
                    if tensor.data is None
                    else (
                        str(np.asarray(tensor.data).dtype),
                        tuple(np.asarray(tensor.data).shape),
                        tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                    )
                ),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        repr(model_ir.metadata),
    )


def _make_model_ir() -> ModelIR:
    model_ir = ModelIR("batchmatmul_transpose_input_to_adj_flags_test")
    model_ir.inputs = ["lhs_a", "rhs_a", "lhs_b", "rhs_b"]
    model_ir.outputs = ["out_a", "out_b"]

    def _add_tensor(
        name: str,
        shape: list[int],
        *,
        dtype: str = "FLOAT32",
        data: np.ndarray | None = None,
    ) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=data is None,
        )

    _add_tensor("lhs_a", [2, 3, 4])
    _add_tensor("lhs_a_t", [2, 4, 3])
    _add_tensor("rhs_a", [2, 3, 6])
    _add_tensor("out_a", [2, 4, 6])
    _add_tensor(
        "perm_swap_last_two",
        [3],
        dtype="INT32",
        data=np.asarray([0, 2, 1], dtype=np.int32),
    )

    _add_tensor("lhs_b", [1, 3, 4])
    _add_tensor("rhs_b", [6, 1, 4])
    _add_tensor("rhs_b_t", [1, 4, 6])
    _add_tensor("out_b", [1, 3, 6])
    _add_tensor(
        "perm_singleton",
        [3],
        dtype="INT32",
        data=np.asarray([1, 2, 0], dtype=np.int32),
    )

    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["lhs_a", "perm_swap_last_two"],
            outputs=["lhs_a_t"],
            options={},
        ),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["lhs_a_t", "rhs_a"],
            outputs=["out_a"],
            options={"adjX": False, "adjY": False},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["rhs_b", "perm_singleton"],
            outputs=["rhs_b_t"],
            options={},
        ),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["lhs_b", "rhs_b_t"],
            outputs=["out_b"],
            options={"adjX": False, "adjY": False},
        ),
    ]
    return model_ir


def test_flatbuffer_direct_batchmatmul_transpose_input_to_adj_flags() -> None:
    owner_model_ir = _make_model_ir()
    wrapper_model_ir = deepcopy(owner_model_ir)

    owner_stats = optimize_batchmatmul_transpose_input_to_adj_flags(owner_model_ir)
    wrapper_stats = _optimize_batchmatmul_transpose_input_to_adj_flags(
        wrapper_model_ir
    )

    assert owner_stats == wrapper_stats == {
        "optimized_batchmatmul_transpose_input_to_adj_flags": 2
    }
    assert _fingerprint(owner_model_ir) == _fingerprint(wrapper_model_ir)
    assert [operator.op_type for operator in owner_model_ir.operators] == [
        "BATCH_MATMUL",
        "RESHAPE",
        "BATCH_MATMUL",
    ]

    first_bmm = owner_model_ir.operators[0]
    assert first_bmm.inputs == ["lhs_a", "rhs_a"]
    assert first_bmm.options == {"adjX": True, "adjY": False}

    singleton_reshape = owner_model_ir.operators[1]
    assert singleton_reshape.inputs == ["rhs_b", "rhs_b_t_reshape_shape"]
    assert singleton_reshape.outputs == ["rhs_b_t"]
    assert singleton_reshape.options == {"newShape": [1, 6, 4]}
    assert owner_model_ir.tensors["rhs_b_t"].shape == [1, 6, 4]
    assert owner_model_ir.tensors["rhs_b_t"].shape_signature == [1, 6, 4]
    assert np.array_equal(
        owner_model_ir.tensors["rhs_b_t_reshape_shape"].data,
        np.asarray([1, 6, 4], dtype=np.int32),
    )

    second_bmm = owner_model_ir.operators[2]
    assert second_bmm.inputs == ["lhs_b", "rhs_b_t"]
    assert second_bmm.options == {"adjX": False, "adjY": True}
    assert "lhs_a_t" not in owner_model_ir.tensors
    assert "perm_swap_last_two" not in owner_model_ir.tensors
    assert "perm_singleton" not in owner_model_ir.tensors

    before_noop = _fingerprint(owner_model_ir)
    assert optimize_batchmatmul_transpose_input_to_adj_flags(owner_model_ir) == {
        "optimized_batchmatmul_transpose_input_to_adj_flags": 0
    }
    assert _fingerprint(owner_model_ir) == before_noop
