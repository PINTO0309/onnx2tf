from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains,
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


def _make_model_ir(
    *,
    leak_pre_transpose: bool = False,
    expose_post_output: bool = False,
) -> ModelIR:
    model_ir = ModelIR("elementwise_roundtrip_nchw_nhwc_test")
    model_ir.inputs = ["x_nchw", "y_nchw"]
    model_ir.outputs = ["root_nchw" if expose_post_output else "final"]
    if leak_pre_transpose:
        model_ir.outputs.append("leaked")

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
            shape=[int(value) for value in shape],
            shape_signature=[int(value) for value in shape],
            data=data,
            is_variable=data is None,
        )

    _add_tensor("x_nchw", [1, 3, 8, 8])
    _add_tensor("y_nchw", [1, 3, 8, 8])
    _add_tensor("x_nhwc", [1, 8, 8, 3])
    _add_tensor("y_nhwc", [1, 8, 8, 3])
    _add_tensor("sum_nhwc", [1, 8, 8, 3])
    _add_tensor("root_nhwc", [1, 8, 8, 3])
    _add_tensor("root_nchw", [1, 3, 8, 8])
    _add_tensor("final", [1, 3, 8, 8])
    _add_tensor(
        "perm_to_nhwc",
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    _add_tensor(
        "perm_to_nchw",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _add_tensor(
        "bias",
        [1],
        data=np.asarray([0.5], dtype=np.float32),
    )
    if leak_pre_transpose:
        _add_tensor("leaked", [1, 8, 8, 3])

    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x_nchw", "perm_to_nhwc"],
            outputs=["x_nhwc"],
            options={},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["y_nchw", "perm_to_nhwc"],
            outputs=["y_nhwc"],
            options={},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["x_nhwc", "bias"],
            outputs=["sum_nhwc"],
            options={},
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["sum_nhwc", "y_nhwc"],
            outputs=["root_nhwc"],
            options={},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["root_nhwc", "perm_to_nchw"],
            outputs=["root_nchw"],
            options={},
        ),
        OperatorIR(
            op_type="RELU",
            inputs=["root_nchw"],
            outputs=["final"],
            options={},
        ),
    ]
    if leak_pre_transpose:
        model_ir.operators.append(
            OperatorIR(
                op_type="RELU",
                inputs=["x_nhwc"],
                outputs=["leaked"],
                options={},
            )
        )
    return model_ir


def test_elementwise_roundtrip_nchw_nhwc_rewrites_closed_subgraph() -> None:
    model_ir = _make_model_ir()

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == {
        "optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": 1
    }
    assert [operator.op_type for operator in model_ir.operators] == [
        "ADD",
        "MUL",
        "RELU",
    ]
    assert model_ir.operators[0].inputs == ["x_nchw", "bias"]
    assert model_ir.operators[1].inputs == ["sum_nhwc", "y_nchw"]
    assert model_ir.operators[1].outputs == ["root_nchw"]
    assert model_ir.operators[2].inputs == ["root_nchw"]
    assert model_ir.tensors["sum_nhwc"].shape == [1, 3, 8, 8]
    assert model_ir.tensors["root_nchw"].shape == [1, 3, 8, 8]
    assert "x_nhwc" not in model_ir.tensors
    assert "y_nhwc" not in model_ir.tensors
    assert "root_nhwc" not in model_ir.tensors
    assert "perm_to_nhwc" not in model_ir.tensors
    assert "perm_to_nchw" not in model_ir.tensors

    before_noop = _fingerprint(model_ir)
    assert _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(
        model_ir
    ) == {"optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": 0}
    assert _fingerprint(model_ir) == before_noop


def test_elementwise_roundtrip_nchw_nhwc_rejects_pre_transpose_fanout() -> None:
    model_ir = _make_model_ir(leak_pre_transpose=True)
    before = _fingerprint(deepcopy(model_ir))

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == {
        "optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": 0
    }
    assert _fingerprint(model_ir) == before


def test_elementwise_roundtrip_nchw_nhwc_rejects_public_post_output() -> None:
    model_ir = _make_model_ir(expose_post_output=True)
    before = _fingerprint(deepcopy(model_ir))

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == {
        "optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": 0
    }
    assert _fingerprint(model_ir) == before
