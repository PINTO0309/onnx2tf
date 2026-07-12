from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_unary_fanout_inverse_post_bridges,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=False,
    )


def _base_model() -> ModelIR:
    model_ir = ModelIR("transpose_unary_fanout")
    model_ir.inputs = ["input"]
    model_ir.tensors = {
        "input": _tensor("input", [1, 2, 3, 4]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "nchw": _tensor("nchw", [1, 4, 2, 3]),
        "relu_nchw": _tensor("relu_nchw", [1, 4, 2, 3]),
        "to_nhwc_0": _tensor(
            "to_nhwc_0",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "branch_0": _tensor("branch_0", [1, 2, 3, 4]),
        "output_0": _tensor("output_0", [1, 2, 3, 4]),
    }
    return model_ir


def test_transpose_unary_fanout_characterization_merges_inverse_posts() -> None:
    model_ir = _base_model()
    model_ir.outputs = ["output_0", "output_1"]
    model_ir.tensors.update(
        {
            "to_nhwc_1": _tensor(
                "to_nhwc_1",
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            "branch_1": _tensor("branch_1", [1, 2, 3, 4]),
            "output_1": _tensor("output_1", [1, 2, 3, 4]),
        }
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "to_nchw"], ["nchw"]),
        OperatorIR("RELU", ["nchw"], ["relu_nchw"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc_0"], ["branch_0"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc_1"], ["branch_1"]),
        OperatorIR("IDENTITY", ["branch_0"], ["output_0"]),
        OperatorIR("IDENTITY", ["branch_1"], ["output_1"]),
    ]

    stats = _optimize_transpose_unary_fanout_inverse_post_bridges(model_ir)

    assert stats["rewritten_transpose_unary_fanout_inverse_post_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "IDENTITY",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["input"]
    assert model_ir.operators[0].outputs == ["branch_0"]
    assert model_ir.operators[1].inputs == ["branch_0"]
    assert model_ir.operators[2].inputs == ["branch_0"]


def test_transpose_unary_fanout_characterization_keeps_legacy_adapter() -> None:
    model_ir = _base_model()
    model_ir.outputs = ["output_0", "legacy_output"]
    model_ir.tensors["legacy_output"] = _tensor(
        "legacy_output",
        [1, 4, 2, 3],
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "to_nchw"], ["nchw"]),
        OperatorIR("RELU", ["nchw"], ["relu_nchw"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc_0"], ["branch_0"]),
        OperatorIR("IDENTITY", ["branch_0"], ["output_0"]),
        OperatorIR("IDENTITY", ["relu_nchw"], ["legacy_output"]),
    ]

    stats = _optimize_transpose_unary_fanout_inverse_post_bridges(model_ir)

    assert stats["rewritten_transpose_unary_fanout_inverse_post_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "TRANSPOSE",
        "IDENTITY",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["input"]
    assert model_ir.operators[0].outputs == ["branch_0"]
    adapter = model_ir.operators[1]
    assert adapter.inputs == ["branch_0", "to_nhwc_0"]
    assert adapter.outputs == ["relu_nchw"]
    np.testing.assert_array_equal(
        model_ir.tensors["to_nhwc_0"].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert model_ir.operators[3].inputs == ["relu_nchw"]
