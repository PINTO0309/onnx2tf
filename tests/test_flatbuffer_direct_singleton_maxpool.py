from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_singleton_layout_reshape_maxpool_binary_cast_chains,
)


def _tensor(
    name: str,
    dtype: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _model(*, post_pool_fanout: bool) -> ModelIR:
    model_ir = ModelIR("singleton_maxpool_binary_cast")
    model_ir.inputs = ["a_nchw"]
    model_ir.outputs = ["d_nchw"] + (["side"] if post_pool_fanout else [])
    nchw = [1, 1, 2, 3]
    nhwc = [1, 2, 3, 1]
    model_ir.tensors = {
        "a_nchw": _tensor("a_nchw", "FLOAT32", nchw),
        "nhwc_shape1": _tensor(
            "nhwc_shape1", "INT32", [4], data=np.asarray(nhwc, dtype=np.int32)
        ),
        "a_nhwc": _tensor("a_nhwc", "FLOAT32", nhwc),
        "b_nhwc": _tensor("b_nhwc", "FLOAT32", nhwc),
        "nchw_shape1": _tensor(
            "nchw_shape1", "INT32", [4], data=np.asarray(nchw, dtype=np.int32)
        ),
        "b_nchw": _tensor("b_nchw", "FLOAT32", nchw),
        "cmp_nchw": _tensor("cmp_nchw", "BOOL", nchw),
        "cast_nchw": _tensor("cast_nchw", "FLOAT32", nchw),
        "nhwc_shape2": _tensor(
            "nhwc_shape2", "INT32", [4], data=np.asarray(nhwc, dtype=np.int32)
        ),
        "cast_nhwc": _tensor("cast_nhwc", "FLOAT32", nhwc),
        "d_nhwc": _tensor("d_nhwc", "FLOAT32", nhwc),
        "nchw_shape2": _tensor(
            "nchw_shape2", "INT32", [4], data=np.asarray(nchw, dtype=np.int32)
        ),
        "d_nchw": _tensor("d_nchw", "FLOAT32", nchw),
    }
    model_ir.operators = [
        OperatorIR("RESHAPE", ["a_nchw", "nhwc_shape1"], ["a_nhwc"]),
        OperatorIR("MAX_POOL_2D", ["a_nhwc"], ["b_nhwc"]),
        OperatorIR("RESHAPE", ["b_nhwc", "nchw_shape1"], ["b_nchw"]),
        OperatorIR("EQUAL", ["a_nchw", "b_nchw"], ["cmp_nchw"]),
        OperatorIR("CAST", ["cmp_nchw"], ["cast_nchw"]),
        OperatorIR("RESHAPE", ["cast_nchw", "nhwc_shape2"], ["cast_nhwc"]),
        OperatorIR("MAX_POOL_2D", ["cast_nhwc"], ["d_nhwc"]),
        OperatorIR("RESHAPE", ["d_nhwc", "nchw_shape2"], ["d_nchw"]),
    ]
    if post_pool_fanout:
        model_ir.tensors["side"] = _tensor("side", "FLOAT32", nchw)
        model_ir.operators.append(OperatorIR("IDENTITY", ["b_nchw"], ["side"]))
    return model_ir


def test_singleton_maxpool_binary_cast_characterization() -> None:
    model_ir = _model(post_pool_fanout=False)

    stats = _optimize_singleton_layout_reshape_maxpool_binary_cast_chains(model_ir)

    assert stats == {
        "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains": 1
    }
    assert [op.op_type for op in model_ir.operators] == [
        "RESHAPE",
        "MAX_POOL_2D",
        "EQUAL",
        "CAST",
        "MAX_POOL_2D",
        "RESHAPE",
    ]
    equal_op = model_ir.operators[2]
    assert equal_op.inputs == ["a_nhwc", "b_nhwc"]
    assert model_ir.operators[4].inputs == ["cast_nchw"]
    assert model_ir.tensors["cmp_nchw"].shape == [1, 2, 3, 1]


def test_singleton_maxpool_binary_cast_rejects_post_pool_fanout() -> None:
    model_ir = _model(post_pool_fanout=True)

    stats = _optimize_singleton_layout_reshape_maxpool_binary_cast_chains(model_ir)

    assert stats == {
        "rewritten_singleton_layout_reshape_maxpool_binary_cast_chains": 0
    }
    assert [op.op_type for op in model_ir.operators[:3]] == [
        "RESHAPE",
        "MAX_POOL_2D",
        "RESHAPE",
    ]
