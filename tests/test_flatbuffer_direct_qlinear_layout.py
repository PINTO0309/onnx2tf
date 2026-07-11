from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_nhwc_propagation_qlinear_concat_conv,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def test_qlinear_concat_propagates_nhwc_across_singleton_layout_reshape() -> None:
    model_ir = ModelIR("qlinear_concat_singleton_layout_reshape")
    model_ir.inputs = ["gap_f_nhwc", "pool_f_nhwc"]
    model_ir.outputs = ["y"]

    _tensor(model_ir, "gap_f_nhwc", [1, 1, 1, 8])
    _tensor(model_ir, "gap_q_nhwc", [1, 1, 1, 8], "INT8")
    _tensor(model_ir, "gap_dq_nhwc", [1, 1, 1, 8])
    _tensor(model_ir, "pool_f_nhwc", [1, 1, 1, 8])
    _tensor(model_ir, "pool_f_nchw", [1, 8, 1, 1])
    _tensor(model_ir, "pool_q_nchw", [1, 8, 1, 1], "INT8")
    _tensor(model_ir, "pool_dq_nchw", [1, 8, 1, 1])
    _tensor(model_ir, "cat_f_nchw", [1, 16, 1, 1])
    _tensor(model_ir, "cat_q_nchw", [1, 16, 1, 1], "INT8")
    _tensor(model_ir, "cat_q_nhwc", [1, 1, 1, 16], "INT8")
    _tensor(model_ir, "conv_w", [3, 1, 1, 16], "INT8", np.ones((3, 1, 1, 16), np.int8))
    _tensor(model_ir, "conv_b", [3], "INT32", np.zeros((3,), np.int32))
    _tensor(model_ir, "y", [1, 1, 1, 3], "INT8")
    _tensor(model_ir, "nchw_shape", [4], "INT32", np.asarray([1, 8, 1, 1], np.int32))
    _tensor(model_ir, "cat_nhwc_shape", [4], "INT32", np.asarray([1, 1, 1, 16], np.int32))

    model_ir.operators = [
        OperatorIR("QUANTIZE", ["gap_f_nhwc"], ["gap_q_nhwc"]),
        OperatorIR("DEQUANTIZE", ["gap_q_nhwc"], ["gap_dq_nhwc"]),
        OperatorIR("RESHAPE", ["pool_f_nhwc", "nchw_shape"], ["pool_f_nchw"]),
        OperatorIR("QUANTIZE", ["pool_f_nchw"], ["pool_q_nchw"]),
        OperatorIR("DEQUANTIZE", ["pool_q_nchw"], ["pool_dq_nchw"]),
        OperatorIR(
            "CONCATENATION",
            ["gap_dq_nhwc", "pool_dq_nchw"],
            ["cat_f_nchw"],
            {"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR("QUANTIZE", ["cat_f_nchw"], ["cat_q_nchw"]),
        OperatorIR("RESHAPE", ["cat_q_nchw", "cat_nhwc_shape"], ["cat_q_nhwc"]),
        OperatorIR(
            "CONV_2D",
            ["cat_q_nhwc", "conv_w", "conv_b"],
            ["y"],
            {"padding": "SAME", "strideH": 1, "strideW": 1},
        ),
    ]

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)

    assert stats == {"propagated_qlinear_concat_conv_nhwc_chains": 1}
    assert all(op.op_type not in {"RESHAPE", "TRANSPOSE"} for op in model_ir.operators)
    pool_quantize = next(
        op for op in model_ir.operators if op.op_type == "QUANTIZE" and op.outputs == ["pool_q_nchw"]
    )
    assert pool_quantize.inputs == ["pool_f_nhwc"]
    concat = next(op for op in model_ir.operators if op.op_type == "CONCATENATION")
    assert concat.options["axis"] == 3
    assert model_ir.tensors["pool_q_nchw"].shape == [1, 1, 1, 8]
    assert model_ir.tensors["cat_q_nchw"].shape == [1, 1, 1, 16]
    conv = next(op for op in model_ir.operators if op.op_type == "CONV_2D")
    assert conv.inputs[0] == "cat_q_nchw"
