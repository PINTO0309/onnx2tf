from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_nhwc_propagation_qlinear_concat_conv,
    _optimize_transpose_pre_add_nhwc_chains,
    _repair_unbound_nonconstant_operator_inputs_with_layout_transpose,
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


def test_transpose_pre_add_rewrite_is_idempotent_with_legacy_consumer() -> None:
    model_ir = ModelIR("qlinear_residual_add_idempotency")
    model_ir.inputs = ["left_nhwc", "right_nhwc"]
    model_ir.outputs = ["legacy_f32", "tail"]
    _tensor(model_ir, "left_nhwc", [1, 8, 8, 16], "INT8")
    _tensor(model_ir, "right_nhwc", [1, 8, 8, 16], "INT8")
    _tensor(model_ir, "left_nchw", [1, 16, 8, 8], "INT8")
    _tensor(model_ir, "right_nchw", [1, 16, 8, 8], "INT8")
    _tensor(model_ir, "sum_nchw", [1, 16, 8, 8], "INT8")
    _tensor(model_ir, "sum_nhwc", [1, 8, 8, 16], "INT8")
    _tensor(model_ir, "legacy_f32", [1, 16, 8, 8])
    _tensor(model_ir, "tail", [1, 8, 8, 16], "INT8")
    _tensor(
        model_ir,
        "to_nchw_perm",
        [4],
        "INT32",
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "to_nhwc_perm",
        [4],
        "INT32",
        np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["left_nhwc", "to_nchw_perm"], ["left_nchw"]),
        OperatorIR("TRANSPOSE", ["right_nhwc", "to_nchw_perm"], ["right_nchw"]),
        OperatorIR("ADD", ["left_nchw", "right_nchw"], ["sum_nchw"]),
        OperatorIR("TRANSPOSE", ["sum_nchw", "to_nhwc_perm"], ["sum_nhwc"]),
        OperatorIR("DEQUANTIZE", ["sum_nchw"], ["legacy_f32"]),
        OperatorIR("RELU", ["sum_nhwc"], ["tail"]),
    ]

    first = _optimize_transpose_pre_add_nhwc_chains(model_ir)
    second = _optimize_transpose_pre_add_nhwc_chains(model_ir)

    assert first == {"optimized_transpose_pre_add_nhwc_chains": 1}
    assert second == {"optimized_transpose_pre_add_nhwc_chains": 0}
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.options["__transpose_pre_add_nhwc_optimized__"] is True
    producers = {
        output_name
        for op in model_ir.operators
        for output_name in op.outputs
    }
    assert "sum_nchw" in producers
    assert next(op for op in model_ir.operators if op.op_type == "DEQUANTIZE").inputs == [
        "sum_nchw"
    ]


def test_repairs_unbound_qlinear_residual_from_exact_nhwc_bridge() -> None:
    model_ir = ModelIR("qlinear_residual_unbound_bridge")
    model_ir.inputs = ["left", "right"]
    model_ir.outputs = ["sum_f32"]
    _tensor(model_ir, "left", [1, 8, 8, 16], "INT8")
    _tensor(model_ir, "right", [1, 8, 8, 16], "INT8")
    _tensor(model_ir, "sum_quantized_nhwc_bridge", [1, 8, 8, 16], "INT8")
    _tensor(model_ir, "sum_quantized", [1, 16, 8, 8], "INT8")
    _tensor(model_ir, "sum_f32", [1, 16, 8, 8])
    model_ir.operators = [
        OperatorIR(
            "ADD",
            ["left", "right"],
            ["sum_quantized_nhwc_bridge"],
        ),
        OperatorIR("DEQUANTIZE", ["sum_quantized"], ["sum_f32"]),
    ]

    stats = _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
        model_ir
    )

    assert stats == {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": 1
    }
    assert [op.op_type for op in model_ir.operators] == [
        "ADD",
        "TRANSPOSE",
        "DEQUANTIZE",
    ]
    transpose_op = model_ir.operators[1]
    assert transpose_op.inputs[0] == "sum_quantized_nhwc_bridge"
    assert transpose_op.outputs == ["sum_quantized"]
