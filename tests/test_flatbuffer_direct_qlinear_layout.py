from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_nhwc_propagation_qlinear_concat_conv,
    _optimize_transpose_pre_add_nhwc_chains,
    _repair_unbound_nonconstant_operator_inputs_with_layout_transpose,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)


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


def _shape_signature(tensor: TensorIR) -> list[int]:
    return (
        [int(value) for value in tensor.shape_signature]
        if tensor.shape_signature is not None
        else [int(value) for value in tensor.shape]
    )


def _fingerprint(model_ir: ModelIR) -> bytes:
    return ModelIRPassState(model_ir).fingerprint()


def _build_qlinear_concat_conv_chain() -> ModelIR:
    model_ir = ModelIR("qlinear_concat_conv_characterization")
    model_ir.inputs = ["a_q_nhwc", "b_q_nhwc"]
    model_ir.outputs = ["y"]

    _tensor(model_ir, "a_q_nhwc", [1, 3, 5, 2], "INT8")
    _tensor(model_ir, "b_q_nhwc", [1, 3, 5, 2], "INT8")
    _tensor(model_ir, "a_q_nchw", [1, 2, 3, 5], "INT8")
    _tensor(model_ir, "b_q_nchw", [1, 2, 3, 5], "INT8")
    _tensor(model_ir, "a_f_nchw", [1, 2, 3, 5])
    _tensor(model_ir, "b_f_nchw", [1, 2, 3, 5])
    _tensor(model_ir, "cat_f_nchw", [1, 4, 3, 5])
    _tensor(model_ir, "cat_q_nchw", [1, 4, 3, 5], "INT8")
    _tensor(model_ir, "cat_q_nhwc", [1, 3, 5, 4], "INT8")
    _tensor(
        model_ir,
        "conv_w",
        [3, 1, 1, 4],
        "INT8",
        np.ones((3, 1, 1, 4), dtype=np.int8),
    )
    _tensor(
        model_ir,
        "conv_b",
        [3],
        "INT32",
        np.zeros((3,), dtype=np.int32),
    )
    _tensor(model_ir, "y", [1, 3, 5, 3], "INT8")
    _tensor(
        model_ir,
        "perm_nhwc_to_nchw",
        [4],
        "INT32",
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "perm_nchw_to_nhwc",
        [4],
        "INT32",
        np.asarray([0, 2, 3, 1], dtype=np.int32),
    )

    model_ir.operators = [
        OperatorIR(
            "TRANSPOSE",
            ["a_q_nhwc", "perm_nhwc_to_nchw"],
            ["a_q_nchw"],
        ),
        OperatorIR("DEQUANTIZE", ["a_q_nchw"], ["a_f_nchw"]),
        OperatorIR(
            "TRANSPOSE",
            ["b_q_nhwc", "perm_nhwc_to_nchw"],
            ["b_q_nchw"],
        ),
        OperatorIR("DEQUANTIZE", ["b_q_nchw"], ["b_f_nchw"]),
        OperatorIR(
            "CONCATENATION",
            ["a_f_nchw", "b_f_nchw"],
            ["cat_f_nchw"],
            {"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR("QUANTIZE", ["cat_f_nchw"], ["cat_q_nchw"]),
        OperatorIR(
            "TRANSPOSE",
            ["cat_q_nchw", "perm_nchw_to_nhwc"],
            ["cat_q_nhwc"],
        ),
        OperatorIR(
            "CONV_2D",
            ["cat_q_nhwc", "conv_w", "conv_b"],
            ["y"],
            {"padding": "SAME", "strideH": 1, "strideW": 1},
        ),
    ]
    return model_ir


def test_qlinear_concat_conv_pattern1_is_rewritten_and_idempotent() -> None:
    model_ir = _build_qlinear_concat_conv_chain()

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)

    assert stats == {"propagated_qlinear_concat_conv_nhwc_chains": 1}
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    dequantize_ops = [
        op for op in model_ir.operators if op.op_type == "DEQUANTIZE"
    ]
    assert [op.inputs for op in dequantize_ops] == [
        ["a_q_nhwc"],
        ["b_q_nhwc"],
    ]
    concat = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat.options["axis"] == 3
    assert model_ir.tensors["cat_f_nchw"].shape == [1, 3, 5, 4]
    assert model_ir.tensors["cat_q_nchw"].shape == [1, 3, 5, 4]
    conv = next(op for op in model_ir.operators if op.op_type == "CONV_2D")
    assert conv.inputs[0] == "cat_q_nchw"

    after_first = deepcopy(model_ir)
    assert _optimize_nhwc_propagation_qlinear_concat_conv(model_ir) == {
        "propagated_qlinear_concat_conv_nhwc_chains": 0
    }
    assert _fingerprint(model_ir) == _fingerprint(after_first)


def test_qlinear_concat_conv_pattern2_bypasses_float_pretranspose() -> None:
    model_ir = _build_qlinear_concat_conv_chain()
    model_ir.inputs[0] = "a_f_nhwc"
    _tensor(model_ir, "a_f_nhwc", [1, 3, 5, 2])
    _tensor(model_ir, "a_f_pre_nchw", [1, 2, 3, 5])
    model_ir.operators[0] = OperatorIR(
        "TRANSPOSE",
        ["a_f_nhwc", "perm_nhwc_to_nchw"],
        ["a_f_pre_nchw"],
    )
    model_ir.operators.insert(
        1,
        OperatorIR("QUANTIZE", ["a_f_pre_nchw"], ["a_q_nchw"]),
    )

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)

    assert stats == {"propagated_qlinear_concat_conv_nhwc_chains": 1}
    a_quantize = next(
        op
        for op in model_ir.operators
        if op.op_type == "QUANTIZE" and op.outputs == ["a_q_nchw"]
    )
    assert a_quantize.inputs == ["a_f_nhwc"]
    assert model_ir.tensors["a_q_nchw"].shape == [1, 3, 5, 2]
    assert model_ir.tensors["a_f_nchw"].shape == [1, 3, 5, 2]


def test_qlinear_concat_conv_pattern4_reinterprets_singleton_input() -> None:
    model_ir = _build_qlinear_concat_conv_chain()
    singleton_shapes = {
        "a_q_nhwc": [1, 1, 1, 2],
        "b_q_nhwc": [1, 1, 1, 2],
        "a_q_nchw": [1, 2, 1, 1],
        "b_q_nchw": [1, 2, 1, 1],
        "a_f_nchw": [1, 2, 1, 1],
        "b_f_nchw": [1, 2, 1, 1],
        "cat_f_nchw": [1, 4, 1, 1],
        "cat_q_nchw": [1, 4, 1, 1],
        "cat_q_nhwc": [1, 1, 1, 4],
        "y": [1, 1, 1, 3],
    }
    for tensor_name, shape in singleton_shapes.items():
        model_ir.tensors[tensor_name].shape = list(shape)
        model_ir.tensors[tensor_name].shape_signature = list(shape)
    model_ir.operators[0] = OperatorIR(
        "MAX_POOL_2D",
        ["a_q_nhwc"],
        ["a_q_nchw"],
    )

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)

    assert stats == {"propagated_qlinear_concat_conv_nhwc_chains": 1}
    assert model_ir.tensors["a_q_nchw"].shape == [1, 1, 1, 2]
    assert model_ir.tensors["a_f_nchw"].shape == [1, 1, 1, 2]
    concat = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat.options["axis"] == 3


def test_qlinear_concat_conv_preserves_dynamic_batch_and_remaps_qdim() -> None:
    model_ir = _build_qlinear_concat_conv_chain()
    signatures = {
        "a_q_nhwc": [-1, 3, 5, 2],
        "b_q_nhwc": [-1, 3, 5, 2],
        "a_q_nchw": [-1, 2, 3, 5],
        "b_q_nchw": [-1, 2, 3, 5],
        "a_f_nchw": [-1, 2, 3, 5],
        "b_f_nchw": [-1, 2, 3, 5],
        "cat_f_nchw": [-1, 4, 3, 5],
        "cat_q_nchw": [-1, 4, 3, 5],
        "cat_q_nhwc": [-1, 3, 5, 4],
        "y": [-1, 3, 5, 3],
    }
    for tensor_name, signature in signatures.items():
        model_ir.tensors[tensor_name].shape_signature = list(signature)
    model_ir.tensors["cat_q_nchw"].quantization = QuantParamIR(
        scale=[0.1, 0.2, 0.3, 0.4],
        zero_point=[0, 0, 0, 0],
        quantized_dimension=1,
    )

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)

    assert stats == {"propagated_qlinear_concat_conv_nhwc_chains": 1}
    assert _shape_signature(model_ir.tensors["cat_f_nchw"]) == [
        -1,
        3,
        5,
        4,
    ]
    assert _shape_signature(model_ir.tensors["cat_q_nchw"]) == [
        -1,
        3,
        5,
        4,
    ]
    assert (
        model_ir.tensors["cat_q_nchw"].quantization.quantized_dimension
        == 3
    )


@pytest.mark.parametrize(
    "case",
    [
        "wrong_pre_perm",
        "pre_fanout",
        "public_concat_output",
        "public_quantized_output",
        "public_post_output",
        "unexpected_concat_user",
        "mismatched_input_shape",
        "invalid_concat_axis",
        "wrong_post_perm",
    ],
)
def test_qlinear_concat_conv_rejects_unsafe_boundaries(case: str) -> None:
    model_ir = _build_qlinear_concat_conv_chain()
    if case == "wrong_pre_perm":
        model_ir.tensors["perm_nhwc_to_nchw"].data = np.asarray(
            [0, 2, 3, 1], dtype=np.int32
        )
    elif case == "pre_fanout":
        _tensor(model_ir, "a_fanout", [1, 2, 3, 5], "INT8")
        model_ir.operators.append(
            OperatorIR("RELU", ["a_q_nchw"], ["a_fanout"])
        )
    elif case == "public_concat_output":
        model_ir.outputs.append("cat_f_nchw")
    elif case == "public_quantized_output":
        model_ir.outputs.append("cat_q_nchw")
    elif case == "public_post_output":
        model_ir.outputs.append("cat_q_nhwc")
    elif case == "unexpected_concat_user":
        _tensor(model_ir, "concat_fanout", [1, 4, 3, 5])
        model_ir.operators.append(
            OperatorIR("RELU", ["cat_f_nchw"], ["concat_fanout"])
        )
    elif case == "mismatched_input_shape":
        model_ir.tensors["b_q_nhwc"].shape = [1, 4, 5, 2]
        model_ir.tensors["b_q_nhwc"].shape_signature = [1, 4, 5, 2]
    elif case == "invalid_concat_axis":
        model_ir.operators[4].options["axis"] = 4
    elif case == "wrong_post_perm":
        model_ir.tensors["perm_nchw_to_nhwc"].data = np.asarray(
            [0, 3, 1, 2], dtype=np.int32
        )
    before = deepcopy(model_ir)

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)

    assert stats == {"propagated_qlinear_concat_conv_nhwc_chains": 0}
    assert _fingerprint(model_ir) == _fingerprint(before)


@pytest.mark.parametrize(
    "missing_tensor_name",
    ["cat_f_nchw", "cat_q_nchw"],
)
@pytest.mark.xfail(
    strict=True,
    reason="required output tensors are checked after candidate mutation",
)
def test_qlinear_concat_conv_missing_output_tensor_rejection_is_atomic(
    missing_tensor_name: str,
) -> None:
    model_ir = _build_qlinear_concat_conv_chain()
    del model_ir.tensors[missing_tensor_name]
    before = deepcopy(model_ir)

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)

    assert stats == {"propagated_qlinear_concat_conv_nhwc_chains": 0}
    assert _fingerprint(model_ir) == _fingerprint(before)


@pytest.mark.xfail(
    strict=True,
    reason="a public Dequantize output is changed from NCHW to NHWC",
)
def test_qlinear_concat_conv_preserves_public_dequantize_output() -> None:
    model_ir = _build_qlinear_concat_conv_chain()
    model_ir.outputs.append("a_f_nchw")
    before = deepcopy(model_ir)

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)

    assert stats == {"propagated_qlinear_concat_conv_nhwc_chains": 0}
    assert _fingerprint(model_ir) == _fingerprint(before)


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


def test_flatbuffer_direct_qlinear_concat_conv_layout_propagation_with_concat_post_transpose() -> None:
    model_ir = ModelIR("qlinear_concat_conv_layout_with_concat_post_transpose_test")
    model_ir.inputs = ["a_q_nhwc", "b_q_nhwc"]
    model_ir.outputs = ["y0", "pool_out"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("a_q_nhwc", [1, 3, 5, 2], "INT8")
    _add_tensor("b_q_nhwc", [1, 3, 5, 2], "INT8")
    _add_tensor("a_q_nchw", [1, 2, 3, 5], "INT8")
    _add_tensor("b_q_nchw", [1, 2, 3, 5], "INT8")
    _add_tensor("a_f_nchw", [1, 2, 3, 5])
    _add_tensor("b_f_nchw", [1, 2, 3, 5])
    _add_tensor("cat_f_nchw", [1, 4, 3, 5])
    _add_tensor("cat_f_nhwc", [1, 3, 5, 4])
    _add_tensor("pool_out", [1, 3, 5, 4])
    _add_tensor("cat_q", [1, 4, 3, 5], "INT8")
    _add_tensor("cat_q_nhwc_0", [1, 3, 5, 4], "INT8")
    _add_tensor("cat_q_nhwc_1", [1, 3, 5, 4], "INT8")
    _add_tensor("conv_w0", [3, 1, 1, 4], "INT8", np.ones((3, 1, 1, 4), dtype=np.int8))
    _add_tensor("conv_b0", [3], "INT32", np.zeros((3,), dtype=np.int32))
    _add_tensor("conv_w1", [3, 1, 1, 4], "INT8", np.ones((3, 1, 1, 4), dtype=np.int8))
    _add_tensor("conv_b1", [3], "INT32", np.zeros((3,), dtype=np.int32))
    _add_tensor("y0", [1, 3, 5, 3], "INT8")
    _add_tensor("y1", [1, 3, 5, 3], "INT8")
    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_nchw_to_nhwc", [4], "INT32", np.asarray([0, 2, 3, 1], dtype=np.int32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_q_nhwc", "perm_nhwc_to_nchw"], outputs=["a_q_nchw"]),
        OperatorIR(op_type="DEQUANTIZE", inputs=["a_q_nchw"], outputs=["a_f_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_q_nhwc", "perm_nhwc_to_nchw"], outputs=["b_q_nchw"]),
        OperatorIR(op_type="DEQUANTIZE", inputs=["b_q_nchw"], outputs=["b_f_nchw"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["a_f_nchw", "b_f_nchw"],
            outputs=["cat_f_nchw"],
            options={"axis": 1, "fused_activation_function": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat_f_nchw", "perm_nchw_to_nhwc"], outputs=["cat_f_nhwc"]),
        OperatorIR(
            op_type="MAX_POOL_2D",
            inputs=["cat_f_nhwc"],
            outputs=["pool_out"],
            options={
                "padding": "SAME",
                "stride_w": 1,
                "stride_h": 1,
                "filter_width": 3,
                "filter_height": 3,
                "fused_activation_function": "NONE",
            },
        ),
        OperatorIR(op_type="QUANTIZE", inputs=["cat_f_nchw"], outputs=["cat_q"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat_q", "perm_nchw_to_nhwc"], outputs=["cat_q_nhwc_0"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["cat_q_nhwc_0", "conv_w0", "conv_b0"],
            outputs=["y0"],
            options={
                "padding": "SAME",
                "stride_w": 1,
                "stride_h": 1,
                "dilation_w_factor": 1,
                "dilation_h_factor": 1,
                "fused_activation_function": "NONE",
                "quantized_bias_type": "INT32",
            },
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat_q", "perm_nchw_to_nhwc"], outputs=["cat_q_nhwc_1"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["cat_q_nhwc_1", "conv_w1", "conv_b1"],
            outputs=["y1"],
            options={
                "padding": "SAME",
                "stride_w": 1,
                "stride_h": 1,
                "dilation_w_factor": 1,
                "dilation_h_factor": 1,
                "fused_activation_function": "NONE",
                "quantized_bias_type": "INT32",
            },
        ),
    ]

    stats = _optimize_nhwc_propagation_qlinear_concat_conv(model_ir)
    assert stats["propagated_qlinear_concat_conv_nhwc_chains"] == 1

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0

    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3

    dq_ops = [op for op in model_ir.operators if str(op.op_type) == "DEQUANTIZE"]
    assert len(dq_ops) == 2
    assert [str(dq_ops[0].inputs[0]), str(dq_ops[1].inputs[0])] == ["a_q_nhwc", "b_q_nhwc"]

    maxpool_op = next(op for op in model_ir.operators if str(op.op_type) == "MAX_POOL_2D")
    assert list(maxpool_op.inputs) == ["cat_f_nchw"]

    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 2
    assert all(str(op.inputs[0]) == "cat_q" for op in conv_ops)

    cat_f_tensor = model_ir.tensors["cat_f_nchw"]
    cat_q_tensor = model_ir.tensors["cat_q"]
    assert list(cat_f_tensor.shape) == [1, 3, 5, 4]
    assert list(cat_q_tensor.shape) == [1, 3, 5, 4]
    assert _shape_signature(cat_q_tensor) == [1, 3, 5, 4]
