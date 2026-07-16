from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_nhwc_prefix_qlinear_silu_chains,
)


_INTERNAL_PERM_NAME = "__nhwc_to_nchw_perm_rank4__"


def _tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    quantized: bool = False,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        quantization=(
            QuantParamIR(scale=[0.125], zero_point=[0])
            if quantized
            else None
        ),
    )


def _fingerprint(model_ir: ModelIR) -> bytes:
    return ModelIRPassState(model_ir).fingerprint()


def _build_qlinear_silu_prefix_chain(
    *,
    pattern: str = "logistic",
    legacy_consumer: bool = False,
) -> ModelIR:
    model_ir = ModelIR(f"qlinear_silu_prefix_{pattern}")
    model_ir.inputs = ["q_raw_nhwc"]
    model_ir.outputs = ["final_y"]

    _tensor(
        model_ir,
        "q_raw_nhwc",
        [1, 2, 3, 4],
        dtype="INT8",
        quantized=True,
    )
    _tensor(
        model_ir,
        "q_nchw",
        [1, 4, 2, 3],
        dtype="INT8",
        quantized=True,
    )
    _tensor(model_ir, "dq_nchw", [1, 4, 2, 3])
    _tensor(
        model_ir,
        "sig_q_nchw",
        [1, 4, 2, 3],
        dtype="INT8",
        quantized=True,
    )
    _tensor(
        model_ir,
        "mul_out_nchw",
        [1, 4, 2, 3],
        dtype="INT8",
        quantized=True,
    )
    _tensor(
        model_ir,
        "perm_nhwc_to_nchw",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _tensor(
        model_ir,
        "perm_nchw_to_nhwc",
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )

    operators = [
        OperatorIR(
            "TRANSPOSE",
            ["q_raw_nhwc", "perm_nhwc_to_nchw"],
            ["q_nchw"],
        ),
        OperatorIR("DEQUANTIZE", ["q_nchw"], ["dq_nchw"]),
    ]

    if pattern == "logistic":
        _tensor(model_ir, "sig_f_nchw", [1, 4, 2, 3])
        operators.extend(
            [
                OperatorIR("LOGISTIC", ["dq_nchw"], ["sig_f_nchw"]),
                OperatorIR("QUANTIZE", ["sig_f_nchw"], ["sig_q_nchw"]),
            ]
        )
    elif pattern == "hard_sigmoid":
        intermediate_names = [
            "hs_mul_nchw",
            "hs_add_nchw",
            "hs_max_nchw",
            "hs_min_nchw",
        ]
        for intermediate_name in intermediate_names:
            _tensor(model_ir, intermediate_name, [1, 4, 2, 3])
        for name, value in (
            ("hs_scale", np.float32(1.0 / 6.0)),
            ("hs_bias", np.float32(0.5)),
            ("hs_floor", np.float32(0.0)),
            ("hs_ceiling", np.float32(1.0)),
        ):
            _tensor(
                model_ir,
                name,
                [1],
                data=np.asarray([value], dtype=np.float32),
            )
        operators.extend(
            [
                OperatorIR(
                    "MUL",
                    ["dq_nchw", "hs_scale"],
                    ["hs_mul_nchw"],
                ),
                OperatorIR(
                    "ADD",
                    ["hs_bias", "hs_mul_nchw"],
                    ["hs_add_nchw"],
                ),
                OperatorIR(
                    "MAXIMUM",
                    ["hs_add_nchw", "hs_floor"],
                    ["hs_max_nchw"],
                ),
                OperatorIR(
                    "MINIMUM",
                    ["hs_ceiling", "hs_max_nchw"],
                    ["hs_min_nchw"],
                ),
                OperatorIR("QUANTIZE", ["hs_min_nchw"], ["sig_q_nchw"]),
            ]
        )
    else:
        raise ValueError(f"unsupported pattern: {pattern}")

    operators.append(
        OperatorIR(
            "MUL",
            ["q_nchw", "sig_q_nchw"],
            ["mul_out_nchw"],
        )
    )
    if legacy_consumer:
        _tensor(
            model_ir,
            "legacy_relu_out",
            [1, 4, 2, 3],
            dtype="INT8",
            quantized=True,
        )
        _tensor(
            model_ir,
            "final_y",
            [1, 4, 2, 3],
            dtype="INT8",
            quantized=True,
        )
        operators.extend(
            [
                OperatorIR("RELU", ["mul_out_nchw"], ["legacy_relu_out"]),
                OperatorIR("IDENTITY", ["legacy_relu_out"], ["final_y"]),
            ]
        )
    else:
        _tensor(
            model_ir,
            "post_nhwc",
            [1, 2, 3, 4],
            dtype="INT8",
            quantized=True,
        )
        _tensor(
            model_ir,
            "final_y",
            [1, 2, 3, 4],
            dtype="INT8",
            quantized=True,
        )
        operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["mul_out_nchw", "perm_nchw_to_nhwc"],
                    ["post_nhwc"],
                ),
                OperatorIR("RELU", ["post_nhwc"], ["final_y"]),
            ]
        )

    model_ir.operators = operators
    return model_ir


def _prefix_model_ir(model_ir: ModelIR, prefix: str) -> ModelIR:
    prefixed = copy.deepcopy(model_ir)
    tensor_names = {
        name: f"{prefix}{name}" for name in prefixed.tensors
    }
    prefixed.tensors = {
        tensor_names[name]: tensor
        for name, tensor in prefixed.tensors.items()
    }
    for name, tensor in prefixed.tensors.items():
        tensor.name = name
    prefixed.inputs = [tensor_names[name] for name in prefixed.inputs]
    prefixed.outputs = [tensor_names[name] for name in prefixed.outputs]
    for op in prefixed.operators:
        op.inputs = [tensor_names[name] for name in op.inputs]
        op.outputs = [tensor_names[name] for name in op.outputs]
    return prefixed


@pytest.mark.parametrize(
    ("pattern", "intermediate_names"),
    [
        ("logistic", ["dq_nchw", "sig_f_nchw"]),
        (
            "hard_sigmoid",
            [
                "dq_nchw",
                "hs_mul_nchw",
                "hs_add_nchw",
                "hs_max_nchw",
                "hs_min_nchw",
            ],
        ),
    ],
)
def test_qlinear_silu_prefix_patterns_rewrite_to_nhwc(
    pattern: str,
    intermediate_names: list[str],
) -> None:
    model_ir = _build_qlinear_silu_prefix_chain(pattern=pattern)

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 1}
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    dequantize = next(op for op in model_ir.operators if op.op_type == "DEQUANTIZE")
    final_mul = next(
        op
        for op in model_ir.operators
        if op.op_type == "MUL" and op.outputs == ["mul_out_nchw"]
    )
    final_relu = next(op for op in model_ir.operators if op.outputs == ["final_y"])
    assert dequantize.inputs == ["q_raw_nhwc"]
    assert final_mul.inputs == ["q_raw_nhwc", "sig_q_nchw"]
    assert final_relu.inputs == ["mul_out_nchw"]
    for tensor_name in intermediate_names + ["sig_q_nchw", "mul_out_nchw"]:
        assert model_ir.tensors[tensor_name].shape == [1, 2, 3, 4]
        assert model_ir.tensors[tensor_name].shape_signature == [1, 2, 3, 4]
    assert "q_nchw" not in model_ir.tensors
    assert "post_nhwc" not in model_ir.tensors
    assert _INTERNAL_PERM_NAME not in model_ir.tensors


def test_qlinear_silu_prefix_fixed_point_rewrites_multiple_patterns() -> None:
    model_ir = _build_qlinear_silu_prefix_chain(pattern="logistic")
    second = _prefix_model_ir(
        _build_qlinear_silu_prefix_chain(pattern="hard_sigmoid"),
        "second_",
    )
    model_ir.inputs.extend(second.inputs)
    model_ir.outputs.extend(second.outputs)
    model_ir.tensors.update(second.tensors)
    model_ir.operators.extend(second.operators)

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 2}
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    dequantize_inputs = [
        op.inputs
        for op in model_ir.operators
        if op.op_type == "DEQUANTIZE"
    ]
    assert dequantize_inputs == [["q_raw_nhwc"], ["second_q_raw_nhwc"]]


def test_qlinear_silu_prefix_inserts_adapter_for_legacy_consumer() -> None:
    model_ir = _build_qlinear_silu_prefix_chain(legacy_consumer=True)

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 1}
    transpose_ops = [op for op in model_ir.operators if op.op_type == "TRANSPOSE"]
    assert len(transpose_ops) == 1
    adapter = transpose_ops[0]
    assert adapter.inputs == ["mul_out_nchw", _INTERNAL_PERM_NAME]
    assert adapter.outputs == ["mul_out_nchw_nchw_adapter"]
    relu = next(op for op in model_ir.operators if op.op_type == "RELU")
    assert relu.inputs == ["mul_out_nchw_nchw_adapter"]
    assert model_ir.tensors["mul_out_nchw"].shape == [1, 2, 3, 4]
    assert model_ir.tensors["mul_out_nchw_nchw_adapter"].shape == [1, 4, 2, 3]
    assert np.array_equal(
        np.asarray(model_ir.tensors[_INTERNAL_PERM_NAME].data),
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )


def test_qlinear_silu_prefix_reuses_exact_internal_perm_tensor() -> None:
    model_ir = _build_qlinear_silu_prefix_chain(legacy_consumer=True)
    _tensor(
        model_ir,
        _INTERNAL_PERM_NAME,
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 1}
    adapter = next(op for op in model_ir.operators if op.op_type == "TRANSPOSE")
    assert adapter.inputs[1] == _INTERNAL_PERM_NAME
    assert f"{_INTERNAL_PERM_NAME}_1" not in model_ir.tensors


def test_qlinear_silu_prefix_plans_each_legacy_consumer_slot_once() -> None:
    model_ir = _build_qlinear_silu_prefix_chain(legacy_consumer=True)
    legacy_op = next(
        op for op in model_ir.operators if op.outputs == ["legacy_relu_out"]
    )
    legacy_op.op_type = "ADD"
    legacy_op.inputs = ["mul_out_nchw", "mul_out_nchw"]

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 1}
    adapter_ops = [op for op in model_ir.operators if op.op_type == "TRANSPOSE"]
    assert [op.outputs for op in adapter_ops] == [
        ["mul_out_nchw_nchw_adapter"],
        ["mul_out_nchw_nchw_adapter_1"],
    ]
    assert legacy_op.inputs == [
        "mul_out_nchw_nchw_adapter",
        "mul_out_nchw_nchw_adapter_1",
    ]


def test_qlinear_silu_prefix_preserves_distinct_legacy_consumer_order() -> None:
    model_ir = _build_qlinear_silu_prefix_chain(legacy_consumer=True)
    _tensor(
        model_ir,
        "side_relu_out",
        [1, 4, 2, 3],
        dtype="INT8",
        quantized=True,
    )
    _tensor(
        model_ir,
        "side_final_y",
        [1, 4, 2, 3],
        dtype="INT8",
        quantized=True,
    )
    model_ir.outputs.append("side_final_y")
    model_ir.operators.extend(
        [
            OperatorIR("RELU", ["mul_out_nchw"], ["side_relu_out"]),
            OperatorIR("IDENTITY", ["side_relu_out"], ["side_final_y"]),
        ]
    )

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 1}
    adapter_ops = [op for op in model_ir.operators if op.op_type == "TRANSPOSE"]
    assert [op.outputs for op in adapter_ops] == [
        ["mul_out_nchw_nchw_adapter"],
        ["mul_out_nchw_nchw_adapter_1"],
    ]
    relu_inputs = [
        op.inputs for op in model_ir.operators if op.op_type == "RELU"
    ]
    assert relu_inputs == [
        ["mul_out_nchw_nchw_adapter"],
        ["mul_out_nchw_nchw_adapter_1"],
    ]


@pytest.mark.parametrize(
    "case",
    [
        "wrong_pre_perm",
        "public_pre_output",
        "pre_output_fanout",
        "per_axis_pre_quantization",
        "shared_sigmoid_quantize_output",
        "blocked_mul_consumer",
        "public_post_output",
        "non_singleton_hard_sigmoid_constant",
    ],
)
def test_qlinear_silu_prefix_rejects_unsafe_boundaries(case: str) -> None:
    pattern = (
        "hard_sigmoid"
        if case == "non_singleton_hard_sigmoid_constant"
        else "logistic"
    )
    model_ir = _build_qlinear_silu_prefix_chain(pattern=pattern)
    if case == "wrong_pre_perm":
        model_ir.tensors["perm_nhwc_to_nchw"].data = np.asarray(
            [0, 1, 2, 3],
            dtype=np.int32,
        )
    elif case == "public_pre_output":
        model_ir.outputs.append("q_nchw")
    elif case == "pre_output_fanout":
        _tensor(model_ir, "side_out", [1, 4, 2, 3])
        model_ir.operators.append(OperatorIR("RELU", ["q_nchw"], ["side_out"]))
    elif case == "per_axis_pre_quantization":
        model_ir.tensors["q_raw_nhwc"].quantization = QuantParamIR(
            scale=[0.1, 0.2],
            zero_point=[0, 0],
            quantized_dimension=3,
        )
    elif case == "shared_sigmoid_quantize_output":
        _tensor(model_ir, "sig_side_out", [1, 4, 2, 3])
        model_ir.operators.append(
            OperatorIR("RELU", ["sig_q_nchw"], ["sig_side_out"])
        )
    elif case == "blocked_mul_consumer":
        post_op = next(op for op in model_ir.operators if op.outputs == ["post_nhwc"])
        post_op.op_type = "SOFTMAX"
    elif case == "public_post_output":
        model_ir.outputs = ["post_nhwc"]
    elif case == "non_singleton_hard_sigmoid_constant":
        model_ir.tensors["hs_scale"].data = np.asarray(
            [1.0 / 6.0, 1.0 / 6.0],
            dtype=np.float32,
        )
        model_ir.tensors["hs_scale"].shape = [2]
        model_ir.tensors["hs_scale"].shape_signature = [2]
    else:
        raise AssertionError(case)

    before_operators = copy.deepcopy(model_ir.operators)
    before_shapes = {
        name: (list(tensor.shape), list(tensor.shape_signature or []))
        for name, tensor in model_ir.tensors.items()
    }

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 0}
    assert model_ir.operators == before_operators
    assert {
        name: (list(tensor.shape), list(tensor.shape_signature or []))
        for name, tensor in model_ir.tensors.items()
    } == before_shapes


def test_qlinear_silu_prefix_rejection_is_complete_model_ir_noop() -> None:
    model_ir = _build_qlinear_silu_prefix_chain()
    model_ir.tensors["perm_nhwc_to_nchw"].data = np.asarray(
        [0, 1, 2, 3],
        dtype=np.int32,
    )
    before = _fingerprint(model_ir)
    before_metadata = copy.deepcopy(model_ir.metadata)

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 0}
    assert _fingerprint(model_ir) == before
    assert model_ir.metadata == before_metadata


def test_qlinear_silu_prefix_is_idempotent_after_rewrite() -> None:
    model_ir = _build_qlinear_silu_prefix_chain()
    assert _optimize_nhwc_prefix_qlinear_silu_chains(model_ir) == {
        "optimized_nhwc_prefix_qlinear_silu_chains": 1,
    }
    after_first = _fingerprint(model_ir)
    metadata_after_first = copy.deepcopy(model_ir.metadata)

    assert _optimize_nhwc_prefix_qlinear_silu_chains(model_ir) == {
        "optimized_nhwc_prefix_qlinear_silu_chains": 0,
    }
    assert _fingerprint(model_ir) == after_first
    assert model_ir.metadata == metadata_after_first


def test_qlinear_silu_prefix_does_not_reuse_colliding_internal_perm_name() -> None:
    model_ir = _build_qlinear_silu_prefix_chain(legacy_consumer=True)
    _tensor(
        model_ir,
        _INTERNAL_PERM_NAME,
        [4],
        dtype="INT32",
        data=np.asarray([0, 1, 2, 3], dtype=np.int32),
    )
    model_ir.outputs.append(_INTERNAL_PERM_NAME)

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 1}
    adapter = next(op for op in model_ir.operators if op.op_type == "TRANSPOSE")
    adapter_perm_name = str(adapter.inputs[1])
    assert adapter_perm_name != _INTERNAL_PERM_NAME
    assert np.array_equal(
        np.asarray(model_ir.tensors[adapter_perm_name].data),
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert np.array_equal(
        np.asarray(model_ir.tensors[_INTERNAL_PERM_NAME].data),
        np.asarray([0, 1, 2, 3], dtype=np.int32),
    )


@pytest.mark.parametrize(
    "tensor_name",
    ["dq_nchw", "sig_f_nchw", "sig_q_nchw", "mul_out_nchw"],
)
def test_qlinear_silu_prefix_rejects_short_target_signature_atomically(
    tensor_name: str,
) -> None:
    model_ir = _build_qlinear_silu_prefix_chain(legacy_consumer=True)
    model_ir.tensors[tensor_name].shape_signature = [1, 4]
    before = _fingerprint(model_ir)

    stats = _optimize_nhwc_prefix_qlinear_silu_chains(model_ir)

    assert stats == {"optimized_nhwc_prefix_qlinear_silu_chains": 0}
    assert _fingerprint(model_ir) == before
