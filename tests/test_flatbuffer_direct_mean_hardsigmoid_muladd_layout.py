from __future__ import annotations

from copy import deepcopy

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
    _optimize_transpose_mean_hardsigmoid_muladd_chains,
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


def _build_chain() -> ModelIR:
    nhwc = [1, 4, 4, 3]
    nchw = [1, 3, 4, 4]
    reduced_nchw = [1, 3, 1, 1]
    model_ir = ModelIR("mean_hardsigmoid_muladd")
    model_ir.inputs = ["q0_raw", "q1_raw"]
    model_ir.outputs = ["mean_user_out", "legacy_out"]
    model_ir.tensors = {
        "q0_raw": _tensor("q0_raw", nhwc, dtype="INT8"),
        "q0_pre_perm": _tensor(
            "q0_pre_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "q0_nchw": _tensor("q0_nchw", nchw, dtype="INT8"),
        "dq0_out": _tensor("dq0_out", nchw),
        "mean_axes": _tensor(
            "mean_axes",
            [2],
            dtype="INT32",
            data=np.asarray([2, 3], dtype=np.int32),
        ),
        "mean_out": _tensor("mean_out", reduced_nchw),
        "qmean_out": _tensor("qmean_out", reduced_nchw, dtype="INT8"),
        "mean_post_perm": _tensor(
            "mean_post_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "mean_post_out": _tensor("mean_post_out", [1, 1, 1, 3], dtype="INT8"),
        "mean_user_out": _tensor("mean_user_out", [1, 1, 1, 3], dtype="INT8"),
        "q1_raw": _tensor("q1_raw", nhwc, dtype="INT8"),
        "q1_pre_perm": _tensor(
            "q1_pre_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "q1_nchw": _tensor("q1_nchw", nchw, dtype="INT8"),
        "dq1_out": _tensor("dq1_out", nchw),
        "mul_c": _tensor(
            "mul_c",
            [1],
            data=np.asarray([1.0 / 6.0], dtype=np.float32),
        ),
        "hsig_mul_out": _tensor("hsig_mul_out", nchw),
        "add_c": _tensor(
            "add_c",
            [1],
            data=np.asarray([0.5], dtype=np.float32),
        ),
        "hsig_add_out": _tensor("hsig_add_out", nchw),
        "max_c": _tensor(
            "max_c",
            [1],
            data=np.asarray([0.0], dtype=np.float32),
        ),
        "hsig_max_out": _tensor("hsig_max_out", nchw),
        "min_c": _tensor(
            "min_c",
            [1],
            data=np.asarray([1.0], dtype=np.float32),
        ),
        "hsig_min_out": _tensor("hsig_min_out", nchw),
        "sig_q": _tensor("sig_q", nchw, dtype="INT8"),
        "mul0_out": _tensor("mul0_out", nchw, dtype="INT8"),
        "add0_out": _tensor("add0_out", nchw, dtype="INT8"),
        "legacy_out": _tensor("legacy_out", nchw, dtype="INT8"),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["q0_raw", "q0_pre_perm"], ["q0_nchw"]),
        OperatorIR("DEQUANTIZE", ["q0_nchw"], ["dq0_out"]),
        OperatorIR(
            "MEAN",
            ["dq0_out", "mean_axes"],
            ["mean_out"],
            options={"keepDims": True, "axes": [2, 3]},
        ),
        OperatorIR("QUANTIZE", ["mean_out"], ["qmean_out"]),
        OperatorIR(
            "TRANSPOSE",
            ["qmean_out", "mean_post_perm"],
            ["mean_post_out"],
        ),
        OperatorIR("RELU", ["mean_post_out"], ["mean_user_out"]),
        OperatorIR("TRANSPOSE", ["q1_raw", "q1_pre_perm"], ["q1_nchw"]),
        OperatorIR("DEQUANTIZE", ["q1_nchw"], ["dq1_out"]),
        OperatorIR("MUL", ["dq1_out", "mul_c"], ["hsig_mul_out"]),
        OperatorIR("ADD", ["hsig_mul_out", "add_c"], ["hsig_add_out"]),
        OperatorIR("MAXIMUM", ["hsig_add_out", "max_c"], ["hsig_max_out"]),
        OperatorIR("MINIMUM", ["hsig_max_out", "min_c"], ["hsig_min_out"]),
        OperatorIR("QUANTIZE", ["hsig_min_out"], ["sig_q"]),
        OperatorIR("MUL", ["q0_nchw", "sig_q"], ["mul0_out"]),
        OperatorIR("ADD", ["q0_nchw", "mul0_out"], ["add0_out"]),
        OperatorIR("RELU", ["add0_out"], ["legacy_out"]),
    ]
    return model_ir


def _fingerprint(model_ir: ModelIR) -> bytes:
    return ModelIRPassState(model_ir).fingerprint()


def test_mean_hardsigmoid_muladd_chain_is_rewritten() -> None:
    model_ir = _build_chain()

    stats = _optimize_transpose_mean_hardsigmoid_muladd_chains(model_ir)

    assert stats == {"optimized_transpose_mean_hardsigmoid_muladd_chains": 1}
    assert model_ir.operators[0].op_type == "DEQUANTIZE"
    assert model_ir.operators[0].inputs == ["q0_raw"]
    dq1 = next(op for op in model_ir.operators if op.outputs == ["dq1_out"])
    assert dq1.inputs == ["q1_raw"]
    np.testing.assert_array_equal(
        model_ir.tensors["mean_axes"].data,
        np.asarray([1, 2], dtype=np.int32),
    )
    mean_user = next(
        op for op in model_ir.operators if op.outputs == ["mean_user_out"]
    )
    assert mean_user.inputs == ["qmean_out"]
    mul0 = next(op for op in model_ir.operators if op.outputs == ["mul0_out"])
    add0 = next(op for op in model_ir.operators if op.outputs == ["add0_out"])
    assert mul0.inputs == ["q0_raw", "sig_q"]
    assert add0.inputs == ["q0_raw", "mul0_out"]
    legacy = next(op for op in model_ir.operators if op.outputs == ["legacy_out"])
    assert legacy.inputs == ["add0_out_nchw_adapter"]
    adapter = next(
        op
        for op in model_ir.operators
        if op.outputs == ["add0_out_nchw_adapter"]
    )
    assert adapter.op_type == "TRANSPOSE"
    assert adapter.inputs == ["add0_out", "__nhwc_to_nchw_perm_rank4__"]
    assert not any(
        op.op_type == "TRANSPOSE"
        and op.outputs[0] in {"q0_nchw", "mean_post_out", "q1_nchw"}
        for op in model_ir.operators
    )

    after_first = deepcopy(model_ir)
    assert _optimize_transpose_mean_hardsigmoid_muladd_chains(model_ir) == {
        "optimized_transpose_mean_hardsigmoid_muladd_chains": 0
    }
    assert _fingerprint(model_ir) == _fingerprint(after_first)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_q0_perm",
        "public_q0_bridge",
        "q0_fanout",
        "mean_not_keepdims",
        "wrong_post_perm",
        "sig_q_fanout",
        "wrong_q1_perm",
        "per_axis_quantization",
    ],
)
def test_mean_hardsigmoid_muladd_rejects_unsafe_boundaries(case: str) -> None:
    model_ir = _build_chain()
    if case == "wrong_q0_perm":
        model_ir.tensors["q0_pre_perm"].data = np.asarray(
            [0, 2, 3, 1], dtype=np.int32
        )
    elif case == "public_q0_bridge":
        model_ir.outputs.append("q0_nchw")
    elif case == "q0_fanout":
        model_ir.tensors["q0_fanout"] = _tensor(
            "q0_fanout", [1, 3, 4, 4], dtype="INT8"
        )
        model_ir.operators.append(
            OperatorIR("RELU", ["q0_nchw"], ["q0_fanout"])
        )
    elif case == "mean_not_keepdims":
        model_ir.operators[2].options["keepDims"] = False
    elif case == "wrong_post_perm":
        model_ir.tensors["mean_post_perm"].data = np.asarray(
            [0, 3, 1, 2], dtype=np.int32
        )
    elif case == "sig_q_fanout":
        model_ir.tensors["sig_fanout"] = _tensor(
            "sig_fanout", [1, 3, 4, 4], dtype="INT8"
        )
        model_ir.operators.append(OperatorIR("RELU", ["sig_q"], ["sig_fanout"]))
    elif case == "wrong_q1_perm":
        model_ir.tensors["q1_pre_perm"].data = np.asarray(
            [0, 2, 3, 1], dtype=np.int32
        )
    elif case == "per_axis_quantization":
        model_ir.tensors["q0_raw"].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3],
            zero_point=[0, 0, 0],
            quantized_dimension=3,
        )
    before = deepcopy(model_ir)

    stats = _optimize_transpose_mean_hardsigmoid_muladd_chains(model_ir)

    assert stats == {"optimized_transpose_mean_hardsigmoid_muladd_chains": 0}
    assert _fingerprint(model_ir) == _fingerprint(before)


@pytest.mark.xfail(
    strict=True,
    reason="mean axes are validated after the first input/metadata mutation",
)
def test_mean_hardsigmoid_muladd_invalid_axes_rejection_is_atomic() -> None:
    model_ir = _build_chain()
    model_ir.tensors["mean_axes"].data = np.asarray([4], dtype=np.int32)
    model_ir.tensors["mean_axes"].shape = [1]
    model_ir.tensors["mean_axes"].shape_signature = [1]
    before = deepcopy(model_ir)

    stats = _optimize_transpose_mean_hardsigmoid_muladd_chains(model_ir)

    assert stats == {"optimized_transpose_mean_hardsigmoid_muladd_chains": 0}
    assert _fingerprint(model_ir) == _fingerprint(before)


@pytest.mark.xfail(
    strict=True,
    reason="the public residual output is changed from NCHW to NHWC",
)
def test_mean_hardsigmoid_muladd_preserves_public_residual_output() -> None:
    model_ir = _build_chain()
    model_ir.outputs.append("add0_out")
    before = deepcopy(model_ir)

    stats = _optimize_transpose_mean_hardsigmoid_muladd_chains(model_ir)

    assert stats == {"optimized_transpose_mean_hardsigmoid_muladd_chains": 0}
    assert _fingerprint(model_ir) == _fingerprint(before)
