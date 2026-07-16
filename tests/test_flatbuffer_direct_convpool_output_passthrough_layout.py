from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_convpool_output_transpose_nhwc_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.convpool_output_passthrough_compat import (
    optimize_convpool_output_transpose_nhwc_passthrough_chains,
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
    model_ir = ModelIR("convpool_output_passthrough")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 6, 6, 8]),
        "conv_filter": _tensor(
            "conv_filter",
            [1, 1, 8, 8],
            data=np.ones((1, 1, 8, 8), dtype=np.float32),
        ),
        "conv_bias": _tensor(
            "conv_bias",
            [8],
            data=np.zeros((8,), dtype=np.float32),
        ),
        "conv_out_nhwc": _tensor("conv_out_nhwc", [1, 6, 6, 8]),
        "pre_perm": _tensor(
            "pre_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "conv_out_nchw": _tensor("conv_out_nchw", [1, 8, 6, 6]),
        "scale": _tensor(
            "scale",
            [1, 8, 1, 1],
            data=np.ones((1, 8, 1, 1), dtype=np.float32),
        ),
        "mul_out": _tensor("mul_out", [1, 8, 6, 6]),
        "add_bias": _tensor(
            "add_bias",
            [1, 8, 1, 1],
            data=np.zeros((1, 8, 1, 1), dtype=np.float32),
        ),
        "tail": _tensor("tail", [1, 8, 6, 6]),
        "z": _tensor("z", [1, 8, 6, 6]),
    }
    model_ir.operators = [
        OperatorIR(
            "CONV_2D",
            ["x_nhwc", "conv_filter", "conv_bias"],
            ["conv_out_nhwc"],
        ),
        OperatorIR(
            "TRANSPOSE",
            ["conv_out_nhwc", "pre_perm"],
            ["conv_out_nchw"],
        ),
        OperatorIR("MUL", ["conv_out_nchw", "scale"], ["mul_out"]),
        OperatorIR("ADD", ["mul_out", "add_bias"], ["tail"]),
        OperatorIR("RELU", ["tail"], ["z"]),
    ]
    return model_ir


def _fingerprint(model_ir: ModelIR) -> bytes:
    return ModelIRPassState(model_ir).fingerprint()


def _run_owner_and_wrapper(
    model_ir: ModelIR,
) -> tuple[ModelIR, dict[str, int]]:
    owner_model_ir = deepcopy(model_ir)
    wrapper_model_ir = deepcopy(model_ir)
    owner_stats = optimize_convpool_output_transpose_nhwc_passthrough_chains(
        owner_model_ir
    )
    wrapper_stats = (
        _optimize_convpool_output_transpose_nhwc_passthrough_chains(
            wrapper_model_ir
        )
    )
    assert owner_stats == wrapper_stats
    assert _fingerprint(owner_model_ir) == _fingerprint(wrapper_model_ir)
    return wrapper_model_ir, wrapper_stats


def test_convpool_output_passthrough_rewrites_elementwise_region() -> None:
    model_ir, stats = _run_owner_and_wrapper(_build_chain())

    assert stats == {
        "optimized_convpool_output_transpose_nhwc_passthrough_chains": 1
    }
    assert [op.op_type for op in model_ir.operators] == [
        "CONV_2D",
        "MUL",
        "ADD",
        "RELU",
        "TRANSPOSE",
    ]
    mul_op = model_ir.operators[1]
    assert mul_op.inputs == ["conv_out_nhwc", "scale"]
    assert mul_op.outputs == ["mul_out"]
    assert model_ir.operators[2].outputs == ["tail__to_nhwc"]
    assert model_ir.operators[3].inputs == ["tail"]
    assert model_ir.operators[4].inputs[0] == "tail__to_nhwc"
    np.testing.assert_array_equal(
        model_ir.tensors[model_ir.operators[4].inputs[1]].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert "conv_out_nhwc" in model_ir.metadata[
        "assume_channel_last_layout_tensor_names"
    ]


@pytest.mark.parametrize(
    "case",
    [
        "wrong_perm",
        "public_pre_output",
        "non_convpool_producer",
        "no_elementwise_region",
        "pre_output_non_elementwise_fanout",
        "public_elementwise_output",
        "multi_output_elementwise",
    ],
)
def test_convpool_output_passthrough_rejects_unsafe_boundaries(case: str) -> None:
    model_ir = _build_chain()
    if case == "wrong_perm":
        model_ir.tensors["pre_perm"].data = np.asarray(
            [0, 2, 3, 1], dtype=np.int32
        )
    elif case == "public_pre_output":
        model_ir.outputs.append("conv_out_nchw")
    elif case == "non_convpool_producer":
        model_ir.operators[0].op_type = "FULLY_CONNECTED"
    elif case == "no_elementwise_region":
        model_ir.operators[2].op_type = "RELU"
    elif case == "pre_output_non_elementwise_fanout":
        model_ir.tensors["fanout"] = _tensor("fanout", [1, 8, 6, 6])
        model_ir.operators.append(
            OperatorIR("RELU", ["conv_out_nchw"], ["fanout"])
        )
    elif case == "public_elementwise_output":
        model_ir.outputs.append("tail")
    elif case == "multi_output_elementwise":
        model_ir.tensors["tail_2"] = _tensor("tail_2", [1, 8, 6, 6])
        model_ir.operators[2].outputs.append("tail_2")
    before = deepcopy(model_ir)

    model_ir, stats = _run_owner_and_wrapper(model_ir)

    assert stats == {
        "optimized_convpool_output_transpose_nhwc_passthrough_chains": 0
    }
    assert _fingerprint(model_ir) == _fingerprint(before)


def test_convpool_output_passthrough_adapts_external_runtime_input() -> None:
    model_ir = _build_chain()
    scale = model_ir.tensors["scale"]
    scale.data = None
    scale.shape = [1, 8, 6, 6]
    scale.shape_signature = [1, 8, 6, 6]
    model_ir.inputs.append("scale")

    model_ir, stats = _run_owner_and_wrapper(model_ir)

    assert stats == {
        "optimized_convpool_output_transpose_nhwc_passthrough_chains": 1
    }
    mul_op = next(op for op in model_ir.operators if op.op_type == "MUL")
    assert mul_op.inputs == ["conv_out_nhwc", "scale__to_nhwc"]
    assert model_ir.tensors["scale__to_nhwc"].shape == [1, 6, 6, 8]
    adapter = next(
        op
        for op in model_ir.operators
        if op.op_type == "TRANSPOSE" and op.outputs == ["scale__to_nhwc"]
    )
    np.testing.assert_array_equal(
        model_ir.tensors[adapter.inputs[1]].data,
        np.asarray([0, 2, 3, 1], dtype=np.int32),
    )


def test_convpool_output_passthrough_absorbs_keepdims_mean_boundary() -> None:
    model_ir = _build_chain()
    model_ir.tensors["axes"] = _tensor(
        "axes",
        [2],
        dtype="INT32",
        data=np.asarray([2, 3], dtype=np.int32),
    )
    model_ir.tensors["z"].shape = [1, 8, 1, 1]
    model_ir.tensors["z"].shape_signature = [1, 8, 1, 1]
    model_ir.operators[-1] = OperatorIR(
        "MEAN",
        ["tail", "axes"],
        ["z"],
        options={"keepDims": True, "axes": [2, 3]},
    )

    model_ir, stats = _run_owner_and_wrapper(model_ir)

    assert stats == {
        "optimized_convpool_output_transpose_nhwc_passthrough_chains": 1
    }
    assert [op.op_type for op in model_ir.operators] == [
        "CONV_2D",
        "MUL",
        "ADD",
        "MEAN",
    ]
    mean_op = model_ir.operators[-1]
    assert mean_op.inputs == ["tail__to_nhwc", "axes"]
    assert mean_op.options["axes"] == [1, 2]
    assert mean_op.options["__convpool_output_nhwc_axes_remapped__"] is True
    np.testing.assert_array_equal(
        model_ir.tensors["axes"].data,
        np.asarray([1, 2], dtype=np.int32),
    )
    assert model_ir.tensors["z"].shape == [1, 1, 1, 8]
    assert model_ir.metadata[
        "convpool_output_nhwc_remapped_axes_tensor_names"
    ] == ["axes"]


def test_convpool_output_passthrough_invalid_external_input_is_atomic() -> None:
    model_ir = _build_chain()
    add_bias = model_ir.tensors["add_bias"]
    add_bias.data = None
    add_bias.shape = [1, 8, 6, 6]
    add_bias.shape_signature = [1, 8, 6, 6]
    scale = model_ir.tensors["scale"]
    scale.data = None
    scale.shape = [8, 1, 1]
    scale.shape_signature = [8, 1, 1]
    model_ir.inputs.extend(["add_bias", "scale"])
    before = deepcopy(model_ir)

    model_ir, stats = _run_owner_and_wrapper(model_ir)

    assert stats == {
        "optimized_convpool_output_transpose_nhwc_passthrough_chains": 0
    }
    assert _fingerprint(model_ir) == _fingerprint(before)
