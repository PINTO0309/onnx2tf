from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_slice_prepost_nhwc_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.slice_prepost_layout import (
    optimize_transpose_slice_prepost_nhwc_passthrough_chains,
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
        is_variable=data is None,
    )


def _snapshot(model_ir: ModelIR) -> dict[str, object]:
    return {
        "inputs": list(model_ir.inputs),
        "outputs": list(model_ir.outputs),
        "tensors": {
            name: {
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "shape_signature": (
                    list(tensor.shape_signature)
                    if tensor.shape_signature is not None
                    else None
                ),
                "data": (
                    tensor.data.tolist()
                    if isinstance(tensor.data, np.ndarray)
                    else tensor.data
                ),
                "quantization": copy.deepcopy(tensor.quantization),
                "logical_layout": tensor.logical_layout,
                "physical_layout": tensor.physical_layout,
            }
            for name, tensor in model_ir.tensors.items()
        },
        "operators": [
            {
                "op_type": operator.op_type,
                "inputs": list(operator.inputs),
                "outputs": list(operator.outputs),
                "options": copy.deepcopy(operator.options),
            }
            for operator in model_ir.operators
        ],
    }


def _make_slice_prepost_model_ir(*, already_nhwc: bool = False) -> ModelIR:
    model_ir = ModelIR("slice_prepost_nhwc")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = _tensor("x", [1, 5, 7, 3])
    model_ir.tensors["pre_perm"] = _tensor(
        "pre_perm",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["post_perm"] = _tensor(
        "post_perm",
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["x_nchw"] = _tensor("x_nchw", [1, 3, 5, 7])
    begin = [0, 1, 2, 1] if already_nhwc else [0, 1, 1, 2]
    size = [1, 3, 4, 2] if already_nhwc else [1, 2, 3, 4]
    model_ir.tensors["begin"] = _tensor(
        "begin",
        [4],
        dtype="INT32",
        data=np.asarray(begin, dtype=np.int32),
    )
    model_ir.tensors["size"] = _tensor(
        "size",
        [4],
        dtype="INT32",
        data=np.asarray(size, dtype=np.int32),
    )
    model_ir.tensors["slice_nchw"] = _tensor(
        "slice_nchw", [1, 2, 3, 4]
    )
    model_ir.tensors["y"] = _tensor("y", [1, 3, 4, 2])
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "pre_perm"], ["x_nchw"]),
        OperatorIR(
            "SLICE",
            ["x_nchw", "begin", "size"],
            ["slice_nchw"],
        ),
        OperatorIR(
            "TRANSPOSE",
            ["slice_nchw", "post_perm"],
            ["y"],
        ),
    ]
    return model_ir


@pytest.mark.parametrize("already_nhwc", [False, True])
def test_slice_prepost_layout_rewrites_and_is_idempotent(
    already_nhwc: bool,
) -> None:
    model_ir = _make_slice_prepost_model_ir(already_nhwc=already_nhwc)

    stats = _optimize_transpose_slice_prepost_nhwc_passthrough_chains(model_ir)

    assert stats == {
        "optimized_transpose_slice_prepost_nhwc_passthrough_chains": 1,
    }
    assert len(model_ir.operators) == 1
    slice_op = model_ir.operators[0]
    assert slice_op.op_type == "SLICE"
    assert list(slice_op.inputs) == ["x", "begin", "size"]
    assert list(slice_op.outputs) == ["y"]
    np.testing.assert_array_equal(
        model_ir.tensors["begin"].data,
        np.asarray([0, 1, 2, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["size"].data,
        np.asarray([1, 3, 4, 2], dtype=np.int32),
    )
    assert _optimize_transpose_slice_prepost_nhwc_passthrough_chains(
        model_ir
    ) == {
        "optimized_transpose_slice_prepost_nhwc_passthrough_chains": 0,
    }


@pytest.mark.parametrize(
    "guard",
    [
        "public_pre",
        "public_slice",
        "shared_begin",
        "pre_fanout",
        "shape_mismatch",
        "wrong_pre_perm",
    ],
)
def test_slice_prepost_layout_rejects_unsafe_or_inconsistent_graphs(
    guard: str,
) -> None:
    model_ir = _make_slice_prepost_model_ir()
    if guard == "public_pre":
        model_ir.outputs.append("x_nchw")
    elif guard == "public_slice":
        model_ir.outputs.append("slice_nchw")
    elif guard == "shared_begin":
        model_ir.tensors["begin_tap"] = _tensor(
            "begin_tap", [4], dtype="INT32"
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["begin"], ["begin_tap"])
        )
    elif guard == "pre_fanout":
        model_ir.tensors["pre_tap"] = _tensor("pre_tap", [1, 3, 5, 7])
        model_ir.operators.append(
            OperatorIR("RELU", ["x_nchw"], ["pre_tap"])
        )
    elif guard == "shape_mismatch":
        model_ir.tensors["y"].shape = [1, 4, 4, 2]
        model_ir.tensors["y"].shape_signature = [1, 4, 4, 2]
    elif guard == "wrong_pre_perm":
        model_ir.tensors["pre_perm"].data = np.asarray(
            [0, 2, 3, 1], dtype=np.int32
        )
    else:
        raise AssertionError(f"unsupported guard: {guard}")
    before = _snapshot(model_ir)

    stats = _optimize_transpose_slice_prepost_nhwc_passthrough_chains(model_ir)

    assert stats == {
        "optimized_transpose_slice_prepost_nhwc_passthrough_chains": 0,
    }
    assert _snapshot(model_ir) == before


def test_slice_prepost_owner_matches_lowerer_compatibility_wrapper() -> None:
    direct_model_ir = _make_slice_prepost_model_ir()
    wrapper_model_ir = copy.deepcopy(direct_model_ir)

    direct_stats = optimize_transpose_slice_prepost_nhwc_passthrough_chains(
        direct_model_ir
    )
    wrapper_stats = (
        _optimize_transpose_slice_prepost_nhwc_passthrough_chains(
            wrapper_model_ir
        )
    )

    assert direct_stats == {
        "optimized_transpose_slice_prepost_nhwc_passthrough_chains": 1,
    }
    assert wrapper_stats == direct_stats
    assert _snapshot(wrapper_model_ir) == _snapshot(direct_model_ir)
