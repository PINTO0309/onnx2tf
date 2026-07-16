from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_shape_extract_nhwc_to_nchw_chains,
)
from onnx2tf.tflite_builder.passes.shape_extract_layout import (
    optimize_transpose_shape_extract_nhwc_to_nchw_chains,
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


def _int_tensor(name: str, values: list[int]) -> TensorIR:
    return _tensor(
        name,
        [len(values)],
        dtype="INT64",
        data=np.asarray(values, dtype=np.int64),
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
            }
            for name, tensor in model_ir.tensors.items()
        },
        "operators": [
            {
                "op_type": operator.op_type,
                "inputs": list(operator.inputs),
                "outputs": list(operator.outputs),
                "options": copy.deepcopy(operator.options),
                "version": operator.version,
            }
            for operator in model_ir.operators
        ],
    }


def _base_model_ir() -> ModelIR:
    model_ir = ModelIR("shape_extract_nhwc")
    model_ir.inputs = ["x"]
    model_ir.tensors["x"] = _tensor("x", [1, 5, 7, 3])
    model_ir.tensors["perm"] = _tensor(
        "perm",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["x_nchw"] = _tensor("x_nchw", [1, 3, 5, 7])
    model_ir.tensors["shape_nchw"] = _tensor(
        "shape_nchw", [4], dtype="INT64"
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["x_nchw"]),
        OperatorIR("SHAPE", ["x_nchw"], ["shape_nchw"]),
    ]
    return model_ir


@pytest.mark.parametrize("shared_indices", [False, True])
def test_shape_extract_layout_remaps_gather_indices(
    shared_indices: bool,
) -> None:
    model_ir = _base_model_ir()
    model_ir.outputs = ["selected"]
    model_ir.tensors["indices"] = _int_tensor("indices", [1, -1])
    model_ir.tensors["selected"] = _tensor(
        "selected", [2], dtype="INT64"
    )
    model_ir.operators.append(
        OperatorIR(
            "GATHER",
            ["shape_nchw", "indices"],
            ["selected"],
            options={"axis": 0, "batchDims": 0},
        )
    )
    if shared_indices:
        model_ir.inputs.append("other_shape")
        model_ir.tensors["other_shape"] = _tensor(
            "other_shape", [4], dtype="INT64"
        )
        model_ir.tensors["other_selected"] = _tensor(
            "other_selected", [2], dtype="INT64"
        )
        model_ir.operators.append(
            OperatorIR(
                "GATHER",
                ["other_shape", "indices"],
                ["other_selected"],
                options={"axis": 0, "batchDims": 0},
            )
        )

    stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)

    assert stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 1,
    }
    shape_op = next(op for op in model_ir.operators if op.op_type == "SHAPE")
    assert list(shape_op.inputs) == ["x"]
    target_gather = next(
        op for op in model_ir.operators if op.outputs == ["selected"]
    )
    target_indices = model_ir.tensors[target_gather.inputs[1]].data
    np.testing.assert_array_equal(
        target_indices,
        np.asarray([3, 2], dtype=np.int64),
    )
    if shared_indices:
        np.testing.assert_array_equal(
            model_ir.tensors["indices"].data,
            np.asarray([1, -1], dtype=np.int64),
        )
        assert target_gather.inputs[1] != "indices"
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    assert _optimize_transpose_shape_extract_nhwc_to_nchw_chains(
        model_ir
    ) == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 0,
    }


def _slice_model_ir(*, begin: int, size: int) -> ModelIR:
    model_ir = _base_model_ir()
    model_ir.outputs = ["selected"]
    model_ir.tensors["begin"] = _int_tensor("begin", [begin])
    model_ir.tensors["size"] = _int_tensor("size", [size])
    model_ir.tensors["selected"] = _tensor(
        "selected", [2], dtype="INT64"
    )
    model_ir.operators.append(
        OperatorIR(
            "SLICE",
            ["shape_nchw", "begin", "size"],
            ["selected"],
        )
    )
    return model_ir


def test_shape_extract_layout_remaps_contiguous_slice() -> None:
    model_ir = _slice_model_ir(begin=2, size=2)

    stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)

    assert stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 1,
    }
    slice_op = next(op for op in model_ir.operators if op.op_type == "SLICE")
    assert list(slice_op.inputs) == ["shape_nchw", "begin", "size"]
    np.testing.assert_array_equal(
        model_ir.tensors["begin"].data,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["size"].data,
        np.asarray([2], dtype=np.int64),
    )


def test_shape_extract_layout_converts_noncontiguous_slice_to_gather() -> None:
    model_ir = _slice_model_ir(begin=1, size=2)

    stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)

    assert stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 1,
    }
    gather_op = next(op for op in model_ir.operators if op.outputs == ["selected"])
    assert gather_op.op_type == "GATHER"
    assert gather_op.version == 1
    assert gather_op.options == {"axis": 0, "batchDims": 0}
    assert list(gather_op.inputs[:1]) == ["shape_nchw"]
    np.testing.assert_array_equal(
        model_ir.tensors[gather_op.inputs[1]].data,
        np.asarray([3, 1], dtype=np.int64),
    )
    assert model_ir.tensors["selected"].shape == [2]
    assert model_ir.tensors["selected"].shape_signature == [2]


@pytest.mark.parametrize(
    "guard",
    [
        "public_transpose",
        "public_shape",
        "transpose_fanout",
        "unsupported_shape_user",
        "gather_axis",
        "invalid_gather_index",
        "nonconstant_indices",
        "empty_slice",
    ],
)
def test_shape_extract_layout_rejects_unsafe_or_unsupported_graphs(
    guard: str,
) -> None:
    model_ir = _base_model_ir()
    model_ir.outputs = ["selected"]
    model_ir.tensors["indices"] = _int_tensor("indices", [1])
    model_ir.tensors["selected"] = _tensor(
        "selected", [1], dtype="INT64"
    )
    model_ir.operators.append(
        OperatorIR(
            "GATHER",
            ["shape_nchw", "indices"],
            ["selected"],
            options={"axis": 0, "batchDims": 0},
        )
    )
    if guard == "public_transpose":
        model_ir.outputs.append("x_nchw")
    elif guard == "public_shape":
        model_ir.outputs.append("shape_nchw")
    elif guard == "transpose_fanout":
        model_ir.tensors["tap"] = _tensor("tap", [1, 3, 5, 7])
        model_ir.operators.append(OperatorIR("RELU", ["x_nchw"], ["tap"]))
    elif guard == "unsupported_shape_user":
        model_ir.tensors["shape_tap"] = _tensor(
            "shape_tap", [4], dtype="INT64"
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["shape_nchw"], ["shape_tap"])
        )
    elif guard == "gather_axis":
        model_ir.operators[-1].options["axis"] = 1
    elif guard == "invalid_gather_index":
        model_ir.tensors["indices"].data = np.asarray([4], dtype=np.int64)
    elif guard == "nonconstant_indices":
        model_ir.tensors["indices"].data = None
    elif guard == "empty_slice":
        model_ir.operators[-1] = OperatorIR(
            "SLICE",
            ["shape_nchw", "indices", "indices"],
            ["selected"],
        )
        model_ir.tensors["indices"].data = np.asarray([4], dtype=np.int64)
    else:
        raise AssertionError(f"unsupported guard: {guard}")
    before = _snapshot(model_ir)

    stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)

    assert stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 0,
    }
    assert _snapshot(model_ir) == before


def test_shape_extract_owner_matches_lowerer_compatibility_wrapper() -> None:
    direct_model_ir = _slice_model_ir(begin=1, size=2)
    wrapper_model_ir = copy.deepcopy(direct_model_ir)

    direct_stats = optimize_transpose_shape_extract_nhwc_to_nchw_chains(
        direct_model_ir
    )
    wrapper_stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(
        wrapper_model_ir
    )

    assert direct_stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 1,
    }
    assert wrapper_stats == direct_stats
    assert _snapshot(wrapper_model_ir) == _snapshot(direct_model_ir)
