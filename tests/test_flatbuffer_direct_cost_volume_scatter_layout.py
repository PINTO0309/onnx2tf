from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_cost_volume_scatter_ndhwc_chains,
)
from onnx2tf.tflite_builder.passes.cost_volume_scatter_layout import (
    run_cost_volume_scatter_layout_cleanup,
)


def _tensor(
    name: str,
    dtype: str,
    shape: list[int],
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=None if data is None else np.asarray(data),
        is_variable=False,
    )


def _model(*, boundary: str | None = None) -> ModelIR:
    model_ir = ModelIR("cost_volume_scatter_ndhwc")
    model_ir.inputs = ["desc0_nhwc", "desc1_nhwc"]
    model_ir.outputs = ["conv_out"]
    model_ir.tensors = {
        "desc0_nhwc": _tensor("desc0_nhwc", "FLOAT32", [1, 3, 6, 4]),
        "desc1_nhwc": _tensor("desc1_nhwc", "FLOAT32", [1, 3, 6, 4]),
        "desc0_nchw": _tensor("desc0_nchw", "FLOAT32", [1, 4, 3, 6]),
        "desc1_nchw": _tensor("desc1_nchw", "FLOAT32", [1, 4, 3, 6]),
        "slice_begin": _tensor(
            "slice_begin",
            "INT32",
            [4],
            np.asarray([0, 0, 0, 1], dtype=np.int32),
        ),
        "slice_size": _tensor(
            "slice_size",
            "INT32",
            [4],
            np.asarray([1, 4, 3, 5], dtype=np.int32),
        ),
        "slice0": _tensor("slice0", "FLOAT32", [1, 4, 3, 5]),
        "slice1": _tensor("slice1", "FLOAT32", [1, 4, 3, 5]),
        "mul_out": _tensor("mul_out", "FLOAT32", [1, 4, 3, 5]),
        "mean_axes": _tensor(
            "mean_axes",
            "INT32",
            [1],
            np.asarray([1], dtype=np.int32),
        ),
        "mean_out": _tensor("mean_out", "FLOAT32", [1, 1, 3, 5]),
        "reshape_shape": _tensor(
            "reshape_shape",
            "INT32",
            [5],
            np.asarray([1, 1, 1, 3, 5], dtype=np.int32),
        ),
        "reshape_out": _tensor("reshape_out", "FLOAT32", [1, 1, 1, 3, 5]),
        "scatter_shape": _tensor(
            "scatter_shape",
            "INT32",
            [4] if boundary == "scatter_shape_rank" else [5],
            np.asarray(
                [1, 1, 3, 4]
                if boundary == "scatter_shape_rank"
                else [1, 1, 3, 4, 6],
                dtype=np.int32,
            ),
        ),
        "indices_i32": _tensor("indices_i32", "INT32", [1, 1, 1, 3, 5, 5]),
        "scatter_ncdhw": _tensor(
            "scatter_ncdhw",
            "FLOAT32",
            [1, 1, 3, 4, 6],
        ),
        "scatter_ndhwc": _tensor(
            "scatter_ndhwc",
            "FLOAT32",
            [1, 3, 4, 6, 1],
        ),
        "conv_w": _tensor("conv_w", "FLOAT32", [1, 1, 1, 1, 2]),
        "conv_b": _tensor("conv_b", "FLOAT32", [2]),
        "conv_out": _tensor("conv_out", "FLOAT32", [1, 3, 4, 6, 2]),
        "perm_nhwc_to_nchw": _tensor(
            "perm_nhwc_to_nchw",
            "INT32",
            [4],
            np.asarray(
                [0, 1, 2, 3] if boundary == "permutation" else [0, 3, 1, 2],
                dtype=np.int32,
            ),
        ),
        "perm_ncdhw_to_ndhwc": _tensor(
            "perm_ncdhw_to_ndhwc",
            "INT32",
            [5],
            np.asarray([0, 2, 3, 4, 1], dtype=np.int32),
        ),
    }
    coordinate_width = 4 if boundary == "indices_rank" else 5
    index_data = np.zeros(
        (1, 1, 1, 3, 5, coordinate_width),
        dtype=np.float32,
    )
    if coordinate_width == 5:
        for height in range(3):
            for width in range(5):
                index_data[0, 0, 0, height, width] = np.asarray(
                    [0, 0, 0, height, width + 1],
                    dtype=np.float32,
                )
    if boundary == "indices_bounds":
        index_data[0, 0, 0, 0, 0, 4] = 99
    model_ir.tensors["indices_f32"] = _tensor(
        "indices_f32",
        "FLOAT32",
        list(index_data.shape),
        index_data,
    )
    model_ir.tensors["indices_i32"].shape = list(index_data.shape)
    model_ir.tensors["indices_i32"].shape_signature = list(index_data.shape)

    model_ir.operators = [
        OperatorIR(
            "TRANSPOSE",
            ["desc0_nhwc", "perm_nhwc_to_nchw"],
            ["desc0_nchw"],
        ),
        OperatorIR(
            "TRANSPOSE",
            ["desc1_nhwc", "perm_nhwc_to_nchw"],
            ["desc1_nchw"],
        ),
        OperatorIR(
            "SLICE",
            ["desc0_nchw", "slice_begin", "slice_size"],
            ["slice0"],
        ),
        OperatorIR(
            "SLICE",
            ["desc1_nchw", "slice_begin", "slice_size"],
            ["slice1"],
        ),
        OperatorIR("MUL", ["slice0", "slice1"], ["mul_out"]),
        OperatorIR(
            "MEAN",
            ["mul_out", "mean_axes"],
            ["mean_out"],
            options={"keepDims": True},
        ),
        OperatorIR(
            "RESHAPE",
            ["mean_out", "reshape_shape"],
            ["reshape_out"],
            options={
                "newShape": [1, 1, 1, 3, 5],
                "onnxRawNewShape": [1, 1, 1, 3, 5],
            },
        ),
        OperatorIR(
            "CAST",
            ["indices_f32"],
            ["indices_i32"],
            options={"inDataType": "FLOAT32", "outDataType": "INT32"},
        ),
        OperatorIR(
            "SCATTER_ND",
            ["indices_i32", "reshape_out", "scatter_shape"],
            ["scatter_ncdhw"],
        ),
        OperatorIR(
            "TRANSPOSE",
            ["scatter_ncdhw", "perm_ncdhw_to_ndhwc"],
            ["scatter_ndhwc"],
        ),
        OperatorIR(
            "IDENTITY" if boundary == "downstream_contract" else "CONV_3D",
            ["scatter_ndhwc", "conv_w", "conv_b"],
            ["conv_out"],
            options={"padding": "SAME"},
        ),
    ]
    side_sources = {
        "leading_adapter_fanout": "desc0_nchw",
        "post_input_fanout": "scatter_ncdhw",
        "post_output_fanout": "scatter_ndhwc",
    }
    if boundary in side_sources:
        source = side_sources[boundary]
        model_ir.tensors["side"] = _tensor(
            "side",
            "FLOAT32",
            list(model_ir.tensors[source].shape),
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    if boundary == "public_intermediate":
        model_ir.outputs.append("scatter_ncdhw")
    return model_ir


def _snapshot(model_ir: ModelIR) -> ModelIR:
    return deepcopy(model_ir)


def _assert_model_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in actual.operators
    ] == [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in expected.operators
    ]
    assert actual.tensors.keys() == expected.tensors.keys()
    for name, tensor in actual.tensors.items():
        expected_tensor = expected.tensors[name]
        assert tensor.dtype == expected_tensor.dtype
        assert tensor.shape == expected_tensor.shape
        assert tensor.shape_signature == expected_tensor.shape_signature
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


def test_cost_volume_scatter_layout_characterization() -> None:
    model_ir = _model()

    stats = _optimize_transpose_cost_volume_scatter_ndhwc_chains(model_ir)

    assert stats["optimized_transpose_cost_volume_scatter_ndhwc_chains"] == 1
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    slice_ops = [op for op in model_ir.operators if op.op_type == "SLICE"]
    assert {op.inputs[0] for op in slice_ops} == {"desc0_nhwc", "desc1_nhwc"}
    for slice_op in slice_ops:
        np.testing.assert_array_equal(
            model_ir.tensors[slice_op.inputs[1]].data,
            np.asarray([0, 0, 1, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            model_ir.tensors[slice_op.inputs[2]].data,
            np.asarray([1, 3, 5, 4], dtype=np.int32),
        )
    mean_op = next(op for op in model_ir.operators if op.op_type == "MEAN")
    np.testing.assert_array_equal(
        model_ir.tensors[mean_op.inputs[1]].data,
        np.asarray([3], dtype=np.int32),
    )
    scatter_op = next(
        op for op in model_ir.operators if op.op_type == "SCATTER_ND"
    )
    np.testing.assert_array_equal(
        model_ir.tensors[scatter_op.inputs[2]].data,
        np.asarray([1, 3, 4, 6, 1], dtype=np.int32),
    )
    cast_op = next(op for op in model_ir.operators if op.op_type == "CAST")
    remapped_indices = np.asarray(model_ir.tensors[cast_op.inputs[0]].data)
    assert remapped_indices[0, 0, 0, 0, 0].tolist() == [0, 0, 0, 1, 0]
    conv_op = next(op for op in model_ir.operators if op.op_type == "CONV_3D")
    assert conv_op.inputs[0] == "scatter_ncdhw"
    assert model_ir.tensors["scatter_ncdhw"].shape == [1, 3, 4, 6, 1]
    assert model_ir.tensors["slice0"].shape == [1, 3, 5, 4]


@pytest.mark.parametrize(
    "boundary",
    [
        "leading_adapter_fanout",
        "post_input_fanout",
        "post_output_fanout",
        "public_intermediate",
        "permutation",
        "downstream_contract",
        "scatter_shape_rank",
        "indices_rank",
        "indices_bounds",
    ],
)
def test_cost_volume_scatter_layout_rejects_unsafe_boundary(
    boundary: str,
) -> None:
    model_ir = _model(boundary=boundary)
    original = _snapshot(model_ir)

    stats = _optimize_transpose_cost_volume_scatter_ndhwc_chains(model_ir)

    assert stats["optimized_transpose_cost_volume_scatter_ndhwc_chains"] == 0
    _assert_model_equal(model_ir, original)


def test_cost_volume_scatter_layout_runner_reuses_one_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = _model()
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_cost_volume_scatter_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    stats_key = "optimized_transpose_cost_volume_scatter_ndhwc_chains"
    assert stats[stats_key] == 1
    assert refresh_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.cost_volume_scatter_ndhwc"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "leading_adapter_fanout",
        "post_input_fanout",
        "post_output_fanout",
        "public_intermediate",
        "permutation",
        "downstream_contract",
        "scatter_shape_rank",
        "indices_rank",
        "indices_bounds",
    ],
)
def test_cost_volume_scatter_layout_runner_rejects_before_snapshot(
    boundary: str,
) -> None:
    model_ir = _model(boundary=boundary)
    original = _snapshot(model_ir)
    diagnostics: list[dict[str, object]] = []

    stats = run_cost_volume_scatter_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    stats_key = "optimized_transpose_cost_volume_scatter_ndhwc_chains"
    assert stats[stats_key] == 0
    assert len(diagnostics) == 1
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
    _assert_model_equal(model_ir, original)
