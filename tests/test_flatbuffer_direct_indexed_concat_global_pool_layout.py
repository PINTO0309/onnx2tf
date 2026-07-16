from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.concat_global_pool_layout as global_pool_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.concat_global_pool_layout import (
    _repair_nchw_concat_global_pool_conv_axes,
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


def _add_branch(
    model_ir: ModelIR,
    prefix: str,
    *,
    negative_axes: bool = False,
    shape_dtype=np.int32,
) -> dict[str, str]:
    names = {
        key: f"{prefix}_{key}"
        for key in (
            "left",
            "right",
            "concat",
            "axes",
            "pool",
            "shape",
            "reshape",
            "filter",
            "bias",
            "output",
        )
    }
    model_ir.inputs.extend([names["left"], names["right"]])
    model_ir.outputs.append(names["output"])
    model_ir.tensors.update(
        {
            names["left"]: _tensor(names["left"], [1, 2, 4, 5]),
            names["right"]: _tensor(names["right"], [1, 3, 4, 5]),
            names["concat"]: _tensor(names["concat"], [1, 2, 4, 10]),
            names["axes"]: _tensor(
                names["axes"],
                [2],
                dtype="INT32",
                data=np.asarray([-2, -1] if negative_axes else [2, 3], dtype=np.int32),
            ),
            names["pool"]: _tensor(names["pool"], [1, 2, 1, 1]),
            names["shape"]: _tensor(
                names["shape"],
                [4],
                dtype="INT64" if shape_dtype == np.int64 else "INT32",
                data=np.asarray([1, 1, 1, 2], dtype=shape_dtype),
            ),
            names["reshape"]: _tensor(names["reshape"], [1, 1, 1, 2]),
            names["filter"]: _tensor(
                names["filter"],
                [4, 1, 1, 5],
                data=np.ones([4, 1, 1, 5], dtype=np.float32),
            ),
            names["bias"]: _tensor(
                names["bias"],
                [4],
                data=np.zeros([4], dtype=np.float32),
            ),
            names["output"]: _tensor(names["output"], [1, 1, 1, 4]),
        }
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                "CONCATENATION",
                [names["left"], names["right"]],
                [names["concat"]],
                {"axis": 3, "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                "MEAN",
                [names["concat"], names["axes"]],
                [names["pool"]],
                {"keepDims": True},
            ),
            OperatorIR(
                "RESHAPE",
                [names["pool"], names["shape"]],
                [names["reshape"]],
                {"newShape": [1, 1, 1, 2]},
            ),
            OperatorIR(
                "CONV_2D",
                [names["reshape"], names["filter"], names["bias"]],
                [names["output"]],
                {"padding": "SAME", "strideH": 1, "strideW": 1},
                onnx_node_name=f"{prefix}_conv",
                onnx_op_type="Conv",
            ),
        ]
    )
    return names


def _assert_index_current(model_ir: ModelIR, index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert index.producers == fresh.producers
    assert index.consumers == fresh.consumers
    assert index.duplicate_producers == fresh.duplicate_producers
    assert index._operator_indices_by_id == fresh._operator_indices_by_id
    assert index._operator_indices_by_type == fresh._operator_indices_by_type


def test_concat_global_pool_repairs_multiple_chains_with_one_index(monkeypatch) -> None:
    model_ir = ModelIR("indexed_concat_global_pool")
    branches = [
        _add_branch(model_ir, "first"),
        _add_branch(model_ir, "second", negative_axes=True, shape_dtype=np.int64),
    ]
    original_conv_options = [
        copy.deepcopy(model_ir.operators[index].options) for index in (3, 7)
    ]
    refreshes = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refreshes
        refreshes += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _repair_nchw_concat_global_pool_conv_axes(model_ir)

    assert stats == {"repaired_nchw_concat_global_pool_conv_axes": 2}
    assert refreshes == 1
    for branch_index, names in enumerate(branches):
        offset = branch_index * 4
        assert model_ir.operators[offset].options == {
            "axis": 1,
            "fusedActivationFunction": "NONE",
        }
        assert model_ir.operators[offset + 2].options["newShape"] == [1, 1, 1, 5]
        assert (
            model_ir.operators[offset + 3].options
            == original_conv_options[branch_index]
        )
        assert model_ir.tensors[names["concat"]].shape == [1, 5, 4, 5]
        assert model_ir.tensors[names["pool"]].shape == [1, 5, 1, 1]
        assert model_ir.tensors[names["reshape"]].shape == [1, 1, 1, 5]
        assert np.asarray(model_ir.tensors[names["shape"]].data).tolist() == [
            1,
            1,
            1,
            5,
        ]
    assert np.asarray(model_ir.tensors[branches[1]["shape"]].data).dtype == np.int64


def test_concat_global_pool_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_concat_global_pool")
    _add_branch(model_ir, "branch")
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _repair_nchw_concat_global_pool_conv_axes(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"repaired_nchw_concat_global_pool_conv_axes": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_concat_global_pool_accepts_indexed_self_gating_mul_fanout() -> None:
    model_ir = ModelIR("gated_concat_global_pool")
    names = _add_branch(model_ir, "branch")
    model_ir.tensors[names["filter"]].shape = [5, 1, 1, 5]
    model_ir.tensors[names["filter"]].shape_signature = [5, 1, 1, 5]
    model_ir.tensors[names["filter"]].data = np.ones(
        [5, 1, 1, 5],
        dtype=np.float32,
    )
    model_ir.tensors[names["bias"]].shape = [5]
    model_ir.tensors[names["bias"]].shape_signature = [5]
    model_ir.tensors[names["bias"]].data = np.zeros([5], dtype=np.float32)
    model_ir.tensors[names["output"]].shape = [1, 1, 1, 5]
    model_ir.tensors[names["output"]].shape_signature = [1, 1, 1, 5]
    model_ir.tensors["gated"] = _tensor("gated", [1, 5, 4, 5])
    model_ir.operators.append(
        OperatorIR(
            "MUL",
            [names["concat"], names["output"]],
            ["gated"],
            {"fusedActivationFunction": "NONE"},
        )
    )
    model_ir.outputs.append("gated")

    stats = _repair_nchw_concat_global_pool_conv_axes(model_ir)

    assert stats == {"repaired_nchw_concat_global_pool_conv_axes": 1}
    assert model_ir.operators[0].options["axis"] == 1
    assert model_ir.tensors[names["concat"]].shape == [1, 5, 4, 5]
    assert model_ir.tensors[names["reshape"]].shape == [1, 1, 1, 5]


@pytest.mark.parametrize(
    "case",
    [
        "conv_arity",
        "duplicate_reshape_producer",
        "reshape_type",
        "reshape_arity",
        "reshape_fanout",
        "reshape_public",
        "pool_type",
        "pool_keepdims",
        "pool_axes",
        "pool_axes_produced",
        "concat_type",
        "concat_arity",
        "concat_axis",
        "concat_axis_invalid",
        "concat_fanout",
        "concat_public",
        "input_rank",
        "input_spatial_mismatch",
        "input_nonpositive",
        "filter_channels",
        "filter_missing_data",
        "filter_shape_mismatch",
        "filter_produced",
        "shape_missing_data",
        "shape_float",
        "shape_size",
        "shape_shared",
        "shape_public",
        "shape_produced",
    ],
)
def test_concat_global_pool_rejects_unsafe_contracts_transactionally(case: str) -> None:
    model_ir = ModelIR("rejected_concat_global_pool")
    names = _add_branch(model_ir, "branch")
    concat, pool, reshape, conv = model_ir.operators

    if case == "conv_arity":
        conv.inputs = [names["reshape"]]
    elif case == "duplicate_reshape_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["pool"]], [names["reshape"]])
        )
    elif case == "reshape_type":
        reshape.op_type = "SQUEEZE"
    elif case == "reshape_arity":
        reshape.inputs = [names["pool"]]
    elif case in {"reshape_fanout", "concat_fanout", "shape_shared"}:
        source = {
            "reshape_fanout": names["reshape"],
            "concat_fanout": names["concat"],
            "shape_shared": names["shape"],
        }[case]
        model_ir.tensors["side"] = _tensor("side", [1])
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    elif case in {"reshape_public", "concat_public", "shape_public"}:
        name = {
            "reshape_public": names["reshape"],
            "concat_public": names["concat"],
            "shape_public": names["shape"],
        }[case]
        model_ir.outputs.append(name)
    elif case == "pool_type":
        pool.op_type = "SUM"
    elif case == "pool_keepdims":
        pool.options["keepDims"] = False
    elif case == "pool_axes":
        model_ir.tensors[names["axes"]].data = np.asarray([1, 2], dtype=np.int32)
    elif case == "pool_axes_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["shape"]], [names["axes"]])
        )
    elif case == "concat_type":
        concat.op_type = "PACK"
    elif case == "concat_arity":
        concat.inputs = [names["left"]]
    elif case == "concat_axis":
        concat.options["axis"] = 1
    elif case == "concat_axis_invalid":
        concat.options["axis"] = "channel"
    elif case == "input_rank":
        model_ir.tensors[names["right"]].shape = [1, 3, 20]
    elif case == "input_spatial_mismatch":
        model_ir.tensors[names["right"]].shape[2] = 6
    elif case == "input_nonpositive":
        model_ir.tensors[names["right"]].shape[1] = -1
    elif case == "filter_channels":
        model_ir.tensors[names["filter"]].shape[3] = 4
    elif case == "filter_missing_data":
        model_ir.tensors[names["filter"]].data = None
    elif case == "filter_shape_mismatch":
        model_ir.tensors[names["filter"]].data = np.ones([4, 1, 1, 4], dtype=np.float32)
    elif case == "filter_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["bias"]], [names["filter"]])
        )
    elif case == "shape_missing_data":
        model_ir.tensors[names["shape"]].data = None
    elif case == "shape_float":
        model_ir.tensors[names["shape"]].data = np.asarray(
            [1, 1, 1, 2], dtype=np.float32
        )
    elif case == "shape_size":
        model_ir.tensors[names["shape"]].data = np.asarray([1, 1, 2], dtype=np.int32)
    elif case == "shape_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axes"]], [names["shape"]])
        )

    before = repr(model_ir)
    stats = _repair_nchw_concat_global_pool_conv_axes(model_ir)

    assert stats == {"repaired_nchw_concat_global_pool_conv_axes": 0}
    assert repr(model_ir) == before


def test_concat_global_pool_skips_index_without_complete_family(monkeypatch) -> None:
    model_ir = ModelIR("no_concat_global_pool")
    model_ir.tensors["x"] = _tensor("x", [1])
    model_ir.tensors["y"] = _tensor("y", [1])
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(global_pool_module, "ModelIRGraphIndex", unexpected_index)

    assert _repair_nchw_concat_global_pool_conv_axes(model_ir) == {
        "repaired_nchw_concat_global_pool_conv_axes": 0
    }
