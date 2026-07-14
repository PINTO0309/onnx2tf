from __future__ import annotations

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.concat_transpose_conv_layout as concat_conv_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.concat_transpose_conv_layout import (
    _repair_nchw_concat_transpose_conv_axes,
)


def _tensor(name: str, shape: list[int], data=None) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _add_branch(
    model_ir: ModelIR,
    prefix: str,
    *,
    pre_relu: bool = False,
    post_prefix: bool = False,
    transpose_conv: bool = False,
) -> dict[str, str]:
    names = {
        key: f"{prefix}_{key}"
        for key in (
            "left",
            "right",
            "concat",
            "relu",
            "perm",
            "transposed",
            "padded",
            "cast",
            "zero",
            "centered",
            "filter",
            "bias",
            "shape",
            "output",
        )
    }
    model_ir.inputs.extend([names["left"], names["right"]])
    model_ir.outputs.append(names["output"])
    tensors = {
        names["left"]: _tensor(names["left"], [1, 2, 4, 5]),
        names["right"]: _tensor(names["right"], [1, 3, 4, 5]),
        names["concat"]: _tensor(names["concat"], [1, 2, 4, 10]),
        names["perm"]: _tensor(
            names["perm"], [4], np.asarray([0, 2, 3, 1], dtype=np.int32)
        ),
        names["transposed"]: _tensor(names["transposed"], [1, 4, 10, 2]),
        names["filter"]: _tensor(
            names["filter"], [4, 1, 1, 5], np.ones([4, 1, 1, 5], dtype=np.float32)
        ),
        names["bias"]: _tensor(names["bias"], [4], np.zeros([4], dtype=np.float32)),
        names["output"]: _tensor(names["output"], [1, 4, 10, 4]),
    }
    model_ir.tensors.update(tensors)
    model_ir.operators.append(
        OperatorIR(
            "CONCATENATION",
            [names["left"], names["right"]],
            [names["concat"]],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        )
    )
    transpose_input = names["concat"]
    if pre_relu:
        model_ir.tensors[names["relu"]] = _tensor(names["relu"], [1, 2, 4, 10])
        model_ir.operators.append(
            OperatorIR("RELU", [transpose_input], [names["relu"]])
        )
        transpose_input = names["relu"]
    model_ir.operators.append(
        OperatorIR("TRANSPOSE", [transpose_input, names["perm"]], [names["transposed"]])
    )
    conv_input = names["transposed"]
    if post_prefix:
        for key, op_type in (("padded", "PAD"), ("cast", "CAST"), ("centered", "SUB")):
            model_ir.tensors[names[key]] = _tensor(names[key], [1, 4, 10, 2])
            inputs = [conv_input]
            if op_type == "SUB":
                model_ir.tensors[names["zero"]] = _tensor(
                    names["zero"], [1], np.asarray([0], dtype=np.float32)
                )
                inputs.append(names["zero"])
            model_ir.operators.append(OperatorIR(op_type, inputs, [names[key]]))
            conv_input = names[key]
    if transpose_conv:
        model_ir.tensors[names["shape"]] = _tensor(
            names["shape"], [4], np.asarray([1, 8, 10, 4], dtype=np.int32)
        )
        model_ir.operators.append(
            OperatorIR(
                "TRANSPOSE_CONV",
                [names["shape"], names["filter"], conv_input],
                [names["output"]],
                {"padding": "SAME", "strideH": 2, "strideW": 2},
            )
        )
    else:
        model_ir.operators.append(
            OperatorIR(
                "CONV_2D",
                [conv_input, names["filter"], names["bias"]],
                [names["output"]],
                {"padding": "SAME", "strideH": 1, "strideW": 1},
            )
        )
    return names


def test_concat_transpose_conv_repairs_multiple_families_with_one_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("indexed_concat_transpose_conv")
    branches = [
        _add_branch(model_ir, "direct"),
        _add_branch(model_ir, "prefix", pre_relu=True, post_prefix=True),
        _add_branch(model_ir, "deconv", transpose_conv=True),
    ]
    refreshes = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index):
        nonlocal refreshes
        refreshes += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    stats = _repair_nchw_concat_transpose_conv_axes(model_ir)

    assert stats == {"repaired_nchw_concat_transpose_conv_axes": 3}
    assert refreshes == 1
    for names in branches:
        assert model_ir.tensors[names["concat"]].shape == [1, 5, 4, 5]
        assert model_ir.tensors[names["transposed"]].shape == [1, 4, 5, 5]
    assert model_ir.tensors[branches[0]["output"]].shape == [1, 4, 5, 4]
    assert model_ir.tensors[branches[1]["output"]].shape == [1, 4, 10, 4]
    assert model_ir.tensors[branches[2]["output"]].shape == [1, 4, 10, 4]


def test_concat_transpose_conv_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_concat_transpose_conv")
    _add_branch(model_ir, "branch", pre_relu=True)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    stats = _repair_nchw_concat_transpose_conv_axes(
        model_ir, graph_index=graph_index, layout_state=layout_state
    )
    fresh = ModelIRGraphIndex(model_ir)
    assert stats == {"repaired_nchw_concat_transpose_conv_axes": 1}
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize(
    "case",
    [
        "conv_arity",
        "duplicate_data_producer",
        "transpose_perm",
        "perm_produced",
        "transpose_fanout",
        "transpose_public",
        "pre_fanout",
        "pre_public",
        "concat_type",
        "concat_arity",
        "concat_axis",
        "concat_axis_invalid",
        "concat_fanout",
        "concat_public",
        "input_rank",
        "input_spatial",
        "input_nonpositive",
        "filter_channels",
        "filter_missing",
        "filter_shape",
        "filter_produced",
        "transpose_already_correct",
    ],
)
def test_concat_transpose_conv_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = ModelIR("rejected_concat_transpose_conv")
    names = _add_branch(
        model_ir, "branch", pre_relu=case in {"pre_fanout", "pre_public"}
    )
    concat, transpose, conv = (
        model_ir.operators[0],
        model_ir.operators[-2],
        model_ir.operators[-1],
    )
    if case == "conv_arity":
        conv.inputs = [names["transposed"]]
    elif case == "duplicate_data_producer":
        model_ir.operators.insert(
            -1, OperatorIR("IDENTITY", [names["concat"]], [names["transposed"]])
        )
    elif case == "transpose_perm":
        model_ir.tensors[names["perm"]].data = np.asarray([0, 3, 1, 2], dtype=np.int32)
    elif case == "perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["bias"]], [names["perm"]])
        )
    elif case in {"transpose_fanout", "pre_fanout", "concat_fanout"}:
        source = {
            "transpose_fanout": names["transposed"],
            "pre_fanout": names["relu"],
            "concat_fanout": names["concat"],
        }[case]
        model_ir.tensors["side"] = _tensor("side", [1])
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    elif case in {"transpose_public", "pre_public", "concat_public"}:
        model_ir.outputs.append(
            {
                "transpose_public": names["transposed"],
                "pre_public": names["relu"],
                "concat_public": names["concat"],
            }[case]
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
    elif case == "input_spatial":
        model_ir.tensors[names["right"]].shape[2] = 6
    elif case == "input_nonpositive":
        model_ir.tensors[names["right"]].shape[1] = -1
        model_ir.tensors[names["filter"]].shape[3] = 1
        model_ir.tensors[names["filter"]].data = np.ones([4, 1, 1, 1], dtype=np.float32)
    elif case == "filter_channels":
        model_ir.tensors[names["filter"]].shape[3] = 4
    elif case == "filter_missing":
        model_ir.tensors[names["filter"]].data = None
    elif case == "filter_shape":
        model_ir.tensors[names["filter"]].data = np.ones([4, 1, 1, 4], dtype=np.float32)
    elif case == "filter_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["bias"]], [names["filter"]])
        )
    elif case == "transpose_already_correct":
        model_ir.tensors[names["transposed"]].shape = [1, 4, 5, 5]

    before = repr(model_ir)
    stats = _repair_nchw_concat_transpose_conv_axes(model_ir)
    assert stats == {"repaired_nchw_concat_transpose_conv_axes": 0}
    assert repr(model_ir) == before


def test_concat_transpose_conv_skips_index_without_required_families(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_concat_transpose_conv")
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]
    monkeypatch.setattr(
        concat_conv_module,
        "ModelIRGraphIndex",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected index")
        ),
    )
    assert _repair_nchw_concat_transpose_conv_axes(model_ir) == {
        "repaired_nchw_concat_transpose_conv_axes": 0
    }
