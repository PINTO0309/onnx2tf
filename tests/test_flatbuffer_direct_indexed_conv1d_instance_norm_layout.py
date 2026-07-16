from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.conv1d_instance_norm_layout as norm_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_instance_norm_layout import (
    _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains,
)


_STATS = (
    "optimized_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains"
)
_UNARY_TYPES = (
    "RELU",
    "RELU6",
    "RELU_0_TO_1",
    "LEAKY_RELU",
    "LOGISTIC",
    "TANH",
    "GELU",
    "ABS",
    "NEG",
    "SQRT",
    "EXP",
    "CAST",
    "FLOOR",
    "CEIL",
    "ROUND",
    "HARD_SWISH",
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data=None,
    quantization=None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _add_branch(
    model_ir: ModelIR,
    prefix: str,
    *,
    unary_type: str = "GELU",
    produced_source: bool = False,
    shared_reshape2_shape: str | None = None,
    shared_expand_axis: str | None = None,
) -> dict[str, str]:
    keys = (
        "upstream",
        "source",
        "pre_perm",
        "pre_output",
        "squeezed",
        "reshape1_shape",
        "flat",
        "axes",
        "mean1",
        "centered",
        "square",
        "mean2",
        "epsilon",
        "variance",
        "standard_deviation",
        "one",
        "inverse",
        "norm",
        "gamma",
        "scaled",
        "beta",
        "biased",
        "reshape2_shape",
        "reshape2",
        "unary",
        "expand_axis",
        "expanded",
        "post_perm",
        "post",
        "terminal",
    )
    names = {key: f"{prefix}_{key}" for key in keys}
    if shared_reshape2_shape is not None:
        names["reshape2_shape"] = shared_reshape2_shape
    if shared_expand_axis is not None:
        names["expand_axis"] = shared_expand_axis

    n, c, w = 1, 6, 5
    k = c * w
    source_shape = [n, 1, w, c]
    pre_shape = [n, c, 1, w]
    rank3_shape = [n, c, w]
    flat_shape = [n, 1, k]
    mean_shape = [n, 1, 1]
    output_dtype = "FLOAT16" if unary_type == "CAST" else "FLOAT32"
    data_shapes = {
        "source": source_shape,
        "pre_output": pre_shape,
        "squeezed": rank3_shape,
        "flat": flat_shape,
        "mean1": mean_shape,
        "centered": flat_shape,
        "square": flat_shape,
        "mean2": mean_shape,
        "variance": mean_shape,
        "standard_deviation": mean_shape,
        "inverse": mean_shape,
        "norm": flat_shape,
        "scaled": flat_shape,
        "biased": flat_shape,
        "reshape2": rank3_shape,
    }
    for key, shape in data_shapes.items():
        model_ir.tensors[names[key]] = _tensor(names[key], shape)
    for key, shape in {
        "unary": rank3_shape,
        "expanded": pre_shape,
        "post": source_shape,
        "terminal": source_shape,
    }.items():
        model_ir.tensors[names[key]] = _tensor(
            names[key],
            shape,
            dtype=output_dtype,
        )
    constants = {
        "pre_perm": ("INT32", [4], np.asarray([0, 3, 1, 2], dtype=np.int32)),
        "reshape1_shape": (
            "INT32",
            [3],
            np.asarray(flat_shape, dtype=np.int32),
        ),
        "axes": ("INT32", [1], np.asarray([2], dtype=np.int32)),
        "epsilon": ("FLOAT32", [1], np.asarray([1e-5], dtype=np.float32)),
        "one": ("FLOAT32", [1], np.asarray([1.0], dtype=np.float32)),
        "gamma": (
            "FLOAT32",
            [1, 1, 1],
            np.asarray([[[1.0]]], dtype=np.float32),
        ),
        "beta": (
            "FLOAT32",
            [1, 1, 1],
            np.asarray([[[0.0]]], dtype=np.float32),
        ),
        "post_perm": (
            "INT32",
            [4],
            np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
    }
    for key, (dtype, shape, data) in constants.items():
        model_ir.tensors[names[key]] = _tensor(
            names[key],
            shape,
            dtype=dtype,
            data=data,
        )
    if names["reshape2_shape"] not in model_ir.tensors:
        model_ir.tensors[names["reshape2_shape"]] = _tensor(
            names["reshape2_shape"],
            [3],
            dtype="INT32",
            data=np.asarray(rank3_shape, dtype=np.int32),
        )
    if names["expand_axis"] not in model_ir.tensors:
        model_ir.tensors[names["expand_axis"]] = _tensor(
            names["expand_axis"],
            [1],
            dtype="INT32",
            data=np.asarray([2], dtype=np.int32),
        )
    if produced_source:
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"],
            source_shape,
        )
        model_ir.inputs.append(names["upstream"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["upstream"]], [names["source"]])
        )
    else:
        model_ir.inputs.append(names["source"])
    model_ir.outputs.append(names["terminal"])
    unary_options = (
        {"inDataType": "FLOAT32", "outDataType": output_dtype}
        if unary_type == "CAST"
        else {"marker": prefix}
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [names["source"], names["pre_perm"]],
                [names["pre_output"]],
            ),
            OperatorIR(
                "SQUEEZE",
                [names["pre_output"]],
                [names["squeezed"]],
                {"squeezeDims": [2]},
            ),
            OperatorIR(
                "RESHAPE",
                [names["squeezed"], names["reshape1_shape"]],
                [names["flat"]],
                {"newShape": flat_shape},
            ),
            OperatorIR(
                "MEAN",
                [names["flat"], names["axes"]],
                [names["mean1"]],
                {"keepDims": True},
            ),
            OperatorIR(
                "SUB",
                [names["flat"], names["mean1"]],
                [names["centered"]],
            ),
            OperatorIR(
                "MUL",
                [names["centered"], names["centered"]],
                [names["square"]],
            ),
            OperatorIR(
                "MEAN",
                [names["square"], names["axes"]],
                [names["mean2"]],
                {"keepDims": True},
            ),
            OperatorIR(
                "ADD",
                [names["mean2"], names["epsilon"]],
                [names["variance"]],
            ),
            OperatorIR(
                "SQRT",
                [names["variance"]],
                [names["standard_deviation"]],
            ),
            OperatorIR(
                "DIV",
                [names["one"], names["standard_deviation"]],
                [names["inverse"]],
            ),
            OperatorIR(
                "MUL",
                [names["centered"], names["inverse"]],
                [names["norm"]],
            ),
            OperatorIR(
                "MUL",
                [names["norm"], names["gamma"]],
                [names["scaled"]],
            ),
            OperatorIR(
                "ADD",
                [names["scaled"], names["beta"]],
                [names["biased"]],
            ),
            OperatorIR(
                "RESHAPE",
                [names["biased"], names["reshape2_shape"]],
                [names["reshape2"]],
                {"newShape": rank3_shape},
            ),
            OperatorIR(
                unary_type,
                [names["reshape2"]],
                [names["unary"]],
                unary_options,
                axis_semantics={"marker": "preserved"},
                version=3,
                onnx_node_name=f"{prefix}_unary",
                onnx_op_type=str(unary_type).title(),
            ),
            OperatorIR(
                "EXPAND_DIMS",
                [names["unary"], names["expand_axis"]],
                [names["expanded"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["expanded"], names["post_perm"]],
                [names["post"]],
            ),
            OperatorIR("IDENTITY", [names["post"]], [names["terminal"]]),
        ]
    )
    return names


def _operator(model_ir: ModelIR, output_name: str) -> OperatorIR:
    return next(op for op in model_ir.operators if op.outputs == [output_name])


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_conv1d_instance_norm_rewrites_multiple_chains_with_one_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("indexed_conv1d_instance_norm")
    branches = [
        _add_branch(model_ir, "gelu"),
        _add_branch(model_ir, "produced", produced_source=True),
        _add_branch(model_ir, "cast", unary_type="CAST"),
        _add_branch(
            model_ir,
            "shared0",
            shared_reshape2_shape="shared_reshape2_shape",
            shared_expand_axis="shared_expand_axis",
        ),
        _add_branch(
            model_ir,
            "shared1",
            shared_reshape2_shape="shared_reshape2_shape",
            shared_expand_axis="shared_expand_axis",
        ),
    ]
    original_unaries = [
        copy.deepcopy(_operator(model_ir, names["unary"])) for names in branches
    ]
    refreshes = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refreshes
        refreshes += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 5}
    assert refreshes == 1
    assert validate_model_ir_invariants(model_ir) == []
    for names, original_unary in zip(branches, original_unaries):
        squeeze = _operator(model_ir, names["squeezed"])
        reshape2 = _operator(model_ir, names["reshape2"])
        unary = _operator(model_ir, names["unary"])
        expand = _operator(model_ir, names["expanded"])
        terminal = _operator(model_ir, names["terminal"])
        assert squeeze.inputs == [names["source"]]
        assert squeeze.options["squeezeDims"] == [1]
        assert model_ir.tensors[names["squeezed"]].shape == [1, 5, 6]
        assert reshape2.options["newShape"] == [1, 5, 6]
        assert model_ir.tensors[names["reshape2"]].shape == [1, 5, 6]
        assert unary.op_type == original_unary.op_type
        assert unary.options == original_unary.options
        assert unary.axis_semantics == original_unary.axis_semantics
        assert unary.version == original_unary.version
        assert unary.onnx_node_name == original_unary.onnx_node_name
        assert unary.onnx_op_type == original_unary.onnx_op_type
        assert model_ir.tensors[names["unary"]].shape == [1, 5, 6]
        assert model_ir.tensors[names["expanded"]].shape == [1, 1, 5, 6]
        assert np.asarray(
            model_ir.tensors[reshape2.inputs[1]].data
        ).reshape(-1).tolist() == [1, 5, 6]
        assert np.asarray(
            model_ir.tensors[expand.inputs[1]].data
        ).reshape(-1).tolist() == [1]
        assert terminal.inputs == [names["expanded"]]
    assert branches[3]["reshape2_shape"] != _operator(
        model_ir, branches[3]["reshape2"]
    ).inputs[1]
    assert branches[4]["reshape2_shape"] == _operator(
        model_ir, branches[4]["reshape2"]
    ).inputs[1]
    assert branches[3]["expand_axis"] != _operator(
        model_ir, branches[3]["expanded"]
    ).inputs[1]
    assert branches[4]["expand_axis"] == _operator(
        model_ir, branches[4]["expanded"]
    ).inputs[1]


def test_conv1d_instance_norm_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_conv1d_instance_norm")
    _add_branch(model_ir, "branch")
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("unary_type", _UNARY_TYPES)
def test_conv1d_instance_norm_preserves_supported_unary_family(
    unary_type: str,
) -> None:
    model_ir = ModelIR(f"conv1d_instance_norm_{unary_type.lower()}")
    names = _add_branch(model_ir, "branch", unary_type=unary_type)

    stats = _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    unary = _operator(model_ir, names["unary"])
    assert unary.op_type == unary_type
    assert model_ir.tensors[names["expanded"]].dtype == (
        "FLOAT16" if unary_type == "CAST" else "FLOAT32"
    )
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("dynamic_axis", ("batch", "width", "channel"))
def test_conv1d_instance_norm_preserves_one_dynamic_dimension(
    dynamic_axis: str,
) -> None:
    model_ir = ModelIR(f"dynamic_conv1d_instance_norm_{dynamic_axis}")
    names = _add_branch(model_ir, "branch")
    n_sig = -1 if dynamic_axis == "batch" else 1
    w_sig = -1 if dynamic_axis == "width" else 5
    c_sig = -1 if dynamic_axis == "channel" else 6
    source_signature = [n_sig, 1, w_sig, c_sig]
    pre_signature = [n_sig, c_sig, 1, w_sig]
    old_rank3_signature = [n_sig, c_sig, w_sig]
    flat_signature = [n_sig, 1, -1 if -1 in (w_sig, c_sig) else 30]
    mean_signature = [n_sig, 1, 1]
    for key in ("source", "post", "terminal"):
        model_ir.tensors[names[key]].shape_signature = source_signature
    model_ir.tensors[names["pre_output"]].shape_signature = pre_signature
    for key in ("squeezed", "reshape2", "unary"):
        model_ir.tensors[names[key]].shape_signature = old_rank3_signature
    for key in ("flat", "centered", "square", "norm", "scaled", "biased"):
        model_ir.tensors[names[key]].shape_signature = flat_signature
    for key in (
        "mean1",
        "mean2",
        "variance",
        "standard_deviation",
        "inverse",
    ):
        model_ir.tensors[names[key]].shape_signature = mean_signature
    model_ir.tensors[names["expanded"]].shape_signature = pre_signature

    stats = _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    expected_rank3 = [n_sig, w_sig, c_sig]
    assert model_ir.tensors[names["reshape2"]].shape_signature == expected_rank3
    assert model_ir.tensors[names["expanded"]].shape_signature == source_signature
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "case",
    [
        "source_height",
        "source_dynamic_height",
        "floating_pre_perm",
        "produced_pre_perm",
        "pre_public",
        "pre_fanout",
        "pre_shape",
        "squeeze_axis",
        "squeezed_public",
        "reshape1_shape",
        "produced_reshape1_shape",
        "flat_fanout",
        "mean_axis",
        "mean_keepdims",
        "sub_reversed",
        "square_inputs",
        "negative_epsilon",
        "produced_epsilon",
        "div_reversed",
        "one_value",
        "gamma_vector",
        "produced_gamma",
        "beta_vector",
        "reshape2_shape",
        "reshape2_public",
        "unary_type",
        "unary_arity",
        "expand_axis",
        "produced_expand_axis",
        "expand_public",
        "floating_post_perm",
        "post_duplicate",
        "post_backward_consumer",
        "mixed_dtype",
        "quantized_data",
        "inconsistent_signature",
        "two_dynamic_target_dimensions",
    ],
)
def test_conv1d_instance_norm_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = ModelIR("rejected_conv1d_instance_norm")
    names = _add_branch(model_ir, "branch")
    squeeze = _operator(model_ir, names["squeezed"])
    mean1 = _operator(model_ir, names["mean1"])
    sub = _operator(model_ir, names["centered"])
    square = _operator(model_ir, names["square"])
    div = _operator(model_ir, names["inverse"])
    unary = _operator(model_ir, names["unary"])
    post = _operator(model_ir, names["post"])

    def add_fanout(source: str, output: str, shape: list[int]) -> None:
        model_ir.tensors[output] = _tensor(output, shape)
        model_ir.outputs.append(output)
        model_ir.operators.append(OperatorIR("IDENTITY", [source], [output]))

    if case == "source_height":
        model_ir.tensors[names["source"]].shape[1] = 2
        model_ir.tensors[names["source"]].shape_signature[1] = 2
    elif case == "source_dynamic_height":
        model_ir.tensors[names["source"]].shape_signature[1] = -1
    elif case == "floating_pre_perm":
        tensor = model_ir.tensors[names["pre_perm"]]
        tensor.dtype = "FLOAT32"
        tensor.data = np.asarray([0, 3, 1, 2], dtype=np.float32)
    elif case == "produced_pre_perm":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axes"]], [names["pre_perm"]])
        )
    elif case == "pre_public":
        model_ir.outputs.append(names["pre_output"])
    elif case == "pre_fanout":
        add_fanout(names["pre_output"], "extra", [1, 6, 1, 5])
    elif case == "pre_shape":
        model_ir.tensors[names["pre_output"]].shape[1] = 5
        model_ir.tensors[names["pre_output"]].shape_signature[1] = 5
    elif case == "squeeze_axis":
        squeeze.options["squeezeDims"] = [3]
    elif case == "squeezed_public":
        model_ir.outputs.append(names["squeezed"])
    elif case == "reshape1_shape":
        model_ir.tensors[names["reshape1_shape"]].data[2] = 29
    elif case == "produced_reshape1_shape":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axes"]], [names["reshape1_shape"]])
        )
    elif case == "flat_fanout":
        add_fanout(names["flat"], "extra", [1, 1, 30])
    elif case == "mean_axis":
        model_ir.tensors[names["axes"]].data[0] = 1
    elif case == "mean_keepdims":
        mean1.options["keepDims"] = False
    elif case == "sub_reversed":
        sub.inputs.reverse()
    elif case == "square_inputs":
        square.inputs[1] = names["mean1"]
    elif case == "negative_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = -1e-5
    elif case == "produced_epsilon":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["one"]], [names["epsilon"]])
        )
    elif case == "div_reversed":
        div.inputs.reverse()
    elif case == "one_value":
        model_ir.tensors[names["one"]].data[0] = 2.0
    elif case == "gamma_vector":
        model_ir.tensors[names["gamma"]].data = np.asarray(
            [1.0, 1.0], dtype=np.float32
        )
    elif case == "produced_gamma":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["one"]], [names["gamma"]])
        )
    elif case == "beta_vector":
        model_ir.tensors[names["beta"]].data = np.asarray(
            [0.0, 0.0], dtype=np.float32
        )
    elif case == "reshape2_shape":
        model_ir.tensors[names["reshape2_shape"]].data[1] = 5
    elif case == "reshape2_public":
        model_ir.outputs.append(names["reshape2"])
    elif case == "unary_type":
        unary.op_type = "ADD"
    elif case == "unary_arity":
        unary.inputs.append(names["one"])
    elif case == "expand_axis":
        model_ir.tensors[names["expand_axis"]].data[0] = 1
    elif case == "produced_expand_axis":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axes"]], [names["expand_axis"]])
        )
    elif case == "expand_public":
        model_ir.outputs.append(names["expanded"])
    elif case == "floating_post_perm":
        tensor = model_ir.tensors[names["post_perm"]]
        tensor.dtype = "FLOAT32"
        tensor.data = np.asarray([0, 2, 3, 1], dtype=np.float32)
    elif case == "post_duplicate":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["post"]])
        )
    elif case == "post_backward_consumer":
        model_ir.tensors["extra"] = _tensor("extra", [1, 1, 5, 6])
        model_ir.outputs.append("extra")
        model_ir.operators.insert(
            model_ir.operators.index(post),
            OperatorIR("IDENTITY", [names["post"]], ["extra"]),
        )
    elif case == "mixed_dtype":
        model_ir.tensors[names["centered"]].dtype = "FLOAT16"
    elif case == "quantized_data":
        model_ir.tensors[names["flat"]].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "inconsistent_signature":
        model_ir.tensors[names["flat"]].shape_signature[2] = -1
    elif case == "two_dynamic_target_dimensions":
        for key in ("source", "post", "terminal"):
            model_ir.tensors[names[key]].shape_signature = [-1, 1, -1, 6]
        model_ir.tensors[names["pre_output"]].shape_signature = [-1, 6, 1, -1]
        for key in ("squeezed", "reshape2", "unary"):
            model_ir.tensors[names[key]].shape_signature = [-1, 6, -1]
        for key in ("flat", "centered", "square", "norm", "scaled", "biased"):
            model_ir.tensors[names[key]].shape_signature = [-1, 1, -1]
        for key in (
            "mean1",
            "mean2",
            "variance",
            "standard_deviation",
            "inverse",
        ):
            model_ir.tensors[names[key]].shape_signature = [-1, 1, 1]
        model_ir.tensors[names["expanded"]].shape_signature = [-1, 6, 1, -1]

    before = repr(model_ir)
    stats = _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_conv1d_instance_norm_shared_constant_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir = ModelIR("unclonable_conv1d_instance_norm")
    first = _add_branch(
        model_ir,
        "first",
        shared_reshape2_shape="shared_shape",
        shared_expand_axis="shared_axis",
    )
    _add_branch(
        model_ir,
        "second",
        shared_reshape2_shape="shared_shape",
        shared_expand_axis="shared_axis",
    )
    model_ir.tensors[first["reshape2_shape"]].quantization = {
        "fault": Unclonable()
    }
    before = repr(model_ir)

    stats = _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_conv1d_instance_norm_preflight_prunes_without_index(monkeypatch) -> None:
    model_ir = ModelIR("no_conv1d_instance_norm")
    model_ir.tensors["unused"] = _tensor("unused", [1])
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(norm_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
        model_ir
    ) == {_STATS: 0}
    assert "unused" not in model_ir.tensors
