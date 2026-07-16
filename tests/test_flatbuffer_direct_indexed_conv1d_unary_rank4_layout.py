from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.conv1d_unary_layout as unary_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains,
)


_STATS = (
    "optimized_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains"
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


def _quantization(scale: float = 0.25, zero_point: int = 3) -> QuantParamIR:
    return QuantParamIR(scale=[scale], zero_point=[zero_point])


def _add_branch(
    model_ir: ModelIR,
    prefix: str,
    *,
    unary_type: str = "GELU",
    produced_source: bool = False,
    quantized: bool = False,
    fanout: bool = False,
    shared_shape: str | None = None,
    shared_axis: str | None = None,
) -> dict[str, str]:
    names = {
        key: f"{prefix}_{key}"
        for key in (
            "upstream",
            "source",
            "pre_perm",
            "pre_output",
            "unary_output",
            "mid_perm",
            "mid_output",
            "shape",
            "reshape_output",
            "axis",
            "expanded",
            "post_perm",
            "post_output",
            "terminal",
            "side_bias",
            "side_output",
        )
    }
    if shared_shape is not None:
        names["shape"] = shared_shape
    if shared_axis is not None:
        names["axis"] = shared_axis
    source_shape = [1, 5, 4, 3]
    pre_shape = [1, 3, 5, 4]
    mid_shape = [1, 5, 3, 4]
    reshape_shape = [5, 3, 4]
    expand_shape = [5, 3, 1, 4]
    post_shape = [5, 1, 4, 3]
    cast = str(unary_type) == "CAST"
    input_dtype = "INT8" if quantized else "FLOAT32"
    output_dtype = "INT32" if cast else input_dtype
    input_quantization = _quantization() if quantized else None
    output_quantization = _quantization() if quantized and not cast else None
    model_ir.tensors.update(
        {
            names["source"]: _tensor(
                names["source"],
                source_shape,
                dtype=input_dtype,
                quantization=copy.deepcopy(input_quantization),
            ),
            names["pre_perm"]: _tensor(
                names["pre_perm"],
                [4],
                dtype="INT32",
                data=np.asarray([0, 3, 1, 2], dtype=np.int32),
            ),
            names["pre_output"]: _tensor(
                names["pre_output"],
                pre_shape,
                dtype=input_dtype,
                quantization=copy.deepcopy(input_quantization),
            ),
            names["unary_output"]: _tensor(
                names["unary_output"],
                pre_shape,
                dtype=output_dtype,
                quantization=copy.deepcopy(output_quantization),
            ),
            names["mid_perm"]: _tensor(
                names["mid_perm"],
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 1, 3], dtype=np.int32),
            ),
            names["mid_output"]: _tensor(
                names["mid_output"],
                mid_shape,
                dtype=output_dtype,
                quantization=copy.deepcopy(output_quantization),
            ),
            names["reshape_output"]: _tensor(
                names["reshape_output"],
                reshape_shape,
                dtype=output_dtype,
                quantization=copy.deepcopy(output_quantization),
            ),
            names["expanded"]: _tensor(
                names["expanded"],
                expand_shape,
                dtype="UINT8",
                quantization=_quantization(0.5, 7),
            ),
            names["post_perm"]: _tensor(
                names["post_perm"],
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            names["post_output"]: _tensor(
                names["post_output"],
                post_shape,
                dtype=output_dtype,
                quantization=copy.deepcopy(output_quantization),
            ),
            names["terminal"]: _tensor(
                names["terminal"],
                post_shape,
                dtype=output_dtype,
                quantization=copy.deepcopy(output_quantization),
            ),
        }
    )
    if names["shape"] not in model_ir.tensors:
        model_ir.tensors[names["shape"]] = _tensor(
            names["shape"],
            [3],
            dtype="INT32",
            data=np.asarray(reshape_shape, dtype=np.int32),
        )
    if names["axis"] not in model_ir.tensors:
        model_ir.tensors[names["axis"]] = _tensor(
            names["axis"],
            [1],
            dtype="INT64",
            data=np.asarray([2], dtype=np.int64),
        )
    if produced_source:
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"],
            source_shape,
            dtype=input_dtype,
            quantization=copy.deepcopy(input_quantization),
        )
        model_ir.inputs.append(names["upstream"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["upstream"]], [names["source"]])
        )
    else:
        model_ir.inputs.append(names["source"])
    model_ir.outputs.append(names["terminal"])
    unary_options = (
        {"inDataType": input_dtype, "outDataType": output_dtype}
        if cast
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
                unary_type,
                [names["pre_output"]],
                [names["unary_output"]],
                unary_options,
                axis_semantics={"marker": "preserved"},
                version=3,
                onnx_node_name=f"{prefix}_unary",
                onnx_op_type=str(unary_type).title(),
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["unary_output"], names["mid_perm"]],
                [names["mid_output"]],
            ),
            OperatorIR(
                "RESHAPE",
                [names["mid_output"], names["shape"]],
                [names["reshape_output"]],
                {
                    "newShape": list(reshape_shape),
                    "onnxRawNewShape": list(reshape_shape),
                },
            ),
            OperatorIR(
                "EXPAND_DIMS",
                [names["reshape_output"], names["axis"]],
                [names["expanded"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["expanded"], names["post_perm"]],
                [names["post_output"]],
            ),
            OperatorIR(
                "IDENTITY",
                [names["post_output"]],
                [names["terminal"]],
            ),
        ]
    )
    if fanout:
        model_ir.tensors[names["side_bias"]] = _tensor(
            names["side_bias"],
            reshape_shape,
            dtype=output_dtype,
            data=np.zeros(reshape_shape, dtype=np.float32),
            quantization=copy.deepcopy(output_quantization),
        )
        model_ir.tensors[names["side_output"]] = _tensor(
            names["side_output"],
            reshape_shape,
            dtype=output_dtype,
            quantization=copy.deepcopy(output_quantization),
        )
        model_ir.outputs.append(names["side_output"])
        model_ir.operators.append(
            OperatorIR(
                "ADD",
                [names["reshape_output"], names["side_bias"]],
                [names["side_output"]],
            )
        )
    return names


def _operators(model_ir: ModelIR, names: dict[str, str]):
    pre = next(op for op in model_ir.operators if op.outputs == [names["pre_output"]])
    unary = next(
        op for op in model_ir.operators if op.outputs == [names["unary_output"]]
    )
    mid = next(op for op in model_ir.operators if op.outputs == [names["mid_output"]])
    reshape = next(
        op for op in model_ir.operators if op.outputs == [names["reshape_output"]]
    )
    expand = next(
        op for op in model_ir.operators if op.outputs == [names["expanded"]]
    )
    post = next(
        op for op in model_ir.operators if op.outputs == [names["post_output"]]
    )
    return pre, unary, mid, reshape, expand, post


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_rank4_conv1d_unary_rewrites_multiple_chains_with_one_index(monkeypatch) -> None:
    model_ir = ModelIR("indexed_rank4_conv1d_unary")
    branches = [
        _add_branch(model_ir, "gelu"),
        _add_branch(model_ir, "produced", unary_type="RELU", produced_source=True),
        _add_branch(model_ir, "quantized", unary_type="LOGISTIC", quantized=True),
        _add_branch(model_ir, "cast", unary_type="CAST"),
        _add_branch(model_ir, "fanout", unary_type="ABS", quantized=True, fanout=True),
        _add_branch(
            model_ir,
            "shared0",
            shared_shape="shared_shape",
            shared_axis="shared_axis",
        ),
        _add_branch(
            model_ir,
            "shared1",
            shared_shape="shared_shape",
            shared_axis="shared_axis",
        ),
    ]
    originals = [copy.deepcopy(_operators(model_ir, names)[1]) for names in branches]
    refreshes = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refreshes
        refreshes += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 7}
    assert refreshes == 1
    for names, original in zip(branches, originals):
        unary = next(op for op in model_ir.operators if op.onnx_node_name == original.onnx_node_name)
        reshape = next(
            op for op in model_ir.operators if op.outputs == [names["reshape_output"]]
        )
        expand = next(op for op in model_ir.operators if op.outputs == [names["expanded"]])
        terminal = next(op for op in model_ir.operators if op.outputs == [names["terminal"]])
        assert unary.inputs == [names["source"]]
        assert unary.options == original.options
        assert unary.axis_semantics == original.axis_semantics
        assert unary.version == original.version
        assert unary.onnx_op_type == original.onnx_op_type
        assert model_ir.tensors[names["unary_output"]].shape == [1, 5, 4, 3]
        assert reshape.inputs[0] == names["unary_output"]
        assert reshape.options["newShape"] == [5, 4, 3]
        assert model_ir.tensors[names["reshape_output"]].shape == [5, 4, 3]
        assert np.asarray(model_ir.tensors[reshape.inputs[1]].data).tolist() == [5, 4, 3]
        assert np.asarray(model_ir.tensors[expand.inputs[1]].data).tolist() == [1]
        assert model_ir.tensors[names["expanded"]].shape == [5, 1, 4, 3]
        assert terminal.inputs == [names["expanded"]]

    first_shared = next(
        op for op in model_ir.operators if op.outputs == [branches[-2]["reshape_output"]]
    )
    second_shared = next(
        op for op in model_ir.operators if op.outputs == [branches[-1]["reshape_output"]]
    )
    assert first_shared.inputs[1] == "shared_shape_nhwc_shape"
    assert second_shared.inputs[1] == "shared_shape"
    assert np.asarray(model_ir.tensors["shared_shape"].data).tolist() == [5, 4, 3]
    assert validate_model_ir_invariants(model_ir) == []


def test_rank4_conv1d_unary_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_rank4_conv1d_unary")
    _add_branch(model_ir, "branch", fanout=True)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []


def test_rank4_conv1d_unary_inserts_fanout_bridge_before_consumer() -> None:
    model_ir = ModelIR("topological_rank4_fanout")
    names = _add_branch(model_ir, "branch", fanout=True)

    stats = _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    bridge_index = next(
        index
        for index, op in enumerate(model_ir.operators)
        if op.op_type == "TRANSPOSE" and op.inputs[0] == names["reshape_output"]
    )
    side_index = next(
        index
        for index, op in enumerate(model_ir.operators)
        if op.outputs == [names["side_output"]]
    )
    bridge = model_ir.operators[bridge_index]
    side = model_ir.operators[side_index]
    assert bridge_index < side_index
    assert side.inputs[0] == bridge.outputs[0]
    assert model_ir.tensors[bridge.outputs[0]].shape == [5, 3, 4]
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    ("source_signature", "pre_signature", "mid_signature", "reshape_signature", "post_signature"),
    [
        ([1, -1, 4, 3], [1, 3, -1, 4], [1, -1, 3, 4], [-1, 3, 4], [-1, 1, 4, 3]),
        ([1, 5, -1, 3], [1, 3, 5, -1], [1, 5, 3, -1], [5, 3, -1], [5, 1, -1, 3]),
        ([1, 5, 4, -1], [1, -1, 5, 4], [1, 5, -1, 4], [5, -1, 4], [5, 1, 4, -1]),
    ],
)
def test_rank4_conv1d_unary_preserves_one_dynamic_reshape_dimension(
    source_signature: list[int],
    pre_signature: list[int],
    mid_signature: list[int],
    reshape_signature: list[int],
    post_signature: list[int],
) -> None:
    model_ir = ModelIR("dynamic_rank4_conv1d_unary")
    names = _add_branch(model_ir, "branch")
    model_ir.tensors[names["source"]].shape_signature = source_signature
    model_ir.tensors[names["pre_output"]].shape_signature = pre_signature
    model_ir.tensors[names["unary_output"]].shape_signature = pre_signature
    model_ir.tensors[names["mid_output"]].shape_signature = mid_signature
    model_ir.tensors[names["reshape_output"]].shape_signature = reshape_signature
    model_ir.tensors[names["expanded"]].shape_signature = [
        reshape_signature[0],
        reshape_signature[1],
        1,
        reshape_signature[2],
    ]
    model_ir.tensors[names["post_output"]].shape_signature = post_signature
    model_ir.tensors[names["terminal"]].shape_signature = post_signature

    stats = _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
        model_ir
    )

    expected_reshape_signature = [
        reshape_signature[0],
        reshape_signature[2],
        reshape_signature[1],
    ]
    assert stats == {_STATS: 1}
    reshape = next(
        op for op in model_ir.operators if op.outputs == [names["reshape_output"]]
    )
    assert reshape.options["newShape"] == expected_reshape_signature
    assert reshape.options["preserveDynamicShape"] is True
    assert np.asarray(model_ir.tensors[reshape.inputs[1]].data).tolist() == (
        expected_reshape_signature
    )
    assert model_ir.tensors[names["expanded"]].shape_signature == post_signature


@pytest.mark.parametrize(
    "case",
    [
        "pre_arity",
        "pre_perm",
        "pre_perm_dtype",
        "pre_perm_produced",
        "source_unbound",
        "source_late_producer",
        "source_boundary_and_producer",
        "batch_not_one",
        "batch_dynamic",
        "pre_public",
        "pre_fanout",
        "pre_shape",
        "unary_type",
        "unary_arity",
        "unary_shape",
        "unary_public",
        "mid_arity",
        "mid_perm",
        "mid_perm_produced",
        "mid_shape",
        "mid_public",
        "reshape_arity",
        "reshape_shape_values",
        "reshape_shape_dtype",
        "reshape_shape_produced",
        "reshape_shape_input",
        "reshape_shape_public",
        "reshape_options_mismatch",
        "reshape_output_shape",
        "reshape_output_public",
        "no_expand",
        "expand_arity",
        "expand_axis",
        "expand_axis_dtype",
        "expand_axis_produced",
        "expand_axis_input",
        "expand_axis_public",
        "expand_shape",
        "expand_public",
        "expand_fanout",
        "post_arity",
        "post_perm",
        "post_perm_produced",
        "post_output_missing",
        "post_output_input",
        "post_output_duplicate",
        "post_output_shape",
        "post_output_backward_consumer",
        "input_dtype_mismatch",
        "output_dtype_mismatch",
        "quantization_mismatch",
        "per_axis_quantization",
        "two_dynamic_reshape_dimensions",
    ],
)
def test_rank4_conv1d_unary_rejects_unsafe_contracts_transactionally(case: str) -> None:
    model_ir = ModelIR("rejected_rank4_conv1d_unary")
    names = _add_branch(model_ir, "branch")
    pre, unary, mid, reshape, expand, post = _operators(model_ir, names)

    def add_fanout(source: str, output: str, shape: list[int]) -> None:
        model_ir.tensors[output] = _tensor(output, shape)
        model_ir.outputs.append(output)
        model_ir.operators.append(OperatorIR("IDENTITY", [source], [output]))

    if case == "pre_arity":
        pre.inputs.append("extra")
    elif case == "pre_perm":
        model_ir.tensors[names["pre_perm"]].data[1] = 2
    elif case == "pre_perm_dtype":
        model_ir.tensors[names["pre_perm"]].dtype = "FLOAT32"
        model_ir.tensors[names["pre_perm"]].data = np.asarray(
            [0, 3, 1, 2], dtype=np.float32
        )
    elif case == "pre_perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axis"]], [names["pre_perm"]])
        )
    elif case == "source_unbound":
        model_ir.inputs.remove(names["source"])
    elif case == "source_late_producer":
        model_ir.inputs.remove(names["source"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["terminal"]], [names["source"]])
        )
    elif case == "source_boundary_and_producer":
        model_ir.operators.insert(
            0, OperatorIR("IDENTITY", [names["terminal"]], [names["source"]])
        )
    elif case == "batch_not_one":
        for key in ("source", "pre_output", "unary_output", "mid_output"):
            model_ir.tensors[names[key]].shape[0] = 2
            model_ir.tensors[names[key]].shape_signature[0] = 2
    elif case == "batch_dynamic":
        for key in ("source", "pre_output", "unary_output", "mid_output"):
            model_ir.tensors[names[key]].shape_signature[0] = -1
    elif case == "pre_public":
        model_ir.outputs.append(names["pre_output"])
    elif case == "pre_fanout":
        add_fanout(names["pre_output"], "side", [1, 3, 5, 4])
    elif case == "pre_shape":
        model_ir.tensors[names["pre_output"]].shape[1] = 4
        model_ir.tensors[names["pre_output"]].shape_signature[1] = 4
    elif case == "unary_type":
        unary.op_type = "ADD"
    elif case == "unary_arity":
        unary.inputs.append(names["axis"])
    elif case == "unary_shape":
        model_ir.tensors[names["unary_output"]].shape[3] = 5
        model_ir.tensors[names["unary_output"]].shape_signature[3] = 5
    elif case == "unary_public":
        model_ir.outputs.append(names["unary_output"])
    elif case == "mid_arity":
        mid.inputs.append("extra")
    elif case == "mid_perm":
        model_ir.tensors[names["mid_perm"]].data[1] = 1
    elif case == "mid_perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axis"]], [names["mid_perm"]])
        )
    elif case == "mid_shape":
        model_ir.tensors[names["mid_output"]].shape[2] = 4
        model_ir.tensors[names["mid_output"]].shape_signature[2] = 4
    elif case == "mid_public":
        model_ir.outputs.append(names["mid_output"])
    elif case == "reshape_arity":
        reshape.inputs.append(names["axis"])
    elif case == "reshape_shape_values":
        model_ir.tensors[names["shape"]].data[1] = 4
    elif case == "reshape_shape_dtype":
        model_ir.tensors[names["shape"]].dtype = "FLOAT32"
        model_ir.tensors[names["shape"]].data = np.asarray(
            [5, 3, 4], dtype=np.float32
        )
    elif case == "reshape_shape_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axis"]], [names["shape"]])
        )
    elif case == "reshape_shape_input":
        model_ir.inputs.append(names["shape"])
    elif case == "reshape_shape_public":
        model_ir.outputs.append(names["shape"])
    elif case == "reshape_options_mismatch":
        reshape.options["newShape"] = [5, 4, 3]
    elif case == "reshape_output_shape":
        model_ir.tensors[names["reshape_output"]].shape[1] = 4
        model_ir.tensors[names["reshape_output"]].shape_signature[1] = 4
    elif case == "reshape_output_public":
        model_ir.outputs.append(names["reshape_output"])
    elif case == "no_expand":
        expand.op_type = "RESHAPE"
    elif case == "expand_arity":
        expand.inputs.append("extra")
    elif case == "expand_axis":
        model_ir.tensors[names["axis"]].data[0] = 1
    elif case == "expand_axis_dtype":
        model_ir.tensors[names["axis"]].dtype = "FLOAT32"
        model_ir.tensors[names["axis"]].data = np.asarray([2], dtype=np.float32)
    elif case == "expand_axis_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["pre_perm"]], [names["axis"]])
        )
    elif case == "expand_axis_input":
        model_ir.inputs.append(names["axis"])
    elif case == "expand_axis_public":
        model_ir.outputs.append(names["axis"])
    elif case == "expand_shape":
        model_ir.tensors[names["expanded"]].shape[1] = 4
    elif case == "expand_public":
        model_ir.outputs.append(names["expanded"])
    elif case == "expand_fanout":
        add_fanout(names["expanded"], "side", [5, 3, 1, 4])
    elif case == "post_arity":
        post.inputs.append("extra")
    elif case == "post_perm":
        model_ir.tensors[names["post_perm"]].data[1] = 3
    elif case == "post_perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axis"]], [names["post_perm"]])
        )
    elif case == "post_output_missing":
        del model_ir.tensors[names["post_output"]]
    elif case == "post_output_input":
        model_ir.inputs.append(names["post_output"])
    elif case == "post_output_duplicate":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["post_output"]])
        )
    elif case == "post_output_shape":
        model_ir.tensors[names["post_output"]].shape[2] = 5
    elif case == "post_output_backward_consumer":
        model_ir.tensors["side"] = _tensor("side", [5, 1, 4, 3])
        model_ir.outputs.append("side")
        post_index = model_ir.operators.index(post)
        model_ir.operators.insert(
            post_index,
            OperatorIR("IDENTITY", [names["post_output"]], ["side"]),
        )
    elif case == "input_dtype_mismatch":
        model_ir.tensors[names["pre_output"]].dtype = "FLOAT16"
    elif case == "output_dtype_mismatch":
        model_ir.tensors[names["mid_output"]].dtype = "FLOAT16"
    elif case == "quantization_mismatch":
        model_ir.tensors[names["unary_output"]].quantization = _quantization()
    elif case == "per_axis_quantization":
        for key in (
            "source",
            "pre_output",
            "unary_output",
            "mid_output",
            "reshape_output",
            "post_output",
        ):
            model_ir.tensors[names[key]].quantization = QuantParamIR(
                scale=[0.25, 0.5],
                zero_point=[0, 0],
                quantized_dimension=1,
            )
    elif case == "two_dynamic_reshape_dimensions":
        model_ir.tensors[names["source"]].shape_signature = [1, -1, 4, -1]
        model_ir.tensors[names["pre_output"]].shape_signature = [1, -1, -1, 4]
        model_ir.tensors[names["unary_output"]].shape_signature = [1, -1, -1, 4]
        model_ir.tensors[names["mid_output"]].shape_signature = [1, -1, -1, 4]
        model_ir.tensors[names["reshape_output"]].shape_signature = [-1, -1, 4]
        model_ir.tensors[names["expanded"]].shape_signature = [-1, -1, 1, 4]
        model_ir.tensors[names["post_output"]].shape_signature = [-1, 1, 4, -1]

    before = repr(model_ir)
    stats = _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_rank4_conv1d_unary_quantization_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir = ModelIR("unclonable_rank4_conv1d_unary")
    names = _add_branch(model_ir, "branch", quantized=True)
    quantization = {
        "scale": [0.25],
        "zero_point": [3],
        "fault": Unclonable(),
    }
    for key in (
        "source",
        "pre_output",
        "unary_output",
        "mid_output",
        "reshape_output",
        "post_output",
    ):
        model_ir.tensors[names[key]].quantization = quantization
    before = repr(model_ir)

    stats = _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_rank4_conv1d_unary_preflight_prunes_without_index(monkeypatch) -> None:
    model_ir = ModelIR("no_rank4_conv1d_unary")
    model_ir.tensors["unused"] = _tensor("unused", [1])
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(unary_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
        model_ir
    ) == {_STATS: 0}
    assert "unused" not in model_ir.tensors
