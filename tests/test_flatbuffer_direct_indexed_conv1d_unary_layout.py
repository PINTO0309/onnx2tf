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
    _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains,
    _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains,
)


_STATS = "optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_chains"
_FANOUT_STATS = (
    "optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains"
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


def _quantization(scale: float = 0.25, zero_point: int = 3) -> QuantParamIR:
    return QuantParamIR(scale=[scale], zero_point=[zero_point])


def _add_branch(
    model_ir: ModelIR,
    prefix: str,
    *,
    unary_type: str = "GELU",
    produced_source: bool = False,
    quantized: bool = False,
    squeeze_axis: int = 2,
    explicit_squeeze_axis: bool = True,
    fanout: bool = False,
) -> dict[str, str]:
    names = {
        key: f"{prefix}_{key}"
        for key in (
            "upstream",
            "source",
            "pre_perm",
            "pre_output",
            "squeezed",
            "unary_output",
            "axis",
            "expanded",
            "post_perm",
            "output",
            "side_output",
        )
    }
    height, width = (1, 5) if int(squeeze_axis) == 2 else (5, 1)
    source_shape = [1, height, width, 3]
    pre_shape = [1, 3, height, width]
    squeezed_shape = [
        value for index, value in enumerate(pre_shape) if index != int(squeeze_axis)
    ]
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
            names["squeezed"]: _tensor(
                names["squeezed"],
                squeezed_shape,
                dtype=input_dtype,
                quantization=copy.deepcopy(input_quantization),
            ),
            names["unary_output"]: _tensor(
                names["unary_output"],
                squeezed_shape,
                dtype=output_dtype,
                quantization=copy.deepcopy(output_quantization),
            ),
            names["axis"]: _tensor(
                names["axis"],
                [1],
                dtype="INT64",
                data=np.asarray([squeeze_axis], dtype=np.int64),
            ),
            names["expanded"]: _tensor(
                names["expanded"],
                pre_shape,
                dtype=output_dtype,
                quantization=copy.deepcopy(output_quantization),
            ),
            names["post_perm"]: _tensor(
                names["post_perm"],
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            names["output"]: _tensor(
                names["output"],
                source_shape,
                dtype="UINT8",
                quantization=_quantization(0.5, 7),
            ),
        }
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
    model_ir.outputs.append(names["output"])
    squeeze_options = (
        {"squeezeDims": [int(squeeze_axis)]}
        if explicit_squeeze_axis
        else {"legacy": True}
    )
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
                "SQUEEZE",
                [names["pre_output"]],
                [names["squeezed"]],
                squeeze_options,
            ),
            OperatorIR(
                unary_type,
                [names["squeezed"]],
                [names["unary_output"]],
                unary_options,
                axis_semantics={"marker": "preserved"},
                version=3,
                onnx_node_name=f"{prefix}_unary",
                onnx_op_type=str(unary_type).title(),
            ),
            OperatorIR(
                "EXPAND_DIMS",
                [names["unary_output"], names["axis"]],
                [names["expanded"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["expanded"], names["post_perm"]],
                [names["output"]],
            ),
        ]
    )
    if fanout:
        model_ir.tensors[names["side_output"]] = _tensor(
            names["side_output"],
            squeezed_shape,
            dtype=output_dtype,
            quantization=copy.deepcopy(output_quantization),
        )
        model_ir.outputs.append(names["side_output"])
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                [names["unary_output"]],
                [names["side_output"]],
            )
        )
    return names


def _operators(model_ir: ModelIR, names: dict[str, str]):
    pre = next(op for op in model_ir.operators if op.outputs == [names["pre_output"]])
    squeeze = next(
        op for op in model_ir.operators if op.outputs == [names["squeezed"]]
    )
    unary = next(
        op for op in model_ir.operators if op.outputs == [names["unary_output"]]
    )
    expand = next(
        op for op in model_ir.operators if op.outputs == [names["expanded"]]
    )
    post = next(op for op in model_ir.operators if op.outputs == [names["output"]])
    return pre, squeeze, unary, expand, post


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_conv1d_unary_rewrites_multiple_chains_with_one_index(monkeypatch) -> None:
    model_ir = ModelIR("indexed_conv1d_unary")
    branches = [
        _add_branch(model_ir, "gelu"),
        _add_branch(model_ir, "produced", unary_type="RELU", produced_source=True),
        _add_branch(model_ir, "quantized", unary_type="LOGISTIC", quantized=True),
        _add_branch(model_ir, "cast", unary_type="CAST"),
        _add_branch(
            model_ir,
            "inferred",
            unary_type="ABS",
            squeeze_axis=3,
            explicit_squeeze_axis=False,
        ),
    ]
    original_unaries = [
        copy.deepcopy(_operators(model_ir, names)[2]) for names in branches
    ]
    expected_tensors = [
        copy.deepcopy(model_ir.tensors[names["unary_output"]])
        for names in branches
    ]
    refreshes = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refreshes
        refreshes += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 5}
    assert refreshes == 1
    assert [operator.op_type for operator in model_ir.operators].count("IDENTITY") == 1
    for names, original, expected_tensor in zip(
        branches,
        original_unaries,
        expected_tensors,
    ):
        unary = next(
            operator
            for operator in model_ir.operators
            if operator.outputs == [names["output"]]
        )
        output = model_ir.tensors[names["output"]]
        assert unary.op_type == original.op_type
        assert unary.inputs == [names["source"]]
        assert unary.options == original.options
        assert unary.axis_semantics == original.axis_semantics
        assert unary.version == original.version
        assert unary.onnx_node_name == original.onnx_node_name
        assert unary.onnx_op_type == original.onnx_op_type
        assert output.dtype == expected_tensor.dtype
        assert output.quantization == expected_tensor.quantization
        if expected_tensor.quantization is not None:
            assert output.quantization is not expected_tensor.quantization


def test_conv1d_unary_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_conv1d_unary")
    _add_branch(model_ir, "branch")
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("unary_type", _UNARY_TYPES)
def test_conv1d_unary_preserves_supported_unary_family(unary_type: str) -> None:
    model_ir = ModelIR(f"conv1d_{unary_type.lower()}")
    names = _add_branch(model_ir, "branch", unary_type=unary_type)

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == unary_type
    assert model_ir.operators[0].inputs == [names["source"]]
    assert model_ir.operators[0].outputs == [names["output"]]


@pytest.mark.parametrize(
    ("input_signature", "pre_signature", "squeezed_signature"),
    [
        ([-1, 1, 5, 3], [-1, 3, 1, 5], [-1, 3, 5]),
        ([1, 1, -1, 3], [1, 3, 1, -1], [1, 3, -1]),
        ([-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1]),
    ],
)
def test_conv1d_unary_preserves_consistent_dynamic_signatures(
    input_signature: list[int],
    pre_signature: list[int],
    squeezed_signature: list[int],
) -> None:
    model_ir = ModelIR("dynamic_conv1d_unary")
    names = _add_branch(model_ir, "branch")
    model_ir.tensors[names["source"]].shape_signature = input_signature
    model_ir.tensors[names["pre_output"]].shape_signature = pre_signature
    model_ir.tensors[names["squeezed"]].shape_signature = squeezed_signature
    model_ir.tensors[names["unary_output"]].shape_signature = squeezed_signature
    model_ir.tensors[names["expanded"]].shape_signature = pre_signature
    model_ir.tensors[names["output"]].shape_signature = input_signature

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    assert model_ir.tensors[names["output"]].shape_signature == input_signature


@pytest.mark.parametrize(
    "case",
    [
        "pre_arity",
        "pre_perm",
        "pre_perm_dtype",
        "pre_perm_produced",
        "pre_perm_input",
        "source_missing",
        "source_unbound",
        "source_late_producer",
        "source_boundary_and_producer",
        "source_duplicate_producer",
        "pre_output_public",
        "pre_output_fanout",
        "pre_output_duplicate",
        "pre_shape",
        "pre_signature",
        "squeeze_type",
        "squeeze_arity",
        "squeeze_axis_multiple",
        "squeeze_axis_invalid",
        "squeeze_axis_dynamic",
        "squeeze_shape",
        "squeezed_public",
        "squeezed_fanout",
        "unary_type",
        "unary_arity",
        "unary_shape",
        "unary_signature",
        "unary_public",
        "unary_fanout",
        "expand_type",
        "expand_arity",
        "expand_axis",
        "expand_axis_dtype",
        "expand_axis_produced",
        "expand_axis_input",
        "expand_shape",
        "expanded_public",
        "expanded_fanout",
        "post_arity",
        "post_perm",
        "post_perm_produced",
        "output_missing",
        "output_input",
        "output_duplicate",
        "output_shape",
        "output_signature",
        "output_backward_consumer",
        "input_dtype_mismatch",
        "output_dtype_mismatch",
        "input_quantization_mismatch",
        "output_quantization_mismatch",
        "per_axis_quantization",
    ],
)
def test_conv1d_unary_rejects_unsafe_contracts_transactionally(case: str) -> None:
    model_ir = ModelIR("rejected_conv1d_unary")
    names = _add_branch(model_ir, "branch")
    pre, squeeze, unary, expand, post = _operators(model_ir, names)

    def add_fanout(source: str, output: str, shape: list[int]) -> None:
        model_ir.tensors[output] = _tensor(output, shape)
        model_ir.outputs.append(output)
        model_ir.operators.append(OperatorIR("IDENTITY", [source], [output]))

    if case == "pre_arity":
        pre.inputs.append("extra")
    elif case == "pre_perm":
        model_ir.tensors[names["pre_perm"]].data[1] = 2
    elif case == "pre_perm_dtype":
        model_ir.tensors[names["pre_perm"]].data = np.asarray(
            [0, 3, 1, 2], dtype=np.int64
        )
    elif case == "pre_perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axis"]], [names["pre_perm"]])
        )
    elif case == "pre_perm_input":
        model_ir.inputs.append(names["pre_perm"])
    elif case == "source_missing":
        del model_ir.tensors[names["source"]]
    elif case == "source_unbound":
        model_ir.inputs.remove(names["source"])
    elif case == "source_late_producer":
        model_ir.inputs.remove(names["source"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["output"]], [names["source"]])
        )
    elif case == "source_boundary_and_producer":
        model_ir.operators.insert(
            0, OperatorIR("IDENTITY", [names["output"]], [names["source"]])
        )
    elif case == "source_duplicate_producer":
        model_ir.inputs.remove(names["source"])
        model_ir.operators.extend(
            [
                OperatorIR("IDENTITY", [names["output"]], [names["source"]]),
                OperatorIR("IDENTITY", [names["output"]], [names["source"]]),
            ]
        )
    elif case == "pre_output_public":
        model_ir.outputs.append(names["pre_output"])
    elif case == "pre_output_fanout":
        add_fanout(names["pre_output"], "side", [1, 3, 1, 5])
    elif case == "pre_output_duplicate":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["pre_output"]])
        )
    elif case == "pre_shape":
        model_ir.tensors[names["pre_output"]].shape[1] = 4
        model_ir.tensors[names["pre_output"]].shape_signature[1] = 4
    elif case == "pre_signature":
        model_ir.tensors[names["pre_output"]].shape_signature[3] = -1
    elif case == "squeeze_type":
        squeeze.op_type = "RESHAPE"
    elif case == "squeeze_arity":
        squeeze.inputs.append(names["axis"])
    elif case == "squeeze_axis_multiple":
        squeeze.options["squeezeDims"] = [0, 2]
    elif case == "squeeze_axis_invalid":
        squeeze.options["squeezeDims"] = [4]
    elif case == "squeeze_axis_dynamic":
        model_ir.tensors[names["source"]].shape_signature[1] = -1
        model_ir.tensors[names["pre_output"]].shape_signature[2] = -1
    elif case == "squeeze_shape":
        model_ir.tensors[names["squeezed"]].shape[2] = 4
        model_ir.tensors[names["squeezed"]].shape_signature[2] = 4
    elif case == "squeezed_public":
        model_ir.outputs.append(names["squeezed"])
    elif case == "squeezed_fanout":
        add_fanout(names["squeezed"], "side", [1, 3, 5])
    elif case == "unary_type":
        unary.op_type = "ADD"
    elif case == "unary_arity":
        unary.inputs.append(names["axis"])
    elif case == "unary_shape":
        model_ir.tensors[names["unary_output"]].shape[2] = 4
        model_ir.tensors[names["unary_output"]].shape_signature[2] = 4
    elif case == "unary_signature":
        model_ir.tensors[names["unary_output"]].shape_signature[2] = -1
    elif case == "unary_public":
        model_ir.outputs.append(names["unary_output"])
    elif case == "unary_fanout":
        add_fanout(names["unary_output"], "side", [1, 3, 5])
    elif case == "expand_type":
        expand.op_type = "RESHAPE"
    elif case == "expand_arity":
        expand.inputs.append("extra")
    elif case == "expand_axis":
        model_ir.tensors[names["axis"]].data[0] = 1
    elif case == "expand_axis_dtype":
        model_ir.tensors[names["axis"]].dtype = "FLOAT32"
        model_ir.tensors[names["axis"]].data = np.asarray([2.0], dtype=np.float32)
    elif case == "expand_axis_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["pre_perm"]], [names["axis"]])
        )
    elif case == "expand_axis_input":
        model_ir.inputs.append(names["axis"])
    elif case == "expand_shape":
        model_ir.tensors[names["expanded"]].shape[3] = 4
        model_ir.tensors[names["expanded"]].shape_signature[3] = 4
    elif case == "expanded_public":
        model_ir.outputs.append(names["expanded"])
    elif case == "expanded_fanout":
        add_fanout(names["expanded"], "side", [1, 3, 1, 5])
    elif case == "post_arity":
        post.inputs.append("extra")
    elif case == "post_perm":
        model_ir.tensors[names["post_perm"]].data[1] = 3
    elif case == "post_perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axis"]], [names["post_perm"]])
        )
    elif case == "output_missing":
        del model_ir.tensors[names["output"]]
    elif case == "output_input":
        model_ir.inputs.append(names["output"])
    elif case == "output_duplicate":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["output"]])
        )
    elif case == "output_shape":
        model_ir.tensors[names["output"]].shape[2] = 4
    elif case == "output_signature":
        model_ir.tensors[names["output"]].shape_signature[2] = -1
    elif case == "output_backward_consumer":
        model_ir.tensors["side"] = _tensor("side", [1, 1, 5, 3])
        model_ir.outputs.append("side")
        model_ir.operators.insert(-1, OperatorIR("IDENTITY", [names["output"]], ["side"]))
    elif case == "input_dtype_mismatch":
        model_ir.tensors[names["pre_output"]].dtype = "FLOAT16"
    elif case == "output_dtype_mismatch":
        model_ir.tensors[names["unary_output"]].dtype = "FLOAT16"
        model_ir.tensors[names["expanded"]].dtype = "FLOAT16"
    elif case == "input_quantization_mismatch":
        model_ir.tensors[names["source"]].quantization = _quantization()
    elif case == "output_quantization_mismatch":
        model_ir.tensors[names["unary_output"]].quantization = _quantization()
        model_ir.tensors[names["expanded"]].quantization = _quantization(0.5, 3)
    elif case == "per_axis_quantization":
        for key in (
            "source",
            "pre_output",
            "squeezed",
            "unary_output",
            "expanded",
        ):
            model_ir.tensors[names[key]].quantization = QuantParamIR(
                scale=[0.25, 0.5],
                zero_point=[0, 0],
                quantized_dimension=1,
            )

    before = repr(model_ir)
    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_conv1d_unary_quantization_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir = ModelIR("unclonable_conv1d_unary")
    names = _add_branch(model_ir, "branch", quantized=True)
    quantization = {
        "scale": [0.25],
        "zero_point": [3],
        "fault": Unclonable(),
    }
    for key in ("source", "pre_output", "squeezed", "unary_output", "expanded"):
        model_ir.tensors[names[key]].quantization = quantization
    before = repr(model_ir)

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_conv1d_unary_preflight_preserves_pruning_without_index(monkeypatch) -> None:
    model_ir = ModelIR("no_conv1d_unary")
    model_ir.tensors["unused"] = _tensor("unused", [1])
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(unary_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
        model_ir
    ) == {_STATS: 0}
    assert "unused" not in model_ir.tensors


def test_conv1d_unary_fanout_rewrites_multiple_chains_with_one_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("indexed_conv1d_unary_fanout")
    branches = [
        _add_branch(model_ir, "gelu", fanout=True),
        _add_branch(
            model_ir,
            "produced",
            unary_type="RELU",
            produced_source=True,
            fanout=True,
        ),
        _add_branch(
            model_ir,
            "quantized",
            unary_type="LOGISTIC",
            quantized=True,
            fanout=True,
        ),
        _add_branch(model_ir, "cast", unary_type="CAST", fanout=True),
        _add_branch(
            model_ir,
            "inferred",
            unary_type="ABS",
            squeeze_axis=3,
            explicit_squeeze_axis=False,
            fanout=True,
        ),
    ]
    original_unaries = [
        copy.deepcopy(_operators(model_ir, names)[2]) for names in branches
    ]
    expected_tensors = [
        copy.deepcopy(model_ir.tensors[names["unary_output"]])
        for names in branches
    ]
    refreshes = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refreshes
        refreshes += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    )

    assert stats == {_FANOUT_STATS: 5}
    assert refreshes == 1
    assert validate_model_ir_invariants(model_ir) == []
    for names, original, expected_tensor in zip(
        branches,
        original_unaries,
        expected_tensors,
    ):
        unary = next(
            operator
            for operator in model_ir.operators
            if operator.outputs == [names["output"]]
        )
        pre = next(
            operator
            for operator in model_ir.operators
            if operator.outputs == [names["pre_output"]]
        )
        squeeze = next(
            operator
            for operator in model_ir.operators
            if operator.outputs == [names["unary_output"]]
        )
        side = next(
            operator
            for operator in model_ir.operators
            if operator.outputs == [names["side_output"]]
        )
        assert model_ir.operators.index(unary) < model_ir.operators.index(pre)
        assert model_ir.operators.index(pre) < model_ir.operators.index(squeeze)
        assert model_ir.operators.index(squeeze) < model_ir.operators.index(side)
        assert unary.op_type == original.op_type
        assert unary.inputs == [names["source"]]
        assert unary.options == original.options
        assert unary.axis_semantics == original.axis_semantics
        assert unary.version == original.version
        assert unary.onnx_node_name == original.onnx_node_name
        assert unary.onnx_op_type == original.onnx_op_type
        assert pre.inputs[0] == names["output"]
        assert squeeze.outputs == [names["unary_output"]]
        assert side.inputs == [names["unary_output"]]
        assert model_ir.tensors[names["output"]].dtype == expected_tensor.dtype
        assert model_ir.tensors[names["pre_output"]].dtype == expected_tensor.dtype
        assert (
            model_ir.tensors[names["pre_output"]].quantization
            == expected_tensor.quantization
        )
        if expected_tensor.quantization is not None:
            assert (
                model_ir.tensors[names["pre_output"]].quantization
                is not expected_tensor.quantization
            )


def test_conv1d_unary_fanout_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_conv1d_unary_fanout")
    _add_branch(model_ir, "branch", fanout=True)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_FANOUT_STATS: 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("unary_type", _UNARY_TYPES)
def test_conv1d_unary_fanout_preserves_supported_unary_family(
    unary_type: str,
) -> None:
    model_ir = ModelIR(f"conv1d_fanout_{unary_type.lower()}")
    names = _add_branch(
        model_ir,
        "branch",
        unary_type=unary_type,
        fanout=True,
    )

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    )

    assert stats == {_FANOUT_STATS: 1}
    assert model_ir.operators[0].op_type == unary_type
    assert model_ir.operators[0].inputs == [names["source"]]
    assert model_ir.operators[0].outputs == [names["output"]]
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    ("source_signature", "pre_signature", "squeezed_signature"),
    [
        ([-1, 1, 5, 3], [-1, 3, 1, 5], [-1, 3, 5]),
        ([1, 1, -1, 3], [1, 3, 1, -1], [1, 3, -1]),
        ([-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1]),
    ],
)
def test_conv1d_unary_fanout_preserves_consistent_dynamic_signatures(
    source_signature: list[int],
    pre_signature: list[int],
    squeezed_signature: list[int],
) -> None:
    model_ir = ModelIR("dynamic_conv1d_unary_fanout")
    names = _add_branch(model_ir, "branch", fanout=True)
    model_ir.tensors[names["source"]].shape_signature = source_signature
    model_ir.tensors[names["pre_output"]].shape_signature = pre_signature
    model_ir.tensors[names["squeezed"]].shape_signature = squeezed_signature
    model_ir.tensors[names["unary_output"]].shape_signature = squeezed_signature
    model_ir.tensors[names["expanded"]].shape_signature = pre_signature
    model_ir.tensors[names["output"]].shape_signature = source_signature
    model_ir.tensors[names["side_output"]].shape_signature = squeezed_signature

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    )

    assert stats == {_FANOUT_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []


def test_conv1d_unary_fanout_preserves_public_nchw_output() -> None:
    model_ir = ModelIR("public_conv1d_unary_fanout")
    names = _add_branch(model_ir, "branch", fanout=True)
    side = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == [names["side_output"]]
    )
    model_ir.operators.remove(side)
    model_ir.outputs.remove(names["side_output"])
    del model_ir.tensors[names["side_output"]]
    model_ir.outputs.append(names["unary_output"])

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    )

    assert stats == {_FANOUT_STATS: 1}
    assert names["unary_output"] in model_ir.outputs
    assert validate_model_ir_invariants(model_ir) == []


def test_conv1d_unary_fanout_bypasses_only_first_eligible_nhwc_branch() -> None:
    model_ir = ModelIR("multiple_nhwc_conv1d_unary_fanout")
    names = _add_branch(model_ir, "branch", fanout=True)
    output_dtype = model_ir.tensors[names["unary_output"]].dtype
    output_quantization = copy.deepcopy(
        model_ir.tensors[names["unary_output"]].quantization
    )
    model_ir.tensors["expanded2"] = _tensor(
        "expanded2",
        list(model_ir.tensors[names["pre_output"]].shape),
        dtype=output_dtype,
        quantization=copy.deepcopy(output_quantization),
    )
    model_ir.tensors["output2"] = _tensor(
        "output2",
        list(model_ir.tensors[names["source"]].shape),
        dtype=output_dtype,
        quantization=copy.deepcopy(output_quantization),
    )
    model_ir.outputs.append("output2")
    second_expand = OperatorIR(
        "EXPAND_DIMS",
        [names["unary_output"], names["axis"]],
        ["expanded2"],
    )
    second_post = OperatorIR(
        "TRANSPOSE",
        ["expanded2", names["post_perm"]],
        ["output2"],
    )
    model_ir.operators.extend([second_expand, second_post])

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    )

    assert stats == {_FANOUT_STATS: 1}
    assert second_expand in model_ir.operators
    assert second_post in model_ir.operators
    assert second_expand.inputs[0] == names["unary_output"]
    assert validate_model_ir_invariants(model_ir) == []


def test_conv1d_unary_fanout_accepts_equivalent_negative_axes() -> None:
    model_ir = ModelIR("negative_axis_conv1d_unary_fanout")
    names = _add_branch(model_ir, "branch", fanout=True)
    _, squeeze, _, _, _ = _operators(model_ir, names)
    squeeze.options["squeezeDims"] = [-2]
    model_ir.tensors[names["axis"]].data = np.asarray([-2], dtype=np.int64)

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    )

    assert stats == {_FANOUT_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "case",
    [
        "no_fanout",
        "pre_arity",
        "floating_pre_perm",
        "produced_pre_perm",
        "source_late_producer",
        "pre_public",
        "pre_fanout",
        "pre_shape",
        "squeezed_public",
        "squeezed_shape",
        "unary_arity",
        "unary_signature",
        "backward_side_consumer",
        "expand_arity",
        "expand_axis",
        "produced_axis",
        "expand_public",
        "expand_shape",
        "post_arity",
        "floating_post_perm",
        "post_output_input",
        "post_output_duplicate",
        "post_output_signature",
        "post_output_backward_consumer",
        "input_dtype_mismatch",
        "output_quantization_mismatch",
        "per_axis_quantization",
    ],
)
def test_conv1d_unary_fanout_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = ModelIR("rejected_conv1d_unary_fanout")
    names = _add_branch(model_ir, "branch", fanout=True)
    pre, squeeze, unary, expand, post = _operators(model_ir, names)
    side = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == [names["side_output"]]
    )

    def add_fanout(source: str, output: str, shape: list[int]) -> None:
        model_ir.tensors[output] = _tensor(output, shape)
        model_ir.outputs.append(output)
        model_ir.operators.append(OperatorIR("IDENTITY", [source], [output]))

    if case == "no_fanout":
        model_ir.operators.remove(side)
        model_ir.outputs.remove(names["side_output"])
        del model_ir.tensors[names["side_output"]]
    elif case == "pre_arity":
        pre.inputs.append(names["axis"])
    elif case == "floating_pre_perm":
        model_ir.tensors[names["pre_perm"]].dtype = "FLOAT32"
        model_ir.tensors[names["pre_perm"]].data = np.asarray(
            [0, 3, 1, 2], dtype=np.float32
        )
    elif case == "produced_pre_perm":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axis"]], [names["pre_perm"]])
        )
    elif case == "source_late_producer":
        model_ir.inputs.remove(names["source"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["output"]], [names["source"]])
        )
    elif case == "pre_public":
        model_ir.outputs.append(names["pre_output"])
    elif case == "pre_fanout":
        add_fanout(names["pre_output"], "extra", [1, 3, 1, 5])
    elif case == "pre_shape":
        model_ir.tensors[names["pre_output"]].shape[1] = 4
        model_ir.tensors[names["pre_output"]].shape_signature[1] = 4
    elif case == "squeezed_public":
        model_ir.outputs.append(names["squeezed"])
    elif case == "squeezed_shape":
        model_ir.tensors[names["squeezed"]].shape[2] = 4
        model_ir.tensors[names["squeezed"]].shape_signature[2] = 4
    elif case == "unary_arity":
        unary.inputs.append(names["axis"])
    elif case == "unary_signature":
        model_ir.tensors[names["unary_output"]].shape_signature[2] = -1
    elif case == "backward_side_consumer":
        model_ir.operators.remove(side)
        model_ir.operators.insert(model_ir.operators.index(unary), side)
    elif case == "expand_arity":
        expand.inputs.append(names["pre_perm"])
    elif case == "expand_axis":
        model_ir.tensors[names["axis"]].data[0] = 1
    elif case == "produced_axis":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["pre_perm"]], [names["axis"]])
        )
    elif case == "expand_public":
        model_ir.outputs.append(names["expanded"])
    elif case == "expand_shape":
        model_ir.tensors[names["expanded"]].shape[1] = 4
        model_ir.tensors[names["expanded"]].shape_signature[1] = 4
    elif case == "post_arity":
        post.inputs.append(names["axis"])
    elif case == "floating_post_perm":
        model_ir.tensors[names["post_perm"]].dtype = "FLOAT32"
        model_ir.tensors[names["post_perm"]].data = np.asarray(
            [0, 2, 3, 1], dtype=np.float32
        )
    elif case == "post_output_input":
        model_ir.inputs.append(names["output"])
    elif case == "post_output_duplicate":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["output"]])
        )
    elif case == "post_output_signature":
        model_ir.tensors[names["output"]].shape_signature[2] = -1
    elif case == "post_output_backward_consumer":
        model_ir.tensors["extra"] = _tensor("extra", [1, 1, 5, 3])
        model_ir.outputs.append("extra")
        model_ir.operators.insert(
            model_ir.operators.index(post),
            OperatorIR("IDENTITY", [names["output"]], ["extra"]),
        )
    elif case == "input_dtype_mismatch":
        model_ir.tensors[names["pre_output"]].dtype = "FLOAT16"
    elif case == "output_quantization_mismatch":
        model_ir.tensors[names["unary_output"]].quantization = _quantization()
        model_ir.tensors[names["expanded"]].quantization = _quantization(0.5, 3)
    elif case == "per_axis_quantization":
        for key in (
            "source",
            "pre_output",
            "squeezed",
            "unary_output",
            "expanded",
        ):
            model_ir.tensors[names[key]].quantization = QuantParamIR(
                scale=[0.25, 0.5],
                zero_point=[0, 0],
                quantized_dimension=1,
            )

    before = repr(model_ir)
    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    )

    assert stats == {_FANOUT_STATS: 0}
    assert repr(model_ir) == before


def test_conv1d_unary_fanout_quantization_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir = ModelIR("unclonable_conv1d_unary_fanout")
    names = _add_branch(model_ir, "branch", quantized=True, fanout=True)
    quantization = {
        "scale": [0.25],
        "zero_point": [3],
        "fault": Unclonable(),
    }
    for key in ("source", "pre_output", "squeezed", "unary_output", "expanded"):
        model_ir.tensors[names[key]].quantization = quantization
    before = repr(model_ir)

    stats = _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    )

    assert stats == {_FANOUT_STATS: 0}
    assert repr(model_ir) == before


def test_conv1d_unary_fanout_preflight_preserves_pruning_without_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_conv1d_unary_fanout")
    model_ir.tensors["unused"] = _tensor("unused", [1])
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(unary_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
        model_ir
    ) == {_FANOUT_STATS: 0}
    assert "unused" not in model_ir.tensors
