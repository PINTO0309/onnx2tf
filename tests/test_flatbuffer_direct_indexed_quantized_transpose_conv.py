from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.quantized_transpose_conv as transpose_conv_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.quantized_transpose_conv import (
    _optimize_dequant_transposeconv_quantize_chains,
)


def _tensor(
    name: str,
    *,
    dtype: str,
    shape: list[int],
    signature: list[int],
    data: np.ndarray | None = None,
    quantization: QuantParamIR | dict | None = None,
    is_variable: bool = True,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(signature),
        data=data,
        is_variable=is_variable,
        quantization=quantization,
    )


def _activation_grid(scale: float, zero_point: int) -> QuantParamIR:
    return QuantParamIR(
        scale=[scale],
        zero_point=[zero_point],
        quantized_dimension=0,
    )


def _quantized_transpose_conv_model(*, branches: int = 2) -> ModelIR:
    model_ir = ModelIR("indexed_quantized_transpose_conv")
    weight_shape = [3, 3, 3, 2]
    float_weight = np.linspace(
        -0.5,
        0.5,
        num=int(np.prod(weight_shape)),
        dtype=np.float32,
    ).reshape(weight_shape)
    int8_weight = np.arange(
        int(np.prod(weight_shape)),
        dtype=np.int16,
    ).reshape(weight_shape)
    int8_weight = ((int8_weight % 31) - 15).astype(np.int8)

    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        names = {
            "quantized_input": f"{prefix}_quantized_input",
            "float_input": f"{prefix}_float_input",
            "output_shape": f"{prefix}_output_shape",
            "weight": f"{prefix}_weight",
            "float_output": f"{prefix}_float_output",
            "quantized_output": f"{prefix}_quantized_output",
        }
        model_ir.inputs.append(names["quantized_input"])
        model_ir.outputs.append(names["quantized_output"])
        model_ir.tensors.update(
            {
                names["quantized_input"]: _tensor(
                    names["quantized_input"],
                    dtype="INT8",
                    shape=[1, 2, 2, 2],
                    signature=[-1, 2, 2, 2],
                    quantization=_activation_grid(0.1, 0),
                ),
                names["float_input"]: _tensor(
                    names["float_input"],
                    dtype="FLOAT32",
                    shape=[1, 2, 2, 2],
                    signature=[-1, 2, 2, 2],
                ),
                names["output_shape"]: _tensor(
                    names["output_shape"],
                    dtype="INT32",
                    shape=[4],
                    signature=[4],
                    data=np.asarray([1, 4, 4, 3], dtype=np.int32),
                    is_variable=False,
                ),
                names["weight"]: _tensor(
                    names["weight"],
                    dtype="FLOAT32" if branch_index == 0 else "INT8",
                    shape=weight_shape,
                    signature=weight_shape,
                    data=(
                        np.asarray(float_weight)
                        if branch_index == 0
                        else np.asarray(int8_weight)
                    ),
                    quantization=(
                        None if branch_index == 0 else _activation_grid(0.05, 0)
                    ),
                    is_variable=False,
                ),
                names["float_output"]: _tensor(
                    names["float_output"],
                    dtype="FLOAT32",
                    shape=[1, 4, 4, 3],
                    signature=[-1, 4, 4, 3],
                ),
                names["quantized_output"]: _tensor(
                    names["quantized_output"],
                    dtype="INT8",
                    shape=[1, 4, 4, 3],
                    signature=[-1, 4, 4, 3],
                    quantization=_activation_grid(0.2, -5),
                ),
            }
        )
        model_ir.operators.extend(
            [
                OperatorIR(
                    "DEQUANTIZE",
                    [names["quantized_input"]],
                    [names["float_input"]],
                ),
                OperatorIR(
                    "TRANSPOSE_CONV",
                    [
                        names["output_shape"],
                        names["weight"],
                        names["float_input"],
                    ],
                    [names["float_output"]],
                    options={
                        "padding": "SAME",
                        "strideW": 2,
                        "strideH": 2,
                        "fusedActivationFunction": "NONE",
                    },
                    axis_semantics={"kernel": "OHWI"},
                    version=1 if branch_index == 0 else 4,
                    onnx_node_name=f"{prefix}_transpose_conv",
                    onnx_op_type="ConvTranspose",
                ),
                OperatorIR(
                    "QUANTIZE",
                    [names["float_output"]],
                    [names["quantized_output"]],
                ),
            ]
        )
    return model_ir


def _assert_index_current(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def _share_branch0_weight(model_ir: ModelIR) -> None:
    model_ir.tensors["weight_side"] = _tensor(
        "weight_side",
        dtype="FLOAT32",
        shape=[3, 3, 3, 2],
        signature=[3, 3, 3, 2],
    )
    model_ir.outputs.append("weight_side")
    model_ir.operators.append(
        OperatorIR("IDENTITY", ["branch0_weight"], ["weight_side"])
    )


def test_quantized_transpose_conv_folds_float_and_int8_weights_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _quantized_transpose_conv_model()
    original_int8_weight = copy.deepcopy(model_ir.tensors["branch1_weight"])
    output_quantizations = [
        model_ir.tensors[f"branch{index}_quantized_output"].quantization
        for index in range(2)
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_dequant_transposeconv_quantize_chains(model_ir)

    assert stats == {"folded_dequant_transposeconv_quantize_chains": 2}
    assert refresh_count == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE_CONV",
        "TRANSPOSE_CONV",
    ]
    for branch_index, transpose_conv in enumerate(model_ir.operators):
        prefix = f"branch{branch_index}"
        assert transpose_conv.inputs == [
            f"{prefix}_output_shape",
            f"{prefix}_weight",
            f"{prefix}_quantized_input",
        ]
        assert transpose_conv.outputs == [f"{prefix}_quantized_output"]
        assert transpose_conv.options == {
            "padding": "SAME",
            "strideW": 2,
            "strideH": 2,
            "fusedActivationFunction": "NONE",
        }
        assert transpose_conv.axis_semantics == {"kernel": "OHWI"}
        assert transpose_conv.version == (3 if branch_index == 0 else 4)
        assert transpose_conv.onnx_node_name == f"{prefix}_transpose_conv"
        assert transpose_conv.onnx_op_type == "ConvTranspose"
        output = model_ir.tensors[f"{prefix}_quantized_output"]
        assert output.quantization == output_quantizations[branch_index]
        assert output.quantization is not output_quantizations[branch_index]
    float_weight = model_ir.tensors["branch0_weight"]
    assert float_weight.dtype == "INT8"
    assert np.asarray(float_weight.data).dtype == np.int8
    assert float_weight.quantization is not None
    int8_weight = model_ir.tensors["branch1_weight"]
    assert int8_weight.dtype == original_int8_weight.dtype
    assert int8_weight.quantization is original_int8_weight.quantization or (
        int8_weight.quantization == original_int8_weight.quantization
    )
    np.testing.assert_array_equal(int8_weight.data, original_int8_weight.data)


def test_quantized_transpose_conv_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _quantized_transpose_conv_model(branches=1)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_dequant_transposeconv_quantize_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"folded_dequant_transposeconv_quantize_chains": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_quantized_transpose_conv_clones_shared_float_weight() -> None:
    model_ir = _quantized_transpose_conv_model(branches=1)
    _share_branch0_weight(model_ir)
    original_weight = copy.deepcopy(model_ir.tensors["branch0_weight"])

    stats = _optimize_dequant_transposeconv_quantize_chains(model_ir)

    assert stats == {"folded_dequant_transposeconv_quantize_chains": 1}
    retained = model_ir.tensors["branch0_weight"]
    assert retained.dtype == "FLOAT32"
    np.testing.assert_array_equal(retained.data, original_weight.data)
    cloned = model_ir.tensors["branch0_weight_q"]
    assert cloned.dtype == "INT8"
    assert np.asarray(cloned.data).dtype == np.int8
    assert model_ir.operators[0].inputs[1] == "branch0_weight_q"


def test_quantized_transpose_conv_clones_public_float_weight() -> None:
    model_ir = _quantized_transpose_conv_model(branches=1)
    model_ir.outputs.append("branch0_weight")
    original_weight = copy.deepcopy(model_ir.tensors["branch0_weight"])

    stats = _optimize_dequant_transposeconv_quantize_chains(model_ir)

    assert stats == {"folded_dequant_transposeconv_quantize_chains": 1}
    retained = model_ir.tensors["branch0_weight"]
    assert retained.dtype == "FLOAT32"
    np.testing.assert_array_equal(retained.data, original_weight.data)
    assert model_ir.tensors["branch0_weight_q"].dtype == "INT8"


def test_quantized_transpose_conv_clone_failure_is_transactional(
    monkeypatch,
) -> None:
    model_ir = _quantized_transpose_conv_model(branches=1)
    before = repr(model_ir)

    def injected_failure(quantization):
        raise RuntimeError("injected output quantization clone failure")

    monkeypatch.setattr(
        transpose_conv_module,
        "_clone_quantization",
        injected_failure,
    )

    stats = _optimize_dequant_transposeconv_quantize_chains(model_ir)

    assert stats == {"folded_dequant_transposeconv_quantize_chains": 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize(
    "case",
    [
        "float_input_public_output",
        "float_input_public_input",
        "float_output_public_output",
        "float_output_public_input",
        "quantized_output_public_input",
        "float_input_fanout",
        "float_output_fanout",
        "duplicate_float_input_producer",
        "duplicate_float_output_producer",
        "duplicate_quantized_output_producer",
        "reverse_dequantize_transpose_conv_order",
        "reverse_transpose_conv_quantize_order",
        "wrong_dequantize_type",
        "wrong_transpose_conv_type",
        "wrong_quantize_type",
        "wrong_dequantize_arity",
        "wrong_transpose_conv_arity",
        "wrong_quantize_arity",
        "wrong_data_input_slot",
        "duplicate_input_roles",
        "missing_quantized_input",
        "missing_float_input",
        "missing_output_shape",
        "missing_weight",
        "missing_float_output",
        "missing_quantized_output",
        "unsupported_input_dtype",
        "unsupported_output_dtype",
        "missing_input_quantization",
        "missing_output_quantization",
        "per_axis_input_quantization",
        "per_axis_output_quantization",
        "negative_input_scale",
        "nonfinite_output_scale",
        "input_zero_point_out_of_range",
        "nonfloat_input_bridge",
        "float_dtype_mismatch",
        "input_shape_mismatch",
        "input_signature_mismatch",
        "output_shape_mismatch",
        "output_signature_mismatch",
        "invalid_input_shape",
        "invalid_output_signature",
        "weight_not_array",
        "weight_rank_three",
        "weight_shape_mismatch",
        "weight_signature_mismatch",
        "unsupported_weight_dtype",
        "nonfinite_float_weight",
        "produced_weight",
        "int8_weight_array_dtype_mismatch",
        "int8_weight_missing_quantization",
        "int8_weight_negative_scale",
        "int8_weight_zero_point_out_of_range",
    ],
)
def test_quantized_transpose_conv_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = _quantized_transpose_conv_model(branches=1)
    dequantize, transpose_conv, quantize = model_ir.operators
    quantized_input = model_ir.tensors["branch0_quantized_input"]
    float_input = model_ir.tensors["branch0_float_input"]
    weight = model_ir.tensors["branch0_weight"]
    float_output = model_ir.tensors["branch0_float_output"]
    quantized_output = model_ir.tensors["branch0_quantized_output"]

    if case == "float_input_public_output":
        model_ir.outputs.append("branch0_float_input")
    elif case == "float_input_public_input":
        model_ir.inputs.append("branch0_float_input")
    elif case == "float_output_public_output":
        model_ir.outputs.append("branch0_float_output")
    elif case == "float_output_public_input":
        model_ir.inputs.append("branch0_float_output")
    elif case == "quantized_output_public_input":
        model_ir.inputs.append("branch0_quantized_output")
    elif case in {"float_input_fanout", "float_output_fanout"}:
        source = (
            "branch0_float_input"
            if case == "float_input_fanout"
            else "branch0_float_output"
        )
        shape = [1, 2, 2, 2] if case == "float_input_fanout" else [1, 4, 4, 3]
        model_ir.tensors["side"] = _tensor(
            "side",
            dtype="FLOAT32",
            shape=shape,
            signature=shape,
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    elif case == "duplicate_float_input_producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_quantized_input"],
                ["branch0_float_input"],
            )
        )
    elif case == "duplicate_float_output_producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_float_input"],
                ["branch0_float_output"],
            )
        )
    elif case == "duplicate_quantized_output_producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_quantized_input"],
                ["branch0_quantized_output"],
            )
        )
    elif case == "reverse_dequantize_transpose_conv_order":
        model_ir.operators = [transpose_conv, dequantize, quantize]
    elif case == "reverse_transpose_conv_quantize_order":
        model_ir.operators = [dequantize, quantize, transpose_conv]
    elif case == "wrong_dequantize_type":
        dequantize.op_type = "CAST"
    elif case == "wrong_transpose_conv_type":
        transpose_conv.op_type = "CONV_2D"
    elif case == "wrong_quantize_type":
        quantize.op_type = "CAST"
    elif case == "wrong_dequantize_arity":
        dequantize.inputs.append("branch0_quantized_input")
    elif case == "wrong_transpose_conv_arity":
        transpose_conv.inputs.append("branch0_float_input")
    elif case == "wrong_quantize_arity":
        quantize.inputs.append("branch0_float_output")
    elif case == "wrong_data_input_slot":
        transpose_conv.inputs[2] = "branch0_weight"
    elif case == "duplicate_input_roles":
        transpose_conv.inputs[0] = "branch0_weight"
    elif case.startswith("missing_") and case not in {
        "missing_input_quantization",
        "missing_output_quantization",
    }:
        tensor_name = {
            "missing_quantized_input": "branch0_quantized_input",
            "missing_float_input": "branch0_float_input",
            "missing_output_shape": "branch0_output_shape",
            "missing_weight": "branch0_weight",
            "missing_float_output": "branch0_float_output",
            "missing_quantized_output": "branch0_quantized_output",
        }[case]
        del model_ir.tensors[tensor_name]
    elif case == "unsupported_input_dtype":
        quantized_input.dtype = "UINT8"
    elif case == "unsupported_output_dtype":
        quantized_output.dtype = "UINT8"
    elif case == "missing_input_quantization":
        quantized_input.quantization = None
    elif case == "missing_output_quantization":
        quantized_output.quantization = None
    elif case == "per_axis_input_quantization":
        quantized_input.quantization = QuantParamIR([0.1, 0.2], [0, 0], 3)
    elif case == "per_axis_output_quantization":
        quantized_output.quantization = QuantParamIR([0.2, 0.3], [-5, -5], 3)
    elif case == "negative_input_scale":
        quantized_input.quantization.scale = [-0.1]
    elif case == "nonfinite_output_scale":
        quantized_output.quantization.scale = [np.inf]
    elif case == "input_zero_point_out_of_range":
        quantized_input.quantization.zero_point = [-129]
    elif case == "nonfloat_input_bridge":
        float_input.dtype = "INT32"
    elif case == "float_dtype_mismatch":
        float_output.dtype = "FLOAT16"
    elif case == "input_shape_mismatch":
        float_input.shape = [1, 2, 1, 4]
    elif case == "input_signature_mismatch":
        float_input.shape_signature = [1, 2, 2, 2]
    elif case == "output_shape_mismatch":
        quantized_output.shape = [1, 4, 2, 6]
    elif case == "output_signature_mismatch":
        quantized_output.shape_signature = [1, 4, 4, 3]
    elif case == "invalid_input_shape":
        float_input.shape = [1, None, 2, 2]
    elif case == "invalid_output_signature":
        float_output.shape_signature = [-1, 4, None, 3]
    elif case == "weight_not_array":
        weight.data = [0.0]
    elif case == "weight_rank_three":
        weight.data = np.zeros([3, 3, 2], dtype=np.float32)
        weight.shape = [3, 3, 2]
        weight.shape_signature = [3, 3, 2]
    elif case == "weight_shape_mismatch":
        weight.shape = [3, 3, 2, 3]
    elif case == "weight_signature_mismatch":
        weight.shape_signature = [-1, 3, 3, 2]
    elif case == "unsupported_weight_dtype":
        weight.dtype = "INT16"
    elif case == "nonfinite_float_weight":
        weight.data = np.asarray(weight.data)
        weight.data.reshape(-1)[0] = np.inf
    elif case == "produced_weight":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_quantized_input"],
                ["branch0_weight"],
            )
        )
    elif case.startswith("int8_weight_"):
        weight.dtype = "INT8"
        if case == "int8_weight_array_dtype_mismatch":
            weight.data = np.asarray(weight.data, dtype=np.float32)
            weight.quantization = _activation_grid(0.05, 0)
        else:
            weight.data = np.asarray(np.round(weight.data), dtype=np.int8)
            weight.quantization = _activation_grid(0.05, 0)
            if case == "int8_weight_missing_quantization":
                weight.quantization = None
            elif case == "int8_weight_negative_scale":
                weight.quantization.scale = [-0.05]
            elif case == "int8_weight_zero_point_out_of_range":
                weight.quantization.zero_point = [-129]

    before = repr(model_ir)
    stats = _optimize_dequant_transposeconv_quantize_chains(model_ir)

    assert stats == {"folded_dequant_transposeconv_quantize_chains": 0}
    assert repr(model_ir) == before


def test_quantized_transpose_conv_skips_index_without_complete_family_and_prunes(
    monkeypatch,
) -> None:
    model_ir = ModelIR("quantized_transpose_conv_without_chain")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor(
            "x",
            dtype="FLOAT32",
            shape=[1, 2],
            signature=[1, 2],
        ),
        "y": _tensor(
            "y",
            dtype="FLOAT32",
            shape=[1, 2],
            signature=[1, 2],
        ),
        "unused": _tensor(
            "unused",
            dtype="FLOAT32",
            shape=[1, 2],
            signature=[1, 2],
        ),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]
    layout_state = LayoutState.from_model_ir(model_ir)

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(
        transpose_conv_module,
        "ModelIRGraphIndex",
        unexpected_index,
    )

    stats = _optimize_dequant_transposeconv_quantize_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"folded_dequant_transposeconv_quantize_chains": 0}
    assert set(model_ir.tensors) == {"x", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
