from __future__ import annotations

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.quantized_logistic as logistic_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.quantized_logistic import (
    _optimize_dequant_logistic_quantize_chains,
)


def _tensor(
    name: str,
    *,
    dtype: str,
    shape: list[int],
    signature: list[int],
    quantization: QuantParamIR | dict | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(signature),
        is_variable=True,
        quantization=quantization,
    )


def _input_grid(dtype: str, branch_index: int = 0) -> QuantParamIR:
    zero_point = -3 - int(branch_index) if dtype == "INT8" else 7 + int(branch_index)
    return QuantParamIR(
        scale=[0.125 * (branch_index + 1)],
        zero_point=[zero_point],
        quantized_dimension=0,
    )


def _output_grid(dtype: str) -> QuantParamIR:
    return QuantParamIR(
        scale=[1.0 / 256.0],
        zero_point=[-128 if dtype == "INT8" else 0],
        quantized_dimension=0,
    )


def _quantized_logistic_model(
    *,
    branches: int = 2,
    shape: list[int] | None = None,
    signature: list[int] | None = None,
) -> ModelIR:
    tensor_shape = [1, 4, 4, 2] if shape is None else list(shape)
    tensor_signature = [-1, 4, 4, 2] if signature is None else list(signature)
    model_ir = ModelIR("indexed_quantized_logistic")
    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        dtype = "INT8" if branch_index % 2 == 0 else "UINT8"
        quantized_input = f"{prefix}_quantized_input"
        float_input = f"{prefix}_float_input"
        float_output = f"{prefix}_float_output"
        quantized_output = f"{prefix}_quantized_output"
        model_ir.inputs.append(quantized_input)
        model_ir.outputs.append(quantized_output)
        model_ir.tensors.update(
            {
                quantized_input: _tensor(
                    quantized_input,
                    dtype=dtype,
                    shape=tensor_shape,
                    signature=tensor_signature,
                    quantization=_input_grid(dtype, branch_index),
                ),
                float_input: _tensor(
                    float_input,
                    dtype="FLOAT32",
                    shape=tensor_shape,
                    signature=tensor_signature,
                ),
                float_output: _tensor(
                    float_output,
                    dtype="FLOAT32",
                    shape=tensor_shape,
                    signature=tensor_signature,
                ),
                quantized_output: _tensor(
                    quantized_output,
                    dtype=dtype,
                    shape=tensor_shape,
                    signature=tensor_signature,
                    quantization=_output_grid(dtype),
                ),
            }
        )
        model_ir.operators.extend(
            [
                OperatorIR(
                    "DEQUANTIZE",
                    [quantized_input],
                    [float_input],
                ),
                OperatorIR(
                    "LOGISTIC",
                    [float_input],
                    [float_output],
                    options={"fusedActivationFunction": "NONE"},
                    axis_semantics={"elementwise": True},
                    version=7,
                    onnx_node_name=f"{prefix}_logistic",
                    onnx_op_type="Sigmoid",
                ),
                OperatorIR(
                    "QUANTIZE",
                    [float_output],
                    [quantized_output],
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


def test_quantized_logistic_folds_multiple_dtypes_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _quantized_logistic_model()
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

    stats = _optimize_dequant_logistic_quantize_chains(model_ir)

    assert stats == {"folded_dequant_logistic_quantize_chains": 2}
    assert refresh_count == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "LOGISTIC",
        "LOGISTIC",
    ]
    for branch_index, logistic in enumerate(model_ir.operators):
        prefix = f"branch{branch_index}"
        expected_version = 2 if branch_index == 0 else 1
        assert logistic.inputs == [f"{prefix}_quantized_input"]
        assert logistic.outputs == [f"{prefix}_quantized_output"]
        assert logistic.options == {"fusedActivationFunction": "NONE"}
        assert logistic.axis_semantics == {"elementwise": True}
        assert logistic.version == expected_version
        assert logistic.onnx_node_name == f"{prefix}_logistic"
        assert logistic.onnx_op_type == "Sigmoid"
        output = model_ir.tensors[f"{prefix}_quantized_output"]
        assert output.quantization is output_quantizations[branch_index]
        assert output.shape == [1, 4, 4, 2]
        assert output.shape_signature == [-1, 4, 4, 2]
    assert set(model_ir.tensors) == {
        "branch0_quantized_input",
        "branch0_quantized_output",
        "branch1_quantized_input",
        "branch1_quantized_output",
    }


def test_quantized_logistic_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _quantized_logistic_model(branches=1)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_dequant_logistic_quantize_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"folded_dequant_logistic_quantize_chains": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert set(layout_state.logical) == {
        "branch0_quantized_input",
        "branch0_quantized_output",
    }


def test_quantized_logistic_preserves_quantized_input_and_output_fanout() -> None:
    model_ir = _quantized_logistic_model(branches=1)
    model_ir.tensors["input_side"] = _tensor(
        "input_side",
        dtype="INT8",
        shape=[1, 4, 4, 2],
        signature=[-1, 4, 4, 2],
        quantization=_input_grid("INT8"),
    )
    model_ir.tensors["output_side"] = _tensor(
        "output_side",
        dtype="INT8",
        shape=[1, 4, 4, 2],
        signature=[-1, 4, 4, 2],
        quantization=_output_grid("INT8"),
    )
    model_ir.outputs.extend(["input_side", "output_side"])
    model_ir.operators.extend(
        [
            OperatorIR(
                "IDENTITY",
                ["branch0_quantized_input"],
                ["input_side"],
            ),
            OperatorIR(
                "IDENTITY",
                ["branch0_quantized_output"],
                ["output_side"],
            ),
        ]
    )

    stats = _optimize_dequant_logistic_quantize_chains(model_ir)

    assert stats == {"folded_dequant_logistic_quantize_chains": 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "LOGISTIC",
        "IDENTITY",
        "IDENTITY",
    ]


def test_quantized_logistic_accepts_dict_grids_and_rank_one() -> None:
    model_ir = _quantized_logistic_model(
        branches=1,
        shape=[7],
        signature=[-1],
    )
    model_ir.tensors["branch0_quantized_input"].quantization = {
        "scale": np.asarray([0.125], dtype=np.float32),
        "zero_point": np.asarray([-3], dtype=np.int64),
        "quantized_dimension": 0,
    }
    model_ir.tensors["branch0_quantized_output"].quantization = {
        "scale": np.asarray([1.0 / 256.0], dtype=np.float32),
        "zero_point": np.asarray([-128], dtype=np.int64),
        "quantized_dimension": 0,
    }

    stats = _optimize_dequant_logistic_quantize_chains(model_ir)

    assert stats == {"folded_dequant_logistic_quantize_chains": 1}
    assert model_ir.tensors["branch0_quantized_output"].shape == [7]
    assert model_ir.tensors["branch0_quantized_output"].shape_signature == [-1]


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
        "reverse_dequantize_logistic_order",
        "reverse_logistic_quantize_order",
        "wrong_dequantize_type",
        "extra_dequantize_input",
        "extra_dequantize_output",
        "wrong_logistic_type",
        "extra_logistic_input",
        "extra_logistic_output",
        "wrong_quantize_type",
        "extra_quantize_input",
        "extra_quantize_output",
        "missing_quantized_input",
        "missing_float_input",
        "missing_float_output",
        "missing_quantized_output",
        "unsupported_quantized_dtype",
        "quantized_dtype_mismatch",
        "missing_input_quantization",
        "missing_output_quantization",
        "per_axis_input_quantization",
        "per_axis_output_quantization",
        "noncanonical_output_scale",
        "near_canonical_output_scale",
        "wrong_output_zero_point",
        "negative_input_scale",
        "nonfinite_input_scale",
        "input_zero_point_out_of_range",
        "nonfloat_intermediate",
        "float_dtype_mismatch",
        "quantized_input_shape_mismatch",
        "float_input_signature_mismatch",
        "logistic_output_shape_mismatch",
        "logistic_output_signature_mismatch",
        "quantized_output_shape_mismatch",
        "quantized_output_signature_mismatch",
        "invalid_shape",
        "invalid_signature",
    ],
)
def test_quantized_logistic_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = _quantized_logistic_model(branches=1)
    dequantize, logistic, quantize = model_ir.operators
    quantized_input = model_ir.tensors["branch0_quantized_input"]
    float_input = model_ir.tensors["branch0_float_input"]
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
        model_ir.tensors["side"] = _tensor(
            "side",
            dtype="FLOAT32",
            shape=[1, 4, 4, 2],
            signature=[-1, 4, 4, 2],
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
    elif case == "reverse_dequantize_logistic_order":
        model_ir.operators = [logistic, dequantize, quantize]
    elif case == "reverse_logistic_quantize_order":
        model_ir.operators = [dequantize, quantize, logistic]
    elif case == "wrong_dequantize_type":
        dequantize.op_type = "CAST"
    elif case == "extra_dequantize_input":
        dequantize.inputs.append("branch0_quantized_input")
    elif case == "extra_dequantize_output":
        model_ir.tensors["extra"] = _tensor(
            "extra",
            dtype="FLOAT32",
            shape=[1, 4, 4, 2],
            signature=[-1, 4, 4, 2],
        )
        dequantize.outputs.append("extra")
    elif case == "wrong_logistic_type":
        logistic.op_type = "TANH"
    elif case == "extra_logistic_input":
        logistic.inputs.append("branch0_float_input")
    elif case == "extra_logistic_output":
        model_ir.tensors["extra"] = _tensor(
            "extra",
            dtype="FLOAT32",
            shape=[1, 4, 4, 2],
            signature=[-1, 4, 4, 2],
        )
        logistic.outputs.append("extra")
    elif case == "wrong_quantize_type":
        quantize.op_type = "CAST"
    elif case == "extra_quantize_input":
        quantize.inputs.append("branch0_float_output")
    elif case == "extra_quantize_output":
        model_ir.tensors["extra"] = _tensor(
            "extra",
            dtype="INT8",
            shape=[1, 4, 4, 2],
            signature=[-1, 4, 4, 2],
            quantization=_output_grid("INT8"),
        )
        quantize.outputs.append("extra")
    elif case == "missing_quantized_input":
        del model_ir.tensors["branch0_quantized_input"]
    elif case == "missing_float_input":
        del model_ir.tensors["branch0_float_input"]
    elif case == "missing_float_output":
        del model_ir.tensors["branch0_float_output"]
    elif case == "missing_quantized_output":
        del model_ir.tensors["branch0_quantized_output"]
    elif case == "unsupported_quantized_dtype":
        quantized_input.dtype = quantized_output.dtype = "INT16"
    elif case == "quantized_dtype_mismatch":
        quantized_output.dtype = "UINT8"
    elif case == "missing_input_quantization":
        quantized_input.quantization = None
    elif case == "missing_output_quantization":
        quantized_output.quantization = None
    elif case == "per_axis_input_quantization":
        quantized_input.quantization = QuantParamIR([0.125, 0.25], [-3, -3], 3)
    elif case == "per_axis_output_quantization":
        quantized_output.quantization = QuantParamIR(
            [1.0 / 256.0, 1.0 / 256.0],
            [-128, -128],
            3,
        )
    elif case == "noncanonical_output_scale":
        quantized_output.quantization.scale = [0.25]
    elif case == "near_canonical_output_scale":
        quantized_output.quantization.scale = [1.0 / 256.0 + 5e-8]
    elif case == "wrong_output_zero_point":
        quantized_output.quantization.zero_point = [-127]
    elif case == "negative_input_scale":
        quantized_input.quantization.scale = [-0.125]
    elif case == "nonfinite_input_scale":
        quantized_input.quantization.scale = [np.inf]
    elif case == "input_zero_point_out_of_range":
        quantized_input.quantization.zero_point = [-129]
    elif case == "nonfloat_intermediate":
        float_input.dtype = float_output.dtype = "INT32"
    elif case == "float_dtype_mismatch":
        float_output.dtype = "FLOAT16"
    elif case == "quantized_input_shape_mismatch":
        quantized_input.shape = [1, 4, 2, 4]
    elif case == "float_input_signature_mismatch":
        float_input.shape_signature = [1, 4, 4, 2]
    elif case == "logistic_output_shape_mismatch":
        float_output.shape = [1, 4, 2, 4]
    elif case == "logistic_output_signature_mismatch":
        float_output.shape_signature = [1, 4, 4, 2]
    elif case == "quantized_output_shape_mismatch":
        quantized_output.shape = [1, 4, 2, 4]
    elif case == "quantized_output_signature_mismatch":
        quantized_output.shape_signature = [1, 4, 4, 2]
    elif case == "invalid_shape":
        float_input.shape = [1, 4, None, 2]
    elif case == "invalid_signature":
        float_output.shape_signature = [-1, 4, None, 2]

    before = repr(model_ir)
    stats = _optimize_dequant_logistic_quantize_chains(model_ir)

    assert stats == {"folded_dequant_logistic_quantize_chains": 0}
    assert repr(model_ir) == before


def test_quantized_logistic_skips_index_without_complete_family_and_prunes(
    monkeypatch,
) -> None:
    model_ir = ModelIR("quantized_logistic_without_chain")
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

    monkeypatch.setattr(logistic_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_dequant_logistic_quantize_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"folded_dequant_logistic_quantize_chains": 0}
    assert set(model_ir.tensors) == {"x", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
