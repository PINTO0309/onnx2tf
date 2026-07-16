from __future__ import annotations

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.quantized_pool as pool_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.quantized_pool import (
    _optimize_dequant_maxpool_quantize_chains,
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


def _grid(dtype: str, branch_index: int = 0) -> QuantParamIR:
    zero_point = -3 - int(branch_index) if dtype == "INT8" else 7 + int(branch_index)
    return QuantParamIR(
        scale=[0.125 * (branch_index + 1)],
        zero_point=[zero_point],
        quantized_dimension=0,
    )


def _quantized_pool_model(*, branches: int = 2) -> ModelIR:
    model_ir = ModelIR("indexed_quantized_maxpool")
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
                    shape=[1, 4, 4, 2],
                    signature=[-1, 4, 4, 2],
                    quantization=_grid(dtype, branch_index),
                ),
                float_input: _tensor(
                    float_input,
                    dtype="FLOAT32",
                    shape=[1, 4, 4, 2],
                    signature=[-1, 4, 4, 2],
                ),
                float_output: _tensor(
                    float_output,
                    dtype="FLOAT32",
                    shape=[1, 2, 2, 2],
                    signature=[-1, 2, 2, 2],
                ),
                quantized_output: _tensor(
                    quantized_output,
                    dtype=dtype,
                    shape=[1, 2, 2, 2],
                    signature=[-1, 2, 2, 2],
                    quantization=_grid(dtype, branch_index),
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
                    "MAX_POOL_2D",
                    [float_input],
                    [float_output],
                    options={
                        "padding": "SAME",
                        "strideW": 2,
                        "strideH": 2,
                        "filterWidth": 3,
                        "filterHeight": 3,
                        "fusedActivationFunction": "NONE",
                    },
                    axis_semantics={"spatial": "NHWC"},
                    version=2,
                    onnx_node_name=f"{prefix}_maxpool",
                    onnx_op_type="MaxPool",
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


def test_quantized_pool_folds_multiple_dtypes_with_one_index(monkeypatch) -> None:
    model_ir = _quantized_pool_model()
    input_quantizations = [
        model_ir.tensors[f"branch{index}_quantized_input"].quantization
        for index in range(2)
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_dequant_maxpool_quantize_chains(model_ir)

    assert stats == {"folded_dequant_maxpool_quantize_chains": 2}
    assert refresh_count == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "MAX_POOL_2D",
        "MAX_POOL_2D",
    ]
    for branch_index, pool in enumerate(model_ir.operators):
        prefix = f"branch{branch_index}"
        assert pool.inputs == [f"{prefix}_quantized_input"]
        assert pool.outputs == [f"{prefix}_quantized_output"]
        assert pool.options == {
            "padding": "SAME",
            "strideW": 2,
            "strideH": 2,
            "filterWidth": 3,
            "filterHeight": 3,
            "fusedActivationFunction": "NONE",
        }
        assert pool.axis_semantics == {"spatial": "NHWC"}
        assert pool.version == 2
        assert pool.onnx_node_name == f"{prefix}_maxpool"
        assert pool.onnx_op_type == "MaxPool"
        output = model_ir.tensors[f"{prefix}_quantized_output"]
        assert output.quantization == input_quantizations[branch_index]
        assert output.quantization is not input_quantizations[branch_index]
        assert output.shape == [1, 2, 2, 2]
        assert output.shape_signature == [-1, 2, 2, 2]
    assert set(model_ir.tensors) == {
        "branch0_quantized_input",
        "branch0_quantized_output",
        "branch1_quantized_input",
        "branch1_quantized_output",
    }


def test_quantized_pool_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _quantized_pool_model(branches=1)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_dequant_maxpool_quantize_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"folded_dequant_maxpool_quantize_chains": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert set(layout_state.logical) == {
        "branch0_quantized_input",
        "branch0_quantized_output",
    }


def test_quantized_pool_preserves_quantized_input_and_output_fanout() -> None:
    model_ir = _quantized_pool_model(branches=1)
    model_ir.tensors["input_side"] = _tensor(
        "input_side",
        dtype="INT8",
        shape=[1, 4, 4, 2],
        signature=[-1, 4, 4, 2],
        quantization=_grid("INT8"),
    )
    model_ir.tensors["output_side"] = _tensor(
        "output_side",
        dtype="INT8",
        shape=[1, 2, 2, 2],
        signature=[-1, 2, 2, 2],
        quantization=_grid("INT8"),
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

    stats = _optimize_dequant_maxpool_quantize_chains(model_ir)

    assert stats == {"folded_dequant_maxpool_quantize_chains": 1}
    assert [operator.op_type for operator in model_ir.operators] == [
        "MAX_POOL_2D",
        "IDENTITY",
        "IDENTITY",
    ]


def test_quantized_pool_accepts_exact_dict_quantization_grid() -> None:
    model_ir = _quantized_pool_model(branches=1)
    grid = {
        "scale": np.asarray([0.125], dtype=np.float32),
        "zero_point": np.asarray([-3], dtype=np.int64),
        "quantized_dimension": 0,
    }
    model_ir.tensors["branch0_quantized_input"].quantization = grid
    model_ir.tensors["branch0_quantized_output"].quantization = {
        key: np.copy(value) if isinstance(value, np.ndarray) else value
        for key, value in grid.items()
    }

    stats = _optimize_dequant_maxpool_quantize_chains(model_ir)

    assert stats == {"folded_dequant_maxpool_quantize_chains": 1}


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
        "reverse_dequantize_pool_order",
        "reverse_pool_quantize_order",
        "wrong_dequantize_type",
        "extra_dequantize_input",
        "extra_dequantize_output",
        "wrong_pool_type",
        "extra_pool_input",
        "extra_pool_output",
        "nondict_pool_options",
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
        "per_axis_quantization",
        "scale_mismatch",
        "near_scale_mismatch",
        "zero_point_mismatch",
        "negative_scale",
        "nonfinite_scale",
        "zero_point_out_of_range",
        "nonfloat_intermediate",
        "float_dtype_mismatch",
        "quantized_input_rank_three",
        "float_input_shape_mismatch",
        "float_input_signature_mismatch",
        "float_output_rank_three",
        "quantized_output_shape_mismatch",
        "quantized_output_signature_mismatch",
        "invalid_signature",
    ],
)
def test_quantized_pool_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = _quantized_pool_model(branches=1)
    dequantize, pool, quantize = model_ir.operators
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
        shape = [1, 4, 4, 2] if case == "float_input_fanout" else [1, 2, 2, 2]
        signature = [-1, 4, 4, 2] if case == "float_input_fanout" else [-1, 2, 2, 2]
        model_ir.tensors["side"] = _tensor(
            "side",
            dtype="FLOAT32",
            shape=shape,
            signature=signature,
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
    elif case == "reverse_dequantize_pool_order":
        model_ir.operators = [pool, dequantize, quantize]
    elif case == "reverse_pool_quantize_order":
        model_ir.operators = [dequantize, quantize, pool]
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
    elif case == "wrong_pool_type":
        pool.op_type = "AVERAGE_POOL_2D"
    elif case == "extra_pool_input":
        pool.inputs.append("branch0_float_input")
    elif case == "extra_pool_output":
        model_ir.tensors["extra"] = _tensor(
            "extra",
            dtype="FLOAT32",
            shape=[1, 2, 2, 2],
            signature=[-1, 2, 2, 2],
        )
        pool.outputs.append("extra")
    elif case == "nondict_pool_options":
        pool.options = None
    elif case == "wrong_quantize_type":
        quantize.op_type = "CAST"
    elif case == "extra_quantize_input":
        quantize.inputs.append("branch0_float_output")
    elif case == "extra_quantize_output":
        model_ir.tensors["extra"] = _tensor(
            "extra",
            dtype="INT8",
            shape=[1, 2, 2, 2],
            signature=[-1, 2, 2, 2],
            quantization=_grid("INT8"),
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
    elif case == "per_axis_quantization":
        quantized_input.quantization = QuantParamIR([0.125, 0.25], [-3, -3], 3)
        quantized_output.quantization = QuantParamIR([0.125, 0.25], [-3, -3], 3)
    elif case == "scale_mismatch":
        quantized_output.quantization.scale = [0.25]
    elif case == "near_scale_mismatch":
        quantized_output.quantization.scale = [0.12500001]
    elif case == "zero_point_mismatch":
        quantized_output.quantization.zero_point = [-2]
    elif case == "negative_scale":
        quantized_input.quantization.scale = quantized_output.quantization.scale = [
            -0.125
        ]
    elif case == "nonfinite_scale":
        quantized_input.quantization.scale = quantized_output.quantization.scale = [
            np.inf
        ]
    elif case == "zero_point_out_of_range":
        quantized_input.quantization.zero_point = (
            quantized_output.quantization.zero_point
        ) = [-129]
    elif case == "nonfloat_intermediate":
        float_input.dtype = float_output.dtype = "INT32"
    elif case == "float_dtype_mismatch":
        float_output.dtype = "FLOAT16"
    elif case == "quantized_input_rank_three":
        quantized_input.shape = [1, 4, 4]
        quantized_input.shape_signature = [-1, 4, 4]
    elif case == "float_input_shape_mismatch":
        float_input.shape = [1, 4, 2, 4]
    elif case == "float_input_signature_mismatch":
        float_input.shape_signature = [1, 4, 4, 2]
    elif case == "float_output_rank_three":
        float_output.shape = [1, 2, 2]
        float_output.shape_signature = [-1, 2, 2]
    elif case == "quantized_output_shape_mismatch":
        quantized_output.shape = [1, 2, 1, 4]
    elif case == "quantized_output_signature_mismatch":
        quantized_output.shape_signature = [1, 2, 2, 2]
    elif case == "invalid_signature":
        float_output.shape_signature = [-1, 2, None, 2]

    before = repr(model_ir)
    stats = _optimize_dequant_maxpool_quantize_chains(model_ir)

    assert stats == {"folded_dequant_maxpool_quantize_chains": 0}
    assert repr(model_ir) == before


def test_quantized_pool_skips_index_without_complete_family_and_prunes(
    monkeypatch,
) -> None:
    model_ir = ModelIR("quantized_pool_without_chain")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor(
            "x",
            dtype="FLOAT32",
            shape=[1, 2, 2, 1],
            signature=[1, 2, 2, 1],
        ),
        "y": _tensor(
            "y",
            dtype="FLOAT32",
            shape=[1, 2, 2, 1],
            signature=[1, 2, 2, 1],
        ),
        "unused": _tensor(
            "unused",
            dtype="FLOAT32",
            shape=[1, 2, 2, 1],
            signature=[1, 2, 2, 1],
        ),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]
    layout_state = LayoutState.from_model_ir(model_ir)

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(pool_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_dequant_maxpool_quantize_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"folded_dequant_maxpool_quantize_chains": 0}
    assert set(model_ir.tensors) == {"x", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
