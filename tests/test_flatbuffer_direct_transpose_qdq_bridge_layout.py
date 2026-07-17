from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_quant_dequant_bridges,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze
from onnx2tf.tflite_builder.passes.transpose_qdq_bridge_layout import (
    optimize_transpose_quant_dequant_bridges,
)


def _quantization(*, per_tensor: bool = True) -> QuantParamIR:
    return QuantParamIR(
        scale=[0.1] if per_tensor else [0.1, 0.2, 0.3, 0.4],
        zero_point=[0] if per_tensor else [0, 0, 0, 0],
        quantized_dimension=1,
    )


def _tensor(
    name: str,
    dtype: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        quantization=quantization,
    )


def _snapshot(model_ir: ModelIR) -> tuple:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or tensor.shape),
                _freeze(tensor.data),
                _freeze(tensor.quantization),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                _freeze(operator.options),
            )
            for operator in model_ir.operators
        ),
    )


def _pattern_a_model(
    *,
    per_tensor: bool = True,
    post_permutation: list[int] | None = None,
    public_bridge: str | None = None,
) -> ModelIR:
    nhwc = [1, 2, 3, 4]
    nchw = [1, 4, 2, 3]
    pre_permutation = np.asarray([0, 3, 1, 2], dtype=np.int32)
    post_permutation = (
        [0, 2, 3, 1]
        if post_permutation is None
        else list(post_permutation)
    )
    model_ir = ModelIR("transpose_qdq_pattern_a")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    if public_bridge is not None:
        model_ir.outputs.append(public_bridge)
    model_ir.tensors = {
        "x": _tensor("x", "FLOAT32", nhwc),
        "pre_perm": _tensor(
            "pre_perm", "INT32", [4], data=pre_permutation
        ),
        "pre": _tensor("pre", "FLOAT32", nchw),
        "quantized": _tensor(
            "quantized",
            "INT8",
            nchw,
            quantization=_quantization(per_tensor=per_tensor),
        ),
        "dequantized": _tensor("dequantized", "FLOAT32", nchw),
        "post_perm": _tensor(
            "post_perm",
            "INT32",
            [4],
            data=np.asarray(post_permutation, dtype=np.int32),
        ),
        "y": _tensor("y", "FLOAT32", nhwc),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "pre_perm"],
            outputs=["pre"],
        ),
        OperatorIR(
            op_type="QUANTIZE", inputs=["pre"], outputs=["quantized"]
        ),
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=["quantized"],
            outputs=["dequantized"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["dequantized", "post_perm"],
            outputs=["y"],
        ),
    ]
    return model_ir


def _pattern_b_dequantize_model() -> ModelIR:
    nhwc = [1, 2, 3, 4]
    nchw = [1, 4, 2, 3]
    model_ir = ModelIR("transpose_dequantize_pattern_b")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", "INT8", nhwc),
        "pre_perm": _tensor(
            "pre_perm",
            "INT32",
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "pre": _tensor(
            "pre", "INT8", nchw, quantization=_quantization()
        ),
        "dequantized": _tensor("dequantized", "FLOAT32", nchw),
        "post_perm": _tensor(
            "post_perm",
            "INT32",
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "y": _tensor("y", "FLOAT32", nhwc),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "pre_perm"],
            outputs=["pre"],
        ),
        OperatorIR(
            op_type="DEQUANTIZE",
            inputs=["pre"],
            outputs=["dequantized"],
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["dequantized", "post_perm"],
            outputs=["y"],
        ),
    ]
    return model_ir


def test_pattern_a_removes_inverse_transposes_and_is_idempotent() -> None:
    model_ir = _pattern_a_model()

    stats = optimize_transpose_quant_dequant_bridges(model_ir)

    assert stats == {
        "removed_transpose_quantize_dequantize_bridges": 1,
        "rewritten_add_qdq_residual_transpose_bridges": 0,
        "rewritten_mixed_add_qdq_residual_transpose_bridges": 0,
    }
    assert [operator.op_type for operator in model_ir.operators] == [
        "QUANTIZE",
        "DEQUANTIZE",
    ]
    quantize, dequantize = model_ir.operators
    assert quantize.inputs == ["x"]
    assert dequantize.outputs == ["y"]
    assert model_ir.tensors["y"].dtype == "FLOAT32"
    before = _snapshot(model_ir)
    assert optimize_transpose_quant_dequant_bridges(model_ir) == {
        "removed_transpose_quantize_dequantize_bridges": 0,
        "rewritten_add_qdq_residual_transpose_bridges": 0,
        "rewritten_mixed_add_qdq_residual_transpose_bridges": 0,
    }
    assert _snapshot(model_ir) == before


def test_pattern_b_dequantize_clones_input_grid_and_removes_transposes() -> None:
    model_ir = _pattern_b_dequantize_model()
    source_grid = model_ir.tensors["pre"].quantization

    stats = optimize_transpose_quant_dequant_bridges(model_ir)

    assert stats["removed_transpose_quantize_dequantize_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "DEQUANTIZE"
    ]
    dequantize = model_ir.operators[0]
    assert dequantize.inputs == ["x"]
    assert dequantize.outputs == ["y"]
    assert _freeze(model_ir.tensors["x"].quantization) == _freeze(
        source_grid
    )
    assert model_ir.tensors["x"].quantization is not source_grid


@pytest.mark.parametrize(
    "model_ir",
    [
        pytest.param(
            _pattern_a_model(per_tensor=False), id="per-channel-grid"
        ),
        pytest.param(
            _pattern_a_model(post_permutation=[0, 2, 1, 3]),
            id="non-inverse-permutation",
        ),
        pytest.param(
            _pattern_a_model(public_bridge="pre"),
            id="public-intermediate",
        ),
    ],
)
def test_pattern_a_guards_are_noop(model_ir: ModelIR) -> None:
    before = _snapshot(model_ir)

    assert optimize_transpose_quant_dequant_bridges(model_ir) == {
        "removed_transpose_quantize_dequantize_bridges": 0,
        "rewritten_add_qdq_residual_transpose_bridges": 0,
        "rewritten_mixed_add_qdq_residual_transpose_bridges": 0,
    }
    assert _snapshot(model_ir) == before


def test_compatibility_wrapper_matches_module_owner() -> None:
    direct_model = _pattern_a_model()
    wrapper_model = copy.deepcopy(direct_model)

    direct_stats = optimize_transpose_quant_dequant_bridges(direct_model)
    wrapper_stats = _optimize_transpose_quant_dequant_bridges(wrapper_model)

    assert wrapper_stats == direct_stats
    assert _snapshot(wrapper_model) == _snapshot(direct_model)
