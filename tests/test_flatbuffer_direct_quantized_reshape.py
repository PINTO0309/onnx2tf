from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_dequant_reshape_quantize_chains,
)


def _tensor(
    name: str,
    dtype: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
    scale: float | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        quantization=(
            QuantParamIR(
                scale=[float(scale)],
                zero_point=[0],
                quantized_dimension=0,
            )
            if scale is not None
            else None
        ),
    )


def _model(*, output_scale: float) -> ModelIR:
    model_ir = ModelIR("quantized_reshape")
    model_ir.inputs = ["xq"]
    model_ir.outputs = ["yq"]
    model_ir.tensors = {
        "xq": _tensor("xq", "INT8", [1, 2, 3], scale=0.1),
        "xf": _tensor("xf", "FLOAT32", [1, 2, 3]),
        "shape": _tensor(
            "shape",
            "INT32",
            [2],
            data=np.asarray([1, 6], dtype=np.int32),
        ),
        "yf": _tensor("yf", "FLOAT32", [1, 6]),
        "yq": _tensor("yq", "INT8", [1, 6], scale=output_scale),
    }
    model_ir.operators = [
        OperatorIR("DEQUANTIZE", ["xq"], ["xf"]),
        OperatorIR("RESHAPE", ["xf", "shape"], ["yf"]),
        OperatorIR("QUANTIZE", ["yf"], ["yq"]),
    ]
    return model_ir


def test_dequant_reshape_quantize_characterization() -> None:
    model_ir = _model(output_scale=0.1)

    stats = _optimize_dequant_reshape_quantize_chains(model_ir)

    assert stats == {"folded_dequant_reshape_quantize_chains": 1}
    assert [op.op_type for op in model_ir.operators] == ["RESHAPE"]
    assert model_ir.operators[0].inputs == ["xq", "shape"]
    assert model_ir.operators[0].outputs == ["yq"]
    assert model_ir.tensors["yq"].dtype == "INT8"
    assert model_ir.tensors["yq"].quantization is not None
    assert model_ir.tensors["yq"].quantization.scale == [0.1]


def test_dequant_reshape_quantize_rejects_mismatched_quantization() -> None:
    model_ir = _model(output_scale=0.2)

    stats = _optimize_dequant_reshape_quantize_chains(model_ir)

    assert stats == {"folded_dequant_reshape_quantize_chains": 0}
    assert [op.op_type for op in model_ir.operators] == [
        "DEQUANTIZE",
        "RESHAPE",
        "QUANTIZE",
    ]
