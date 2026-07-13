from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.pytorch_capabilities import (
    _DIRECT_CODEGEN_SUPPORTED_OP_TYPES,
    _ensure_no_custom_ops,
    _ensure_supported_ops,
    get_supported_pytorch_kernel_op_types,
)
from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError


def _model_with_op(op_type: str) -> ModelIR:
    model_ir = ModelIR(name=f"capability_{op_type.lower()}")
    model_ir.operators.append(
        OperatorIR(
            op_type=op_type,
            inputs=[],
            outputs=[],
            options={},
        )
    )
    return model_ir


def test_supported_kernel_query_returns_independent_copy() -> None:
    first = get_supported_pytorch_kernel_op_types()
    assert first
    first.add("MUTATED_BY_TEST")

    assert "MUTATED_BY_TEST" not in get_supported_pytorch_kernel_op_types()


def test_direct_codegen_registry_accepts_declared_op() -> None:
    assert "CONV_2D" in _DIRECT_CODEGEN_SUPPORTED_OP_TYPES

    _ensure_supported_ops(_model_with_op("CONV_2D"))


def test_capability_validation_reports_unknown_op() -> None:
    with pytest.raises(
        ModelIRPyTorchExportError,
        match=r"unsupported_op_types=\['NOT_REAL'\]",
    ):
        _ensure_supported_ops(_model_with_op("NOT_REAL"))


def test_custom_op_validation_is_explicit() -> None:
    with pytest.raises(
        ModelIRPyTorchExportError,
        match="does not support CUSTOM ops",
    ):
        _ensure_no_custom_ops(_model_with_op("CUSTOM"))
