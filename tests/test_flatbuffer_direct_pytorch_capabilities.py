from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_capabilities import (
    _DIRECT_CODEGEN_SUPPORTED_OP_TYPES,
    _can_emit_direct_module_call_for_codegen,
    _ensure_direct_codegen_supported,
    _ensure_native_export_supported_ops,
    _ensure_no_custom_ops,
    _ensure_supported_ops,
    _is_direct_codegen_unsupported_error,
    _is_channel_last_layout_for_codegen,
    _supports_runtime_wrapper_model_ir,
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


def test_direct_codegen_validation_keeps_runtime_only_ops_out() -> None:
    _ensure_direct_codegen_supported(_model_with_op("CONV_2D"))

    with pytest.raises(
        ModelIRPyTorchExportError,
        match=r"unsupported_op_types=\['WHILE'\]",
    ) as error:
        _ensure_direct_codegen_supported(_model_with_op("WHILE"))

    assert _is_direct_codegen_unsupported_error(error.value)
    assert not _is_direct_codegen_unsupported_error(
        ModelIRPyTorchExportError("unrelated export failure")
    )


def test_custom_op_validation_is_explicit() -> None:
    with pytest.raises(
        ModelIRPyTorchExportError,
        match="does not support CUSTOM ops",
    ):
        _ensure_no_custom_ops(_model_with_op("CUSTOM"))


def test_native_export_capability_validation_preserves_error_precedence() -> None:
    root_custom = _model_with_op("CUSTOM")
    root_custom.operators.append(
        OperatorIR(op_type="NOT_REAL", inputs=[], outputs=[], options={})
    )
    with pytest.raises(
        ModelIRPyTorchExportError,
        match="does not support CUSTOM ops",
    ):
        _ensure_native_export_supported_ops(root_custom)

    subgraph_custom = _model_with_op("ADD")
    subgraph_custom.subgraphs.append(_model_with_op("CUSTOM"))
    with pytest.raises(
        ModelIRPyTorchExportError,
        match=r"unsupported_op_types=\['CUSTOM'\]",
    ):
        _ensure_native_export_supported_ops(subgraph_custom)


def test_native_export_capability_validation_scans_root_operators_once() -> None:
    class CountingOperators(list[OperatorIR]):
        def __init__(self, values: list[OperatorIR]) -> None:
            super().__init__(values)
            self.iteration_count = 0

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.iteration_count += 1
            return super().__iter__()

    model_ir = _model_with_op("ADD")
    counting_operators = CountingOperators(model_ir.operators)
    model_ir.operators = counting_operators

    _ensure_native_export_supported_ops(model_ir)

    assert counting_operators.iteration_count == 1


def test_runtime_wrapper_capability_allows_runtime_ops_and_onnx_slice_custom() -> None:
    model_ir = _model_with_op("ADD")
    model_ir.operators.append(
        OperatorIR(
            op_type="CUSTOM",
            inputs=["y"],
            outputs=["z"],
            options={"customCode": "onnx_slice"},
        )
    )

    assert _supports_runtime_wrapper_model_ir(model_ir)


def test_runtime_wrapper_capability_rejects_unknown_and_other_custom_ops() -> None:
    assert not _supports_runtime_wrapper_model_ir(_model_with_op("UNKNOWN_RUNTIME_OP"))
    model_ir = _model_with_op("CUSTOM")
    model_ir.operators[0].options["customCode"] = "other"
    assert not _supports_runtime_wrapper_model_ir(model_ir)


def test_direct_module_capability_requires_channel_first_static_channels() -> None:
    op = OperatorIR(
        op_type="CONV_2D",
        inputs=["input", "weight"],
        outputs=["output"],
    )
    model_ir = ModelIR(
        name="direct_module_capability",
        tensors={
            "input": TensorIR(
                name="input",
                dtype="FLOAT32",
                shape=[1, 3, 8, 8],
                logical_layout="NCHW",
            ),
            "output": TensorIR(
                name="output",
                dtype="FLOAT32",
                shape=[1, 4, 8, 8],
                logical_layout="NCHW",
            ),
        },
        operators=[op],
    )

    assert _can_emit_direct_module_call_for_codegen(
        model_ir=model_ir,
        is_channel_last_layout_fn=_is_channel_last_layout_for_codegen,
        op=op,
    )
    model_ir.tensors["output"].logical_layout = "NHWC"
    assert not _can_emit_direct_module_call_for_codegen(
        model_ir=model_ir,
        is_channel_last_layout_fn=_is_channel_last_layout_for_codegen,
        op=op,
    )
