from __future__ import annotations

from onnx2tf.tflite_builder.artifact_metadata import (
    collect_custom_op_artifact_metadata,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


def test_custom_op_artifact_metadata_preserves_legacy_normalization() -> None:
    model_ir = ModelIR(name="custom_metadata")
    model_ir.operators = [
        OperatorIR(op_type="ADD", inputs=[], outputs=[]),
        OperatorIR(
            op_type="CUSTOM",
            inputs=[],
            outputs=[],
            options={
                "customCode": " ZETA ",
                "onnxOp": " Einsum ",
                "onnxNodeName": " node_b ",
            },
        ),
        OperatorIR(
            op_type="CUSTOM",
            inputs=[],
            outputs=[],
            options={
                "customCode": "",
                "onnxOp": "TopK",
                "onnxNodeName": "node_a",
            },
        ),
        OperatorIR(
            op_type="CUSTOM",
            inputs=[],
            outputs=[],
            options={
                "customCode": " ZETA ",
                "onnxOp": " Einsum ",
                "onnxNodeName": " node_b ",
            },
        ),
    ]

    custom_ops_used, custom_op_nodes = collect_custom_op_artifact_metadata(model_ir)

    assert custom_ops_used == ["", " ZETA "]
    assert custom_op_nodes == [
        {
            "custom_code": "CUSTOM",
            "onnx_op": "TopK",
            "onnx_node_name": "node_a",
        },
        {
            "custom_code": "ZETA",
            "onnx_op": "Einsum",
            "onnx_node_name": "node_b",
        },
    ]


def test_custom_op_artifact_metadata_reads_each_operator_once() -> None:
    op_type_reads = 0

    class _CountingOperator:
        options = {}

        @property
        def op_type(self) -> str:
            nonlocal op_type_reads
            op_type_reads += 1
            return "ADD"

    model_ir = ModelIR(name="single_artifact_metadata_scan")
    model_ir.operators = [_CountingOperator() for _ in range(256)]

    assert collect_custom_op_artifact_metadata(model_ir) == ([], [])
    assert op_type_reads == 256
