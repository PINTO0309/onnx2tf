from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_layout_bridge_policy import (
    _fold_single_consumer_public_input_bridge_for_codegen,
    _match_single_consumer_layout_bridge_transpose_for_codegen,
)


def _tensor(name: str, shape: list[int], logical_layout: str) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        logical_layout=logical_layout,
    )


def test_fold_single_consumer_public_input_bridge() -> None:
    bridge_name = "input_public_layout_bridge"
    model_ir = ModelIR(
        name="public_input_bridge",
        operators=[
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["input"],
                outputs=[bridge_name],
                options={"perm": [0, 3, 1, 2]},
            )
        ],
        inputs=["input"],
    )

    assert _fold_single_consumer_public_input_bridge_for_codegen(
        model_ir=model_ir,
        producer_index={bridge_name: 0},
        consumer_index={bridge_name: [1]},
        public_layout_bridge_tensor_names={bridge_name},
        public_input_names={"input"},
        tensor_name=bridge_name,
        downstream_permute=None,
    ) == ("input", [0, 3, 1, 2], 0)


def test_match_single_consumer_layout_bridge_requires_exact_layout_perm() -> None:
    model_ir = ModelIR(
        name="layout_bridge",
        tensors={
            "source": _tensor("source", [1, 3, 4, 5], "NCHW"),
            "output": _tensor("output", [1, 4, 5, 3], "NHWC"),
        },
        operators=[
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["source"],
                outputs=["output"],
                options={"perm": [0, 2, 3, 1]},
            )
        ],
    )

    assert _match_single_consumer_layout_bridge_transpose_for_codegen(
        model_ir=model_ir,
        consumer_index={"source": [0]},
        tensor_name="source",
        required_output_layout="NHWC",
    ) == ("output", 0)
    assert (
        _match_single_consumer_layout_bridge_transpose_for_codegen(
            model_ir=model_ir,
            consumer_index={"source": [0, 1]},
            tensor_name="source",
        )
        is None
    )
