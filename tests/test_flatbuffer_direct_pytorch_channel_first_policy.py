from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_channel_first_policy import (
    _can_emit_channel_first_shape_preserving_unary_op_for_codegen,
    _can_resolve_channel_first_expr_statically_for_codegen,
    _channel_first_passthrough_input_expr_for_codegen,
)


def _tensor(name: str, shape: list[int], *, layout: str) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        logical_layout=layout,
    )


def test_channel_first_expression_traces_inverse_layout_bridge() -> None:
    model_ir = ModelIR(
        name="channel_first_trace",
        tensors={
            "source": _tensor("source", [1, 3, 4, 5], layout="NCHW"),
            "output": _tensor("output", [1, 4, 5, 3], layout="NHWC"),
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

    assert _channel_first_passthrough_input_expr_for_codegen(
        model_ir=model_ir,
        producer_index={"output": 0},
        channel_first_tensor_expr_aliases={},
        tensor_expr_fn=lambda name: f"expr_{name}",
        tensor_name="output",
    ) == "expr_source"
    assert _can_resolve_channel_first_expr_statically_for_codegen(
        model_ir=model_ir,
        producer_index={"output": 0},
        channel_first_tensor_expr_aliases={},
        direct_codegen_unary_expressions={"RELU"},
        tensor_name="output",
    )


def test_shape_preserving_unary_uses_static_channel_first_input() -> None:
    op = OperatorIR(
        op_type="RELU",
        inputs=["input"],
        outputs=["output"],
    )
    model_ir = ModelIR(
        name="channel_first_unary",
        tensors={
            "input": _tensor("input", [1, 3, 4, 5], layout="NCHW"),
            "output": _tensor("output", [1, 3, 4, 5], layout="NCHW"),
        },
        operators=[op],
    )

    assert _can_emit_channel_first_shape_preserving_unary_op_for_codegen(
        model_ir=model_ir,
        direct_codegen_unary_expressions={"RELU"},
        tensor_shape_list_fn=lambda name: list(model_ir.tensors[name].shape),
        can_resolve_channel_first_expr_statically_fn=lambda _name: True,
        op=op,
    )
