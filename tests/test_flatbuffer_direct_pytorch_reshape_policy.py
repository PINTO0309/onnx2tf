from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_reshape_policy import (
    _reshape_codegen_is_plain_data_only_for_codegen,
    _reshape_prefers_feature_last_for_adjx_batch_matmul_for_codegen,
    _static_sequence_length_for_model_ir,
    _tensor_exact_static_shape_list_for_model_ir,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
    layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        shape_signature=signature,
        logical_layout=layout,
    )


def test_lowered_flatten_is_plain_reshape_data() -> None:
    op = OperatorIR(
        op_type="RESHAPE",
        inputs=["input"],
        outputs=["output"],
        options={"onnxFlattenAxis": 1},
    )
    model_ir = ModelIR(
        name="flatten",
        tensors={
            "input": _tensor("input", [1, 2, 3, 4], layout="NCHW"),
            "output": _tensor("output", [1, 24]),
        },
        operators=[op],
    )

    assert _reshape_codegen_is_plain_data_only_for_codegen(
        model_ir=model_ir,
        op=op,
        infer_effective_rank4_runtime_layout_fn=lambda _name: "NCHW",
        reshape_preserves_channel_last_sequence_fn=lambda *_args: None,
        reshape_prefers_feature_last_for_adjx_batch_matmul_fn=lambda *_args: None,
    )


def test_static_reshape_shape_queries_require_positive_signature() -> None:
    model_ir = ModelIR(
        name="static_shape",
        tensors={
            "input": _tensor("input", [2, 3], signature=[2, 3]),
            "dynamic": _tensor("dynamic", [1, 3], signature=[-1, 3]),
        },
    )

    assert _tensor_exact_static_shape_list_for_model_ir(
        model_ir=model_ir,
        tensor_name="input",
    ) == [2, 3]
    assert _static_sequence_length_for_model_ir(
        model_ir=model_ir,
        tensor_name="input",
    ) == 2
    assert (
        _tensor_exact_static_shape_list_for_model_ir(
            model_ir=model_ir,
            tensor_name="dynamic",
        )
        is None
    )


def test_adjx_batch_matmul_requests_feature_last_reshape() -> None:
    batch_matmul = OperatorIR(
        op_type="BATCH_MATMUL",
        inputs=["reshaped", "rhs"],
        outputs=["output"],
        options={"adjX": True},
    )
    model_ir = ModelIR(
        name="adjx_reshape",
        tensors={
            "input": _tensor("input", [1, 4, 8], layout="NCW"),
            "reshaped": _tensor("reshaped", [8, 1, 4], layout="NCW"),
            "rhs": _tensor("rhs", [1, 4, 6]),
            "output": _tensor("output", [8, 1, 6]),
        },
        operators=[batch_matmul],
    )

    assert _reshape_prefers_feature_last_for_adjx_batch_matmul_for_codegen(
        model_ir=model_ir,
        consumer_index={"reshaped": [0]},
        input_tensor_name="input",
        output_name="reshaped",
    ) == ([0, 1, 2], [8, 4, 1])
