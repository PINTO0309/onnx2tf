from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_graph_policy import (
    _base_target_shape_values_for_model_ir,
    _channel_first_shape_for_tensor_for_codegen,
    _channel_first_shape_values_for_model_ir,
    _expected_channel_dim_for_tensor_for_codegen,
    _gather_input_pre_permute_for_codegen,
    _infer_effective_rank4_runtime_layout_for_codegen,
    _native_codegen_cache_bucket_for_model_ir,
    _native_codegen_graph_index_for_model_ir,
    _producer_op_for_model_ir,
    _rank4_channel_first_shape_for_tensor_for_codegen,
    _resize_target_shape_literal_for_model_ir,
    _target_shape_literal_for_model_ir,
    _target_shape_values_for_model_ir,
    _tensor_shape_list_for_model_ir,
    _to_channel_first_shape_for_model_ir,
)


def _tensor(name: str, shape: list[int], *, layout: str = "UNKNOWN") -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        logical_layout=layout,
    )


def test_gather_graph_policy_recovers_channel_first_to_last_boundary() -> None:
    model_ir = ModelIR(
        name="gather_boundary",
        tensors={
            "params": _tensor("params", [1, 3, 4, 5]),
            "output": _tensor("output", [1, 4, 5, 3]),
        },
    )

    assert _gather_input_pre_permute_for_codegen(
        model_ir=model_ir,
        params_name="params",
        output_name="output",
        axis=0,
        batch_dims=0,
    ) == [0, 2, 3, 1]
    assert (
        _gather_input_pre_permute_for_codegen(
            model_ir=model_ir,
            params_name="params",
            output_name="output",
            axis=0,
            batch_dims=1,
        )
        is None
    )


def test_runtime_layout_graph_policy_traces_passthrough_to_conv() -> None:
    model_ir = ModelIR(
        name="runtime_layout",
        tensors={
            "input": _tensor("input", [1, 3, 8, 8]),
            "weight": _tensor("weight", [4, 3, 3, 3]),
            "conv_output": _tensor("conv_output", [1, 4, 8, 8]),
            "output": _tensor("output", [1, 4, 8, 8]),
        },
        operators=[
            OperatorIR(
                op_type="CONV_2D",
                inputs=["input", "weight"],
                outputs=["conv_output"],
            ),
            OperatorIR(
                op_type="IDENTITY",
                inputs=["conv_output"],
                outputs=["output"],
            ),
        ],
        inputs=["input"],
        outputs=["output"],
    )
    graph_index = ModelIRGraphIndex(model_ir)

    assert _infer_effective_rank4_runtime_layout_for_codegen(
        model_ir=model_ir,
        producer_index=graph_index.producers,
        consumer_index=graph_index.consumers,
        tensor_name="output",
    ) == "NCHW"


def test_codegen_graph_cache_reuses_index_for_channel_and_producer_queries() -> None:
    model_ir = ModelIR(
        name="codegen_cache",
        tensors={
            "input": _tensor("input", [1, 3, 8, 8], layout="NCHW"),
            "weight": _tensor("weight", [4, 3, 3, 3]),
            "output": _tensor("output", [1, 4, 8, 8], layout="NCHW"),
        },
        operators=[
            OperatorIR(
                op_type="CONV_2D",
                inputs=["input", "weight"],
                outputs=["output"],
            ),
        ],
        inputs=["input"],
        outputs=["output"],
    )
    graph_index = ModelIRGraphIndex(model_ir)
    cache = _native_codegen_cache_bucket_for_model_ir(model_ir=model_ir)
    cache["graph_index"] = graph_index

    assert _native_codegen_graph_index_for_model_ir(model_ir=model_ir) is graph_index
    assert (
        _expected_channel_dim_for_tensor_for_codegen(
            model_ir=model_ir,
            tensor_name="input",
        )
        == 3
    )
    assert _producer_op_for_model_ir(
        model_ir=model_ir,
        tensor_name="output",
    ) is model_ir.operators[0]


def test_graph_shape_policy_preserves_public_layout_contract() -> None:
    model_ir = ModelIR(
        name="public_shape",
        tensors={
            "input": _tensor("input", [1, 8, 8, 3], layout="NHWC"),
        },
        inputs=["input"],
        outputs=["input"],
    )

    assert _base_target_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name="input",
    ) == [1, 8, 8, 3]
    assert _to_channel_first_shape_for_model_ir(
        model_ir=model_ir,
        tensor_name="input",
        shape_values=[1, 8, 8, 3],
    ) == [1, 3, 8, 8]
    assert _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name="input",
    ) == [1, 3, 8, 8]
    assert _target_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name="input",
    ) == [1, 8, 8, 3]
    assert _target_shape_literal_for_model_ir(
        model_ir=model_ir,
        tensor_name="input",
    ) == "[1, 8, 8, 3]"
    assert _tensor_shape_list_for_model_ir(
        model_ir=model_ir,
        tensor_name="input",
    ) == [1, 8, 8, 3]
    assert _rank4_channel_first_shape_for_tensor_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases={},
        tensor_name="input",
    ) == [1, 8, 3, 8]
    assert _channel_first_shape_for_tensor_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases={},
        tensor_name="input",
    ) == [1, 8, 3, 8]


def test_resize_target_literal_recovers_channel_first_storage() -> None:
    model_ir = ModelIR(
        name="resize_shape",
        tensors={
            "input": _tensor("input", [1, 3, 8, 8], layout="NCHW"),
            "output": _tensor("output", [1, 16, 16, 3], layout="NCHW"),
        },
    )

    assert _resize_target_shape_literal_for_model_ir(
        model_ir=model_ir,
        output_name="output",
        input_name="input",
    ) == "[1, 3, 16, 16]"
