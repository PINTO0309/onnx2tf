from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_graph_policy import (
    _gather_input_pre_permute_for_codegen,
    _infer_effective_rank4_runtime_layout_for_codegen,
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
