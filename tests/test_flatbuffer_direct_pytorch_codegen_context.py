from pathlib import Path

from onnx2tf.tflite_builder._pytorch_exporter_native_codegen_common import (
    _NativeModelFileWriterContext,
)
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def test_native_writer_context_preserves_one_shared_graph_index() -> None:
    model_ir = ModelIR(
        name="context_index",
        tensors={
            "input": TensorIR(name="input", dtype="FLOAT32", shape=[1]),
            "output": TensorIR(name="output", dtype="FLOAT32", shape=[1]),
        },
        operators=[OperatorIR(op_type="IDENTITY", inputs=["input"], outputs=["output"])],
        inputs=["input"],
        outputs=["output"],
    )
    graph_index = ModelIRGraphIndex(model_ir)
    context = _NativeModelFileWriterContext(
        output_folder_path="unused",
        model_ir=model_ir,
        metadata={},
        tensor_storage_name_map={},
        package_dir=Path("unused"),
        preserve_channel_last_tensor_names=set(),
        tensor_var_names={},
        graph_index=graph_index,
    )

    assert context.graph_index is graph_index
    assert context.producer_index is graph_index.producers
    assert context.consumer_index is graph_index.consumers
