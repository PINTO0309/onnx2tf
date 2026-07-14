import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_constant_policy import (
    _is_constant_tensor_name_for_codegen,
    _reshape_shape_tensor_uses_runtime_dims_for_codegen,
    _shape_tensor_constant_is_non_zero_int_vector_for_codegen,
    _static_int_tensor_values_for_codegen,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(name=name, dtype="INT64", shape=shape, data=data)


def test_constant_shape_vector_contract() -> None:
    model_ir = ModelIR(
        name="constant_vector",
        tensors={
            "shape": _tensor(
                "shape",
                [3],
                data=np.asarray([1, 2, 3], dtype=np.int64),
            ),
            "zero_shape": _tensor(
                "zero_shape",
                [2],
                data=np.asarray([1, 0], dtype=np.int64),
            ),
        },
    )

    assert _is_constant_tensor_name_for_codegen(
        model_ir=model_ir,
        tensor_name="shape",
    )
    assert _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
        model_ir=model_ir,
        tensor_name="shape",
    )
    assert not _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
        model_ir=model_ir,
        tensor_name="zero_shape",
    )


def test_static_shape_gather_is_evaluated_without_runtime() -> None:
    model_ir = ModelIR(
        name="shape_gather",
        tensors={
            "input": _tensor("input", [2, 3, 4]),
            "shape": _tensor("shape", [3]),
            "index": _tensor(
                "index",
                [1],
                data=np.asarray([1], dtype=np.int32),
            ),
            "dimension": _tensor("dimension", [1]),
        },
        operators=[
            OperatorIR(op_type="SHAPE", inputs=["input"], outputs=["shape"]),
            OperatorIR(
                op_type="GATHER",
                inputs=["shape", "index"],
                outputs=["dimension"],
                options={"axis": 0, "batchDims": 0},
            ),
        ],
    )
    graph_index = ModelIRGraphIndex(model_ir)

    assert _static_int_tensor_values_for_codegen(
        model_ir=model_ir,
        producer_index=graph_index.producers,
        tensor_name="shape",
    ) == [2, 3, 4]
    assert _static_int_tensor_values_for_codegen(
        model_ir=model_ir,
        producer_index=graph_index.producers,
        tensor_name="dimension",
    ) == [3]


def test_runtime_shape_dimension_propagates_through_identity() -> None:
    model_ir = ModelIR(
        name="runtime_shape",
        tensors={
            "input": _tensor("input", [2, 3, 4]),
            "shape": _tensor("shape", [3]),
            "identity": _tensor("identity", [3]),
        },
        operators=[
            OperatorIR(op_type="SHAPE", inputs=["input"], outputs=["shape"]),
            OperatorIR(
                op_type="IDENTITY",
                inputs=["shape"],
                outputs=["identity"],
            ),
        ],
    )
    graph_index = ModelIRGraphIndex(model_ir)

    assert _reshape_shape_tensor_uses_runtime_dims_for_codegen(
        model_ir=model_ir,
        producer_index=graph_index.producers,
        tensor_name="identity",
    )
