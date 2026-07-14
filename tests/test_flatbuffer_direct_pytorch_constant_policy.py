import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_constant_policy import (
    _axis_expr_from_input_for_codegen,
    _constant_pad_pairs_for_codegen,
    _int_scalar_literal_expr_for_codegen,
    _is_constant_tensor_name_for_codegen,
    _pad_literal_expr_for_codegen,
    _reshape_shape_tensor_uses_runtime_dims_for_codegen,
    _scalar_literal_expr_for_codegen,
    _shape_tensor_constant_is_non_zero_int_vector_for_codegen,
    _static_mirror_pad_expr_for_codegen,
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


def test_constant_pad_and_scalar_literals() -> None:
    model_ir = ModelIR(
        name="constant_literals",
        tensors={
            "pads": _tensor(
                "pads",
                [2, 2],
                data=np.asarray([[0, 0], [1, 2]], dtype=np.int64),
            ),
            "scalar": _tensor(
                "scalar",
                [],
                data=np.asarray(1.25, dtype=np.float32),
            ),
        },
    )

    assert _constant_pad_pairs_for_codegen(
        model_ir=model_ir,
        tensor_name="pads",
    ) == [[0, 0], [1, 2]]
    assert _pad_literal_expr_for_codegen(
        model_ir=model_ir,
        tensor_name="pads",
    ) == "[1, 2]"
    assert _scalar_literal_expr_for_codegen(
        model_ir=model_ir,
        tensor_name="scalar",
    ) == repr(float(np.float32(1.25)))


def test_static_and_runtime_axis_expressions() -> None:
    assert _int_scalar_literal_expr_for_codegen(
        static_int_tensor_values_fn=lambda _name: [-2],
        tensor_name="axis",
    ) == "-2"

    static_imports: set[str] = set()
    assert _axis_expr_from_input_for_codegen(
        runtime_imports=static_imports,
        int_scalar_literal_expr_fn=lambda _name: "3",
        tensor_expr_fn=lambda name: f"value_{name}",
        tensor_name="axis",
        device_expr="input_tensor",
    ) == "3"
    assert static_imports == set()

    runtime_imports: set[str] = set()
    assert _axis_expr_from_input_for_codegen(
        runtime_imports=runtime_imports,
        int_scalar_literal_expr_fn=lambda _name: None,
        tensor_expr_fn=lambda name: f"value_{name}",
        tensor_name="axis",
        device_expr="input_tensor",
    ) == "_coerce_scalar_axis(value_axis, device=input_tensor.device)"
    assert runtime_imports == {"_coerce_scalar_axis"}


def test_static_mirror_pad_moves_only_padded_axes_to_the_tail() -> None:
    model_ir = ModelIR(
        name="mirror_pad",
        tensors={"input": _tensor("input", [1, 2, 3, 4])},
    )
    runtime_imports: set[str] = set()

    assert _static_mirror_pad_expr_for_codegen(
        model_ir=model_ir,
        runtime_imports=runtime_imports,
        constant_pad_pairs_fn=lambda _name: [
            [0, 0],
            [1, 2],
            [0, 0],
            [3, 4],
        ],
        tensor_expr_fn=lambda _name: "input_expr",
        input_tensor_name="input",
        pads_tensor_name="pads",
    ) == (
        "_torch_permute(F.pad(_torch_permute(input_expr, [0, 2, 1, 3]), "
        "[3, 4, 1, 2], mode='reflect'), [0, 2, 1, 3])"
    )
    assert runtime_imports == {"_torch_permute"}
