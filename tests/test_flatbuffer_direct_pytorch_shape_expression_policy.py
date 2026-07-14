from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_shape_expression_policy import (
    _reconstruct_shape_list_expr_for_codegen,
    _reconstruct_shape_scalar_expr_for_codegen,
    _shape_tensor_length_for_codegen,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(name=name, dtype="INT64", shape=shape, data=data)


def test_shape_tensor_length_requires_a_static_vector() -> None:
    model_ir = ModelIR(
        name="shape_lengths",
        tensors={
            "scalar": _tensor("scalar", []),
            "vector": _tensor("vector", [3]),
            "matrix": _tensor("matrix", [1, 3]),
        },
    )

    assert _shape_tensor_length_for_codegen(
        model_ir=model_ir, tensor_name="scalar"
    ) == 0
    assert _shape_tensor_length_for_codegen(
        model_ir=model_ir, tensor_name="vector"
    ) == 3
    assert (
        _shape_tensor_length_for_codegen(model_ir=model_ir, tensor_name="matrix")
        is None
    )


def test_reconstruct_shape_list_uses_runtime_helper_for_dynamic_shape() -> None:
    shape_op = OperatorIR(op_type="SHAPE", inputs=["input"], outputs=["shape"])
    model_ir = ModelIR(
        name="dynamic_shape",
        tensors={
            "input": TensorIR(name="input", dtype="FLOAT32", shape=[1, 3, 4]),
            "shape": _tensor("shape", [3]),
        },
        operators=[shape_op],
    )
    runtime_imports: set[str] = set()

    assert _reconstruct_shape_list_expr_for_codegen(
        model_ir=model_ir,
        producer_by_output_name={"shape": shape_op},
        tensor_exact_static_shape_list_fn=lambda _name: None,
        tensor_expr_fn=lambda name: f"value_{name}",
        runtime_imports=runtime_imports,
        tensor_name="shape",
    ) == "_tensor_shape_list(value_input)"
    assert runtime_imports == {"_tensor_shape_list"}


def test_reconstruct_shape_scalar_preserves_constant_arithmetic() -> None:
    add_op = OperatorIR(op_type="ADD", inputs=["lhs", "rhs"], outputs=["sum"])
    model_ir = ModelIR(
        name="shape_scalar",
        tensors={
            "lhs": _tensor("lhs", [1], data=np.asarray([2], dtype=np.int64)),
            "rhs": _tensor("rhs", [1], data=np.asarray([3], dtype=np.int64)),
            "sum": _tensor("sum", [1]),
        },
        operators=[add_op],
    )

    assert _reconstruct_shape_scalar_expr_for_codegen(
        model_ir=model_ir,
        producer_by_output_name={"sum": add_op},
        tensor_exact_static_shape_list_fn=lambda _name: None,
        tensor_expr_fn=lambda name: f"value_{name}",
        runtime_imports=set(),
        tensor_name="sum",
    ) == "(2 + 3)"
