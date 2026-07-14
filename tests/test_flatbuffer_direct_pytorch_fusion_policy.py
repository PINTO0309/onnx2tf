from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_fusion_policy import (
    _match_affine_layer_norm_for_codegen,
    _match_if_axis0_tensor_mux_slice_for_codegen,
    _match_swish_activation_pattern_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_naming import (
    _canonical_codegen_name_for_codegen,
)


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(name=name, dtype="FLOAT32", shape=shape)


def test_affine_layer_norm_pattern_builds_module_spec() -> None:
    add_op = OperatorIR(
        op_type="ADD",
        inputs=["mul_output", "bert/encoder/FakeLayerNorm/beta"],
        outputs=["bert/encoder/FakeLayerNorm/add"],
    )
    model_ir = ModelIR(
        name="affine_layer_norm",
        tensors={
            "input": _tensor("input", [1, 8, 16]),
            "bert/encoder/FakeLayerNorm/gamma": _tensor(
                "bert/encoder/FakeLayerNorm/gamma", [16]
            ),
            "bert/encoder/FakeLayerNorm/beta": _tensor(
                "bert/encoder/FakeLayerNorm/beta", [16]
            ),
            "mul_output": _tensor("mul_output", [1, 8, 16]),
            "bert/encoder/FakeLayerNorm/add": _tensor(
                "bert/encoder/FakeLayerNorm/add", [1, 8, 16]
            ),
        },
        operators=[
            OperatorIR(
                op_type="MUL",
                inputs=["input", "bert/encoder/FakeLayerNorm/gamma"],
                outputs=["mul_output"],
            ),
            add_op,
        ],
    )
    constant_names = {
        "bert/encoder/FakeLayerNorm/gamma",
        "bert/encoder/FakeLayerNorm/beta",
    }

    assert _match_affine_layer_norm_for_codegen(
        model_ir=model_ir,
        producer_index={"mul_output": 0},
        is_constant_tensor_name_fn=constant_names.__contains__,
        canonical_codegen_name_fn=_canonical_codegen_name_for_codegen,
        next_unique_attr_name_fn=lambda base: f"attr::{base}",
        op_index=1,
        op=add_op,
    ) == {
        "attr_name": "attr::encoder_layer_norm",
        "input_name": "input",
        "output_name": "bert/encoder/FakeLayerNorm/add",
        "gamma_name": "bert/encoder/FakeLayerNorm/gamma",
        "beta_name": "bert/encoder/FakeLayerNorm/beta",
        "gamma_shape": [16],
        "gamma_dtype": "FLOAT32",
        "mul_op_index": 0,
    }


def test_swish_pattern_requires_exclusive_logistic_output() -> None:
    model_ir = ModelIR(
        name="swish",
        tensors={
            "input": _tensor("input", [1, 8]),
            "sigmoid": _tensor("sigmoid", [1, 8]),
            "output": _tensor("output", [1, 8]),
        },
        operators=[
            OperatorIR(
                op_type="LOGISTIC",
                inputs=["input"],
                outputs=["sigmoid"],
            ),
            OperatorIR(
                op_type="MUL",
                inputs=["input", "sigmoid"],
                outputs=["output"],
            ),
        ],
    )

    assert _match_swish_activation_pattern_for_codegen(
        model_ir=model_ir,
        consumer_index={"input": [0, 1], "sigmoid": [1]},
        tensor_name="input",
        consumer_indices=[0, 1],
    ) == ("output", {0, 1})
    assert (
        _match_swish_activation_pattern_for_codegen(
            model_ir=model_ir,
            consumer_index={"input": [0, 1], "sigmoid": [1, 2]},
            tensor_name="input",
            consumer_indices=[0, 1],
        )
        is None
    )


def test_axis0_tensor_mux_slice_pattern_recovers_conditional_inputs() -> None:
    operators = [
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["then_value", "else_value"],
            outputs=["merged"],
            options={"axis": 0},
        ),
        OperatorIR(
            op_type="CAST",
            inputs=["condition"],
            outputs=["cond_i32"],
        ),
        OperatorIR(
            op_type="SUB",
            inputs=["one", "cond_i32"],
            outputs=["not_cond_i32"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["not_cond_i32", "then_dim"],
            outputs=["begin"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["cond_i32", "then_dim"],
            outputs=["size_then"],
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["not_cond_i32", "else_dim"],
            outputs=["size_else"],
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["size_then", "size_else"],
            outputs=["size"],
        ),
    ]
    slice_op = OperatorIR(
        op_type="SLICE",
        inputs=["merged", "begin", "size"],
        outputs=["output"],
    )
    model_ir = ModelIR(
        name="axis0_tensor_mux",
        tensors={
            "then_dim": TensorIR(
                name="then_dim",
                dtype="INT32",
                shape=[1],
                data=np.asarray([2], dtype=np.int32),
            ),
            "else_dim": TensorIR(
                name="else_dim",
                dtype="INT32",
                shape=[1],
                data=np.asarray([3], dtype=np.int32),
            ),
        },
        operators=[*operators, slice_op],
    )
    producers = {
        str(output): op
        for op in operators
        for output in op.outputs
    }

    assert _match_if_axis0_tensor_mux_slice_for_codegen(
        model_ir=model_ir,
        producer_by_output_name=producers,
        op=slice_op,
    ) == {
        "cond_name": "condition",
        "then_name": "then_value",
        "else_name": "else_value",
    }
