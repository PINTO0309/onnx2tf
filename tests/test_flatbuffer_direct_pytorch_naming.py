from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_naming import (
    _build_buffer_attr_name_map,
    _build_tensor_var_name_map,
    _canonical_codegen_name_for_codegen,
    _direct_codegen_module_attr_base,
    _make_tensor_storage_name_map,
    _next_unique_attr_name_for_codegen,
    _sanitize_python_identifier,
    _shorten_generated_python_identifier,
)


def _constant_tensor(name: str) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0], dtype=np.float32),
    )


def test_python_identifier_sanitization_handles_keywords_and_digits() -> None:
    assert _sanitize_python_identifier("class", prefix="t") == "class_t"
    assert _sanitize_python_identifier("123-x", prefix="t") == "t_123_x"
    assert _sanitize_python_identifier("---", prefix="tensor") == "tensor"


def test_tensor_storage_names_are_deterministic_and_collision_free() -> None:
    model_ir = ModelIR(name="storage_names")
    model_ir.tensors["1 weight"] = _constant_tensor("1 weight")
    model_ir.tensors["a-b"] = _constant_tensor("a-b")
    model_ir.tensors["a_b"] = _constant_tensor("a_b")
    model_ir.tensors["dynamic"] = TensorIR(
        name="dynamic",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )

    assert _make_tensor_storage_name_map(model_ir) == {
        "1 weight": "tensor_1_weight",
        "a-b": "a_b",
        "a_b": "a_b_1",
    }


def test_generated_identifier_shortening_is_bounded_and_distinguishes_siblings() -> (
    None
):
    common = (
        "model_BLOCK_4_4_BN_2_FusedBatchNormV3_model_BLOCK_4_1_CONV_1_Conv2D_"
        "model_BLOCK_4_4_CONV_2_depthwise1"
    )
    first = _shorten_generated_python_identifier(
        f"{common}_input_nhwc__channel_first",
        prefix="t",
    )
    second = _shorten_generated_python_identifier(
        f"{common}_output_nhwc__channel_first",
        prefix="t",
    )

    assert len(first) <= 40
    assert len(second) <= 40
    assert first != second
    assert first == _shorten_generated_python_identifier(
        f"{common}_input_nhwc__channel_first",
        prefix="t",
    )


def test_tensor_variable_names_respect_layout_and_resolve_collisions() -> None:
    model_ir = ModelIR(name="tensor_var_names")
    model_ir.inputs = ["a-b", "ab", "x_nhwc"]
    model_ir.outputs = ["y_nhwc"]
    for name in model_ir.inputs:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1, 3, 8, 8],
            shape_signature=[1, 3, 8, 8],
            logical_layout="NCHW",
        )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 4],
        shape_signature=[1, 8, 8, 4],
        logical_layout="NHWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="ADD",
            inputs=["a-b", "ab"],
            outputs=["y_nhwc"],
            options={},
        )
    )

    mapping = _build_tensor_var_name_map(model_ir)

    assert mapping["a-b"] == "ab"
    assert mapping["ab"] == "ab_1"
    assert mapping["x_nhwc"] == "x"
    assert mapping["y_nhwc"].endswith("nhwc")


def test_buffer_attribute_names_use_storage_names_and_exclusions() -> None:
    model_ir = ModelIR(name="buffer_attr_names")
    model_ir.tensors["a-b"] = _constant_tensor("a-b")
    model_ir.tensors["a_b"] = _constant_tensor("a_b")

    assert _build_buffer_attr_name_map(
        model_ir=model_ir,
        tensor_storage_name_map=_make_tensor_storage_name_map(model_ir),
        excluded_tensor_names={"a-b"},
    ) == {"a_b": "const_a_b_1"}


def test_direct_codegen_module_attribute_bases_are_stable() -> None:
    assert _direct_codegen_module_attr_base("CONV_2D") == "conv2d"
    assert _direct_codegen_module_attr_base("TRANSPOSE_CONV") == "conv_transpose2d"
    assert _direct_codegen_module_attr_base("UNIDIRECTIONAL_SEQUENCE_LSTM") == (
        "sequence_lstm"
    )
    assert _direct_codegen_module_attr_base("FUTURE_OP") == "future_op"


def test_codegen_attribute_names_are_canonical_and_collision_free() -> None:
    module_attr_counts = {"encoder_layer_norm": 1}
    affine_specs = {0: {"attr_name": "encoder_layer_norm_1"}}
    op_attr_names = {1: "encoder_layer_norm_2"}

    assert _canonical_codegen_name_for_codegen(
        "Bert/FakeLayerNorm:Gamma"
    ) == "bert_fakelayernorm_gamma"
    assert _next_unique_attr_name_for_codegen(
        base_name="Encoder/Layer Norm",
        module_attr_counts=module_attr_counts,
        affine_layer_norm_specs=affine_specs,
        op_module_attr_names=op_attr_names,
    ) == "encoder_layer_norm_3"
    assert module_attr_counts["encoder_layer_norm_3"] == 1
