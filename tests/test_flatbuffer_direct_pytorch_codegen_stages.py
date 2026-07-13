from __future__ import annotations

import ast

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.pytorch_codegen_stages import (
    _build_forward_stage_methods,
    _build_named_encoder_methods_composite,
    _fold_single_use_static_reshape_chains,
)


def test_static_reshape_chain_folder_combines_single_use_pair() -> None:
    model_ir = ModelIR(name="reshape_chain")

    assert _fold_single_use_static_reshape_chains(
        [
            "intermediate = torch.reshape(x, [1, 2, 3])",
            "y = torch.reshape(intermediate, "
            "_resolve_reshape_shape([1, -1], intermediate))",
        ],
        tensor_var_names={"x_tensor": "x", "y_tensor": "y"},
        model_ir=model_ir,
    ) == ["y = torch.reshape(x, [1, 6])"]


def test_static_reshape_chain_folder_preserves_fanout() -> None:
    model_ir = ModelIR(name="reshape_fanout")
    lines = [
        "intermediate = torch.reshape(x, [1, 2, 3])",
        "side = torch.add(intermediate, 1)",
        "y = torch.reshape(intermediate, [1, 6])",
    ]

    assert (
        _fold_single_use_static_reshape_chains(
            lines,
            tensor_var_names={"x_tensor": "x", "y_tensor": "y"},
            model_ir=model_ir,
        )
        == lines
    )


def test_forward_stage_builder_partitions_large_linear_source() -> None:
    model_ir = ModelIR(name="large_linear")
    model_ir.inputs = ["input_tensor"]
    model_ir.outputs = ["output_tensor"]
    model_ir.tensors = {
        "input_tensor": TensorIR("input_tensor", "FLOAT32", [1]),
        "output_tensor": TensorIR("output_tensor", "FLOAT32", [1]),
    }
    lines = ["value_0 = torch.add(x, 1)"]
    lines.extend(
        f"value_{index} = torch.add(value_{index - 1}, 1)" for index in range(1, 89)
    )
    lines.append("y = torch.add(value_88, 1)")

    result = _build_forward_stage_methods(
        lines,
        tensor_var_names={"input_tensor": "x", "output_tensor": "y"},
        model_ir=model_ir,
    )
    repeated = _build_forward_stage_methods(
        lines,
        tensor_var_names={"input_tensor": "x", "output_tensor": "y"},
        model_ir=model_ir,
    )

    assert result == repeated
    stage_source, forward_calls, stage_specs = result
    ast.parse("class Model:\n" + stage_source)
    assert len(stage_specs) >= 2
    assert len(forward_calls) == len(stage_specs)
    assert stage_specs[0]["inputs"] == ["x"]
    assert stage_specs[-1]["outputs"] == ["y"]


def test_forward_stage_builder_keeps_small_source_inline() -> None:
    model_ir = ModelIR(name="small")
    model_ir.inputs = ["input_tensor"]
    model_ir.outputs = ["output_tensor"]

    assert _build_forward_stage_methods(
        ["y = torch.relu(x)"],
        tensor_var_names={"input_tensor": "x", "output_tensor": "y"},
        model_ir=model_ir,
    ) == ("", ["        y = torch.relu(x)"], [])


def test_named_encoder_builder_keeps_unmatched_stages_inline() -> None:
    class_source, init_lines, forward_lines = _build_named_encoder_methods_composite(
        [
            {
                "method_name": "_forward_stage_0",
                "inputs": ["x"],
                "outputs": ["hidden"],
            }
        ],
        final_output_names={"hidden"},
    )

    assert class_source == ""
    assert init_lines == []
    assert forward_lines == ["        hidden = self._forward_stage_0(x)"]


def test_named_encoder_builder_groups_layer_and_preserves_tail_stage() -> None:
    class_source, init_lines, forward_lines = _build_named_encoder_methods_composite(
        [
            {
                "method_name": "_forward_stage_0",
                "inputs": ["x"],
                "outputs": ["bert_encoder_layer_0_attention_context"],
            },
            {
                "method_name": "_forward_stage_1",
                "inputs": ["bert_encoder_layer_0_attention_context"],
                "outputs": ["y"],
            },
        ],
        final_output_names={"y"},
    )

    ast.parse(class_source)
    assert class_source.count("class _GeneratedEncoderLayer0(") == 1
    assert init_lines == [
        "self.encoder_layer_0 = _GeneratedEncoderLayer0(",
        "            _forward_stage_0=self._forward_stage_0,",
        ")",
    ]
    assert forward_lines == [
        "        bert_encoder_layer_0_attention_context = self.encoder_layer_0(x)",
        "        y = self._forward_stage_1(bert_encoder_layer_0_attention_context)",
    ]


def test_named_encoder_builder_splits_attention_and_ffn_submodules() -> None:
    class_source, init_lines, forward_lines = _build_named_encoder_methods_composite(
        [
            {
                "method_name": "_forward_stage_0",
                "inputs": ["x"],
                "outputs": ["bert_encoder_layer_0_attention_context"],
            },
            {
                "method_name": "_forward_stage_1",
                "inputs": ["bert_encoder_layer_0_attention_context"],
                "outputs": ["bert_encoder_layer_0_ffn_hidden"],
            },
            {
                "method_name": "_forward_stage_2",
                "inputs": ["bert_encoder_layer_0_ffn_hidden"],
                "outputs": ["bert_encoder_layer_0_output_value"],
            },
        ],
        final_output_names={"bert_encoder_layer_0_output_value"},
    )

    ast.parse(class_source)
    assert "class _GeneratedEncoderLayer0Attention(" in class_source
    assert "class _GeneratedEncoderLayer0FFN(" in class_source
    assert "class _GeneratedEncoderLayer0(" in class_source
    assert init_lines[0] == ("self.encoder_layer_0 = _GeneratedEncoderLayer0(")
    assert forward_lines == [
        "        bert_encoder_layer_0_output_value = self.encoder_layer_0(x)"
    ]
