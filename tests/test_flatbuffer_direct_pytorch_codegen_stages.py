from __future__ import annotations

import ast

from onnx2tf.tflite_builder.pytorch_codegen_stages import (
    _build_named_encoder_methods_composite,
)


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
