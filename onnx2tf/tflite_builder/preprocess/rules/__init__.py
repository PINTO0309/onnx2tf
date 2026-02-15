from __future__ import annotations

from onnx2tf.tflite_builder.preprocess.rules.constant_fold import (
    CONSTANT_FOLD_RULE_ID,
    register_constant_fold_rule,
)
from onnx2tf.tflite_builder.preprocess.rules.normalize_attrs import (
    NORMALIZE_ATTRS_RULE_ID,
    register_normalize_attrs_rule,
)
from onnx2tf.tflite_builder.preprocess.rules.pattern_fusion import (
    PATTERN_FUSION_WAVE2_RULE_ID,
    register_pattern_fusion_wave2_rule,
)
from onnx2tf.tflite_builder.preprocess.rules.quant_chain_fusion import (
    QUANT_CHAIN_FUSION_WAVE3_RULE_ID,
    register_quant_chain_fusion_wave3_rule,
)
from onnx2tf.tflite_builder.preprocess.rules.pseudo_ops import (
    PSEUDO_OPS_WAVE1_RULE_ID,
    register_pseudo_ops_wave1_rule,
)


def register_default_preprocess_rules() -> None:
    register_pattern_fusion_wave2_rule()
    register_quant_chain_fusion_wave3_rule()
    register_pseudo_ops_wave1_rule()
    register_constant_fold_rule()
    register_normalize_attrs_rule()


__all__ = [
    "CONSTANT_FOLD_RULE_ID",
    "NORMALIZE_ATTRS_RULE_ID",
    "PATTERN_FUSION_WAVE2_RULE_ID",
    "QUANT_CHAIN_FUSION_WAVE3_RULE_ID",
    "PSEUDO_OPS_WAVE1_RULE_ID",
    "register_default_preprocess_rules",
    "register_constant_fold_rule",
    "register_normalize_attrs_rule",
    "register_pattern_fusion_wave2_rule",
    "register_quant_chain_fusion_wave3_rule",
    "register_pseudo_ops_wave1_rule",
]
