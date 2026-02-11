from __future__ import annotations

from onnx2tf.tflite_builder.preprocess.rules.pattern_fusion import (
    PATTERN_FUSION_WAVE2_RULE_ID,
    register_pattern_fusion_wave2_rule,
)
from onnx2tf.tflite_builder.preprocess.rules.pseudo_ops import (
    PSEUDO_OPS_WAVE1_RULE_ID,
    register_pseudo_ops_wave1_rule,
)


def register_default_preprocess_rules() -> None:
    register_pattern_fusion_wave2_rule()
    register_pseudo_ops_wave1_rule()


__all__ = [
    "PATTERN_FUSION_WAVE2_RULE_ID",
    "PSEUDO_OPS_WAVE1_RULE_ID",
    "register_default_preprocess_rules",
    "register_pattern_fusion_wave2_rule",
    "register_pseudo_ops_wave1_rule",
]
