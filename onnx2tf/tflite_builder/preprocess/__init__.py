from __future__ import annotations

from onnx2tf.tflite_builder.preprocess.pipeline import (
    clear_preprocess_rules,
    get_registered_preprocess_rule_ids,
    register_preprocess_rule,
    run_preprocess_pipeline,
)
from onnx2tf.tflite_builder.preprocess.rules import (
    CONSTANT_FOLD_RULE_ID,
    NORMALIZE_ATTRS_RULE_ID,
    PATTERN_FUSION_WAVE2_RULE_ID,
    PSEUDO_OPS_WAVE1_RULE_ID,
    register_constant_fold_rule,
    register_default_preprocess_rules,
    register_normalize_attrs_rule,
    register_pattern_fusion_wave2_rule,
    register_pseudo_ops_wave1_rule,
)

__all__ = [
    "CONSTANT_FOLD_RULE_ID",
    "NORMALIZE_ATTRS_RULE_ID",
    "PATTERN_FUSION_WAVE2_RULE_ID",
    "PSEUDO_OPS_WAVE1_RULE_ID",
    "clear_preprocess_rules",
    "get_registered_preprocess_rule_ids",
    "register_constant_fold_rule",
    "register_default_preprocess_rules",
    "register_normalize_attrs_rule",
    "register_pattern_fusion_wave2_rule",
    "register_preprocess_rule",
    "register_pseudo_ops_wave1_rule",
    "run_preprocess_pipeline",
]
