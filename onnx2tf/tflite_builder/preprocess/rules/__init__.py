from __future__ import annotations

from onnx2tf.tflite_builder.preprocess.rules.pseudo_ops import (
    PSEUDO_OPS_WAVE1_RULE_ID,
    register_pseudo_ops_wave1_rule,
)


def register_default_preprocess_rules() -> None:
    register_pseudo_ops_wave1_rule()


__all__ = [
    "PSEUDO_OPS_WAVE1_RULE_ID",
    "register_default_preprocess_rules",
    "register_pseudo_ops_wave1_rule",
]

