from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.attention_gather_cleanup_layout import (
    _optimize_attention_gather_transpose_reshape_cleanup_chains,
)
from onnx2tf.tflite_builder.passes.attention_preproj_ranklift_layout import (
    _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains,
)
from onnx2tf.tflite_builder.passes.attention_qkv_reshape_compat_layout import (
    optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat,
)
from onnx2tf.tflite_builder.passes.gather_reshape_cleanup import (
    _optimize_gather_axis0_singleton_to_reshape_input_chains,
)


LATE_ATTENTION_LAYOUT_PASS_IDS = (
    "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains",
    "_optimize_attention_gather_transpose_reshape_cleanup_chains",
    "_optimize_gather_axis0_singleton_to_reshape_input_chains",
    "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains",
)


LateAttentionLayoutContext = ModelIRPassContext


def run_late_attention_layout_cleanup(
    context: LateAttentionLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run adjacent late attention/layout repairs in their fixed order."""

    return (
        optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_attention_gather_transpose_reshape_cleanup_chains(
            context.model_ir,
        ),
        _optimize_gather_axis0_singleton_to_reshape_input_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(
            context.model_ir,
        ),
    )
