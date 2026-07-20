from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LayoutAttentionQuantizedSuffixContext,
    run_layout_attention_quantized_suffix,
)
from onnx2tf.tflite_builder.passes.quantized_recovery_orchestration import (
    run_safe_binary_recovery,
)


CleanupResults = Tuple[Any, ...]


def run_layout_pass_set_1_attention_quantized_safe_binary_cleanup(
    context: LayoutAttentionQuantizedSuffixContext,
    *,
    include_duplicate_transpose: bool,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run the post-binary quantized suffix and safe-binary cleanup."""

    return (
        run_layout_attention_quantized_suffix(
            context,
            include_duplicate_transpose=include_duplicate_transpose,
        ),
        run_safe_binary_recovery(context.pass_context),
    )
