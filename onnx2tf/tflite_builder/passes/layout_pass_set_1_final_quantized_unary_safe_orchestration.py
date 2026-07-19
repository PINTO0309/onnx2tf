from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LayoutAttentionQuantizedSuffixContext,
    run_layout_attention_quantized_suffix,
)
from onnx2tf.tflite_builder.passes.quantized_recovery_orchestration import (
    run_safe_binary_recovery,
)
from onnx2tf.tflite_builder.passes.transpose_unary_fanout_orchestration import (
    run_transpose_unary_fanout,
)


CleanupResults = Tuple[Any, ...]


def run_layout_pass_set_1_final_quantized_unary_safe_cleanup(
    context: LayoutAttentionQuantizedSuffixContext,
    *,
    include_duplicate_transpose: bool,
) -> Tuple[CleanupResults, CleanupResults, CleanupResults]:
    """Run the final quantized, transpose/unary, and safe-binary tail."""

    return (
        run_layout_attention_quantized_suffix(
            context,
            include_duplicate_transpose=include_duplicate_transpose,
        ),
        run_transpose_unary_fanout(
            context.pass_context,
            include_layout_transpose=True,
            include_unary_passthrough=False,
        ),
        run_safe_binary_recovery(context.pass_context),
    )
