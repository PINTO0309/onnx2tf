from __future__ import annotations

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    run_indexed_binary_layout_adapter_cleanup,
)
from onnx2tf.tflite_builder.passes.static_shape_signature_sanitization import (
    sanitize_static_shape_signature_consistency,
)


LATE_BINARY_REPAIR_PASS_IDS = (
    "_sanitize_static_shape_signature_consistency",
    "run_indexed_binary_layout_adapter_cleanup",
)


LateBinaryRepairContext = ModelIRPassContext


def run_late_binary_repair_cleanup(
    context: LateBinaryRepairContext,
) -> bool:
    """Run late binary repairs and report whether shapes need reconciliation."""

    initial_tensor_count = len(context.model_ir.tensors)
    signature_stats = sanitize_static_shape_signature_consistency(
        context.model_ir,
    )
    binary_adapter_stats, singleton_adapter_stats = (
        run_indexed_binary_layout_adapter_cleanup(context.model_ir)
    )
    mutation_count = (
        int(
            signature_stats.get(
                "sanitized_static_shape_signature_consistency",
                0,
            )
        )
        + int(
            binary_adapter_stats.get(
                "inserted_rank4_binary_layout_fix_transpose",
                0,
            )
        )
        + int(
            singleton_adapter_stats.get(
                "repaired_rank4_binary_singleton_broadcast_layout_mismatch",
                0,
            )
        )
    )
    return (
        mutation_count > 0
        or len(context.model_ir.tensors) < initial_tensor_count
    )
