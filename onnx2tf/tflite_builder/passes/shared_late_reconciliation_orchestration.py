from __future__ import annotations

from typing import Dict

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    run_indexed_binary_layout_adapter_cleanup,
)
from onnx2tf.tflite_builder.passes.conv_input_layout import (
    sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv,
)
from onnx2tf.tflite_builder.passes.hardswish_shape_sanitization import (
    sanitize_hardswish_tensor_shapes,
)
from onnx2tf.tflite_builder.passes.singleton_consecutive_reshape_orchestration import (
    run_singleton_consecutive_reshape,
)
from onnx2tf.tflite_builder.passes.squeeze_shape_sanitization import (
    sanitize_squeeze_axes_with_static_input_shapes,
)
from onnx2tf.tflite_builder.passes.static_shape_signature_sanitization import (
    realign_dynamic_boundary_shape_signature_map,
)


SHARED_LATE_RECONCILIATION_PASS_IDS = (
    "_realign_dynamic_boundary_shape_signature_map",
    "_sanitize_hardswish_tensor_shapes",
    "_sanitize_squeeze_axes_with_static_input_shapes",
    "_sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv",
    "run_indexed_binary_layout_adapter_cleanup",
    "_run_singleton_consecutive_reshape_pass_cluster",
)


SharedLateReconciliationContext = ModelIRPassContext


def _stats_have_positive_count(*stats: Dict[str, int]) -> bool:
    return any(
        int(value) > 0
        for result in stats
        for value in result.values()
    )


def run_shared_late_reconciliation_cleanup(
    context: SharedLateReconciliationContext,
) -> bool:
    """Run shared-late cleanup and report whether shapes need reconciliation."""

    initial_tensor_count = len(context.model_ir.tensors)
    boundary_signature_stats = realign_dynamic_boundary_shape_signature_map(
        context.model_ir,
    )
    # Keep serialized HARD_SWISH metadata shape-preserving for renderers.
    hardswish_stats = sanitize_hardswish_tensor_shapes(context.model_ir)
    # Ensure final SQUEEZE axes still target singleton dimensions.
    squeeze_stats = sanitize_squeeze_axes_with_static_input_shapes(
        context.model_ir,
    )
    conv_transpose_stats = (
        sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv(
            context.model_ir,
        )
    )
    binary_adapter_stats, singleton_adapter_stats = (
        run_indexed_binary_layout_adapter_cleanup(context.model_ir)
    )
    (
        singleton_channel_stats,
        duplicate_fanout_stats,
        consecutive_reshape_stats,
    ) = run_singleton_consecutive_reshape(context)

    return _stats_have_positive_count(
        boundary_signature_stats,
        hardswish_stats,
        squeeze_stats,
        conv_transpose_stats,
        binary_adapter_stats,
        singleton_adapter_stats,
        singleton_channel_stats,
        duplicate_fanout_stats,
        consecutive_reshape_stats,
    ) or len(context.model_ir.tensors) < initial_tensor_count
