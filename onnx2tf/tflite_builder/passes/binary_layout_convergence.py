from __future__ import annotations

from typing import Dict

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    repair_rank4_channelwise_broadcast_constants_to_runtime_layout,
)
from onnx2tf.tflite_builder.passes.stale_binary_adapter_repair import (
    _repair_stale_nchw_to_nhwc_channelwise_binary_transposes as repair_stale_nchw_to_nhwc_channelwise_binary_transposes,
)
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    reconcile_static_tensor_shapes,
)


def _stats_have_positive_count(*stats: Dict[str, int]) -> bool:
    return any(
        int(value) > 0
        for result in stats
        for value in result.values()
    )


def run_indexed_binary_layout_convergence(model_ir: ModelIR) -> Dict[str, int]:
    """Run up to three terminal binary-layout convergence rounds with one index."""

    graph_index = ModelIRGraphIndex(model_ir)
    repaired_constants = 0
    removed_transposes = 0
    reconciled_shapes = 0
    for _ in range(3):
        broadcast_stats = (
            repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
                model_ir,
                graph_index=graph_index,
            )
        )
        transpose_stats = (
            repair_stale_nchw_to_nhwc_channelwise_binary_transposes(
                model_ir,
                graph_index=graph_index,
            )
        )
        reconcile_stats = reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
        repaired_constants += int(
            broadcast_stats.get(
                "repaired_rank4_channelwise_broadcast_constants",
                0,
            )
        )
        removed_transposes += int(
            transpose_stats.get(
                "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes",
                0,
            )
        )
        reconciled_shapes += int(
            reconcile_stats.get("reconciled_static_tensor_shapes", 0)
        )
        if not _stats_have_positive_count(
            broadcast_stats,
            transpose_stats,
            reconcile_stats,
        ):
            break
    return {
        "repaired_rank4_channelwise_broadcast_constants": int(
            repaired_constants
        ),
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": int(
            removed_transposes
        ),
        "reconciled_static_tensor_shapes": int(reconciled_shapes),
    }
