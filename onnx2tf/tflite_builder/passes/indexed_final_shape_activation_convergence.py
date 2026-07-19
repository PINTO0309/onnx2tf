from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.activation_fusion import (
    optimize_fuse_activation_chains,
)
from onnx2tf.tflite_builder.passes.dynamic_reshape_resolution import (
    resolve_dynamic_reshape_shapes,
)
from onnx2tf.tflite_builder.passes.graph_cleanup import prune_dead_operators
from onnx2tf.tflite_builder.passes.hardswish_shape_sanitization import (
    sanitize_hardswish_tensor_shapes,
)
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    reconcile_static_tensor_shapes,
)


def _stats_have_positive_count(*stats: Dict[str, int]) -> bool:
    """Return whether pure mutation-count dictionaries report a change."""

    return any(
        int(value) > 0
        for result in stats
        for value in result.values()
    )


def run_indexed_shape_convergence_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Converge terminal pruning and shape metadata with one graph index."""

    graph_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    prune_stats = prune_dead_operators(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    first_reconcile_stats = reconcile_static_tensor_shapes(
        model_ir,
        graph_index=graph_index,
    )
    reshape_stats = resolve_dynamic_reshape_shapes(
        model_ir,
        graph_index=graph_index,
    )
    final_reconcile_stats = {"reconciled_static_tensor_shapes": 0}
    if _stats_have_positive_count(
        prune_stats,
        first_reconcile_stats,
        reshape_stats,
    ):
        final_reconcile_stats = reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
    return {
        "removed_dead_operators": int(
            prune_stats.get("removed_dead_operators", 0)
        ),
        "resolved_dynamic_reshape_shapes": int(
            reshape_stats.get("resolved_dynamic_reshape_shapes", 0)
        ),
        "reconciled_static_tensor_shapes": int(
            first_reconcile_stats.get("reconciled_static_tensor_shapes", 0)
            + final_reconcile_stats.get(
                "reconciled_static_tensor_shapes",
                0,
            )
        ),
    }


def run_indexed_final_shape_activation_convergence(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Run terminal metadata and activation convergence with one graph index."""

    graph_index = ModelIRGraphIndex(model_ir)
    convergence_stats = run_indexed_shape_convergence_cleanup(
        model_ir,
        layout_state=layout_state,
        graph_index=graph_index,
    )
    hardswish_stats = sanitize_hardswish_tensor_shapes(
        model_ir,
        graph_index=graph_index,
    )
    first_reconcile_stats = {"reconciled_static_tensor_shapes": 0}
    if _stats_have_positive_count(
        convergence_stats,
        hardswish_stats,
    ):
        first_reconcile_stats = reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
    reshape_stats = resolve_dynamic_reshape_shapes(
        model_ir,
        graph_index=graph_index,
    )
    second_reconcile_stats = {"reconciled_static_tensor_shapes": 0}
    if _stats_have_positive_count(
        first_reconcile_stats,
        reshape_stats,
    ):
        second_reconcile_stats = reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
    fusion_tensor_count = len(model_ir.tensors)
    fusion_stats = optimize_fuse_activation_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    final_reconcile_stats = {"reconciled_static_tensor_shapes": 0}
    if (
        _stats_have_positive_count(
            second_reconcile_stats,
            fusion_stats,
        )
        or len(model_ir.tensors) < fusion_tensor_count
    ):
        final_reconcile_stats = reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
    return {
        **convergence_stats,
        "sanitized_hardswish_tensor_shapes": int(
            hardswish_stats.get("sanitized_hardswish_tensor_shapes", 0)
        ),
        "resolved_dynamic_reshape_shapes": int(
            convergence_stats.get("resolved_dynamic_reshape_shapes", 0)
            + reshape_stats.get("resolved_dynamic_reshape_shapes", 0)
        ),
        "reconciled_static_tensor_shapes": int(
            convergence_stats.get("reconciled_static_tensor_shapes", 0)
            + first_reconcile_stats.get(
                "reconciled_static_tensor_shapes",
                0,
            )
            + second_reconcile_stats.get(
                "reconciled_static_tensor_shapes",
                0,
            )
            + final_reconcile_stats.get(
                "reconciled_static_tensor_shapes",
                0,
            )
        ),
        **fusion_stats,
    }
