from __future__ import annotations

from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    optimize_transpose_mul_add_const_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.dual_pre_add_layout import (
    optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
)
from onnx2tf.tflite_builder.passes.prelu_passthrough_layout import (
    optimize_prelu_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.terminal_affine_fc_layout import (
    optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.terminal_prelu_bmm_layout import (
    optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains,
)


_LAYOUT_MUTATION_KEYS = (
    "removed_identity_transpose",
    "removed_inverse_transpose_pairs",
    "removed_inverse_transpose_fanout_branches",
    "composed_consecutive_transpose_pairs",
)


def run_late_binary_layout_recovery(
    model_ir: ModelIR,
    *,
    include_layout_transpose: bool,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run the late binary-layout recovery cluster and return mutation counts."""

    tensor_count_before = len(model_ir.tensors)
    prelu_stats = optimize_prelu_transpose_passthrough_chains(
        model_ir,
        layout_state=layout_state,
    )
    dual_pre_add_stats = (
        optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
            model_ir
        )
    )
    terminal_fc_stats = (
        optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(model_ir)
    )
    terminal_prelu_bmm_stats = {
        "optimized_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains": 0,
    }
    if include_layout_transpose:
        terminal_prelu_bmm_stats = (
            optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
                model_ir
            )
        )
    affine_prepost_stats = (
        optimize_transpose_mul_add_const_prepost_nhwc_chains(
            model_ir,
            layout_state=layout_state,
        )
    )
    layout_stats = {key: 0 for key in _LAYOUT_MUTATION_KEYS}
    if include_layout_transpose:
        raw_layout_stats = run_layout_transpose_cleanup(
            model_ir,
            layout_state=layout_state,
            diagnostics=diagnostics,
        )
        layout_stats = {
            key: int(raw_layout_stats.get(key, 0))
            for key in _LAYOUT_MUTATION_KEYS
        }

    return {
        "rewritten_prelu_transpose_passthrough_chains": int(
            prelu_stats.get(
                "rewritten_prelu_transpose_passthrough_chains",
                0,
            )
        ),
        "optimized_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains": int(
            dual_pre_add_stats.get(
                "optimized_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains",
                0,
            )
        ),
        "optimized_terminal_transpose_mul_add_reshape_fc_nhwc_chains": int(
            terminal_fc_stats.get(
                "optimized_terminal_transpose_mul_add_reshape_fc_nhwc_chains",
                0,
            )
        ),
        "optimized_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains": int(
            terminal_prelu_bmm_stats.get(
                "optimized_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains",
                0,
            )
        ),
        "optimized_transpose_mul_add_const_prepost_nhwc_chains": int(
            affine_prepost_stats.get(
                "optimized_transpose_mul_add_const_prepost_nhwc_chains",
                0,
            )
        ),
        **layout_stats,
        "pruned_unused_tensors": max(
            0,
            int(tensor_count_before - len(model_ir.tensors)),
        ),
    }
