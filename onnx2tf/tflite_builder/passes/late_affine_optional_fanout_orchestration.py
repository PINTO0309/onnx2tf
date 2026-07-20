from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.elementwise_fanout_layout import (
    optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
)
from onnx2tf.tflite_builder.passes.late_affine_concat_orchestration import (
    run_late_affine_concat_cleanup,
)


LateAffineResults = Tuple[Dict[str, int], Tuple[Dict[str, int], ...]]


def run_late_affine_optional_fanout_cleanup(
    context: ModelIRPassContext,
    *,
    include_elementwise_fanout: bool,
) -> Tuple[LateAffineResults, Optional[Dict[str, int]]]:
    """Run late affine/Concat cleanup and optional elementwise fan-out."""

    affine_results = run_late_affine_concat_cleanup(context)
    fanout_results: Optional[Dict[str, int]] = None
    if include_elementwise_fanout:
        fanout_results = (
            optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
                context.model_ir
            )
        )
    return affine_results, fanout_results
