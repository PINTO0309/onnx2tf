from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.expanddims_reshape_compat_layout import (
    optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat,
)
from onnx2tf.tflite_builder.passes.flatten_hw_reshape_compat_layout import (
    optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat,
)
from onnx2tf.tflite_builder.passes.reshape_transpose_collapse_layout import (
    _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains,
)


LATE_RESHAPE_LAYOUT_PASS_IDS = (
    "_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains",
    "_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains",
    "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains",
)


LateReshapeLayoutContext = ModelIRPassContext


def run_late_reshape_layout_cleanup(
    context: LateReshapeLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run adjacent late reshape/layout repairs in their fixed order."""

    return (
        optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            context.model_ir,
        ),
    )
