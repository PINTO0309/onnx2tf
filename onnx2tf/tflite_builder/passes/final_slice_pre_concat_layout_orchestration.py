from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.pre_concat_nhwc_layout import (
    optimize_transpose_pre_concat_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.slice_prepost_layout import (
    optimize_transpose_slice_prepost_nhwc_passthrough_chains,
)


FINAL_SLICE_PRE_CONCAT_LAYOUT_PASS_IDS = (
    "_optimize_transpose_slice_prepost_nhwc_passthrough_chains",
    "_optimize_transpose_pre_concat_nhwc_chains",
)


FinalSlicePreConcatLayoutContext = ModelIRPassContext


def run_final_slice_pre_concat_layout_cleanup(
    context: FinalSlicePreConcatLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run final Slice passthrough then pre-Concat NHWC cleanup."""

    return (
        optimize_transpose_slice_prepost_nhwc_passthrough_chains(
            context.model_ir,
        ),
        optimize_transpose_pre_concat_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        ),
    )
